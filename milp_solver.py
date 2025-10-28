from __future__ import annotations

"""
MILP engine for peer-to-peer energy trading (rolling horizon).

Objective
---------
Minimise the total amount of energy imported from the main grid
(system-level self-sufficiency).  No second-stage “fairness” step.

CLI example
-----------
$ python -m energynet.milp_solver --folder 1 --out results --verbose

Outputs
-------
results/
└── folder_1/
    ├─ trades_hourly.json      # list[list[dict]] (seller, buyer, kWh)
    ├─ grid_import.csv         # house_id, total_grid_import_kWh
    └─ soc_timeseries.csv      # hour_idx, house_id, soc_kWh
"""
# ── stdlib ───────────────────────────────────────────────────────────────
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, List, Tuple, Generator
import json
import argparse
import re

# ── third-party ──────────────────────────────────────────────────────────
import pandas as pd
import pulp as pl

# ── local ────────────────────────────────────────────────────────────────
from energynet.data_loader import read_data_from_generated_folder


# ─────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────
WINDOW_HOURS = 24  # optimisation horizon per MILP solve
MIP_GAP = 0.01  # CBC relative optimality gap


# ─────────────────────────────────────────────────────────────────────────
@dataclass
class WindowResult:
    __slots__ = ("end_soc", "trades", "grid_imp", "soc_series")
    end_soc: Dict[int, float]
    trades: List[List[Dict[str, float]]]
    grid_imp: Dict[int, float]
    soc_series: List[Tuple[int, int, float]]


# ─────────────────────────────────────────────────────────────────────────
# Data utilities
# ─────────────────────────────────────────────────────────────────────────
def _tidy_house_frames(folder_data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Stack per-house DataFrames into one tidy frame with helper columns."""
    frames: List[pd.DataFrame] = []
    pat = re.compile(r"folder\d+_house(\d+)$")

    for key, df in folder_data.items():
        m = pat.match(key)
        if not m:
            continue  # skip monthly_balances etc.
        house_id = int(m.group(1))
        d = df.copy()
        d["house_id"] = house_id
        d["hour_idx"] = range(len(d))  # assumes rows already hourly
        d["load_kw"] = d["hourly_load_kw_house"]
        d["solar_kw"] = d["solar_hourly_watt_generation_house"] / 1000.0
        d["surplus_kw"] = d["solar_kw"] - d["load_kw"]
        frames.append(d)

    if not frames:
        raise ValueError("No house CSV frames detected. Check folder naming.")

    return (
        pd.concat(frames, ignore_index=True)
        .sort_values(["hour_idx", "house_id"])
        .reset_index(drop=True)
    )


def _rolling_windows(
    df: pd.DataFrame,
    step: int = WINDOW_HOURS,
) -> Generator[pd.DataFrame, None, None]:
    """Yield consecutive hour-slices of length `step`."""
    max_hour = int(df["hour_idx"].max())
    for start in range(0, max_hour + 1, step):
        yield df[df["hour_idx"].between(start, start + step - 1)].copy()


# ─────────────────────────────────────────────────────────────────────────
# MILP build / solve helpers
# ─────────────────────────────────────────────────────────────────────────
def _build_problem(
    df_win: pd.DataFrame,
    init_soc: Dict[int, float],
) -> Tuple[pl.LpProblem, dict]:
    """Create PuLP problem and variable dictionaries for one window."""
    houses = sorted(df_win["house_id"].unique())
    hours = sorted(df_win["hour_idx"].unique())

    prob = pl.LpProblem("p2p_window", pl.LpMinimize)

    # Decision variables
    b = pl.LpVariable.dicts("b", (houses, hours), lowBound=0)  # SOC
    ch = pl.LpVariable.dicts("ch", (houses, hours), lowBound=0)  # charge
    dis = pl.LpVariable.dicts("dis", (houses, hours), lowBound=0)  # discharge
    imp = pl.LpVariable.dicts("imp", (houses, hours), lowBound=0)  # grid import
    exp = pl.LpVariable.dicts("exp", (houses, hours), lowBound=0)  # grid export
    tr = pl.LpVariable.dicts("tr", (houses, houses, hours), lowBound=0)  # trades

    first_hour = hours[0]

    # Constraints
    for t in hours:
        hour_df = df_win[df_win["hour_idx"] == t]
        for h in houses:
            row = hour_df[hour_df["house_id"] == h].iloc[0]
            cap = row["Battery_capacity_kw"]
            solar = max(0.0, row["solar_kw"])  # generation cannot be negative
            load = row["load_kw"]

            # SOC dynamics
            if t == first_hour:
                prob += b[h][t] == init_soc[h] + ch[h][t] - dis[h][t]
                # Discharge cannot exceed available SOC at start of window
                prob += dis[h][t] <= init_soc[h]
            else:
                prob += b[h][t] == b[h][t - 1] + ch[h][t] - dis[h][t]
                # Discharge cannot exceed previous hour SOC
                prob += dis[h][t] <= b[h][t - 1]
            prob += b[h][t] <= cap

            # Power balance (use raw solar generation, not net surplus)
            prob += (dis[h][t] + solar + imp[h][t] + pl.lpSum(tr[j][h][t] for j in houses)) == (
                load + ch[h][t] + exp[h][t] + pl.lpSum(tr[h][j][t] for j in houses)
            )

            # Prevent using grid import to fund exports or peer trades.
            # Outgoing energy to grid+peers must come solely from local sources
            # (battery discharge and contemporaneous solar generation).
            prob += (exp[h][t] + pl.lpSum(tr[h][j][t] for j in houses)) <= (dis[h][t] + solar)

            # NOTE: Do NOT also constrain load to be covered by local supply.
            # Load may be served by grid import; only outgoing flows are
            # restricted to local sources to avoid grid-funded trades/exports.

    return prob, dict(b=b, ch=ch, dis=dis, imp=imp, exp=exp, trade=tr)


def _solve_window(
    df_win: pd.DataFrame,
    init_soc: Dict[int, float],
    verbose: bool,
) -> WindowResult:
    """Solve one 24-hour MILP window."""
    prob, vars = _build_problem(df_win, init_soc)
    houses = sorted(df_win["house_id"].unique())
    hours = sorted(df_win["hour_idx"].unique())

    # Objective: minimise total grid import across all houses and hours
    imp = vars["imp"]
    prob.setObjective(pl.lpSum(imp[h][t] for h in houses for t in hours))
    prob.solve(pl.PULP_CBC_CMD(msg=verbose, gapRel=MIP_GAP, threads=1))
    if pl.LpStatus[prob.status] != "Optimal":
        raise RuntimeError(f"CBC status: {pl.LpStatus[prob.status]}")

    end_soc = {h: vars["b"][h][hours[-1]].varValue for h in houses}
    grid_imp = {h: sum(vars["imp"][h][t].varValue for t in hours) for h in houses}

    # Collect trades & SOC series
    trades_hourly, soc_series = [], []
    tr = vars["trade"]
    for t in hours:
        trades_hourly.append(
            [
                {"seller": int(s), "buyer": int(b), "amount": float(tr[s][b][t].varValue)}
                for s in houses
                for b in houses
                if tr[s][b][t].varValue > 1e-6
            ]
        )
        for h in houses:
            soc_series.append((t, h, vars["b"][h][t].varValue))

    return WindowResult(end_soc, trades_hourly, grid_imp, soc_series)


# ─────────────────────────────────────────────────────────────────────────
# Folder-level orchestration
# ─────────────────────────────────────────────────────────────────────────
def solve_folder(
    folder: int,
    out_root: str = "results",
    verbose: bool = False,
) -> None:
    """Run optimisation for one 'Generated Data – <n>' folder."""
    # 1 — load & tidy
    folder_data = read_data_from_generated_folder(folder)
    df = _tidy_house_frames(folder_data)

    houses = sorted(df["house_id"].unique())
    soc = {h: 0.0 for h in houses}  # empty batteries at t = 0

    total_grid_import: Dict[int, float] = {h: 0.0 for h in houses}
    trades_hourly_all: List[List[Dict[str, float]]] = []
    soc_series_all: List[Tuple[int, int, float]] = []

    # 2 — rolling-horizon MILP solves
    for win_df in _rolling_windows(df):
        if verbose:
            print(f"Solving window {win_df['hour_idx'].min()}-{win_df['hour_idx'].max()} …")

        win_res = _solve_window(win_df, soc, verbose)

        soc = win_res.end_soc  # feed SOC into next window

        for h, g_imp in win_res.grid_imp.items():
            total_grid_import[h] += g_imp
        trades_hourly_all.extend(win_res.trades)
        soc_series_all.extend(win_res.soc_series)

    # 3 — write outputs
    out_dir = Path(out_root) / f"folder_{folder}"
    out_dir.mkdir(parents=True, exist_ok=True)

    pd.DataFrame(
        {
            "house_id": list(total_grid_import.keys()),
            "total_grid_import_kWh": list(total_grid_import.values()),
        }
    ).to_csv(out_dir / "grid_import.csv", index=False)

    # Write a trades-tagged SoC so the viewer can switch between strategies
    soc_df = pd.DataFrame(soc_series_all, columns=["hour_idx", "house_id", "soc_kWh"]).sort_values(
        ["hour_idx", "house_id"]
    )
    soc_df.to_csv(out_dir / "soc_timeseries.csv", index=False)
    soc_df.to_csv(out_dir / "soc_timeseries_trades.csv", index=False)

    with open(out_dir / "trades_hourly.json", "w", encoding="utf-8") as f:
        json.dump(trades_hourly_all, f, indent=2)

    if verbose:
        print(f"Results written to {out_dir.resolve()}")


# ─────────────────────────────────────────────────────────────────────────
# CLI entry-point
# ─────────────────────────────────────────────────────────────────────────
def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="milp_solver",
        description="Solve rolling-horizon MILPs for peer-to-peer energy trading.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--folder",
        type=int,
        required=True,
        help="Index n corresponding to the folder ‘Generated Data – <n>’.",
    )
    p.add_argument(
        "--out",
        "--out_root",
        dest="out_root",
        default="results",
        help="Root directory for results.",
    )
    p.add_argument(
        "--verbose",
        action="store_true",
        help="Show CBC solver output and progress.",
    )
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    solve_folder(folder=args.folder, out_root=args.out_root, verbose=args.verbose)


if __name__ == "__main__":
    main()
