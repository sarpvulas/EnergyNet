from __future__ import annotations
"""MILP engine for peer‑to‑peer energy trading (rolling horizon).

Two‑stage lexicographic objective:
  1. Minimise total grid import (system self‑sufficiency).
  2. Given that minimum, minimise the *largest* grid import borne by any
     single house (fairness).

CLI example
────────────
$ python -m energynet.milp_solver --folder 1 --out results --verbose

Outputs
───────
results/
└── folder_1/
    ├─ trades_hourly.json      # list[list[dict]] (seller, buyer, kWh)
    ├─ grid_import.csv         # house_id, total_grid_import_kWh
    └─ soc_timeseries.csv      # hour_idx, house_id, soc_kWh
"""
# stdlib
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, List, Tuple, Generator
import json
import argparse
import re

# third‑party
import pandas as pd
import pulp as pl

# local (make sure PYTHONPATH includes project root)
from data_loader import read_data_from_generated_folder

# ───────────────────────────────────────────────────────────────────────────
# Constants
# ───────────────────────────────────────────────────────────────────────────
WINDOW_HOURS = 24        # optimisation horizon per MILP solve
MIP_GAP      = 0.01      # relative optimality gap for CBC
TOL          = 1e-6      # tolerance when fixing stage‑2 total import

# ───────────────────────────────────────────────────────────────────────────
@dataclass(slots=True)
class WindowResult:
    """Container for results from a single WINDOW_HOURS optimisation."""
    end_soc: Dict[int, float]
    trades: List[List[Dict[str, float]]]
    grid_imp: Dict[int, float]
    soc_series: List[Tuple[int, int, float]]  # (hour_idx, house_id, soc_kWh)

# ───────────────────────────────────────────────────────────────────────────
# Data helpers
# ───────────────────────────────────────────────────────────────────────────

def _tidy_house_frames(folder_data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Stack per‑house DataFrames into one tidy frame with helper columns."""
    frames: List[pd.DataFrame] = []
    pat = re.compile(r"folder\d+_house(\d+)$")

    for key, df in folder_data.items():
        m = pat.match(key)
        if not m:
            continue  # skip monthly_balances etc.
        house_id = int(m.group(1))
        d = df.copy()
        d["house_id"] = house_id
        d["hour_idx"] = range(len(d))  # assumes file already hourly ordered
        d["load_kw"]    = d["hourly_load_kw_house"]
        d["surplus_kw"] = (
            d["solar_hourly_watt_generation_house"] / 1000.0
            - d["load_kw"]
        )
        frames.append(d)

    if not frames:
        raise ValueError("No house CSV frames detected. Check folder naming.")

    return (
        pd.concat(frames, ignore_index=True)
          .sort_values(["hour_idx", "house_id"])
          .reset_index(drop=True)
    )


def _rolling_windows(df: pd.DataFrame, step: int = WINDOW_HOURS
                     ) -> Generator[pd.DataFrame, None, None]:
    """Yield consecutive hour‑slices of `step` length."""
    max_hour = int(df["hour_idx"].max())
    for start in range(0, max_hour + 1, step):
        yield df[df["hour_idx"].between(start, start + step - 1)].copy()

# ───────────────────────────────────────────────────────────────────────────
# MILP build/solve helpers
# ───────────────────────────────────────────────────────────────────────────

def _build_problem(
    df_win: pd.DataFrame,
    init_soc: Dict[int, float],
) -> Tuple[pl.LpProblem, dict]:
    """Create PuLP problem and variable dictionaries for one window."""
    houses = sorted(df_win["house_id"].unique())
    hours  = sorted(df_win["hour_idx"].unique())

    prob = pl.LpProblem("p2p_window", pl.LpMinimize)

    # Variables
    b    = pl.LpVariable.dicts("b",   (houses, hours), lowBound=0)
    ch   = pl.LpVariable.dicts("ch",  (houses, hours), lowBound=0)
    dis  = pl.LpVariable.dicts("dis", (houses, hours), lowBound=0)
    imp  = pl.LpVariable.dicts("imp", (houses, hours), lowBound=0)
    exp  = pl.LpVariable.dicts("exp", (houses, hours), lowBound=0)
    tr   = pl.LpVariable.dicts("tr",  (houses, houses, hours), lowBound=0)
    gmax = pl.LpVariable("G_max", lowBound=0)

    first_hour = hours[0]

    # Constraints per house/hour
    for t in hours:
        hour_df = df_win[df_win["hour_idx"] == t]
        for h in houses:
            row = hour_df[hour_df["house_id"] == h].iloc[0]
            cap     = row["Battery_capacity_kw"]
            surplus = max(0.0, row["surplus_kw"])
            load    = row["load_kw"]

            # SOC dynamics
            if t == first_hour:
                prob += b[h][t] == init_soc[h] + ch[h][t] - dis[h][t]
            else:
                prob += b[h][t] == b[h][t-1] + ch[h][t] - dis[h][t]
            prob += b[h][t] <= cap

            # Power balance
            prob += (
                dis[h][t] + surplus + imp[h][t] +
                pl.lpSum(tr[j][h][t] for j in houses)
            ) == (
                load + ch[h][t] + exp[h][t] +
                pl.lpSum(tr[h][j][t] for j in houses)
            )

    # Link total import of each house to G_max
    for h in houses:
        prob += pl.lpSum(imp[h][t] for t in hours) <= gmax

    vars = dict(b=b, ch=ch, dis=dis, imp=imp, exp=exp, trade=tr, gmax=gmax)
    return prob, vars


def _solve_two_stage(
    prob: pl.LpProblem,
    vars: dict,
    houses: List[int],
    hours: List[int],
    verbose: bool,
) -> None:
    """Lexicographic solve: minimise total import, then minimise G_max."""
    imp, gmax = vars["imp"], vars["gmax"]

    # Stage‑1 objective: minimise total grid import
    prob.setObjective(pl.lpSum(imp[h][t] for h in houses for t in hours))
    prob.solve(pl.PULP_CBC_CMD(msg=verbose, gapRel=MIP_GAP))
    if pl.LpStatus[prob.status] != "Optimal":
        raise RuntimeError(f"Stage‑1 CBC status: {pl.LpStatus[prob.status]}")
    total_imp = pl.value(prob.objective)

    # Stage‑2: minimise G_max, keeping total import ≈ optimal
    prob2 = prob.deepcopy()
    prob2 += (
        pl.lpSum(imp[h][t] for h in houses for t in hours)
        <= total_imp + TOL
    )
    prob2.setObjective(gmax)
    prob2.solve(pl.PULP_CBC_CMD(msg=verbose, gapRel=MIP_GAP))
    if pl.LpStatus[prob2.status] != "Optimal":
        raise RuntimeError(f"Stage‑2 CBC status: {pl.LpStatus[prob2.status]}")

    # Copy solution back to original problem's variables
    for v in prob2.variables():
        prob.variablesDict()[v.name].varValue = v.varValue


def _solve_window(
    df_win: pd.DataFrame,
    init_soc: Dict[int, float],
    verbose: bool,
) -> WindowResult:
    prob, vars = _build_problem(df_win, init_soc)
    houses = sorted(df_win["house_id"].unique())
    hours  = sorted(df_win["hour_idx"].unique())

    _solve_two_stage(prob, vars, houses, hours, verbose)

    end_soc  = {h: vars["b"][h][hours[-1]].varValue for h in houses}
    grid_imp = {h: sum(vars["imp"][h][t].varValue for t in hours) for h in houses}

    trades_hourly: List[List[Dict[str, float]]] = []
    soc_series:    List[Tuple[int, int, float]] = []

    tr = vars["trade"]
    for t in hours:
        hour_trades: List[Dict[str, float]] = []
        for s in houses:
            for b in houses:
                amt = tr[s][b][t].varValue
                if amt and amt > 1e-6:
                    hour_trades.append({"seller": s, "buyer": b, "amount": amt})
        trades_hourly.append(hour_trades)

        for h in houses:
            soc_series.append((t, h, vars["b"][h][t].varValue))

    return WindowResult(end_soc, trades_hourly, grid_imp, soc_series)

# ───────────────────────────────────────────────────────────────────────────
# Folder‑level orchestration
# ───────────────────────────────────────────────────────────────────────────

def solve_folder(
    folder: int,
    out_root: str = "results",
    verbose: bool = False,
) -> None:
    """Run optimisation for one "Generated Data - <n>" folder.

    Parameters
    ----------
    folder : int
        Index *n* in “Generated Data - <n>”.
    out_root : str, default "results"
        Root directory in which a sub-folder “folder_<n>” will be created.
    verbose : bool, default False
        If True, passes `msg=True` to CBC so PuLP prints solver logs.
    """
    # ── 1. Load + tidy data ────────────────────────────────────────────────
    folder_data = read_data_from_generated_folder(folder)
    df          = _tidy_house_frames(folder_data)

    houses: List[int] = sorted(df["house_id"].unique())
    # Initial state-of-charge: assume empty batteries at t = 0
    soc: Dict[int, float] = {h: 0.0 for h in houses}

    # Accumulators across all windows
    total_grid_import: Dict[int, float] = {h: 0.0 for h in houses}
    trades_hourly_all: List[List[Dict[str, float]]] = []
    soc_series_all:   List[Tuple[int, int, float]]   = []

    # ── 2. Rolling-horizon solve ──────────────────────────────────────────
    for win_df in _rolling_windows(df):
        if verbose:
            first_hr = win_df["hour_idx"].min()
            last_hr  = win_df["hour_idx"].max()
            print(f"Solving window {first_hr}–{last_hr} …")

        win_res = _solve_window(win_df, soc, verbose)

        # Update initial SOC for next window
        soc = win_res.end_soc

        # Aggregate results
        for h, g_imp in win_res.grid_imp.items():
            total_grid_import[h] += g_imp
        trades_hourly_all.extend(win_res.trades)
        soc_series_all.extend(win_res.soc_series)

    # ── 3. Write outputs ──────────────────────────────────────────────────
    out_dir = Path(out_root) / f"folder_{folder}"
    out_dir.mkdir(parents=True, exist_ok=True)

    # 3-a. Grid import per house
    pd.DataFrame(
        {
            "house_id": list(total_grid_import.keys()),
            "total_grid_import_kWh": list(total_grid_import.values()),
        }
    ).to_csv(out_dir / "grid_import.csv", index=False)

    # 3-b. SOC time-series
    (
        pd.DataFrame(soc_series_all, columns=["hour_idx", "house_id", "soc_kWh"])
          .sort_values(["hour_idx", "house_id"])
          .to_csv(out_dir / "soc_timeseries.csv", index=False)
    )

    # 3-c. Trades per hour (JSON)
    with open(out_dir / "trades_hourly.json", "w", encoding="utf-8") as f:
        json.dump(trades_hourly_all, f, indent=2)

    if verbose:
        print(f"Results written to {out_dir.resolve()}.")


# ───────────────────────────────────────────────────────────────────────────
# CLI entry-point
# ───────────────────────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog        = "milp_solver",
        description = "Solve rolling-horizon MILPs for peer-to-peer energy trading.",
        formatter_class = argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--folder", type=int, required=True,
        help="Index n corresponding to the folder “Generated Data - <n>”."
    )
    p.add_argument(
        "--out", "--out_root", dest="out_root", default="results",
        help="Root directory into which results are written."
    )
    p.add_argument(
        "--verbose", action="store_true",
        help="Print CBC solver output and progress messages."
    )
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    solve_folder(folder=args.folder, out_root=args.out_root, verbose=args.verbose)

if __name__ == "__main__":
    main()

