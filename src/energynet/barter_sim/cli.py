from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List
import numpy as np
import pandas as pd

from .adapters import load_hour_inputs, load_price_series, load_initial_balances, load_all_hourly_series
from .engine import BarterSimulator, Config
from .models import Peer
from .io import ensure_out_dir, write_grid_import, write_trades, write_barters, write_open_claims, write_paid_earned, write_summary, append_log, write_soc_timeseries, write_claim_returns, write_claim_expiries


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="barter_sim",
        description="Non-MILP simulator for P2P trading and bartering",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--folder", type=int, required=True)
    p.add_argument("--algo", choices=["T_AND_B", "C_EB", "P2P_EB", "S_EB", "AI_P2P_EB"], default="T_AND_B")
    p.add_argument("--eta", type=float, default=0.6)
    p.add_argument("--tau", type=int, default=6)
    p.add_argument("--price-window", type=int, default=3)
    p.add_argument("--priority-rule", choices=["claims_first"], default="claims_first")
    p.add_argument("--balance-mode", choices=["fixed", "percent_of_grid_bill"], default="percent_of_grid_bill")
    p.add_argument("--balance-percent", type=float, default=5.0)
    p.add_argument("--expiry-action", choices=["consumer_keeps", "return_to_lender"], default="consumer_keeps",
                   help="What happens to unredeemed claim energy at expiry: stays with consumer or returns to lender")
    p.add_argument("--ai-model-path", type=str, default="")
    p.add_argument("--out", dest="out_root", default="results")
    p.add_argument("--verbose", action="store_true")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main() -> None:
    args = _parse_args()

    # Load all series in a single pass (cached) to avoid repeated IO
    loads_series, solars_series, caps, util_prices, fit_prices = load_all_hourly_series(args.folder)

    if not loads_series:
        raise RuntimeError("No hourly data loaded; cannot run simulation.")

    # Build peers with capacities and balances
    balances = load_initial_balances(args.folder, mode=args.balance_mode, percent=args.balance_percent)
    peers = {hid: Peer(hid, battery_cap_kWh=caps.get(hid, 0.0), soc_kWh=0.0, balance_currency=balances.get(hid, 0.0)) for hid in caps}

    cfg = Config(
        folder=args.folder,
        algo=args.algo,
        eta=args.eta,
        tau=args.tau,
        price_window=args.price_window,
        priority_rule=args.priority_rule,
        random_seed=args.seed,
        balance_mode=args.balance_mode,
        balance_percent=args.balance_percent,
        out_root=args.out_root,
        verbose=args.verbose,
        expiry_action=args.expiry_action,
    )

    sim = BarterSimulator(
        peers=peers,
        util_prices=util_prices,
        fit_prices=fit_prices,
        loads=loads_series,
        solars=solars_series,
        caps=caps,
        config=cfg,
        predictor=None,  # will use heuristic unless AI_P2P_EB provided with model
    )

    out_dir = ensure_out_dir(args.out_root, args.folder)
    append_log(out_dir, f"Config: {vars(args)}")

    res = sim.run()

    # Outputs
    write_grid_import(out_dir, res.grid_import_by_house)
    write_trades(out_dir, res.trades_all)
    write_barters(out_dir, res.barters_all)
    write_open_claims(out_dir, res.open_claims, t_end=len(loads_series) - 1)
    write_paid_earned(out_dir, res.paid_earned)
    write_soc_timeseries(out_dir, res.soc_series)
    # New: claim return events for viewer
    write_claim_returns(out_dir, res.claim_returns)
    # New: claim expiry events for viewer
    write_claim_expiries(out_dir, res.claim_expiries)

    summary = {
        "grid_import": sum(res.grid_import_by_house.values()),
        "p2p_traded": sum(tr.qty_kWh for tr in res.trades_all),
        "barter_consumed": sum(b.Es_kWh * cfg.eta for b in res.barters_all),
        "coordinator_payouts": res.coordinator_payouts if args.algo == "C_EB" else 0.0,
        "avg_price": sum(res.price_series) / max(1, len(res.price_series)),
    }
    write_summary(out_dir, summary)

    if args.verbose:
        # Bartering article statistics (existing summary)
        print("Bartering article statistics:")
        print(f"  Total grid import: {summary['grid_import']:.3f} kWh")
        print(f"  Total P2P traded (monetary): {summary['p2p_traded']:.3f} kWh")
        print(f"  Total bartered and consumed: {summary['barter_consumed']:.3f} kWh")
        tot_paid_minus_earned = sum(pe[0] - pe[1] for pe in res.paid_earned.values())
        print(f"  Net paid-earnings across peers: {tot_paid_minus_earned:.2f}")

        # Newly implemented statistics (aggregates)
        print("\nNewly implemented statistics:")
        peer_ids = sorted(peers.keys())

        # Build per-hour trade list for metrics
        T = len(loads_series)
        trades_by_hour: List[List[Dict]] = [[] for _ in range(T)]
        for tr in res.trades_all:
            if 0 <= tr.t < T:
                trades_by_hour[tr.t].append({"seller": tr.seller, "buyer": tr.buyer, "amount": float(tr.qty_kWh)})

        # Build load and solar DataFrames [T x houses]
        load_rows = []
        solar_rows = []
        for t_idx in range(T):
            load_row = {hid: float(loads_series[t_idx].get(hid, 0.0)) for hid in peer_ids}
            solar_row = {hid: float(solars_series[t_idx].get(hid, 0.0)) for hid in peer_ids}
            load_rows.append(load_row)
            solar_rows.append(solar_row)
        load_df = pd.DataFrame(load_rows)
        solar_df = pd.DataFrame(solar_rows)

        # Battery charged/discharged and SOC%
        soc_df = pd.DataFrame(res.soc_series, columns=["hour_idx", "house_id", "soc_kWh"]) if res.soc_series else pd.DataFrame(columns=["hour_idx", "house_id", "soc_kWh"])
        if not soc_df.empty:
            soc_pivot = soc_df.pivot(index="hour_idx", columns="house_id", values="soc_kWh").reindex(columns=peer_ids, fill_value=0.0).sort_index()
        else:
            soc_pivot = pd.DataFrame(columns=peer_ids)

        battery_charged_total = 0.0
        battery_discharged_total = 0.0
        if not soc_pivot.empty:
            diffs = soc_pivot.diff().fillna(0.0)
            battery_charged_total = float(diffs.clip(lower=0).sum().sum())
            battery_discharged_total = float((-diffs.clip(upper=0)).sum().sum())

        # Average SOC (%) across houses and time
        if not soc_pivot.empty:
            soc_pct = soc_pivot.copy()
            for hid in soc_pct.columns:
                cap = float(caps.get(hid, 1.0)) or 1.0
                soc_pct[hid] = soc_pct[hid] / cap * 100.0
            avg_soc_pct = float(soc_pct.values.mean())
        else:
            avg_soc_pct = 0.0

        # Totals and coverage
        total_load = float(load_df.values.sum()) if not load_df.empty else 0.0
        total_solar = float(solar_df.values.sum()) if not solar_df.empty else 0.0

        MIN_TRADE_KWH = 1e-3
        H = min(len(trades_by_hour), len(load_df))
        peer_buy_total_cov = 0.0
        peer_sell_total_cov = 0.0
        grid_draw_total_cov = 0.0
        grid_sell_total_cov = 0.0
        cov_vals: List[float] = []
        for h in range(H):
            deficit_h = float(np.maximum(load_df.iloc[h] - solar_df.iloc[h], 0.0).sum())
            surplus_h = float(np.maximum(solar_df.iloc[h] - load_df.iloc[h], 0.0).sum())
            peer_buy_h = sum(float(t.get("amount", 0.0)) for t in trades_by_hour[h] if float(t.get("amount", 0.0)) > MIN_TRADE_KWH and t.get("seller") != t.get("buyer"))
            grid_draw_h = max(deficit_h - peer_buy_h, 0.0)
            grid_sell_h = max(surplus_h - peer_buy_h, 0.0)
            peer_buy_total_cov += peer_buy_h
            peer_sell_total_cov += peer_buy_h
            grid_draw_total_cov += grid_draw_h
            grid_sell_total_cov += grid_sell_h
            denom_h = peer_buy_h + grid_draw_h
            if denom_h > 1e-9:
                cov_vals.append(peer_buy_h / denom_h)
        trade_cov = float(np.mean(cov_vals)) if cov_vals else 0.0

        # Self-sufficiency and self-consumption ratios
        self_suff = 0.0 if total_load <= 1e-9 else max(0.0, 1.0 - grid_draw_total_cov / total_load)
        self_cons = 0.0 if total_solar <= 1e-9 else max(0.0, 1.0 - (grid_sell_total_cov + peer_sell_total_cov) / total_solar)

        # Degree and weighted degree centralities (time-averaged)
        n = len(peer_ids)
        deg_sum_by_house: Dict[int, float] = {hid: 0.0 for hid in peer_ids}
        wdeg_norm_sum_by_house: Dict[int, float] = {hid: 0.0 for hid in peer_ids}
        for hour in trades_by_hour:
            partners_h: Dict[int, set] = {hid: set() for hid in peer_ids}
            strength_h: Dict[int, float] = {hid: 0.0 for hid in peer_ids}
            for t in hour:
                s = t["seller"]; b = t["buyer"]; amt = float(t.get("amount", 0.0))
                if s == b or amt <= MIN_TRADE_KWH:
                    continue
                partners_h[s].add(b)
                partners_h[b].add(s)
                strength_h[s] += amt
                strength_h[b] += amt
            if n > 1:
                for hid in peer_ids:
                    deg_sum_by_house[hid] += len(partners_h[hid]) / (n - 1)
            max_strength = max(strength_h.values()) if strength_h else 0.0
            if max_strength > 0:
                for hid in peer_ids:
                    wdeg_norm_sum_by_house[hid] += strength_h[hid] / max_strength
        Hn = len(trades_by_hour)
        if Hn > 0:
            deg_time_avg = {hid: deg_sum_by_house[hid] / Hn for hid in peer_ids}
            wdeg_time_avg = {hid: wdeg_norm_sum_by_house[hid] / Hn for hid in peer_ids}
        else:
            deg_time_avg = {hid: 0.0 for hid in peer_ids}
            wdeg_time_avg = {hid: 0.0 for hid in peer_ids}
        avg_deg = float(np.mean(list(deg_time_avg.values()))) if peer_ids else 0.0
        avg_wdeg = float(np.mean(list(wdeg_time_avg.values()))) if peer_ids else 0.0
        hub = max(deg_time_avg.items(), key=lambda kv: kv[1]) if peer_ids else (None, 0.0)
        hub_w = max(wdeg_time_avg.items(), key=lambda kv: kv[1]) if peer_ids else (None, 0.0)

        # Import variance across houses
        imports = np.array([float(res.grid_import_by_house.get(hid, 0.0)) for hid in peer_ids], dtype=float)
        imp_var = float(np.var(imports)) if imports.size > 0 else 0.0

        # Print aggregate blocks
        print("Grid & Imports:")
        print(f"  Grid draw (kWh): {grid_draw_total_cov:.3f}")
        print(f"  Grid sell (kWh): {grid_sell_total_cov:.3f}")
        print(f"  Import variance (kWh^2): {imp_var:.3f}")

        total_trade_volume = float(sum(float(t.get("amount", 0.0)) for hour in trades_by_hour for t in hour))
        print("Trading:")
        print(f"  Trade volume (kWh): {total_trade_volume:.3f}")
        print(f"  Trade coverage: {trade_cov:.3f}")
        print(f"  Avg degree centrality: {avg_deg:.3f}")
        print(f"  Top hub (house, degree): {hub[0]} @ {hub[1]:.2f}")
        print(f"  Avg weighted degree (norm.): {avg_wdeg:.3f}")
        print(f"  Top weighted hub (house, degree): {hub_w[0]} @ {hub_w[1]:.2f}")

        print("Battery:")
        print(f"  Battery charged (kWh): {battery_charged_total:.3f}")
        print(f"  Battery discharged (kWh): {battery_discharged_total:.3f}")
        print(f"  Average SOC (%): {avg_soc_pct:.2f}")

        print("Autarky & Self-consumption:")
        print(f"  Self-sufficiency ratio: {self_suff:.3f}")
        print(f"  Self-consumption ratio: {self_cons:.3f}")


if __name__ == "__main__":
    main()


