from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

try:
    # Optional: Only available if a compatible statistics module is packaged with energynet
    from energynet.statistics import statistics_after_trade_and_share  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    statistics_after_trade_and_share = None  # type: ignore


def compute_stats_dataframe(
    clients_data_list: List[pd.DataFrame],
    trades: List[List[dict]],
    prices: List[List[float]],
    charges_list: List[List[Dict[str, float]]],
    discharges_list: List[List[Dict[str, float]]],
) -> pd.DataFrame:
    if statistics_after_trade_and_share is not None:
        try:
            return statistics_after_trade_and_share(
                clients_data=clients_data_list,
                tradedEnergy=trades,
                prices=prices,
                chargesDf=charges_list,
                dischargesDf=discharges_list,
                sharedEnergy=trades,
                allTradedEnergyOfSharing=[],
            )
        except Exception:
            pass
    # Fallback: return empty, and downstream aggregations will recompute from inputs
    return pd.DataFrame()


def compute_aggregates(
    stats_df: pd.DataFrame,
    trades: List[List[dict]],
    load_w: pd.DataFrame,
    solar_w: pd.DataFrame,
    soc_df: pd.DataFrame,
    capacity_map: Dict[int, float],
    houses: List[int],
) -> Dict[str, Dict[str, float]]:
    agg: Dict[str, Dict[str, float]] = {}
    try:
        grid_draw_total = float(stats_df["grid_draw_kWh"].sum()) if not stats_df.empty else 0.0
        grid_sell_total = float(stats_df["grid_sell_kWh"].sum()) if not stats_df.empty else 0.0
        peer_buy_total = float(stats_df.get("peer_buy_kWh", pd.Series(0)).sum()) if stats_df is not None else 0.0
        peer_sell_total = float(stats_df.get("peer_sell_kWh", pd.Series(0)).sum()) if stats_df is not None else 0.0

        if peer_buy_total == 0.0 and peer_sell_total == 0.0 and trades is not None:
            by_buyer: Dict[int, float] = {}
            by_seller: Dict[int, float] = {}
            for hour in trades:
                for t in hour:
                    by_buyer[t["buyer"]] = by_buyer.get(t["buyer"], 0.0) + float(t["amount"]) 
                    by_seller[t["seller"]] = by_seller.get(t["seller"], 0.0) + float(t["amount"]) 
            peer_buy_total = float(sum(by_buyer.values()))
            peer_sell_total = float(sum(by_seller.values()))

        total_load = float(load_w.values.sum()) if load_w is not None else 0.0
        total_solar = float(solar_w.values.sum()) if solar_w is not None else 0.0

        # Threshold for ignoring numeric dust in trades (also reused below)
        MIN_TRADE_KWH = 1e-3

        self_suff = 0.0 if total_load <= 1e-9 else max(0.0, 1.0 - grid_draw_total / total_load)
        denom_prod = total_solar if total_solar > 1e-9 else 0.0
        self_cons = 0.0 if denom_prod == 0.0 else max(0.0, 1.0 - (grid_sell_total + peer_sell_total) / denom_prod)

        # Robust trade coverage: compute per-hour from load, solar, and trades
        H = min(len(trades) if trades is not None else 0, len(load_w) if load_w is not None else 0)
        peer_buy_total_cov = 0.0
        peer_sell_total_cov = 0.0
        grid_draw_total_cov = 0.0
        grid_sell_total_cov = 0.0
        cov_vals = []
        for h in range(H):
            deficit_h = float(np.maximum(load_w.iloc[h] - solar_w.iloc[h], 0.0).sum())
            surplus_h = float(np.maximum(solar_w.iloc[h] - load_w.iloc[h], 0.0).sum())
            peer_buy_h = sum(
                float(t.get("amount", 0.0))
                for t in trades[h]
                if float(t.get("amount", 0.0)) > MIN_TRADE_KWH and t.get("seller") != t.get("buyer")
            )
            grid_draw_h = max(deficit_h - peer_buy_h, 0.0)
            grid_sell_h = max(surplus_h - peer_buy_h, 0.0)

            peer_buy_total_cov += peer_buy_h
            peer_sell_total_cov += peer_buy_h  # symmetry: peers' buys == peers' sells
            grid_draw_total_cov += grid_draw_h
            grid_sell_total_cov += grid_sell_h
            denom_h = peer_buy_h + grid_draw_h
            if denom_h > 1e-9:
                cov_vals.append(peer_buy_h / denom_h)
        # Time-averaged coverage across hours with a non-zero deficit
        trade_cov = float(np.mean(cov_vals)) if cov_vals else 0.0

        # Override self-sufficiency and self-consumption using consistent per-hour recomputation
        self_suff = 0.0 if total_load <= 1e-9 else max(0.0, 1.0 - grid_draw_total_cov / total_load)
        self_cons = 0.0 if denom_prod == 0.0 else max(0.0, 1.0 - (grid_sell_total_cov + peer_sell_total_cov) / denom_prod)

        # ── Time-averaged degree centrality (and weighted/strength variant) ──
        n = len(houses)
        H = len(trades) if trades is not None else 0

        # Accumulators per house
        deg_sum_by_house: Dict[int, float] = {hid: 0.0 for hid in houses}
        wdeg_norm_sum_by_house: Dict[int, float] = {hid: 0.0 for hid in houses}

        for hour in (trades or []):
            # per-hour partner sets and per-hour trade strengths
            partners_h: Dict[int, set] = {hid: set() for hid in houses}
            strength_h: Dict[int, float] = {hid: 0.0 for hid in houses}

            for t in hour:
                s = t["seller"]; b = t["buyer"]
                amt = float(t.get("amount", 0.0))
                if s == b or amt <= MIN_TRADE_KWH:
                    continue
                partners_h[s].add(b)
                partners_h[b].add(s)
                strength_h[s] += amt
                strength_h[b] += amt

            # degree centrality for this hour
            if n > 1:
                for hid in houses:
                    deg_sum_by_house[hid] += len(partners_h[hid]) / (n - 1)
            # weighted degree (normalize by max strength that hour → [0, 1])
            max_strength = max(strength_h.values()) if strength_h else 0.0
            if max_strength > 0:
                for hid in houses:
                    wdeg_norm_sum_by_house[hid] += strength_h[hid] / max_strength
            # else add 0 for everyone

        # Convert sums to time-averages over hours (include zero-trade hours as 0)
        if H > 0:
            deg_time_avg_by_house = {hid: deg_sum_by_house[hid] / H for hid in houses}
            wdeg_time_avg_by_house = {hid: wdeg_norm_sum_by_house[hid] / H for hid in houses}
        else:
            deg_time_avg_by_house = {hid: 0.0 for hid in houses}
            wdeg_time_avg_by_house = {hid: 0.0 for hid in houses}

        # Overall averages across houses
        avg_deg = float(np.mean(list(deg_time_avg_by_house.values()))) if houses else 0.0
        avg_wdeg = float(np.mean(list(wdeg_time_avg_by_house.values()))) if houses else 0.0

        # Hubs based on time-averaged centralities
        hub = max(deg_time_avg_by_house.items(), key=lambda kv: kv[1]) if houses else (None, 0.0)
        hub_w = max(wdeg_time_avg_by_house.items(), key=lambda kv: kv[1]) if houses else (None, 0.0)

        if soc_df is not None and capacity_map:
            soc_pct_df = soc_df.copy()
            for hid in soc_pct_df.columns:
                cap = float(capacity_map.get(hid, 1.0)) or 1.0
                soc_pct_df[hid] = soc_pct_df[hid] / cap * 100.0
            avg_soc_pct = float(soc_pct_df.values.mean())
        else:
            avg_soc_pct = 0.0

        if not stats_df.empty and "grid_draw_kWh" in stats_df:
            imports_per_house = stats_df["grid_draw_kWh"].astype(float)
            imp_var = float(imports_per_house.var())
        else:
            imp_var = 0.0

        agg = {
            "Grid & Imports": {
                "Grid draw (kWh)": grid_draw_total,
                "Grid sell (kWh)": grid_sell_total,
                "Import variance (kWh^2)": imp_var,
            },
            "Trading": {
                "Trade volume (kWh)": float(sum(t.get("amount", 0.0) for hour in trades for t in hour)),
                "Trade coverage": trade_cov,
                "Avg degree centrality": avg_deg,
                "Top hub (house, degree)": f"{hub[0]} @ {hub[1]:.2f}",
                "Avg weighted degree (norm.)": avg_wdeg,
                "Top weighted hub (house, degree)": f"{hub_w[0]} @ {hub_w[1]:.2f}",
            },
            "Battery": {
                "Battery charged (kWh)": float(stats_df.get("battery_charged_kWh", pd.Series(0)).sum()) if stats_df is not None else 0.0,
                "Battery discharged (kWh)": float(stats_df.get("battery_discharged_kWh", pd.Series(0)).sum()) if stats_df is not None else 0.0,
                "Average SOC (%)": avg_soc_pct,
            },
            "Autarky & Self-consumption": {
                "Self-sufficiency ratio": self_suff,
                "Self-consumption ratio": self_cons,
            },
        }
    except Exception:
        agg = {}
    return agg


