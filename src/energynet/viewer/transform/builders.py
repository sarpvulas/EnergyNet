from __future__ import annotations

import re
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd


def _norm(x) -> str:
    return str(x).lower().replace(" ", "").replace("_", "")


def build_load_and_solar_wide(frames: Dict[int, pd.DataFrame]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    LOAD_KEYS = ("hourly_load", "load", "consumption", "demand")
    SOLAR_KEYS = (
        "solar_hourly_watt_generation",
        "solar_hourly", "solar_generation", "solargen",
        "solar", "pv", "generation",
    )

    def _match(txt: str, keys: Tuple[str, ...]) -> bool:
        low = txt.lower()
        return any(k in low for k in keys)

    def _pick_hourly(df: pd.DataFrame, cand: list[str]) -> str:
        if len(cand) == 1:
            return cand[0]
        hourly = [c for c in cand if re.search(r"hourly|kwh", c, re.I)]
        if hourly:
            return hourly[0]
        varied = sorted(cand, key=lambda c: df[c].astype(float).std(), reverse=True)
        return varied[0]

    s_load, s_solar = {}, {}
    for hid, df in frames.items():
        cols = [c.lower() for c in df.columns]
        if any(_match(c, LOAD_KEYS) for c in cols) and any(_match(c, SOLAR_KEYS) for c in cols):
            load_col = next(c for c in df.columns if _match(c, LOAD_KEYS))
            solar_candidates = [c for c in df.columns if _match(c, SOLAR_KEYS)]
            solar_col = _pick_hourly(df, solar_candidates)
            l = df[load_col].astype(float)
            s = df[solar_col].astype(float)
            if "watt" in load_col.lower():
                l = l / 1000.0
            if "watt" in solar_col.lower():
                s = s / 1000.0
            s_load[hid] = pd.Series(l.values)
            s_solar[hid] = pd.Series(s.values)
            continue

        if any(_match(idx, LOAD_KEYS) for idx in df.index.map(str)) and any(_match(idx, SOLAR_KEYS) for idx in df.index.map(str)):
            l_row = next(r for r in df.index if _match(r, LOAD_KEYS))
            solar_rows = [r for r in df.index if _match(r, SOLAR_KEYS)]
            s_row = _pick_hourly(df.T, solar_rows)
            l = df.loc[l_row].astype(float)
            s = df.loc[s_row].astype(float)
            if "watt" in str(l_row).lower():
                l = l / 1000.0
            if "watt" in str(s_row).lower():
                s = s / 1000.0
            s_load[hid] = pd.Series(l.values)
            s_solar[hid] = pd.Series(s.values)
            continue

        first_vals = df.iloc[:, 0].astype(str)
        if any(_match(v, LOAD_KEYS) for v in first_vals) and any(_match(v, SOLAR_KEYS) for v in first_vals):
            load_mask = first_vals.apply(lambda x: _match(x, LOAD_KEYS))
            solar_mask = first_vals.apply(lambda x: _match(x, SOLAR_KEYS))
            l_vals = df[load_mask].iloc[:, 1:].astype(float).values.flatten()
            s_df = df[solar_mask].iloc[:, 1:].astype(float)
            s_row = _pick_hourly(s_df, list(s_df.index.astype(str)))
            s_vals = s_df.loc[s_row].values.flatten()
            if "watt" in first_vals[load_mask.idxmax()].lower():
                l_vals /= 1000.0
            if "watt" in first_vals[solar_mask.idxmax()].lower():
                s_vals /= 1000.0
            s_load[hid] = pd.Series(l_vals)
            s_solar[hid] = pd.Series(s_vals)
            continue

        if df.index.nlevels == 1 and any(_match(r, LOAD_KEYS) for r in df.index.map(str)) and any(_match(r, SOLAR_KEYS) for r in df.index.map(str)):
            l_row = next(r for r in df.index if _match(r, LOAD_KEYS))
            solar_rows = [r for r in df.index if _match(r, SOLAR_KEYS)]
            s_row = _pick_hourly(df.T, solar_rows)
            l = df.loc[l_row].astype(float)
            s = df.loc[s_row].astype(float)
            if "watt" in str(l_row).lower():
                l = l / 1000.0
            if "watt" in str(s_row).lower():
                s = s / 1000.0
            s_load[hid] = pd.Series(l.values)
            s_solar[hid] = pd.Series(s.values)
            continue

    if not s_load:
        raise RuntimeError("All house CSVs were skipped – none expose load/solar. Update keyword lists if needed.")

    H = max(len(v) for v in s_load.values())
    idx = pd.RangeIndex(0, H, name="hour_idx")
    load_w = pd.DataFrame(index=idx, columns=sorted(s_load)).astype(float)
    solar_w = pd.DataFrame(index=idx, columns=sorted(s_solar)).astype(float)
    for hid in load_w.columns:
        load_w[hid] = s_load[hid].values
        solar_w[hid] = s_solar[hid].values
    return load_w, solar_w


def derive_prices_charges_from_soc(frames: Dict[int, pd.DataFrame], soc_df: pd.DataFrame):
    prices: Dict[int, List[float]] = {}
    for hid, df in frames.items():
        price_col = next((c for c in df.columns if c == "Electricity_price_watt"), None)
        if price_col is None:
            prices[hid] = [0.0] * len(df)
        else:
            prices[hid] = df[price_col].astype(float).tolist()

    n_hours = len(soc_df.index)
    chargesDf: Dict[int, List[Dict[str, float]]] = {hid: [] for hid in soc_df.columns}
    dischargesDf: Dict[int, List[Dict[str, float]]] = {hid: [] for hid in soc_df.columns}
    for h in range(n_hours):
        for hid in soc_df.columns:
            if h == 0:
                delta = 0.0
            else:
                delta = float(soc_df.at[h, hid] - soc_df.at[h-1, hid])
            charge_amt = max(delta, 0.0)
            discharge_amt = max(-delta, 0.0)
            chargesDf[hid].append({"client": int(hid), "amount": float(charge_amt)})
            dischargesDf[hid].append({"client": int(hid), "amount": float(discharge_amt)})

    # timestamp from first house with a datetime-like column
    hour_dt = None
    for df in frames.values():
        dt_col = next((c for c in df.columns if _norm(c) in {"datetime", "timestamp", "date_time", "date", "time", "dt"}), None)
        if dt_col is not None:
            hour_dt = pd.to_datetime(df[dt_col], utc=True).dt.tz_convert(None).reset_index(drop=True)
            break

    capacity = {}
    for hid, df in frames.items():
        col = next((c for c in df.columns if _norm(c) == "batterycapacitykw"), None)
        capacity[hid] = float(df[col].iloc[0]) if col else soc_df[hid].max()

    # convert dicts to lists aligned by sorted house ids
    house_ids: List[int] = sorted([int(h) for h in soc_df.columns])
    prices_list = [prices.get(hid, [0.0]) for hid in house_ids]
    charges_list = [[{"client": idx, "amount": entry["amount"]} for entry in chargesDf[hid]] for idx, hid in enumerate(house_ids)]
    discharges_list = [[{"client": idx, "amount": entry["amount"]} for entry in dischargesDf[hid]] for idx, hid in enumerate(house_ids)]

    return prices_list, charges_list, discharges_list, capacity, hour_dt


