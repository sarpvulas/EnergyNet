from __future__ import annotations

from typing import Dict, Tuple, List, Optional
import re

import pandas as pd

from energynet.data_loader import read_data_from_generated_folder


# Simple in-module cache to prevent re-reading the same folder repeatedly
_HOUSES_CACHE: Dict[int, Dict[int, pd.DataFrame]] = {}
_MB_CACHE: Dict[int, Optional[pd.DataFrame]] = {}


def _keep_only_houses(data: Dict[str, pd.DataFrame]) -> Dict[int, pd.DataFrame]:
    pat = re.compile(r"folder\d+_house(\d+)$")
    out: Dict[int, pd.DataFrame] = {}
    for key, df in data.items():
        m = pat.match(key)
        if not m:
            continue
        hid = int(m.group(1))
        d = df.copy()
        d["hour_idx"] = range(len(d))
        out[hid] = d
    if not out:
        raise ValueError("No house CSV frames detected. Check folder naming.")
    return out


def load_hour_inputs(folder: int, t: int) -> Tuple[Dict[int, float], Dict[int, float], Dict[int, float]]:
    """
    For step t, return dicts: load_kWh, solar_kWh, battery_cap_kWh.
    Assumes hourly rows. Converts W to kW then to kWh where relevant.
    """
    if folder not in _HOUSES_CACHE:
        data = read_data_from_generated_folder(folder)
        _HOUSES_CACHE[folder] = _keep_only_houses(data)
        _MB_CACHE[folder] = data.get("monthly_balances") if "monthly_balances" in data else None
    houses = _HOUSES_CACHE[folder]

    load_kWh: Dict[int, float] = {}
    solar_kWh: Dict[int, float] = {}
    cap_kWh: Dict[int, float] = {}

    for hid, df in houses.items():
        if t >= len(df):
            continue
        row = df.iloc[t]
        # load is already hourly in kW; convert to kWh over 1 hour
        load_kw = float(row.get("hourly_load_kw_house", 0.0))
        load_kWh[hid] = max(0.0, load_kw * 1.0)

        # solar generation hourly is given in watts; convert to kWh for 1 hour
        solar_w = float(row.get("solar_hourly_watt_generation_house", 0.0))
        solar_kWh[hid] = max(0.0, (solar_w / 1000.0) * 1.0)

        cap_kWh[hid] = float(row.get("Battery_capacity_kw", 0.0))

    return load_kWh, solar_kWh, cap_kWh


def load_price_series(folder: int, t: int) -> Tuple[float, float]:
    """Return utility price u_t and FiT_t for hour t.
    If not present, synthesise simple diurnal pattern.
    """
    if folder not in _HOUSES_CACHE:
        data = read_data_from_generated_folder(folder)
        _HOUSES_CACHE[folder] = _keep_only_houses(data)
        _MB_CACHE[folder] = data.get("monthly_balances") if "monthly_balances" in data else None
    houses = _HOUSES_CACHE[folder]

    # Use the first house as proxy for price series if available
    sample = houses[sorted(houses.keys())[0]]
    if "to_grid_prices" in sample.columns and "Electricity_price_watt" in sample.columns:
        if t < len(sample):
            fit = float(sample.iloc[t]["to_grid_prices"])  # already unit price
            util = float(sample.iloc[t]["Electricity_price_watt"])  # proxy utility
            return util, fit

    # Fallback simple diurnal cycle
    hour_of_day = t % 24
    fit = 0.05 + 0.03 * (1.0 if 10 <= hour_of_day <= 16 else 0.0)
    util = 0.20 + 0.10 * (hour_of_day in (18, 19, 20))
    return util, fit


def load_initial_balances(folder: int, mode: str = "percent_of_grid_bill", percent: float = 5.0) -> Dict[int, float]:
    """
    Provide initial P2P spending balance per peer.
    If monthly_balances.csv exists and mode == percent_of_grid_bill,
    allocate 'percent' percent of the historical grid spend.
    """
    if folder not in _HOUSES_CACHE:
        data = read_data_from_generated_folder(folder)
        _HOUSES_CACHE[folder] = _keep_only_houses(data)
        _MB_CACHE[folder] = data.get("monthly_balances") if "monthly_balances" in data else None
    houses = _HOUSES_CACHE[folder]
    balances: Dict[int, float] = {hid: 0.0 for hid in houses}

    mb = _MB_CACHE.get(folder)
    if mb is not None and mode == "percent_of_grid_bill":
        # Expect rows indexed by house_id with a column named 'grid_spend' or similar
        col = None
        for candidate in ("grid_spend", "grid_bill", "total_spend", mb.columns[0] if len(mb.columns) == 1 else None):
            if candidate in mb.columns:
                col = candidate
                break
        if col is not None:
            for hid in balances.keys():
                if hid in mb.index:
                    balances[hid] = float(mb.loc[hid, col]) * (percent / 100.0)
    return balances


def load_all_hourly_series(folder: int) -> Tuple[List[Dict[int, float]], List[Dict[int, float]], Dict[int, float], List[float], List[float]]:
    """
    Load the folder once and return:
    - loads_series: List[Dict[house_id, kWh]] per hour
    - solars_series: List[Dict[house_id, kWh]] per hour
    - battery_cap_kWh: Dict[house_id, kWh]
    - util_prices: List[float] per hour
    - fit_prices: List[float] per hour
    """
    if folder not in _HOUSES_CACHE:
        data = read_data_from_generated_folder(folder)
        _HOUSES_CACHE[folder] = _keep_only_houses(data)
        _MB_CACHE[folder] = data.get("monthly_balances") if "monthly_balances" in data else None
    houses = _HOUSES_CACHE[folder]

    if not houses:
        return [], [], {}, [], []

    # Determine time horizon by the first house
    first_key = sorted(houses.keys())[0]
    T = len(houses[first_key])

    caps: Dict[int, float] = {}
    for hid, df in houses.items():
        if not df.empty:
            caps[hid] = float(df.iloc[0].get("Battery_capacity_kw", 0.0))

    loads_series: List[Dict[int, float]] = []
    solars_series: List[Dict[int, float]] = []
    util_prices: List[float] = []
    fit_prices: List[float] = []

    for t in range(T):
        load_kWh: Dict[int, float] = {}
        solar_kWh: Dict[int, float] = {}
        for hid, df in houses.items():
            if t >= len(df):
                continue
            row = df.iloc[t]
            load_kw = float(row.get("hourly_load_kw_house", 0.0))
            load_kWh[hid] = max(0.0, load_kw)
            solar_w = float(row.get("solar_hourly_watt_generation_house", 0.0))
            solar_kWh[hid] = max(0.0, solar_w / 1000.0)
        loads_series.append(load_kWh)
        solars_series.append(solar_kWh)

        # Prices from sample house if present, else synthetic
        sample = houses[first_key]
        if "to_grid_prices" in sample.columns and "Electricity_price_watt" in sample.columns:
            fit = float(sample.iloc[t]["to_grid_prices"]) if t < len(sample) else 0.05
            util = float(sample.iloc[t]["Electricity_price_watt"]) if t < len(sample) else 0.20
        else:
            hour_of_day = t % 24
            fit = 0.05 + 0.03 * (1.0 if 10 <= hour_of_day <= 16 else 0.0)
            util = 0.20 + 0.10 * (hour_of_day in (18, 19, 20))
        util_prices.append(util)
        fit_prices.append(fit)

    return loads_series, solars_series, caps, util_prices, fit_prices


