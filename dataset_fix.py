# dataset_fix.py  (keep this in your project root)

from __future__ import annotations
import os, re
import numpy as np, pandas as pd
from typing import Dict, List

__all__: List[str] = ["fix_missing_energy_data", "write_fixed_data"]

# timestamps to repair
_SINGLE_TS = pd.Timestamp("2015-01-05 03:00:00+01:00")
_RANGE = pd.date_range("2015-01-05 12:00:00+01:00", "2015-01-05 17:00:00+01:00", freq="h")
_COLS = ["solar_hourly_watt_generation_house",
         "hourly_load_kw_house",
         "hourly_load_watt_house"]

def _mean(a, b):
    if np.isnan(a) and np.isnan(b): return np.nan
    if np.isnan(a): return b
    if np.isnan(b): return a
    return (a + b) / 2

def _update_excess(df: pd.DataFrame, idx):
    df.loc[idx, "Excess_energy_watt"] = (
        df.loc[idx, "hourly_load_watt_house"] -
        df.loc[idx, "solar_hourly_watt_generation_house"]
    )

def fix_missing_energy_data(all_data: Dict[int, Dict[str, pd.DataFrame]], verbose=True):
    fixed = 0
    for houses in all_data.values():
        for df in houses.values():
            df.sort_values("datetime", inplace=True, ignore_index=True)
            m_single = df["datetime"] == _SINGLE_TS
            if m_single.any():
                i = m_single.idxmax()
                if pd.isna(df.at[i, _COLS[0]]):
                    df.at[i, _COLS[0]] = _mean(df.at[i-1, _COLS[0]], df.at[i+1, _COLS[0]])
                    _update_excess(df, i); fixed += 1
            m_range = df["datetime"].isin(_RANGE)
            if m_range.any():
                df[_COLS] = df[_COLS].interpolate(method="linear", limit_direction="both")
                _update_excess(df, df.index[m_range]); fixed += int(m_range.sum())
    if verbose: print(f"🔧 fixed {fixed} row(s)")
    return all_data

def write_fixed_data(all_data: Dict[int, Dict[str, pd.DataFrame]],
                     base_folder: str = "/Users/sarpvulas/Datasets/energynetdata/icc_combined",
                     verbose=True):
    count = 0
    for folder_no, houses in all_data.items():
        folder_path = os.path.join(base_folder, f"Generated Data - {folder_no}")
        for key, df in houses.items():
            m = re.match(r"folder\d+_house(\d+)", key)
            if not m: continue
            file_path = os.path.join(folder_path, f"house{m.group(1)}.csv")
            df.to_csv(file_path, index=False)
            count += 1
            if verbose: print(f"📝 wrote {file_path}")
    if verbose: print(f"✅ updated {count} CSV file(s)")
    return count

if __name__ == "__main__":
    from data_loader import read_all_generated_data
    data, _ = read_all_generated_data(verbose=False, fraction=1)
    fix_missing_energy_data(data)
    write_fixed_data(data)
