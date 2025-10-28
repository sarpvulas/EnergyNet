from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd


def _norm(x) -> str:
    return str(x).lower().replace(" ", "").replace("_", "")


def discover_generated_folder(folder_num: int) -> Path:
    try:
        import config  # type: ignore
        base = Path(config.BASE_DIR)
    except (ModuleNotFoundError, AttributeError):
        base = Path.cwd()
    cand = base / f"Generated Data - {folder_num}"
    if not cand.exists():
        raise FileNotFoundError(f"Cannot find data folder: {cand}")
    return cand


def _keep_only_houses(raw: Dict) -> Dict[int, pd.DataFrame]:
    out: Dict[int, pd.DataFrame] = {}
    for k, v in raw.items():
        if isinstance(k, int):
            out[k] = v
        elif isinstance(k, str) and (m := re.search(r"house(\d+)", k.lower())):
            out[int(m.group(1))] = v
    return out


def load_house_csv_frames(folder: int) -> Dict[int, pd.DataFrame]:
    try:
        from energynet.data_loader import read_data_from_generated_folder  # type: ignore
        return _keep_only_houses(read_data_from_generated_folder(folder))
    except Exception:
        pass
    root = discover_generated_folder(folder)
    out: Dict[int, pd.DataFrame] = {}
    for csv in root.glob("house*.csv"):
        if m := re.search(r"house(\d+)", csv.stem, flags=re.I):
            out[int(m.group(1))] = pd.read_csv(csv)
    if not out:
        raise RuntimeError(f"No house*.csv in {root}")
    return out


def read_soc_timeseries(results_dir: Path, prefer_barters: bool = False) -> Tuple[pd.DataFrame, pd.Series | None]:
    # Allow switching between different SoC series if present
    barters_path = results_dir / "soc_timeseries_barters.csv"
    trades_path = results_dir / "soc_timeseries_trades.csv"
    default_path = results_dir / "soc_timeseries.csv"
    if prefer_barters and barters_path.exists():
        soc_path = barters_path
    elif (not prefer_barters) and trades_path.exists():
        soc_path = trades_path
    else:
        soc_path = default_path
    soc = pd.read_csv(soc_path)
    ts_col = next((c for c in soc.columns if _norm(c) in {"timestamp", "datetime", "date_time"}), None)
    ts = None
    if ts_col is not None:
        ts = (
            soc[["hour_idx", ts_col]].drop_duplicates("hour_idx").set_index("hour_idx")[ts_col]
        )
    wide = soc.pivot(index="hour_idx", columns="house_id", values="soc_kWh")
    wide.sort_index(axis=1, inplace=True)
    return wide, ts


def read_hourly_trades(results_dir: Path) -> List[List[dict]]:
    with open(results_dir / "trades_hourly.json") as f:
        return json.load(f)


def read_barters_csv(results_dir: Path) -> pd.DataFrame:
    path = results_dir / "barters.csv"
    if not path.exists():
        return pd.DataFrame(columns=["t", "seller", "buyer", "Es_kWh", "eta", "claim_kWh", "expiry_t"])
    df = pd.read_csv(path)
    # Ensure expected columns exist
    req = {"t", "seller", "buyer", "Es_kWh", "eta", "claim_kWh"}
    missing = req - set(df.columns)
    if missing:
        raise RuntimeError(f"barters.csv missing columns: {missing}")
    return df


def read_claim_returns_csv(results_dir: Path) -> pd.DataFrame:
    path = results_dir / "claim_returns.csv"
    if not path.exists():
        return pd.DataFrame(columns=["t", "owner", "stored_on", "buyer", "qty_kWh"])
    df = pd.read_csv(path)
    req = {"t", "owner", "stored_on", "buyer", "qty_kWh"}
    missing = req - set(df.columns)
    if missing:
        raise RuntimeError(f"claim_returns.csv missing columns: {missing}")
    return df


def read_claim_expiries_csv(results_dir: Path) -> pd.DataFrame:
    path = results_dir / "claim_expiries.csv"
    if not path.exists():
        return pd.DataFrame(columns=["t", "owner", "stored_on", "qty_kWh"])
    df = pd.read_csv(path)
    req = {"t", "owner", "stored_on", "qty_kWh"}
    missing = req - set(df.columns)
    if missing:
        raise RuntimeError(f"claim_expiries.csv missing columns: {missing}")
    return df

