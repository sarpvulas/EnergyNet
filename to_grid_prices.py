#!/usr/bin/env python3
"""
batch_add_to_grid_prices.py (finalised)
────────────────────────────────────────────────────────
• **Deletes** any previously created `*_priced.csv` files.
• Adds **``to_grid_prices`` = ``Electricity_price_watt`` / 3** to every
  original `house*.csv` and **overwrites the file in‑place**.

Usage
─────
    # default – scans the canonical dataset root
    python batch_add_to_grid_prices.py

    # point to a different root directory
    python batch_add_to_grid_prices.py /some/other/root
"""
from __future__ import annotations

import os
import re
import sys
from pathlib import Path

import pandas as pd

# ───────────────────────── helper: add column ─────────────────────────

def add_to_grid_prices(df: pd.DataFrame, col: str = "Electricity_price_watt") -> pd.DataFrame:
    """Return *df* with a new/updated column ``to_grid_prices`` = ``col`` / 3."""
    if col not in df.columns:
        raise KeyError(f"Column '{col}' not found in DataFrame")
    df["to_grid_prices"] = df[col] / 3.0
    return df

# ───────────────────── default data directory logic ───────────────────

_DEFAULT_PATH = Path("/Users/sarpvulas/Datasets/energynetdata/icc_combined")

def default_data_dir() -> Path:
    return Path(os.getenv("ENERGENET_DATA_DIR", _DEFAULT_PATH))

# ───────────────────────── housekeeping helpers ──────────────────────

def delete_priced_files(root: Path) -> None:
    """Remove every *_priced.csv under *root* (recursive)."""
    priced_files = list(root.rglob("*_priced.csv"))
    for fp in priced_files:
        try:
            fp.unlink()
            print(f"🗑️  Deleted {fp.relative_to(root)}")
        except Exception as exc:
            print(f"❌  Failed to delete {fp}: {exc}")


def collect_house_csvs(root: Path) -> list[Path]:
    """Return house*.csv paths under *root*, excluding *_priced.csv."""
    csv_paths: list[Path] = []
    for gd in root.rglob("*"):
        if gd.is_dir() and re.match(r"Generated Data - \d+", gd.name):
            csv_paths.extend([p for p in gd.glob("house*.csv") if p.is_file() and "_priced" not in p.stem])
    return csv_paths

# ───────────────────────── file processing ───────────────────────────

def process_file(path: Path) -> None:
    try:
        df = pd.read_csv(path)
    except Exception as exc:
        print(f"❌  Failed to read {path}: {exc}")
        return

    try:
        add_to_grid_prices(df)
    except KeyError:
        print(f"⚠️   Skipping {path.name} – no 'Electricity_price_watt' column")
        return

    try:
        df.to_csv(path, index=False)
        print(f"✅  Updated {path.relative_to(path.parents[3])}")
    except Exception as exc:
        print(f"❌  Failed to write {path}: {exc}")

# ─────────────────────────────── main ────────────────────────────────

def main(argv: list[str] | None = None) -> None:
    # positional argument: optional root directory
    root_dir = Path(argv[0]) if argv else default_data_dir()

    if not root_dir.exists():
        print("🚫  Folder does not exist:", root_dir)
        sys.exit(1)

    # 1. delete *_priced.csv artefacts
    delete_priced_files(root_dir)

    # 2. update originals
    csv_files = collect_house_csvs(root_dir)
    if not csv_files:
        print("No house*.csv files found under", root_dir)
        sys.exit(1)

    for csv in csv_files:
        process_file(csv)

if __name__ == "__main__":
    main(sys.argv[1:])
