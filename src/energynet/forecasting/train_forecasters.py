#!/usr/bin/env python3
"""
train_forecasters.py
───────────────────────────────────────────────────────────────────
Train global CatBoost models to predict **hour‑ahead** load and
solar generation (kWh) for every house. Models are evaluated on an
*unseen* chronological **test set** and the resulting metrics (MAE,
RMSE) are written to CSV.  Optionally, full test‑set predictions can
also be exported.

Outputs
=======
models/
  ├─ load_kwh_catboost.cbm
  └─ solar_kwh_catboost.cbm
metrics/test_metrics.csv
predictions/ (only when --dump-test)
  ├─ load_kwh_test.csv
  └─ solar_kwh_test.csv
"""

from __future__ import annotations

# ── Standard library ──────────────────────────────────────────────
import argparse
import re
from pathlib import Path

# ── Third‑party ───────────────────────────────────────────────────
import numpy as np
import pandas as pd
from catboost import CatBoostRegressor, Pool
from sklearn.metrics import mean_absolute_error, mean_squared_error
from tqdm import tqdm

# ── Local modules ─────────────────────────────────────────────────
from ..data_loader import read_all_generated_data

# ── Hyper‑parameters & constants ──────────────────────────────────
DT_COL_CANDIDATES = ["datetime", "Datetime", "timestamp", "Timestamp"]
CAT_COLS: list[str] = ["hour", "dow", "month"]
LAGS = range(1, 49)     # up to 48‑hour lag
ROLLS = [6, 12, 24]     # rolling‑mean windows
SEED = 42

# ───────────────────────────────────────────────────────────────────
# Utility helpers
# ───────────────────────────────────────────────────────────────────


def _find_datetime_col(df: pd.DataFrame) -> str:
    for c in DT_COL_CANDIDATES:
        if c in df.columns:
            return c
    raise KeyError(f"No datetime column found; expected one of {DT_COL_CANDIDATES}")


def ensure_datetime(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure the datetime column is named 'datetime' and is of dtype datetime64."""
    col = _find_datetime_col(df)
    if col != "datetime":
        df = df.rename(columns={col: "datetime"})
    if not np.issubdtype(df["datetime"].dtype, np.datetime64):
        df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce")
    return df.dropna(subset=["datetime"]).copy()


def tidy_all_folders(fraction: float) -> pd.DataFrame:
    """Load a fraction of the *Generated Data* and return a tidy DataFrame."""
    all_data, _ = read_all_generated_data(verbose=True, fraction=fraction)
    frames: list[pd.DataFrame] = []

    for folder in tqdm(all_data.values(), desc="Tidying houses"):
        for key, df in folder.items():
            m = re.match(r"folder\d+_house(\d+)", key)
            if not m:
                continue
            hid = int(m.group(1))
            d = ensure_datetime(df)[
                ["datetime", "solar_hourly_watt_generation_house", "hourly_load_kw_house"]
            ].copy()
            d["house_id"] = hid
            d["solar_kwh"] = d["solar_hourly_watt_generation_house"] / 1000.0
            d["load_kwh"] = d["hourly_load_kw_house"]
            frames.append(d[["datetime", "house_id", "solar_kwh", "load_kwh"]])

    return pd.concat(frames, ignore_index=True)


def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add calendar time features and lag / rolling statistics."""
    base = df.copy()
    base["hour"] = base["datetime"].dt.hour.astype("int16")
    base["dow"] = base["datetime"].dt.dayofweek.astype("int16")
    base["month"] = base["datetime"].dt.month.astype("int16")

    feat_dict: dict[str, pd.Series] = {}
    gb_load = base.groupby("house_id")["load_kwh"]
    gb_solar = base.groupby("house_id")["solar_kwh"]

    for lag in LAGS:
        feat_dict[f"load_lag{lag}"] = gb_load.shift(lag)
        feat_dict[f"solar_lag{lag}"] = gb_solar.shift(lag)
    for r in ROLLS:
        feat_dict[f"load_mean{r}"] = gb_load.shift(1).rolling(r).mean()
        feat_dict[f"solar_mean{r}"] = gb_solar.shift(1).rolling(r).mean()

    return pd.concat([base, pd.DataFrame(feat_dict)], axis=1).dropna().reset_index(drop=True)


# ───────────────────────────────────────────────────────────────────
# Chronological split helper
# ───────────────────────────────────────────────────────────────────


def chron_split(df: pd.DataFrame, train: float = 0.70, val: float = 0.15):
    """Return boolean masks for train / val / test chronological split."""
    t1 = df["datetime"].quantile(train)
    t2 = df["datetime"].quantile(train + val)
    train_mask = df["datetime"] < t1
    val_mask = (df["datetime"] >= t1) & (df["datetime"] < t2)
    test_mask = df["datetime"] >= t2
    return train_mask, val_mask, test_mask


# ───────────────────────────────────────────────────────────────────
# Model training + evaluation
# ───────────────────────────────────────────────────────────────────


def train_and_eval(df: pd.DataFrame, target: str, dump_pred: bool):
    """Train a CatBoost model and evaluate on the chronological test split."""
    train_m, val_m, test_m = chron_split(df)

    features = df.drop(columns=["datetime", "solar_kwh", "load_kwh"])

    X_tr, y_tr = features[train_m], df.loc[train_m, target]
    X_val, y_val = features[val_m], df.loc[val_m, target]
    X_te, y_te = features[test_m], df.loc[test_m, target]

    model = CatBoostRegressor(
        loss_function="RMSE",
        iterations=800,
        depth=6,
        learning_rate=0.05,
        random_seed=SEED,
        early_stopping_rounds=50,
        verbose=100,
    )
    model.fit(
        Pool(X_tr, y_tr, cat_features=CAT_COLS),
        eval_set=Pool(X_val, y_val, cat_features=CAT_COLS),
    )

    Path("models").mkdir(exist_ok=True)
    model.save_model(Path("models") / f"{target}_catboost.cbm")

    test_pred = model.predict(X_te)

    mae = mean_absolute_error(y_te, test_pred)
    rmse = np.sqrt(mean_squared_error(y_te, test_pred))

    dump_df = None
    if dump_pred:
        dump_df = df.loc[test_m, ["datetime", "house_id"]].assign(
            actual=y_te, pred=test_pred
        )

    return mae, rmse, dump_df


# ───────────────────────────────────────────────────────────────────
# CLI wrapper
# ───────────────────────────────────────────────────────────────────


def main(fraction: float, dump_test: bool):
    tidy = add_time_features(tidy_all_folders(fraction))

    metrics: dict[str, dict[str, float]] = {}
    dump_frames: dict[str, pd.DataFrame] = {}

    for target in ["load_kwh", "solar_kwh"]:
        print(f"\n▶ Training model for {target} …")
        mae, rmse, dump_df = train_and_eval(tidy, target, dump_test)
        metrics[target] = {"mae": mae, "rmse": rmse}
        if dump_df is not None:
            dump_frames[target] = dump_df

    # Save metrics
    Path("metrics").mkdir(exist_ok=True)
    pd.DataFrame(metrics).T.to_csv("metrics/test_metrics.csv", float_format="%.4f")

    # Save predictions if requested
    if dump_test and dump_frames:
        Path("predictions").mkdir(exist_ok=True)
        for target, df_ in dump_frames.items():
            df_.to_csv(Path("predictions") / f"{target}_test.csv", index=False)

    print("\n✅ All done! Metrics written to metrics/test_metrics.csv")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train CatBoost forecasters on EnergyNet data"
    )
    parser.add_argument(
        "--fraction",
        type=float,
        default=0.05,
        help="Fraction of folders to load (0–1)",
    )
    parser.add_argument(
        "--dump-test",
        action="store_true",
        help="Dump full test‑set predictions to CSV",
    )
    args = parser.parse_args()

    main(fraction=args.fraction, dump_test=args.dump_test)
