#!/usr/bin/env python3
# train_keep_reserve_model.py
# ------------------------------------------------------------------
# • Label  = 24-h sum of the *raw* hourly deficit  (before battery use)
# • Features = present storage state  + 6-h expected surplus  + seasonality
# • Head-room is NOT given as a feature (model must learn < headroom)
# ------------------------------------------------------------------

import os, numpy as np, pandas as pd
from typing import List
from catboost import CatBoostRegressor, Pool
from sklearn.metrics import mean_absolute_error, mean_squared_error
from data_loader     import read_all_generated_data
from role_identifier import classify_house_roles

from math import sqrt


# ---------------------------------------------------------------------
# 1. Load data  (use 10 % sample while iterating; set to 1.0 for full run)
# ---------------------------------------------------------------------
all_data, _ = read_all_generated_data(verbose=False, fraction=0.1)
prosumers = set(classify_house_roles(all_data)['prosumers'])

records: List[pd.DataFrame] = []
HORIZON = 24   # target horizon (h)
LOOK6   = 6    # short-term outlook (h)

for folder in all_data.values():
    for house, df in folder.items():
        if house not in prosumers:
            continue

        d = df.copy()
        d['datetime'] = pd.to_datetime(d['datetime'], utc=True)
        d.sort_values('datetime', inplace=True, ignore_index=True)

        # ── base kWh series ───────────────────────────────────────────
        d['solar_kWh'] = d['solar_hourly_watt_generation_house'] / 1000.0
        d['load_kWh']  = d['hourly_load_kw_house']

        # optional numeric columns (fill if absent)
        for col in ('Electricity_price_watt', 'Excess_energy_watt'):
            d[col] = d.get(col, 0.0)

        # ── raw hourly deficit  (before battery/credit) ───────────────
        d['raw_deficit'] = (d['load_kWh'] - d['solar_kWh']).clip(lower=0)

        # ── barter simulation to update battery / credit *states* ─────
        d[['battery_state_kWh', 'credit_kWh']] = np.nan
        batt = d.at[0, 'Battery_charge_kw']
        cred = 0.0
        cap0 = d.at[0, 'Battery_capacity_kw']

        for i in range(len(d)):
            load  = d.at[i, 'load_kWh']
            solar = d.at[i, 'solar_kWh']
            cap   = d.at[i, 'Battery_capacity_kw'] if 'Battery_capacity_kw' in d else cap0
            balance = solar - load

            if balance > 0:                              # surplus
                charge = min(balance, cap - batt)
                batt  += charge
                cred  += balance - charge                # credited to others
            else:                                        # deficit
                need  = -balance
                fb = min(need, batt); batt -= fb; need -= fb
                fc = min(need, cred); cred -= fc; need -= fc
                # any remaining need would come from grid (not used here)

            d.at[i, 'battery_state_kWh'] = batt
            d.at[i,  'credit_kWh']       = cred

        # ── label: 24-h forward *raw* grid need (no head-room cap) ────
        d['grid_need_24h'] = (
            d['raw_deficit'].shift(-1).rolling(HORIZON, min_periods=HORIZON).sum()
        )

        # ── feature: expected surplus next 6 h  ───────────────────────
        d['exp_surplus_6h'] = (
            (d['solar_kWh'] - d['load_kWh'])
              .shift(-1).rolling(LOOK6, min_periods=1).sum()
        )

        # ── derived battery % fullness ────────────────────────────────
        d['battery_pct'] = (
            100 * d['battery_state_kWh'] /
            d['Battery_capacity_kw'].replace(0, np.nan)
        )

        # ── seasonality features ──────────────────────────────────────
        m = d['datetime'].dt.month
        w = d['datetime'].dt.dayofweek
        d['month_sin'] = np.sin(2*np.pi*m/12); d['month_cos'] = np.cos(2*np.pi*m/12)
        d['dow_sin']   = np.sin(2*np.pi*w/7 ); d['dow_cos']   = np.cos(2*np.pi*w/7)

        # rolling means
        for win in (24, 48, 168):
            d[f'load_mean_{win}h']  = d['load_kWh'] .rolling(win,1).mean().shift(1)
            d[f'solar_mean_{win}h'] = d['solar_kWh'].rolling(win,1).mean().shift(1)

        d['house'] = house
        records.append(d)

# ---------------------------------------------------------------------
# 2. Combine & clean
# ---------------------------------------------------------------------
df = pd.concat(records, ignore_index=True)
df = df.dropna(subset=['grid_need_24h'])     # need full look-ahead label
df.fillna(0.0, inplace=True)                # CatBoost can handle zeros

# ---------------------------------------------------------------------
# 3. Chronological split
# ---------------------------------------------------------------------
df.sort_values('datetime', inplace=True)
train_end, val_end = '2017-12-31', '2018-06-30'
train = df[df['datetime'] <= train_end]
val   = df[(df['datetime'] > train_end) & (df['datetime'] <= val_end)]
test  = df[df['datetime'] > val_end]

feature_cols = [
    'solar_kWh','load_kWh','Electricity_price_watt',
    'battery_state_kWh','battery_pct','credit_kWh',
    'exp_surplus_6h',
    'month_sin','month_cos','dow_sin','dow_cos',
    'load_mean_24h','solar_mean_24h',
]

X_train, y_train = train[feature_cols], train['grid_need_24h']
X_val,   y_val   = val  [feature_cols], val  ['grid_need_24h']
X_test,  y_test  = test [feature_cols], test ['grid_need_24h']

# ---------------------------------------------------------------------
# 4. Train CatBoost
# ---------------------------------------------------------------------
model = CatBoostRegressor(
    iterations     = 1500,
    learning_rate  = 0.05,
    loss_function  = 'MAE',
    eval_metric    = 'MAE',
    random_seed    = 42,
    verbose        = 100
)
model.fit(
    Pool(X_train, y_train),
    eval_set      = Pool(X_val, y_val),
    use_best_model=True
)

# ---------------------------------------------------------------------
# 5. Evaluate
# ---------------------------------------------------------------------
preds = model.predict(X_test)
print(f"Test MAE  (24h): {mean_absolute_error(y_test, preds):.2f} kWh")

mse  = mean_squared_error(y_test, preds)     # plain MSE
rmse = sqrt(mse)                             # convert to RMSE
print(f"Test RMSE (24h): {rmse:.2f} kWh")

# ---------------------------------------------------------------------
# 6. Save
# ---------------------------------------------------------------------
os.makedirs("pretrained_models", exist_ok=True)
model_path = "pretrained_models/catboost_keep_reserve_24h.cbm"
model.save_model(model_path)
print(f"✔ model saved to {model_path}")
