## EnergyNet — Simulator, Viewer, and Forecasters

This project provides:

- `energynet.barter_sim`: a discrete‑time simulator for peer‑to‑peer trading and bartering with batteries.
- `energynet.viewer`: an interactive desktop viewer to explore hourly states and bartering activity.
- `energynet.forecasting`: CatBoost models for 1‑hour‑ahead load and solar forecasting.

The repository uses a `src/` layout and exposes console scripts for easy use.

### Installation

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e .
```

This registers two commands in your environment: `energynet-barter` and `energynet-viewer`.

### Data prerequisites

- Input data is expected under your generated data root (used by `energynet.data_loader`).
- Simulation results are written under `results/folder_<n>/` (created on demand).

---

## Simulator CLI — `energynet-barter` (or `python -m energynet.barter_sim.cli`)

Run the non‑MILP simulator over all hours for a folder.

```bash
energynet-barter --folder 1 \
  --algo P2P_EB \
  --eta 0.6 \
  --tau 6 \
  --expiry-action return_to_lender \
  --price-window 3 \
  --priority-rule claims_first \
  --balance-mode percent_of_grid_bill \
  --balance-percent 5.0 \
  --out results \
  --seed 42 \
  --verbose
```

### Arguments (detailed)
- `--folder <int>`: Results will be written to `results/folder_<n>/`.
- `--algo {T_AND_B,C_EB,P2P_EB,S_EB,AI_P2P_EB}`: Trading/Bartering mode.
  - `T_AND_B`: trading + batteries only; no barters created.
  - `C_EB`: centralized bartering; coordinator (-1) owns claims.
  - `P2P_EB`: decentralized bartering; lenders own claims and can sell later.
  - `S_EB`: self‑prioritized bartering; reclaim own claims to serve load first.
  - `AI_P2P_EB`: as P2P_EB but caps bartered energy via a predictor.
- `--eta <float>`: Fraction of a barter consumed immediately by the receiver (0–1).
- `--tau <int>`: Claim lifetime in hours; after `expiry_t = t + tau` it lapses.
- `--expiry-action {consumer_keeps,return_to_lender}`: What to do at expiry.
  - `consumer_keeps` (default): lapsed energy stays at the consumer; logged as expiry.
  - `return_to_lender`: attempt physical return to the owner, bounded by
    consumer SoC and owner free capacity; any remainder is logged as expiry.
- `--price-window <int>`: Smoother window for dynamic price formation.
- `--priority-rule claims_first`: Sells existing claims before new P2P trades.
- `--balance-mode {fixed,percent_of_grid_bill}` & `--balance-percent <float>`:
  initial buyer allowances used to constrain monetary trades.
- `--ai-model-path <path>`: Optional model path for `AI_P2P_EB`.
- `--out <dir>`: Output root (default: `results`).
- `--seed <int>`: Random seed.
- `--verbose`: Print summary statistics to stdout.

### Outputs (under `results/folder_<n>/`)
- `grid_import.csv`: total grid draw by house.
- `p2p_trades.csv`: all monetary trades (time, seller, buyer, kWh, price).
- `barters.csv`: all barters (time, seller, buyer, Es_kWh, eta, claim_kWh, expiry_t).
- `claim_returns.csv`: physical redemptions of claims (time, owner, stored_on, buyer, qty_kWh).
- `claim_expiries.csv`: quantities that lapsed at expiry (time, owner, stored_on, qty_kWh).
- `claims_open.csv`: claims still alive at the end of the run.
- `paid_earned.csv`: money paid/earned by each house.
- `soc_timeseries.csv`: SoC per house per hour (used by the viewer).
- `summary.json`, `logs.txt`.

Tips:
- If you expect returns specifically at expiry, use `--expiry-action return_to_lender`.
- If many expiries show up, it often means the owner had no spare capacity at those hours.

---

## Viewer — `energynet-viewer`

Open the interactive viewer for a simulation folder.

```bash
energynet-viewer --folder 1
# or
python -m energynet.viewer.main --folder 1
```

What you can do:
- Navigate time with the left/right buttons or the slider.
- Switch between `Trades` and `Barters` modes.
- Inspect the Barters Table (button at the top‑right):
  - Filter by hour/seller/buyer/min Es_kWh.
  - Toggle inclusion of purple “returns” and orange “expiries”.
  - Sort by any column; export the filtered rows to CSV.
  - Open a dedicated Returns table.
- Hover any bar to see numeric tooltips; click arrows for detailed popups.
- In Barters mode, stored‑claim badges appear as colored squares around consumers; hover to see seller and kWh.

Legend:
- Green arrows: bartered energy actually used immediately (η·Es).
- Purple arrows: claim redemptions (either sold or returned at expiry under `return_to_lender`).
- Orange self‑loops: expired energy that remained with the consumer.

---

## Forecasters — `python -m energynet.forecasting.train_forecasters`

Train and evaluate CatBoost models for hour‑ahead load and solar across all houses.

```bash
python -m energynet.forecasting.train_forecasters \
  --fraction 0.10 \
  --dump-test
```

Arguments:
- `--fraction <0..1>`: fraction of generated folders to load (speeds up experiments).
- `--dump-test`: save full test‑set predictions to `predictions/`.

Outputs:
- `models/load_kwh_catboost.cbm`, `models/solar_kwh_catboost.cbm`.
- `metrics/test_metrics.csv` (MAE, RMSE).
- `predictions/*_test.csv` when `--dump-test` is set.

---

## Development

```bash
pip install -e .
pip install -r requirements.txt  # if you plan to retrain forecasters
pip install pre-commit && pre-commit install

# Run tests (subset lives under src/energynet/barter_sim/tests)
pytest -q
```

### Repository layout (selected)
- `src/energynet/barter_sim/`: simulator code (engine, matching, IO, CLI, tests).
- `src/energynet/viewer/`: interactive viewer (Tkinter + Matplotlib).
- `src/energynet/forecasting/`: model training utilities.
- `results/folder_<n>/`: simulation outputs consumed by the viewer.

---

## Notes on expiry semantics
- A claim created at hour `t` with `--tau k` is redeemable through hour `t + k`.
- After that, it lapses. Under `consumer_keeps`, lapsed energy stays at the consumer.
- Under `return_to_lender`, we first try to return up to the minimum of
  remaining claim quantity, consumer SoC and owner’s free capacity; any remainder
  is logged as an expiry. This processing happens at the start of each hour, so
  the consumer cannot “use” expired energy before a return is attempted.


