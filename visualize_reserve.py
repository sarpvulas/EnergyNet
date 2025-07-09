#!/usr/bin/env python3
"""
keep_reserve_visualiser.py
────────────────────────────────────────────────────────────
Grid-top-up rule
  • After covering a deficit, if battery ≤ 5 % of capacity,
    buy 20 % of capacity from the grid (added to `grid_draw_kWh`).

GUI workflow
  • A background thread loads every hourly CSV, runs the per-house battery
    simulation, then calls a CatBoost model to predict the 24-h reserve.
  • When that heavy work finishes, the “Plot” button becomes clickable.
  • Clicking “Plot”
        1. writes the full DataFrame to
              visualization/full_simulation_dump.csv
        2. creates a PNG for the selected prosumer and month.

PNG layout
  1. Raw 24 h need (grey dotted) · predicted keep (green) · lendable band
     (cyan fill) · battery-cap line (red dashed)
  2. Load vs. solar
  3. Credit exported (purple bars) — panel is hidden if all-zero
  4. Grid draw (red bars) — panel is hidden if all-zero
  5. Battery % (green) — auto-zooms when the line is nearly flat
"""
# ═════════════════════════════════════════════════════════════════════
import os
import threading
import queue
import tkinter as tk
from tkinter import ttk, messagebox
from typing import Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from catboost import CatBoostRegressor

from data_loader import read_all_generated_data
from role_identifier import classify_house_roles

# ───────────────────────── helper ────────────────────────────────────
def ensure(df: pd.DataFrame, col: str, default=0.0):
    """Add column `col` filled with *default* if it is missing."""
    if col not in df:
        df[col] = default

# ────────────────────── heavy builder ────────────────────────────────
def build_test_df(progress, cancelled, *, fraction=0.1) -> Optional[pd.DataFrame]:
    """
    1. Load every house’s CSVs.
    2. Simulate battery / credit hour-by-hour.
    3. Add CatBoost ‘keep reserve’ prediction.
    4. Return a single concatenated DataFrame (or None if cancelled).
    """
    say = progress

    # 1 ── load data
    say("1/7  Loading CSVs …")
    all_data, _ = read_all_generated_data(False, fraction)
    if cancelled():
        return None

    # 2 ── find prosumers
    say("2/7  Classifying roles …")
    prosumers = set(classify_house_roles(all_data)["prosumers"])
    say(f"   → {len(prosumers)} prosumers")
    if not prosumers:
        return None

    # constants
    HORIZON, LOOK6  = 24, 6        # hours
    FALLBACK_CAP    = 10.0         # kWh if capacity row is 0 / NaN / negative
    INIT_SOC        = 0.5          # 50 % initial state of charge
    TOPUP_FRAC      = 0.2          # 20 % top-up when under threshold
    EMPTY_THR       = 0.05         # 5 % threshold to trigger top-up

    recs, done = [], 0
    for folder in all_data.values():
        for house, df in folder.items():
            if house not in prosumers or cancelled():
                continue
            done += 1
            say(f"3/7  Simulating {house}  ({done}/{len(prosumers)})")

            # ── prep basic frame
            d = df.copy()
            d["datetime"] = pd.to_datetime(d["datetime"], utc=True)
            d.sort_values("datetime", inplace=True, ignore_index=True)
            for c in ("Electricity_price_watt", "Excess_energy_watt"):
                ensure(d, c, 0.0)

            d["solar_kWh"] = d["solar_hourly_watt_generation_house"] / 1000.0
            d["load_kWh"]  = d["hourly_load_kw_house"]
            if len(d) < 3:
                continue

            # ── Capacity: use ONE fixed value per house ─────────────
            raw_cap = d["Battery_capacity_kw"].astype(float)
            pos_vals = raw_cap[raw_cap > 0]
            fixed_cap = pos_vals.iloc[0] if not pos_vals.empty else FALLBACK_CAP
            cap = pd.Series(fixed_cap, index=d.index, name="Battery_capacity_kWh")

            # optional diagnostics
            d["Battery_capacity_raw_kW"] = raw_cap
            d["Battery_capacity_kWh"]    = cap

            # init states
            batt = fixed_cap * INIT_SOC   # current battery state in kWh
            cred = 0.0                    # current credit balance in kWh

            d[["battery_state_kWh","credit_kWh",
               "credit_out_kWh","grid_draw_kWh"]] = np.nan
            d["raw_deficit"] = (d["load_kWh"] - d["solar_kWh"]).clip(lower=0)

            # ── hour-by-hour bookkeeping ────────────────────────────
            for i in range(len(d)):
                load  = d.at[i, "load_kWh"]
                solar = d.at[i, "solar_kWh"]

                bal = solar - load  # positive = surplus, negative = deficit

                if bal > 0:
                    # Surplus hour: charge battery and credit leftover
                    charge   = min(bal, fixed_cap - batt)
                    batt    += charge
                    leftover = bal - charge
                    cred    += leftover
                    credit_out, grid_draw = leftover, 0.0
                else:
                    # Deficit hour: draw from battery, then credit, then grid
                    need      = -bal
                    use_batt  = min(need, batt)
                    batt     -= use_batt
                    need     -= use_batt

                    use_cred  = min(need, cred)
                    cred     -= use_cred
                    need     -= use_cred

                    grid_draw, credit_out = need, 0.0

                    # If battery at or below threshold, top up from grid
                    if batt <= EMPTY_THR * fixed_cap:
                        topup = min(TOPUP_FRAC * fixed_cap, fixed_cap - batt)
                        batt += topup
                        grid_draw += topup

                # Clamp to [0, fixed_cap] to avoid negative or over-capacity due to precision
                if batt < 0:
                    batt = 0.0
                if batt > fixed_cap:
                    batt = fixed_cap

                d.loc[i, ["battery_state_kWh","credit_kWh",
                          "credit_out_kWh","grid_draw_kWh"]] = (
                    batt, cred, credit_out, grid_draw
                )

            # ── derived columns ─────────────────────────────────────
            d["battery_pct"]   = 100 * d["battery_state_kWh"] / fixed_cap
            d["grid_need_24h"] = d["raw_deficit"].shift(-1).rolling(HORIZON, 1).sum()
            d["exp_surplus_6h"] = (
                d["solar_kWh"] - d["load_kWh"]
            ).shift(-1).rolling(LOOK6, 1).sum()

            m, w = d["datetime"].dt.month, d["datetime"].dt.dayofweek
            d["month_sin"], d["month_cos"] = np.sin(2 * np.pi * m / 12), np.cos(2 * np.pi * m / 12)
            d["dow_sin"],   d["dow_cos"]   = np.sin(2 * np.pi * w / 7), np.cos(2 * np.pi * w / 7)
            d["load_mean_24h"]  = d["load_kWh"].rolling(24, 1).mean().shift(1)
            d["solar_mean_24h"] = d["solar_kWh"].rolling(24, 1).mean().shift(1)
            d["headroom_kWh"]   = (fixed_cap - d["battery_state_kWh"]).clip(lower=0)
            d["house"] = house

            recs.append(d)

    if not recs:
        say("❗ Nothing survived")
        return None

    # 4 ── concat all houses
    say("4/7  Concatenating …")
    df = pd.concat(recs, ignore_index=True)

    # 5 ── CatBoost prediction
    say("5/7  Loading CatBoost …")
    model = CatBoostRegressor()
    model.load_model("pretrained_models/catboost_keep_reserve_24h.cbm")

    feats = [
        "solar_kWh", "load_kWh", "Electricity_price_watt",
        "battery_state_kWh", "battery_pct", "credit_kWh",
        "exp_surplus_6h",
        "month_sin", "month_cos", "dow_sin", "dow_cos",
        "load_mean_24h", "solar_mean_24h"
    ]

    say("6/7  Predicting keep-reserve …")
    df["keep_pred_kWh"] = model.predict(df[feats])
    df["lendable_kWh"]  = (df["headroom_kWh"] - df["keep_pred_kWh"]).clip(lower=0)

    say("7/7  Done ✔")
    return df

# ───────────────────────── plotting ──────────────────────────────────
def plot_for_house_month(df: pd.DataFrame, house: str, month_str: str):
    """
    Create a PNG for `house` and `month_str` (format 'YYYY-MM') and save in /visualization.
    """
    sub = df[df["house"] == house]
    if sub.empty:
        messagebox.showinfo("No data", f"No rows for {house}")
        return

    period = pd.Period(month_str, freq="M")
    grp = sub[sub["datetime"].dt.to_period("M") == period].sort_values("datetime")
    if grp.empty:
        messagebox.showinfo("No data", f"No rows for {house} in {month_str}")
        return

    os.makedirs("visualization", exist_ok=True)
    fig, axes = plt.subplots(5, 1, sharex=True, figsize=(10, 12))
    ax1, ax2, ax3, ax4, ax5 = axes

    # Row 1: Raw 24h need, Predicted keep, Lendable band, Battery cap line
    ax1.plot(
        grp["datetime"], grp["grid_need_24h"], ":", color="grey",
        label="Raw 24 h need"
    )
    ax1.plot(
        grp["datetime"], grp["keep_pred_kWh"], color="green", lw=1.8,
        label="Predicted keep"
    )
    ax1.fill_between(
        grp["datetime"], 0, grp["lendable_kWh"],
        color="cyan", alpha=0.25, label="Lendable now"
    )
    cap_kWh = grp["Battery_capacity_kWh"].iloc[0]
    ax1.axhline(cap_kWh, color="red", ls="--", lw=1.2, label="Battery cap")
    ax1.set_ylabel("kWh")
    ax1.set_title(f"{house} — {month_str}")
    ax1.legend(loc="upper left")

    # Row 2: Load vs. Solar
    ax2.plot(grp["datetime"], grp["load_kWh"],  label="Load",  alpha=0.7)
    ax2.plot(grp["datetime"], grp["solar_kWh"], label="Solar", alpha=0.7)
    ax2.set_ylabel("kWh/h")
    ax2.legend(loc="upper left")

    # Row 3: Credit exported (purple bars) or "no credit exported"
    if grp["credit_out_kWh"].abs().sum() > 0:
        ax3.bar(
            grp["datetime"], grp["credit_out_kWh"], width=0.4,
            color="purple", alpha=0.6, label="Credit out"
        )
        ax3.set_ylabel("kWh")
        ax3.set_ylim(0, grp["credit_out_kWh"].max() * 1.1)
        ax3.legend(loc="upper left")
    else:
        ax3.text(
            0.5, 0.5, "no credit exported", ha="center", va="center",
            transform=ax3.transAxes, color="grey", fontsize=9
        )
        ax3.set_axis_off()

    # Row 4: Grid draw (red bars) or "no grid draw"
    if grp["grid_draw_kWh"].abs().sum() > 0:
        ax4.bar(
            grp["datetime"], grp["grid_draw_kWh"], width=0.4,
            color="red", alpha=0.6, label="Grid draw"
        )
        ax4.set_ylabel("kWh")
        ax4.set_ylim(0, grp["grid_draw_kWh"].max() * 1.1)
        ax4.legend(loc="upper left")
    else:
        ax4.text(
            0.5, 0.5, "no grid draw", ha="center", va="center",
            transform=ax4.transAxes, color="grey", fontsize=9
        )
        ax4.set_axis_off()

    # Row 5: Battery % (green line), auto-zoom if nearly flat
    ax5.plot(
        grp["datetime"], grp["battery_pct"],
        color="darkgreen", marker=".", ms=2, lw=1, label="Battery %"
    )
    low, high = grp["battery_pct"].min(), grp["battery_pct"].max()
    if high - low < 20:  # nearly flat → zoom in
        pad = 2
        ax5.set_ylim(max(0, low - pad), min(100, high + pad))
    else:
        ax5.set_ylim(0, 100)
    ax5.set_ylabel("% full")
    ax5.legend(loc="upper left")

    fig.autofmt_xdate()
    plt.tight_layout()
    fname = f"{house}_{month_str}.png"
    fig.savefig(os.path.join("visualization", fname))
    plt.close(fig)
    print("✅ saved", fname)

# ─────────────────────────── GUI ─────────────────────────────────────
def run_gui():
    root = tk.Tk()
    root.title("24-h Prosumer Reserve Visualiser")

    # Queue and event for background processing (must be defined before cancel_btn)
    q, stop_evt = queue.Queue(), threading.Event()

    # Prosumer selector
    ttk.Label(root, text="Select a prosumer:").grid(row=0, column=0, padx=10, pady=10)
    house_var = tk.StringVar(value="(loading…)")
    combo_house = ttk.Combobox(root, textvariable=house_var, width=36, state="disabled")
    combo_house.grid(row=0, column=1, padx=10, pady=10)

    # Month selector
    ttk.Label(root, text="Select month:").grid(row=1, column=0, padx=10, pady=10)
    month_var = tk.StringVar(value="(select a prosumer first)")
    combo_month = ttk.Combobox(root, textvariable=month_var, width=36, state="disabled")
    combo_month.grid(row=1, column=1, padx=10, pady=10)

    # Plot button
    btn_plot = ttk.Button(root, text="Plot", state="disabled")
    btn_plot.grid(row=2, column=0, columnspan=2, pady=6)

    # Status label
    status = ttk.Label(root, text="Loading …", foreground="steelblue")
    status.grid(row=3, column=0, columnspan=2)

    # Cancel button
    cancel_btn = ttk.Button(root, text="Cancel")
    cancel_btn.grid(row=4, column=0, columnspan=2, pady=(0, 8))
    cancel_btn.config(command=stop_evt.set)

    # Poll status queue
    def poll():
        try:
            while True:
                msg = q.get_nowait()
                status.config(
                    text=msg,
                    foreground=(
                        "green" if msg.endswith("✔")
                        else "red" if msg.startswith(("❗", "⚠", "❌"))
                        else "steelblue"
                    )
                )
        except queue.Empty:
            pass
        root.after(150, poll)

    poll()

    # Background worker
    def worker():
        df = build_test_df(q.put, stop_evt.is_set, fraction=0.01)
        if df is None:
            q.put("❌ No data built.")
            return

        houses = sorted(df["house"].unique())

        # Enable prosumer selector once data is ready
        def enable_house_selection():
            combo_house.config(values=houses, state="readonly")
            house_var.set(houses[0])
            populate_months(houses[0])

        root.after(0, enable_house_selection)

        # Populate month Combobox based on selected house
        def populate_months(selected_house):
            sub = df[df["house"] == selected_house]
            periods = sorted(
                sub["datetime"].dt.to_period("M")
                .unique()
                .astype(str)
            )
            combo_month.config(values=periods, state="readonly")
            if periods:
                month_var.set(periods[0])
                btn_plot.config(state="normal")
            else:
                month_var.set("(no data)")
                btn_plot.config(state="disabled")

        # When a prosumer is selected, update the months
        def on_house_selected(event):
            selected_house = house_var.get()
            populate_months(selected_house)

        combo_house.bind("<<ComboboxSelected>>", on_house_selected)

        # Configure Plot button to use both house and month
        btn_plot.config(
            command=lambda: plot_and_save(df, house_var.get(), month_var.get())
        )

    threading.Thread(target=worker, daemon=True).start()
    root.mainloop()

# ─────────────────── helper: save CSV + plot ─────────────────────────
def plot_and_save(df: pd.DataFrame, house: str, month_str: str):
    os.makedirs("visualization", exist_ok=True)
    csv_path = "visualization/full_simulation_dump.csv"
    df.to_csv(csv_path, index=False)
    print("💾 DataFrame written →", csv_path)
    plot_for_house_month(df, house, month_str)

# ───────────────────────── main ──────────────────────────────────────
if __name__ == "__main__":
    print("🌟 GUI starting — heavy work in background")
    run_gui()
