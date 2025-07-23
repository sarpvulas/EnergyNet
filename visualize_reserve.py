#!/usr/bin/env python3
"""
simple_reserve_visualiser.py
──────────────────────────────────────────────────────────────────────
Visualise **every** house’s energy flows & profit — *no ML, no role
classification*.

Key points
==========
* Loads hourly CSV data for **all** houses (sampled via `fraction`).
* Simulates battery SOC and grid transactions with a **5 %-threshold/20 %-top-up** rule.
* Tkinter GUI lets you pick a house + month and saves a **three-panel PNG**.
* **No CSV dump** is created.
* Grid-usage and sold-energy bars are deliberately **thin (width = 0.1)**.

Assumptions
-----------
* Price columns are per **watt-hour**; dividing by 1000 converts to per-kWh.
* Missing/invalid battery capacity → **10 kWh** default.
* Default `fraction=0.01` = quick demo; set to `1.0` for full data.
"""

import os
import threading
import queue
import tkinter as tk
from tkinter import ttk, messagebox
from typing import Optional, Callable, List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from data_loader import read_all_generated_data  # project helper — unchanged

# ───────────────────────── helper ────────────────────────────────────

def ensure(df: pd.DataFrame, col: str, default: float = 0.0) -> None:
    """Ensure *col* exists in *df*; if missing, create and fill with *default*."""
    if col not in df:
        df[col] = default


# ────────────────────── heavy builder ────────────────────────────────

def build_df(progress: Callable[[str], None], cancelled: Callable[[], bool], *,
             fraction: float = 0.1) -> Optional[pd.DataFrame]:
    """Load → simulate → concatenate across **all** houses."""

    say = progress

    # 1 ── load data
    say("1/4  Loading CSVs …")
    all_data, _ = read_all_generated_data(False, fraction)
    if cancelled():
        return None

    # Simulation constants
    FALLBACK_CAP = 10.0   # kWh fallback capacity
    INIT_SOC     = 0.5    # 50 % starting SOC
    TOPUP_FRAC   = 0.2    # 20 % top-up amount
    EMPTY_THR    = 0.05   # ≤ 5 % triggers top-up

    total_houses = sum(len(folder) for folder in all_data.values())
    recs: List[pd.DataFrame] = []
    done = 0

    # 2 ── simulate every house
    for folder in all_data.values():
        for house, df in folder.items():
            if cancelled():
                continue

            done += 1
            say(f"2/4  Simulating {house}  ({done}/{total_houses})")

            d = df.copy()
            d["datetime"] = pd.to_datetime(d["datetime"], utc=True)
            d.sort_values("datetime", inplace=True, ignore_index=True)

            # Ensure required columns exist
            for col in ("Electricity_price_watt", "Excess_energy_watt", "to_grid_prices"):
                ensure(d, col, 0.0)

            d["solar_kWh"] = d["solar_hourly_watt_generation_house"].astype(float) / 1000.0
            d["load_kWh"]  = d["hourly_load_kw_house"].astype(float)
            if len(d) < 3:
                continue

            # Fixed battery capacity per house
            raw_cap  = d["Battery_capacity_kw"].astype(float)
            pos_vals = raw_cap[raw_cap > 0]
            fixed_cap = pos_vals.iloc[0] if not pos_vals.empty else FALLBACK_CAP
            d["Battery_capacity_kWh"] = fixed_cap
            d["Battery_capacity_raw_kW"] = raw_cap

            # Initial state
            batt = fixed_cap * INIT_SOC  # kWh
            cred = 0.0                   # kWh

            d[["battery_state_kWh", "credit_kWh", "credit_out_kWh", "grid_draw_kWh"]] = np.nan

            # Hour-by-hour loop
            for i in range(len(d)):
                load  = d.at[i, "load_kWh"]
                solar = d.at[i, "solar_kWh"]
                bal   = solar - load  # surplus if positive

                if bal > 0:
                    charge   = min(bal, fixed_cap - batt)
                    batt    += charge
                    leftover = bal - charge
                    cred    += leftover
                    credit_out, grid_draw = leftover, 0.0
                else:
                    need      = -bal
                    use_batt  = min(need, batt)
                    batt     -= use_batt
                    need     -= use_batt

                    use_cred  = min(need, cred)
                    cred     -= use_cred
                    need     -= use_cred

                    grid_draw, credit_out = need, 0.0

                    if batt <= EMPTY_THR * fixed_cap:
                        topup = min(TOPUP_FRAC * fixed_cap, fixed_cap - batt)
                        batt += topup
                        grid_draw += topup

                batt = max(0.0, min(batt, fixed_cap))

                d.loc[i, ["battery_state_kWh", "credit_kWh", "credit_out_kWh", "grid_draw_kWh"]] = (
                    batt, cred, credit_out, grid_draw
                )

            # Derived columns
            d["battery_pct"]        = 100 * d["battery_state_kWh"] / fixed_cap
            d["grid_cost"]         = d["grid_draw_kWh"]   * d["Electricity_price_watt"] / 1000.0
            d["grid_revenue"]      = d["credit_out_kWh"] * d["to_grid_prices"]         / 1000.0
            d["profit"]            = d["grid_revenue"] - d["grid_cost"]
            d["cumulative_profit"] = d["profit"].cumsum()
            d["house"]             = house

            recs.append(d)

    if not recs:
        say("❗ Nothing survived")
        return None

    # 3 ── concatenate all
    say("3/4  Concatenating …")
    df_all = pd.concat(recs, ignore_index=True)

    say("4/4  Done ✔")
    return df_all


# ───────────────────────── plotting ──────────────────────────────────

def plot_for_house_month(df: pd.DataFrame, house: str, month_str: str) -> None:
    """Generate and save a 3-row PNG for *house* & *month*."""

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

    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, sharex=True, figsize=(10, 10))

    # Panel 1: Solar vs Load
    ax1.plot(grp["datetime"], grp["solar_kWh"], label="Solar", alpha=0.8)
    ax1.plot(grp["datetime"], grp["load_kWh"],  label="Load",  alpha=0.8)
    ax1.set_ylabel("kWh/h")
    ax1.set_title(f"{house} — {month_str}")
    ax1.legend(loc="upper left")

    # Panel 2: Grid usage vs Sold (thin bars)
    bar_w = 0.1
    ax2.bar(grp["datetime"], grp["grid_draw_kWh"],   width=bar_w,
            color="red",   alpha=0.6, label="Grid usage (buy)")
    ax2.bar(grp["datetime"], grp["credit_out_kWh"], width=bar_w,
            color="green", alpha=0.6, label="Sold to grid")
    ax2.set_ylabel("kWh")
    ax2.legend(loc="upper left")

    # Panel 3: Profit / Loss
    colors = ["green" if p >= 0 else "red" for p in grp["profit"]]
    ax3.bar(grp["datetime"], grp["profit"], width=0.1, color=colors, alpha=0.7,
            label="Hourly P/L")
    ax3.plot(grp["datetime"], grp["cumulative_profit"], color="grey", lw=1.2,
             label="Cumulative P/L")
    ax3.set_ylabel("Currency")
    ax3.legend(loc="upper left")

    fig.autofmt_xdate()
    plt.tight_layout()
    fname = f"{house}_{month_str}_energystats.png"
    fig.savefig(os.path.join("visualization", fname))
    plt.close(fig)
    print("✅ saved", fname)


# ─────────────────────────── GUI ─────────────────────────────────────

def run_gui() -> None:
    root = tk.Tk()
    root.title("House Energy Visualiser (no ML)")

    q, stop_evt = queue.Queue(), threading.Event()

    # ——— UI widgets ———
    ttk.Label(root, text="Select a house:").grid(row=0, column=0, padx=10, pady=10)
    house_var = tk.StringVar(value="(loading…)")
    combo_house = ttk.Combobox(root, textvariable=house_var, width=36, state="disabled")
    combo_house.grid(row=0, column=1, padx=10, pady=10)

    ttk.Label(root, text="Select month:").grid(row=1, column=0, padx=10, pady=10)
    month_var = tk.StringVar(value="(select house first)")
    combo_month = ttk.Combobox(root, textvariable=month_var, width=36, state="disabled")
    combo_month.grid(row=1, column=1, padx=10, pady=10)

    btn_plot = ttk.Button(root, text="Plot", state="disabled")
    btn_plot.grid(row=2, column=0, columnspan=2, pady=6)

    status = ttk.Label(root, text="Loading …", foreground="steelblue")
    status.grid(row=3, column=0, columnspan=2)

    cancel_btn = ttk.Button(root, text="Cancel", command=stop_evt.set)
    cancel_btn.grid(row=4, column=0, columnspan=2, pady=(0, 8))

    # ——— helper to poll status queue ———
    def poll_status():
        try:
            while True:
                msg = q.get_nowait()
                colour = (
                    "green" if msg.endswith("✔") else
                    "red" if msg.startswith(("❗", "⚠", "❌")) else
                    "steelblue"
                )
                status.config(text=msg, foreground=colour)
        except queue.Empty:
            pass
        root.after(150, poll_status)

    poll_status()

    # ——— background worker thread ———
    def worker():
        df = build_df(q.put, stop_evt.is_set, fraction=0.01)
        if df is None:
            q.put("❌ No data built.")
            return

        houses = sorted(df["house"].unique())

        def populate_months(selected_house: str):
            sub = df[df["house"] == selected_house]
            periods = sorted(sub["datetime"].dt.to_period("M").unique().astype(str))
            combo_month.config(values=periods, state="readonly")
            if periods:
                month_var.set(periods[0])
                btn_plot.config(state="normal")
            else:
                month_var.set("(no data)")
                btn_plot.config(state="disabled")

        def on_house_selected(_: tk.Event):
            populate_months(house_var.get())

        # enable selectors in GUI thread
        def enable_selectors():
            combo_house.config(values=houses, state="readonly")
            house_var.set(houses[0])
            populate_months(houses[0])
            combo_house.bind("<<ComboboxSelected>>", on_house_selected)
            btn_plot.config(command=lambda: plot_for_house_month(df, house_var.get(), month_var.get()))

        root.after(0, enable_selectors)

    threading.Thread(target=worker, daemon=True).start()

    root.mainloop()


# ───────────────────────── main ─────────────────────────────────────

if __name__ == "__main__":
    print("🌟 GUI starting — no ML, full-house visualisation")
    run_gui()
