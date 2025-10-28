# gui/controller.py

from __future__ import annotations

from pathlib import Path
from typing import Dict, List
from concurrent.futures import ThreadPoolExecutor

import tkinter as tk
from tkinter import ttk, messagebox

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

from ..io.filesystem import (
    read_soc_timeseries,
    read_hourly_trades,
    discover_generated_folder,
    load_house_csv_frames,
    read_barters_csv,
    read_claim_returns_csv,
    read_claim_expiries_csv,
)
from ..transform.builders import (
    build_load_and_solar_wide,
    derive_prices_charges_from_soc,
)
from ..stats.aggregates import compute_stats_dataframe, compute_aggregates
from ..plotting.panels import (
    draw_battery_panels,
    draw_middle_bars,
    draw_grid_panel,
    draw_trade_network,
    draw_stats_panel,
)

FONT_LBL = 5
FONT_TITLE = 7


def launch_viewer(folder: int) -> None:
    results_dir = Path("results") / f"folder_{folder}"
    if not results_dir.exists():
        raise FileNotFoundError(f"{results_dir} not found")

    # I/O
    frames = load_house_csv_frames(folder)
    # Load both SoC variants if available
    soc_trades_df, _ = read_soc_timeseries(results_dir, prefer_barters=False)
    soc_barters_df, _ = read_soc_timeseries(results_dir, prefer_barters=True)
    # Use trades SoC as default; barters SoC may be identical if not present
    soc_df = soc_trades_df
    soc_df_map = {"Trades": soc_trades_df, "Barters": soc_barters_df}
    # Trades (MILP) – tolerate absence for bartering-only runs
    try:
        trades = read_hourly_trades(results_dir)
    except Exception:
        trades = []

    # Barters (sim) and claim returns (physical redemptions)
    barters_df = read_barters_csv(results_dir)
    claim_returns_df = read_claim_returns_csv(results_dir)
    claim_expiries_df = read_claim_expiries_csv(results_dir)

    # Transforms
    load_w, solar_w = build_load_and_solar_wide(frames)
    prices_list, charges_list, discharges_list, capacity_map, hour_dt = derive_prices_charges_from_soc(
        frames, soc_df
    )

    houses = soc_df.columns.tolist()
    hours = soc_df.index.values

    # Build hourly barters: split into used (green arrows), stored markers (badges), and returns (purple arrows)
    T = len(hours)
    barters_used_by_hour: List[List[Dict]] = [[] for _ in range(T)]
    barters_stored_markers_by_hour: List[List[Dict]] = [[] for _ in range(T)]
    returns_by_hour: List[List[Dict]] = [[] for _ in range(T)]
    expiry_self_by_hour: List[List[Dict]] = [[] for _ in range(T)]
    if not barters_df.empty:
        for _, row in barters_df.iterrows():
            t = int(row["t"]) if "t" in row else 0
            if 0 <= t < T:
                seller = int(row.get("seller", 0))
                buyer = int(row.get("buyer", 0))
                eta = float(row.get("eta", 0.0))
                Es = float(row.get("Es_kWh", 0.0))
                claim_kWh = float(row.get("claim_kWh", 0.0))
                used_amt = eta * Es
                if used_amt > 1e-9:
                    barters_used_by_hour[t].append({
                        "type": "barter-used",
                        "seller": seller,
                        "buyer": buyer,
                        "amount": float(used_amt),
                        "eta": eta,
                        "claim_kWh": claim_kWh,
                    })
                if claim_kWh > 1e-9:
                    barters_stored_markers_by_hour[t].append({
                        "type": "barter-stored",
                        "seller": seller,
                        "buyer": buyer,
                        "amount": float(claim_kWh),
                    })
    if not claim_returns_df.empty:
        for _, row in claim_returns_df.iterrows():
            t = int(row["t"]) if "t" in row else 0
            if 0 <= t < T:
                returns_by_hour[t].append({
                    "type": "barter-return",
                    "seller": int(row.get("stored_on", 0)),  # physical source battery
                    "buyer": int(row.get("buyer", 0)),       # consuming peer
                    "owner": int(row.get("owner", 0)),
                    "amount": float(row.get("qty_kWh", 0.0)),
                })
    if not claim_expiries_df.empty:
        for _, row in claim_expiries_df.iterrows():
            t = int(row.get("t", 0))
            if 0 <= t < T:
                store = int(row.get("stored_on", 0))
                qty = float(row.get("qty_kWh", 0.0))
                if qty > 1e-9 and store in houses:
                    expiry_self_by_hour[t].append({
                        "type": "expiry-used",
                        "seller": store,
                        "buyer": store,
                        "amount": qty,
                    })
    if not trades:
        trades = [[] for _ in range(T)]

    # Seller→color mapping for badges/legend (stable mapping across houses)
    def _to_hex(rgb_tuple):
        r, g, b, *a = rgb_tuple
        return f"#{int(r*255):02x}{int(g*255):02x}{int(b*255):02x}"
    cmap = plt.cm.get_cmap("tab20", max(10, len(houses)))
    seller_colors: Dict[int, str] = {hid: _to_hex(cmap(i % cmap.N)) for i, hid in enumerate(houses)}

    # Stats inputs (clients, charges/discharges are invariant across view modes for this results set)
    clients_data_list = [frames[hid] for hid in houses if hid in frames]

    # Precompute aggregates for both modes to keep UI interactions snappy
    # Parallel precomputation for trades/barters aggregates
    def _compute_pair(edge_list: List[List[Dict]]):
        df = compute_stats_dataframe(
            clients_data_list=clients_data_list,
            trades=edge_list,
            prices=prices_list,
            charges_list=charges_list,
            discharges_list=discharges_list,
        )
        ag = compute_aggregates(
            stats_df=df,
            trades=edge_list,
            load_w=load_w,
            solar_w=solar_w,
            soc_df=soc_df,
            capacity_map=capacity_map,
            houses=houses,
        )
        return df, ag

    with ThreadPoolExecutor(max_workers=2) as ex:
        fut_trades = ex.submit(_compute_pair, trades)
        # Use only 'used' barter flows for aggregate stats
        fut_barters = ex.submit(_compute_pair, barters_used_by_hour)
        stats_df_trades, agg_trades = fut_trades.result()
        stats_df_barters, agg_barters = fut_barters.result()

    # GUI scaffolding
    root = tk.Tk()
    root.title(f"EnergyNet Viewer – folder {folder}")

    # dedicated centered title label above the figure
    title_frm = ttk.Frame(root)
    title_frm.pack(fill=tk.X, padx=6, pady=(8, 2))
    title_lbl = ttk.Label(title_frm, font=("Arial", 12, "bold"), anchor="center")
    title_lbl.pack(fill=tk.X)

    ctrl = ttk.Frame(root)
    ctrl.pack(fill=tk.X, padx=6, pady=4)
    h_var = tk.IntVar(value=0)
    ttk.Button(ctrl, text="« Prev", command=lambda: h_var.set(max(0, h_var.get() - 1))).pack(side=tk.LEFT)
    lbl = ttk.Label(ctrl, font=("Arial", 9, "bold"))
    lbl.pack(side=tk.LEFT, padx=10)
    ttk.Button(ctrl, text="Next »", command=lambda: h_var.set(min(len(hours) - 1, h_var.get() + 1))).pack(
        side=tk.LEFT
    )
    ttk.Scale(ctrl, from_=0, to=len(hours) - 1, orient="horizontal", variable=h_var).pack(
        side=tk.LEFT, fill=tk.X, expand=True, padx=10
    )

    # View mode selector: Trades vs Barters
    mode_var = tk.StringVar(value="Trades")
    ttk.Label(ctrl, text="View:").pack(side=tk.LEFT, padx=(8, 2))
    mode_sel = ttk.Combobox(ctrl, width=10, textvariable=mode_var, values=["Trades", "Barters"], state="readonly")
    mode_sel.pack(side=tk.LEFT, padx=(0, 8))

    # Figure layout
    fig = plt.figure(figsize=(12, 6.6), constrained_layout=True)
    fig.set_dpi(150)  # crisper, smaller-looking text

    # 3 rows x 3 cols grid; let constrained_layout place things safely
    gs = gridspec.GridSpec(
        3,
        3,
        figure=fig,
        height_ratios=[2, 1, 1],
    )

    ax_full = fig.add_subplot(gs[0, 0])     # Battery fullness
    ax_cap = fig.add_subplot(gs[1, 0])      # Battery capacity
    ax_stats = fig.add_subplot(gs[2, 0])    # Statistics cards (our new robust panel)

    ax_load = fig.add_subplot(gs[0:2, 1])   # Solar/Load (left)
    ax_grid = fig.add_subplot(gs[2, 1])     # Grid draw/sell
    ax_trade = fig.add_subplot(gs[:, 2])    # Trade network
    ax_load_r = ax_load.twinx()             # Load (right axis)

    canvas = FigureCanvasTkAgg(fig, master=root)
    canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    def open_barters_view():
        top = tk.Toplevel(root)
        top.title("Barters Inspector")
        top.geometry("1000x600")
        for i in range(3):
            top.columnconfigure(i, weight=1)
        top.rowconfigure(2, weight=1)

        # Filters
        filt = ttk.LabelFrame(top, text="Filters")
        filt.grid(row=0, column=0, columnspan=3, sticky="ew", padx=8, pady=(8, 4))
        for i in range(10):
            filt.columnconfigure(i, weight=1)

        tk.Label(filt, text="Hour from:").grid(row=0, column=0, sticky="e", padx=(6, 2), pady=4)
        e_from = ttk.Entry(filt, width=8); e_from.grid(row=0, column=1, sticky="w", padx=(0, 8))
        tk.Label(filt, text="to:").grid(row=0, column=2, sticky="e", padx=(6, 2))
        e_to = ttk.Entry(filt, width=8); e_to.grid(row=0, column=3, sticky="w", padx=(0, 8))

        tk.Label(filt, text="Seller:").grid(row=0, column=4, sticky="e", padx=(6, 2))
        e_seller = ttk.Entry(filt, width=10); e_seller.grid(row=0, column=5, sticky="w", padx=(0, 8))
        tk.Label(filt, text="Buyer:").grid(row=0, column=6, sticky="e", padx=(6, 2))
        e_buyer = ttk.Entry(filt, width=10); e_buyer.grid(row=0, column=7, sticky="w", padx=(0, 8))

        tk.Label(filt, text="min Es_kWh:").grid(row=0, column=8, sticky="e", padx=(6, 2))
        e_min_es = ttk.Entry(filt, width=8); e_min_es.grid(row=0, column=9, sticky="w", padx=(0, 8))

        # Controls row
        ctrl_row = ttk.Frame(top)
        ctrl_row.grid(row=1, column=0, columnspan=3, sticky="ew", padx=8, pady=(0, 6))
        for i in range(6):
            ctrl_row.columnconfigure(i, weight=1)

        btn_apply = ttk.Button(ctrl_row, text="Apply filters")
        btn_export = ttk.Button(ctrl_row, text="Export filtered CSV")
        btn_open_returns = ttk.Button(ctrl_row, text="Open returns table")
        include_returns_var = tk.BooleanVar(value=True)
        cb_include = ttk.Checkbutton(ctrl_row, text="Include returns", variable=include_returns_var)
        include_expiries_var = tk.BooleanVar(value=True)
        cb_include_exp = ttk.Checkbutton(ctrl_row, text="Include expiries", variable=include_expiries_var)
        ttk.Label(ctrl_row, text="Page size:").grid(row=0, column=2, sticky="e")
        e_ps = ttk.Combobox(ctrl_row, width=6, values=[100, 500, 1000, 2000], state="readonly")
        e_ps.set(1000)
        e_ps.grid(row=0, column=3, sticky="w")
        btn_prev = ttk.Button(ctrl_row, text="◀ Prev")
        btn_next = ttk.Button(ctrl_row, text="Next ▶")
        lbl_page = ttk.Label(ctrl_row, text="Page 1/1")

        btn_apply.grid(row=0, column=0, sticky="w")
        btn_export.grid(row=0, column=1, sticky="w", padx=(6, 0))
        cb_include.grid(row=0, column=2, sticky="w", padx=(90, 0))
        cb_include_exp.grid(row=0, column=2, sticky="w", padx=(200, 0))
        btn_open_returns.grid(row=0, column=2, sticky="w", padx=(320, 0))
        btn_prev.grid(row=0, column=4, sticky="e")
        lbl_page.grid(row=0, column=5, sticky="e", padx=(8, 8))
        btn_next.grid(row=0, column=6, sticky="e")

        # Table
        table_frm = ttk.Frame(top)
        table_frm.grid(row=2, column=0, columnspan=3, sticky="nsew", padx=8, pady=(0, 8))
        table_frm.rowconfigure(0, weight=1)
        table_frm.columnconfigure(0, weight=1)
        cols = ("mark", "type", "t", "seller", "buyer", "Es_kWh", "eta", "used_kWh", "stored_kWh", "reclaimed_kWh", "expiry_t")
        tree = ttk.Treeview(table_frm, columns=cols, show="headings", height=20)
        vsb = ttk.Scrollbar(table_frm, orient="vertical", command=tree.yview)
        hsb = ttk.Scrollbar(table_frm, orient="horizontal", command=tree.xview)
        tree.configure(yscroll=vsb.set, xscroll=hsb.set)
        tree.grid(row=0, column=0, sticky="nsew")
        vsb.grid(row=0, column=1, sticky="ns")
        hsb.grid(row=1, column=0, sticky="ew")

        for c in cols:
            tree.heading(c, text=c)
            tree.column(c, width=100, anchor="center")

        # State for pagination and sorting
        _df_barters = barters_df.copy() if not barters_df.empty else pd.DataFrame(columns=["t","seller","buyer","Es_kWh","eta","claim_kWh","expiry_t"]) 
        if not _df_barters.empty:
            try:
                _df_barters["used_kWh"] = _df_barters["eta"].astype(float) * _df_barters["Es_kWh"].astype(float)
            except Exception:
                _df_barters["used_kWh"] = 0.0
            _df_barters["stored_kWh"] = _df_barters.get("claim_kWh", pd.Series(0.0)).astype(float)
            _df_barters["reclaimed_kWh"] = 0.0
            _df_barters["mark"] = "🟢"  # lend rows
            _df_barters["type"] = "barter"

        # Map returns to unified schema
        _df_returns_raw = claim_returns_df.copy() if not claim_returns_df.empty else pd.DataFrame(columns=["t","owner","stored_on","buyer","qty_kWh"]) 
        if not _df_returns_raw.empty:
            _df_ret = pd.DataFrame()
            _df_ret["t"] = _df_returns_raw["t"]
            _df_ret["seller"] = _df_returns_raw["stored_on"]
            _df_ret["buyer"] = _df_returns_raw["buyer"]
            _df_ret["Es_kWh"] = 0.0
            _df_ret["eta"] = 0.0
            _df_ret["used_kWh"] = 0.0
            _df_ret["stored_kWh"] = 0.0
            _df_ret["reclaimed_kWh"] = _df_returns_raw["qty_kWh"].astype(float)
            _df_ret["expiry_t"] = 0
            _df_ret["mark"] = "🟣"
            _df_ret["type"] = "return"
        else:
            _df_ret = pd.DataFrame(columns=["t","seller","buyer","Es_kWh","eta","used_kWh","stored_kWh","reclaimed_kWh","expiry_t","mark","type"]) 

        # Expiries to unified schema (no energy movement; ownership lapsed)
        _df_exp_raw = claim_expiries_df.copy() if not claim_expiries_df.empty else pd.DataFrame(columns=["t","owner","stored_on","qty_kWh"]) 
        if not _df_exp_raw.empty:
            _df_exp = pd.DataFrame()
            _df_exp["t"] = _df_exp_raw["t"]
            # show direction stored_on -> owner to mirror a would-be return
            _df_exp["seller"] = _df_exp_raw["stored_on"]
            _df_exp["buyer"] = _df_exp_raw["owner"]
            _df_exp["Es_kWh"] = 0.0
            _df_exp["eta"] = 0.0
            _df_exp["used_kWh"] = 0.0
            _df_exp["stored_kWh"] = 0.0
            _df_exp["reclaimed_kWh"] = 0.0
            _df_exp["expiry_t"] = 0
            _df_exp["mark"] = "🟠"
            _df_exp["type"] = "expiry"
        else:
            _df_exp = pd.DataFrame(columns=["t","seller","buyer","Es_kWh","eta","used_kWh","stored_kWh","reclaimed_kWh","expiry_t","mark","type"]) 

        # Base combined (avoid pandas FutureWarning by skipping empty frames)
        parts = []
        for _df in (_df_barters, _df_ret, _df_exp):
            if _df is not None and not _df.empty:
                parts.append(_df)
        if parts:
            _df_init = pd.concat(parts, ignore_index=True, sort=False)
        else:
            _df_init = pd.DataFrame(columns=list(cols))

        # Ensure all expected columns exist with correct basic dtypes
        for c in cols:
            if c not in _df_init.columns:
                if c in ("mark", "type"):
                    _df_init[c] = ""
                elif c in ("t", "seller", "buyer", "expiry_t"):
                    _df_init[c] = 0
                else:
                    _df_init[c] = 0.0

        # Reorder columns
        _df_init = _df_init[list(cols)].copy()

        state_tbl = {
            "df": _df_init,
            "filtered": None,
            "page": 0,
            "pages": 1,
            "page_size": int(e_ps.get()),
            "sort_col": "t",
            "sort_asc": True,
        }

        def _filter_df():
            df = state_tbl["df"]
            if df is None or df.empty:
                return df
            q = df
            # include/exclude returns and expiries
            if not include_returns_var.get():
                q = q[q.get("type", "barter") != "return"]
            if not include_expiries_var.get():
                q = q[q.get("type", "barter") != "expiry"]
            try:
                t0 = int(e_from.get()) if e_from.get() else None
            except Exception:
                t0 = None
            try:
                t1 = int(e_to.get()) if e_to.get() else None
            except Exception:
                t1 = None
            if t0 is not None:
                q = q[q["t"] >= t0]
            if t1 is not None:
                q = q[q["t"] <= t1]
            if e_seller.get().strip():
                try:
                    s = int(e_seller.get().strip())
                    q = q[q["seller"] == s]
                except Exception:
                    pass
            if e_buyer.get().strip():
                try:
                    b = int(e_buyer.get().strip())
                    q = q[q["buyer"] == b]
                except Exception:
                    pass
            if e_min_es.get().strip():
                try:
                    mn = float(e_min_es.get().strip())
                    q = q[q["Es_kWh"].astype(float) >= mn]
                except Exception:
                    pass
            return q

        def _apply_filters_and_refresh():
            q = _filter_df()
            if q is None:
                q = pd.DataFrame(columns=list(cols))
            # sort
            sort_col = state_tbl["sort_col"]
            asc = state_tbl["sort_asc"]
            if sort_col in q.columns:
                try:
                    q = q.sort_values(sort_col, ascending=asc)
                except Exception:
                    pass
            state_tbl["filtered"] = q
            ps = state_tbl["page_size"]
            state_tbl["pages"] = max(1, int((len(q) + ps - 1) // ps))
            state_tbl["page"] = 0
            _render_page()

        def _render_page():
            q = state_tbl["filtered"]
            if q is None:
                q = pd.DataFrame(columns=list(cols))
            ps = state_tbl["page_size"]
            p = max(0, min(state_tbl["page"], state_tbl["pages"] - 1))
            state_tbl["page"] = p
            i0 = p * ps; i1 = i0 + ps
            view = q.iloc[i0:i1] if not q.empty else q
            # clear
            for iid in tree.get_children():
                tree.delete(iid)
            # insert
            for _, row in view.iterrows():
                vals = [row.get(c, "") for c in cols]
                tree.insert("", "end", values=vals)
            lbl_page.configure(text=f"Page {p+1}/{state_tbl['pages']}")

        def _export_filtered():
            q = state_tbl["filtered"]
            if q is None or q.empty:
                messagebox.showinfo("Export", "No filtered rows to export.")
                return
            try:
                out = Path("barters_filtered.csv").resolve()
                q.to_csv(out, index=False)
                messagebox.showinfo("Export", f"Saved to {out}")
            except Exception as ex:
                messagebox.showerror("Export", str(ex))

        def _on_heading_click(col):
            prev = state_tbl["sort_col"]
            if prev == col:
                state_tbl["sort_asc"] = not state_tbl["sort_asc"]
            else:
                state_tbl["sort_col"] = col
                state_tbl["sort_asc"] = True
            _apply_filters_and_refresh()

        for c in cols:
            tree.heading(c, command=lambda cc=c: _on_heading_click(cc))

        def _set_ps(_ev=None):
            try:
                state_tbl["page_size"] = int(e_ps.get())
            except Exception:
                state_tbl["page_size"] = 1000
            _apply_filters_and_refresh()

        def _prev():
            state_tbl["page"] = max(0, state_tbl["page"] - 1)
            _render_page()

        def _next():
            state_tbl["page"] = min(state_tbl["pages"] - 1, state_tbl["page"] + 1)
            _render_page()

        def open_returns_view():
            top2 = tk.Toplevel(top)
            top2.title("Claim Returns")
            top2.geometry("900x520")
            frm_f = ttk.LabelFrame(top2, text="Filters")
            frm_f.pack(fill=tk.X, padx=8, pady=8)
            tk.Label(frm_f, text="Hour from:").pack(side=tk.LEFT, padx=(6,2))
            er_from = ttk.Entry(frm_f, width=8); er_from.pack(side=tk.LEFT)
            tk.Label(frm_f, text="to:").pack(side=tk.LEFT, padx=(6,2))
            er_to = ttk.Entry(frm_f, width=8); er_to.pack(side=tk.LEFT)
            tk.Label(frm_f, text="Owner:").pack(side=tk.LEFT, padx=(10,2))
            er_owner = ttk.Entry(frm_f, width=8); er_owner.pack(side=tk.LEFT)
            tk.Label(frm_f, text="Stored_on:").pack(side=tk.LEFT, padx=(10,2))
            er_store = ttk.Entry(frm_f, width=8); er_store.pack(side=tk.LEFT)
            tk.Label(frm_f, text="Buyer:").pack(side=tk.LEFT, padx=(10,2))
            er_buyer = ttk.Entry(frm_f, width=8); er_buyer.pack(side=tk.LEFT)

            frm_tbl = ttk.Frame(top2); frm_tbl.pack(fill=tk.BOTH, expand=True, padx=8, pady=(0,8))
            cols_r = ("mark","t","owner","stored_on","buyer","qty_kWh")
            tree_r = ttk.Treeview(frm_tbl, columns=cols_r, show="headings")
            vsb_r = ttk.Scrollbar(frm_tbl, orient="vertical", command=tree_r.yview)
            tree_r.configure(yscroll=vsb_r.set)
            tree_r.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
            vsb_r.pack(side=tk.RIGHT, fill=tk.Y)
            for c in cols_r:
                tree_r.heading(c, text=c)

            def _refresh_ret():
                df = claim_returns_df.copy() if not claim_returns_df.empty else pd.DataFrame(columns=list(cols_r))
                if not df.empty:
                    try:
                        if er_from.get():
                            df = df[df["t"] >= int(er_from.get())]
                        if er_to.get():
                            df = df[df["t"] <= int(er_to.get())]
                        if er_owner.get():
                            df = df[df["owner"] == int(er_owner.get())]
                        if er_store.get():
                            df = df[df["stored_on"] == int(er_store.get())]
                        if er_buyer.get():
                            df = df[df["buyer"] == int(er_buyer.get())]
                    except Exception:
                        pass
                for iid in tree_r.get_children():
                    tree_r.delete(iid)
                for _, r in df.iterrows():
                    vals = ["🟣"] + [r.get(c, "") for c in cols_r if c != "mark"]
                    tree_r.insert("", "end", values=vals)

            for wdg in (er_from, er_to, er_owner, er_store, er_buyer):
                wdg.bind("<Return>", lambda *_: _refresh_ret())
                wdg.bind("<FocusOut>", lambda *_: _refresh_ret())
            _refresh_ret()

        btn_apply.configure(command=_apply_filters_and_refresh)
        btn_export.configure(command=_export_filtered)
        e_ps.bind("<<ComboboxSelected>>", _set_ps)
        btn_prev.configure(command=_prev)
        btn_next.configure(command=_next)
        btn_open_returns.configure(command=open_returns_view)
        cb_include.configure(command=_apply_filters_and_refresh)
        cb_include_exp.configure(command=_apply_filters_and_refresh)

        _apply_filters_and_refresh()

    # Quick access button in the top control bar (more discoverable)
    ttk.Button(ctrl, text="Barters Table", command=open_barters_view).pack(side=tk.RIGHT, padx=(8, 0))

    # Bottom actions bar (optional duplicate button)
    bottom_frm = ttk.Frame(root)
    bottom_frm.pack(fill=tk.X, padx=6, pady=(4, 8))
    ttk.Button(bottom_frm, text="Open Barters Table", command=open_barters_view).pack(side=tk.RIGHT)

    # Cache for hover and patches used by plotting functions
    state = {
        "hover_text": fig.text(
            0, 0, "",
            ha="left", va="bottom", fontsize=7,
            bbox=dict(boxstyle="round,pad=0.2", facecolor="#ffffe0", edgecolor="#999999", alpha=0.95),
            visible=False, zorder=10,
        ),
        "hover_targets": [],
        "arrow_patches": [],
        "stats_tab_index": 0,
        "stats_groups": ["Grid & Imports", "Trading", "Battery", "Autarky & Self-consumption"],
        "stats_hover_items": [],
    }

    def goto(h: int) -> None:
        # Title (label, not on the figure)
        if hour_dt is not None and h < len(hour_dt):
            dt_txt = hour_dt.iloc[h].strftime("%Y-%m-%d  %H:%M")
        else:
            dt_txt = f"Hour {h}"
        title_lbl.configure(text=f"Energynet — {dt_txt}")

        # Per-hour slices
        mode = mode_var.get()
        # Swap SoC based on mode if bartering-specific SoC is available
        soc_df_current = soc_df_map.get(mode, soc_df_map["Trades"])
        soc = soc_df_current.iloc[h]
        load = load_w.iloc[h]
        solar = solar_w.iloc[h]
        # cache for other handlers (e.g., pick popups)
        state["current_soc_df"] = soc_df_current
        if mode == "Barters":
            used_edges = barters_used_by_hour[h] if h < len(barters_used_by_hour) else []
            ret_edges = returns_by_hour[h] if h < len(returns_by_hour) else []
            exp_edges = expiry_self_by_hour[h] if h < len(expiry_self_by_hour) else []
            hour_edges = list(used_edges) + list(ret_edges) + list(exp_edges)
            state["stored_markers"] = barters_stored_markers_by_hour[h] if h < len(barters_stored_markers_by_hour) else []
            state["seller_colors"] = seller_colors
        else:
            hour_edges = trades[h] if h < len(trades) else []
            state["stored_markers"] = []

        # Clear axes
        ax_full.clear()
        ax_cap.clear()
        ax_load.clear()
        ax_load_r.clear()
        ax_grid.clear()
        ax_trade.clear()
        ax_trade.axis("off")

        # Draw panels
        bars_full, bars_cap = draw_battery_panels(ax_full, ax_cap, houses, soc, capacity_map)
        # Update title to reflect current view mode
        ax_full.set_title(f"Battery fullness (%) — {mode}", fontsize=FONT_TITLE)
        bars_solar, bars_load = draw_middle_bars(ax_load, ax_load_r, houses, solar, load)
        bars_grid = draw_grid_panel(ax_grid, houses, solar, load, hour_edges)
        draw_trade_network(ax_trade, houses, solar, load, hour_edges, state)

        # Pick precomputed aggregates for selected mode (fast)
        agg_mode = agg_barters if mode == "Barters" else agg_trades

        # Stats panel (auto-fitting; no overlaps)
        draw_stats_panel(ax_stats, state, agg_mode)

        # Register hover items for tooltips
        state["hover_targets"] = []
        state["hover_targets"].extend(state["stats_hover_items"])  # stats overlays first

        # Battery fullness (%)
        pct_series = (pd.Series(soc).reindex(houses) / pd.Series(capacity_map).reindex(houses)) * 100.0
        pct_series = pct_series.fillna(0.0)
        for rect, hid, val in zip(bars_full.patches, houses, pct_series.values):
            rect._hover_title = f"House {hid} — Fullness"
            rect._hover_value = float(val)
            rect._hover_units = "%"; state["hover_targets"].append(rect)

        # Battery capacity (kWh)
        for rect, hid, val in zip(bars_cap.patches, houses, pd.Series(capacity_map).reindex(houses).fillna(0.0).values):
            rect._hover_title = f"House {hid} — Capacity"
            rect._hover_value = float(val)
            rect._hover_units = "kWh"; state["hover_targets"].append(rect)

        # Solar / Load (kWh)
        for rect, hid, val in zip(bars_solar.patches, houses, pd.Series(solar).reindex(houses).fillna(0.0).values):
            rect._hover_title = f"House {hid} — Solar"
            rect._hover_value = float(val)
            rect._hover_units = "kWh"; state["hover_targets"].append(rect)
        for rect, hid, val in zip(bars_load.patches, houses, pd.Series(load).reindex(houses).fillna(0.0).values):
            rect._hover_title = f"House {hid} — Load"
            rect._hover_value = float(val)
            rect._hover_units = "kWh"; state["hover_targets"].append(rect)

        # Grid net (kWh) before trades shown; here we expose (solar - load) for hover
        for rect, hid, val in zip(bars_grid.patches, houses, (pd.Series(solar) - pd.Series(load)).reindex(houses).values):
            rect._hover_title = f"House {hid} — Grid"
            rect._hover_value = float(val)
            rect._hover_units = "kWh"; state["hover_targets"].append(rect)

        # Add marker squares (stored claims) to hover targets if present
        for rect in state.get("marker_hover_items", []):
            state["hover_targets"].append(rect)

        fig.canvas.draw_idle()

    def on_motion(event):
        new_cursor, show_tip = "arrow", False

        # Arrow hit-test first (trade & grid arrows)
        for patch in state.get("arrow_patches", []):
            inside, _ = patch.contains(event)
            if inside:
                new_cursor = "hand2"
                break

        # Bars / stat cards hover
        if new_cursor == "arrow" and event.inaxes is not None:
            for rect in state.get("hover_targets", []):
                same_axes = (rect.axes is event.inaxes)
                # twin-axis group for middle bars
                twin_group = (rect.axes in (ax_load, ax_load_r) and event.inaxes in (ax_load, ax_load_r))
                if not (same_axes or twin_group):
                    continue
                inside, _ = rect.contains(event)
                if inside:
                    fx, fy = fig.transFigure.inverted().transform((event.x, event.y))
                    state["hover_text"].set_position((fx + 0.005, fy + 0.005))
                    title = getattr(rect, "_hover_title", "")
                    value = getattr(rect, "_hover_value", None)
                    units = getattr(rect, "_hover_units", "")
                    expl = getattr(rect, "_hover_expl", None)
                    line2 = expl if expl is not None else (
                        f"{value:.2f} {units}".strip() if value is not None else ""
                    )
                    txt = title if not line2 else f"{title}\n{line2}"
                    state["hover_text"].set_text(txt)
                    state["hover_text"].set_visible(True)
                    new_cursor = "hand2"
                    show_tip = True
                    break

        if not show_tip:
            state["hover_text"].set_visible(False)

        widget = canvas.get_tk_widget()
        if str(widget["cursor"]) != new_cursor:
            widget.configure(cursor=new_cursor)
        fig.canvas.draw_idle()

    def on_pick(event):
        art = event.artist
        # Stats tab switching
        if hasattr(art, "_tab_index"):
            state["stats_tab_index"] = int(getattr(art, "_tab_index"))
            goto(h_var.get())
            return
        # Trade arrow clicked → detailed popup
        if hasattr(art, "_trade_info"):
            info = art._trade_info
            h = h_var.get()
            s_id = info.get("seller")
            b_id = info.get("buyer")
            amt = float(info.get("amount", 0.0))
            typ = info.get("type")

            # Special popup for expiry self-usage (no actual transfer)
            if typ == "expiry-used" and s_id == b_id and isinstance(s_id, int):
                top = tk.Toplevel(root)
                top.title(f"Expired claim used locally @ hour {h} — House {s_id}")
                ttk.Label(top, text=f"House {s_id} used {amt:.2f} kWh from expired claims", font=("Arial", 12, "bold")).pack(padx=16, pady=(14, 6))
                ttk.Label(top, text="Reservation lifted at expiry; no external transfer.", foreground="#555").pack(padx=16, pady=(0, 10))
                ttk.Button(top, text="Close", command=top.destroy).pack(pady=(0, 12))
                top.update_idletasks()
                w, h_ = top.winfo_width(), top.winfo_height()
                x = root.winfo_x() + (root.winfo_width() - w) // 2
                y = root.winfo_y() + (root.winfo_height() - h_) // 2
                top.geometry(f"+{x}+{y}")
                return

            def _soc_triplet(house_id):
                if house_id == "GRID":
                    return None, None, None
                soc_df_pick = state.get("current_soc_df", soc_df)
                soc_kwh = float(soc_df_pick.at[h, house_id])
                cap_kwh = float(capacity_map.get(house_id, 0.0))
                pct = (soc_kwh / cap_kwh * 100.0) if cap_kwh else 0.0
                return soc_kwh, cap_kwh, pct

            soc_s, cap_s, pct_s = _soc_triplet(s_id)
            soc_b, cap_b, pct_b = _soc_triplet(b_id)

            pct_before_s = pct_after_s = pct_before_b = pct_after_b = None
            kwh_before_s = kwh_after_s = kwh_before_b = kwh_after_b = None
            if s_id != "GRID" and soc_s is not None and cap_s:
                pct_after_s = pct_s
                pct_before_s = (soc_s + amt) / cap_s * 100
                kwh_after_s = soc_s
                kwh_before_s = soc_s + amt
            if b_id != "GRID" and soc_b is not None and cap_b:
                pct_after_b = pct_b
                pct_before_b = (soc_b - amt) / cap_b * 100
                kwh_after_b = soc_b
                kwh_before_b = soc_b - amt

            top = tk.Toplevel(root)
            top.title(f"Trade @ hour {h}  –  {s_id} → {b_id}")
            for col in (0, 1):
                top.columnconfigure(col, weight=1)
            ttk.Label(top, text=f"⚡  {s_id}  →  {b_id}        {amt:.2f} kWh transferred", font=("Arial", 16, "bold")).grid(row=0, column=0, columnspan=2, pady=(12, 6))

            def house_row(row, hid, cap, before_pct, after_pct, before_kwh, after_kwh, color):
                if hid == "GRID" or cap in (None, 0):
                    return
                ttk.Label(top, text=f"House {hid}", font=("Arial", 10, "bold")).grid(row=row, column=0, sticky="e", padx=(0, 6), pady=3)
                frm = ttk.Frame(top); frm.grid(row=row, column=1, sticky="w", pady=3)
                ttk.Label(frm, text=f"capacity = {cap:.1f} kWh", foreground="#555").pack(anchor="w")
                ttk.Label(frm, text=f"before = {before_kwh:,.2f} kWh ({before_pct:5.1f} %)", width=28).pack(anchor="w")
                ttk.Label(frm, text=f"after  = {after_kwh:,.2f} kWh ({after_pct:5.1f} %)", width=28).pack(anchor="w")
                CAN_W, CAN_H = 220, 18
                c = tk.Canvas(frm, width=CAN_W, height=CAN_H, highlightthickness=0); c.pack(anchor="w", pady=(2, 0))
                c.create_rectangle(0, 0, CAN_W, CAN_H, fill="#e0e0e0", width=0)
                c.create_rectangle(0, 0, CAN_W * before_pct / 100, CAN_H, fill="#b0c4de", width=0)
                c.create_rectangle(0, 0, CAN_W * after_pct / 100, CAN_H, fill=color, width=0)

            house_row(1, s_id, cap_s, pct_before_s, pct_after_s, kwh_before_s, kwh_after_s, "#4da6ff")
            house_row(2, b_id, cap_b, pct_before_b, pct_after_b, kwh_before_b, kwh_after_b, "#4caf50")

            if "GRID" in (s_id, b_id):
                sign = "supplied" if s_id == "GRID" else "absorbed"
                ttk.Label(top, text=f"⚡  Grid {sign} {amt:.2f} kWh", font=("Arial", 14, "italic"), foreground="#666").grid(row=3, column=0, columnspan=2, pady=(4, 8))

            ttk.Button(top, text="Close", command=top.destroy).grid(row=4, column=0, columnspan=2, pady=(4, 14))
            top.update_idletasks()
            w, h_ = top.winfo_width(), top.winfo_height()
            x = root.winfo_x() + (root.winfo_width() - w) // 2
            y = root.winfo_y() + (root.winfo_height() - h_) // 2
            top.geometry(f"+{x}+{y}")
            return

        # House node clicked → show details popup (This hour vs Outstanding)
        if hasattr(art, "_house_id"):
            h = h_var.get()
            hid = int(getattr(art, "_house_id"))

            top = tk.Toplevel(root)
            top.title(f"House {hid} — Hour {h}")
            for col in (0, 1, 2):
                top.columnconfigure(col, weight=1)

            # Toggle buttons
            mode_local = tk.StringVar(value="This hour")
            frm_tabs = ttk.Frame(top); frm_tabs.grid(row=0, column=0, columnspan=3, sticky="ew", pady=(8, 6), padx=8)
            for i, name in enumerate(["This hour", "Outstanding"]):
                rb = ttk.Radiobutton(frm_tabs, text=name, value=name, variable=mode_local)
                rb.pack(side=tk.LEFT, padx=6)

            content = ttk.Frame(top); content.grid(row=1, column=0, columnspan=3, sticky="nsew", padx=8)
            content.columnconfigure(0, weight=1)
            content.columnconfigure(1, weight=1)

            def _refresh_content(*_):
                for w in content.winfo_children():
                    w.destroy()
                sel = mode_local.get()
                if sel == "This hour":
                    # Incoming barters to this house (used vs stored)
                    ttk.Label(content, text="Incoming barters", font=("Arial", 10, "bold")).grid(row=0, column=0, sticky="w", pady=(2, 4))
                    inc = [r for r in (barters_df[barters_df["t"] == h].to_dict("records") if not barters_df.empty else []) if int(r.get("buyer", -1)) == hid]
                    if inc:
                        for i, r in enumerate(inc, start=1):
                            seller = int(r.get("seller", 0))
                            Es = float(r.get("Es_kWh", 0.0))
                            eta = float(r.get("eta", 0.0))
                            used = eta * Es
                            stored = float(r.get("claim_kWh", 0.0))
                            color = seller_colors.get(seller, "#777777")
                            ttk.Label(content, text=f"Seller {seller}: used {used:.2f} kWh ({eta*100:.1f}%), stored {stored:.2f} kWh", foreground=color).grid(row=i, column=0, sticky="w")
                    else:
                        ttk.Label(content, text="None").grid(row=1, column=0, sticky="w")

                    # Outgoing (lent) from this house
                    ttk.Label(content, text="Outgoing (lent)", font=("Arial", 10, "bold")).grid(row=0, column=1, sticky="w", pady=(2, 4))
                    out = [r for r in (barters_df[barters_df["t"] == h].to_dict("records") if not barters_df.empty else []) if int(r.get("seller", -1)) == hid]
                    if out:
                        for i, r in enumerate(out, start=1):
                            buyer = int(r.get("buyer", 0))
                            Es = float(r.get("Es_kWh", 0.0))
                            eta = float(r.get("eta", 0.0))
                            used = eta * Es
                            stored = float(r.get("claim_kWh", 0.0))
                            expiry = int(r.get("expiry_t", 0))
                            ttk.Label(content, text=f"→ Buyer {buyer}: used {used:.2f} kWh, stored {stored:.2f} kWh (exp t={expiry})").grid(row=i, column=1, sticky="w")
                    else:
                        ttk.Label(content, text="None").grid(row=1, column=1, sticky="w")

                else:
                    # Outstanding at hour h
                    ttk.Label(content, text="Stored in me (by owner)", font=("Arial", 10, "bold")).grid(row=0, column=0, sticky="w", pady=(2, 4))
                    outstanding_in_me: Dict[int, float] = {}
                    if not barters_df.empty:
                        df_b = barters_df[barters_df["t"] <= h]
                        for _, r in df_b[df_b["buyer"] == hid].iterrows():
                            owner = int(r["seller"])  # owner approximated as seller at creation
                            outstanding_in_me[owner] = outstanding_in_me.get(owner, 0.0) + float(r["claim_kWh"])
                    if not claim_returns_df.empty:
                        df_ret = claim_returns_df[claim_returns_df["t"] <= h]
                        for _, r in df_ret[df_ret["stored_on"] == hid].iterrows():
                            owner = int(r["owner"])
                            outstanding_in_me[owner] = outstanding_in_me.get(owner, 0.0) - float(r["qty_kWh"])
                    if outstanding_in_me:
                        rowi = 1
                        for owner, qty in sorted(outstanding_in_me.items()):
                            ttk.Label(content, text=f"Owner {owner}: {max(0.0, qty):.2f} kWh", foreground=seller_colors.get(owner, "#555")).grid(row=rowi, column=0, sticky="w")
                            rowi += 1
                    else:
                        ttk.Label(content, text="None").grid(row=1, column=0, sticky="w")

                    ttk.Label(content, text="My claims stored elsewhere", font=("Arial", 10, "bold")).grid(row=0, column=1, sticky="w", pady=(2, 4))
                    outstanding_my_claims: Dict[int, float] = {}
                    if not barters_df.empty:
                        df_b = barters_df[barters_df["t"] <= h]
                        for _, r in df_b[df_b["seller"] == hid].iterrows():
                            store = int(r["buyer"])
                            outstanding_my_claims[store] = outstanding_my_claims.get(store, 0.0) + float(r["claim_kWh"])
                    if not claim_returns_df.empty:
                        df_ret = claim_returns_df[claim_returns_df["t"] <= h]
                        for _, r in df_ret[df_ret["owner"] == hid].iterrows():
                            store = int(r["stored_on"])
                            outstanding_my_claims[store] = outstanding_my_claims.get(store, 0.0) - float(r["qty_kWh"])
                    if outstanding_my_claims:
                        rowi = 1
                        for store, qty in sorted(outstanding_my_claims.items()):
                            ttk.Label(content, text=f"On {store}: {max(0.0, qty):.2f} kWh").grid(row=rowi, column=1, sticky="w")
                            rowi += 1
                    else:
                        ttk.Label(content, text="None").grid(row=1, column=1, sticky="w")

                # Legend with wrapping layout
                leg = ttk.Frame(content)
                leg.grid(row=99, column=0, columnspan=2, sticky="ew", pady=(8, 8))
                leg.columnconfigure(0, weight=1)
                ttk.Label(leg, text="Seller colors:").grid(row=0, column=0, sticky="w")

                items_frame = ttk.Frame(leg)
                items_frame.grid(row=1, column=0, sticky="ew")
                items_frame.columnconfigure(0, weight=1)

                # Create one small item per seller: [square][id]
                legend_items: List[ttk.Frame] = []
                for sid in sorted(houses):
                    itm = ttk.Frame(items_frame)
                    c = tk.Canvas(itm, width=10, height=10, highlightthickness=0)
                    c.pack(side=tk.LEFT)
                    c.create_rectangle(0, 0, 10, 10, fill=seller_colors.get(sid, "#777"), width=0)
                    ttk.Label(itm, text=str(sid)).pack(side=tk.LEFT, padx=(2, 6))
                    legend_items.append(itm)

                def _reflow_legend(_evt=None):
                    # Approximate per-item width (square+label+padding)
                    w = items_frame.winfo_width() or leg.winfo_width() or 280
                    cell_w = 40
                    cols = max(1, int(w // cell_w))
                    for i, itm in enumerate(legend_items):
                        r, cidx = divmod(i, cols)
                        itm.grid(row=r, column=cidx, padx=2, pady=1, sticky="w")

                items_frame.bind("<Configure>", _reflow_legend)
                _reflow_legend()

            mode_local.trace_add("write", _refresh_content)
            _refresh_content()

            ttk.Button(top, text="Close", command=top.destroy).grid(row=2, column=0, columnspan=3, pady=(6, 10))
            top.update_idletasks()
            w, h_ = top.winfo_width(), top.winfo_height()
            x = root.winfo_x() + (root.winfo_width() - w) // 2
            y = root.winfo_y() + (root.winfo_height() - h_) // 2
            top.geometry(f"+{x}+{y}")
            return

    fig.canvas.mpl_connect("pick_event", on_pick)
    fig.canvas.mpl_connect("motion_notify_event", on_motion)

    def on_scale_change(*_):
        idx = h_var.get()
        lbl.config(text=f"Hour: {idx+1}/{len(hours)}")
        goto(idx)

    h_var.trace_add("write", on_scale_change)

    def on_mode_change(*_):
        goto(h_var.get())
    mode_var.trace_add("write", on_mode_change)

    # Initial draw
    goto(0)
    root.mainloop()
