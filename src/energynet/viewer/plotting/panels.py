# panels.py
# ──────────────────────────────────────────────────────────────────────────────
# Small, pure drawing helpers for the EnergyNet visualization.
# Each function receives precomputed data (no I/O, no heavy logic) and draws on
# the given Matplotlib Axes. Keep this module UI-agnostic and side-effect free.

from __future__ import annotations

from typing import Dict, List, Tuple

import math
import textwrap

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch


# -----------------------------------------------------------------------------
# Global style knobs (kept small so figures scale well on dense dashboards)
# -----------------------------------------------------------------------------
FONT_LBL = 5
FONT_TITLE = 7

# Public API
__all__ = [
    "draw_battery_panels",
    "draw_middle_bars",
    "draw_grid_panel",
    "draw_trade_network",
    "draw_stats_panel",
]


# =============================================================================
# Utility formatters (numbers, titles) – keep all “presentation decisions” here
# =============================================================================

# ---------- helpers (put near the other utilities) ----------
def _estimate_text_size_px(text: str, fontsize: float) -> Tuple[float, float]:
    """
    Cheap width/height estimate in pixels (no renderer needed).
    Rough rule: avg glyph width ≈ 0.6 * fontsize px; height ≈ 1.2 * fontsize px per line.
    """
    lines = text.split("\n")
    max_chars = max((len(l) for l in lines), default=0)
    width = 0.6 * fontsize * max_chars
    height = 1.2 * fontsize * len(lines)
    return width, height

def _fit_fontsize_to_box(text: str, max_w_px: float, max_h_px: float,
                         fs_min: float, fs_max: float, max_lines: int = 2) -> float:
    """
    Find a fontsize that makes `text` fit into (max_w_px, max_h_px).
    Uses binary search on the simple _estimate_text_size_px model.
    """
    lo, hi = fs_min, fs_max
    best = fs_min
    for _ in range(18):
        mid = (lo + hi) / 2
        w, h = _estimate_text_size_px(text, mid)
        if w <= max_w_px and h <= max_h_px and text.count("\n") + 1 <= max_lines:
            best = mid
            lo = mid
        else:
            hi = mid
    return max(fs_min, min(best, fs_max))

def _wrap_and_maybe_shorten(title: str, max_chars_per_line: int, max_lines: int = 2) -> str:
    import textwrap
    lines = textwrap.wrap(title, width=max_chars_per_line)
    if len(lines) <= max_lines:
        return "\n".join(lines)
    # keep first (max_lines) lines; ellipsize the last
    kept = lines[:max_lines]
    if len(kept[-1]) > 3:
        kept[-1] = kept[-1][:-3] + "..."
    return "\n".join(kept)


def _abbr_number(x: float) -> str:
    """
    Abbreviate numbers: 1_234 -> '1.23K', 5_600_000 -> '5.6M'.
    Robust against inf/NaN; trims useless '.0'.
    """
    try:
        x = float(x)
    except Exception:
        return str(x)

    if not math.isfinite(x):
        return str(x)

    sign = "-" if x < 0 else ""
    x = abs(x)
    units = ["", "K", "M", "B", "T"]
    i = 0
    while x >= 1000.0 and i < len(units) - 1:
        x /= 1000.0
        i += 1

    # Keep one decimal for better readability on mid-range values.
    s = f"{x:.1f}" if x < 100 else f"{x:.0f}"
    if s.endswith(".0"):
        s = s[:-2]
    return f"{sign}{s}{units[i]}"


def _fmt_value_for_stat(name: str, val) -> str:
    """
    Human-readable value with units inferred from the metric name.
    - Ratios / degrees: 3 decimals
    - Shares / coverage / self-* : percentage (if in [−2, 2] assume 0–1 and convert)
    - Variance: append kWh²
    - Energy totals (grid draw/sell, charged, discharged, trade volume): kWh
    - Default: abbreviated number
    """
    try:
        v = float(val)
    except Exception:
        return str(val)

    low = name.lower()

    if ("ratio" in low) or ("degree" in low):
        return f"{v:.3f}"

    if any(k in low for k in [
        "share", "coverage", "self-sufficiency", "self consumption", "self-consumption",
        "utilization", "rate", "percent"
    ]):
        if -2.0 <= v <= 2.0:
            return f"{v * 100:.1f}%"
        return f"{v:.1f}%"

    if "variance" in low:
        return _abbr_number(v) + " kWh²"

    if any(k in low for k in [
        "grid draw", "grid sell", "charged", "discharged",
        "trade volume", "imports", "exports", "kwh"
    ]):
        return _abbr_number(v) + " kWh"

    return _abbr_number(v)


def _wrap_title(txt: str, width_chars: int = 22) -> str:
    """
    Wrap long metric titles to at most two lines. Ellipsize the 2nd if needed.
    """
    lines = textwrap.wrap(txt, width=width_chars)
    if len(lines) <= 2:
        return "\n".join(lines)
    second = lines[1]
    if len(second) > 3:
        second = second[:-3] + "..."
    return lines[0] + "\n" + second


def _metric_explanation(name: str) -> str:
    """
    Short tooltip strings used by the GUI layer.
    """
    low = name.lower()
    if "grid draw" in low:
        return (
            "Energy imported from the main grid over the horizon.\n"
            "Formula: $G = \\sum_{h} \\max(0,\; L_h - S_h - P_h)$, where $L_h$ is load, $S_h$ is solar, $P_h$ is peer energy used to cover deficit."
        )
    if "grid sell" in low:
        return (
            "Energy exported to the main grid over the horizon.\n"
            "Formula: $G_{sell} = \\sum_{h} \\max(0,\; S_h - L_h - P_h)$."
        )
    if "import variance" in low:
        return (
            "Variance of per-house total grid imports (higher = less even).\n"
            "Formula: $\\mathrm{Var}(g_i)$ over houses $i$, where $g_i$ is total grid draw for house $i$."
        )
    if "trade volume" in low:
        return (
            "Total peer-to-peer energy traded over the horizon.\n"
            "Formula: $V = \\sum_{h} \\sum_{(i,j)\\in E_h} \\mathrm{amt}_{ij}(h)$."
        )
    if "trade coverage" in low:
        return (
            "Share of deficit met by peers instead of the grid (time-averaged over deficit hours).\n"
            "Per hour: $c_h = \\frac{P_h}{P_h + G_h}$; Overall: $C = \\frac{1}{|H^{\\prime}|} \\sum_{h\\in H^{\\prime}} c_h$,\n"
            "where $G_h=\\max(0, L_h - S_h - P_h)$ and $H^{\\prime}$ are hours with $P_h+G_h>0$."
        )
    if ("avg degree centrality" in low) or ("degree centrality" in low and "weighted" not in low):
        return (
            "Time-averaged degree centrality across houses.\n"
            "Per hour: $d_i(h)=\\frac{|N_i(h)|}{n-1}$. Per house: $\\bar d_i=\\frac{1}{H}\\sum_h d_i(h)$. Overall: $\\frac{1}{n}\\sum_i \\bar d_i$."
        )
    if "top hub (house, degree)" in low or ("top hub" in low and "weighted" not in low):
        return (
            "House with highest time-averaged degree centrality and its value.\n"
            "Formula: $\\mathrm{argmax}_i\\, \\bar d_i$ with degree $\\max_i \\bar d_i$."
        )
    if "avg weighted degree" in low or "weighted degree" in low:
        return (
            "Time-averaged normalized trade strength per house (0–1), averaged across houses.\n"
            "Per hour: $s_i(h)=\\sum_j \\mathrm{amt}_{ij}(h)$, $w_i(h)=\\frac{s_i(h)}{\\max_k s_k(h)}$. Per house: $\\bar w_i=\\frac{1}{H}\\sum_h w_i(h)$. Overall: $\\frac{1}{n}\\sum_i \\bar w_i$."
        )
    if "top weighted hub" in low or ("top hub" in low and "weighted" in low):
        return (
            "House with highest time-averaged normalized trade strength and its value.\n"
            "Formula: $\\mathrm{argmax}_i\\, \\bar w_i$ with strength $\\max_i \\bar w_i$."
        )
    if "charged" in low:
        return (
            "Total energy charged into all batteries over the horizon.\n"
            "Formula: $C_{bat} = \\sum_{h,i} \\text{charge}_{i}(h)$."
        )
    if "discharged" in low:
        return (
            "Total energy discharged from all batteries over the horizon.\n"
            "Formula: $D_{bat} = \\sum_{h,i} \\text{discharge}_{i}(h)$."
        )
    if "average soc" in low:
        return (
            "Mean state-of-charge across houses and time, as percent of capacity.\n"
            "Formula: $\\tfrac{1}{Hn} \\sum_{h,i} 100 \\times \\tfrac{\\text{SOC}_i(h)}{\\text{Cap}_i}$."
        )
    if "self-sufficiency" in low:
        return (
            "Share of total load served without using the main grid (higher is better).\n"
            "Formula: $1 - \\frac{G}{\\sum_h L_h}$, with $G$ recomputed per hour as above."
        )
    if "self-consumption" in low:
        return (
            "Share of solar production used on-site (not exported to grid or peers).\n"
            "Formula: $1 - \\frac{G_{sell} + P_{sell}}{\\sum_h S_h}$, where $G_{sell}$ is grid export and $P_{sell}$ is total sold to peers."
        )
    return ""


# =============================================================================
# Panels
# =============================================================================

def draw_battery_panels(
    ax_full: plt.Axes,
    ax_cap: plt.Axes,
    houses: List[int],
    soc: pd.Series,
    capacity_map: Dict[int, float]
) -> Tuple[plt.BarContainer, plt.BarContainer]:
    """
    Top-left: battery fullness (% of capacity) by house.
    Top-right: battery capacity (kWh) by house.
    """
    # Align series by house id to avoid index mismatches.
    caps = pd.Series(capacity_map, dtype=float).reindex(houses)
    pct = (pd.Series(soc, dtype=float).reindex(houses) / caps.replace(0, np.nan)) * 100.0
    pct = pct.fillna(0.0)

    colors = pct.apply(lambda p: "seagreen" if p >= 0 else "#d62728")
    bars_full = ax_full.bar(houses, pct, color=colors)
    ax_full.set_title("Battery fullness (%)", fontsize=FONT_TITLE)
    ymin = min(-10, float(pct.min()) * 1.1) if len(pct) else -10
    ax_full.set_ylim(ymin, 110)
    ax_full.axhline(0, color="k", linewidth=.8)
    ax_full.set_xlabel("House", fontsize=FONT_LBL)
    ax_full.set_ylabel("%", fontsize=FONT_LBL)
    ax_full.tick_params(axis="both", labelsize=FONT_LBL)
    ax_full.set_xticks(houses)
    ax_full.set_xticklabels([str(h) for h in houses], fontsize=FONT_LBL)

    bars_cap = ax_cap.bar(houses, caps.fillna(0.0), color="lightgray", edgecolor="k")
    ax_cap.set_title("Battery capacity (kWh)", fontsize=FONT_TITLE)
    ymax = float(caps.max()) * 1.1 if len(caps) and np.isfinite(caps.max()) else 1.0
    ax_cap.set_ylim(0, ymax)
    ax_cap.set_xlabel("House", fontsize=FONT_LBL)
    ax_cap.set_ylabel("kWh", fontsize=FONT_LBL)
    ax_cap.tick_params(axis="both", labelsize=FONT_LBL)
    ax_cap.set_xticks(houses)
    ax_cap.set_xticklabels([str(h) for h in houses], fontsize=FONT_LBL)

    return bars_full, bars_cap


def draw_middle_bars(
    ax_left: plt.Axes,
    ax_right: plt.Axes,
    houses: List[int],
    solar: pd.Series,
    load: pd.Series
) -> Tuple[plt.BarContainer, plt.BarContainer]:
    """
    Middle row: left axis used for Solar, right axis for Load (twin layout).
    """
    x = np.arange(len(houses))
    w = 0.4
    s = pd.Series(solar, dtype=float).reindex(houses).fillna(0.0)
    l = pd.Series(load, dtype=float).reindex(houses).fillna(0.0)

    bars_solar = ax_left.bar(x - w / 2, s, w, color="goldenrod", label="Solar")
    bars_load = ax_right.bar(x + w / 2, l, w, color="firebrick", label="Load")

    ax_left.set_title("Load vs Solar (kWh)", fontsize=FONT_TITLE)
    ax_left.set_ylabel("Solar (kWh)", fontsize=FONT_LBL, color="goldenrod", labelpad=2)
    ax_left.yaxis.set_label_position("left")
    ax_left.yaxis.tick_left()

    ax_right.set_ylabel("Load (kWh)", fontsize=FONT_LBL, color="firebrick", labelpad=2)
    ax_right.yaxis.set_label_position("right")
    ax_right.yaxis.tick_right()

    ax_left.set_xlabel("House", fontsize=FONT_LBL)
    ax_left.set_xticks(x)
    ax_left.set_xticklabels(houses, fontsize=FONT_LBL)

    max_solar = float(s.max()) if len(s) else 0.0
    max_load  = float(l.max()) if len(l) else 0.0
    ax_left.set_ylim(0, max_solar * 1.1 if max_solar else 1)
    ax_right.set_ylim(0, max_load * 1.1 if max_load else 1)

    ax_left.tick_params(axis="y", labelsize=FONT_LBL, colors="goldenrod")
    ax_right.tick_params(axis="y", labelsize=FONT_LBL, colors="firebrick")

    # Legend: place outside to avoid label collisions in tight layouts
    ax_left.legend(
        [bars_solar[0], bars_load[0]], ["Solar", "Load"],
        loc="upper center", bbox_to_anchor=(0.5, -0.10),
        ncol=2, frameon=False, fontsize=FONT_LBL
    )
    return bars_solar, bars_load


def draw_grid_panel(
    ax: plt.Axes,
    houses: List[int],
    solar: pd.Series,
    load: pd.Series,
    trades_for_hour: List[dict]
) -> plt.BarContainer:
    """
    Bottom-left small panel: net flow with the grid per house after trades.
    Positive = selling to grid; negative = drawing from grid.
    """
    s = pd.Series(solar, dtype=float).reindex(houses).fillna(0.0)
    l = pd.Series(load, dtype=float).reindex(houses).fillna(0.0)
    grid_flow_house = s - l

    # Apply peer trades: seller reduces net, buyer increases net
    for t in trades_for_hour:
        grid_flow_house[t["seller"]] -= t["amount"]
        grid_flow_house[t["buyer"]] += t["amount"]

    x = np.arange(len(houses))
    bar_w = 0.35
    colors = ["#ff9300" if v > 0 else "#d62728" for v in grid_flow_house]
    bars_grid = ax.bar(x, grid_flow_house, width=bar_w, color=colors)

    ax.axhline(0, color="k", linewidth=.6)
    ax.set_xticks(x)
    ax.set_xticklabels(houses, fontsize=FONT_LBL)
    ax.set_ylabel("kWh", fontsize=FONT_LBL)
    ax.set_xlabel("House", fontsize=FONT_LBL)
    ax.set_title("Grid draw / sell", fontsize=FONT_TITLE)

    max_grid_abs = float((s - l).abs().max()) * 1.1 if len(s) else 1.0
    ax.set_ylim(-max_grid_abs, max_grid_abs)
    ax.tick_params(axis="y", labelsize=FONT_LBL)

    return bars_grid


def draw_trade_network(
    ax: plt.Axes,
    houses: List[int],
    solar: pd.Series,
    load: pd.Series,
    trades_for_hour: List[dict],
    state: dict
) -> None:
    """
    Network-style panel: nodes are houses (blue=net seller, red=net buyer).
    Green arrows = peer trades; orange/red dashed arrows = grid interactions.
    """
    ax.clear()
    ax.axis("off")
    ax.set_title("Trades (kWh)", fontsize=FONT_TITLE)
    # Slightly looser bounds to keep badges in frame
    ax.set_xlim(-1.2, 1.2)
    ax.set_ylim(-1.4, 1.4)
    ax.set_aspect("equal", adjustable="box")

    n = len(houses)
    s = pd.Series(solar, dtype=float).reindex(houses).fillna(0.0)
    l = pd.Series(load, dtype=float).reindex(houses).fillna(0.0)
    net = s - l
    for t in trades_for_hour:
        net[t["seller"]] -= t["amount"]
        net[t["buyer"]] += t["amount"]

    # Layout on a horizontal ellipse, slightly shifted right for the Grid square
    MARGIN = 0.05
    ELLIPSE_A = 1.0 - MARGIN
    ELLIPSE_B = 1.2
    SHIFT_X = 0.06
    coords = {
        hid: (SHIFT_X + ELLIPSE_A * np.cos(2 * np.pi * i / n),
              ELLIPSE_B * np.sin(2 * np.pi * i / n))
        for i, hid in enumerate(houses)
    }

    # Node size adapts to node count
    min_axis = min(ELLIPSE_A, ELLIPSE_B)
    NODE_R = 0.9 * np.pi * min_axis / n if n else 0.02
    NODE_R = max(0.020, min(0.070, NODE_R))

    # Draw nodes
    for hid, (x0, y0) in coords.items():
        face = "tab:blue" if net[hid] >= 0 else "tab:red"
        circ = Circle((x0, y0), radius=NODE_R, facecolor=face, edgecolor="k", linewidth=1, picker=True)
        circ._house_id = hid  # used by the GUI picker
        ax.add_patch(circ)
        ax.text(x0, y0, str(hid), ha="center", va="center", color="white", fontweight="bold", fontsize=9)

    # Grid center (square)
    grid_flow = float(net.sum())
    grid_color = "#444444"
    if grid_flow > 0:
        grid_color = "#ff9300"  # net export
    elif grid_flow < 0:
        grid_color = "#d62728"  # net import

    grid_xy = (SHIFT_X, 0)
    ax.scatter(*grid_xy, s=550, marker="s", color=grid_color, edgecolor="k", zorder=4)
    ax.text(*grid_xy, "Grid", ha="center", va="center", color="white", fontweight="bold", fontsize=9, zorder=5)

    # Arrows: grid ↔ houses (dashed, width ∝ |flow|)
    state["arrow_patches"] = []
    max_grid = float(np.max(np.abs(list(net.values)))) if len(net) else 0.0
    for hid, diff in net.items():
        if abs(diff) < 1e-6:
            continue
        if diff > 0:
            start, end = coords[hid], grid_xy
            arr_color = "#ff9300"
            seller, buyer = hid, "GRID"
        else:
            start, end = grid_xy, coords[hid]
            arr_color = "#d62728"
            seller, buyer = "GRID", hid
        lw = 2 * abs(diff) / max_grid if max_grid else 1.5
        arrow = FancyArrowPatch(
            start, end, arrowstyle="->", mutation_scale=10,
            linewidth=lw, alpha=0.6, color=arr_color, linestyle="--",
            picker=6, zorder=2
        )
        arrow._trade_info = {"seller": seller, "buyer": buyer, "amount": float(abs(diff))}
        ax.add_patch(arrow)
        state["arrow_patches"].append(arrow)

    # Arrows: used barters (solid green), returns (solid purple), expiry self-usage (orange self-loop)
    max_amt_house = max((t.get("amount", 0.0) for t in trades_for_hour), default=0.0)
    for t in trades_for_hour:
        s_id, b_id, amt = t.get("seller"), t.get("buyer"), float(t.get("amount", 0.0))
        if s_id not in coords or b_id not in coords or amt <= 1e-12:
            continue
        (x0, y0), (x1, y1) = coords[s_id], coords[b_id]
        lw = 2 * amt / max_amt_house if max_amt_house else 1.5
        typ = t.get("type", "trade")
        if typ == "expiry-used" and s_id == b_id:
            # Draw a small self-loop arc around the node
            r = NODE_R * 1.9
            ang0 = np.deg2rad(40)
            ang1 = -np.deg2rad(40)
            xs, ys = x0 + r * np.cos(ang0), y0 + r * np.sin(ang0)
            xe, ye = x0 + r * np.cos(ang1), y0 + r * np.sin(ang1)
            arrow = FancyArrowPatch(
                (xs, ys), (xe, ye), connectionstyle="arc3,rad=-1.1",
                arrowstyle="->", mutation_scale=12,
                linewidth=lw, alpha=0.9, color="#f59e0b",
                picker=6, zorder=4
            )
            arrow._trade_info = {"seller": s_id, "buyer": b_id, "amount": amt, "type": "expiry-used"}
        else:
            color = "#2ca02c" if typ in ("barter-used", "trade") else "#7e22ce"
            arrow = FancyArrowPatch(
                (x0, y0), (x1, y1), arrowstyle="->", mutation_scale=12,
                linewidth=lw, alpha=0.85, color=color,
                picker=6, zorder=3
            )
            arrow._trade_info = {"seller": s_id, "buyer": b_id, "amount": amt, "type": typ}
        ax.add_patch(arrow)
        state["arrow_patches"].append(arrow)

    # Badges: stored claim markers arranged radially around each buyer node
    state["marker_hover_items"] = []
    markers = state.get("stored_markers", []) or []
    seller_colors: Dict[int, str] = state.get("seller_colors", {}) or {}
    # Group markers by buyer to position them without overlap
    by_buyer: Dict[int, List[dict]] = {}
    for m in markers:
        try:
            buyer = int(m.get("buyer", -1))
        except Exception:
            buyer = -1
        if buyer not in by_buyer:
            by_buyer[buyer] = []
        by_buyer[buyer].append(m)

    for buyer, ms in by_buyer.items():
        if buyer not in coords:
            continue
        (xb, yb) = coords[buyer]
        k = len(ms)
        if k <= 0:
            continue
        # Base box side and base ring radius
        side = NODE_R * 0.70
        ring_base = NODE_R * 1.5
        # Aim to place badges on the inward-facing arc towards the grid center
        center_x, center_y = SHIFT_X, 0.0
        inward_ang = math.atan2(center_y - yb, center_x - xb)
        ARC = math.radians(140.0)  # arc width around inward direction
        # Capacity per ring based on arc length and badge size with gap factor
        gap = 1.35
        per_ring = max(4, int((ARC * ring_base) / (side * gap)))
        rings = max(1, int(math.ceil(k / per_ring)))

        idx = 0
        for ring in range(rings):
            ring_r = ring_base + ring * (side * 0.9)
            # Recompute per-ring capacity for larger radius (more space)
            cap = max(4, int((ARC * ring_r) / (side * gap)))
            take = min(cap, k - idx)
            if take <= 0:
                break
            if take == 1:
                angles = [inward_ang]
            else:
                angles = [inward_ang - ARC / 2 + i * (ARC / (take - 1)) for i in range(take)]
            for i in range(take):
                m = ms[idx + i]
                seller = int(m.get("seller", -1))
                amt = float(m.get("amount", 0.0))
                if seller not in seller_colors or amt <= 1e-12:
                    continue
                ang = angles[i]
                bx, by = xb + ring_r * math.cos(ang), yb + ring_r * math.sin(ang)
                rect = plt.Rectangle(
                    (bx - side / 2, by - side / 2), side, side,
                    facecolor=seller_colors.get(seller, "#666"), edgecolor="k", linewidth=0.9,
                    zorder=4
                )
                rect._hover_title = f"Claim stored from seller {seller}"
                rect._hover_value = float(amt)
                rect._hover_units = "kWh"
                rect._hover_expl = None
                ax.add_patch(rect)
                state["marker_hover_items"].append(rect)
                ax.text(bx, by, str(seller), ha="center", va="center", fontsize=7, color="white", zorder=5)
            idx += take


def draw_stats_panel(ax, state: dict, aggregates: Dict[str, Dict[str, float]]) -> None:
    groups = state.get("stats_groups", []) or ["Statistics"]
    tab_idx = int(state.get("stats_tab_index", 0))
    tab_idx = max(0, min(tab_idx, len(groups) - 1))
    grp_name = groups[tab_idx]

    ax.clear()
    ax.axis("off")

    # Axes size in pixels
    fig = ax.figure
    try:
        bbox = ax.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
        ax_w_in, ax_h_in = bbox.width, bbox.height
    except Exception:
        ax_w_in, ax_h_in = fig.get_size_inches()
    dpi = fig.dpi
    ax_w_px, ax_h_px = ax_w_in * dpi, ax_h_in * dpi

    # Panel title
    title_fs = max(8, min(14, 0.035 * ax_h_px))
    ax.set_title(f"Statistics — {grp_name}", fontsize=title_fs, pad=6)

    # Items
    items = list(aggregates.get(grp_name, {}).items())
    n = len(items)

    if n <= 6:
        cols = 2
    elif n <= 12:
        cols = 3
    else:
        cols = 4
    rows = max(1, (n + cols - 1) // cols)

    # Layout for cards
    left, right = 0.035, 0.965
    top_y = 0.92
    bottom_tabs = 0.20        # ⬅ more reserved space for tabs (margin)
    gap_x, gap_y = 0.020, 0.025

    avail_w = (right - left)
    avail_h = (top_y - bottom_tabs)
    card_w = (avail_w - gap_x * (cols - 1)) / max(1, cols)
    card_h = (avail_h - gap_y * (rows - 1)) / max(1, rows)

    # Pixel sizes
    card_w_px = card_w * ax_w_px
    card_h_px = card_h * ax_h_px

    pad_px = max(5.0, 0.025 * min(card_w_px, card_h_px))
    title_frac = 0.44
    title_box_h_px = title_frac * card_h_px
    value_box_h_px = card_h_px - title_box_h_px - 2 * pad_px

    title_fs_min, title_fs_max = 3, 4
    value_fs_min, value_fs_max = 3, 4

    state["stats_hover_items"] = []

    for i, (name, val) in enumerate(items):
        r = i // cols
        c = i % cols
        x0 = left + c * (card_w + gap_x)
        y_top = top_y - r * (card_h + gap_y)
        y0 = y_top - card_h

        # Card
        card = FancyBboxPatch(
            (x0, y0), card_w, card_h,
            boxstyle="round,pad=0.010,rounding_size=0.02",
            transform=ax.transAxes,
            facecolor="#ffffff", edgecolor="#e5e7eb",
            linewidth=0.9, zorder=2
        )
        ax.add_patch(card)

        # Title
        approx_chars = max(12, int((card_w_px / (0.6 * 10)) * 0.9))
        title_wrapped = _wrap_and_maybe_shorten(name, max_chars_per_line=approx_chars, max_lines=2)
        title_fs = _fit_fontsize_to_box(
            title_wrapped,
            max_w_px=card_w_px - 2 * pad_px,
            max_h_px=title_box_h_px - 0.5 * pad_px,
            fs_min=title_fs_min, fs_max=title_fs_max, max_lines=2
        )
        ax.text(
            x0 + 0.012, y_top - 0.012,
            title_wrapped, transform=ax.transAxes,
            ha="left", va="top",
            fontsize=title_fs, color="#111827", zorder=3, clip_on=True
        )

        # Divider line
        y_div = y_top - title_frac * card_h - 0.004
        ax.plot(
            [x0 + 0.010, x0 + card_w - 0.010], [y_div, y_div],
            transform=ax.transAxes, color="#e5e7eb", linewidth=0.9, zorder=3
        )

        # Value
        txt_val = _fmt_value_for_stat(name, val)
        value_fs = _fit_fontsize_to_box(
            txt_val,
            max_w_px=card_w_px - 2 * pad_px,
            max_h_px=value_box_h_px - 0.5 * pad_px,
            fs_min=value_fs_min, fs_max=value_fs_max, max_lines=1
        )
        value_center_y = y0 + (card_h * (1.0 - title_frac)) * 0.52
        ax.text(
            x0 + 0.012, value_center_y,
            txt_val, transform=ax.transAxes,
            ha="left", va="center",
            fontsize=value_fs, color="#374151", zorder=3, clip_on=True
        )

        # Hover overlay
        overlay = plt.Rectangle(
            (x0, y0), card_w, card_h,
            transform=ax.transAxes,
            facecolor="none", edgecolor="none",
            picker=True, zorder=4
        )
        overlay._hover_title = name
        overlay._hover_expl = _metric_explanation(name)
        overlay.axes = ax
        ax.add_patch(overlay)
        state["stats_hover_items"].append(overlay)

    # Tabs (bigger)
    n_tabs = len(groups)
    x = 0.035
    total_gap = 0.025 * max(0, n_tabs - 1)
    max_w_each = (0.965 - 0.035 - total_gap) / max(1, n_tabs)
    tab_h = 0.140                             # ⬅ taller
    tab_fs_min, tab_fs_max = 4, 5

    def _wrap_tab_label(label: str, tab_w_px: float) -> str:
        label = label.replace(" & ", " &\n")  # force break before/after &
        if "\n" not in label:
            max_chars = max(10, int((tab_w_px / 6.0)))
            label = _wrap_and_maybe_shorten(label, max_chars_per_line=max_chars, max_lines=2)
        return label

    for i, title in enumerate(groups):
        w = max(0.25, min(0.40, max_w_each))  # ⬅ wider tabs
        rect = plt.Rectangle(
            (x, 0.02), w, tab_h,
            transform=ax.transAxes,
            facecolor=("#dbeafe" if i == tab_idx else "#f3f4f6"),
            edgecolor="#9ca3af",
            picker=True, zorder=3
        )
        rect._tab_index = i
        ax.add_patch(rect)

        tab_w_px = w * ax_w_px
        label_wrapped = _wrap_tab_label(title, tab_w_px)
        tab_fs = max(tab_fs_min, min(tab_fs_max, 0.026 * ax_h_px))

        ax.text(
            x + w / 2, 0.02 + tab_h / 2,
            label_wrapped, transform=ax.transAxes,
            ha="center", va="center",
            fontsize=tab_fs, color="#111827", zorder=4
        )
        x += w + 0.025
