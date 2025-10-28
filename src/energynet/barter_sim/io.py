from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, List, Tuple
import json

import pandas as pd

from .models import Trade, BarterEvent, BarterClaim, ClaimReturnEvent, ClaimExpiryEvent


def ensure_out_dir(out_root: str, folder: int) -> Path:
    out_dir = Path(out_root) / f"folder_{folder}"
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


def write_grid_import(out_dir: Path, grid_import: Dict[int, float]) -> None:
    pd.DataFrame({
        "house_id": list(grid_import.keys()),
        "total_grid_import_kWh": list(grid_import.values()),
    }).to_csv(out_dir / "grid_import.csv", index=False)


def write_trades(out_dir: Path, trades: Iterable[Trade]) -> None:
    df = pd.DataFrame([
        {"t": tr.t, "seller": tr.seller, "buyer": tr.buyer, "qty_kWh": tr.qty_kWh, "price": tr.price}
        for tr in trades
    ])
    if not df.empty:
        df.sort_values(["t", "seller", "buyer"]).to_csv(out_dir / "p2p_trades.csv", index=False)
    else:
        pd.DataFrame(columns=["t", "seller", "buyer", "qty_kWh", "price"]).to_csv(out_dir / "p2p_trades.csv", index=False)


def write_barters(out_dir: Path, barters: Iterable[BarterEvent]) -> None:
    df = pd.DataFrame([
        {
            "t": b.t,
            "seller": b.seller,
            "buyer": b.buyer,
            "Es_kWh": b.Es_kWh,
            "eta": b.eta,
            "claim_kWh": b.claim_kWh,
            "expiry_t": b.expiry_t,
        }
        for b in barters
    ])
    if not df.empty:
        df.sort_values(["t", "seller", "buyer"]).to_csv(out_dir / "barters.csv", index=False)
    else:
        pd.DataFrame(columns=["t", "seller", "buyer", "Es_kWh", "eta", "claim_kWh", "expiry_t"]).to_csv(out_dir / "barters.csv", index=False)


def write_open_claims(out_dir: Path, claims: Iterable[BarterClaim], t_end: int) -> None:
    df = pd.DataFrame([
        {
            "owner": c.owner_id,
            "stored_on": c.stored_on_id,
            "qty_kWh": c.qty_kWh,
            "hours_remaining": max(0, c.expiry_t - t_end),
        }
        for c in claims
        if c.qty_kWh > 1e-9
    ])
    if not df.empty:
        df.sort_values(["owner", "stored_on"]).to_csv(out_dir / "claims_open.csv", index=False)
    else:
        pd.DataFrame(columns=["owner", "stored_on", "qty_kWh", "hours_remaining"]).to_csv(out_dir / "claims_open.csv", index=False)


def write_soc_timeseries(out_dir: Path, soc_series: Iterable[tuple[int, int, float]], tag: str | None = None) -> None:
    df = pd.DataFrame(list(soc_series), columns=["hour_idx", "house_id", "soc_kWh"]) if soc_series else pd.DataFrame(columns=["hour_idx", "house_id", "soc_kWh"])
    if not df.empty:
        df = df.sort_values(["hour_idx", "house_id"]) 
        if tag:
            df.to_csv(out_dir / f"soc_timeseries_{tag}.csv", index=False)
        df.to_csv(out_dir / "soc_timeseries.csv", index=False)
    else:
        df.to_csv(out_dir / "soc_timeseries.csv", index=False)


def write_paid_earned(out_dir: Path, paid_earned: Dict[int, Tuple[float, float]]) -> None:
    df = pd.DataFrame([
        {"house_id": hid, "paid": pe[0], "earned": pe[1], "paid_minus_earned": pe[0] - pe[1]}
        for hid, pe in paid_earned.items()
    ])
    df.sort_values(["house_id"]).to_csv(out_dir / "paid_earned.csv", index=False)


def write_summary(out_dir: Path, summary: Dict) -> None:
    with open(out_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)


def append_log(out_dir: Path, text: str) -> None:
    with open(out_dir / "logs.txt", "a", encoding="utf-8") as f:
        f.write(text.rstrip("\n") + "\n")


# New: write claim return events so the viewer can render return flows
def write_claim_returns(out_dir: Path, return_events: Iterable[ClaimReturnEvent]) -> None:
    df = pd.DataFrame([
        {"t": e.t, "owner": e.owner, "stored_on": e.stored_on, "buyer": e.buyer, "qty_kWh": e.qty_kWh}
        for e in return_events
    ])
    if not df.empty:
        df.sort_values(["t", "owner", "stored_on", "buyer"]).to_csv(out_dir / "claim_returns.csv", index=False)
    else:
        pd.DataFrame(columns=["t", "owner", "stored_on", "buyer", "qty_kWh"]).to_csv(out_dir / "claim_returns.csv", index=False)


# New: write claim expiry events for visibility in the viewer
def write_claim_expiries(out_dir: Path, expiry_events: Iterable[ClaimExpiryEvent]) -> None:
    df = pd.DataFrame([
        {"t": e.t, "owner": e.owner, "stored_on": e.stored_on, "qty_kWh": e.qty_kWh}
        for e in expiry_events
    ])
    if not df.empty:
        df.sort_values(["t", "owner", "stored_on"]).to_csv(out_dir / "claim_expiries.csv", index=False)
    else:
        pd.DataFrame(columns=["t", "owner", "stored_on", "qty_kWh"]).to_csv(out_dir / "claim_expiries.csv", index=False)
