from __future__ import annotations

from typing import Dict, Iterable, List, Tuple

from .models import BarterEvent, BarterClaim, Trade


def sanity_battery_bounds(soc: Dict[int, float], caps: Dict[int, float]) -> List[str]:
    msgs: List[str] = []
    for hid, s in soc.items():
        cap = caps.get(hid, 0.0)
        if s < -1e-6:
            msgs.append(f"SoC negative for house {hid}: {s}")
        if s - cap > 1e-6:
            msgs.append(f"SoC above capacity for house {hid}: {s} > {cap}")
    return msgs


def sanity_barter_energy(barters: Iterable[BarterEvent]) -> List[str]:
    msgs: List[str] = []
    for b in barters:
        claim = (1.0 - b.eta) * b.Es_kWh
        if abs(claim - b.claim_kWh) > 1e-6:
            msgs.append(
                f"Barter mismatch at t={b.t} seller={b.seller} buyer={b.buyer}: claim {b.claim_kWh} vs (1-eta)Es {claim}"
            )
    return msgs


def sanity_trade_accounting(trades: Iterable[Trade], paid: Dict[int, float], earned: Dict[int, float]) -> List[str]:
    # Here we only check that paid/earned are non-negative and exist; deeper
    # checks require internal engine state (balances per t).
    msgs: List[str] = []
    for hid, val in paid.items():
        if val < -1e-6:
            msgs.append(f"Negative paid for {hid}: {val}")
    for hid, val in earned.items():
        if val < -1e-6:
            msgs.append(f"Negative earned for {hid}: {val}")
    return msgs


def energy_conservation_report(
    solar_kWh: float,
    discharge_kWh: float,
    grid_import_kWh: float,
    in_trades_kWh: float,
    load_kWh: float,
    charge_kWh: float,
    export_kWh: float,
    out_trades_kWh: float,
    tol: float = 1e-6,
) -> Tuple[float, bool]:
    lhs = solar_kWh + discharge_kWh + grid_import_kWh + in_trades_kWh
    rhs = load_kWh + charge_kWh + export_kWh + out_trades_kWh
    diff = lhs - rhs
    return diff, abs(diff) <= tol


