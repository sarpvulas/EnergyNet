from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple


# ─────────────────────────────────────────────────────────────────────────────
# Core data objects
# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class Peer:
    peer_id: int
    battery_cap_kWh: float
    soc_kWh: float = 0.0
    balance_currency: float = 0.0  # initial allowance to spend on P2P
    paid_currency: float = 0.0     # cumulative spending (P2P + grid imports)
    earned_currency: float = 0.0   # cumulative earnings (P2P + grid exports)

    def balance_remaining(self) -> float:
        return max(0.0, self.balance_currency - self.paid_currency)


@dataclass
class Offer:
    seller_id: int
    qty_kWh: float


@dataclass
class Request:
    buyer_id: int
    qty_kWh: float


@dataclass
class BarterClaim:
    owner_id: int
    stored_on_id: int
    qty_kWh: float
    expiry_t: int  # inclusive
    created_t: int


@dataclass
class Trade:
    t: int
    seller: int
    buyer: int
    qty_kWh: float
    price: float


@dataclass
class BarterEvent:
    t: int
    seller: int
    buyer: int
    Es_kWh: float
    eta: float
    claim_kWh: float
    expiry_t: int


@dataclass
class ClaimReturnEvent:
    t: int
    owner: int
    stored_on: int
    buyer: int
    qty_kWh: float


@dataclass
class ClaimExpiryEvent:
    t: int
    owner: int
    stored_on: int
    qty_kWh: float


# ─────────────────────────────────────────────────────────────────────────────
# Containers for simulation state
# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class HourInputs:
    t: int
    load_kWh: Dict[int, float]
    solar_kWh: Dict[int, float]
    battery_cap_kWh: Dict[int, float]
    u_t: float
    fit_t: float


@dataclass
class TimestepState:
    t: int
    offers: List[Offer] = field(default_factory=list)
    requests: List[Request] = field(default_factory=list)
    claims: List[BarterClaim] = field(default_factory=list)
    trades: List[Trade] = field(default_factory=list)
    barters: List[BarterEvent] = field(default_factory=list)
    grid_import_kWh: Dict[int, float] = field(default_factory=dict)
    grid_export_kWh: Dict[int, float] = field(default_factory=dict)


@dataclass
class RunOutputs:
    grid_import_by_house: Dict[int, float]
    trades_all: List[Trade]
    barters_all: List[BarterEvent]
    open_claims: List[BarterClaim]
    paid_earned: Dict[int, Tuple[float, float]]  # paid, earned
    coordinator_payouts: float
    price_series: List[float]
    soc_series: List[Tuple[int, int, float]]
    claim_returns: List[ClaimReturnEvent] = field(default_factory=list)
    claim_expiries: List[ClaimExpiryEvent] = field(default_factory=list)
    grid_export_by_house: Dict[int, float] = field(default_factory=dict)
    total_requests_energy_kWh: float = 0.0


# ─────────────────────────────────────────────────────────────────────────────
# Predictors
# ─────────────────────────────────────────────────────────────────────────────


class SaleabilityPredictor:
    """Interface for AI predictor used by AI_P2P_EB.

    predictor(features_dict) -> saleable_kWh within τ
    """

    def __call__(self, features: Dict) -> float:  # pragma: no cover - interface only
        raise NotImplementedError


class HeuristicPredictor(SaleabilityPredictor):
    """A conservative heuristic predictor used when no model is provided.

    It caps barter energy based on recent supply/demand imbalance and balance
    availability. The aim is to avoid creating unsellable claims.
    """

    def __init__(self, default_ratio: float = 0.5) -> None:
        self.default_ratio = max(0.0, min(1.0, default_ratio))

    def __call__(self, features: Dict) -> float:
        offer_qty = float(features.get("offer_qty", 0.0))
        req_qty = float(features.get("req_qty", 0.0))
        imbalance = float(features.get("recent_R_over_O", 1.0))

        # If there has been abundant demand vs supply (R/O large), allow more
        # else restrict to default conservative fraction of the smaller side.
        base = min(offer_qty, req_qty)
        if base <= 0.0:
            return 0.0
        # Map imbalance (>1 good) to a cap between 0.3..1.0 of base
        if imbalance <= 0:
            imbalance = 1.0
        cap_ratio = max(0.3, min(1.0, 0.6 + 0.2 * (imbalance - 1.0)))
        return max(0.0, min(base, cap_ratio * base * self.default_ratio))


