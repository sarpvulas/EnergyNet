from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional, Iterable
import random

from .models import (
    Peer, Offer, Request, BarterClaim, Trade, BarterEvent, ClaimReturnEvent, ClaimExpiryEvent,
    HourInputs, TimestepState, RunOutputs, SaleabilityPredictor, HeuristicPredictor,
)
from .pricing import PriceSmoother, compute_price, profitability_warning_eta
from .matching import fcfs_match_affordable, fcfs_barter_pairs


@dataclass
class Config:
    folder: int
    algo: str
    eta: float
    tau: int
    price_window: int
    priority_rule: str
    random_seed: int
    balance_mode: str
    balance_percent: float
    out_root: str
    verbose: bool
    expiry_action: str  # "consumer_keeps" | "return_to_lender"


class BarterSimulator:
    def __init__(
        self,
        peers: Dict[int, Peer],
        util_prices: List[float],
        fit_prices: List[float],
        loads: List[Dict[int, float]],
        solars: List[Dict[int, float]],
        caps: Dict[int, float],
        config: Config,
        predictor: Optional[SaleabilityPredictor] = None,
    ) -> None:
        self.peers = peers
        self.util_prices = util_prices
        self.fit_prices = fit_prices
        self.loads = loads
        self.solars = solars
        self.caps = caps
        self.cfg = config
        self.predictor = predictor or HeuristicPredictor()
        random.seed(self.cfg.random_seed)

        self.price_smoother = PriceSmoother(self.cfg.price_window)
        self.claims: List[BarterClaim] = []
        self.trades: List[Trade] = []
        self.barters: List[BarterEvent] = []
        self.claim_returns: List[ClaimReturnEvent] = []
        self.claim_expiries: List[ClaimExpiryEvent] = []
        self.grid_import_tot: Dict[int, float] = {pid: 0.0 for pid in peers}
        self.coordinator_payouts: float = 0.0
        self.price_series: List[float] = []
        self.soc_series: List[Tuple[int, int, float]] = []

    # ─────────────────────────────────────────────────────────────────────
    # Helpers
    # ─────────────────────────────────────────────────────────────────────
    def _handle_expiries_upfront(self, t: int) -> None:
        """Process claims that expired before this hour (expiry_t < t).

        If expiry_action is 'return_to_lender', attempt to physically return
        energy to the owner (bounded by store SoC and owner's free capacity),
        logging a ClaimReturnEvent. Any remaining quantity is logged as a
        ClaimExpiryEvent. Finally, drop processed expired claims.
        """
        expired = [c for c in self.claims if c.expiry_t < t and c.qty_kWh > 1e-9]
        for c in expired:
            exp_qty = float(c.qty_kWh)
            if self.cfg.expiry_action == "return_to_lender":
                store_id = c.stored_on_id
                owner_id = c.owner_id
                if store_id in self.peers and owner_id in self.peers:
                    store_soc = max(0.0, self.peers[store_id].soc_kWh)
                    owner_free_cap = max(0.0, self.caps.get(owner_id, 0.0) - self.peers[owner_id].soc_kWh)
                    take = min(exp_qty, store_soc, owner_free_cap)
                    if take > 1e-12:
                        self.peers[store_id].soc_kWh = max(0.0, self.peers[store_id].soc_kWh - take)
                        self.peers[owner_id].soc_kWh = min(self.caps.get(owner_id, 0.0), self.peers[owner_id].soc_kWh + take)
                        self.claim_returns.append(ClaimReturnEvent(
                            t=t, owner=owner_id, stored_on=store_id, buyer=owner_id, qty_kWh=float(take)
                        ))
                        exp_qty = max(0.0, exp_qty - take)
            if exp_qty > 1e-12:
                self.claim_expiries.append(ClaimExpiryEvent(
                    t=t, owner=c.owner_id, stored_on=c.stored_on_id, qty_kWh=float(exp_qty)
                ))

        # Remove processed expired claims; keep live claims only
        if expired:
            self.claims = [c for c in self.claims if not (c.expiry_t < t) and c.qty_kWh > 1e-9]
    def _build_physical_balances(self, t: int) -> Tuple[List[Offer], List[Request]]:
        offers: List[Offer] = []
        requests: List[Request] = []

        # Reserved energy per peer from active claims (cannot be used by that peer)
        reserved_by_peer: Dict[int, float] = {}
        for c in self.claims:
            if c.expiry_t >= t and c.qty_kWh > 1e-9:
                reserved_by_peer[c.stored_on_id] = reserved_by_peer.get(c.stored_on_id, 0.0) + c.qty_kWh

        for pid, peer in self.peers.items():
            load = max(0.0, self.loads[t].get(pid, 0.0))
            solar = max(0.0, self.solars[t].get(pid, 0.0))

            # Serve load from solar first
            direct_solar = min(solar, load)
            load -= direct_solar
            solar -= direct_solar

            # If self-prioritized bartering is enabled, consume own claims first
            if self.cfg.algo == "S_EB" and load > 1e-12:
                for c in self.claims:
                    if load <= 1e-12:
                        break
                    if c.owner_id != pid or c.expiry_t < t or c.qty_kWh <= 1e-12:
                        continue
                    store_id = c.stored_on_id
                    # Energy actually available to withdraw is limited by reserved on store and its SoC
                    reserved_available = min(c.qty_kWh, reserved_by_peer.get(store_id, 0.0))
                    if reserved_available <= 1e-12:
                        continue
                    phys_avail = min(reserved_available, self.peers[store_id].soc_kWh)
                    if phys_avail <= 1e-12:
                        continue
                    take = min(load, phys_avail)
                    if take <= 1e-12:
                        continue
                    # Apply withdrawal
                    c.qty_kWh -= take
                    self.peers[store_id].soc_kWh = max(0.0, self.peers[store_id].soc_kWh - take)
                    reserved_by_peer[store_id] = max(0.0, reserved_by_peer.get(store_id, 0.0) - take)
                    load -= take
                    # Log self-consumption of claims as return flow (owner reclaims from stored_on)
                    self.claim_returns.append(ClaimReturnEvent(
                        t=t, owner=pid, stored_on=store_id, buyer=pid, qty_kWh=float(take)
                    ))

            # Battery discharge to serve remaining load, excluding reserved energy
            available_soc = max(0.0, peer.soc_kWh - reserved_by_peer.get(pid, 0.0))
            discharge = min(load, available_soc)
            peer.soc_kWh -= discharge
            load -= discharge

            if load > 1e-12:
                requests.append(Request(pid, load))

            # Store surplus solar into battery
            free_cap = max(0.0, self.caps.get(pid, 0.0) - peer.soc_kWh)
            charge = min(solar, free_cap)
            peer.soc_kWh += charge
            solar -= charge

            if solar > 1e-12:
                offers.append(Offer(pid, solar))

        return offers, requests

    def _compute_price(self, t: int, offers: List[Offer], requests: List[Request]) -> float:
        R_t = sum(r.qty_kWh for r in requests)
        O_t = sum(o.qty_kWh for o in offers)
        u_t = self.util_prices[t]
        fit_t = self.fit_prices[t]
        avg_recent = self.price_smoother.average() if self.price_series else None
        p_t = compute_price(u_t, fit_t, R_t, O_t, avg_recent)
        self.price_smoother.append(p_t)
        self.price_series.append(p_t)
        return p_t

    def _sell_existing_claims(self, t: int, p_t: float, requests: List[Request]) -> Tuple[List[Trade], List[Request]]:
        # Convert active claims into owner offers and FCFS match
        active_claims = [c for c in self.claims if c.expiry_t >= t and c.qty_kWh > 1e-9]
        owner_offer_by_owner: Dict[int, float] = {}
        for c in active_claims:
            owner_offer_by_owner[c.owner_id] = owner_offer_by_owner.get(c.owner_id, 0.0) + c.qty_kWh

        offers = [Offer(owner, qty) for owner, qty in owner_offer_by_owner.items() if qty > 1e-9]

        # Buyers' balances for affordability
        balances = {pid: self.peers[pid].balance_remaining() for pid in self.peers}
        trades_raw, rem_offers, rem_requests = fcfs_match_affordable(offers, requests, p_t, balances)

        # Apply accounting: buyers pay, sellers earn; reduce claims FIFO
        trades: List[Trade] = []
        for seller_id, buyer_id, qty in trades_raw:
            cost = qty * p_t
            self.peers[buyer_id].paid_currency += cost
            # Coordinator (id -1) is not a real peer account; skip credit in that case
            if seller_id in self.peers:
                self.peers[seller_id].earned_currency += cost
            trades.append(Trade(t, seller_id, buyer_id, qty, p_t))

            # Reduce seller's claims FIFO
            need = qty
            for c in self.claims:
                if need <= 1e-12:
                    break
                if c.owner_id != seller_id or c.expiry_t < t or c.qty_kWh <= 0:
                    continue
                take = min(c.qty_kWh, need)
                if take <= 1e-12:
                    continue
                c.qty_kWh -= take
                # Physically withdraw from the storing consumer's battery
                store_id = c.stored_on_id
                if store_id in self.peers:
                    self.peers[store_id].soc_kWh = max(0.0, self.peers[store_id].soc_kWh - take)
                # Log physical claim redemption (return) event
                self.claim_returns.append(ClaimReturnEvent(
                    t=t, owner=seller_id, stored_on=store_id, buyer=buyer_id, qty_kWh=float(take)
                ))
                need -= take

        # Keep claims with remaining quantity; expiry handling (and logging) is done in the per-step maintenance.
        self.claims = [c for c in self.claims if c.qty_kWh > 1e-9]
        return trades, rem_requests

    def _p2p_trading(self, t: int, p_t: float, offers: List[Offer], requests: List[Request]) -> Tuple[List[Trade], List[Offer], List[Request]]:
        balances = {pid: self.peers[pid].balance_remaining() for pid in self.peers}
        trades_raw, rem_offers, rem_requests = fcfs_match_affordable(offers, requests, p_t, balances)
        trades: List[Trade] = []
        for seller_id, buyer_id, qty in trades_raw:
            cost = qty * p_t
            self.peers[buyer_id].paid_currency += cost
            self.peers[seller_id].earned_currency += cost
            trades.append(Trade(t, seller_id, buyer_id, qty, p_t))
        return trades, rem_offers, rem_requests

    def _barter_phase(self, t: int, p_t: float, offers: List[Offer], requests: List[Request]) -> Tuple[List[BarterEvent], List[Offer], List[Request]]:
        events: List[BarterEvent] = []
        pairs = fcfs_barter_pairs(offers, requests)
        for seller_id, buyer_id, es in pairs:
            if es <= 1e-12:
                continue
            eta = self.cfg.eta
            claim_qty = (1.0 - eta) * es
            # Algorithm-specific rules
            algo = self.cfg.algo
            if algo == "S_EB":
                # Disabled SoC guard: always allow creating claims under S_EB
                pass
            if algo == "AI_P2P_EB":
                features = {
                    "t": t,
                    "eta": eta,
                    "tau": self.cfg.tau,
                    "u_t": self.util_prices[t],
                    "fit_t": self.fit_prices[t],
                    "offer_qty": es,
                    "req_qty": es,
                    "peer_balance": self.peers[buyer_id].balance_remaining(),
                    "recent_R_over_O": (sum(r.qty_kWh for r in requests) + 1e-9) / (sum(o.qty_kWh for o in offers) + 1e-9),
                    "battery_cap": self.caps.get(seller_id, 0.0),
                }
                e_hat = max(0.0, float(self.predictor(features)))
                es_cap = e_hat / max(1e-12, (1.0 - eta))
                if es > es_cap:
                    es = es_cap
                    claim_qty = (1.0 - eta) * es

            # Capacity check on consumer battery for storing (1-eta)*Es
            if (1.0 - eta) > 1e-12:
                free_cap = max(0.0, self.caps.get(buyer_id, 0.0) - self.peers[buyer_id].soc_kWh)
                es_cap_by_cap = free_cap / max(1e-12, (1.0 - eta))
                if es > es_cap_by_cap:
                    es = es_cap_by_cap
                    claim_qty = (1.0 - eta) * es
                if es <= 1e-12:
                    continue

            # Receiver consumes immediately η*Es — non-monetary; remove from request
            # Accounting already ensured these are non-affordable requests.
            expiry = t + self.cfg.tau
            # Transfer claim ownership depending on algorithm
            owner_id = seller_id  # default P2P_EB & AI_P2P_EB
            if self.cfg.algo == "C_EB":
                owner_id = -1  # coordinator identifier
                self.coordinator_payouts += es * p_t
                self.peers[seller_id].earned_currency += es * p_t

            # Reduce seller's available offer
            for off in offers:
                if off.seller_id == seller_id and off.qty_kWh > 0:
                    take = min(off.qty_kWh, es)
                    off.qty_kWh -= take
                    es -= take
                    if es <= 1e-12:
                        break

            # Immediate consumption reduces buyer's request by eta*Es
            immediate = eta * (claim_qty / max(1e-12, (1.0 - eta))) if (1.0 - eta) > 1e-12 else eta * es
            for r in requests:
                if r.buyer_id == buyer_id and r.qty_kWh > 1e-12:
                    reduce = min(r.qty_kWh, immediate)
                    r.qty_kWh -= reduce
                    break

            # Store claim energy physically on consumer battery (reserved)
            if claim_qty > 1e-12:
                # Increase consumer SoC by claim amount, capped earlier
                self.peers[buyer_id].soc_kWh += claim_qty
                self.claims.append(BarterClaim(owner_id, stored_on_id=buyer_id, qty_kWh=claim_qty, expiry_t=expiry, created_t=t))
                events.append(BarterEvent(t, seller_id, buyer_id, Es_kWh=claim_qty / max(1e-12, (1.0 - eta)), eta=eta, claim_kWh=claim_qty, expiry_t=expiry))

        rem_offers = [o for o in offers if o.qty_kWh > 1e-9]
        rem_requests = [r for r in requests if r.qty_kWh > 1e-9]
        return events, rem_offers, rem_requests

    def _grid_and_export(self, t: int, p_t: float, fit_t: float, offers: List[Offer], requests: List[Request]) -> None:
        # Grid fallback for remaining requests
        for req in requests:
            if req.qty_kWh <= 1e-12:
                continue
            # pay grid price, add to paid_currency; energy comes from grid
            self.grid_import_tot[req.buyer_id] += req.qty_kWh
            self.peers[req.buyer_id].paid_currency += req.qty_kWh * self.util_prices[t]

        # Export leftovers to grid at FiT
        for off in offers:
            if off.qty_kWh <= 1e-12:
                continue
            self.peers[off.seller_id].earned_currency += off.qty_kWh * fit_t

    # ─────────────────────────────────────────────────────────────────────
    # Public API
    # ─────────────────────────────────────────────────────────────────────
    def run(self) -> RunOutputs:
        T = min(len(self.loads), len(self.solars), len(self.util_prices), len(self.fit_prices))
        # Precompute average FiT/u ratio for profitability guardrail
        fit_over_u = [self.fit_prices[t] / max(1e-12, self.util_prices[t]) for t in range(T)]
        avg_ratio = sum(fit_over_u) / max(1, len(fit_over_u))
        if profitability_warning_eta(self.cfg.eta, avg_ratio) and self.cfg.verbose:
            # soft warning — logging is handled by caller
            pass

        for t in range(T):
            # Upfront: process expiries before any consumption/building requests
            self._handle_expiries_upfront(t)

            offers, requests = self._build_physical_balances(t)
            # record SoC snapshot for all peers
            for pid, peer in self.peers.items():
                self.soc_series.append((t, pid, peer.soc_kWh))
            p_t = self._compute_price(t, offers, requests)
            fit_t = self.fit_prices[t]

            # Priority: old claims first
            claim_trades, requests = self._sell_existing_claims(t, p_t, requests)
            self.trades.extend(claim_trades)

            # Fresh P2P trading (monetary)
            trades, offers, requests = self._p2p_trading(t, p_t, offers, requests)
            self.trades.extend(trades)

            # Barter (non-monetary)
            if self.cfg.algo in ("C_EB", "P2P_EB", "S_EB", "AI_P2P_EB"):
                events, offers, requests = self._barter_phase(t, p_t, offers, requests)
                self.barters.extend(events)

            # Grid fallback and export
            self._grid_and_export(t, p_t, fit_t, offers, requests)

            # End-of-hour maintenance: bounds + drop depleted (expiry handled upfront)
            for pid, peer in self.peers.items():
                peer.soc_kWh = max(0.0, min(peer.soc_kWh, self.caps.get(pid, 0.0)))
            self.claims = [c for c in self.claims if c.qty_kWh > 1e-9]

        paid_earned = {pid: (p.paid_currency, p.earned_currency) for pid, p in self.peers.items()}
        return RunOutputs(
            grid_import_by_house=self.grid_import_tot,
            trades_all=self.trades,
            barters_all=self.barters,
            open_claims=self.claims,
            paid_earned=paid_earned,
            coordinator_payouts=self.coordinator_payouts,
            price_series=self.price_series,
            soc_series=self.soc_series,
            claim_returns=self.claim_returns,
                claim_expiries=self.claim_expiries,
        )


