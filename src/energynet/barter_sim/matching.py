from __future__ import annotations

from typing import Dict, List, Tuple

from .models import Offer, Request


def fcfs_match_affordable(
    offers: List[Offer],
    requests: List[Request],
    price: float,
    balances: Dict[int, float],
) -> Tuple[List[Tuple[int, int, float]], List[Offer], List[Request]]:
    """
    FCFS monetary matching. Respects buyer balances.

    Returns (trades, remaining_offers, remaining_requests)
    where trades is a list of (seller_id, buyer_id, qty_kWh).
    """
    trades: List[Tuple[int, int, float]] = []
    rem_offers = [Offer(o.seller_id, o.qty_kWh) for o in offers]
    rem_requests = [Request(r.buyer_id, r.qty_kWh) for r in requests]

    for i, req in enumerate(rem_requests):
        if req.qty_kWh <= 0:
            continue
        affordable_qty = balances.get(req.buyer_id, 0.0) / max(price, 1e-12)
        if affordable_qty <= 0:
            continue
        needed = min(req.qty_kWh, affordable_qty)
        for j, off in enumerate(rem_offers):
            if needed <= 0:
                break
            if off.qty_kWh <= 0:
                continue
            qty = min(off.qty_kWh, needed)
            if qty <= 0:
                continue

            trades.append((off.seller_id, req.buyer_id, qty))
            off.qty_kWh -= qty
            req.qty_kWh -= qty
            needed -= qty

    rem_offers = [o for o in rem_offers if o.qty_kWh > 1e-9]
    rem_requests = [r for r in rem_requests if r.qty_kWh > 1e-9]
    return trades, rem_offers, rem_requests


def fcfs_barter_pairs(
    offers: List[Offer],
    requests: List[Request],
) -> List[Tuple[int, int, float]]:
    """FCFS pairing for bartering on non-affordable requests.

    Returns list of (seller_id, buyer_id, Es_kWh) with Es = min(offer, request).
    """
    pairs: List[Tuple[int, int, float]] = []
    rem_offers = [Offer(o.seller_id, o.qty_kWh) for o in offers]
    rem_requests = [Request(r.buyer_id, r.qty_kWh) for r in requests]

    for req in rem_requests:
        need = req.qty_kWh
        if need <= 0:
            continue
        for off in rem_offers:
            if need <= 0:
                break
            if off.qty_kWh <= 0:
                continue
            es = min(off.qty_kWh, need)
            if es <= 0:
                continue
            pairs.append((off.seller_id, req.buyer_id, es))
            off.qty_kWh -= es
            need -= es

    return pairs


