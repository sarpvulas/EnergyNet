from __future__ import annotations

from energynet.barter_sim.matching import fcfs_match_affordable
from energynet.barter_sim.models import Offer, Request


def test_fcfs_respects_balances():
    offers = [Offer(1, 10.0)]
    requests = [Request(2, 10.0)]
    price = 2.0
    balances = {2: 10.0}  # can afford 5 kWh
    trades, rem_offers, rem_requests = fcfs_match_affordable(offers, requests, price, balances)

    assert trades == [(1, 2, 5.0)]
    assert abs(rem_offers[0].qty_kWh - 5.0) < 1e-9
    assert abs(rem_requests[0].qty_kWh - 5.0) < 1e-9


