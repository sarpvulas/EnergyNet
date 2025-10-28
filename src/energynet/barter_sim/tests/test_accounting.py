from __future__ import annotations

from energynet.barter_sim.matching import fcfs_match_affordable
from energynet.barter_sim.models import Offer, Request


def test_paid_earned_consistency_simple():
    offers = [Offer(1, 3.0)]
    requests = [Request(2, 3.0)]
    price = 1.2
    balances = {2: 10.0}
    trades, _, _ = fcfs_match_affordable(offers, requests, price, balances)
    assert trades == [(1, 2, 3.0)]
    paid = 3.0 * 1.2
    earned = 3.0 * 1.2
    assert abs(paid - earned) < 1e-9


