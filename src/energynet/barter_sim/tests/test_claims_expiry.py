from __future__ import annotations

from energynet.barter_sim.models import BarterClaim


def test_claims_drop_on_expiry():
    t = 10
    claims = [
        BarterClaim(owner_id=1, stored_on_id=1, qty_kWh=2.0, expiry_t=11, created_t=5),
        BarterClaim(owner_id=1, stored_on_id=1, qty_kWh=1.0, expiry_t=9, created_t=5),
    ]

    # Drop those with expiry < t and zero-qty
    alive = [c for c in claims if c.expiry_t >= t and c.qty_kWh > 1e-9]
    assert len(alive) == 1
    assert alive[0].qty_kWh == 2.0


