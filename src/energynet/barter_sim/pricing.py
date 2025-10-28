from __future__ import annotations

from collections import deque
from typing import Deque, List


class PriceSmoother:
    def __init__(self, window: int = 3) -> None:
        self.window: int = max(1, int(window))
        self._hist: Deque[float] = deque(maxlen=self.window)

    def average(self) -> float:
        if not self._hist:
            return 0.0
        return sum(self._hist) / len(self._hist)

    def append(self, price: float) -> None:
        self._hist.append(float(price))

    def values(self) -> List[float]:
        return list(self._hist)


def compute_price(u_t: float, fit_t: float, R_t: float, O_t: float, avg_recent: float | None) -> float:
    """Compute p_t with given rule and guardrails.

    p_t = max(FiT_t, min(u_t, (R_t/O_t) * average(last 3 prices)))
    Guard against O_t = 0 and cold start (use mid-point between u_t and FiT_t).
    """
    u_t = float(u_t)
    fit_t = float(fit_t)

    if O_t <= 0.0 or avg_recent is None or avg_recent <= 0.0:
        return 0.5 * (u_t + fit_t)

    ratio = max(0.0, R_t / O_t)
    raw = ratio * float(avg_recent)
    return max(fit_t, min(u_t, raw))


def profitability_warning_eta(eta: float, fit_over_u_avg: float) -> bool:
    """Return True if η ≥ 1 − (FiT/u)avg, which collapses resale window."""
    threshold = 1.0 - fit_over_u_avg
    return eta >= threshold


