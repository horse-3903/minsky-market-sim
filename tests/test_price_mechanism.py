"""Tests for price-impact mechanism."""

import numpy as np
import pytest

from market.config import PriceConfig
from market.price_mechanism import PriceMechanism


def _pm(seed: int = 0, noise: float = 0.0) -> PriceMechanism:
    cfg = PriceConfig(initial_price=100.0, price_impact=0.01, noise_volatility=noise)
    return PriceMechanism(cfg, np.random.default_rng(seed))


def test_zero_demand_price_stable():
    pm = _pm(noise=0.0)
    p0 = pm.price
    p1 = pm.step(0.0)
    # With zero noise and zero demand, price should stay exactly at p0
    assert abs(p1 - p0) < 1e-10


def test_positive_demand_raises_price():
    pm = _pm(noise=0.0)
    p0 = pm.price
    p1 = pm.step(10.0)
    assert p1 > p0


def test_negative_demand_lowers_price():
    pm = _pm(noise=0.0)
    p0 = pm.price
    p1 = pm.step(-10.0)
    assert p1 < p0


def test_price_positive():
    """Price should never go negative even with large sell pressure."""
    pm = _pm(noise=0.0)
    for _ in range(100):
        pm.step(-1000.0)
    assert pm.price > 0


def test_history_grows():
    pm = _pm()
    for _ in range(5):
        pm.step(0.0)
    assert len(pm.history) == 6  # initial + 5 steps


def test_reset():
    pm = _pm(noise=0.0)
    pm.step(50.0)
    pm.reset()
    assert pm.price == 100.0
    assert len(pm.history) == 1
