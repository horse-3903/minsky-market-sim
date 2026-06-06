"""Tests for leverage, margin calls, forced liquidation, and defaults."""

import numpy as np
import pytest

from agents.base_agent import BaseAgent, reset_id_counter
from market.config import LeverageConfig
from market.margin import apply_interest, check_and_liquidate


def _lev_config(max_lev: float = 3.0) -> LeverageConfig:
    return LeverageConfig(
        max_leverage=max_lev,
        borrowing_rate=0.001,
        margin_call_threshold=max_lev,
        liquidation_fraction=0.5,
    )


def _leveraged_agent(price: float = 100.0, leverage_target: float = 4.0) -> BaseAgent:
    """Create an agent with leverage > max_leverage for testing."""
    reset_id_counter()
    a = BaseAgent(initial_cash=200.0)
    # Buy shares on margin to hit leverage_target
    # wealth ≈ 200; exposure = leverage_target * 200 = 800 → shares = 8
    shares = (leverage_target * a.initial_cash) / price
    cost = shares * price
    borrow = max(cost - a.cash, 0.0)
    a.shares = shares
    a.cash = max(a.cash - cost, 0.0)
    a.debt = borrow
    return a


def test_interest_increases_debt():
    reset_id_counter()
    a = BaseAgent(initial_cash=500.0)
    a.debt = 200.0
    apply_interest(a, borrowing_rate=0.01)
    assert a.debt == pytest.approx(202.0)


def test_no_margin_call_when_leverage_ok():
    reset_id_counter()
    a = BaseAgent(initial_cash=1000.0)
    a.shares = 5.0  # leverage very low
    cfg = _lev_config(max_lev=3.0)
    result = check_and_liquidate(a, price=100.0, config=cfg)
    assert result is None


def test_margin_call_fires_when_over_max_leverage():
    a = _leveraged_agent(price=100.0, leverage_target=5.0)
    cfg = _lev_config(max_lev=3.0)
    initial_shares = a.shares
    result = check_and_liquidate(a, price=100.0, config=cfg)
    assert result is not None
    assert result.shares_liquidated > 0
    assert a.shares < initial_shares


def test_post_liquidation_leverage_reduced():
    a = _leveraged_agent(price=100.0, leverage_target=5.0)
    cfg = _lev_config(max_lev=3.0)
    check_and_liquidate(a, price=100.0, config=cfg)
    new_lev = a.leverage(100.0)
    # After liquidation leverage should be lower (may not be ≤ 3 in one pass, but must drop)
    assert new_lev < 5.0


def test_forced_sell_increases_sell_demand():
    """Confirm that liquidated shares are returned in the result."""
    a = _leveraged_agent(price=100.0, leverage_target=6.0)
    cfg = _lev_config(max_lev=3.0)
    result = check_and_liquidate(a, price=100.0, config=cfg)
    assert result is not None
    assert result.shares_liquidated > 0


def test_default_when_wealth_negative():
    reset_id_counter()
    a = BaseAgent(initial_cash=0.0)
    a.shares = 1.0
    a.debt = 200.0   # wealth = 0 + 100 - 200 = -100 at price=100
    cfg = _lev_config(max_lev=1.0)
    result = check_and_liquidate(a, price=100.0, config=cfg)
    assert result is not None
    assert result.defaulted
    assert not a.is_active
