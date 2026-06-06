"""Tests for agent balance-sheet accounting and decision logic."""

import numpy as np
import pytest

from agents.base_agent import BaseAgent, reset_id_counter
from agents.fundamental_agent import FundamentalAgent
from agents.momentum_agent import MomentumAgent
from agents.noise_agent import NoiseAgent


def rng():
    return np.random.default_rng(0)


# ---------------------------------------------------------------------------
# Balance-sheet tests
# ---------------------------------------------------------------------------

def test_wealth_no_shares():
    reset_id_counter()
    a = BaseAgent(initial_cash=1000.0)
    assert a.wealth(price=100.0) == 1000.0


def test_wealth_with_shares():
    reset_id_counter()
    a = BaseAgent(initial_cash=500.0)
    a.shares = 5.0
    assert a.wealth(price=100.0) == pytest.approx(1000.0)


def test_wealth_with_debt():
    reset_id_counter()
    a = BaseAgent(initial_cash=0.0)
    a.shares = 10.0
    a.debt = 500.0
    assert a.wealth(price=100.0) == pytest.approx(500.0)


def test_leverage_no_position():
    reset_id_counter()
    a = BaseAgent(initial_cash=1000.0)
    assert a.leverage(price=100.0) == pytest.approx(0.0)


def test_leverage_calculation():
    reset_id_counter()
    a = BaseAgent(initial_cash=500.0)
    a.shares = 10.0  # exposure = 1000; wealth = 500 + 1000 - 0 = 1500; lev = 1000/1500
    assert a.leverage(price=100.0) == pytest.approx(1000.0 / 1500.0)


def test_execute_buy_uses_cash_first():
    reset_id_counter()
    a = BaseAgent(initial_cash=500.0)
    a.execute_buy(3.0, price=100.0)
    assert a.shares == pytest.approx(3.0)
    assert a.cash == pytest.approx(200.0)
    assert a.debt == pytest.approx(0.0)


def test_execute_buy_borrows_when_insufficient_cash():
    reset_id_counter()
    a = BaseAgent(initial_cash=100.0)
    a.execute_buy(5.0, price=100.0)   # cost = 500; borrow 400
    assert a.shares == pytest.approx(5.0)
    assert a.cash == pytest.approx(0.0)
    assert a.debt == pytest.approx(400.0)


def test_execute_sell_repays_debt():
    reset_id_counter()
    a = BaseAgent(initial_cash=0.0)
    a.shares = 10.0
    a.debt = 500.0
    a.execute_sell(3.0, price=100.0)   # proceeds = 300 → repay debt
    assert a.shares == pytest.approx(7.0)
    assert a.debt == pytest.approx(200.0)
    assert a.cash == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# Decision logic tests
# ---------------------------------------------------------------------------

def test_fundamental_agent_buys_when_undervalued():
    reset_id_counter()
    a = FundamentalAgent(initial_cash=1000.0, sensitivity=0.5, max_position=50.0)
    bd, sd = a.decide(price=80.0, fundamental=100.0,
                      rolling_volatility=0.0, rolling_momentum=0.0,
                      aggregate_leverage=0.0, rng=rng())
    assert bd > 0
    assert sd == 0.0


def test_fundamental_agent_sells_when_overvalued():
    reset_id_counter()
    a = FundamentalAgent(initial_cash=1000.0)
    a.shares = 30.0
    bd, sd = a.decide(price=120.0, fundamental=100.0,
                      rolling_volatility=0.0, rolling_momentum=0.0,
                      aggregate_leverage=0.0, rng=rng())
    assert sd > 0
    assert bd == 0.0


def test_momentum_agent_buys_on_positive_momentum():
    reset_id_counter()
    a = MomentumAgent(initial_cash=1000.0, sensitivity=0.3, max_position=50.0)
    bd, sd = a.decide(price=100.0, fundamental=100.0,
                      rolling_volatility=0.0, rolling_momentum=0.1,
                      aggregate_leverage=0.0, rng=rng())
    assert bd > 0
    assert sd == 0.0


def test_momentum_agent_sells_on_negative_momentum():
    reset_id_counter()
    a = MomentumAgent(initial_cash=1000.0, sensitivity=0.3, max_position=50.0)
    a.shares = 10.0
    bd, sd = a.decide(price=100.0, fundamental=100.0,
                      rolling_volatility=0.0, rolling_momentum=-0.1,
                      aggregate_leverage=0.0, rng=rng())
    assert sd > 0
    assert bd == 0.0
