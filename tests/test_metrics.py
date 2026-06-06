"""Tests for metrics recording and Gini calculation."""

import numpy as np
import pytest

from market.config import MinskyThresholds, SimulationConfig
from market.metrics import MetricsRecorder, _gini
from agents.base_agent import BaseAgent, reset_id_counter


def _make_agent(cash: float = 1000.0, shares: float = 0.0, debt: float = 0.0) -> BaseAgent:
    reset_id_counter()
    a = BaseAgent(initial_cash=cash)
    a.shares = shares
    a.debt = debt
    return a


def test_gini_equal_wealth_is_zero():
    arr = np.array([100.0, 100.0, 100.0])
    assert _gini(arr) == pytest.approx(0.0, abs=1e-10)


def test_gini_all_zero_is_zero():
    assert _gini(np.array([0.0, 0.0, 0.0])) == pytest.approx(0.0)


def test_gini_max_inequality():
    # One agent has all the wealth
    arr = np.array([0.0, 0.0, 100.0])
    g = _gini(arr)
    assert g > 0.6  # high inequality


def test_rolling_volatility_empty():
    cfg = SimulationConfig()
    rec = MetricsRecorder(cfg)
    assert rec.rolling_volatility() == 0.0


def test_metrics_record_increments():
    cfg = SimulationConfig(n_steps=5)
    rec = MetricsRecorder(cfg)
    rec._price_history.append(100.0)  # seed initial price

    agents = [_make_agent(cash=1000.0) for _ in range(3)]
    thr = MinskyThresholds()

    for step in range(5):
        m = rec.record(
            step=step,
            price=100.0 + step,
            fundamental=100.0,
            buy_demand=5.0,
            sell_demand=3.0,
            n_margin_calls=0,
            n_defaults=0,
            agents=agents,
            minsky_thresholds=thr,
        )

    assert len(rec.records) == 5
    df = rec.to_dataframe()
    assert len(df) == 5
    assert "price" in df.columns


def test_minsky_classification_all_hedge():
    cfg = SimulationConfig()
    rec = MetricsRecorder(cfg)
    rec._price_history.append(100.0)

    # All agents with zero leverage → all hedge
    agents = [_make_agent(cash=1000.0) for _ in range(4)]
    m = rec.record(
        step=0,
        price=100.0,
        fundamental=100.0,
        buy_demand=0.0,
        sell_demand=0.0,
        n_margin_calls=0,
        n_defaults=0,
        agents=agents,
        minsky_thresholds=MinskyThresholds(),
    )
    assert m.pct_hedge == pytest.approx(1.0)
    assert m.pct_speculative == pytest.approx(0.0)
    assert m.pct_ponzi == pytest.approx(0.0)
