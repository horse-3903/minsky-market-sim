"""Market metrics: recording, rolling statistics, and Minsky classification."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from ..agents.base_agent import BaseAgent
    from .config import MinskyThresholds, SimulationConfig


@dataclass
class StepMetrics:
    step: int
    price: float
    fundamental: float
    return_: float
    mispricing: float         # (P - F) / F
    abs_mispricing: float     # |P - F| / F
    rolling_volatility: float
    rolling_momentum: float
    buy_demand: float
    sell_demand: float
    net_demand: float
    volume: float
    aggregate_leverage: float
    avg_leverage: float
    max_leverage: float
    n_margin_calls: int
    n_defaults: int
    avg_wealth: float
    wealth_fundamental: float
    wealth_momentum: float
    wealth_noise: float
    wealth_rl: float
    pct_hedge: float
    pct_speculative: float
    pct_ponzi: float
    bubble: bool
    crash: bool
    drawdown: float
    gini: float


class MetricsRecorder:
    """Maintains rolling windows and accumulates per-step metrics."""

    def __init__(
        self,
        config: "SimulationConfig",
        volatility_window: int = 20,
        momentum_window: int = 10,
    ) -> None:
        self.config = config
        self.vol_window = volatility_window
        self.mom_window = momentum_window
        self._returns: deque[float] = deque(maxlen=volatility_window)
        self._prices: deque[float] = deque(maxlen=momentum_window + 1)
        self._price_history: list[float] = []
        self.records: list[StepMetrics] = []

    # ------------------------------------------------------------------
    # Rolling statistics
    # ------------------------------------------------------------------

    def rolling_volatility(self) -> float:
        if len(self._returns) < 2:
            return 0.0
        return float(np.std(self._returns))

    def rolling_momentum(self) -> float:
        """Return over last mom_window steps."""
        if len(self._prices) < 2:
            return 0.0
        oldest = self._prices[0]
        newest = self._prices[-1]
        if oldest <= 0:
            return 0.0
        return (newest - oldest) / oldest

    def current_return(self) -> float:
        if len(self._returns) == 0:
            return 0.0
        return self._returns[-1]

    # ------------------------------------------------------------------
    # Main record call
    # ------------------------------------------------------------------

    def record(
        self,
        step: int,
        price: float,
        fundamental: float,
        buy_demand: float,
        sell_demand: float,
        n_margin_calls: int,
        n_defaults: int,
        agents: list["BaseAgent"],
        minsky_thresholds: "MinskyThresholds",
    ) -> StepMetrics:
        # Update rolling buffers
        ret = (price / self._price_history[-1] - 1.0) if self._price_history else 0.0
        self._returns.append(ret)
        self._prices.append(price)
        self._price_history.append(price)

        net_demand = buy_demand - sell_demand
        vol = self.rolling_volatility()
        mom = self.rolling_momentum()
        mispricing = (price - fundamental) / max(fundamental, 1e-8)
        abs_misp = abs(mispricing)

        # Drawdown from peak
        peak = max(self._price_history) if self._price_history else price
        drawdown = (price - peak) / peak

        # Crash detection
        crash_window = self.config.crash_window
        if len(self._price_history) >= crash_window + 1:
            past_price = self._price_history[-(crash_window + 1)]
            price_change = (price - past_price) / max(past_price, 1e-8)
            crash = price_change < self.config.crash_threshold
        else:
            crash = False

        bubble = mispricing > self.config.bubble_threshold

        # Agent-level aggregates
        active = [a for a in agents if a.is_active]
        leverages = [a.leverage(price) for a in active]
        wealths = [a.wealth(price) for a in active]

        agg_lev = float(np.sum(leverages)) if leverages else 0.0
        avg_lev = float(np.mean(leverages)) if leverages else 0.0
        max_lev = float(np.max(leverages)) if leverages else 0.0
        avg_wealth = float(np.mean(wealths)) if wealths else 0.0

        # Wealth by type
        def type_wealth(atype: str) -> float:
            w = [a.wealth(price) for a in active if a.agent_type == atype]
            return float(np.mean(w)) if w else 0.0

        # Minsky classification
        thr = minsky_thresholds
        hedge = sum(1 for lv in leverages if lv < thr.hedge_max)
        spec = sum(1 for lv in leverages if thr.hedge_max <= lv < thr.speculative_max)
        ponzi = sum(1 for lv in leverages if lv >= thr.speculative_max)
        n_active = len(active) or 1
        pct_h = hedge / n_active
        pct_s = spec / n_active
        pct_p = ponzi / n_active

        # Gini coefficient
        gini = _gini(np.array(wealths)) if len(wealths) > 1 else 0.0

        m = StepMetrics(
            step=step,
            price=price,
            fundamental=fundamental,
            return_=ret,
            mispricing=mispricing,
            abs_mispricing=abs_misp,
            rolling_volatility=vol,
            rolling_momentum=mom,
            buy_demand=buy_demand,
            sell_demand=sell_demand,
            net_demand=net_demand,
            volume=buy_demand + abs(sell_demand),
            aggregate_leverage=agg_lev,
            avg_leverage=avg_lev,
            max_leverage=max_lev,
            n_margin_calls=n_margin_calls,
            n_defaults=n_defaults,
            avg_wealth=avg_wealth,
            wealth_fundamental=type_wealth("fundamental"),
            wealth_momentum=type_wealth("momentum"),
            wealth_noise=type_wealth("noise"),
            wealth_rl=type_wealth("rl"),
            pct_hedge=pct_h,
            pct_speculative=pct_s,
            pct_ponzi=pct_p,
            bubble=bubble,
            crash=crash,
            drawdown=drawdown,
            gini=gini,
        )
        self.records.append(m)
        return m

    def to_dataframe(self) -> pd.DataFrame:
        rows = [vars(r) for r in self.records]
        return pd.DataFrame(rows)

    def reset(self) -> None:
        self._returns.clear()
        self._prices.clear()
        self._price_history.clear()
        self.records.clear()


def _gini(arr: np.ndarray) -> float:
    """Gini coefficient over non-negative wealth values."""
    arr = np.maximum(arr, 0.0)
    if arr.sum() == 0:
        return 0.0
    arr = np.sort(arr)
    n = len(arr)
    idx = np.arange(1, n + 1)
    return float((2 * (idx * arr).sum()) / (n * arr.sum()) - (n + 1) / n)
