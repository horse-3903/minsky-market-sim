"""Optional market maker: provides liquidity around fundamental value."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .base_agent import BaseAgent
from ..market.config import AgentConfig


@dataclass
class MarketMaker(BaseAgent):
    """Simple market maker.

    Provides a small stabilising order on each side proportional to how far
    price deviates from fundamental value.  Keeps inventory near zero.
    """

    agent_type: str = field(default="market_maker")
    spread_sensitivity: float = field(default=0.2)
    max_position: float = field(default=30.0)

    def decide(
        self,
        price: float,
        fundamental: float,
        rolling_volatility: float,
        rolling_momentum: float,
        aggregate_leverage: float,
        rng: np.random.Generator,
    ) -> tuple[float, float]:
        # If price is above fundamental, lean towards selling; below → buying
        signal = (fundamental - price) / max(fundamental, 1e-8)
        desired_shares = self.spread_sensitivity * signal * self.max_position
        delta = desired_shares - self.shares

        if delta > 0.0:
            return delta, 0.0
        else:
            sell = min(abs(delta), max(self.shares, 0.0))
            return 0.0, sell


def make_market_makers(config: AgentConfig) -> list[MarketMaker]:
    return [
        MarketMaker(initial_cash=config.initial_cash * 2)
        for _ in range(config.n_market_maker)
    ]
