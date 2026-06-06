"""Fundamental trader: buys when undervalued, sells when overvalued."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .base_agent import BaseAgent
from ..market.config import AgentConfig


@dataclass
class FundamentalAgent(BaseAgent):
    """Fundamental trader.

    Signal: s = (F - P) / F
    Demand = sensitivity * s * max_position (positive → buy, negative → sell)
    """

    agent_type: str = field(default="fundamental")
    sensitivity: float = field(default=0.5)
    max_position: float = field(default=50.0)

    def decide(
        self,
        price: float,
        fundamental: float,
        rolling_volatility: float,
        rolling_momentum: float,
        aggregate_leverage: float,
        rng: np.random.Generator,
    ) -> tuple[float, float]:
        signal = (fundamental - price) / max(fundamental, 1e-8)
        desired_shares = self.sensitivity * signal * self.max_position
        current_shares = self.shares

        delta = desired_shares - current_shares

        if delta > 0.0:
            return delta, 0.0
        else:
            sell = min(abs(delta), max(current_shares, 0.0))
            return 0.0, sell


def make_fundamental_agents(config: AgentConfig) -> list[FundamentalAgent]:
    return [
        FundamentalAgent(
            initial_cash=config.initial_cash,
            sensitivity=config.fundamental_sensitivity,
            max_position=config.fundamental_max_position,
        )
        for _ in range(config.n_fundamental)
    ]
