"""Momentum trader: follows price trends."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .base_agent import BaseAgent
from market.config import AgentConfig


@dataclass
class MomentumAgent(BaseAgent):
    """Momentum trader.

    Signal: m = (P_t - P_{t-k}) / P_{t-k}  (rolling momentum passed in)
    Demand = sensitivity * m * max_position
    """

    agent_type: str = field(default="momentum")
    sensitivity: float = field(default=0.3)
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
        desired_shares = self.sensitivity * rolling_momentum * self.max_position
        delta = desired_shares - self.shares

        if delta > 0.0:
            return delta, 0.0
        else:
            sell = min(abs(delta), max(self.shares, 0.0))
            return 0.0, sell


def make_momentum_agents(config: AgentConfig) -> list[MomentumAgent]:
    return [
        MomentumAgent(
            initial_cash=config.initial_cash,
            sensitivity=config.momentum_sensitivity,
            max_position=config.momentum_max_position,
        )
        for _ in range(config.n_momentum)
    ]
