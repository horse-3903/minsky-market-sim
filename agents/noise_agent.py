"""Noise trader: trades randomly."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .base_agent import BaseAgent
from market.config import AgentConfig


@dataclass
class NoiseAgent(BaseAgent):
    """Noise trader. Random buy, sell, or hold each step."""

    agent_type: str = field(default="noise")
    max_trade: float = field(default=10.0)

    def decide(
        self,
        price: float,
        fundamental: float,
        rolling_volatility: float,
        rolling_momentum: float,
        aggregate_leverage: float,
        rng: np.random.Generator,
    ) -> tuple[float, float]:
        action = rng.choice(["buy", "sell", "hold"])
        if action == "buy":
            size = rng.uniform(0, self.max_trade)
            return size, 0.0
        elif action == "sell":
            size = min(rng.uniform(0, self.max_trade), max(self.shares, 0.0))
            return 0.0, size
        return 0.0, 0.0


def make_noise_agents(config: AgentConfig) -> list[NoiseAgent]:
    return [
        NoiseAgent(
            initial_cash=config.initial_cash,
            max_trade=config.noise_max_trade,
        )
        for _ in range(config.n_noise)
    ]
