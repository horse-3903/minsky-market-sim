"""RL agent stub — full implementation added in Phase 4."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .base_agent import BaseAgent


@dataclass
class RLAgent(BaseAgent):
    """Placeholder RL trader. Phase 4 replaces decide() with a learned policy."""

    agent_type: str = field(default="rl")

    def decide(
        self,
        price: float,
        fundamental: float,
        rolling_volatility: float,
        rolling_momentum: float,
        aggregate_leverage: float,
        rng: np.random.Generator,
    ) -> tuple[float, float]:
        # Hold by default until a real policy is attached
        return 0.0, 0.0
