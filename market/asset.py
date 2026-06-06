"""Risky asset and fundamental value process."""

from __future__ import annotations

import numpy as np

from .config import FundamentalConfig


class Asset:
    """Single risky asset with a slowly evolving fundamental value.

    F_{t+1} = F_t * (1 + mu_F + epsilon_t),  epsilon_t ~ N(0, sigma_F)
    """

    def __init__(self, config: FundamentalConfig, rng: np.random.Generator) -> None:
        self.config = config
        self.rng = rng
        self.fundamental_value: float = config.initial_value
        self._history: list[float] = [config.initial_value]

    def step(self) -> float:
        """Advance fundamental value by one time step. Returns new value."""
        eps = self.rng.normal(0.0, self.config.volatility)
        self.fundamental_value *= 1.0 + self.config.drift + eps
        self.fundamental_value = max(self.fundamental_value, 1e-4)
        self._history.append(self.fundamental_value)
        return self.fundamental_value

    @property
    def history(self) -> list[float]:
        return list(self._history)

    def reset(self) -> float:
        self.fundamental_value = self.config.initial_value
        self._history = [self.config.initial_value]
        return self.fundamental_value
