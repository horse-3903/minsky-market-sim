"""Price-impact mechanism: maps net demand to a new market price."""

from __future__ import annotations

import numpy as np

from .config import PriceConfig


class PriceMechanism:
    """Log-linear price-impact model.

    P_{t+1} = P_t * exp(alpha * D_t / liquidity + eta_t)

    where D_t = buy_demand - sell_demand (normalised units),
    alpha is price impact, and eta_t is iid noise.
    """

    def __init__(self, config: PriceConfig, rng: np.random.Generator) -> None:
        self.config = config
        self.rng = rng
        self.price: float = config.initial_price
        self._history: list[float] = [config.initial_price]

    def step(self, net_demand: float) -> float:
        """Update price given net demand. Returns new price."""
        impact = self.config.price_impact * net_demand / max(self.config.liquidity, 1e-8)
        noise = self.rng.normal(0.0, self.config.noise_volatility)
        # Clip exponent to prevent overflow; ±1.5 caps single-step move at ~4.5×/0.22×
        exponent = float(np.clip(impact + noise, -1.5, 1.5))
        self.price = self.price * np.exp(exponent)
        self.price = max(self.price, 1e-4)
        self._history.append(self.price)
        return self.price

    @property
    def history(self) -> list[float]:
        return list(self._history)

    def reset(self) -> float:
        self.price = self.config.initial_price
        self._history = [self.config.initial_price]
        return self.price
