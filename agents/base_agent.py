"""Base agent: balance sheet, wealth, leverage."""

from __future__ import annotations

from dataclasses import dataclass, field


_AGENT_COUNTER: int = 0


def _next_id() -> int:
    global _AGENT_COUNTER
    _AGENT_COUNTER += 1
    return _AGENT_COUNTER


def reset_id_counter() -> None:
    global _AGENT_COUNTER
    _AGENT_COUNTER = 0


@dataclass
class BaseAgent:
    """Agent balance sheet and portfolio accounting.

    All subclasses should call super().__post_init__() if they define __post_init__.
    """

    agent_type: str = field(default="base")
    initial_cash: float = field(default=1000.0)

    # Runtime state (not set at construction via dataclass field)
    agent_id: int = field(init=False)
    cash: float = field(init=False)
    shares: float = field(init=False)   # risky asset holdings
    debt: float = field(init=False)
    is_active: bool = field(init=False, default=True)

    def __post_init__(self) -> None:
        self.agent_id = _next_id()
        self.cash = self.initial_cash
        self.shares = 0.0
        self.debt = 0.0
        self.is_active = True

    # ------------------------------------------------------------------
    # Balance-sheet quantities
    # ------------------------------------------------------------------

    def asset_exposure(self, price: float) -> float:
        return price * self.shares

    def wealth(self, price: float) -> float:
        return self.cash + price * self.shares - self.debt

    def leverage(self, price: float) -> float:
        w = self.wealth(price)
        if w <= 1e-8:
            return float("inf")
        return abs(self.asset_exposure(price)) / w

    # ------------------------------------------------------------------
    # Action interface (override in subclasses)
    # ------------------------------------------------------------------

    def decide(
        self,
        price: float,
        fundamental: float,
        rolling_volatility: float,
        rolling_momentum: float,
        aggregate_leverage: float,
        rng: "np.random.Generator",  # type: ignore[name-defined]
    ) -> tuple[float, float]:
        """Return (buy_demand, sell_demand) as positive floats.

        buy_demand  — units the agent wants to buy
        sell_demand — units the agent wants to sell (positive value)
        """
        return 0.0, 0.0

    def execute_buy(self, shares: float, price: float) -> None:
        """Buy shares. Borrow cash if insufficient."""
        cost = shares * price
        if cost <= self.cash:
            self.cash -= cost
        else:
            borrowed = cost - self.cash
            self.cash = 0.0
            self.debt += borrowed
        self.shares += shares

    def execute_sell(self, shares: float, price: float) -> None:
        """Sell shares. Use proceeds to repay debt first."""
        shares = min(shares, max(self.shares, 0.0))
        proceeds = shares * price
        self.shares -= shares
        if self.debt > 0:
            repay = min(self.debt, proceeds)
            self.debt -= repay
            self.cash += proceeds - repay
        else:
            self.cash += proceeds

    def reset(self) -> None:
        self.cash = self.initial_cash
        self.shares = 0.0
        self.debt = 0.0
        self.is_active = True
        self.agent_id = _next_id()
