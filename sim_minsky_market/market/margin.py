"""Leverage, debt interest, margin calls, forced liquidation, and defaults."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..agents.base_agent import BaseAgent
    from .config import LeverageConfig


@dataclass
class MarginCallResult:
    agent_id: int
    pre_leverage: float
    post_leverage: float
    shares_liquidated: float
    defaulted: bool


def apply_interest(agent: "BaseAgent", borrowing_rate: float) -> None:
    """Accrue interest on agent debt for one time step."""
    agent.debt *= 1.0 + borrowing_rate


def check_and_liquidate(
    agent: "BaseAgent",
    price: float,
    config: "LeverageConfig",
) -> MarginCallResult | None:
    """Check whether agent exceeds max leverage. If so, force-sell assets.

    Returns a MarginCallResult if a margin call fired, otherwise None.
    The caller is responsible for collecting the resulting sell demand.
    """
    leverage = agent.leverage(price)
    if leverage <= config.max_leverage:
        return None

    pre_leverage = leverage
    shares_to_sell = _shares_needed_to_restore(agent, price, config)

    # Clamp: can't sell more than held (short-selling not modelled here)
    shares_to_sell = min(shares_to_sell, max(agent.shares, 0.0))

    proceeds = shares_to_sell * price
    agent.shares -= shares_to_sell
    agent.cash += proceeds
    agent.debt = max(agent.debt - proceeds, 0.0)

    defaulted = agent.wealth(price) <= 0.0
    if defaulted:
        agent.is_active = False

    return MarginCallResult(
        agent_id=agent.agent_id,
        pre_leverage=pre_leverage,
        post_leverage=agent.leverage(price),
        shares_liquidated=shares_to_sell,
        defaulted=defaulted,
    )


def _shares_needed_to_restore(
    agent: "BaseAgent",
    price: float,
    config: "LeverageConfig",
) -> float:
    """Calculate how many shares to sell to bring leverage to config.max_leverage.

    We want: |price * (shares - x)| / max(wealth_after, eps) == L_max
    Solving for x (assuming wealth_after ≈ wealth_before + proceeds - debt reduction):
        x = liquidation_fraction * shares  (simple heuristic that converges fast)
    """
    # Iterative heuristic: sell a fraction, check if enough
    shares = max(agent.shares, 0.0)
    fraction = config.liquidation_fraction
    # We sell enough shares so that leverage drops back to max_leverage
    # Closed-form approximation:
    #   leverage = price * shares / wealth
    #   after selling dx shares: leverage' = price*(shares-dx) / (wealth + dx*price - dx*price) → unchanged
    # Because selling reduces both numerator and denominator equally, we must repay debt.
    # Repay debt = proceeds = dx * price, so:
    #   leverage' = price*(shares-dx) / (wealth + dx*price - dx*price) is wrong.
    # Correct: wealth = cash + price*shares - debt; selling dx → cash += dx*p, shares -= dx, debt -= dx*p
    #   wealth' = cash + dx*p + price*(shares-dx) - (debt - dx*p) = cash + price*shares - debt + dx*p = wealth + dx*p (wrong)
    # Actually: proceeds pay down debt first:
    #   cash' = cash (proceeds go to debt repayment)
    #   debt' = debt - dx*price (if debt > 0)
    #   shares' = shares - dx
    #   wealth' = cash + price*(shares-dx) - (debt - dx*price) = cash + price*shares - debt = wealth (unchanged)
    # So wealth is invariant when proceeds repay debt exactly. Leverage = price*shares'/wealth
    # L_target = price*(shares - dx)/wealth  =>  dx = shares - L_target*wealth/price
    wealth = agent.wealth(price)
    if wealth <= 1e-8:
        return shares  # bankrupt, liquidate everything

    dx = shares - config.max_leverage * wealth / price
    return max(dx, shares * fraction)  # sell at least the configured fraction
