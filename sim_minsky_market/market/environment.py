"""Main simulation loop — ties together asset, price, agents, and margin calls."""

from __future__ import annotations

from typing import Sequence

import numpy as np

from .asset import Asset
from .config import SimulationConfig
from .margin import apply_interest, check_and_liquidate, MarginCallResult
from .metrics import MetricsRecorder, StepMetrics
from .price_mechanism import PriceMechanism
from ..agents.base_agent import BaseAgent, reset_id_counter
from ..agents.fundamental_agent import make_fundamental_agents
from ..agents.market_maker import make_market_makers
from ..agents.momentum_agent import make_momentum_agents
from ..agents.noise_agent import make_noise_agents
from ..agents.rl_agent import RLAgent


class MarketSimulation:
    """Core agent-based market simulation.

    Parameters
    ----------
    config : SimulationConfig
        Full simulation configuration.
    extra_agents : list[BaseAgent], optional
        Additional agents (e.g. RL agents) injected from outside.
    """

    def __init__(
        self,
        config: SimulationConfig,
        extra_agents: list[BaseAgent] | None = None,
    ) -> None:
        self.config = config
        self.rng = np.random.default_rng(config.seed)

        self.asset = Asset(config.fundamental, self.rng)
        self.price_mech = PriceMechanism(config.price, self.rng)
        self.metrics = MetricsRecorder(config)

        self.agents: list[BaseAgent] = self._build_agents()
        if extra_agents:
            self.agents.extend(extra_agents)

        self._step: int = 0
        self._forced_sell_buffer: float = 0.0  # accumulated forced-sell units

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def reset(self) -> None:
        """Reset simulation to initial conditions."""
        reset_id_counter()
        self.rng = np.random.default_rng(self.config.seed)
        self.asset.reset()
        self.price_mech.reset()
        self.metrics.reset()
        self.agents = self._build_agents()
        self._step = 0
        self._forced_sell_buffer = 0.0

    def run(self) -> list[StepMetrics]:
        """Run the full simulation and return per-step metrics."""
        # Seed the metrics recorder with the initial price
        self.metrics._price_history.append(self.price_mech.price)

        for _ in range(self.config.n_steps):
            self.step()
        return self.metrics.records

    def step(self) -> StepMetrics:
        """Advance simulation by one time step. Returns step metrics."""
        price = self.price_mech.price
        fundamental = self.asset.fundamental_value

        # 1. Apply interest on all agent debt
        for agent in self.agents:
            if agent.is_active:
                apply_interest(agent, self.config.leverage.borrowing_rate)

        # 2. Compute rolling stats for agent observations
        vol = self.metrics.rolling_volatility()
        mom = self.metrics.rolling_momentum()
        agg_lev = self._aggregate_leverage(price)

        # 3. Collect agent orders
        buy_demand = 0.0
        sell_demand = 0.0  # kept positive; sign flipped when updating price

        for agent in self.agents:
            if not agent.is_active:
                continue
            bd, sd = agent.decide(
                price=price,
                fundamental=fundamental,
                rolling_volatility=vol,
                rolling_momentum=mom,
                aggregate_leverage=agg_lev,
                rng=self.rng,
            )
            bd = max(bd, 0.0)
            sd = max(sd, 0.0)
            agent.execute_buy(bd, price)
            agent.execute_sell(sd, price)
            buy_demand += bd
            sell_demand += sd

        # 4. Add any forced-sell pressure from previous step's margin calls
        if self._forced_sell_buffer > 0:
            sell_demand += self._forced_sell_buffer
            self._forced_sell_buffer = 0.0

        # 5. Normalise demand by agent count so price impact is per-agent-average,
        #    not total. This prevents overflow when many agents submit large orders.
        n_active = max(sum(1 for a in self.agents if a.is_active), 1)
        net_demand = (buy_demand - sell_demand) / n_active
        new_price = self.price_mech.step(net_demand)

        # 6. Update fundamental value
        self.asset.step()

        # 7. Margin calls and forced liquidation
        margin_results: list[MarginCallResult] = []
        forced_sell_units = 0.0

        for agent in self.agents:
            if not agent.is_active:
                continue
            result = check_and_liquidate(agent, new_price, self.config.leverage)
            if result is not None:
                margin_results.append(result)
                forced_sell_units += result.shares_liquidated

        # Buffer forced sales to feed into next step's sell demand
        self._forced_sell_buffer = forced_sell_units

        n_margin_calls = len(margin_results)
        n_defaults = sum(1 for r in margin_results if r.defaulted)

        # 8. Record metrics
        m = self.metrics.record(
            step=self._step,
            price=new_price,
            fundamental=self.asset.fundamental_value,
            buy_demand=buy_demand,
            sell_demand=sell_demand,
            n_margin_calls=n_margin_calls,
            n_defaults=n_defaults,
            agents=self.agents,
            minsky_thresholds=self.config.minsky,
        )

        self._step += 1
        return m

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def price(self) -> float:
        return self.price_mech.price

    @property
    def fundamental(self) -> float:
        return self.asset.fundamental_value

    def get_dataframe(self):
        return self.metrics.to_dataframe()

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _build_agents(self) -> list[BaseAgent]:
        cfg = self.config.agents
        agents: list[BaseAgent] = []
        agents.extend(make_fundamental_agents(cfg))
        agents.extend(make_momentum_agents(cfg))
        agents.extend(make_noise_agents(cfg))
        agents.extend(make_market_makers(cfg))
        return agents

    def _aggregate_leverage(self, price: float) -> float:
        lev = [a.leverage(price) for a in self.agents if a.is_active]
        return float(np.mean(lev)) if lev else 0.0
