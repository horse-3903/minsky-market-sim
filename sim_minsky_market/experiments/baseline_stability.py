"""Experiment 1: Baseline stable market — no RL, low leverage."""

from __future__ import annotations

from pathlib import Path

from ..market.config import (
    AgentConfig,
    FundamentalConfig,
    LeverageConfig,
    PriceConfig,
    SimulationConfig,
)
from ..market.environment import MarketSimulation
from ..analysis.plots import plot_all
from ..analysis.metrics_report import save_summary


def run(seed: int = 42, output_dir: str = "results/baseline") -> dict:
    config = SimulationConfig(
        n_steps=500,
        seed=seed,
        fundamental=FundamentalConfig(
            initial_value=100.0,
            drift=0.0,
            volatility=0.003,
        ),
        price=PriceConfig(
            initial_price=100.0,
            price_impact=0.008,
            noise_volatility=0.002,
            liquidity=1.0,
        ),
        leverage=LeverageConfig(
            max_leverage=2.0,
            borrowing_rate=0.0003,
            liquidation_fraction=0.5,
        ),
        agents=AgentConfig(
            n_fundamental=20,
            n_momentum=5,
            n_noise=10,
            n_rl=0,
            n_market_maker=0,
            initial_cash=1000.0,
        ),
    )

    sim = MarketSimulation(config)
    sim.run()
    df = sim.get_dataframe()

    out = Path(output_dir)
    plot_all(df, out / "plots")
    stats = save_summary(df, out, label=f"baseline_seed{seed}")
    return stats


if __name__ == "__main__":
    stats = run()
    for k, v in stats.items():
        print(f"  {k}: {v}")
