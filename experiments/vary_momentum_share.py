"""Experiment 2: Vary the share of momentum traders.

Sweeps momentum_fraction from 0.0 to 0.8, keeping total agents fixed at 40.
For each fraction, runs n_seeds independent seeds and records summary stats.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from market.config import (
    AgentConfig,
    FundamentalConfig,
    LeverageConfig,
    MinskyThresholds,
    PriceConfig,
    SimulationConfig,
)
from market.environment import MarketSimulation
from analysis.metrics_report import summary as compute_summary


TOTAL_AGENTS = 40
MOMENTUM_FRACTIONS = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
N_SEEDS = 10
N_STEPS = 500

# Calibrated so the market is stable at low momentum fractions and increasingly
# fragile above ~30-40%, demonstrating the Minsky mechanism endogenously.
_NOISE_VOL = 0.009          # enough price variance for meaningful momentum signals
_MOMENTUM_SENS = 2.0        # aggressive trend-following
_FUNDAMENTAL_SENS = 0.5     # stabilising force


def _config(momentum_fraction: float, seed: int) -> SimulationConfig:
    n_momentum = round(TOTAL_AGENTS * momentum_fraction)
    remaining = TOTAL_AGENTS - n_momentum
    n_fundamental = round(remaining * 0.6)
    n_noise = remaining - n_fundamental

    return SimulationConfig(
        n_steps=N_STEPS,
        seed=seed,
        fundamental=FundamentalConfig(initial_value=100.0, drift=0.0, volatility=0.005),
        price=PriceConfig(
            initial_price=100.0,
            price_impact=0.015,
            noise_volatility=_NOISE_VOL,
            liquidity=1.0,
        ),
        leverage=LeverageConfig(
            max_leverage=3.0,
            borrowing_rate=0.0003,
            margin_call_threshold=3.0,
            liquidation_fraction=0.5,
        ),
        agents=AgentConfig(
            n_fundamental=n_fundamental,
            n_momentum=n_momentum,
            n_noise=n_noise,
            n_rl=0,
            initial_cash=1000.0,
            momentum_sensitivity=_MOMENTUM_SENS,
            fundamental_sensitivity=_FUNDAMENTAL_SENS,
        ),
        minsky=MinskyThresholds(),
    )


def run(output_dir: str = "results/vary_momentum") -> pd.DataFrame:
    """Run the full sweep. Returns a DataFrame with one row per (fraction, seed)."""
    records = []

    for frac in MOMENTUM_FRACTIONS:
        for seed in range(N_SEEDS):
            cfg = _config(frac, seed)
            sim = MarketSimulation(cfg)
            sim.run()
            df = sim.get_dataframe()
            stats = compute_summary(df)
            stats["momentum_fraction"] = frac
            stats["n_momentum"] = cfg.agents.n_momentum
            stats["seed"] = seed
            records.append(stats)
            print(
                f"  frac={frac:.1f}  seed={seed}"
                f"  crashes={stats['n_crash_steps']}"
                f"  max_lev={stats['max_avg_leverage']:.2f}"
                f"  ponzi={stats['peak_ponzi_pct']:.1f}%"
            )

    results = pd.DataFrame(records)

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    results.to_csv(out / "raw_results.csv", index=False)

    # Aggregate: mean ± std per fraction
    agg = (
        results
        .groupby("momentum_fraction")
        .agg(
            crash_steps_mean=("n_crash_steps", "mean"),
            crash_steps_std=("n_crash_steps", "std"),
            max_leverage_mean=("max_avg_leverage", "mean"),
            max_leverage_std=("max_avg_leverage", "std"),
            peak_ponzi_mean=("peak_ponzi_pct", "mean"),
            peak_ponzi_std=("peak_ponzi_pct", "std"),
            max_drawdown_mean=("max_drawdown_pct", "mean"),
            max_drawdown_std=("max_drawdown_pct", "std"),
            total_margin_calls_mean=("total_margin_calls", "mean"),
            total_defaults_mean=("total_defaults", "mean"),
            gini_mean=("final_gini", "mean"),
            sharpe_mean=("annualised_sharpe", "mean"),
        )
        .reset_index()
    )
    agg.to_csv(out / "aggregated.csv", index=False)

    with open(out / "sweep_config.json", "w") as f:
        json.dump({
            "total_agents": TOTAL_AGENTS,
            "momentum_fractions": MOMENTUM_FRACTIONS,
            "n_seeds": N_SEEDS,
            "n_steps": N_STEPS,
        }, f, indent=2)

    print(f"\nSweep complete. Results in {out.resolve()}")
    return results


if __name__ == "__main__":
    run()
