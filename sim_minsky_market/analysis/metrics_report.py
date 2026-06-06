"""Summary statistics from a completed simulation run."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


def summary(df: pd.DataFrame) -> dict:
    """Compute key summary statistics from a metrics DataFrame."""
    returns = df["return_"].values

    sharpe = (
        float(np.mean(returns) / np.std(returns) * np.sqrt(252))
        if np.std(returns) > 0
        else 0.0
    )

    return {
        "n_steps": len(df),
        "final_price": float(df["price"].iloc[-1]),
        "final_fundamental": float(df["fundamental"].iloc[-1]),
        "final_mispricing_pct": float(df["mispricing"].iloc[-1] * 100),
        "max_abs_mispricing_pct": float(df["abs_mispricing"].max() * 100),
        "mean_abs_mispricing_pct": float(df["abs_mispricing"].mean() * 100),
        "max_drawdown_pct": float(df["drawdown"].min() * 100),
        "annualised_sharpe": sharpe,
        "mean_rolling_vol": float(df["rolling_volatility"].mean()),
        "total_margin_calls": int(df["n_margin_calls"].sum()),
        "total_defaults": int(df["n_defaults"].sum()),
        "n_crash_steps": int(df["crash"].sum()),
        "n_bubble_steps": int(df["bubble"].sum()),
        "mean_avg_leverage": float(df["avg_leverage"].mean()),
        "max_avg_leverage": float(df["avg_leverage"].max()),
        "peak_ponzi_pct": float(df["pct_ponzi"].max() * 100),
        "final_avg_wealth": float(df["avg_wealth"].iloc[-1]),
        "final_gini": float(df["gini"].iloc[-1]),
    }


def save_summary(df: pd.DataFrame, output_dir: str | Path, label: str = "run") -> None:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    stats = summary(df)
    with open(out / f"{label}_summary.json", "w") as f:
        json.dump(stats, f, indent=2)
    df.to_csv(out / f"{label}_timeseries.csv", index=False)
    print(f"Results saved to {out.resolve()}")
    return stats
