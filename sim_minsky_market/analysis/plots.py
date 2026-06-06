"""Visualisation helpers for market simulation results."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Style helpers
# ---------------------------------------------------------------------------

def _style() -> None:
    plt.rcParams.update(
        {
            "figure.dpi": 120,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.3,
            "font.size": 10,
        }
    )


def _save(fig: plt.Figure, path: Path | str) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Individual plots
# ---------------------------------------------------------------------------

def plot_price_vs_fundamental(df: pd.DataFrame, out: Path | str) -> None:
    _style()
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(df["step"], df["price"], label="Market price", linewidth=1.2)
    ax.plot(df["step"], df["fundamental"], label="Fundamental value", linewidth=1.2, linestyle="--")
    ax.set_xlabel("Step")
    ax.set_ylabel("Price")
    ax.set_title("Price vs Fundamental Value")
    ax.legend()
    _save(fig, out)


def plot_mispricing(df: pd.DataFrame, out: Path | str) -> None:
    _style()
    fig, ax = plt.subplots(figsize=(10, 3))
    ax.plot(df["step"], df["mispricing"] * 100, color="purple", linewidth=1.0)
    ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
    ax.fill_between(df["step"], df["mispricing"] * 100, 0,
                    where=df["mispricing"] > 0, alpha=0.25, color="red", label="Overvalued")
    ax.fill_between(df["step"], df["mispricing"] * 100, 0,
                    where=df["mispricing"] < 0, alpha=0.25, color="green", label="Undervalued")
    ax.set_xlabel("Step")
    ax.set_ylabel("Mispricing (%)")
    ax.set_title("Relative Mispricing  (P − F) / F")
    ax.legend()
    _save(fig, out)


def plot_leverage(df: pd.DataFrame, out: Path | str) -> None:
    _style()
    fig, ax = plt.subplots(figsize=(10, 3))
    ax.plot(df["step"], df["avg_leverage"], label="Average leverage", linewidth=1.2)
    ax.plot(df["step"], df["max_leverage"], label="Max leverage", linewidth=1.0, linestyle=":", color="red")
    ax.set_xlabel("Step")
    ax.set_ylabel("Leverage (×)")
    ax.set_title("Aggregate Leverage over Time")
    ax.legend()
    _save(fig, out)


def plot_volatility(df: pd.DataFrame, out: Path | str) -> None:
    _style()
    fig, ax = plt.subplots(figsize=(10, 3))
    ax.plot(df["step"], df["rolling_volatility"] * 100, color="darkorange", linewidth=1.0)
    ax.set_xlabel("Step")
    ax.set_ylabel("Rolling σ (%)")
    ax.set_title("Rolling Return Volatility")
    _save(fig, out)


def plot_margin_calls(df: pd.DataFrame, out: Path | str) -> None:
    _style()
    fig, axes = plt.subplots(2, 1, figsize=(10, 5), sharex=True)
    axes[0].bar(df["step"], df["n_margin_calls"], color="tomato", width=1.0, label="Margin calls")
    axes[0].set_ylabel("Count")
    axes[0].set_title("Margin Calls and Defaults per Step")
    axes[0].legend()

    axes[1].bar(df["step"], df["n_defaults"], color="darkred", width=1.0, label="Defaults")
    axes[1].set_ylabel("Count")
    axes[1].set_xlabel("Step")
    axes[1].legend()
    fig.tight_layout()
    _save(fig, out)


def plot_minsky_composition(df: pd.DataFrame, out: Path | str) -> None:
    _style()
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.stackplot(
        df["step"],
        df["pct_hedge"] * 100,
        df["pct_speculative"] * 100,
        df["pct_ponzi"] * 100,
        labels=["Hedge", "Speculative", "Ponzi"],
        colors=["#2ecc71", "#f39c12", "#e74c3c"],
        alpha=0.85,
    )
    ax.set_xlabel("Step")
    ax.set_ylabel("% of active agents")
    ax.set_title("Minsky Finance-State Composition")
    ax.yaxis.set_major_formatter(mticker.PercentFormatter())
    ax.legend(loc="upper left")
    _save(fig, out)


def plot_wealth_by_type(df: pd.DataFrame, out: Path | str) -> None:
    _style()
    fig, ax = plt.subplots(figsize=(10, 4))
    for col, label, color in [
        ("wealth_fundamental", "Fundamental", "#3498db"),
        ("wealth_momentum", "Momentum", "#e67e22"),
        ("wealth_noise", "Noise", "#95a5a6"),
        ("wealth_rl", "RL", "#9b59b6"),
    ]:
        if df[col].abs().sum() > 0:
            ax.plot(df["step"], df[col], label=label, color=color, linewidth=1.2)
    ax.set_xlabel("Step")
    ax.set_ylabel("Average wealth")
    ax.set_title("Average Wealth by Agent Type")
    ax.legend()
    _save(fig, out)


def plot_drawdown(df: pd.DataFrame, out: Path | str) -> None:
    _style()
    fig, ax = plt.subplots(figsize=(10, 3))
    ax.fill_between(df["step"], df["drawdown"] * 100, 0, color="crimson", alpha=0.5)
    ax.plot(df["step"], df["drawdown"] * 100, color="crimson", linewidth=0.8)
    ax.set_xlabel("Step")
    ax.set_ylabel("Drawdown (%)")
    ax.set_title("Price Drawdown from Peak")
    _save(fig, out)


def plot_all(df: pd.DataFrame, output_dir: str | Path = "results/plots") -> None:
    """Generate the full standard plot suite into output_dir."""
    out = Path(output_dir)
    plot_price_vs_fundamental(df, out / "price_vs_fundamental.png")
    plot_mispricing(df, out / "mispricing.png")
    plot_leverage(df, out / "leverage.png")
    plot_volatility(df, out / "volatility.png")
    plot_margin_calls(df, out / "margin_calls.png")
    plot_minsky_composition(df, out / "minsky_composition.png")
    plot_wealth_by_type(df, out / "wealth_by_type.png")
    plot_drawdown(df, out / "drawdown.png")
    print(f"Plots saved to {out.resolve()}")
