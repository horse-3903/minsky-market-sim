"""Plotly chart builders for the Streamlit dashboard."""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots


_COLORS = {
    "price": "#2563eb",
    "fundamental": "#16a34a",
    "mispricing_pos": "#ef4444",
    "mispricing_neg": "#22c55e",
    "leverage_avg": "#7c3aed",
    "leverage_max": "#dc2626",
    "volatility": "#ea580c",
    "margin_calls": "#f97316",
    "defaults": "#991b1b",
    "hedge": "#22c55e",
    "speculative": "#f59e0b",
    "ponzi": "#ef4444",
    "fundamental_agent": "#3b82f6",
    "momentum_agent": "#f97316",
    "noise_agent": "#94a3b8",
    "rl_agent": "#a855f7",
}


def price_chart(df: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df["step"], y=df["price"],
        name="Market price", line=dict(color=_COLORS["price"], width=2),
    ))
    fig.add_trace(go.Scatter(
        x=df["step"], y=df["fundamental"],
        name="Fundamental value",
        line=dict(color=_COLORS["fundamental"], width=2, dash="dash"),
    ))

    # Shade crash regions
    crashes = df[df["crash"]]
    for _, row in crashes.iterrows():
        fig.add_vrect(
            x0=row["step"] - 1, x1=row["step"] + 1,
            fillcolor="rgba(239,68,68,0.12)", line_width=0,
        )

    fig.update_layout(
        title="Market Price vs Fundamental Value",
        xaxis_title="Step", yaxis_title="Price",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        height=380, margin=dict(l=0, r=0, t=40, b=0),
    )
    return fig


def mispricing_chart(df: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    mp = df["mispricing"] * 100
    fig.add_trace(go.Scatter(
        x=df["step"], y=mp,
        fill="tozeroy",
        fillcolor="rgba(239,68,68,0.15)",
        line=dict(color=_COLORS["mispricing_pos"], width=1.5),
        name="Mispricing (%)",
    ))
    fig.add_hline(y=0, line_dash="dot", line_color="gray")
    fig.update_layout(
        title="Relative Mispricing  (P − F) / F",
        xaxis_title="Step", yaxis_title="Mispricing (%)",
        height=280, margin=dict(l=0, r=0, t=40, b=0),
        showlegend=False,
    )
    return fig


def leverage_chart(df: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df["step"], y=df["avg_leverage"],
        name="Avg leverage", fill="tozeroy",
        fillcolor="rgba(124,58,237,0.12)",
        line=dict(color=_COLORS["leverage_avg"], width=2),
    ))
    fig.add_trace(go.Scatter(
        x=df["step"], y=df["max_leverage"],
        name="Max leverage",
        line=dict(color=_COLORS["leverage_max"], width=1.5, dash="dot"),
    ))
    fig.update_layout(
        title="Leverage over Time",
        xaxis_title="Step", yaxis_title="Leverage (×)",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        height=280, margin=dict(l=0, r=0, t=40, b=0),
    )
    return fig


def volatility_chart(df: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df["step"], y=df["rolling_volatility"] * 100,
        fill="tozeroy",
        fillcolor="rgba(234,88,12,0.15)",
        line=dict(color=_COLORS["volatility"], width=1.5),
        name="Rolling σ (%)",
    ))
    fig.update_layout(
        title="Rolling Return Volatility",
        xaxis_title="Step", yaxis_title="σ (%)",
        height=260, margin=dict(l=0, r=0, t=40, b=0),
        showlegend=False,
    )
    return fig


def margin_calls_chart(df: pd.DataFrame) -> go.Figure:
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                        subplot_titles=("Margin Calls", "Defaults"),
                        vertical_spacing=0.22)
    fig.add_trace(go.Bar(
        x=df["step"], y=df["n_margin_calls"],
        marker_color=_COLORS["margin_calls"], name="Margin calls",
    ), row=1, col=1)
    fig.add_trace(go.Bar(
        x=df["step"], y=df["n_defaults"],
        marker_color=_COLORS["defaults"], name="Defaults",
    ), row=2, col=1)
    fig.update_layout(
        height=380, margin=dict(l=0, r=0, t=30, b=0),
        showlegend=False,
    )
    return fig


def minsky_chart(df: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df["step"], y=df["pct_hedge"] * 100,
        stackgroup="one", name="Hedge",
        fillcolor="rgba(34,197,94,0.7)",
        line=dict(color=_COLORS["hedge"], width=0.5),
    ))
    fig.add_trace(go.Scatter(
        x=df["step"], y=df["pct_speculative"] * 100,
        stackgroup="one", name="Speculative",
        fillcolor="rgba(245,158,11,0.7)",
        line=dict(color=_COLORS["speculative"], width=0.5),
    ))
    fig.add_trace(go.Scatter(
        x=df["step"], y=df["pct_ponzi"] * 100,
        stackgroup="one", name="Ponzi",
        fillcolor="rgba(239,68,68,0.7)",
        line=dict(color=_COLORS["ponzi"], width=0.5),
    ))
    fig.update_layout(
        xaxis_title="Step", yaxis_title="% of active agents",
        yaxis=dict(range=[0, 100]),
        legend=dict(orientation="h", yanchor="top", y=-0.18, xanchor="center", x=0.5),
        height=360, margin=dict(l=0, r=0, t=10, b=60),
    )
    return fig


def wealth_chart(df: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    type_map = {
        "wealth_fundamental": ("Fundamental", _COLORS["fundamental_agent"]),
        "wealth_momentum": ("Momentum", _COLORS["momentum_agent"]),
        "wealth_noise": ("Noise", _COLORS["noise_agent"]),
        "wealth_rl": ("RL", _COLORS["rl_agent"]),
    }
    for col, (label, color) in type_map.items():
        if df[col].abs().sum() > 0:
            fig.add_trace(go.Scatter(
                x=df["step"], y=df[col],
                name=label, line=dict(color=color, width=2),
            ))
    fig.update_layout(
        title="Average Wealth by Agent Type",
        xaxis_title="Step", yaxis_title="Avg wealth",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        height=300, margin=dict(l=0, r=0, t=40, b=0),
    )
    return fig


def drawdown_chart(df: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df["step"], y=df["drawdown"] * 100,
        fill="tozeroy",
        fillcolor="rgba(220,38,38,0.2)",
        line=dict(color="#dc2626", width=1.5),
        name="Drawdown (%)",
    ))
    fig.add_hline(y=0, line_dash="dot", line_color="gray")
    fig.update_layout(
        title="Price Drawdown from Peak",
        xaxis_title="Step", yaxis_title="Drawdown (%)",
        height=260, margin=dict(l=0, r=0, t=40, b=0),
        showlegend=False,
    )
    return fig


def returns_histogram(df: pd.DataFrame) -> go.Figure:
    returns = df["return_"].dropna() * 100
    fig = go.Figure()
    fig.add_trace(go.Histogram(
        x=returns, nbinsx=60,
        marker_color=_COLORS["price"], opacity=0.75,
        name="Returns",
    ))
    fig.update_layout(
        title="Return Distribution",
        xaxis_title="Return (%)", yaxis_title="Frequency",
        height=260, margin=dict(l=0, r=0, t=40, b=0),
        showlegend=False,
    )
    return fig


def gini_chart(df: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df["step"], y=df["gini"],
        fill="tozeroy",
        fillcolor="rgba(168,85,247,0.15)",
        line=dict(color="#a855f7", width=1.5),
    ))
    fig.update_layout(
        title="Wealth Inequality (Gini Coefficient)",
        xaxis_title="Step", yaxis_title="Gini",
        height=240, margin=dict(l=0, r=0, t=40, b=0),
        showlegend=False,
    )
    return fig


def sweep_crashes_chart(agg: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=agg["momentum_fraction"] * 100,
        y=agg["crash_steps_mean"],
        error_y=dict(type="data", array=agg["crash_steps_std"].tolist(), visible=True),
        marker_color=_COLORS["defaults"],
        name="Avg crash steps",
    ))
    fig.update_layout(
        title="Crash Steps vs Momentum Fraction",
        xaxis_title="Momentum traders (%)",
        yaxis_title="Crash steps (mean ± SD)",
        height=320, margin=dict(l=0, r=0, t=40, b=0),
        showlegend=False,
    )
    return fig


def sweep_leverage_chart(agg: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=agg["momentum_fraction"] * 100,
        y=agg["max_leverage_mean"],
        error_y=dict(type="data", array=agg["max_leverage_std"].tolist(), visible=True),
        mode="lines+markers",
        line=dict(color=_COLORS["leverage_avg"], width=2),
        marker=dict(size=8),
        name="Max avg leverage",
    ))
    fig.update_layout(
        title="Peak Leverage vs Momentum Fraction",
        xaxis_title="Momentum traders (%)",
        yaxis_title="Max avg leverage (×)",
        height=320, margin=dict(l=0, r=0, t=40, b=0),
        showlegend=False,
    )
    return fig


def sweep_drawdown_chart(agg: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=agg["momentum_fraction"] * 100,
        y=agg["max_drawdown_mean"].abs(),
        error_y=dict(type="data", array=agg["max_drawdown_std"].tolist(), visible=True),
        mode="lines+markers",
        fill="tozeroy",
        fillcolor="rgba(220,38,38,0.15)",
        line=dict(color="#dc2626", width=2),
        marker=dict(size=8),
        name="Max drawdown",
    ))
    fig.update_layout(
        title="Max Drawdown vs Momentum Fraction",
        xaxis_title="Momentum traders (%)",
        yaxis_title="Max drawdown (%)",
        height=320, margin=dict(l=0, r=0, t=40, b=0),
        showlegend=False,
    )
    return fig


def sweep_gini_chart(agg: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=agg["momentum_fraction"] * 100,
        y=agg["gini_mean"],
        mode="lines+markers",
        line=dict(color="#a855f7", width=2),
        marker=dict(size=8),
        name="Final Gini",
    ))
    fig.update_layout(
        title="Wealth Inequality (Gini) vs Momentum Fraction",
        xaxis_title="Momentum traders (%)",
        yaxis_title="Final Gini coefficient",
        height=320, margin=dict(l=0, r=0, t=40, b=0),
        showlegend=False,
    )
    return fig


def overview_sparklines(df: pd.DataFrame) -> go.Figure:
    """2×2 mini dashboard for the Overview tab."""
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=(
            "Price vs Fundamental",
            "Avg Leverage",
            "Rolling Volatility",
            "Mispricing %",
        ),
        vertical_spacing=0.18,
        horizontal_spacing=0.1,
    )
    fig.add_trace(go.Scatter(x=df["step"], y=df["price"],
                             line=dict(color=_COLORS["price"], width=1.5),
                             name="Price"), row=1, col=1)
    fig.add_trace(go.Scatter(x=df["step"], y=df["fundamental"],
                             line=dict(color=_COLORS["fundamental"], width=1.5, dash="dash"),
                             name="Fundamental"), row=1, col=1)
    fig.add_trace(go.Scatter(x=df["step"], y=df["avg_leverage"],
                             fill="tozeroy",
                             fillcolor="rgba(124,58,237,0.15)",
                             line=dict(color=_COLORS["leverage_avg"], width=1.5),
                             name="Avg lev"), row=1, col=2)
    fig.add_trace(go.Scatter(x=df["step"], y=df["rolling_volatility"] * 100,
                             fill="tozeroy",
                             fillcolor="rgba(234,88,12,0.15)",
                             line=dict(color=_COLORS["volatility"], width=1.5),
                             name="Vol"), row=2, col=1)
    fig.add_trace(go.Scatter(x=df["step"], y=df["mispricing"] * 100,
                             fill="tozeroy",
                             fillcolor="rgba(239,68,68,0.15)",
                             line=dict(color=_COLORS["mispricing_pos"], width=1.5),
                             name="Mispricing"), row=2, col=2)
    fig.update_layout(
        height=480, showlegend=False,
        margin=dict(l=0, r=0, t=40, b=0),
    )
    return fig
