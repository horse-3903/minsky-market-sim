"""Streamlit dashboard for the Minsky Market Simulation."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

# Ensure the project root is on the path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

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

from charts import (
    drawdown_chart,
    gini_chart,
    leverage_chart,
    margin_calls_chart,
    minsky_chart,
    mispricing_chart,
    overview_sparklines,
    price_chart,
    returns_histogram,
    sweep_crashes_chart,
    sweep_drawdown_chart,
    sweep_gini_chart,
    sweep_leverage_chart,
    volatility_chart,
    wealth_chart,
)

# ──────────────────────────────────────────────────────────────────────────────
# Page config
# ──────────────────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Minsky Market Sim",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="expanded",
)

# ──────────────────────────────────────────────────────────────────────────────
# Sidebar — simulation controls
# ──────────────────────────────────────────────────────────────────────────────

with st.sidebar:
    st.title("Simulation Controls")

    st.subheader("General")
    n_steps = st.slider("Steps", 100, 2000, 500, step=100)
    seed = st.number_input("Random seed", min_value=0, max_value=9999, value=42, step=1)

    st.subheader("Agent Counts")
    n_fundamental = st.slider("Fundamental traders", 0, 50, 20)
    n_momentum = st.slider("Momentum traders", 0, 50, 10)
    n_noise = st.slider("Noise traders", 0, 50, 10)

    st.subheader("Market Parameters")
    price_impact = st.slider("Price impact (α)", 0.001, 0.05, 0.01, step=0.001, format="%.3f")
    liquidity = st.slider("Liquidity", 0.2, 3.0, 1.0, step=0.1)
    noise_vol = st.slider("Price noise σ", 0.0, 0.02, 0.002, step=0.001, format="%.3f")
    fund_vol = st.slider("Fundamental σ", 0.0, 0.02, 0.003, step=0.001, format="%.3f")

    st.subheader("Leverage & Margin")
    max_leverage = st.slider("Max leverage (L_max)", 1.0, 10.0, 3.0, step=0.5)
    borrowing_rate = st.slider("Borrowing rate (per step)", 0.0, 0.005, 0.0003,
                               step=0.0001, format="%.4f")

    st.subheader("Agent Parameters")
    initial_cash = st.slider("Initial cash per agent", 100, 5000, 1000, step=100)
    fund_sens = st.slider("Fundamental sensitivity", 0.05, 2.0, 0.5, step=0.05)
    mom_sens = st.slider("Momentum sensitivity", 0.05, 2.0, 0.3, step=0.05)

    run_btn = st.button("Run Simulation", use_container_width=True, type="primary")

# ──────────────────────────────────────────────────────────────────────────────
# Session state — persist results between reruns
# ──────────────────────────────────────────────────────────────────────────────

if "df" not in st.session_state:
    st.session_state.df = None
if "stats" not in st.session_state:
    st.session_state.stats = None

if run_btn:
    config = SimulationConfig(
        n_steps=n_steps,
        seed=int(seed),
        fundamental=FundamentalConfig(
            initial_value=100.0,
            drift=0.0,
            volatility=fund_vol,
        ),
        price=PriceConfig(
            initial_price=100.0,
            price_impact=price_impact,
            noise_volatility=noise_vol,
            liquidity=liquidity,
        ),
        leverage=LeverageConfig(
            max_leverage=max_leverage,
            borrowing_rate=borrowing_rate,
            margin_call_threshold=max_leverage,
            liquidation_fraction=0.5,
        ),
        agents=AgentConfig(
            n_fundamental=n_fundamental,
            n_momentum=n_momentum,
            n_noise=n_noise,
            n_rl=0,
            initial_cash=initial_cash,
            fundamental_sensitivity=fund_sens,
            momentum_sensitivity=mom_sens,
        ),
        minsky=MinskyThresholds(),
    )

    with st.spinner("Running simulation…"):
        sim = MarketSimulation(config)
        sim.run()
        st.session_state.df = sim.get_dataframe()
        st.session_state.stats = compute_summary(st.session_state.df)

# ──────────────────────────────────────────────────────────────────────────────
# Main content
# ──────────────────────────────────────────────────────────────────────────────

st.title("Minsky Market Simulation Dashboard")
st.caption(
    "Agent-based model of Minsky's Financial Instability Hypothesis. "
    "Configure parameters in the sidebar and press **Run Simulation**."
)

if st.session_state.df is None:
    st.info("Configure parameters in the sidebar and press **Run Simulation** to begin.")
    st.stop()

df: pd.DataFrame = st.session_state.df
stats: dict = st.session_state.stats

# ── KPI metrics row ──────────────────────────────────────────────────────────

st.subheader("Key Metrics")

c1, c2, c3, c4, c5, c6, c7 = st.columns(7)
c1.metric("Final Price", f"{stats['final_price']:.2f}",
          delta=f"{stats['final_mispricing_pct']:.1f}% vs F",
          delta_color="inverse")
c2.metric("Max Mispricing", f"{stats['max_abs_mispricing_pct']:.1f}%")
c3.metric("Max Drawdown", f"{stats['max_drawdown_pct']:.1f}%", delta_color="off")
c4.metric("Sharpe (ann.)", f"{stats['annualised_sharpe']:.2f}")
c5.metric("Margin Calls", f"{stats['total_margin_calls']:,}")
c6.metric("Defaults", f"{stats['total_defaults']:,}")
c7.metric("Peak Ponzi %", f"{stats['peak_ponzi_pct']:.1f}%")

st.divider()

# ── Tabs ─────────────────────────────────────────────────────────────────────

tab_overview, tab_market, tab_risk, tab_agents, tab_experiments, tab_data = st.tabs(
    ["Overview", "Market", "Leverage & Risk", "Agent Performance", "Experiments", "Raw Data"]
)

# ── Overview ─────────────────────────────────────────────────────────────────

with tab_overview:
    st.plotly_chart(overview_sparklines(df), use_container_width=True)

    col_l, col_r = st.columns(2)
    with col_l:
        st.subheader("Crash & Bubble Events")
        n_crashes = int(df["crash"].sum())
        n_bubbles = int(df["bubble"].sum())
        ev_col1, ev_col2 = st.columns(2)
        ev_col1.metric("Crash steps", n_crashes)
        ev_col2.metric("Bubble steps", n_bubbles)

        # Timeline of events
        events = df[(df["crash"]) | (df["bubble"])][["step", "crash", "bubble", "price", "fundamental"]].copy()
        if not events.empty:
            events["event"] = events.apply(
                lambda r: "Crash" if r["crash"] else "Bubble", axis=1
            )
            st.dataframe(events[["step", "event", "price", "fundamental"]].reset_index(drop=True),
                         use_container_width=True, height=200)
        else:
            st.success("No crash or bubble events detected.")

    with col_r:
        st.subheader("Summary Statistics")
        summary_rows = {
            "Mean avg leverage": f"{stats['mean_avg_leverage']:.3f}×",
            "Max avg leverage": f"{stats['max_avg_leverage']:.3f}×",
            "Mean rolling vol": f"{stats['mean_rolling_vol']*100:.3f}%",
            "Mean |mispricing|": f"{stats['mean_abs_mispricing_pct']:.2f}%",
            "Final avg wealth": f"{stats['final_avg_wealth']:.1f}",
            "Final Gini": f"{stats['final_gini']:.3f}",
        }
        for label, val in summary_rows.items():
            col_a, col_b = st.columns([2, 1])
            col_a.write(label)
            col_b.write(f"**{val}**")

# ── Market ────────────────────────────────────────────────────────────────────

with tab_market:
    st.plotly_chart(price_chart(df), use_container_width=True)

    col1, col2 = st.columns(2)
    with col1:
        st.plotly_chart(mispricing_chart(df), use_container_width=True)
    with col2:
        st.plotly_chart(volatility_chart(df), use_container_width=True)

    col3, col4 = st.columns(2)
    with col3:
        st.plotly_chart(drawdown_chart(df), use_container_width=True)
    with col4:
        st.plotly_chart(returns_histogram(df), use_container_width=True)

# ── Leverage & Risk ───────────────────────────────────────────────────────────

with tab_risk:
    st.plotly_chart(leverage_chart(df), use_container_width=True)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Minsky Finance-State Composition**")
        st.plotly_chart(minsky_chart(df), use_container_width=True)
    with col2:
        st.markdown("**Margin Calls & Defaults per Step**")
        st.plotly_chart(margin_calls_chart(df), use_container_width=True)

    # Minsky state breakdown at last step
    st.subheader("Finance-State Snapshot (final step)")
    last = df.iloc[-1]
    m1, m2, m3 = st.columns(3)
    m1.metric("Hedge", f"{last['pct_hedge']*100:.1f}%")
    m2.metric("Speculative", f"{last['pct_speculative']*100:.1f}%",
              delta_color="inverse" if last["pct_speculative"] > 0.2 else "off")
    m3.metric("Ponzi", f"{last['pct_ponzi']*100:.1f}%",
              delta_color="inverse" if last["pct_ponzi"] > 0.05 else "off")

# ── Agent Performance ─────────────────────────────────────────────────────────

with tab_agents:
    st.plotly_chart(wealth_chart(df), use_container_width=True)

    col1, col2 = st.columns(2)
    with col1:
        st.plotly_chart(gini_chart(df), use_container_width=True)
    with col2:
        # Final wealth by type
        st.subheader("Final Average Wealth by Type")
        last = df.iloc[-1]
        wealth_rows = {
            "Fundamental": last["wealth_fundamental"],
            "Momentum": last["wealth_momentum"],
            "Noise": last["wealth_noise"],
            "RL": last["wealth_rl"],
        }
        active_types = {k: v for k, v in wealth_rows.items() if v != 0}
        if active_types:
            w_df = pd.DataFrame(
                {"Agent type": list(active_types.keys()),
                 "Avg wealth": [f"{v:.1f}" for v in active_types.values()]}
            )
            st.dataframe(w_df, use_container_width=True, hide_index=True)
        else:
            st.info("No agent type wealth data available.")

# ── Experiments ──────────────────────────────────────────────────────────────

SWEEP_AGG = ROOT / "results" / "vary_momentum" / "aggregated.csv"
SWEEP_RAW = ROOT / "results" / "vary_momentum" / "raw_results.csv"
SWEEP_STATS = ROOT / "results" / "vary_momentum" / "statistical_tests.json"

with tab_experiments:
    st.subheader("Experiment 2: Momentum Trader Share Sweep")
    st.caption(
        "Varies the fraction of momentum traders from 0% to 80% (40 total agents, "
        "10 seeds per condition). Tests whether increasing momentum-trader dominance "
        "endogenously generates the leverage buildup and crashes described by Minsky."
    )

    if not SWEEP_AGG.exists():
        st.warning(
            "Sweep results not found. Run the experiment first:\n\n"
            "```\npython -m experiments.vary_momentum_share\n```"
        )
    else:
        agg = pd.read_csv(SWEEP_AGG)

        col1, col2 = st.columns(2)
        with col1:
            st.plotly_chart(sweep_crashes_chart(agg), use_container_width=True)
        with col2:
            st.plotly_chart(sweep_leverage_chart(agg), use_container_width=True)

        col3, col4 = st.columns(2)
        with col3:
            st.plotly_chart(sweep_drawdown_chart(agg), use_container_width=True)
        with col4:
            st.plotly_chart(sweep_gini_chart(agg), use_container_width=True)

        # Statistical test summary
        if SWEEP_STATS.exists():
            import json as _json
            with open(SWEEP_STATS) as _f:
                stat_report = _json.load(_f)

            st.subheader("Spearman Correlations with Momentum Fraction")
            stat_rows = []
            for metric, res in stat_report.items():
                sp = res["spearman"]
                kw = res["kruskal_wallis"]
                stat_rows.append({
                    "Metric": metric,
                    "Spearman rho": f"{sp['rho']:+.3f}",
                    "p-value": f"{sp['p']:.4f}",
                    "Significant": "Yes" if sp["significant"] else "No",
                    "Kruskal-Wallis p": f"{kw['p']:.4f}",
                })
            st.dataframe(
                pd.DataFrame(stat_rows),
                use_container_width=True,
                hide_index=True,
            )

        # Raw results download
        raw = pd.read_csv(SWEEP_RAW)
        st.download_button(
            "Download raw sweep results (CSV)",
            data=raw.to_csv(index=False).encode(),
            file_name="sweep_vary_momentum.csv",
            mime="text/csv",
        )


# ── Raw Data ─────────────────────────────────────────────────────────────────

with tab_data:
    st.subheader("Time-Series Data")

    # Column selector
    all_cols = list(df.columns)
    default_cols = ["step", "price", "fundamental", "mispricing",
                    "avg_leverage", "rolling_volatility", "n_margin_calls",
                    "n_defaults", "pct_hedge", "pct_speculative", "pct_ponzi"]
    selected_cols = st.multiselect(
        "Columns to display", all_cols,
        default=[c for c in default_cols if c in all_cols],
    )

    st.dataframe(
        df[selected_cols].round(6) if selected_cols else df.round(6),
        use_container_width=True,
        height=400,
    )

    col_dl1, col_dl2 = st.columns(2)
    with col_dl1:
        csv_bytes = df.to_csv(index=False).encode()
        st.download_button(
            "Download CSV",
            data=csv_bytes,
            file_name="simulation_results.csv",
            mime="text/csv",
            use_container_width=True,
        )
    with col_dl2:
        json_str = json.dumps(stats, indent=2)
        st.download_button(
            "Download Summary JSON",
            data=json_str,
            file_name="summary_stats.json",
            mime="application/json",
            use_container_width=True,
        )
