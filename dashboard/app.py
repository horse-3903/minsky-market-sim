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
# Presets
# ──────────────────────────────────────────────────────────────────────────────

PRESETS = {
    "stable_baseline": {
        "label": "Stable Baseline",
        "description": "Fundamental traders dominate. Price tracks fair value, no crashes, low volatility.",
        "params": dict(
            n_steps=500, seed=42,
            n_fundamental=25, n_momentum=5, n_noise=10,
            price_impact=0.008, liquidity=1.0, noise_vol=0.003, fund_vol=0.003,
            max_leverage=2.0, borrowing_rate=0.0003,
            initial_cash=1000, fund_sens=0.5, mom_sens=0.3,
        ),
    },
    "minsky_moment": {
        "label": "Minsky Moment",
        "description": "Near the critical threshold. Leverage builds steadily; crashes emerge occasionally.",
        "params": dict(
            n_steps=500, seed=7,
            n_fundamental=14, n_momentum=18, n_noise=8,
            price_impact=0.015, liquidity=1.0, noise_vol=0.009, fund_vol=0.005,
            max_leverage=3.0, borrowing_rate=0.0003,
            initial_cash=1000, fund_sens=0.5, mom_sens=2.0,
        ),
    },
    "full_collapse": {
        "label": "Full Collapse",
        "description": "Momentum traders overwhelm stabilisers. Rapid leverage buildup leads to total market collapse.",
        "params": dict(
            n_steps=500, seed=42,
            n_fundamental=6, n_momentum=28, n_noise=6,
            price_impact=0.015, liquidity=1.0, noise_vol=0.009, fund_vol=0.005,
            max_leverage=3.0, borrowing_rate=0.0003,
            initial_cash=1000, fund_sens=0.5, mom_sens=2.0,
        ),
    },
    "high_leverage": {
        "label": "High Leverage / Fragile",
        "description": "High leverage limit with aggressive borrowing. Small shocks cascade into margin call chains.",
        "params": dict(
            n_steps=500, seed=42,
            n_fundamental=15, n_momentum=15, n_noise=10,
            price_impact=0.015, liquidity=0.7, noise_vol=0.008, fund_vol=0.005,
            max_leverage=6.0, borrowing_rate=0.001,
            initial_cash=1000, fund_sens=0.5, mom_sens=1.5,
        ),
    },
}

_DEFAULTS = PRESETS["stable_baseline"]["params"]

def _apply_preset(key: str) -> None:
    p = PRESETS[key]["params"]
    for k, v in p.items():
        st.session_state[k] = v
    st.session_state["_run_preset"] = True

# ──────────────────────────────────────────────────────────────────────────────
# Sidebar: simulation controls
# ──────────────────────────────────────────────────────────────────────────────

with st.sidebar:
    st.title("Simulation Controls")

    st.subheader("General")
    n_steps = st.slider("Steps", 100, 2000, _DEFAULTS["n_steps"], step=100, key="n_steps")
    seed = st.number_input("Random seed", min_value=0, max_value=9999, value=_DEFAULTS["seed"], step=1, key="seed")

    st.subheader("Agent Counts")
    n_fundamental = st.slider("Fundamental traders", 0, 50, _DEFAULTS["n_fundamental"], key="n_fundamental")
    n_momentum = st.slider("Momentum traders", 0, 50, _DEFAULTS["n_momentum"], key="n_momentum")
    n_noise = st.slider("Noise traders", 0, 50, _DEFAULTS["n_noise"], key="n_noise")

    st.subheader("Market Parameters")
    price_impact = st.slider("Price impact (α)", 0.001, 0.05, _DEFAULTS["price_impact"], step=0.001, format="%.3f", key="price_impact")
    liquidity = st.slider("Liquidity", 0.2, 3.0, _DEFAULTS["liquidity"], step=0.1, key="liquidity")
    noise_vol = st.slider("Price noise σ", 0.0, 0.02, _DEFAULTS["noise_vol"], step=0.001, format="%.3f", key="noise_vol")
    fund_vol = st.slider("Fundamental σ", 0.0, 0.02, _DEFAULTS["fund_vol"], step=0.001, format="%.3f", key="fund_vol")

    st.subheader("Leverage & Margin")
    max_leverage = st.slider("Max leverage (L_max)", 1.0, 10.0, _DEFAULTS["max_leverage"], step=0.5, key="max_leverage")
    borrowing_rate = st.slider("Borrowing rate (per step)", 0.0, 0.005, _DEFAULTS["borrowing_rate"],
                               step=0.0001, format="%.4f", key="borrowing_rate")

    st.subheader("Agent Parameters")
    initial_cash = st.slider("Initial cash per agent", 100, 5000, _DEFAULTS["initial_cash"], step=100, key="initial_cash")
    fund_sens = st.slider("Fundamental sensitivity", 0.05, 2.0, _DEFAULTS["fund_sens"], step=0.05, key="fund_sens")
    mom_sens = st.slider("Momentum sensitivity", 0.05, 2.0, _DEFAULTS["mom_sens"], step=0.05, key="mom_sens")

    run_btn = st.button("Run Simulation", use_container_width=True, type="primary")

# ──────────────────────────────────────────────────────────────────────────────
# Session state: persist results between reruns
# ──────────────────────────────────────────────────────────────────────────────

if "df" not in st.session_state:
    st.session_state.df = None
if "stats" not in st.session_state:
    st.session_state.stats = None
if "_run_preset" not in st.session_state:
    st.session_state["_run_preset"] = False

should_run = run_btn or st.session_state.pop("_run_preset", False)

if should_run:
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

# ── Tabs ─────────────────────────────────────────────────────────────────────

tab_overview, tab_market, tab_risk, tab_agents, tab_experiments, tab_data, tab_guide = st.tabs(
    ["Overview", "Market", "Leverage & Risk", "Agent Performance", "Experiments", "Raw Data", "Guide"]
)

_no_sim = st.session_state.df is None

if _no_sim:
    for _t in (tab_overview, tab_market, tab_risk, tab_agents, tab_data):
        with _t:
            st.subheader("Configure parameters in the sidebar and press Run Simulation to begin.")
            st.markdown(
                "New here? The **Guide** tab (far right) explains every parameter and what each chart measures."
            )
            st.subheader("Or try a preset")
            p_cols = st.columns(4)
            for col, key in zip(p_cols, list(PRESETS.keys())):
                preset = PRESETS[key]
                with col:
                    st.markdown(f"**{preset['label']}**")
                    st.caption(preset["description"])
                    st.button(
                        "Load & Run",
                        key=f"preset_btn_{key}_{_t}",
                        on_click=_apply_preset,
                        args=(key,),
                        use_container_width=True,
                    )

df: pd.DataFrame = st.session_state.df
stats: dict = st.session_state.stats if st.session_state.stats else {}

if not _no_sim:
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

# ── Overview ─────────────────────────────────────────────────────────────────

with tab_overview:
  if not _no_sim:
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
  if not _no_sim:
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
  if not _no_sim:
    st.plotly_chart(leverage_chart(df), use_container_width=True)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Minsky Finance-State Composition**")
        st.plotly_chart(minsky_chart(df), use_container_width=True)
    with col2:
        st.markdown("**Margin Calls & Defaults per Step**")
        st.plotly_chart(margin_calls_chart(df), use_container_width=True)

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
  if not _no_sim:
    st.plotly_chart(wealth_chart(df), use_container_width=True)

    col1, col2 = st.columns(2)
    with col1:
        st.plotly_chart(gini_chart(df), use_container_width=True)
    with col2:
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
  if not _no_sim:
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

# ── Guide ─────────────────────────────────────────────────────────────────────

with tab_guide:
    st.subheader("How the simulation works")
    st.markdown(
        "Each run is a discrete-time agent-based market. Agents trade a single risky asset "
        "against a fundamental value that drifts randomly. Price moves in response to net "
        "demand. Agents can borrow to buy more than their cash allows; this leverage is "
        "the core mechanism behind Minsky's Financial Instability Hypothesis: stability "
        "encourages risk-taking, which creates fragility, which eventually causes a crash."
    )

    st.divider()

    # ── Sidebar parameters ────────────────────────────────────────────────────
    st.subheader("Sidebar parameters")

    with st.expander("General", expanded=True):
        st.markdown("""
**Steps**: how many time steps the simulation runs. More steps give leverage cycles more
time to develop. 500 is enough to see a crash in fragile configurations; 1000+ reveals
long-run wealth dynamics.

**Random seed**: fixes the random number generator so results are reproducible. Change
the seed to get a different draw from the same distribution, useful for checking whether
a crash is systematic or a fluke.
""")

    with st.expander("Agent Counts"):
        st.markdown("""
**Fundamental traders**: stabilising force. They buy when price is below fundamental
value and sell when above. More fundamental traders means stronger mean-reversion and
a harder-to-sustain bubble. Set to zero and the price will wander freely.

**Momentum traders**: destabilising force. They buy into rising prices and sell into
falling prices, amplifying trends. This is the key Minsky lever: above ~40% of total
agents, their feedback loop overwhelms the fundamentalists and crashes become likely.
Combined with high *Momentum sensitivity*, even 20-30% is enough.

**Noise traders**: submit random buy/sell orders each step. They add baseline volatility
and prevent the market locking into unrealistic deterministic cycles. Increasing noise
traders increases churn but doesn't systematically drive price in any direction.
""")

    with st.expander("Market Parameters"):
        st.markdown("""
**Price impact (α)**: how much the price moves per unit of net demand. Higher α means
individual orders have more market impact. The model uses:
`P_{t+1} = P_t × exp(α × net_demand / liquidity + noise)`.
At α = 0.001 the market is very deep; at α = 0.05 even small imbalances cause large
price swings.

**Liquidity (λ)**: scales the effective price impact. Lower liquidity amplifies each
order's effect (equivalent to raising α). Think of it as market depth: a thin market
(λ = 0.2) is far more volatile than a deep one (λ = 3.0).

**Price noise σ**: standard deviation of the random noise term added to the log price
each step. This represents background market microstructure noise. It also drives the
momentum signal: if noise is near zero, momentum agents see no trend and barely trade.
Values around 0.008-0.01 are needed for momentum to generate meaningful signals.

**Fundamental σ**: volatility of the true fundamental value process (GBM). A higher
value means fair value itself drifts more, making mispricing harder to detect and giving
fundamental traders a weaker signal. Kept small (0.003-0.005) so crashes are agent-driven,
not fundamental-driven.
""")

    with st.expander("Leverage & Margin"):
        st.markdown("""
**Max leverage (L_max)**: the leverage ratio at which a margin call is triggered.
Leverage is defined as `|position value| / wealth`. At L_max = 3.0, an agent whose
position is worth 3x their net wealth gets force-liquidated. Higher values allow more
risk-taking before the margin call fires, dramatically increasing crash severity when
it eventually does.

**Borrowing rate (per step)**: interest charged on outstanding debt each step as
`debt × (1 + r_b)`. At r_b = 0.0003 and 500 steps, an agent who borrows 1000 at step 0
owes ~1162 by step 500 (14% compounded). Higher rates drain leveraged agents faster,
sometimes triggering defaults even without a price crash.
""")

    with st.expander("Agent Parameters"):
        st.markdown("""
**Initial cash per agent**: starting capital for every agent. All agents begin with
this amount in cash and zero shares. Relative wealth diverges over time through trading
performance. This mainly sets the scale of the simulation; changing it doesn't alter
dynamics, only absolute price/wealth numbers.

**Fundamental sensitivity**: how aggressively fundamental traders act on mispricing.
Their target position is `sensitivity × (F-P)/F × max_position`. At 0.5 they trade
half their maximum position when price is 100% mispriced. Higher values make them trade
larger amounts and more quickly correct mispricings, strengthening the stabilising force.

**Momentum sensitivity**: the single most important parameter for generating crashes.
Momentum traders target `sensitivity × rolling_return × max_position` shares. At the
default of 0.3, rolling returns are small (~0.002) so agents barely trade. At 1.5-2.0
they build large positions aggressively, creating the feedback loop needed for a Minsky
cycle. **If you want crashes, set this to 1.5 or higher.**
""")

    st.divider()

    # ── Charts ────────────────────────────────────────────────────────────────
    st.subheader("What the charts show")

    with st.expander("Overview tab"):
        st.markdown("""
**Price vs Fundamental** (top-left sparkline): market price (blue) and true fair value
(green dashed). A persistent gap is a bubble or crash in progress. Red shaded regions
mark crash steps.

**Avg Leverage** (top-right): cross-sectional mean leverage across active agents.
Rising leverage during a calm period is the Minsky buildup phase.

**Rolling Volatility** (bottom-left): 20-step rolling standard deviation of returns.
Spikes mark periods of rapid price movement.

**Mispricing %** (bottom-right): `(P - F) / F × 100`. Positive = bubble, negative = crash.
""")

    with st.expander("Market tab"):
        st.markdown("""
**Market Price vs Fundamental Value**: full resolution time series. Red vertical bands
show crash steps (price fell >20% over 20 steps).

**Relative Mispricing**: signed percentage gap between price and fundamental. Fill
colour turns red above zero (overvalued) and green below (undervalued).

**Rolling Return Volatility**: annualised volatility measure. Calm-crisis cycles appear
as long flat periods followed by sharp spikes.

**Price Drawdown from Peak**: percentage decline from the running all-time high.
A -99% drawdown means the market essentially collapsed.

**Return Distribution**: histogram of all step-by-step returns. A Minsky collapse
produces fat left tails and a bimodal distribution.
""")

    with st.expander("Leverage & Risk tab"):
        st.markdown("""
**Leverage over Time**: average (filled area) and maximum (dotted) agent leverage.
Watch for the buildup-then-collapse pattern: leverage rises as momentum traders pile in,
then crashes suddenly when margin calls fire.

**Minsky Finance-State Composition**: stacked area chart showing what fraction of
active agents are in each regime each step:
- **Hedge** (green): leverage < 1.5, income covers all obligations
- **Speculative** (amber): leverage 1.5-3.0, must roll over debt
- **Ponzi** (red): leverage >= 3.0, must sell assets or borrow just to service interest

A rising Ponzi fraction is the canonical Minsky warning sign.

**Margin Calls & Defaults**: bar charts of forced liquidations (orange) and
bankruptcies (dark red) per step. Clusters of margin calls create the self-reinforcing
selling cascade.
""")

    with st.expander("Agent Performance tab"):
        st.markdown("""
**Average Wealth by Agent Type**: tracks mean wealth over time for fundamental,
momentum, noise, and RL agent groups. In stable markets, fundamental traders
outperform. In crash scenarios, whoever is short (or cash-heavy) at the right moment
wins, often noise traders.

**Wealth Inequality (Gini)**: 0 = perfectly equal, 1 = one agent holds everything.
Rises during booms as momentum traders profit, then can collapse during crashes as
leveraged agents are wiped out, paradoxically equalising wealth through shared ruin.
""")

    with st.expander("Experiments tab"):
        st.markdown("""
Pre-computed results from **Experiment 2**: sweeping the momentum trader fraction from
0% to 80% across 10 seeds each.

The four charts show how crash frequency, peak leverage, maximum drawdown, and final
wealth inequality all respond to increasing momentum trader dominance. The statistical
table (Spearman ρ and Kruskal-Wallis H) confirms all relationships are significant at
p < 10⁻⁹.

Key finding: there is a sharp phase transition at ~35–45% momentum traders. Below it,
markets are stable across all seeds. Above it, crashes are near-certain.
""")
