# Endogenous Financial Fragility in an Agent-Based Market: A Simulation Study of Minsky's Financial Instability Hypothesis

**Rafael Chong**
Agent-Based Modelling & Quantitative Finance — Independent Research Project

---

## Abstract

This report presents an agent-based simulation designed to test Minsky's Financial Instability Hypothesis (FIH): the proposition that financial stability is self-undermining because rational agents accumulate leverage during calm periods, creating systemic fragility that amplifies small shocks into crises. The simulation features four heterogeneous agent types (fundamental, momentum, noise, and a planned reinforcement learning agent), a log-linear price-impact mechanism, and an explicit balance sheet with debt, interest accrual, margin calls, and forced liquidation. Experiment 2 sweeps the share of momentum traders from 0% to 80% across 10 independent seeds per condition. Results show a sharp phase transition: markets with ≤30% momentum traders produce zero crashes across all seeds; above 40% momentum trader dominance, crash frequency rises dramatically (Spearman ρ = +0.894, p < 10⁻³⁰). Peak leverage and maximum drawdown scale monotonically with momentum share (ρ = +0.977 and ρ = −0.833, respectively, both p < 10⁻²³). These results confirm the Minsky mechanism as an emergent property of agent interaction — no crash is encoded in the model; fragility arises endogenously from individually rational trend-following behaviour.

---

## 1. Introduction

Hyman Minsky argued in *Stabilizing an Unstable Economy* (1986) that market economies are inherently prone to cycles of stability and crisis. His core claim was counterintuitive: prolonged calm is dangerous. When agents observe stability, they rationalise taking on more debt and risk; this demand drives asset prices above fundamental value; the inflated balance sheets become fragile; a small shock triggers forced selling; falling prices trigger more forced selling. The crisis is not caused by an external event — it is the endogenous product of the preceding boom.

Despite its descriptive power — Minsky's work gained mainstream attention after the 2008 financial crisis — the FIH is difficult to test empirically. The causal chain from stability to fragility to crash is confounded by concurrent policy changes, technological shocks, and global capital flows. Agent-based modelling (ABM) offers a controlled alternative: build a minimal model with the proposed mechanism, remove confounders by construction, and ask whether the predicted dynamics emerge.

This project builds such a model. The market begins in a controlled stable equilibrium — price at fundamental value, near-zero volatility, no debt — and evolves entirely from agent interactions. The research questions addressed in this report are:

1. Does increasing the share of momentum traders endogenously generate leverage buildup and market crashes?
2. Is there a critical threshold of momentum trader dominance above which market stability collapses?
3. How does momentum trader dominance affect wealth inequality (Gini coefficient) and risk-adjusted returns (Sharpe ratio)?

---

## 2. Model Description

### 2.1 Market Architecture

The market runs for T discrete time steps. Each step executes the following sequence:

1. Accrue interest on all outstanding agent debt: `d_{t+1} = d_t · (1 + r_b)`
2. Each active agent observes market state `(P_t, F_t, σ_t, m_t, L̄_t)` and submits a buy/sell order
3. Net demand is aggregated, normalised by active agent count, and passed to the price mechanism
4. Price and fundamental value update
5. Agents are marked to market; leverage is recalculated
6. Agents exceeding `L_max` receive margin calls and are force-liquidated
7. Forced-sell units buffer into the next step as additional sell demand
8. Metrics are recorded

The one-step forced-sell buffer (step 7) is a deliberate design choice: it creates the feedback delay that allows cascade propagation across time steps, modelling realistic broker execution lag.

### 2.2 Fundamental Value

Fundamental value follows a discrete-time approximation to Geometric Brownian Motion:

```
F_{t+1} = F_t · (1 + μ_F + ε_t),   ε_t ~ N(0, σ_F²)
```

Parameters: `F_0 = 100`, `μ_F = 0`, `σ_F = 0.005`. The zero drift ensures any long-run price deviation from fundamental is attributable to agent behaviour, not the drift of F itself.

### 2.3 Price Mechanism

The price mechanism is a log-linear impact model (Kyle 1985):

```
P_{t+1} = P_t · exp(clip(α · D̃_t / λ + η_t, −1.5, 1.5))
```

where `D̃_t = (buy − sell) / n_active` is the per-agent net demand, `α` is price impact, `λ` is market liquidity, and `η_t ~ N(0, σ_η²)` is baseline noise. The exponent is clipped to ±1.5 to prevent numerical overflow in extreme momentum cascades; this bound allows single-step price moves of up to ±78% — far exceeding any observed market crash — so it does not artificially constrain normal dynamics.

### 2.4 Agent Balance Sheet

Each agent `i` holds cash `c_i`, shares `q_i`, and debt `d_i`. Derived quantities:

```
Asset exposure:    A_i(P) = P · q_i
Wealth:            W_i(P) = c_i + P·q_i − d_i
Leverage:          L_i(P) = |P·q_i| / max(W_i, ε)
```

Debt arises when an agent buys more than their available cash; cash can go negative. Interest accrues on debt at rate `r_b` per step. An agent with `L_i ≥ L_max` receives a margin call and is forced to sell `liquidation_fraction` of their position.

**Minsky classification** (per-step, per-agent):

| Category | Condition | Interpretation |
|----------|-----------|----------------|
| Hedge | L < 1.5 | Income covers debt service |
| Speculative | 1.5 ≤ L < 3.0 | Income covers interest, not principal |
| Ponzi | L ≥ 3.0 | Must sell assets or roll debt to survive |

### 2.5 Agent Types

**Fundamental traders** are stabilising. They compute the mispricing signal `s = (F − P) / F` and target `desired_q = κ_F · s · q_max` shares. When price exceeds fundamental, they sell; when below, they buy. Sensitivity `κ_F = 0.5`.

**Momentum traders** are destabilising. They observe the rolling return `m_t = mean(r_{t−k:t})` over the lookback window and target `desired_q = κ_M · m_t · q_max`. They buy into rising prices and sell into falling prices, amplifying trends. Sensitivity `κ_M = 2.0` in the sweep experiments.

**Noise traders** submit random fractional orders at each step, providing background liquidity and preventing the market from locking into deterministic cycles.

**RL agents** (Phase 4, planned) will observe an 11-dimensional market state and output a target leverage exposure, learning via DQN with experience replay and a target network.

---

## 3. Experiment Design

### 3.1 Motivation

The Minsky mechanism requires an agent whose trading amplifies rather than dampens price movements. Momentum traders are the natural candidate: they buy when prices rise (pushing prices higher) and sell when prices fall (pushing prices lower). As their population share increases, the feedback loop strengthens.

### 3.2 Parameter Sweep

**Independent variable**: momentum trader fraction `f ∈ {0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8}`. Total agents N = 40 throughout. At fraction `f`, the composition is:

```
n_momentum  = round(40 · f)
n_fundamental = round((40 − n_momentum) · 0.6)
n_noise     = (40 − n_momentum) − n_fundamental
```

This preserves the 60/40 fundamental-to-noise ratio among non-momentum agents.

**Fixed parameters**: `T = 500`, `P_0 = F_0 = 100`, `α = 0.015`, `σ_η = 0.009`, `λ = 1.0`, `L_max = 3.0`, `r_b = 0.0003`, `initial_cash = 1000`.

**Replication**: 10 independent seeds per condition (90 total runs). Summary statistics (mean, SD) are computed per condition; statistical tests are run on the full 90-row sample.

### 3.3 Dependent Variables

- `n_crash_steps`: steps where price fell >20% over a 20-step window
- `max_avg_leverage`: peak of the cross-sectional mean leverage
- `max_drawdown_pct`: maximum peak-to-trough price drawdown (%)
- `total_margin_calls`: cumulative forced-liquidation events
- `total_defaults`: agents who exited the market via bankruptcy
- `final_gini`: Gini coefficient of agent wealth at step T
- `annualised_sharpe`: `mean(r) / std(r) · √252`

### 3.4 Statistical Methods

Two tests are applied to each dependent variable:

- **Kruskal-Wallis H test**: non-parametric one-way ANOVA testing whether the distribution of the outcome differs across the nine momentum-fraction conditions
- **Spearman rank correlation**: tests the monotone relationship between momentum fraction and the outcome across all 90 observations

For pairwise comparisons between specific conditions, Mann-Whitney U tests with Bonferroni correction are used (stored in `results/vary_momentum/pairwise_*.csv`).

---

## 4. Results

### 4.1 Phase Transition in Crash Frequency

**Table 1: Mean crash steps and peak leverage by momentum fraction (N = 10 seeds per condition)**

| Momentum % | Crash steps (mean ± SD) | Max avg leverage (mean ± SD) | Max drawdown (mean ± SD) |
|:----------:|:-----------------------:|:----------------------------:|:------------------------:|
| 0% | 0.0 ± 0.0 | 0.56 ± 0.05 | −17.6% ± 5.0% |
| 10% | 0.0 ± 0.0 | 0.55 ± 0.06 | −18.5% ± 3.2% |
| 20% | 0.0 ± 0.0 | 0.72 ± 0.06 | −20.3% ± 8.1% |
| 30% | 0.0 ± 0.0 | 0.82 ± 0.07 | −18.2% ± 4.7% |
| 40% | 0.6 ± 1.9 | 1.14 ± 0.17 | −22.7% ± 5.4% |
| 50% | 8.3 ± 7.3 | 1.53 ± 0.16 | −35.6% ± 17.2% |
| 60% | 66.4 ± 29.3 | 1.82 ± 0.07 | −100.0% ± 0.0% |
| 70% | 106.6 ± 15.0 | 1.96 ± 0.04 | −100.0% ± 0.0% |
| 80% | 116.9 ± 11.6 | 2.01 ± 0.07 | −100.0% ± 0.0% |

The transition is stark. At 30% momentum traders, no crashes occur across all 10 seeds. At 40%, the first crashes appear (in 1 of 10 seeds). At 50%, 9 of 10 seeds produce crashes. At 60% and above, price collapse to near-zero (−99.99% drawdown) occurs in every single run.

The SD of crash steps narrows at high fractions — high-momentum regimes are deterministically catastrophic regardless of seed.

### 4.2 Statistical Significance

**Table 2: Kruskal-Wallis and Spearman test results**

| Metric | Spearman ρ | p-value | Kruskal-Wallis H | p-value |
|--------|:----------:|:-------:|:----------------:|:-------:|
| Crash steps | +0.894 | 2.1 × 10⁻³² | 83.9 | 8.1 × 10⁻¹⁵ |
| Max avg leverage | **+0.977** | 1.5 × 10⁻⁶⁰ | 85.9 | 3.1 × 10⁻¹⁵ |
| Max drawdown | −0.833 | 2.7 × 10⁻²⁴ | 69.5 | 6.1 × 10⁻¹² |
| Total defaults | +0.797 | 5.5 × 10⁻²¹ | 86.0 | 3.0 × 10⁻¹⁵ |
| Final Gini | −0.860 | 1.8 × 10⁻²⁷ | 72.8 | 1.3 × 10⁻¹² |
| Annualised Sharpe | +0.612 | 1.4 × 10⁻¹⁰ | 66.5 | 2.4 × 10⁻¹¹ |
| Total margin calls | −0.830 | 5.4 × 10⁻²⁴ | 63.4 | 1.0 × 10⁻¹⁰ |

All tested metrics are statistically significant at p < 10⁻⁹. The strongest relationship is between momentum fraction and peak leverage (ρ = +0.977), consistent with Minsky's central claim that stability-seeking behaviour (trend-following in rising markets) is the proximate cause of leverage accumulation.

### 4.3 Wealth Inequality

The Gini coefficient — a measure of wealth inequality — decreases with momentum fraction (ρ = −0.860). This is initially counterintuitive: one might expect high-leverage crashes to redistribute wealth upward.

The explanation is a race-to-the-bottom dynamic. At 0–30% momentum, fundamental traders accumulate wealth steadily by exploiting mispricing; wealth diverges over T = 500 steps (Gini ≈ 0.50–0.53). At 60–80% momentum, crashes are so frequent and severe that the market periodically wipes out all agents — momentum and fundamental alike — resetting wealth towards equality. The surviving agents (those who happened to be short or cash-heavy at crash moments) briefly hold high wealth, but subsequent crashes eliminate them too. Final Gini at 80% momentum is 0.14, indicating near-equal (universally low) wealth.

This result highlights that the Gini coefficient alone is insufficient to distinguish between prosperous equality and catastrophic equality.

### 4.4 Sharpe Ratio Anomaly

Annualised Sharpe increases with momentum fraction (ρ = +0.612), reaching 7.94 at 80% momentum. This is an artefact of how total-collapse-and-recovery dynamics interact with the Sharpe formula.

At 60%+ momentum, price regularly crashes to near-zero and then recovers as the crash clears the market (all leveraged agents are liquidated, removing sell pressure). The recovery constitutes an extremely large positive return from a very low base. The distribution of returns is not approximately normal — it is bimodal, with many small returns during calm periods and occasional large rebounds post-crash. The Sharpe ratio, which assumes normality, is inflated by these rebounds. This is a known limitation: Sharpe is not a reliable risk measure for return distributions with fat tails and bimodal structure.

### 4.5 Margin Calls vs Defaults

Total margin calls is negatively correlated with momentum fraction (ρ = −0.830), while total defaults is positively correlated (ρ = +0.797). These are consistent: at high momentum fractions, price collapses are so swift and severe that agents skip the margin call stage and go directly to default (wealth drops below zero before the margin call mechanism can trigger a partial liquidation). The margin call system, designed to manage risk, is overwhelmed by the speed of the cascade.

---

## 5. Discussion

### 5.1 Confirmation of the Minsky Mechanism

The central finding — that financial instability emerges endogenously from agent-level momentum trading without any exogenous crash mechanism — directly supports the FIH. The model adds a specific, testable claim to Minsky's narrative: the system undergoes a sharp phase transition at approximately 35–45% momentum trader dominance. Below this threshold, fundamental traders can absorb momentum-driven mispricings; above it, the feedback loop overpowers stabilising forces.

This is consistent with empirical work documenting that momentum strategies constituted roughly 30–50% of hedge fund strategies in the period leading up to the 2008 crisis (Brunnermeier & Nagel 2004; Khandani & Lo 2007).

### 5.2 The Stabilising Role of Fundamental Traders

The 0%–30% stability zone is sustained entirely by fundamental traders counteracting momentum-driven mispricings. When price rises above fundamental value, fundamental traders sell, dampening the move. Their effectiveness depends on the ratio of their capital to the momentum traders' capital — at 40%+ momentum trader dominance, the fundamental traders (now fewer in number) lack the aggregate buying/selling power to stabilise the market.

This reflects Shleifer & Vishny's (1997) "limits to arbitrage": arbitrageurs (fundamental traders) are finite in capital and can be overwhelmed by noise/trend traders.

### 5.3 Endogeneity of the Crash

Importantly, the crash is not triggered by any exogenous shock in the model. The simulation has no "crisis event" parameter, no sudden fundamental value collapse, and no external intervention. The crash emerges purely from the balance-sheet dynamics of leveraged momentum traders: rising prices → higher paper wealth → higher safe-to-borrow amount → more buying → higher prices → eventually unsustainable valuations → reversal → margin calls → forced selling → cascade.

This endogeneity is the key theoretical contribution of the FIH and is reproduced here from first principles.

### 5.4 Limitations

| Limitation | Impact | Mitigation |
|------------|--------|------------|
| No persistent agent identity across crashes | Agents default but the population stays fixed (dead agents could be replaced) | Affects long-run wealth dynamics but not crash mechanics |
| Symmetric leverage cap (L_max = 3) | Real markets have heterogeneous margin requirements | Future: agent-type-specific leverage limits |
| No contagion / interbank linkages | Real crises spread through counterparty exposure | Future: balance sheet linkages between agents |
| Momentum signal = rolling average return | Real momentum traders use more sophisticated signals (RSI, MACD, crossovers) | Captures the qualitative mechanism; exact signal form matters for calibration |
| Sharpe anomaly at extreme fractions | Inflated Sharpe in high-momentum regimes is a measure artefact | Use Sortino or Calmar ratio for non-normal return distributions |
| Ponzi fraction always zero | Leverage cap L_max = 3.0 equals the Ponzi threshold — agents are liquidated before entering Ponzi state | Lower Ponzi threshold (e.g. 2.5) or raise L_max to 4.0 |

---

## 6. Next Steps

**Phase 4 — Reinforcement Learning Environment**

The RL agent will observe an 11-dimensional state:

```
s_t = (P_t, F_t, m_t, σ_t, L_i, W_i, C_i, Q_i, D_i, L̄_t, f_ponzi_t)
```

and output a target leverage exposure from a discrete action space {−1.0, −0.5, 0.0, 0.5, 1.0, 1.5} × W_i / P_t. Three reward functions will be compared:

- **Profit-only**: `r_t = ΔW_i / W_i`
- **Risk-adjusted**: `r_t = ΔW_i / W_i − λ · L_i²`
- **System-aware**: `r_t = ΔW_i / W_i − λ · f_ponzi_t`

The core research question: does a profit-maximising RL agent learn to exploit the leverage cycle (buy during calm, unwind before crash) in a way that amplifies or dampens systemic risk? Do risk-adjusted and system-aware rewards produce different emergent behaviour?

**Phase 6 — Regulatory Interventions**

Test counterfactual policies:
- Hard leverage caps (L_max ∈ {1.5, 2.0, 2.5})
- Mandatory countercyclical margin buffers (increase L_max requirements during high-leverage periods)
- Transaction tax (Tobin tax) reducing momentum profitability
- Circuit breakers (halt trading for k steps after a crash signal)

Measure the trade-off between crash prevention and liquidity reduction.

---

## 7. Conclusion

This simulation demonstrates that Minsky's Financial Instability Hypothesis is mechanistically credible: a market populated even partly by rational trend-following agents will endogenously generate leverage cycles and crashes without any external shock. The critical finding is a phase transition at approximately 35–45% momentum trader dominance — a quantitative, testable prediction that could guide empirical research and regulatory design.

The model also highlights two important caveats for interpreting market data. First, the Gini coefficient confounds prosperous and catastrophic equality: high-crash regimes can show low inequality because everyone is equally ruined. Second, the Sharpe ratio is unreliable in markets with bimodal return distributions — precisely the conditions that Minsky's theory predicts will arise. Both points argue for a broader suite of systemic risk metrics when evaluating market health.

---

## References

Brunnermeier, M. K. & Nagel, S. (2004). Hedge funds and the technology bubble. *Journal of Finance*, 59(5), 2013–2040.

Khandani, A. E. & Lo, A. W. (2007). What happened to the quants in August 2007? *Journal of Investment Management*, 5(4), 5–54.

Kyle, A. S. (1985). Continuous auctions and insider trading. *Econometrica*, 53(6), 1315–1335.

Minsky, H. P. (1986). *Stabilizing an Unstable Economy*. Yale University Press.

Minsky, H. P. (1992). The financial instability hypothesis. Levy Economics Institute Working Paper No. 74.

Mnih, V. et al. (2015). Human-level control through deep reinforcement learning. *Nature*, 518, 529–533.

Shleifer, A. & Vishny, R. (1997). The limits of arbitrage. *Journal of Finance*, 52(1), 35–55.

Sutton, R. S. & Barto, A. G. (2018). *Reinforcement Learning: An Introduction* (2nd ed.). MIT Press.
