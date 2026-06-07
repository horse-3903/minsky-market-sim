# Theory: From Stability to Fragility

A mathematical and conceptual reference for the Minsky Market Simulation.

---

## Contents

1. [Minsky's Financial Instability Hypothesis](#1-minskys-financial-instability-hypothesis)
2. [Agent-Based Modelling in Financial Markets](#2-agent-based-modelling-in-financial-markets)
3. [Fundamental Value Process](#3-fundamental-value-process)
4. [Price Formation Mechanism](#4-price-formation-mechanism)
5. [Agent Balance Sheets](#5-agent-balance-sheets)
6. [Leverage, Debt, and Margin Calls](#6-leverage-debt-and-margin-calls)
7. [Agent Decision Models](#7-agent-decision-models)
8. [Reinforcement Learning Formulation](#8-reinforcement-learning-formulation)
9. [Market Instability Metrics](#9-market-instability-metrics)
10. [The Minsky Mechanism End-to-End](#10-the-minsky-mechanism-end-to-end)
11. [Limitations and Modelling Assumptions](#11-limitations-and-modelling-assumptions)
12. [Further Reading](#12-further-reading)

---

## 1. Minsky's Financial Instability Hypothesis

### 1.1 Background

Hyman Minsky (1919–1996) was an American economist whose work, largely ignored during his lifetime, became influential after the 2008 global financial crisis. His central claim, the **Financial Instability Hypothesis (FIH)**, inverts the standard equilibrium view of financial markets: rather than markets being self-correcting, **stability itself sows the seeds of instability**.

The core insight is that during periods of calm, agents systematically underestimate risk and increase leverage. This collective shift in risk-taking transforms a stable system into a fragile one. When a shock eventually arrives — even a small one — the over-leveraged system is unable to absorb it, and cascading forced selling produces a crash far larger than the original shock.

### 1.2 The Three Financing Regimes

Minsky classified borrowers into three regimes based on their ability to service debt from income (or, in a financial market context, from cash flows and asset values):

**Hedge finance**

The borrower can cover both interest payments and principal repayment from expected cash flows. Even a moderate fall in asset prices or income does not threaten solvency.

```
Cash flow >= Interest + Principal repayments
```

In the simulation, this is proxied by low leverage:

```
L < L_hedge   (default: 1.5)
```

**Speculative finance**

The borrower can cover interest payments but cannot repay principal without rolling over (refinancing) the debt. Solvency depends on continued access to credit and stable asset prices.

```
Cash flow >= Interest,  but  Cash flow < Principal repayments
```

Proxied by moderate leverage:

```
L_hedge <= L < L_speculative   (default: 1.5 <= L < 3.0)
```

**Ponzi finance**

The borrower cannot even cover interest payments from cash flows. Solvency depends entirely on rising asset prices — the agent must sell assets at a gain or borrow further just to service existing debt.

```
Cash flow < Interest
```

Proxied by high leverage:

```
L >= L_ponzi   (default: 3.0)
```

### 1.3 Stability Breeds Instability

The critical dynamic is the **endogenous transition** between regimes. In a calm market:

- Observed volatility is low
- Recent returns have been positive
- Defaults are rare
- Lenders are willing to extend credit at low rates

Rational (or boundedly rational) agents respond by increasing leverage to capture higher returns. Individually, each agent's decision is defensible. Collectively, the shift from hedge to speculative to Ponzi finance makes the entire system fragile.

The feedback loop:

```
Low volatility
    → agents increase leverage
    → leveraged buying pushes prices up
    → rising prices validate risk-taking
    → measured volatility stays low (prices trend smoothly upward)
    → agents increase leverage further
    → ...
```

### 1.4 The Minsky Moment

A **Minsky moment** is the point at which the system tips from fragility into crisis. A small negative shock — insufficient to cause a crisis in a low-leverage environment — triggers margin calls for Ponzi-financed agents. Forced selling drives prices down. Falling prices trigger margin calls for speculative agents. Further forced selling drives prices down further. The cascade continues until leverage across the system is dramatically reduced, often overshooting the fundamental value downward.

The key property: the crash is **disproportionate to the trigger**. A 2% price fall from a shock can produce a 30% crash through the forced deleveraging spiral.

---

## 2. Agent-Based Modelling in Financial Markets

### 2.1 Why Agent-Based Models

Standard financial models (CAPM, Black-Scholes, rational expectations equilibrium) assume:

- Homogeneous, fully rational agents
- Equilibrium pricing
- Normally distributed returns
- No feedback between agent actions and prices

These assumptions are mathematically convenient but empirically problematic. Real markets exhibit:

- Fat-tailed return distributions (excess kurtosis)
- Volatility clustering (calm followed by turbulent periods)
- Asset price bubbles and crashes
- Heterogeneous beliefs and strategies

**Agent-based models (ABMs)** replace the representative rational agent with a population of heterogeneous, adaptive agents. Each agent follows a simple local rule. Macro-level phenomena — bubbles, crashes, volatility clustering — **emerge** from micro-level interactions rather than being imposed by assumption.

### 2.2 Emergence

In complex systems, emergence refers to properties of the whole that cannot be predicted from the properties of any individual part. In the simulation:

- No single agent is programmed to cause a crash
- The crash emerges from the interaction of individual leverage decisions, price impact, and the margin call mechanism
- This is what makes the model a genuine test of Minsky's hypothesis: instability is not hard-coded, it must arise endogenously

### 2.3 Heterogeneous Agents

The simulation uses four agent archetypes, each embodying a different belief about how prices work:

| Agent | Belief | Market role |
|-------|--------|-------------|
| Fundamental | Price reverts to intrinsic value | Stabilising |
| Momentum | Trends persist | Destabilising |
| Noise | No systematic view | Provides liquidity and randomness |
| RL | Learns from reward signal | Adaptive, potentially destabilising |

The interaction between stabilising and destabilising agents determines whether prices track fundamental value or deviate into bubbles.

---

## 3. Fundamental Value Process

The **fundamental value** $F_t$ represents the true intrinsic worth of the asset, evolving independently of market prices. It follows a discrete-time geometric Brownian motion:

$$
F_{t+1} = F_t \left(1 + \mu_F + \varepsilon_t\right), \quad \varepsilon_t \sim \mathcal{N}(0, \sigma_F^2)
$$

where:

- $\mu_F$ is the drift (set to 0 in baseline experiments — no secular trend)
- $\sigma_F$ is the fundamental volatility (kept small, e.g. 0.003 per step)
- $\varepsilon_t$ are i.i.d. Gaussian shocks

This is the log-normal process familiar from the Black-Scholes model. The key design choice is keeping $\sigma_F$ small so that large price movements cannot be explained by fundamental shocks alone — any crash must be endogenously generated.

**Why geometric rather than arithmetic?** Geometric Brownian motion ensures $F_t > 0$ for all $t$ (prices cannot go negative) and produces percentage returns that are i.i.d., consistent with the efficient market hypothesis for fundamental value.

In continuous time, the equivalent process is:

$$
dF = \mu_F F \, dt + \sigma_F F \, dW_t
$$

where $W_t$ is a standard Wiener process.

---

## 4. Price Formation Mechanism

### 4.1 Net Demand

At each time step, agents submit buy and sell orders. Aggregating across all $N$ active agents:

$$
D_t^+ = \sum_{i=1}^{N} b_{i,t}, \qquad D_t^- = \sum_{i=1}^{N} s_{i,t}
$$

$$
D_t = D_t^+ - D_t^-
$$

where $b_{i,t} \geq 0$ is agent $i$'s buy quantity and $s_{i,t} \geq 0$ is their sell quantity at step $t$.

To prevent price explosions when many agents submit large simultaneous orders, demand is normalised by the number of active agents:

$$
\tilde{D}_t = \frac{D_t}{N_{\text{active}}}
$$

This is equivalent to measuring price impact in terms of the **average per-agent order**, rather than total market demand.

### 4.2 Price Impact Function

The price updates via a log-linear impact function:

$$
P_{t+1} = P_t \cdot \exp\!\left(\frac{\alpha \tilde{D}_t}{\lambda} + \eta_t\right), \quad \eta_t \sim \mathcal{N}(0, \sigma_\eta^2)
$$

where:

- $\alpha$ is the price impact coefficient (how much a unit of normalised demand moves the price)
- $\lambda$ is the liquidity parameter (higher liquidity dampens impact)
- $\eta_t$ is i.i.d. Gaussian market noise

**Why log-linear?** The exponential form ensures $P_t > 0$ always (no negative prices). It is also consistent with the Kyle (1985) framework for price impact in the presence of informed traders, and with the empirical finding that log returns are approximately normally distributed in liquid markets.

The effective price impact per unit of normalised demand is:

$$
\text{impact} = \frac{\alpha}{\lambda}
$$

Lower liquidity ($\lambda \to 0$) produces larger price swings for the same demand imbalance — a key feature for modelling crisis dynamics where liquidity evaporates.

### 4.3 Forced-Sell Feedback

When margin calls fire in step $t$, the resulting forced sell volume $\Phi_t$ is **buffered** and injected as additional sell demand in step $t+1$:

$$
D_{t+1} \leftarrow D_{t+1} - \Phi_t
$$

This one-step delay is intentional. It:

1. Prevents simultaneous price update and liquidation (avoids logical circularity)
2. Models the realistic lag between a broker issuing a margin call and the resulting market order being executed
3. Allows the price decline from forced selling to propagate and trigger further margin calls in subsequent steps — the mechanism behind the Minsky cascade

---

## 5. Agent Balance Sheets

Each agent $i$ holds a complete balance sheet at every time step $t$:

| Variable | Symbol | Description |
|----------|--------|-------------|
| Cash | $c_{i,t}$ | Undeployed cash holdings |
| Shares | $q_{i,t}$ | Units of risky asset held |
| Debt | $d_{i,t}$ | Outstanding borrowed principal |
| Active | $\mathbf{1}_{i,t}$ | 0 if agent has defaulted |

### 5.1 Derived Quantities

**Asset exposure** — the gross value of the risky asset position:

$$
E_{i,t} = P_t \cdot q_{i,t}
$$

**Wealth (equity)** — the net asset value, or what the agent would have after paying all debts:

$$
W_{i,t} = c_{i,t} + P_t q_{i,t} - d_{i,t}
$$

**Leverage** — the ratio of gross asset exposure to equity:

$$
L_{i,t} = \frac{|E_{i,t}|}{\max(W_{i,t},\, \varepsilon)}
$$

where $\varepsilon > 0$ is a small floor that prevents division by zero. Note that $L = 1$ means the agent is fully invested with no borrowing; $L = 2$ means the agent has borrowed an amount equal to their equity; $L = 0$ means the agent holds only cash.

### 5.2 Trade Mechanics

**Buying** $\Delta q > 0$ shares at price $P_t$:

$$
\text{cost} = \Delta q \cdot P_t
$$

If $\text{cost} \leq c_{i,t}$: deduct from cash.

$$
c_{i,t+1} = c_{i,t} - \text{cost}, \quad d_{i,t+1} = d_{i,t}
$$

If $\text{cost} > c_{i,t}$: use all cash, borrow the shortfall.

$$
c_{i,t+1} = 0, \quad d_{i,t+1} = d_{i,t} + (\text{cost} - c_{i,t})
$$

In both cases: $q_{i,t+1} = q_{i,t} + \Delta q$

**Selling** $\Delta q > 0$ shares: proceeds first repay debt.

$$
\text{proceeds} = \Delta q \cdot P_t
$$

$$
\text{repay} = \min(d_{i,t},\, \text{proceeds})
$$

$$
d_{i,t+1} = d_{i,t} - \text{repay}, \quad c_{i,t+1} = c_{i,t} + (\text{proceeds} - \text{repay})
$$

$$
q_{i,t+1} = q_{i,t} - \Delta q
$$

Note that selling shares does not increase wealth directly — proceeds go to debt repayment first. This is a key constraint: when prices fall, leveraged agents cannot easily restore their balance sheet by selling, because the proceeds are absorbed by debt.

---

## 6. Leverage, Debt, and Margin Calls

### 6.1 Interest Accrual

At the start of each time step, before any orders are placed, debt compounds at the per-step borrowing rate $r_b$:

$$
d_{i,t+1} = d_{i,t} \cdot (1 + r_b)
$$

The annualised equivalent at typical step frequencies (e.g. 500 steps per year) is:

$$
r_b^{\text{annual}} \approx (1 + r_b)^{500} - 1
$$

For $r_b = 0.0003$, this gives approximately 16% per year — a high but not unrealistic unsecured borrowing rate that penalises prolonged heavy leverage.

Interest accrual is the mechanism by which **time harms leveraged positions**. Even if prices stay flat, debt grows. An agent who borrows heavily and holds a stagnant position will see wealth slowly eroded by interest, eventually triggering a margin call even without any price movement.

### 6.2 Leverage Constraint

The simulation enforces a hard leverage cap $L_{\max}$. At the end of each step, after the price has updated, every agent's leverage is checked:

$$
\text{if } L_{i,t} > L_{\max}: \text{ margin call}
$$

### 6.3 Forced Liquidation

When a margin call fires, the agent must sell enough shares to bring leverage back to $L_{\max}$.

Let $W = W_{i,t}$ (current wealth) and $q = q_{i,t}$ (current shares). The target share count $q^*$ that achieves $L = L_{\max}$ is found by solving:

$$
L_{\max} = \frac{P_t \cdot q^*}{W_{i,t}}
$$

$$
q^* = \frac{L_{\max} \cdot W_{i,t}}{P_t}
$$

Shares to sell: $\Delta q = q - q^*$

Note that $W_{i,t}$ does not change when proceeds exactly repay debt (selling shares and repaying debt keeps wealth constant):

$$
W \leftarrow c + P(q - \Delta q) - (d - P \Delta q) = c + Pq - d = W
$$

This is a crucial insight: **selling to reduce leverage does not directly increase wealth** — it just reduces the size of both sides of the balance sheet. Wealth only recovers if prices subsequently rise.

In practice, a minimum liquidation fraction $\phi$ is also enforced so that each margin call makes meaningful progress:

$$
\Delta q = \max\!\left(q - q^*,\; \phi \cdot q\right)
$$

**Default** occurs when, even after full liquidation ($q = 0$), wealth is still non-positive:

$$
W_{i,t} = c_{i,t} - d_{i,t} \leq 0 \implies \text{default}
$$

A defaulted agent is removed from the market (`is_active = False`).

### 6.4 Why Forced Selling Creates a Cascade

The feedback loop that creates Minsky-style crashes operates through three channels:

1. **Price channel**: forced selling increases $D_t^-$, pushing $P_{t+1}$ down
2. **Wealth channel**: falling prices reduce $W_{i,t} = c + P q - d$, increasing leverage $L = Pq / W$ for all leveraged agents
3. **Contagion channel**: rising leverage triggers margin calls for previously safe agents, creating more forced selling

Formally, if a price drop $\Delta P < 0$ occurs, the change in leverage for agent $i$ is:

$$
\Delta L_i \approx \frac{q_i}{W_i} \Delta P - \frac{P q_i}{W_i^2} \cdot q_i \Delta P = \frac{q_i \Delta P}{W_i} \left(1 - L_i\right)
$$

For $L_i > 1$ (leveraged), this is negative times a negative — leverage **increases** when prices fall. This is the mathematical core of the Minsky instability mechanism.

---

## 7. Agent Decision Models

### 7.1 Fundamental Trader

The fundamental trader believes prices are mean-reverting toward intrinsic value. The mispricing signal is:

$$
s_{i,t} = \frac{F_t - P_t}{F_t}
$$

Positive $s$ means the asset is undervalued; negative means overvalued. The desired share position is:

$$
q_{i,t}^* = \kappa_F \cdot s_{i,t} \cdot q_{\max}
$$

where $\kappa_F$ is the sensitivity parameter and $q_{\max}$ is the maximum position size. The agent submits an order to move from their current position $q_{i,t}$ toward $q_{i,t}^*$:

$$
\Delta q = q_{i,t}^* - q_{i,t}
$$

$\Delta q > 0$: buy order. $\Delta q < 0$: sell order (capped at $q_{i,t}$, no short-selling).

Fundamental traders are **stabilising**: they provide a force pulling prices toward $F_t$. Without them, prices have no anchor and can drift arbitrarily far from intrinsic value.

### 7.2 Momentum Trader

The momentum trader extrapolates recent price trends. The signal is the rolling return over the last $k$ steps:

$$
m_t = \frac{P_t - P_{t-k}}{P_{t-k}}
$$

Desired position:

$$
q_{i,t}^* = \kappa_M \cdot m_t \cdot q_{\max}
$$

Positive momentum → buy. Negative momentum → sell.

Momentum traders are **destabilising**: they amplify price moves. A price rise increases $m_t$, generating more buying, which pushes prices higher, which increases $m_t$ further. This positive feedback loop is the mechanism behind bubble formation.

The tension between fundamental and momentum traders is what makes the simulation interesting. With only fundamental traders, prices always converge to $F_t$. With only momentum traders, prices diverge to infinity or zero. The mixture produces realistic boom-bust dynamics.

### 7.3 Noise Trader

The noise trader has no systematic belief. Each step, they draw an action uniformly from $\{\text{buy, sell, hold}\}$ with order size drawn uniformly from $[0, s_{\max}]$.

Noise traders serve two roles:

1. **Prevent trivial equilibrium**: without noise, a market of equal fundamental and momentum traders might simply sit at the equilibrium price indefinitely
2. **Realistic market activity**: real markets have many participants whose trades are not fully explained by any systematic strategy — execution flows, index rebalancing, liquidity needs

### 7.4 Market Maker

The market maker acts as a liquidity provider, leaning against price deviations similarly to the fundamental trader but with a smaller inventory and dedicated to reducing bid-ask spreads rather than taking directional bets. The signal and desired position are the same form as the fundamental trader, parameterised separately.

---

## 8. Reinforcement Learning Formulation

### 8.1 The Markov Decision Process

The RL agent's interaction with the market is formalised as a **Markov Decision Process (MDP)** $(\mathcal{S}, \mathcal{A}, \mathcal{T}, \mathcal{R}, \gamma)$:

- $\mathcal{S}$: state space — a vector of market and agent observables
- $\mathcal{A}$: action space — discrete set of target exposure levels
- $\mathcal{T}: \mathcal{S} \times \mathcal{A} \to \Delta(\mathcal{S})$: stochastic transition function (governed by the simulation)
- $\mathcal{R}: \mathcal{S} \times \mathcal{A} \to \mathbb{R}$: reward function
- $\gamma \in [0,1)$: discount factor

The Markov property requires that the transition distribution depends only on the current state, not the full history. This is approximate in the simulation — the true state includes the positions of all other agents — but the observation vector is designed to capture the most relevant market statistics.

### 8.2 State (Observation) Space

The observation vector $\mathbf{o}_t \in \mathbb{R}^{11}$ at step $t$ is:

$$
\mathbf{o}_t = \left[\frac{P_t}{F_t},\; r_t,\; \sigma_t^{\text{roll}},\; m_t,\; c_{i,t},\; q_{i,t},\; d_{i,t},\; L_{i,t},\; \bar{L}_t,\; n_t^{\text{MC}},\; \text{DD}_t \right]
$$

| Feature | Description |
|---------|-------------|
| $P_t / F_t$ | Price-to-fundamental ratio (mispricing signal) |
| $r_t$ | Last-step log return |
| $\sigma_t^{\text{roll}}$ | Rolling return standard deviation (20-step window) |
| $m_t$ | Rolling momentum (10-step price change) |
| $c_{i,t}$ | Agent's own cash |
| $q_{i,t}$ | Agent's own share holdings |
| $d_{i,t}$ | Agent's own debt |
| $L_{i,t}$ | Agent's own leverage |
| $\bar{L}_t$ | Market-wide average leverage |
| $n_t^{\text{MC}}$ | Number of margin calls in recent steps |
| $\text{DD}_t$ | Current drawdown from peak |

### 8.3 Action Space

The agent chooses a **target exposure level** expressed as a multiple of current wealth:

| Action | Target leverage |
|--------|----------------|
| 0 | 0× (full cash) |
| 1 | 0.5× |
| 2 | 1× (unleveraged long) |
| 3 | 1.5× |
| 4 | 2× |
| 5 | 3× |

This design is preferable to raw buy/sell quantities because:
1. It directly maps to the Minsky leverage dimension
2. The agent can learn leverage strategy rather than trade-sizing heuristics
3. Interpretation is straightforward: action 5 is Ponzi-range leverage, action 2 is hedge-range

### 8.4 Reward Functions

Three reward functions are compared across experiments.

**Profit-only reward**

$$
R_t^{\text{profit}} = W_{i,t+1} - W_{i,t}
$$

or equivalently as a percentage change:

$$
R_t^{\text{profit}} = \frac{W_{i,t+1} - W_{i,t}}{W_{i,t}}
$$

This is the standard financial objective. The hypothesis is that an agent trained purely on this signal will learn to increase leverage during calm periods — rational at the individual level, destabilising at the system level.

**Risk-adjusted reward**

$$
R_t^{\text{risk}} = \Delta W_t - \lambda \cdot \text{DD}_t - \mu \cdot L_{i,t}
$$

where:
- $\text{DD}_t = \frac{P_t - \max_{s \leq t} P_s}{\max_{s \leq t} P_s} \leq 0$ is the drawdown (always non-positive)
- $\lambda > 0$ penalises exposure to large drawdowns
- $\mu > 0$ penalises high leverage directly

The $\lambda \cdot \text{DD}_t$ term punishes the agent for holding through a price decline. The $\mu \cdot L_{i,t}$ term directly penalises leverage, creating an incentive to stay in hedge-finance territory.

**System-aware reward**

$$
R_t^{\text{sys}} = \Delta W_t - \lambda \cdot \text{DD}_t - \mu \cdot L_{i,t} - \gamma \cdot \bar{L}_t
$$

The additional $\gamma \cdot \bar{L}_t$ term penalises the agent when aggregate market leverage is high, regardless of their own leverage. This tests whether an agent can be incentivised to internalise **systemic risk** — the negative externality their leverage imposes on the system.

### 8.5 Q-Learning

For discrete state-action spaces, the value of taking action $a$ in state $s$ and following policy $\pi$ thereafter is the Q-function:

$$
Q^\pi(s, a) = \mathbb{E}_\pi \left[\sum_{k=0}^{\infty} \gamma^k R_{t+k} \,\Big|\, s_t = s,\, a_t = a\right]
$$

The optimal Q-function satisfies the **Bellman optimality equation**:

$$
Q^*(s, a) = \mathbb{E}\left[R_t + \gamma \max_{a'} Q^*(s_{t+1}, a') \,\Big|\, s_t = s,\, a_t = a\right]
$$

Tabular Q-learning iteratively approximates $Q^*$ using the update rule:

$$
Q(s_t, a_t) \leftarrow Q(s_t, a_t) + \alpha \underbrace{\left[R_t + \gamma \max_{a'} Q(s_{t+1}, a') - Q(s_t, a_t)\right]}_{\text{TD error}}
$$

where $\alpha \in (0,1]$ is the learning rate. The **temporal difference (TD) error** is the discrepancy between the current estimate and the Bellman target. The update moves the estimate toward the target at rate $\alpha$.

For the continuous observation space in this simulation, tabular Q-learning requires discretisation of $\mathcal{S}$, which suffers from the curse of dimensionality with 11 features. DQN addresses this.

### 8.6 Deep Q-Network (DQN)

DQN (Mnih et al., 2015) approximates the Q-function with a neural network $Q_\theta(s, a)$ parameterised by weights $\theta$. The loss function is:

$$
\mathcal{L}(\theta) = \mathbb{E}_{(s,a,r,s') \sim \mathcal{D}} \left[\left(y - Q_\theta(s, a)\right)^2\right]
$$

where the target is:

$$
y = r + \gamma \max_{a'} Q_{\theta^-}(s', a')
$$

Two stabilisation techniques are essential:

**Experience replay**: transitions $(s_t, a_t, r_t, s_{t+1})$ are stored in a replay buffer $\mathcal{D}$ and sampled uniformly at training time. This breaks temporal correlations in the gradient updates and improves sample efficiency.

**Target network**: a separate network $Q_{\theta^-}$ with frozen weights is used to compute the target $y$. The target weights are updated every $C$ steps: $\theta^- \leftarrow \theta$. This prevents the instability that arises when both the prediction and the target depend on the same rapidly changing network.

### 8.7 Exploration

During training, the agent uses an $\varepsilon$-greedy policy:

$$
a_t = \begin{cases} \text{random action} & \text{with probability } \varepsilon \\ \arg\max_a Q_\theta(s_t, a) & \text{with probability } 1 - \varepsilon \end{cases}
$$

$\varepsilon$ is annealed from 1.0 (fully random) to a small floor (e.g. 0.05) over the course of training. Early exploration builds diverse experience; later exploitation refines the learned policy.

---

## 9. Market Instability Metrics

### 9.1 Rolling Volatility

The standard deviation of log returns over a window of $W$ steps:

$$
\sigma_t = \sqrt{\frac{1}{W-1} \sum_{k=0}^{W-1} (r_{t-k} - \bar{r})^2}
$$

where $r_t = \ln(P_t / P_{t-1})$ and $\bar{r}$ is the sample mean over the window.

Minsky's hypothesis predicts a characteristic **volatility pattern**: low and falling during the leverage build-up phase, then spiking sharply at the Minsky moment.

### 9.2 Mispricing

The relative deviation of market price from fundamental value:

$$
M_t = \frac{P_t - F_t}{F_t}
$$

$M_t > 0$: overvaluation (bubble territory). $M_t < 0$: undervaluation (post-crash overshoot).

### 9.3 Bubble and Crash Indicators

**Bubble**: price has deviated above fundamental value by more than threshold $\theta_b$:

$$
\text{Bubble}_t = \mathbf{1}\left[M_t > \theta_b\right]
$$

**Crash**: price has fallen by more than threshold $\theta_c$ over a window of $k$ steps:

$$
\text{Crash}_t = \mathbf{1}\left[\frac{P_t - P_{t-k}}{P_{t-k}} < -\theta_c\right]
$$

### 9.4 Maximum Drawdown

The largest peak-to-trough price decline observed up to step $t$:

$$
\text{MDD}_t = \min_{s \leq t} \frac{P_s - \max_{u \leq s} P_u}{\max_{u \leq s} P_u}
$$

Maximum drawdown is a standard risk metric in both academic finance and portfolio management. In the simulation, it captures the severity of the crash phase.

### 9.5 Sharpe Ratio

The annualised risk-adjusted return for agent $i$:

$$
\text{SR}_i = \frac{\bar{r}_i}{\sigma_{r_i}} \cdot \sqrt{N_{\text{steps per year}}}
$$

where $\bar{r}_i$ and $\sigma_{r_i}$ are the mean and standard deviation of the agent's per-step wealth returns. The Sharpe ratio penalises strategies that achieve high returns only by accepting high volatility — exactly what profit-only RL agents tend to do.

### 9.6 Gini Coefficient

The Gini coefficient $G \in [0,1]$ measures wealth inequality across agents. For a vector of non-negative wealth values $W_1 \leq W_2 \leq \cdots \leq W_N$:

$$
G = \frac{2 \sum_{i=1}^{N} i W_i}{N \sum_{i=1}^{N} W_i} - \frac{N+1}{N}
$$

$G = 0$ is perfect equality; $G = 1$ is maximum inequality (one agent holds all wealth). Crashes tend to increase $G$ sharply as heavily leveraged agents default while cash-holding agents survive.

---

## 10. The Minsky Mechanism End-to-End

Putting all the pieces together, here is the full sequence of events in a simulated Minsky cycle:

**Phase 1 — Stable equilibrium** (steps 0–~100)

- $P_t \approx F_t$, $M_t \approx 0$
- All agents in hedge-finance regime ($L < 1.5$)
- Low rolling volatility
- RL agent begins exploration, learning that leveraged positions in the current environment earn positive returns

**Phase 2 — Leverage build-up** (steps ~100–~300)

- RL agent (and momentum traders) increase positions
- Leveraged buying pushes $P_t > F_t$
- Rising prices confirm momentum traders' signals, creating a self-reinforcing cycle
- Measured volatility stays low (prices trend smoothly upward)
- Proportion of speculative-finance agents rises
- Aggregate leverage $\bar{L}_t$ increases
- Fundamental traders sell into the rising market, but are outnumbered

**Phase 3 — Fragility** (steps ~300–~350)

- Price significantly above fundamental value ($M_t > \theta_b$, bubble flag)
- High proportion of agents in speculative or Ponzi regimes
- System is fragile: margin calls would cascade from even a small price fall
- RL agent (if trained on profit-only reward) is at maximum leverage

**Phase 4 — Shock and cascade** (steps ~350)

- Small negative shock ($\eta_t$ slightly negative, or fundamental value dip)
- First margin calls fire on Ponzi agents
- Forced selling enters market as $\Phi_t$ in the next step
- Price falls further, triggering margin calls on speculative agents
- Forced-sell spiral causes a large price decline disproportionate to the initial shock
- RL agent's wealth collapses

**Phase 5 — Post-crash deleveraging** (steps ~350–~500)

- Surviving agents hold mostly cash (low leverage)
- Price undershoots fundamental value ($M_t < 0$)
- Fundamental traders begin buying, gradually pushing price back toward $F_t$
- System returns to a low-leverage hedge-finance equilibrium
- The cycle can begin again

---

## 11. Limitations and Modelling Assumptions

This simulation is a **stylised computational experiment**, not a calibrated model of any real market. The following simplifications are made deliberately for tractability:

| Assumption | What is omitted |
|-----------|-----------------|
| Single risky asset | No cross-asset contagion, portfolio diversification, or correlation |
| Price-impact without order book | No bid-ask spread, no order queue, no market depth curve |
| No banking sector | No credit contraction, no interbank lending, no central bank |
| No collateral chains | Repo markets, rehypothecation, and collateral spirals are absent |
| Homogeneous borrowing rate | In reality, spreads widen as leverage increases and credit quality deteriorates |
| Hard leverage cap | Real margin requirements are dynamic and endogenous to market conditions |
| No short-selling | Agents cannot profit from declining prices (this limits some stabilising forces) |
| Simplified RL state | The true state includes all other agents' positions — partially observed MDP |
| i.i.d. shocks | Real macro shocks are correlated, regime-dependent, and have fat tails |

These limitations do not invalidate the simulation's findings, but they do restrict the generalisability of its conclusions. A finding that "profit-only RL agents increase leverage and contribute to crashes" in this model is evidence of a possible mechanism, not proof that RL trading systems cause financial instability in practice.

---

## 12. Further Reading

**Minsky's Financial Instability Hypothesis**
- Minsky, H.P. (1986). *Stabilizing an Unstable Economy*. Yale University Press.
- Minsky, H.P. (1992). "The Financial Instability Hypothesis." Levy Economics Institute Working Paper No. 74.
- Kindleberger, C.P. & Aliber, R.Z. (2005). *Manias, Panics, and Crashes*. Palgrave Macmillan.

**Agent-Based Models in Finance**
- LeBaron, B. (2006). "Agent-based Computational Finance." *Handbook of Computational Economics*, Vol. 2.
- Farmer, J.D. & Foley, D. (2009). "The economy needs agent-based modelling." *Nature*, 460, 685–686.
- Tesfatsion, L. & Judd, K.L. (eds.) (2006). *Handbook of Computational Economics*. North-Holland.

**Price Impact and Market Microstructure**
- Kyle, A.S. (1985). "Continuous auctions and insider trading." *Econometrica*, 53(6), 1315–1335.
- Geanakoplos, J. (2010). "The Leverage Cycle." *NBER Macroeconomics Annual*, 24, 1–65.

**Reinforcement Learning**
- Sutton, R.S. & Barto, A.G. (2018). *Reinforcement Learning: An Introduction* (2nd ed.). MIT Press.
- Mnih, V. et al. (2015). "Human-level control through deep reinforcement learning." *Nature*, 518, 529–533.

**RL in Finance**
- Deng, Y. et al. (2016). "Deep Direct Reinforcement Learning for Financial Signal Representation and Trading." *IEEE Transactions on Neural Networks and Learning Systems*.
- Spooner, T. et al. (2018). "Market Making via Reinforcement Learning." *AAMAS 2018*.
