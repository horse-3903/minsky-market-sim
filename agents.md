# Agent Reference

Documentation for every agent type in the simulation. All agents share a common balance sheet through `BaseAgent` and expose a single `decide()` interface that the simulation loop calls each time step.

---

## Balance Sheet (`BaseAgent`)

Every agent tracks the following state at each time step:

| Field | Description |
|-------|-------------|
| `cash` | Undeployed cash holdings |
| `shares` | Units of the risky asset held |
| `debt` | Outstanding borrowed principal |
| `is_active` | `False` once the agent has defaulted |

Derived quantities computed on-demand from these four fields:

```
asset_exposure  = price × shares
wealth          = cash + price × shares − debt
leverage        = |asset_exposure| / max(wealth, ε)
```

An agent defaults (`is_active = False`) when `wealth ≤ 0`.

### Buying and selling

- **Buy** — if the agent has enough cash, the cost is deducted directly. If not, the shortfall is added to `debt` (borrowing to buy).
- **Sell** — proceeds first repay `debt`; any surplus goes to `cash`.

### Interest accrual

At the start of every time step, before orders are collected:

```
debt_{t+1} = debt_t × (1 + r_b)
```

where `r_b` is the per-step borrowing rate set in `LeverageConfig`.

### Margin calls

If `leverage > L_max` (configured in `LeverageConfig.max_leverage`), the agent is forced to sell shares. The number of shares liquidated is chosen to restore leverage to `L_max`, subject to a minimum of `liquidation_fraction × shares`. Proceeds repay debt. If wealth is still ≤ 0 after liquidation, the agent defaults.

Forced-sell units are buffered and injected as additional sell demand at the **next** time step, creating a one-step feedback delay that can amplify price declines.

---

## Minsky Finance-State Classification

Each agent is classified at every step based on leverage (thresholds configurable via `MinskyThresholds`):

| State | Default threshold | Economic meaning |
|-------|------------------|-----------------|
| **Hedge** | `L < 1.5` | Can service and repay debt from existing wealth |
| **Speculative** | `1.5 ≤ L < 3.0` | Can service interest but depends on rolling over principal |
| **Ponzi** | `L ≥ 3.0` | Depends on continued asset-price appreciation to remain solvent |

The proportion of agents in each state is recorded every step and drives the Minsky composition chart in the dashboard.

---

## Rule-Based Agents

### Fundamental Trader (`FundamentalAgent`)

**File:** [`agents/fundamental_agent.py`](agents/fundamental_agent.py)

**Role:** Stabilising. Believes price should converge to fundamental value and trades proportionally to mispricing.

**Signal:**
```
s = (F_t − P_t) / F_t
desired_shares = sensitivity × s × max_position
```

**Behaviour:**
- `s > 0` (price below fundamental) → buy
- `s < 0` (price above fundamental) → sell
- Larger mispricing → larger order

**Key parameters** (set in `AgentConfig`):

| Parameter | Default | Effect |
|-----------|---------|--------|
| `fundamental_sensitivity` | `0.5` | Aggressiveness of mean-reversion trades |
| `fundamental_max_position` | `50.0` | Maximum share position |

---

### Momentum Trader (`MomentumAgent`)

**File:** [`agents/momentum_agent.py`](agents/momentum_agent.py)

**Role:** Destabilising. Chases price trends, amplifying moves in both directions.

**Signal:**
```
m = rolling_momentum  (passed in from MetricsRecorder)
desired_shares = sensitivity × m × max_position
```

**Behaviour:**
- Positive recent return → buy
- Negative recent return → sell
- Momentum lookback window controlled by `MetricsRecorder.mom_window` (default: 10 steps)

**Key parameters:**

| Parameter | Default | Effect |
|-----------|---------|--------|
| `momentum_sensitivity` | `0.3` | Trend-following aggressiveness |
| `momentum_max_position` | `50.0` | Maximum share position |

**Minsky relevance:** Momentum traders are a primary engine of bubble formation. A higher proportion of momentum traders in the population increases mispricing, bubble frequency, and crash severity.

---

### Noise Trader (`NoiseAgent`)

**File:** [`agents/noise_agent.py`](agents/noise_agent.py)

**Role:** Random. Provides baseline market activity and prevents the market from becoming trivially stable.

**Behaviour:** Each step, independently and uniformly draws one of:
- `buy` — purchase a uniform random quantity in `[0, max_trade]`
- `sell` — sell a uniform random quantity in `[0, min(max_trade, current_shares)]`
- `hold` — do nothing

**Key parameters:**

| Parameter | Default | Effect |
|-----------|---------|--------|
| `noise_max_trade` | `10.0` | Maximum units per order |

---

### Market Maker (`MarketMaker`)

**File:** [`agents/market_maker.py`](agents/market_maker.py)

**Role:** Optional liquidity provider. Leans against price deviations from fundamental value, similar to the fundamental trader but with smaller inventory limits and a different sensitivity parameter.

**Signal:**
```
signal = (F_t − P_t) / F_t
desired_shares = spread_sensitivity × signal × max_position
```

**Key parameters:**

| Parameter | Default | Effect |
|-----------|---------|--------|
| `spread_sensitivity` | `0.2` | Lean strength |
| `max_position` | `30.0` | Inventory limit |

Enable via `AgentConfig.n_market_maker > 0`. Useful in regulation experiments to test whether adding liquidity support reduces crash severity.

---

## RL Agent (Phase 4)

**File:** [`agents/rl_agent.py`](agents/rl_agent.py)

Currently a stub that holds and does nothing. Full implementation arrives in Phase 4 when the `MinskyMarketEnv` Gymnasium environment is built.

### Planned observation space

| Feature | Description |
|---------|-------------|
| `price / fundamental` | Current mispricing signal |
| `recent_return` | Last-step return |
| `rolling_volatility` | 20-step return std |
| `rolling_momentum` | 10-step price momentum |
| `cash` | Agent's own cash |
| `shares` | Agent's own inventory |
| `debt` | Agent's own debt |
| `leverage` | Agent's own leverage |
| `aggregate_leverage` | Market-wide average leverage |
| `n_margin_calls` | Recent margin call count |
| `drawdown` | Current price drawdown from peak |

### Planned action space (target exposure)

| Action | Target exposure |
|--------|----------------|
| 0 | 0× (fully flat) |
| 1 | 0.5× |
| 2 | 1× (unleveraged long) |
| 3 | 1.5× |
| 4 | 2× |
| 5 | 3× (high leverage) |

### Planned reward functions

| Variant | Formula | Hypothesis |
|---------|---------|-----------|
| Profit-only | `ΔW_t` | Encourages leverage during calm periods |
| Risk-adjusted | `ΔW_t − λ·drawdown_t − μ·L_t` | Reduces Minsky-style instability |
| System-aware | `ΔW_t − λ·drawdown_t − μ·L_t − γ·agg_leverage_t` | Tests whether agents internalise systemic risk |

---

## Adding a Custom Agent

Subclass `BaseAgent`, set `agent_type` to a unique string, and override `decide()`:

```python
from dataclasses import dataclass, field
from agents.base_agent import BaseAgent
import numpy as np

@dataclass
class MyAgent(BaseAgent):
    agent_type: str = field(default="my_agent")

    def decide(
        self,
        price: float,
        fundamental: float,
        rolling_volatility: float,
        rolling_momentum: float,
        aggregate_leverage: float,
        rng: np.random.Generator,
    ) -> tuple[float, float]:
        # return (buy_demand, sell_demand) as positive floats
        return 0.0, 0.0
```

Then pass instances to `MarketSimulation` via the `extra_agents` parameter:

```python
from market.environment import MarketSimulation
from market.config import SimulationConfig

sim = MarketSimulation(config, extra_agents=[MyAgent() for _ in range(5)])
sim.run()
```

Wealth tracking by type uses the `agent_type` string, so custom agents will appear under `wealth_{agent_type}` in the metrics dataframe if `MetricsRecorder.record()` is extended to include the new type.
