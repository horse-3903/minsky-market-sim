<div align="center">

# minsky-market-sim

Agent-based reinforcement learning simulation of Minsky's Financial Instability Hypothesis — testing whether profit-maximising agents can endogenously generate financial fragility from a stable market equilibrium.

[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=flat-square&logo=python&logoColor=white)](https://python.org)
[![NumPy](https://img.shields.io/badge/NumPy-013243?style=flat-square&logo=numpy&logoColor=white)](https://numpy.org)
[![Pandas](https://img.shields.io/badge/Pandas-150458?style=flat-square&logo=pandas&logoColor=white)](https://pandas.pydata.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=flat-square&logo=streamlit&logoColor=white)](https://streamlit.io)
[![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=flat-square&logo=pytorch&logoColor=white)](https://pytorch.org)
[![Gymnasium](https://img.shields.io/badge/Gymnasium-0078D4?style=flat-square)](https://gymnasium.farama.org)
[![License](https://img.shields.io/badge/License-MIT-green?style=flat-square)](LICENSE)
[![Last Commit](https://img.shields.io/github/last-commit/horse-3903/minsky-market-sim?style=flat-square)](https://github.com/horse-3903/minsky-market-sim/commits/main)

</div>

---

## Overview

**minsky-market-sim** is a discrete-time agent-based market simulation that investigates whether individually rational, profit-seeking behaviour can endogenously generate the leverage buildup and cascading deleveraging described in Minsky's Financial Instability Hypothesis. The market begins in a controlled stable equilibrium — price at fundamental value, low volatility, no defaults — and evolves entirely through agent interactions.

The core research question: can reinforcement learning agents, optimising for profit, learn to increase leverage during calm periods in a way that creates systemic fragility? Can a small exogenous shock then trigger margin calls, forced liquidation, and a crash — without the crash being built into the model?

---

## Features

- **Endogenous instability** — market starts stable by design; fragility emerges from agent behaviour, not from external shocks
- **Full agent balance sheets** — each agent tracks cash, shares, debt, wealth, and leverage; interest accrues per step
- **Margin calls and forced liquidation** — agents breaching `L_max` are automatically deleveraged; forced sells feed back into next-step price pressure
- **Minsky finance-state classification** — agents classified as hedge, speculative, or Ponzi at every step based on leverage thresholds
- **Four rule-based agent types** — fundamental (stabilising), momentum (destabilising), noise (random), market maker (optional liquidity)
- **RL agent scaffolding** — `RLAgent` stub ready for Phase 4 Gymnasium environment and DQN/PPO implementation
- **Interactive Streamlit dashboard** — 5-tab UI with Plotly charts, live parameter controls, and CSV/JSON data export
- **Reproducible experiments** — every run is seeded; results saved as timestamped CSV and summary JSON
- **30 unit tests** — covering price mechanics, balance-sheet accounting, margin call logic, and metrics calculations

---

## Tech Stack

[![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![NumPy](https://img.shields.io/badge/NumPy-013243?style=for-the-badge&logo=numpy&logoColor=white)](https://numpy.org)
[![Pandas](https://img.shields.io/badge/Pandas-150458?style=for-the-badge&logo=pandas&logoColor=white)](https://pandas.pydata.org)
[![Matplotlib](https://img.shields.io/badge/Matplotlib-11557C?style=for-the-badge)](https://matplotlib.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://streamlit.io)
[![Plotly](https://img.shields.io/badge/Plotly-3F4F75?style=for-the-badge&logo=plotly&logoColor=white)](https://plotly.com)
[![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)](https://pytorch.org)
[![Gymnasium](https://img.shields.io/badge/Gymnasium-0078D4?style=for-the-badge)](https://gymnasium.farama.org)

---

## Getting Started

### Prerequisites

- Python 3.10 or higher
- pip

### Installation

```bash
git clone https://github.com/horse-3903/minsky-market-sim.git
cd minsky-market-sim
pip install -r requirements.txt
```

### Run the baseline simulation

```bash
python main.py --experiment baseline --seed 42
```

Results are written to `results/baseline/` — a time-series CSV, a summary JSON, and eight matplotlib plots.

### Launch the dashboard

```bash
cd dashboard
PYTHONPATH=.. streamlit run app.py   # macOS / Linux
$env:PYTHONPATH = ".."; streamlit run app.py   # Windows PowerShell
```

Open `http://localhost:8501`. Use the sidebar to configure agent counts, leverage limits, price impact, and other parameters, then press **Run Simulation** to recompute everything live.

### Run the test suite

```bash
python -m pytest tests/ -v
```

---

## Project Structure

```
minsky-market-sim/
│
├── main.py                    # CLI entry point
├── requirements.txt
├── conftest.py                # pytest path setup
├── agents.md                  # agent type reference
│
├── market/                    # core simulation engine
│   ├── config.py              # SimulationConfig dataclasses
│   ├── asset.py               # fundamental value process (GBM)
│   ├── price_mechanism.py     # log-linear price-impact model
│   ├── environment.py         # main simulation loop
│   ├── margin.py              # leverage, interest, margin calls, defaults
│   └── metrics.py             # rolling stats, Minsky classification, Gini
│
├── agents/                    # agent implementations
│   ├── base_agent.py          # balance sheet, wealth, leverage
│   ├── fundamental_agent.py   # mean-reversion trader
│   ├── momentum_agent.py      # trend-following trader
│   ├── noise_agent.py         # random trader
│   ├── market_maker.py        # optional liquidity provider
│   └── rl_agent.py            # RL agent stub (Phase 4)
│
├── rl/                        # RL implementations (Phase 4)
│   ├── q_learning.py
│   ├── dqn.py
│   ├── replay_buffer.py
│   └── networks.py
│
├── experiments/               # runnable experiment scripts
│   ├── baseline_stability.py  # Experiment 1 — stable baseline
│   ├── vary_momentum_share.py # Experiment 2 — momentum trader sweep (Phase 3)
│   ├── train_profit_rl.py     # Experiment 3 — profit-only RL (Phase 4)
│   ├── compare_rewards.py     # Experiment 4 — reward function comparison (Phase 5)
│   └── regulation_tests.py    # Experiment 5 — leverage caps, circuit breakers (Phase 6)
│
├── analysis/                  # post-run analysis
│   ├── plots.py               # matplotlib plot suite
│   ├── metrics_report.py      # summary statistics and CSV/JSON output
│   └── statistical_tests.py   # cross-experiment comparison (Phase 3+)
│
├── dashboard/                 # interactive web dashboard
│   ├── app.py                 # Streamlit application
│   └── charts.py              # Plotly chart builders
│
└── tests/                     # unit tests (30 tests, all passing)
    ├── test_price_mechanism.py
    ├── test_margin_calls.py
    ├── test_agents.py
    └── test_metrics.py
```

---

## Development Phases

| Phase | Scope | Status |
|-------|-------|--------|
| **1 — Market simulator** | Asset, price mechanism, rule-based agents, metrics, matplotlib plots | ✅ Complete |
| **2 — Leverage & margin calls** | Debt interest, forced liquidation, defaults, Minsky classification | ✅ Complete |
| **2.5 — Dashboard** | Streamlit + Plotly interactive UI, 5 tabs, live parameter controls | ✅ Complete |
| **3 — Rule-based experiments** | Vary momentum trader share; baseline stability validation | 🔲 Next |
| **4 — RL environment** | Gymnasium `MinskyMarketEnv`, tabular Q-learning, DQN | 🔲 Planned |
| **5 — Reward comparison** | Profit-only vs risk-adjusted vs system-aware reward functions | 🔲 Planned |
| **6 — Regulation experiments** | Leverage caps, margin requirements, transaction tax, circuit breakers | 🔲 Planned |
| **7 — Report & dashboard polish** | Final visualisations, Streamlit demo, written report | 🔲 Planned |

---

## Research Questions

1. Do profit-maximising RL agents learn to increase leverage during calm market periods?
2. Does leveraged buying create asset-price bubbles relative to fundamental value?
3. Can a small shock trigger margin calls and forced liquidation cascades?
4. Do risk-sensitive reward functions reduce financial instability relative to profit-only rewards?
5. Which regulatory interventions — leverage caps, margin requirements, transaction taxes, circuit breakers — most effectively reduce crash frequency without excessive liquidity loss?

---

## Simulation Mechanics

The market runs for `n_steps` discrete time steps. Each step:

1. Accrue interest on all agent debt
2. Each agent observes the market state and submits a buy/sell order
3. Aggregate net demand is normalised by agent count and passed to the price-impact function
4. Price updates: `P_{t+1} = P_t × exp(α × D_t / liquidity + η_t)`
5. Fundamental value updates: `F_{t+1} = F_t × (1 + μ_F + ε_t)`
6. Agents are marked to market; leverage is recalculated
7. Agents exceeding `L_max` receive margin calls and are force-liquidated
8. Forced-sell units buffer into next step's sell demand, creating a feedback loop
9. Metrics recorded: price, mispricing, leverage, volatility, Minsky states, Gini, crash/bubble flags

See [`agents.md`](agents.md) for full agent documentation.

---

## Disclaimer

This simulation is a stylised computational model built to study one possible mechanism under clearly stated assumptions. It does not predict real financial crises, claim to prove Minsky's hypothesis empirically, or model real banking systems, collateral chains, or macroeconomic feedbacks. Results depend on parameter choices and should be interpreted as exploratory rather than definitive.
