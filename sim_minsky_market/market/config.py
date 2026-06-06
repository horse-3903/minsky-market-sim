"""Market configuration dataclasses."""

from dataclasses import dataclass, field


@dataclass
class FundamentalConfig:
    initial_value: float = 100.0
    drift: float = 0.0
    volatility: float = 0.005  # sigma_F — kept small so crashes are endogenous


@dataclass
class PriceConfig:
    initial_price: float = 100.0
    price_impact: float = 0.01   # alpha — sensitivity to net demand
    noise_volatility: float = 0.002  # eta — baseline market noise
    liquidity: float = 1.0       # scales impact; lower = more impact per unit demand


@dataclass
class LeverageConfig:
    max_leverage: float = 3.0        # L_max — triggers margin call
    borrowing_rate: float = 0.0005   # r_b per step
    margin_call_threshold: float = 3.0
    liquidation_fraction: float = 0.5  # fraction of position sold on forced liquidation


@dataclass
class AgentConfig:
    n_fundamental: int = 20
    n_momentum: int = 10
    n_noise: int = 10
    n_rl: int = 0
    n_market_maker: int = 0

    # Fundamental trader params
    fundamental_sensitivity: float = 0.5   # how aggressively they trade on mispricing
    fundamental_max_position: float = 50.0

    # Momentum trader params
    momentum_lookback: int = 10
    momentum_sensitivity: float = 0.3
    momentum_max_position: float = 50.0

    # Noise trader params
    noise_max_trade: float = 10.0

    # Initial wealth per agent
    initial_cash: float = 1000.0


@dataclass
class MinskyThresholds:
    """Leverage thresholds for Minsky finance-state classification."""
    hedge_max: float = 1.5       # L < 1.5 → hedge
    speculative_max: float = 3.0  # 1.5 ≤ L < 3.0 → speculative
                                  # L ≥ 3.0 → Ponzi


@dataclass
class SimulationConfig:
    n_steps: int = 1000
    seed: int = 42
    fundamental: FundamentalConfig = field(default_factory=FundamentalConfig)
    price: PriceConfig = field(default_factory=PriceConfig)
    leverage: LeverageConfig = field(default_factory=LeverageConfig)
    agents: AgentConfig = field(default_factory=AgentConfig)
    minsky: MinskyThresholds = field(default_factory=MinskyThresholds)

    # Bubble / crash detection
    bubble_threshold: float = 0.20    # price > 20% above fundamental
    crash_threshold: float = -0.20    # price drops > 20% over crash_window
    crash_window: int = 20

    # Output
    output_dir: str = "results"
