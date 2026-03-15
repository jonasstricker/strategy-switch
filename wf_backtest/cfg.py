"""
Walk-Forward Backtesting Framework – Configuration
====================================================
All parameters centralized. No magic numbers in code.
"""

from dataclasses import dataclass, field
from typing import List


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------
@dataclass
class DataConfig:
    """Data download and validation settings."""
    tickers: List[str] = field(
        default_factory=lambda: ["IWDA.AS", "URTH", "ACWI", "SPY"]
    )
    start_date: str = "2005-01-01"
    end_date: str = "2026-02-28"
    min_years: int = 15


# ---------------------------------------------------------------------------
# Transaction costs
# ---------------------------------------------------------------------------
@dataclass
class CostConfig:
    """Realistic cost assumptions."""
    transaction_cost: float = 0.001   # 0.1 % per trade (round-trip)
    slippage: float = 0.0005          # 0.05 % slippage per trade


# ---------------------------------------------------------------------------
# Walk-Forward
# ---------------------------------------------------------------------------
@dataclass
class WalkForwardConfig:
    """Walk-forward optimization settings."""
    train_days: int = 1260            # ~5 years
    test_days: int = 63               # ~3 months
    step_days: int = 63               # ~3 months
    top_pct: float = 0.20             # top 20 % of parameter space
    optimization_metric: str = "sharpe"


# ---------------------------------------------------------------------------
# Strategies
# ---------------------------------------------------------------------------
@dataclass
class StrategyConfig:
    """Parameter grids for each strategy."""
    # Momentum lookback (days)
    mom_lookbacks: List[int] = field(
        default_factory=lambda: list(range(60, 253, 12))
    )
    # Moving-average period (days)
    ma_periods: List[int] = field(
        default_factory=lambda: list(range(50, 251, 10))
    )
    # RSI period (days)
    rsi_periods: List[int] = field(
        default_factory=lambda: [10, 14, 20, 25, 30, 35, 40]
    )
    # RSI threshold (long when RSI > threshold)
    rsi_thresholds: List[int] = field(
        default_factory=lambda: [40, 45, 50, 55, 60]
    )
    # Volatility targeting
    vol_target: float = 0.15
    vol_lookback: int = 63            # ~3 months for realized vol


# ---------------------------------------------------------------------------
# Strategy switching
# ---------------------------------------------------------------------------
@dataclass
class SwitchingConfig:
    """Rolling strategy selection settings."""
    rolling_window: int = 126         # 6 months
    min_sharpe: float = 0.0           # minimum Sharpe to be eligible


# ---------------------------------------------------------------------------
# Stability / robustness
# ---------------------------------------------------------------------------
@dataclass
class StabilityConfig:
    """Monte Carlo / bootstrap settings."""
    n_bootstrap: int = 1_000
    n_monte_carlo: int = 1_000
    confidence_level: float = 0.95
    block_length: int = 21            # block bootstrap length (~1 month)


# ---------------------------------------------------------------------------
# Master config
# ---------------------------------------------------------------------------
@dataclass
class FrameworkConfig:
    """Root configuration object – modify here or override in code."""
    data: DataConfig = field(default_factory=DataConfig)
    costs: CostConfig = field(default_factory=CostConfig)
    wf: WalkForwardConfig = field(default_factory=WalkForwardConfig)
    strategy: StrategyConfig = field(default_factory=StrategyConfig)
    switching: SwitchingConfig = field(default_factory=SwitchingConfig)
    stability: StabilityConfig = field(default_factory=StabilityConfig)
    risk_free_rate: float = 0.02      # annual risk-free rate
