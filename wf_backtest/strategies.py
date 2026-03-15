"""
Trading Strategies
===================
Each strategy returns a *position signal* Series aligned to the price index:
  1 = fully long, 0 = cash.

Signals are always shifted by 1 day to avoid lookahead bias –
signal at close(t) determines position for day t+1.
"""

import numpy as np
import pandas as pd


# ── Momentum ─────────────────────────────────────────────────────────────────

def momentum_signal(prices: pd.Series, lookback: int) -> pd.Series:
    """
    Long when cumulative return over *lookback* days > 0.
    Signal is lagged by 1 day (trade on next open).
    """
    ret = prices / prices.shift(lookback) - 1
    raw = (ret > 0).astype(float)
    return raw.shift(1)                      # no lookahead


# ── Moving Average ───────────────────────────────────────────────────────────

def ma_signal(prices: pd.Series, period: int) -> pd.Series:
    """
    Long when price > SMA(period).
    Signal lagged by 1 day.
    """
    sma = prices.rolling(period, min_periods=period).mean()
    raw = (prices > sma).astype(float)
    return raw.shift(1)


# ── RSI ──────────────────────────────────────────────────────────────────────

def _compute_rsi(prices: pd.Series, period: int) -> pd.Series:
    """Wilder-smoothed RSI."""
    delta = prices.diff()
    gain = delta.clip(lower=0)
    loss = (-delta).clip(lower=0)
    avg_gain = gain.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - 100 / (1 + rs)
    return rsi


def rsi_signal(prices: pd.Series, period: int, threshold: float) -> pd.Series:
    """
    Long when RSI(period) > threshold.
    Signal lagged by 1 day.
    """
    rsi = _compute_rsi(prices, period)
    raw = (rsi > threshold).astype(float)
    return raw.shift(1)


# ── Volatility Targeting (position scaler) ───────────────────────────────────

def vol_target_scaler(returns: pd.Series,
                      target_vol: float = 0.15,
                      lookback: int = 63) -> pd.Series:
    """
    Multiplicative scaler to target annualized volatility.
    Capped at [0, 2] to avoid extreme leverage.
    Lagged by 1 day.
    """
    realized = returns.rolling(lookback, min_periods=lookback).std() * np.sqrt(252)
    scale = target_vol / realized
    scale = scale.clip(0, 2.0)
    return scale.shift(1)


# ── Transaction cost engine ──────────────────────────────────────────────────

def apply_costs(signal: pd.Series,
                returns: pd.Series,
                tx_cost: float = 0.001,
                slippage: float = 0.0005) -> pd.Series:
    """
    Apply one-way transaction cost + slippage each time position changes.
    Returns net daily strategy returns.
    """
    trades = signal.diff().abs().fillna(0)
    cost_series = trades * (tx_cost + slippage)
    gross = signal * returns
    net = gross - cost_series
    return net


# ── Convenience: strategy registry ──────────────────────────────────────────

STRATEGY_REGISTRY = {
    "Momentum": {
        "func": momentum_signal,
        "param_name": "lookback",       # primary param key
    },
    "MA": {
        "func": ma_signal,
        "param_name": "period",
    },
    "RSI": {
        "func": rsi_signal,
        "param_name": ("period", "threshold"),  # 2-D grid
    },
}
