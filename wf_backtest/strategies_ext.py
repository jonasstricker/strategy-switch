"""
Extended Strategy Library
==========================
Additional strategies beyond Momentum / MA / RSI:
  - MACD
  - Bollinger Bands (mean-reversion & breakout)
  - Dual Momentum (absolute + relative via SMA filter)
  - Double Moving Average Crossover
  - Donchian Channel Breakout
  - Combined / Ensemble signals

All signals: 1 = long, 0 = cash, shifted by 1 day (no lookahead).
"""

import numpy as np
import pandas as pd


# ── MACD ─────────────────────────────────────────────────────────────────────

def macd_signal(prices: pd.Series, fast: int = 12, slow: int = 26,
                signal_period: int = 9) -> pd.Series:
    """
    Long when MACD line > signal line.
    """
    ema_fast = prices.ewm(span=fast, min_periods=fast, adjust=False).mean()
    ema_slow = prices.ewm(span=slow, min_periods=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal_period, min_periods=signal_period,
                                adjust=False).mean()
    raw = (macd_line > signal_line).astype(float)
    return raw.shift(1)


# ── Bollinger Bands ──────────────────────────────────────────────────────────

def bollinger_breakout_signal(prices: pd.Series, period: int = 20,
                              n_std: float = 2.0) -> pd.Series:
    """
    Breakout: Long when price > upper band (trending).
    """
    sma = prices.rolling(period, min_periods=period).mean()
    std = prices.rolling(period, min_periods=period).std()
    upper = sma + n_std * std
    raw = (prices > upper).astype(float)
    return raw.shift(1)


def bollinger_mean_reversion_signal(prices: pd.Series, period: int = 20,
                                     n_std: float = 2.0) -> pd.Series:
    """
    Mean-reversion: Long when price < lower band (expect bounce).
    Exit when price > SMA.
    """
    sma = prices.rolling(period, min_periods=period).mean()
    std = prices.rolling(period, min_periods=period).std()
    lower = sma - n_std * std
    # State machine: enter when below lower, exit when above sma
    raw = pd.Series(0.0, index=prices.index)
    in_position = False
    for i in range(1, len(prices)):
        if not in_position and prices.iloc[i] < lower.iloc[i]:
            in_position = True
        elif in_position and prices.iloc[i] > sma.iloc[i]:
            in_position = False
        raw.iloc[i] = 1.0 if in_position else 0.0
    return raw.shift(1)


# ── Dual Momentum ────────────────────────────────────────────────────────────

def dual_momentum_signal(prices: pd.Series,
                         abs_lookback: int = 200,
                         trend_period: int = 200) -> pd.Series:
    """
    Dual momentum:
      1. Absolute momentum: price return over abs_lookback > 0
      2. Trend filter: price > SMA(trend_period)
    Long only when BOTH are true.
    """
    abs_ret = prices / prices.shift(abs_lookback) - 1
    sma = prices.rolling(trend_period, min_periods=trend_period).mean()
    abs_ok = abs_ret > 0
    trend_ok = prices > sma
    raw = (abs_ok & trend_ok).astype(float)
    return raw.shift(1)


# ── Double MA Crossover ─────────────────────────────────────────────────────

def double_ma_signal(prices: pd.Series, fast: int = 50,
                     slow: int = 200) -> pd.Series:
    """
    Golden/Death cross: Long when SMA(fast) > SMA(slow).
    """
    sma_fast = prices.rolling(fast, min_periods=fast).mean()
    sma_slow = prices.rolling(slow, min_periods=slow).mean()
    raw = (sma_fast > sma_slow).astype(float)
    return raw.shift(1)


# ── Donchian Channel ────────────────────────────────────────────────────────

def donchian_signal(prices: pd.Series, period: int = 55) -> pd.Series:
    """
    Donchian breakout: Long when price hits new period-day high.
    """
    highest = prices.rolling(period, min_periods=period).max()
    raw = (prices >= highest).astype(float)
    return raw.shift(1)


# ── Ensemble / Combined Signal ──────────────────────────────────────────────

def ensemble_signal(signals: list, threshold: float = 0.5) -> pd.Series:
    """
    Combine multiple binary signals. Long when fraction of active
    signals >= threshold.

    Parameters
    ----------
    signals : list of pd.Series (each 0/1)
    threshold : fraction of signals that must agree (0.5 = majority vote)
    """
    df = pd.concat(signals, axis=1)
    mean_sig = df.mean(axis=1)
    raw = (mean_sig >= threshold).astype(float)
    return raw  # already shifted if inputs are shifted


# ── Partial Position (gradual risk-on / risk-off) ───────────────────────────

def partial_position_signal(prices: pd.Series, ma_short: int = 50,
                            ma_long: int = 200) -> pd.Series:
    """
    Gradual positioning:
      - Full (1.0) when price > SMA(short) > SMA(long)
      - Half (0.5) when price > SMA(long) but < SMA(short)
      - Zero (0.0) otherwise
    """
    sma_s = prices.rolling(ma_short, min_periods=ma_short).mean()
    sma_l = prices.rolling(ma_long, min_periods=ma_long).mean()
    raw = pd.Series(0.0, index=prices.index)
    raw[prices > sma_l] = 0.5
    raw[(prices > sma_s) & (sma_s > sma_l)] = 1.0
    return raw.shift(1)


# ── Adaptive Momentum (volatility-scaled lookback) ──────────────────────────

def adaptive_momentum_signal(prices: pd.Series, returns: pd.Series,
                              base_lookback: int = 126,
                              vol_lookback: int = 63) -> pd.Series:
    """
    Momentum with lookback that adapts to volatility regime.
    Low vol → longer lookback (trend), high vol → shorter (faster reaction).
    """
    realized = returns.rolling(vol_lookback, min_periods=vol_lookback).std() * np.sqrt(252)
    median_vol = realized.expanding(min_periods=vol_lookback).median()
    vol_ratio = realized / median_vol
    # Scale lookback inversely with vol ratio
    adapted = (base_lookback / vol_ratio).clip(20, 504)
    # Can't vectorize variable lookback easily, so use discrete buckets
    raw = pd.Series(np.nan, index=prices.index)
    for lb in [40, 63, 90, 126, 180, 252]:
        mask = ((adapted >= lb - 20) & (adapted < lb + 20))
        ret = prices / prices.shift(lb) - 1
        raw[mask] = (ret[mask] > 0).astype(float)
    raw = raw.ffill().fillna(0)
    return raw.shift(1)
