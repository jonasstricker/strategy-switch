"""
Performance Metrics
====================
All metrics operate on pandas Series. No lookahead bias possible here –
these are pure post-hoc measurement functions.
"""

import numpy as np
import pandas as pd


# ── Return / growth metrics ─────────────────────────────────────────────────

def cagr(equity: pd.Series) -> float:
    """Compound Annual Growth Rate."""
    if len(equity) < 2 or equity.iloc[0] <= 0:
        return 0.0
    total = equity.iloc[-1] / equity.iloc[0]
    n_years = len(equity) / 252
    if n_years <= 0:
        return 0.0
    return float(total ** (1 / n_years) - 1)


# ── Risk-adjusted metrics ───────────────────────────────────────────────────

def sharpe_ratio(returns: pd.Series, rf_annual: float = 0.02) -> float:
    """Annualized Sharpe Ratio (daily returns input)."""
    rf_daily = (1 + rf_annual) ** (1 / 252) - 1
    excess = returns - rf_daily
    std = excess.std()
    if std == 0 or np.isnan(std):
        return 0.0
    return float(np.sqrt(252) * excess.mean() / std)


def sortino_ratio(returns: pd.Series, rf_annual: float = 0.02) -> float:
    """Annualized Sortino Ratio."""
    rf_daily = (1 + rf_annual) ** (1 / 252) - 1
    excess = returns - rf_daily
    downside = excess[excess < 0]
    if len(downside) < 2:
        return 0.0
    down_std = downside.std()
    if down_std == 0 or np.isnan(down_std):
        return 0.0
    return float(np.sqrt(252) * excess.mean() / down_std)


def calmar_ratio(equity: pd.Series) -> float:
    """Calmar = CAGR / |MaxDD|."""
    c = cagr(equity)
    mdd = abs(max_drawdown(equity))
    if mdd == 0:
        return 0.0
    return float(c / mdd)


# ── Drawdown metrics ────────────────────────────────────────────────────────

def max_drawdown(equity: pd.Series) -> float:
    """Maximum drawdown (negative number)."""
    peak = equity.expanding().max()
    dd = (equity - peak) / peak
    return float(dd.min())


def drawdown_series(equity: pd.Series) -> pd.Series:
    """Full drawdown time series."""
    peak = equity.expanding().max()
    return (equity - peak) / peak


def time_under_water(equity: pd.Series) -> dict:
    """
    Time-under-water statistics.
    Returns dict with max / avg / current days and count of DD periods.
    """
    peak = equity.expanding().max()
    underwater = equity < peak

    periods = []
    start = None
    for idx, is_uw in underwater.items():
        if is_uw and start is None:
            start = idx
        elif not is_uw and start is not None:
            periods.append((idx - start).days)
            start = None
    # still underwater at end
    if start is not None:
        periods.append((underwater.index[-1] - start).days)

    if not periods:
        return {"max_days": 0, "avg_days": 0.0, "current_days": 0, "n_periods": 0}

    current = periods[-1] if underwater.iloc[-1] else 0
    return {
        "max_days": max(periods),
        "avg_days": float(np.mean(periods)),
        "current_days": current,
        "n_periods": len(periods),
    }


# ── Rolling metrics ─────────────────────────────────────────────────────────

def rolling_sharpe(returns: pd.Series, window: int = 126,
                   rf_annual: float = 0.02) -> pd.Series:
    """Rolling annualized Sharpe (default 6-month window)."""
    rf_daily = (1 + rf_annual) ** (1 / 252) - 1
    excess = returns - rf_daily
    rmean = excess.rolling(window, min_periods=window).mean()
    rstd = excess.rolling(window, min_periods=window).std()
    return np.sqrt(252) * rmean / rstd


def rolling_outperformance(strat_ret: pd.Series, bench_ret: pd.Series,
                           window: int = 252) -> pd.Series:
    """Rolling cumulative outperformance (default 1-year window)."""
    diff = strat_ret - bench_ret
    return diff.rolling(window, min_periods=window).sum()


# ── Summary ──────────────────────────────────────────────────────────────────

def summary_table(equity: pd.Series, returns: pd.Series,
                  bench_equity: pd.Series, bench_returns: pd.Series,
                  rf_annual: float = 0.02, label: str = "Strategy") -> pd.DataFrame:
    """Side-by-side comparison table."""
    def _row(eq, ret, name):
        tuw = time_under_water(eq)
        return {
            "Name": name,
            "CAGR": f"{cagr(eq):.2%}",
            "Sharpe": f"{sharpe_ratio(ret, rf_annual):.2f}",
            "Sortino": f"{sortino_ratio(ret, rf_annual):.2f}",
            "Calmar": f"{calmar_ratio(eq):.2f}",
            "Max DD": f"{max_drawdown(eq):.2%}",
            "Max UW (d)": tuw["max_days"],
            "Avg UW (d)": f"{tuw['avg_days']:.0f}",
        }

    rows = [
        _row(equity, returns, label),
        _row(bench_equity, bench_returns, "Buy & Hold"),
    ]
    return pd.DataFrame(rows).set_index("Name")
