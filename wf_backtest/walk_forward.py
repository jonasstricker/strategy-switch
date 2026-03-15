"""
Walk-Forward Optimisation Engine
=================================
Strictly separated in-sample / out-of-sample windows.
No lookahead bias: signals generated only from training data,
parameter selection based only on training Sharpe.

Procedure per step:
  1. Define [train_start .. train_end] and [test_start .. test_end]
  2. For each strategy, grid-search all parameter combos on train set
  3. Rank by Sharpe, keep top 20 %
  4. Take **median** parameter (no peak-picking)
  5. Generate signal with median param, apply to test set
  6. Record OOS returns + selected parameter
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Any

import numpy as np
import pandas as pd
from tqdm import tqdm

from .cfg import FrameworkConfig
from .strategies import (
    momentum_signal, ma_signal, rsi_signal,
    vol_target_scaler, apply_costs,
)
from .metrics import sharpe_ratio


# ── Data structures ──────────────────────────────────────────────────────────

@dataclass
class WindowResult:
    """Result for one OOS window and one strategy."""
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp
    strategy: str
    selected_params: dict
    oos_returns: pd.Series          # daily OOS net returns
    oos_sharpe: float
    is_signal: pd.Series            # OOS position signal


@dataclass
class WalkForwardResult:
    """Aggregated walk-forward results."""
    strategy_name: str
    window_results: List[WindowResult] = field(default_factory=list)
    oos_returns: pd.Series = field(default_factory=lambda: pd.Series(dtype=float))
    param_history: List[dict] = field(default_factory=list)


# ── Window generation ────────────────────────────────────────────────────────

def generate_windows(dates: pd.DatetimeIndex,
                     train_days: int, test_days: int, step_days: int
                     ) -> List[Tuple[int, int, int, int]]:
    """
    Return list of (train_start_idx, train_end_idx,
                     test_start_idx, test_end_idx).
    Indices are positional into *dates*.
    """
    windows = []
    n = len(dates)
    train_end = train_days
    while train_end + test_days <= n:
        train_start = train_end - train_days
        test_start = train_end
        test_end = min(train_end + test_days, n)
        windows.append((train_start, train_end, test_start, test_end))
        train_end += step_days
    return windows


# ── Single-strategy optimiser ────────────────────────────────────────────────

def _sharpe_for_signal(signal: pd.Series, returns: pd.Series,
                       cfg: FrameworkConfig) -> float:
    """Compute Sharpe of a strategy signal on the given return series."""
    if signal.isna().all():
        return -999.0
    clean_sig = signal.fillna(0)
    net = apply_costs(clean_sig, returns, cfg.costs.transaction_cost,
                      cfg.costs.slippage)
    net = net.dropna()
    if len(net) < 20:
        return -999.0
    return sharpe_ratio(net, cfg.risk_free_rate)


def _optimize_momentum(prices_train: pd.Series, returns_train: pd.Series,
                       cfg: FrameworkConfig) -> dict:
    """Grid search momentum lookback."""
    results = []
    for lb in cfg.strategy.mom_lookbacks:
        sig = momentum_signal(prices_train, lb)
        sr = _sharpe_for_signal(sig, returns_train, cfg)
        results.append({"lookback": lb, "sharpe": sr})

    return _select_median_params(results, ["lookback"], cfg.wf.top_pct)


def _optimize_ma(prices_train: pd.Series, returns_train: pd.Series,
                 cfg: FrameworkConfig) -> dict:
    """Grid search MA period."""
    results = []
    for p in cfg.strategy.ma_periods:
        sig = ma_signal(prices_train, p)
        sr = _sharpe_for_signal(sig, returns_train, cfg)
        results.append({"period": p, "sharpe": sr})

    return _select_median_params(results, ["period"], cfg.wf.top_pct)


def _optimize_rsi(prices_train: pd.Series, returns_train: pd.Series,
                  cfg: FrameworkConfig) -> dict:
    """Grid search RSI period × threshold."""
    results = []
    for per, thr in itertools.product(cfg.strategy.rsi_periods,
                                      cfg.strategy.rsi_thresholds):
        sig = rsi_signal(prices_train, per, thr)
        sr = _sharpe_for_signal(sig, returns_train, cfg)
        results.append({"period": per, "threshold": thr, "sharpe": sr})

    return _select_median_params(results, ["period", "threshold"], cfg.wf.top_pct)


def _select_median_params(results: List[dict], param_keys: List[str],
                          top_pct: float) -> dict:
    """
    From grid-search results, pick top *top_pct* by Sharpe,
    return median of each parameter dimension.
    """
    df = pd.DataFrame(results).sort_values("sharpe", ascending=False)
    n_top = max(1, int(len(df) * top_pct))
    top = df.head(n_top)
    median_params = {}
    for key in param_keys:
        med = top[key].median()
        # Snap to nearest grid value
        all_vals = sorted(df[key].unique())
        closest = min(all_vals, key=lambda v: abs(v - med))
        median_params[key] = int(closest)
    return median_params


# ── Strategy dispatcher ──────────────────────────────────────────────────────

_OPTIMIZER = {
    "Momentum": _optimize_momentum,
    "MA": _optimize_ma,
    "RSI": _optimize_rsi,
}


def _generate_signal(strategy: str, prices: pd.Series, params: dict) -> pd.Series:
    """Generate position signal for given strategy + params."""
    if strategy == "Momentum":
        return momentum_signal(prices, params["lookback"])
    elif strategy == "MA":
        return ma_signal(prices, params["period"])
    elif strategy == "RSI":
        return rsi_signal(prices, params["period"], params["threshold"])
    else:
        raise ValueError(f"Unknown strategy: {strategy}")


# ── Main walk-forward runner ─────────────────────────────────────────────────

def run_walk_forward(prices: pd.Series, returns: pd.Series,
                     cfg: FrameworkConfig,
                     use_vol_target: bool = False,
                     ) -> Dict[str, WalkForwardResult]:
    """
    Run walk-forward optimisation for all three strategies.

    Parameters
    ----------
    prices : daily Close prices
    returns : daily simple returns (aligned to *prices*)
    cfg : framework configuration
    use_vol_target : whether to apply vol-targeting overlay

    Returns
    -------
    Dict mapping strategy name → WalkForwardResult
    """
    dates = prices.index
    windows = generate_windows(dates, cfg.wf.train_days,
                               cfg.wf.test_days, cfg.wf.step_days)
    print(f"[wf] {len(windows)} walk-forward windows generated")

    all_results: Dict[str, WalkForwardResult] = {}

    for strat_name, optimizer in _OPTIMIZER.items():
        print(f"\n[wf] ── {strat_name} ──")
        wf_res = WalkForwardResult(strategy_name=strat_name)

        oos_parts = []

        for i, (ts, te, os_s, os_e) in enumerate(tqdm(windows, desc=strat_name)):
            train_prices = prices.iloc[ts:te]
            train_returns = returns.iloc[ts:te]
            test_prices = prices.iloc[os_s:os_e]
            test_returns = returns.iloc[os_s:os_e]

            # 1. Optimise on training data
            best_params = optimizer(train_prices, train_returns, cfg)

            # 2. Generate signal on FULL available data up to test end
            #    (signal needs lookback history but only test-period is scored)
            #    We use data from train_start→test_end to compute the signal,
            #    then slice the test period.
            full_prices = prices.iloc[ts:os_e]
            full_returns = returns.iloc[ts:os_e]
            full_signal = _generate_signal(strat_name, full_prices, best_params)

            # Optional vol-targeting overlay
            if use_vol_target:
                vt = vol_target_scaler(
                    full_returns, cfg.strategy.vol_target,
                    cfg.strategy.vol_lookback)
                full_signal = (full_signal * vt).clip(0, 1)

            # 3. Slice test period
            test_signal = full_signal.iloc[os_e - os_s * 0 - (os_e - os_s):]
            # More robust: locate by date
            test_signal = full_signal.loc[test_returns.index]
            test_net = apply_costs(
                test_signal.fillna(0), test_returns,
                cfg.costs.transaction_cost, cfg.costs.slippage)

            oos_sharpe = sharpe_ratio(test_net.dropna(), cfg.risk_free_rate)

            wr = WindowResult(
                train_start=dates[ts], train_end=dates[te - 1],
                test_start=dates[os_s], test_end=dates[min(os_e - 1, len(dates) - 1)],
                strategy=strat_name,
                selected_params=best_params,
                oos_returns=test_net,
                oos_sharpe=oos_sharpe,
                is_signal=test_signal,
            )
            wf_res.window_results.append(wr)
            wf_res.param_history.append({
                "window": i,
                "test_start": wr.test_start,
                **best_params,
            })
            oos_parts.append(test_net)

        # Concatenate all OOS returns (non-overlapping by construction)
        wf_res.oos_returns = pd.concat(oos_parts).sort_index()
        # Remove duplicates (shouldn't happen but safety)
        wf_res.oos_returns = wf_res.oos_returns[
            ~wf_res.oos_returns.index.duplicated(keep="first")
        ]
        all_results[strat_name] = wf_res
        n_pos = sum(1 for wr in wf_res.window_results if wr.oos_sharpe > 0)
        n_tot = len(wf_res.window_results)
        print(f"[wf] {strat_name}: OOS Sharpe = "
              f"{sharpe_ratio(wf_res.oos_returns, cfg.risk_free_rate):.2f}, "
              f"positive periods = {n_pos}/{n_tot} "
              f"({n_pos / n_tot:.0%})")

    return all_results
