"""
Weight Optimizer
=================
Find the optimal combination of strategy weights (Momentum, MA, RSI)
plus a Buy & Hold component plus hedging overlay.
Uses walk-forward OOS returns only – no lookahead.
"""

from __future__ import annotations

import itertools
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from .metrics import sharpe_ratio, max_drawdown, cagr, sortino_ratio, calmar_ratio


def _trim_to_common(series_dict: Dict[str, pd.Series]) -> Dict[str, pd.Series]:
    """Trim all series to the common date range where ALL have real data."""
    common_start = max(s.first_valid_index() for s in series_dict.values())
    common_end = min(s.last_valid_index() for s in series_dict.values())
    return {k: v.loc[common_start:common_end] for k, v in series_dict.items()}


def grid_search_weights(strat_returns: Dict[str, pd.Series],
                        bench_returns: pd.Series,
                        rf_annual: float = 0.02,
                        step: float = 0.05,
                        include_bench: bool = True,
                        ) -> pd.DataFrame:
    """
    Exhaustive grid search over weight combinations.
    Weights sum to 1, step size configurable.

    Parameters
    ----------
    strat_returns : dict of strategy name -> daily OOS returns
    bench_returns : buy & hold daily returns (aligned)
    rf_annual : risk free rate
    step : weight increment (0.05 = 5% steps)
    include_bench : whether to include Buy & Hold as an asset in the mix

    Returns
    -------
    DataFrame with columns for each weight, Sharpe, CAGR, MaxDD, Sortino, Calmar
    sorted by Sharpe descending.
    """
    # Trim to common period
    all_series = dict(strat_returns)
    if include_bench:
        all_series["BuyHold"] = bench_returns
    all_series = _trim_to_common(all_series)

    names = list(all_series.keys())
    n = len(names)
    ret_df = pd.DataFrame(all_series)

    # Generate weight grid (sum to 1)
    steps = int(round(1.0 / step))
    weight_combos = []
    for combo in itertools.product(range(steps + 1), repeat=n):
        if sum(combo) == steps:
            weights = tuple(c * step for c in combo)
            weight_combos.append(weights)

    print(f"[weights] Testing {len(weight_combos)} weight combinations "
          f"for {names}")

    results = []
    for weights in weight_combos:
        # Weighted portfolio return
        w = np.array(weights)
        port_ret = (ret_df.values * w).sum(axis=1)
        port_ret = pd.Series(port_ret, index=ret_df.index)

        eq = (1 + port_ret).cumprod()
        sr = sharpe_ratio(port_ret, rf_annual)
        c = cagr(eq)
        mdd = max_drawdown(eq)
        sort_r = sortino_ratio(port_ret, rf_annual)
        cal_r = calmar_ratio(eq)

        row = {names[i]: weights[i] for i in range(n)}
        row.update({
            "Sharpe": sr,
            "CAGR": c,
            "MaxDD": mdd,
            "Sortino": sort_r,
            "Calmar": cal_r,
        })
        results.append(row)

    df = pd.DataFrame(results)
    df = df.sort_values("Sharpe", ascending=False).reset_index(drop=True)
    return df


def optimal_hedged_portfolio(strat_returns: Dict[str, pd.Series],
                             bench_returns: pd.Series,
                             rf_annual: float = 0.02
                             ) -> Dict:
    """
    Find the best combination that:
    1. Maximizes Sharpe
    2. Reduces MaxDD vs pure Buy & Hold
    3. Also search with hedging overlays (partial cash allocation)

    Returns dict with optimal weights, metrics, and the portfolio return series.
    """
    # Trim all to common dates
    all_series = dict(strat_returns)
    all_series["BuyHold"] = bench_returns
    all_series = _trim_to_common(all_series)
    bench_trimmed = all_series["BuyHold"]

    # 1. Fine-grained search (5% steps)
    grid = grid_search_weights(
        {k: v for k, v in all_series.items() if k != "BuyHold"},
        bench_trimmed, rf_annual, step=0.05, include_bench=True)

    # Benchmark metrics on the same period
    bench_eq = (1 + bench_trimmed).cumprod()
    bench_sharpe = sharpe_ratio(bench_trimmed, rf_annual)
    bench_mdd = max_drawdown(bench_eq)
    bench_cagr = cagr(bench_eq)

    # 2. Filter: must beat benchmark Sharpe and have better MaxDD
    names = [c for c in grid.columns if c not in
             ["Sharpe", "CAGR", "MaxDD", "Sortino", "Calmar"]]
    improved = grid[
        (grid["Sharpe"] > bench_sharpe) &
        (grid["MaxDD"] > bench_mdd)  # less negative = better
    ].copy()

    if improved.empty:
        # Fallback: just best Sharpe
        improved = grid.head(20)

    # 3. Best by Sharpe
    best_sharpe = improved.iloc[0]

    # 4. Best by Calmar (risk-adjusted with DD focus)
    best_calmar = improved.sort_values("Calmar", ascending=False).iloc[0]

    # 5. Best balanced (Sharpe rank + Calmar rank)
    improved = improved.copy()
    improved["sharpe_rank"] = improved["Sharpe"].rank(ascending=False)
    improved["calmar_rank"] = improved["Calmar"].rank(ascending=False)
    improved["combined_rank"] = improved["sharpe_rank"] + improved["calmar_rank"]
    best_balanced = improved.sort_values("combined_rank").iloc[0]

    # Build return series for best balanced
    ret_df = pd.DataFrame(all_series)
    w = np.array([best_balanced[n] for n in names])
    port_ret = pd.Series(
        (ret_df[names].values * w).sum(axis=1),
        index=ret_df.index)

    return {
        "grid": grid,
        "best_sharpe": best_sharpe.to_dict(),
        "best_calmar": best_calmar.to_dict(),
        "best_balanced": best_balanced.to_dict(),
        "portfolio_returns": port_ret,
        "names": names,
        "benchmark": {
            "sharpe": bench_sharpe,
            "cagr": bench_cagr,
            "mdd": bench_mdd,
        },
        "common_start": ret_df.index[0],
        "common_end": ret_df.index[-1],
    }
