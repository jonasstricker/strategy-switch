"""
Stability & Robustness Analysis
=================================
- Parameter variance over time
- Bootstrap Sharpe CI
- Deflated Sharpe Ratio (Bailey & López de Prado 2014)
- Monte Carlo resampling
"""

from __future__ import annotations

from typing import Dict, List

import numpy as np
import pandas as pd
from scipy import stats as sp_stats

from .metrics import sharpe_ratio


# ── Parameter stability ──────────────────────────────────────────────────────

def parameter_stability(param_history: List[dict],
                        param_keys: List[str]) -> pd.DataFrame:
    """
    Compute rolling coefficient of variation (CV) of each parameter
    selected across walk-forward windows.

    Returns a DataFrame with columns: param, mean, std, cv.
    """
    df = pd.DataFrame(param_history)
    rows = []
    for key in param_keys:
        if key not in df.columns:
            continue
        vals = df[key].astype(float)
        m, s = vals.mean(), vals.std()
        cv = s / m if m != 0 else np.nan
        rows.append({"param": key, "mean": m, "std": s, "cv": cv})
    return pd.DataFrame(rows)


def rolling_parameter_change(param_history: List[dict],
                             param_key: str) -> pd.Series:
    """Absolute change of a parameter between consecutive windows."""
    df = pd.DataFrame(param_history)
    if param_key not in df.columns:
        return pd.Series(dtype=float)
    vals = df[param_key].astype(float)
    return vals.diff().abs()


# ── Fraction of positive OOS windows ────────────────────────────────────────

def positive_oos_fraction(oos_sharpes: List[float]) -> float:
    """Fraction of OOS windows with Sharpe > 0."""
    if not oos_sharpes:
        return 0.0
    return sum(1 for s in oos_sharpes if s > 0) / len(oos_sharpes)


# ── Bootstrap (block bootstrap) ─────────────────────────────────────────────

def block_bootstrap_sharpe(returns: pd.Series,
                           n_bootstrap: int = 1_000,
                           block_length: int = 21,
                           rf_annual: float = 0.02,
                           confidence: float = 0.95) -> dict:
    """
    Block-bootstrap the Sharpe ratio to get confidence intervals.
    Using non-overlapping blocks preserves autocorrelation structure.
    """
    ret_arr = returns.dropna().values
    n = len(ret_arr)
    if n < block_length * 2:
        return {"observed": 0.0, "mean": 0.0, "ci_low": 0.0, "ci_high": 0.0,
                "p_value": 1.0}

    n_blocks = n // block_length
    observed = sharpe_ratio(returns, rf_annual)

    rng = np.random.default_rng(42)
    boot_sharpes = []
    for _ in range(n_bootstrap):
        # Sample blocks with replacement
        block_starts = rng.integers(0, n - block_length, size=n_blocks)
        sample = np.concatenate([
            ret_arr[s:s + block_length] for s in block_starts
        ])
        sr = _sharpe_from_array(sample, rf_annual)
        boot_sharpes.append(sr)

    boot_arr = np.array(boot_sharpes)
    alpha = 1 - confidence
    ci_low = np.percentile(boot_arr, 100 * alpha / 2)
    ci_high = np.percentile(boot_arr, 100 * (1 - alpha / 2))
    # p-value: fraction of bootstrap samples ≤ 0
    p_value = np.mean(boot_arr <= 0)

    return {
        "observed": observed,
        "mean": float(np.mean(boot_arr)),
        "std": float(np.std(boot_arr)),
        "ci_low": ci_low,
        "ci_high": ci_high,
        "p_value": p_value,
        "distribution": boot_arr,
    }


def _sharpe_from_array(ret: np.ndarray, rf_annual: float) -> float:
    """Sharpe from numpy array."""
    rf_daily = (1 + rf_annual) ** (1 / 252) - 1
    excess = ret - rf_daily
    std = excess.std()
    if std == 0:
        return 0.0
    return float(np.sqrt(252) * excess.mean() / std)


# ── Deflated Sharpe Ratio ────────────────────────────────────────────────────

def deflated_sharpe_ratio(observed_sr: float,
                          n_trials: int,
                          n_obs: int,
                          skewness: float,
                          excess_kurtosis: float) -> dict:
    """
    Bailey & López de Prado (2014) Deflated Sharpe Ratio.

    Tests whether the observed Sharpe is significant after adjusting
    for multiple testing (n_trials parameter combinations tried).

    Returns dict with DSR probability + intermediate values.
    """
    # Variance of Sharpe estimator
    sr_var = (1 - skewness * observed_sr
              + (excess_kurtosis - 1) / 4 * observed_sr ** 2) / n_obs
    sr_std = np.sqrt(max(sr_var, 1e-12))

    # Expected maximum Sharpe under null (Euler-Mascheroni approximation)
    gamma_em = 0.5772156649  # Euler–Mascheroni constant
    if n_trials <= 1:
        e_max_sr = 0.0
    else:
        z = sp_stats.norm.ppf(1 - 1 / n_trials)
        z2 = sp_stats.norm.ppf(1 - 1 / (n_trials * np.e))
        e_max_sr = sr_std * ((1 - gamma_em) * z + gamma_em * z2)

    # DSR = Prob that observed SR > expected max SR
    if sr_std == 0:
        dsr = 0.0
    else:
        dsr = float(sp_stats.norm.cdf((observed_sr - e_max_sr) / sr_std))

    return {
        "dsr": dsr,
        "observed_sr": observed_sr,
        "expected_max_sr": e_max_sr,
        "sr_std": sr_std,
        "n_trials": n_trials,
        "significant": dsr > 0.95,
    }


# ── Monte Carlo resampling ──────────────────────────────────────────────────

def monte_carlo_sharpe(returns: pd.Series,
                       n_sims: int = 1_000,
                       rf_annual: float = 0.02) -> dict:
    """
    Shuffle daily returns to destroy any timing skill.
    If the walk-forward Sharpe significantly exceeds the MC distribution,
    the strategy adds value beyond random rebalancing.
    """
    ret_arr = returns.dropna().values
    observed = sharpe_ratio(returns, rf_annual)

    rng = np.random.default_rng(42)
    mc_sharpes = []
    for _ in range(n_sims):
        shuffled = rng.permutation(ret_arr)
        sr = _sharpe_from_array(shuffled, rf_annual)
        mc_sharpes.append(sr)

    mc_arr = np.array(mc_sharpes)
    p_value = np.mean(mc_arr >= observed)

    return {
        "observed": observed,
        "mc_mean": float(np.mean(mc_arr)),
        "mc_std": float(np.std(mc_arr)),
        "p_value": p_value,
        "significant_5pct": p_value < 0.05,
        "distribution": mc_arr,
    }


# ── Full stability report ───────────────────────────────────────────────────

def full_stability_analysis(oos_returns: pd.Series,
                            param_history: List[dict],
                            param_keys: List[str],
                            n_total_trials: int,
                            cfg_stability,
                            rf_annual: float = 0.02) -> dict:
    """
    Run all stability checks.
    Returns a nested dict with all results.
    """
    ret = oos_returns.dropna()

    # Parameter stability
    p_stab = parameter_stability(param_history, param_keys)

    # OOS positive fraction
    # (need per-window Sharpe – computed externally, here we use overall)
    observed_sr = sharpe_ratio(ret, rf_annual)

    # Bootstrap
    boot = block_bootstrap_sharpe(
        ret, cfg_stability.n_bootstrap, cfg_stability.block_length,
        rf_annual, cfg_stability.confidence_level)

    # Deflated Sharpe
    skew = float(sp_stats.skew(ret.values))
    kurt = float(sp_stats.kurtosis(ret.values))  # excess kurtosis
    dsr = deflated_sharpe_ratio(observed_sr, n_total_trials,
                                len(ret), skew, kurt)

    # Monte Carlo
    mc = monte_carlo_sharpe(ret, cfg_stability.n_monte_carlo, rf_annual)

    return {
        "parameter_stability": p_stab,
        "bootstrap": boot,
        "deflated_sharpe": dsr,
        "monte_carlo": mc,
        "observed_sharpe": observed_sr,
    }
