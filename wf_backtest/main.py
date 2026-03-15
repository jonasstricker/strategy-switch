#!/usr/bin/env python3
"""
Walk-Forward Backtesting Framework – MSCI World
=================================================
Main entry point. Downloads data, runs walk-forward optimisation
for three strategies (Momentum, MA, RSI), performs stability analysis,
strategy switching, and generates a comprehensive report.

Usage:
    python -m wf_backtest.main
    # or
    python main.py  (from within the wf_backtest/ directory)
"""

from __future__ import annotations

import sys
import os
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)

# Allow running both as module and as script
if __name__ == "__main__" and __package__ is None:
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    __package__ = "wf_backtest"

import numpy as np
import pandas as pd

from .cfg import FrameworkConfig
from .data_loader import download_data, validate_data
from .walk_forward import run_walk_forward
from .stability import (
    full_stability_analysis, positive_oos_fraction,
)
from .switching import switching_summary
from .metrics import (
    sharpe_ratio, sortino_ratio, max_drawdown, cagr, calmar_ratio,
    summary_table, rolling_sharpe, time_under_water,
)
from .report import (
    plot_equity_curves, plot_drawdowns, plot_rolling_sharpe,
    plot_parameter_history, plot_allocation,
    plot_bootstrap_distribution, plot_monte_carlo_distribution,
    plot_rolling_outperformance, plot_oos_sharpes,
    print_full_report,
)


# ── Helpers ──────────────────────────────────────────────────────────────────

def _equity_from_returns(returns: pd.Series) -> pd.Series:
    """Convert daily returns to a growth-of-$1 equity curve."""
    return (1 + returns).cumprod()


def _count_total_trials(cfg: FrameworkConfig) -> int:
    """Total number of parameter combos evaluated across all strategies."""
    n_mom = len(cfg.strategy.mom_lookbacks)
    n_ma = len(cfg.strategy.ma_periods)
    n_rsi = len(cfg.strategy.rsi_periods) * len(cfg.strategy.rsi_thresholds)
    return n_mom + n_ma + n_rsi


PARAM_KEYS = {
    "Momentum": ["lookback"],
    "MA": ["period"],
    "RSI": ["period", "threshold"],
}


# ── Generate Recommendation ─────────────────────────────────────────────────

def generate_recommendation(
    bench_sharpe: float,
    bench_mdd: float,
    strat_results: dict,
    stability_results: dict,
    switching_results: dict,
) -> str:
    """
    Build a text recommendation based on quantitative criteria:
      - OOS Sharpe > Benchmark + 0.2
      - MaxDD ≥ 25% lower
      - > 60% positive OOS windows
      - Parameters stable
      - Statistically significant (DSR, MC)
    """
    lines = []
    lines.append("PORTFOLIO ALLOCATION RECOMMENDATION")
    lines.append("-" * 40)

    # Find best switching mode
    best_mode = max(["hard", "soft"],
                    key=lambda m: switching_results[m]["sharpe"])
    best_sr = switching_results[best_mode]["sharpe"]

    # Check criteria
    sharpe_pass = best_sr > bench_sharpe + 0.2
    lines.append(f"  1. OOS Sharpe ({best_sr:.2f}) > Bench+0.2 "
                 f"({bench_sharpe + 0.2:.2f}): "
                 f"{'PASS' if sharpe_pass else 'FAIL'}")

    # Check MaxDD of switching strategy
    sw_ret = switching_results[best_mode]["returns"]
    sw_eq = _equity_from_returns(sw_ret.dropna())
    sw_mdd = abs(max_drawdown(sw_eq))
    mdd_reduction = 1 - sw_mdd / abs(bench_mdd) if bench_mdd != 0 else 0
    mdd_pass = mdd_reduction >= 0.25
    lines.append(f"  2. MaxDD reduction ({mdd_reduction:.0%}): "
                 f"{'PASS' if mdd_pass else 'FAIL'}")

    # Check OOS positive fraction per strategy
    oos_fracs = {}
    all_pass_oos = True
    for name, res in strat_results.items():
        sharpes = [wr.oos_sharpe for wr in res.window_results]
        frac = positive_oos_fraction(sharpes)
        oos_fracs[name] = frac
        if frac < 0.60:
            all_pass_oos = False
    frac_str = ", ".join(f"{n}: {f:.0%}" for n, f in oos_fracs.items())
    lines.append(f"  3. OOS > 60% positive ({frac_str}): "
                 f"{'PASS' if all_pass_oos else 'PARTIAL'}")

    # Check parameter stability (CV < 0.3 is "stable")
    stable = True
    for name, res in stability_results.items():
        ps = res["parameter_stability"]
        if not ps.empty and (ps["cv"] > 0.30).any():
            stable = False
    lines.append(f"  4. Parameter stability (CV < 0.30): "
                 f"{'PASS' if stable else 'FAIL'}")

    # Check statistical significance
    sig = True
    for name, res in stability_results.items():
        if not res["deflated_sharpe"]["significant"]:
            sig = False
        if not res["monte_carlo"]["significant_5pct"]:
            sig = False
    lines.append(f"  5. Statistical significance (DSR + MC): "
                 f"{'PASS' if sig else 'FAIL'}")

    # Overall
    n_pass = sum([sharpe_pass, mdd_pass, all_pass_oos, stable, sig])
    lines.append(f"\n  Score: {n_pass}/5 criteria passed")

    if n_pass >= 4:
        alloc = "15-20%"
        conf = "HIGH"
    elif n_pass >= 3:
        alloc = "10-15%"
        conf = "MODERATE"
    elif n_pass >= 2:
        alloc = "5-10%"
        conf = "LOW"
    else:
        alloc = "0% (DO NOT DEPLOY)"
        conf = "NONE"

    lines.append(f"\n  Confidence: {conf}")
    lines.append(f"  Recommended allocation: {alloc} of total portfolio")
    lines.append(f"  Preferred switching mode: {best_mode.capitalize()}")

    if n_pass < 5:
        lines.append("\n  WARNINGS:")
        if not sharpe_pass:
            lines.append("    - Sharpe improvement insufficient after costs")
        if not mdd_pass:
            lines.append("    - Drawdown reduction below target")
        if not all_pass_oos:
            lines.append("    - Some strategies have < 60% positive OOS windows")
        if not stable:
            lines.append("    - Parameter instability detected (high CV)")
        if not sig:
            lines.append("    - Statistical significance not fully established")

    lines.append("\n  IMPLEMENTATION NOTES:")
    lines.append("    - Rebalance quarterly (aligned with walk-forward step)")
    lines.append("    - Monitor rolling 6M Sharpe for regime changes")
    lines.append("    - Combine with core Buy & Hold allocation (80-90%)")
    lines.append("    - Re-run walk-forward annually with fresh data")
    lines.append("    - Consider tax implications of trading frequency")

    return "\n".join(lines)


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    cfg = FrameworkConfig()
    out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")

    # ── 1. Data ──────────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("  STEP 1 / 6 – Downloading data")
    print("=" * 70)
    df = download_data(cfg.data)
    validate_data(df)
    prices = df["Close"]
    returns = df["Returns"]

    # Benchmark: buy & hold
    bench_returns = returns.copy()
    bench_equity = _equity_from_returns(bench_returns)

    # ── 2. Walk-Forward Optimisation ─────────────────────────────────────────
    print("\n" + "=" * 70)
    print("  STEP 2 / 6 – Walk-Forward Optimisation")
    print("=" * 70)
    wf_results = run_walk_forward(prices, returns, cfg, use_vol_target=True)

    # ── 3. Strategy Switching ────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("  STEP 3 / 6 – Strategy Switching")
    print("=" * 70)
    strat_ret_dict = {
        name: res.oos_returns for name, res in wf_results.items()
    }
    sw = switching_summary(
        strat_ret_dict, cfg.switching.rolling_window,
        cfg.switching.min_sharpe, cfg.risk_free_rate)
    print(f"  Hard Switch Sharpe: {sw['hard']['sharpe']:.2f}")
    print(f"  Soft Switch Sharpe: {sw['soft']['sharpe']:.2f}")

    # ── 4. Stability Analysis ────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("  STEP 4 / 6 – Stability & Robustness Analysis")
    print("=" * 70)
    n_trials = _count_total_trials(cfg)
    stability_results = {}
    for name, res in wf_results.items():
        print(f"  Analysing {name} …")
        stab = full_stability_analysis(
            oos_returns=res.oos_returns,
            param_history=res.param_history,
            param_keys=PARAM_KEYS[name],
            n_total_trials=n_trials,
            cfg_stability=cfg.stability,
            rf_annual=cfg.risk_free_rate,
        )
        stability_results[name] = stab

    # ── 5. Reports & Plots ───────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("  STEP 5 / 6 – Generating Reports & Charts")
    print("=" * 70)

    # Build equity curves
    equity_curves = {"Buy & Hold": bench_equity}
    return_series = {"Buy & Hold": bench_returns}
    for name, res in wf_results.items():
        # Align to benchmark dates
        aligned = res.oos_returns.reindex(bench_returns.index).fillna(0)
        eq = _equity_from_returns(aligned)
        equity_curves[name] = eq
        return_series[name] = aligned

    # Switching equity curves
    for mode in ["hard", "soft"]:
        sr = sw[mode]["returns"]
        aligned = sr.reindex(bench_returns.index).fillna(0)
        eq = _equity_from_returns(aligned)
        equity_curves[f"{mode.capitalize()} Switch"] = eq
        return_series[f"{mode.capitalize()} Switch"] = aligned

    # Plots
    plot_equity_curves(equity_curves, out_dir=out_dir)
    plot_drawdowns(equity_curves, out_dir=out_dir)
    plot_rolling_sharpe(return_series, window=126,
                        rf_annual=cfg.risk_free_rate, out_dir=out_dir)

    for name, res in wf_results.items():
        for pk in PARAM_KEYS[name]:
            plot_parameter_history(res.param_history, pk, name, out_dir=out_dir)
        plot_oos_sharpes(res.window_results, name, out_dir=out_dir)

    # Switching allocation plots
    plot_allocation(sw["hard"]["allocation"],
                    title="Hard Switch Allocation", out_dir=out_dir)
    plot_allocation(sw["soft"]["allocation"],
                    title="Soft Switch Allocation", out_dir=out_dir)

    # Bootstrap / MC plots for best switching mode
    best_mode = max(["hard", "soft"], key=lambda m: sw[m]["sharpe"])
    for name, stab in stability_results.items():
        plot_bootstrap_distribution(stab["bootstrap"], label=name, out_dir=out_dir)
        plot_monte_carlo_distribution(stab["monte_carlo"], label=name, out_dir=out_dir)

    # Rolling outperformance
    best_sw_key = f"{best_mode.capitalize()} Switch"
    if best_sw_key in return_series:
        plot_rolling_outperformance(
            return_series[best_sw_key], bench_returns,
            window=252, label=best_sw_key, out_dir=out_dir)

    # Performance table
    best_sw_ret = return_series[best_sw_key]
    best_sw_eq = equity_curves[best_sw_key]
    perf_table = summary_table(
        best_sw_eq, best_sw_ret,
        bench_equity, bench_returns,
        rf_annual=cfg.risk_free_rate,
        label=best_sw_key,
    )

    # Add individual strategies to table
    for name in wf_results:
        if name in equity_curves:
            eq = equity_curves[name]
            ret = return_series[name]
            row = {
                "CAGR": f"{cagr(eq):.2%}",
                "Sharpe": f"{sharpe_ratio(ret, cfg.risk_free_rate):.2f}",
                "Sortino": f"{sortino_ratio(ret, cfg.risk_free_rate):.2f}",
                "Calmar": f"{calmar_ratio(eq):.2f}",
                "Max DD": f"{max_drawdown(eq):.2%}",
                "Max UW (d)": str(time_under_water(eq)["max_days"]),
                "Avg UW (d)": f"{time_under_water(eq)['avg_days']:.0f}",
            }
            perf_table.loc[name] = row

    # ── 6. Recommendation ────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("  STEP 6 / 6 – Generating Recommendation")
    print("=" * 70)

    bench_sr = sharpe_ratio(bench_returns, cfg.risk_free_rate)
    bench_mdd = max_drawdown(bench_equity)
    recommendation = generate_recommendation(
        bench_sr, bench_mdd, wf_results,
        stability_results, sw)

    # Print full text report
    report_text = print_full_report(
        perf_table, stability_results, sw, recommendation)

    # Save report
    report_path = os.path.join(out_dir, "report.txt")
    os.makedirs(out_dir, exist_ok=True)
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report_text)
    print(f"\n  Report saved to: {report_path}")
    print(f"  Charts saved to: {out_dir}/")


if __name__ == "__main__":
    main()
