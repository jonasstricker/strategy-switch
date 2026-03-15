#!/usr/bin/env python3
"""
Enhanced Walk-Forward Analysis
===============================
Fixes:
  - Equity curves only compared from first OOS date (no 5-year zero padding)
  - Optimal weight combinations via grid search
  - Multi-ETF regional comparison (Brazil, China, Europe, Japan)
  - Hedged portfolio construction

Usage:  python -m wf_backtest.run_enhanced
"""

from __future__ import annotations

import sys
import os
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

if __name__ == "__main__" and __package__ is None:
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    __package__ = "wf_backtest"

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns

sns.set_theme(style="whitegrid", palette="muted", font_scale=1.1)

from .cfg import FrameworkConfig
from .data_loader import download_data, validate_data
from .data_multi import download_regional_etfs
from .walk_forward import run_walk_forward
from .stability import full_stability_analysis, positive_oos_fraction
from .switching import switching_summary
from .weights import optimal_hedged_portfolio, grid_search_weights
from .metrics import (
    sharpe_ratio, sortino_ratio, max_drawdown, cagr, calmar_ratio,
    drawdown_series, time_under_water, rolling_sharpe, summary_table,
)
from .report import (
    plot_equity_curves, plot_drawdowns, plot_rolling_sharpe,
    plot_parameter_history, plot_allocation,
    plot_bootstrap_distribution, plot_monte_carlo_distribution,
    plot_rolling_outperformance, plot_oos_sharpes, _ensure_dir,
)


PARAM_KEYS = {
    "Momentum": ["lookback"],
    "MA": ["period"],
    "RSI": ["period", "threshold"],
}

FIGSIZE = (14, 6)


def _equity(returns: pd.Series) -> pd.Series:
    return (1 + returns).cumprod()


def _metrics_row(eq, ret, rf):
    tuw = time_under_water(eq)
    return {
        "CAGR": cagr(eq),
        "Sharpe": sharpe_ratio(ret, rf),
        "Sortino": sortino_ratio(ret, rf),
        "Calmar": calmar_ratio(eq),
        "MaxDD": max_drawdown(eq),
        "MaxUW_d": tuw["max_days"],
        "AvgUW_d": tuw["avg_days"],
    }


# ══════════════════════════════════════════════════════════════════════════════
#  PART 1 – CORRECTED WALK-FORWARD  (only OOS period)
# ══════════════════════════════════════════════════════════════════════════════

def run_corrected_wf(cfg, out_dir):
    """Run WF and compute everything ONLY on the actual OOS period."""

    print("\n" + "=" * 70)
    print("  PART 1: CORRECTED WALK-FORWARD ANALYSIS")
    print("=" * 70)

    # Download
    df = download_data(cfg.data)
    validate_data(df)
    prices = df["Close"]
    returns = df["Returns"]

    # Walk-forward
    wf_results = run_walk_forward(prices, returns, cfg, use_vol_target=True)

    # Find the common OOS start date (= first OOS return for any strategy)
    oos_starts = []
    for name, res in wf_results.items():
        if len(res.oos_returns) > 0:
            oos_starts.append(res.oos_returns.index[0])
    oos_start = min(oos_starts)
    # Also find the common OOS end
    oos_ends = [res.oos_returns.index[-1] for res in wf_results.values()]
    oos_end = max(oos_ends)

    print(f"\n[CORRECTED] OOS period: {oos_start.date()} → {oos_end.date()}")
    oos_days = (oos_end - oos_start).days
    print(f"[CORRECTED] OOS duration: {oos_days} calendar days "
          f"({oos_days / 365.25:.1f} years)")

    # TRIM benchmark to OOS period only
    bench_ret_oos = returns.loc[oos_start:oos_end]
    bench_eq_oos = _equity(bench_ret_oos)

    # Trim strategy returns to OOS period
    strat_oos = {}
    for name, res in wf_results.items():
        trimmed = res.oos_returns.loc[oos_start:oos_end]
        strat_oos[name] = trimmed

    # Strategy switching
    sw = switching_summary(
        strat_oos, cfg.switching.rolling_window,
        cfg.switching.min_sharpe, cfg.risk_free_rate)

    # ── Corrected equity curves (all start at same date) ─────────────────────
    equity_curves = {"Buy & Hold": bench_eq_oos}
    return_series = {"Buy & Hold": bench_ret_oos}

    for name, ret in strat_oos.items():
        # Fill only missing trading days within OOS period (not before!)
        aligned = ret.reindex(bench_ret_oos.index).fillna(0)
        equity_curves[name] = _equity(aligned)
        return_series[name] = aligned

    for mode in ["hard", "soft"]:
        sr = sw[mode]["returns"]
        aligned = sr.reindex(bench_ret_oos.index).fillna(0)
        equity_curves[f"{mode.capitalize()} Switch"] = _equity(aligned)
        return_series[f"{mode.capitalize()} Switch"] = aligned

    # ── Performance Table (corrected) ────────────────────────────────────────
    perf_rows = {}
    for name in equity_curves:
        eq = equity_curves[name]
        ret = return_series[name]
        perf_rows[name] = _metrics_row(eq, ret, cfg.risk_free_rate)

    perf_df = pd.DataFrame(perf_rows).T
    perf_df.index.name = "Strategy"
    # Format for display
    perf_display = perf_df.copy()
    perf_display["CAGR"] = perf_display["CAGR"].apply(lambda x: f"{x:.2%}")
    perf_display["Sharpe"] = perf_display["Sharpe"].apply(lambda x: f"{x:.2f}")
    perf_display["Sortino"] = perf_display["Sortino"].apply(lambda x: f"{x:.2f}")
    perf_display["Calmar"] = perf_display["Calmar"].apply(lambda x: f"{x:.2f}")
    perf_display["MaxDD"] = perf_display["MaxDD"].apply(lambda x: f"{x:.2%}")
    perf_display["MaxUW_d"] = perf_display["MaxUW_d"].astype(int)
    perf_display["AvgUW_d"] = perf_display["AvgUW_d"].apply(lambda x: f"{x:.0f}")

    print("\n── CORRECTED Performance (OOS Period Only) ──")
    print(perf_display.to_string())

    # ── Plots ────────────────────────────────────────────────────────────────
    plot_equity_curves(equity_curves, title="Equity Curves (OOS Only)",
                       out_dir=out_dir)
    plot_drawdowns(equity_curves, out_dir=out_dir)
    plot_rolling_sharpe(return_series, window=126,
                        rf_annual=cfg.risk_free_rate, out_dir=out_dir)

    for name, res in wf_results.items():
        for pk in PARAM_KEYS[name]:
            plot_parameter_history(res.param_history, pk, name, out_dir=out_dir)
        plot_oos_sharpes(res.window_results, name, out_dir=out_dir)

    plot_allocation(sw["hard"]["allocation"],
                    title="Hard Switch Allocation", out_dir=out_dir)
    plot_allocation(sw["soft"]["allocation"],
                    title="Soft Switch Allocation", out_dir=out_dir)

    # Stability
    n_trials = (len(cfg.strategy.mom_lookbacks) +
                len(cfg.strategy.ma_periods) +
                len(cfg.strategy.rsi_periods) * len(cfg.strategy.rsi_thresholds))
    stability_results = {}
    for name, res in wf_results.items():
        stab = full_stability_analysis(
            oos_returns=res.oos_returns,
            param_history=res.param_history,
            param_keys=PARAM_KEYS[name],
            n_total_trials=n_trials,
            cfg_stability=cfg.stability,
            rf_annual=cfg.risk_free_rate)
        stability_results[name] = stab
        plot_bootstrap_distribution(stab["bootstrap"], label=name, out_dir=out_dir)
        plot_monte_carlo_distribution(stab["monte_carlo"], label=name,
                                      out_dir=out_dir)

    return {
        "wf_results": wf_results,
        "strat_oos": strat_oos,
        "bench_ret_oos": bench_ret_oos,
        "bench_eq_oos": bench_eq_oos,
        "sw": sw,
        "perf_df": perf_df,
        "perf_display": perf_display,
        "stability": stability_results,
        "equity_curves": equity_curves,
        "return_series": return_series,
        "oos_start": oos_start,
        "oos_end": oos_end,
        "prices": prices,
        "returns": returns,
    }


# ══════════════════════════════════════════════════════════════════════════════
#  PART 2 – OPTIMAL WEIGHT SEARCH
# ══════════════════════════════════════════════════════════════════════════════

def run_weight_optimization(part1, cfg, out_dir):
    """Find the best strategy combination that beats Buy & Hold."""

    print("\n" + "=" * 70)
    print("  PART 2: OPTIMAL WEIGHT COMBINATIONS")
    print("=" * 70)

    strat_oos = part1["strat_oos"]
    bench_ret_oos = part1["bench_ret_oos"]
    sw = part1["sw"]

    # Include switching strategies as candidates too
    candidates = dict(strat_oos)
    candidates["Hard Switch"] = sw["hard"]["returns"].reindex(
        bench_ret_oos.index).fillna(0)
    candidates["Soft Switch"] = sw["soft"]["returns"].reindex(
        bench_ret_oos.index).fillna(0)

    result = optimal_hedged_portfolio(
        candidates, bench_ret_oos, cfg.risk_free_rate)

    # Print results
    bench = result["benchmark"]
    print(f"\n  Common OOS period: {result['common_start'].date()} → "
          f"{result['common_end'].date()}")
    print(f"\n  Benchmark (Buy & Hold on this period):")
    print(f"    Sharpe: {bench['sharpe']:.2f}  |  CAGR: {bench['cagr']:.2%}  |  "
          f"MaxDD: {bench['mdd']:.2%}")

    for label, key in [("Best Sharpe", "best_sharpe"),
                       ("Best Calmar", "best_calmar"),
                       ("Best Balanced", "best_balanced")]:
        b = result[key]
        print(f"\n  {label}:")
        w_str = ", ".join(f"{n}: {b.get(n, 0):.0%}" for n in result["names"])
        print(f"    Weights: {w_str}")
        print(f"    Sharpe: {b['Sharpe']:.2f}  |  CAGR: {b['CAGR']:.2%}  |  "
              f"MaxDD: {b['MaxDD']:.2%}  |  Calmar: {b['Calmar']:.2f}")

    # Top-10 Sharpe combos
    print("\n  Top-10 Weight Combinations by Sharpe:")
    top10 = result["grid"].head(10)
    for i, row in top10.iterrows():
        w_str = ", ".join(f"{n}: {row[n]:.0%}" for n in result["names"])
        print(f"    #{i+1}: [{w_str}] → Sharpe={row['Sharpe']:.2f}, "
              f"CAGR={row['CAGR']:.2%}, MaxDD={row['MaxDD']:.2%}")

    # Plot equity: benchmark vs best balanced
    port_ret = result["portfolio_returns"]
    port_eq = _equity(port_ret)
    bench_eq = _equity(bench_ret_oos.loc[port_ret.index[0]:port_ret.index[-1]])

    fig, ax = plt.subplots(figsize=FIGSIZE)
    ax.plot(bench_eq.index, bench_eq.values, label="Buy & Hold", linewidth=1.5)
    ax.plot(port_eq.index, port_eq.values, label="Optimal Balanced",
            linewidth=1.5, linestyle="--")
    ax.set_title("Optimal Weighted Portfolio vs. Buy & Hold")
    ax.set_ylabel("Growth of $1")
    ax.legend()
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "optimal_weights_equity.png"), dpi=150)
    plt.close(fig)

    # Plot weight distribution heatmap (top 20 combos)
    top20 = result["grid"].head(20)
    fig, ax = plt.subplots(figsize=(14, 8))
    weight_data = top20[result["names"]].values
    sns.heatmap(weight_data, annot=True, fmt=".0%",
                xticklabels=result["names"],
                yticklabels=[f"#{i+1} (SR={top20.iloc[i]['Sharpe']:.2f})"
                             for i in range(len(top20))],
                cmap="YlOrRd", ax=ax, vmin=0, vmax=1)
    ax.set_title("Top-20 Weight Combinations (sorted by Sharpe)")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "weight_heatmap.png"), dpi=150)
    plt.close(fig)

    return result


# ══════════════════════════════════════════════════════════════════════════════
#  PART 3 – REGIONAL ETF COMPARISON
# ══════════════════════════════════════════════════════════════════════════════

def run_regional_comparison(cfg, out_dir):
    """Download and compare regional ETFs vs MSCI World."""

    print("\n" + "=" * 70)
    print("  PART 3: REGIONAL ETF COMPARISON vs. MSCI WORLD")
    print("=" * 70)

    etf_data = download_regional_etfs(
        start=cfg.data.start_date, end=cfg.data.end_date, min_years=8.0)

    if "MSCI World" not in etf_data:
        print("[ERROR] Could not download MSCI World ETF – aborting comparison")
        return None

    # Find common date range
    common_start = max(d.index[0] for d in etf_data.values())
    common_end = min(d.index[-1] for d in etf_data.values())
    print(f"\n[regional] Common period: {common_start.date()} → {common_end.date()}")

    trimmed = {}
    for region, df in etf_data.items():
        r = df.loc[common_start:common_end, "Returns"]
        trimmed[region] = r

    # Performance table
    rf = cfg.risk_free_rate
    perf_rows = {}
    for region, ret in trimmed.items():
        eq = _equity(ret)
        perf_rows[region] = _metrics_row(eq, ret, rf)

    perf_df = pd.DataFrame(perf_rows).T
    perf_df.index.name = "Region"
    perf_df = perf_df.sort_values("Sharpe", ascending=False)

    # Display
    disp = perf_df.copy()
    disp["CAGR"] = disp["CAGR"].apply(lambda x: f"{x:.2%}")
    disp["Sharpe"] = disp["Sharpe"].apply(lambda x: f"{x:.2f}")
    disp["Sortino"] = disp["Sortino"].apply(lambda x: f"{x:.2f}")
    disp["Calmar"] = disp["Calmar"].apply(lambda x: f"{x:.2f}")
    disp["MaxDD"] = disp["MaxDD"].apply(lambda x: f"{x:.2%}")
    disp["MaxUW_d"] = disp["MaxUW_d"].astype(int)
    disp["AvgUW_d"] = disp["AvgUW_d"].apply(lambda x: f"{x:.0f}")
    print("\n── Regional ETF Performance (Buy & Hold, common period) ──")
    print(disp.to_string())

    msci_ret = trimmed["MSCI World"]
    msci_sharpe = sharpe_ratio(msci_ret, rf)

    # Who beats MSCI?
    print(f"\n  MSCI World Sharpe: {msci_sharpe:.2f}")
    for region in perf_df.index:
        if region == "MSCI World":
            continue
        reg_sr = perf_df.loc[region, "Sharpe"]
        delta = reg_sr - msci_sharpe
        verdict = "BEATS" if delta > 0 else "LOSES"
        print(f"  {region:>12s}: Sharpe {reg_sr:.2f} ({delta:+.2f}) → {verdict}")

    # ── Run WF on each regional ETF that beats MSCI World ──
    print("\n── Walk-Forward on Regional ETFs ──")
    regional_wf = {}
    for region, df_full in etf_data.items():
        if region == "MSCI World":
            continue
        prices = df_full["Close"]
        returns = df_full["Returns"]
        if len(returns) < cfg.wf.train_days + cfg.wf.test_days + 100:
            print(f"  {region}: insufficient data for WF, skipping")
            continue
        print(f"\n  Running WF for {region}...")
        try:
            wf_res = run_walk_forward(prices, returns, cfg, use_vol_target=True)
            # Aggregate
            for strat_name, res in wf_res.items():
                if len(res.oos_returns) > 0:
                    sr = sharpe_ratio(res.oos_returns, rf)
                    n_pos = sum(1 for wr in res.window_results
                                if wr.oos_sharpe > 0)
                    n_tot = len(res.window_results)
                    print(f"    {strat_name}: OOS Sharpe={sr:.2f}, "
                          f"pos={n_pos}/{n_tot}")
            regional_wf[region] = wf_res
        except Exception as exc:
            print(f"    {region} WF failed: {exc}")

    # ── Equity Curves ────────────────────────────────────────────────────────
    eq_curves = {}
    for region, ret in trimmed.items():
        eq_curves[region] = _equity(ret)

    fig, ax = plt.subplots(figsize=FIGSIZE)
    for region, eq in eq_curves.items():
        lw = 2.0 if region == "MSCI World" else 1.2
        ax.plot(eq.index, eq.values, label=region, linewidth=lw)
    ax.set_title("Regional ETFs vs. MSCI World (Buy & Hold)")
    ax.set_ylabel("Growth of $1")
    ax.legend(loc="upper left")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "regional_equity.png"), dpi=150)
    plt.close(fig)

    # ── Rolling 1Y outperformance vs MSCI ────────────────────────────────────
    fig, ax = plt.subplots(figsize=FIGSIZE)
    for region, ret in trimmed.items():
        if region == "MSCI World":
            continue
        diff = ret - msci_ret
        rolling = diff.rolling(252, min_periods=252).sum()
        ax.plot(rolling.index, rolling.values, label=region, linewidth=1.0)
    ax.axhline(0, color="black", linewidth=0.5, linestyle="--")
    ax.set_title("Rolling 1-Year Outperformance vs. MSCI World")
    ax.set_ylabel("Excess Cumulative Return")
    ax.legend()
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "regional_outperformance.png"), dpi=150)
    plt.close(fig)

    # ── Drawdown comparison ──────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=FIGSIZE)
    for region, eq in eq_curves.items():
        dd = drawdown_series(eq)
        ax.fill_between(dd.index, dd.values, alpha=0.15)
        ax.plot(dd.index, dd.values, label=region, linewidth=0.8)
    ax.set_title("Drawdowns – Regional Comparison")
    ax.set_ylabel("Drawdown")
    ax.legend(loc="lower left")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "regional_drawdowns.png"), dpi=150)
    plt.close(fig)

    # ── Correlation matrix ───────────────────────────────────────────────────
    ret_df = pd.DataFrame(trimmed)
    corr = ret_df.corr()
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="RdYlGn_r",
                ax=ax, vmin=-1, vmax=1, center=0)
    ax.set_title("Return Correlation Matrix")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "regional_correlation.png"), dpi=150)
    plt.close(fig)

    # ── Can a regional mix beat MSCI? ────────────────────────────────────────
    print("\n── Optimal Regional Mix ──")
    non_msci = {k: v for k, v in trimmed.items() if k != "MSCI World"}
    if len(non_msci) >= 2:
        mix_grid = grid_search_weights(
            non_msci, msci_ret, rf, step=0.10, include_bench=True)
        print("\n  Top-5 Regional Mixes (step=10%):")
        for i, row in mix_grid.head(5).iterrows():
            names = [c for c in mix_grid.columns if c not in
                     ["Sharpe", "CAGR", "MaxDD", "Sortino", "Calmar"]]
            w_str = ", ".join(f"{n}: {row[n]:.0%}" for n in names)
            print(f"    #{i+1}: [{w_str}] → Sharpe={row['Sharpe']:.2f}, "
                  f"CAGR={row['CAGR']:.2%}, MaxDD={row['MaxDD']:.2%}")

    return {
        "etf_data": etf_data,
        "perf_df": perf_df,
        "regional_wf": regional_wf,
        "trimmed": trimmed,
    }


# ══════════════════════════════════════════════════════════════════════════════
#  PART 4 – COMPREHENSIVE REPORT
# ══════════════════════════════════════════════════════════════════════════════

def print_final_report(part1, weight_result, regional_result, cfg, out_dir):
    """Print and save the comprehensive final report."""

    print("\n" + "=" * 80)
    print("  FINAL COMPREHENSIVE REPORT")
    print("=" * 80)

    lines = []
    lines.append("=" * 80)
    lines.append("  WALK-FORWARD BACKTESTING FRAMEWORK – ENHANCED REPORT")
    lines.append("=" * 80)

    # PART 1: Corrected WF
    lines.append("\n\n══ PART 1: CORRECTED WF-ANALYSE (NUR OOS-ZEITRAUM) ══")
    lines.append(f"OOS Zeitraum: {part1['oos_start'].date()} → "
                 f"{part1['oos_end'].date()}")
    lines.append(f"Dauer: {(part1['oos_end'] - part1['oos_start']).days / 365.25:.1f} Jahre")
    lines.append("\n" + part1["perf_display"].to_string())

    # Stability summary
    lines.append("\n\n── Robustheitsanalyse ──")
    for name, stab in part1["stability"].items():
        lines.append(f"\n  {name}:")
        lines.append(f"    Observed Sharpe:     {stab['observed_sharpe']:.2f}")
        boot = stab["bootstrap"]
        lines.append(f"    Bootstrap 95% CI:    [{boot['ci_low']:.2f}, "
                      f"{boot['ci_high']:.2f}]")
        lines.append(f"    Bootstrap p-value:   {boot['p_value']:.3f}")
        dsr = stab["deflated_sharpe"]
        lines.append(f"    Deflated Sharpe:     {dsr['dsr']:.3f} "
                      f"({'sig.' if dsr['significant'] else 'n.s.'})")
        mc = stab["monte_carlo"]
        lines.append(f"    Monte Carlo p:       {mc['p_value']:.3f} "
                      f"({'sig.' if mc['significant_5pct'] else 'n.s.'})")
        ps = stab["parameter_stability"]
        if not ps.empty:
            for _, row in ps.iterrows():
                lines.append(f"    Param {row['param']:>10s}: "
                              f"CV={row['cv']:.2f}")

    # PART 2: Weights
    if weight_result is not None:
        lines.append("\n\n══ PART 2: OPTIMALE GEWICHTUNG ══")
        bench = weight_result["benchmark"]
        lines.append(f"Benchmark (gleicher Zeitraum): "
                    f"Sharpe={bench['sharpe']:.2f}, CAGR={bench['cagr']:.2%}, "
                    f"MaxDD={bench['mdd']:.2%}")

        for label, key in [("Bestes Sharpe-Portfolio", "best_sharpe"),
                           ("Bestes Calmar-Portfolio", "best_calmar"),
                           ("Bestes Balanced-Portfolio", "best_balanced")]:
            b = weight_result[key]
            lines.append(f"\n  {label}:")
            w_str = ", ".join(f"{n}: {b.get(n, 0):.0%}"
                              for n in weight_result["names"])
            lines.append(f"    Gewichte: {w_str}")
            lines.append(f"    Sharpe: {b['Sharpe']:.2f}  |  "
                        f"CAGR: {b['CAGR']:.2%}  |  MaxDD: {b['MaxDD']:.2%}")

    # PART 3: Regional
    if regional_result is not None:
        lines.append("\n\n══ PART 3: REGIONALE ETF-ANALYSE ══")
        rperf = regional_result["perf_df"]
        disp = rperf.copy()
        disp["CAGR"] = disp["CAGR"].apply(lambda x: f"{x:.2%}")
        disp["Sharpe"] = disp["Sharpe"].apply(lambda x: f"{x:.2f}")
        disp["Sortino"] = disp["Sortino"].apply(lambda x: f"{x:.2f}")
        disp["Calmar"] = disp["Calmar"].apply(lambda x: f"{x:.2f}")
        disp["MaxDD"] = disp["MaxDD"].apply(lambda x: f"{x:.2%}")
        disp["MaxUW_d"] = disp["MaxUW_d"].astype(int)
        disp["AvgUW_d"] = disp["AvgUW_d"].apply(lambda x: f"{x:.0f}")
        lines.append("\n" + disp.to_string())

        msci_sr = rperf.loc["MSCI World", "Sharpe"] if "MSCI World" in rperf.index else 0
        lines.append(f"\nMSCI World Sharpe: {msci_sr:.2f}")
        for region in rperf.index:
            if region == "MSCI World":
                continue
            sr = rperf.loc[region, "Sharpe"]
            delta = sr - msci_sr
            lines.append(f"  {region:>10s}: {sr:.2f} ({delta:+.2f}) → "
                        f"{'ÜBERTRIFFT' if delta > 0 else 'UNTERLIEGT'}")

    # FINAL RECOMMENDATION
    lines.append("\n\n══ ABSCHLIESSENDE EMPFEHLUNG ══")
    lines.append("-" * 40)

    bench_sr = part1["perf_df"].loc["Buy & Hold", "Sharpe"]

    # Find best of ALL options
    all_options = {}
    for name in part1["perf_df"].index:
        all_options[name] = part1["perf_df"].loc[name, "Sharpe"]

    if weight_result is not None:
        bb = weight_result["best_balanced"]
        all_options["Optimal Balanced"] = bb["Sharpe"]

    best_name = max(all_options, key=all_options.get)
    best_sr = all_options[best_name]

    lines.append(f"\n  Benchmark Sharpe (OOS):  {bench_sr:.2f}")
    lines.append(f"  Bestes System:           {best_name} (Sharpe {best_sr:.2f})")
    lines.append(f"  Sharpe-Differenz:        {best_sr - bench_sr:+.2f}")

    if best_sr > bench_sr + 0.1:
        lines.append("\n  → MODERATE EVIDENZ für taktische Allokation (10-15%)")
    elif best_sr > bench_sr:
        lines.append("\n  → SCHWACHE EVIDENZ – max. 5-10% taktisch, Rest Buy & Hold")
    else:
        lines.append("\n  → KEINE EVIDENZ – Buy & Hold bleibt überlegen")

    lines.append("\n  Portfolio-Vorschlag:")
    lines.append("    80-90%  MSCI World Buy & Hold (Kernallokation)")
    if weight_result and weight_result["best_balanced"]["Sharpe"] > bench_sr:
        bb = weight_result["best_balanced"]
        tac = ", ".join(f"{n}: {bb.get(n, 0):.0%}"
                        for n in weight_result["names"] if bb.get(n, 0) > 0)
        lines.append(f"    10-20%  Taktisch: [{tac}]")
    else:
        lines.append("    10-20%  Diversifikation über Regionen (falls Korrelation < 0.8)")

    lines.append("")
    lines.append("=" * 80)

    report = "\n".join(lines)
    print(report)

    path = os.path.join(out_dir, "enhanced_report.txt")
    with open(path, "w", encoding="utf-8") as f:
        f.write(report)
    print(f"\n  Report: {path}")


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    cfg = FrameworkConfig()
    out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")
    os.makedirs(out_dir, exist_ok=True)

    # PART 1 – Corrected WF
    part1 = run_corrected_wf(cfg, out_dir)

    # PART 2 – Optimal Weights
    weight_result = run_weight_optimization(part1, cfg, out_dir)

    # PART 3 – Regional ETF Comparison
    regional_result = run_regional_comparison(cfg, out_dir)

    # PART 4 – Final Report
    print_final_report(part1, weight_result, regional_result, cfg, out_dir)

    print(f"\n  All charts saved to: {out_dir}/")
    print("  DONE.")


if __name__ == "__main__":
    main()
