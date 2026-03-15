"""
Report Generation
==================
All charts and printed summaries. Saves PNGs to an output directory.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")                        # non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns

from .metrics import (
    cagr, sharpe_ratio, sortino_ratio, calmar_ratio,
    max_drawdown, drawdown_series, time_under_water,
    rolling_sharpe, rolling_outperformance, summary_table,
)

sns.set_theme(style="whitegrid", palette="muted", font_scale=1.1)
FIGSIZE = (14, 6)


def _ensure_dir(path: str) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


# ── Equity curves ────────────────────────────────────────────────────────────

def plot_equity_curves(curves: Dict[str, pd.Series],
                       title: str = "Equity Curves",
                       out_dir: str = "output") -> str:
    d = _ensure_dir(out_dir)
    fig, ax = plt.subplots(figsize=FIGSIZE)
    for name, eq in curves.items():
        ax.plot(eq.index, eq.values, label=name, linewidth=1.2)
    ax.set_title(title)
    ax.set_ylabel("Growth of $1")
    ax.legend(loc="upper left")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    fig.tight_layout()
    fp = str(d / "equity_curves.png")
    fig.savefig(fp, dpi=150)
    plt.close(fig)
    return fp


# ── Drawdowns ────────────────────────────────────────────────────────────────

def plot_drawdowns(curves: Dict[str, pd.Series],
                   out_dir: str = "output") -> str:
    d = _ensure_dir(out_dir)
    fig, ax = plt.subplots(figsize=FIGSIZE)
    for name, eq in curves.items():
        dd = drawdown_series(eq)
        ax.fill_between(dd.index, dd.values, alpha=0.3, label=name)
        ax.plot(dd.index, dd.values, linewidth=0.8)
    ax.set_title("Drawdown")
    ax.set_ylabel("Drawdown (%)")
    ax.legend(loc="lower left")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    fig.tight_layout()
    fp = str(d / "drawdowns.png")
    fig.savefig(fp, dpi=150)
    plt.close(fig)
    return fp


# ── Rolling Sharpe ───────────────────────────────────────────────────────────

def plot_rolling_sharpe(return_series: Dict[str, pd.Series],
                        window: int = 126,
                        rf_annual: float = 0.02,
                        out_dir: str = "output") -> str:
    d = _ensure_dir(out_dir)
    fig, ax = plt.subplots(figsize=FIGSIZE)
    for name, ret in return_series.items():
        rs = rolling_sharpe(ret, window, rf_annual)
        ax.plot(rs.index, rs.values, label=name, linewidth=1.0)
    ax.axhline(0, color="black", linewidth=0.5, linestyle="--")
    ax.set_title(f"Rolling {window}-Day Sharpe Ratio")
    ax.set_ylabel("Sharpe")
    ax.legend(loc="upper left")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    fig.tight_layout()
    fp = str(d / "rolling_sharpe.png")
    fig.savefig(fp, dpi=150)
    plt.close(fig)
    return fp


# ── Parameter stability ─────────────────────────────────────────────────────

def plot_parameter_history(param_history: list,
                           param_key: str,
                           strategy_name: str,
                           out_dir: str = "output") -> str:
    d = _ensure_dir(out_dir)
    df = pd.DataFrame(param_history)
    if param_key not in df.columns:
        return ""
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(df.index, df[param_key], "o-", markersize=4)
    ax.set_title(f"{strategy_name} – Selected '{param_key}' per Window")
    ax.set_xlabel("Walk-Forward Window")
    ax.set_ylabel(param_key)
    fig.tight_layout()
    fp = str(d / f"param_{strategy_name}_{param_key}.png")
    fig.savefig(fp, dpi=150)
    plt.close(fig)
    return fp


# ── Strategy allocation over time ───────────────────────────────────────────

def plot_allocation(allocation: pd.DataFrame,
                    title: str = "Strategy Allocation",
                    out_dir: str = "output") -> str:
    d = _ensure_dir(out_dir)
    fig, ax = plt.subplots(figsize=FIGSIZE)
    allocation.plot.area(ax=ax, alpha=0.7, linewidth=0)
    ax.set_title(title)
    ax.set_ylabel("Weight")
    ax.set_ylim(0, 1.05)
    ax.legend(loc="upper right")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    fig.tight_layout()
    fp = str(d / f"allocation_{title.replace(' ', '_').lower()}.png")
    fig.savefig(fp, dpi=150)
    plt.close(fig)
    return fp


# ── Bootstrap / Monte Carlo distributions ────────────────────────────────────

def _safe_hist(ax, data, **kwargs):
    """Histogram with automatic fallback for degenerate data."""
    for bins in [50, 30, 20, 10, 5, "auto"]:
        try:
            ax.hist(data, bins=bins, **kwargs)
            return
        except (ValueError, TypeError):
            continue
    # ultimate fallback – just 2 bins
    ax.hist(data, bins=2, **kwargs)


def plot_bootstrap_distribution(boot_result: dict,
                                label: str = "Strategy",
                                out_dir: str = "output") -> str:
    d = _ensure_dir(out_dir)
    fig, ax = plt.subplots(figsize=(10, 5))
    dist = boot_result["distribution"]
    _safe_hist(ax, dist, alpha=0.7, edgecolor="white", label="Bootstrap")
    ax.axvline(boot_result["observed"], color="red", linewidth=2,
               label=f"Observed = {boot_result['observed']:.2f}")
    ax.axvline(boot_result["ci_low"], color="orange", linestyle="--",
               label=f"95% CI low = {boot_result['ci_low']:.2f}")
    ax.axvline(boot_result["ci_high"], color="orange", linestyle="--",
               label=f"95% CI high = {boot_result['ci_high']:.2f}")
    ax.set_title(f"Block-Bootstrap Sharpe Distribution – {label}")
    ax.set_xlabel("Sharpe Ratio")
    ax.legend()
    fig.tight_layout()
    fp = str(d / f"bootstrap_{label.replace(' ', '_').lower()}.png")
    fig.savefig(fp, dpi=150)
    plt.close(fig)
    return fp


def plot_monte_carlo_distribution(mc_result: dict,
                                  label: str = "Strategy",
                                  out_dir: str = "output") -> str:
    d = _ensure_dir(out_dir)
    fig, ax = plt.subplots(figsize=(10, 5))
    dist = mc_result["distribution"]
    _safe_hist(ax, dist, alpha=0.7, edgecolor="white", color="steelblue",
               label="Monte Carlo (shuffled)")
    ax.axvline(mc_result["observed"], color="red", linewidth=2,
               label=f"Observed = {mc_result['observed']:.2f}")
    ax.set_title(f"Monte Carlo Sharpe Distribution – {label}")
    ax.set_xlabel("Sharpe Ratio")
    ax.legend()
    fig.tight_layout()
    fp = str(d / f"mc_{label.replace(' ', '_').lower()}.png")
    fig.savefig(fp, dpi=150)
    plt.close(fig)
    return fp


# ── Rolling Outperformance ──────────────────────────────────────────────────

def plot_rolling_outperformance(strat_ret: pd.Series,
                                bench_ret: pd.Series,
                                window: int = 252,
                                label: str = "Strategy",
                                out_dir: str = "output") -> str:
    d = _ensure_dir(out_dir)
    fig, ax = plt.subplots(figsize=FIGSIZE)
    ro = rolling_outperformance(strat_ret, bench_ret, window)
    ax.plot(ro.index, ro.values, linewidth=1.0, color="teal")
    ax.fill_between(ro.index, ro.values, alpha=0.2, color="teal")
    ax.axhline(0, color="black", linewidth=0.5, linestyle="--")
    ax.set_title(f"Rolling {window}-Day Outperformance vs. Buy & Hold – {label}")
    ax.set_ylabel("Cumulative Excess Return")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    fig.tight_layout()
    fp = str(d / f"rolling_outperf_{label.replace(' ', '_').lower()}.png")
    fig.savefig(fp, dpi=150)
    plt.close(fig)
    return fp


# ── OOS Sharpe per window ───────────────────────────────────────────────────

def plot_oos_sharpes(window_results: list,
                     strategy_name: str,
                     out_dir: str = "output") -> str:
    d = _ensure_dir(out_dir)
    dates = [wr.test_start for wr in window_results]
    sharpes = [wr.oos_sharpe for wr in window_results]
    fig, ax = plt.subplots(figsize=(12, 4))
    colors = ["green" if s > 0 else "red" for s in sharpes]
    ax.bar(range(len(sharpes)), sharpes, color=colors, alpha=0.7)
    ax.axhline(0, color="black", linewidth=0.5)
    ax.set_title(f"{strategy_name} – OOS Sharpe per Window")
    ax.set_xlabel("Walk-Forward Window")
    ax.set_ylabel("OOS Sharpe")
    fig.tight_layout()
    fp = str(d / f"oos_sharpes_{strategy_name.lower()}.png")
    fig.savefig(fp, dpi=150)
    plt.close(fig)
    return fp


# ── Print text report ────────────────────────────────────────────────────────

def print_full_report(perf_table: pd.DataFrame,
                      stability_results: dict,
                      switching_results: dict,
                      recommendation: str) -> str:
    """Build a text report string and print it."""
    lines = []
    lines.append("=" * 80)
    lines.append("   WALK-FORWARD BACKTESTING REPORT – MSCI WORLD")
    lines.append("=" * 80)
    lines.append("")
    lines.append("── Performance Comparison ──")
    lines.append(perf_table.to_string())
    lines.append("")

    # Switching results
    lines.append("── Strategy Switching ──")
    for mode in ["hard", "soft"]:
        sr = switching_results[mode]["sharpe"]
        lines.append(f"  {mode.capitalize()} Switch Sharpe: {sr:.2f}")
    lines.append("")

    # Stability for each strategy
    lines.append("── Robustness Analysis ──")
    for strat_name, res in stability_results.items():
        lines.append(f"\n  {strat_name}:")
        lines.append(f"    Observed Sharpe:       {res['observed_sharpe']:.2f}")

        boot = res["bootstrap"]
        lines.append(f"    Bootstrap Mean:        {boot['mean']:.2f}")
        lines.append(f"    Bootstrap 95% CI:      [{boot['ci_low']:.2f}, "
                      f"{boot['ci_high']:.2f}]")
        lines.append(f"    Bootstrap p-value:     {boot['p_value']:.3f}")

        dsr = res["deflated_sharpe"]
        lines.append(f"    Deflated Sharpe (DSR): {dsr['dsr']:.3f} "
                      f"({'significant' if dsr['significant'] else 'NOT significant'})")

        mc = res["monte_carlo"]
        lines.append(f"    Monte Carlo p-value:   {mc['p_value']:.3f} "
                      f"({'significant' if mc['significant_5pct'] else 'NOT significant'})")

        ps = res["parameter_stability"]
        if not ps.empty:
            lines.append(f"    Parameter Stability:")
            for _, row in ps.iterrows():
                lines.append(f"      {row['param']:>12s}:  mean={row['mean']:.1f}  "
                              f"std={row['std']:.1f}  CV={row['cv']:.2f}")

    lines.append("")
    lines.append("── Recommendation ──")
    lines.append(recommendation)
    lines.append("")
    lines.append("=" * 80)

    report = "\n".join(lines)
    print(report)
    return report
