#!/usr/bin/env python3
"""
Stock-Level Analysis: Does the switching methodology work on individual stocks?
================================================================================
Tests 3 universes:
  A) 10 Mega-Cap stocks (AAPL, MSFT, AMZN, GOOGL, JPM, JNJ, V, PG, UNH, NVDA)
  B) 11 Sector ETFs (XLB, XLC, XLE, XLF, XLI, XLK, XLP, XLRE, XLU, XLV, XLY)
  C) MSCI World ETF (SPY proxy) — baseline

For each universe:
  1. Run WF-optimized timing per instrument
  2. Apply switching across instruments  
  3. Compare to equal-weight Buy & Hold of the same universe

Key question: Does more dispersion = more alpha, or more noise = worse signals?
"""

from __future__ import annotations

import sys, os, warnings, time
from typing import Dict, List

warnings.filterwarnings("ignore")

if __name__ == "__main__" and __package__ is None:
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    __package__ = "wf_backtest"

import numpy as np
import pandas as pd
import yfinance as yf

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
sns.set_theme(style="whitegrid", palette="muted", font_scale=1.1)

from .strategies import (
    momentum_signal, ma_signal, rsi_signal,
    vol_target_scaler, apply_costs,
)
from .strategies_ext import (
    macd_signal, dual_momentum_signal, double_ma_signal, donchian_signal,
)
from .metrics import (
    sharpe_ratio, sortino_ratio, max_drawdown, cagr, calmar_ratio,
    drawdown_series, rolling_sharpe, time_under_water,
)
from .walk_forward import generate_windows

FIGSIZE = (16, 7)
RF = 0.02
TX = 0.001
SLIP = 0.0005

# ══════════════════════════════════════════════════════════════════════════════
#  UNIVERSES
# ══════════════════════════════════════════════════════════════════════════════

UNIVERSES = {
    "Mega-Cap Stocks": {
        "tickers": ["AAPL", "MSFT", "AMZN", "GOOGL", "JPM",
                     "JNJ", "V", "PG", "UNH", "NVDA"],
        "description": "10 US mega-cap stocks (survivorship bias!)",
    },
    "Sector ETFs": {
        "tickers": ["XLB", "XLC", "XLE", "XLF", "XLI",
                     "XLK", "XLP", "XLRE", "XLU", "XLV", "XLY"],
        "description": "11 SPDR sector ETFs (no survivorship bias)",
    },
    "MSCI World (SPY)": {
        "tickers": ["SPY"],
        "description": "Single broad-market ETF — baseline",
    },
}

# ══════════════════════════════════════════════════════════════════════════════
#  STRATEGY DEFS (compact for speed)
# ══════════════════════════════════════════════════════════════════════════════

STRATEGY_DEFS = {
    "RSI": {
        "grid": [{"period": per, "threshold": thr}
                 for per in [10, 14, 20, 30]
                 for thr in [35, 45, 50, 55]],
        "gen": lambda p, params: rsi_signal(p, params["period"], params["threshold"]),
        "keys": ["period", "threshold"],
    },
    "Momentum": {
        "grid": [{"lookback": lb} for lb in [40, 90, 160, 252]],
        "gen": lambda p, params: momentum_signal(p, params["lookback"]),
        "keys": ["lookback"],
    },
    "MA": {
        "grid": [{"period": p} for p in [50, 100, 200]],
        "gen": lambda p, params: ma_signal(p, params["period"]),
        "keys": ["period"],
    },
    "Double_MA": {
        "grid": [{"fast": f, "slow": s}
                 for f in [20, 50]
                 for s in [150, 200]
                 if f < s],
        "gen": lambda p, params: double_ma_signal(p, params["fast"], params["slow"]),
        "keys": ["fast", "slow"],
    },
    "Dual_Momentum": {
        "grid": [{"abs_lookback": al, "trend_period": tp}
                 for al in [120, 200]
                 for tp in [150, 252]],
        "gen": lambda p, params: dual_momentum_signal(p, params["abs_lookback"], params["trend_period"]),
        "keys": ["abs_lookback", "trend_period"],
    },
}

WF_CFG = {"train": 756, "test": 21, "step": 21}  # 3Y train, 1M test


# ══════════════════════════════════════════════════════════════════════════════
#  HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def _equity(returns):
    return (1 + returns).cumprod()


def _metrics(eq, ret, rf=RF):
    tuw = time_under_water(eq)
    return {
        "CAGR": cagr(eq),
        "Sharpe": sharpe_ratio(ret, rf),
        "Sortino": sortino_ratio(ret, rf),
        "Calmar": calmar_ratio(eq),
        "MaxDD": max_drawdown(eq),
        "MaxUW_d": tuw["max_days"],
    }


def _select_median(results, keys, top_pct=0.20):
    df = pd.DataFrame(results).sort_values("sharpe", ascending=False)
    n_top = max(1, int(len(df) * top_pct))
    top = df.head(n_top)
    out = {}
    for key in keys:
        med = top[key].median()
        all_vals = sorted(df[key].unique())
        out[key] = min(all_vals, key=lambda v: abs(v - med))
    return out


# ══════════════════════════════════════════════════════════════════════════════
#  DOWNLOAD
# ══════════════════════════════════════════════════════════════════════════════

def download_universe(tickers, start="2010-01-01", end="2026-02-28"):
    """Download all tickers and return dict of DataFrames."""
    print(f"  Downloading {len(tickers)} tickers …")
    raw = yf.download(tickers, start=start, end=end,
                      auto_adjust=True, progress=False)

    result = {}

    if isinstance(raw.columns, pd.MultiIndex):
        # yfinance returns (Price, Ticker) MultiIndex
        for t in tickers:
            try:
                close = raw[("Close", t)].dropna()
                ret = close.pct_change().dropna()
                if len(ret) > 500:
                    result[t] = pd.DataFrame({"Close": close.loc[ret.index],
                                              "Returns": ret})
                    print(f"    {t}: {len(ret)} days")
                else:
                    print(f"    {t}: too few days ({len(ret)}), skipped")
            except Exception as e:
                print(f"    {t}: FAILED ({e})")
    else:
        # Non-MultiIndex (shouldn't happen with recent yfinance, but handle)
        if "Close" in raw.columns:
            t = tickers[0]
            close = raw["Close"].dropna()
            ret = close.pct_change().dropna()
            result[t] = pd.DataFrame({"Close": close.loc[ret.index],
                                      "Returns": ret})
            print(f"    {t}: {len(ret)} days")

    return result


# ══════════════════════════════════════════════════════════════════════════════
#  PER-INSTRUMENT WF TIMING
# ══════════════════════════════════════════════════════════════════════════════

def wf_timing_single_instrument(prices, returns, strat_def):
    """Run WF optimization for one strategy on one instrument."""
    windows = generate_windows(prices.index, WF_CFG["train"],
                               WF_CFG["test"], WF_CFG["step"])
    if len(windows) < 4:
        return None

    oos_parts = []
    for ts, te, os_s, os_e in windows:
        train_p = prices.iloc[ts:te]
        train_r = returns.iloc[ts:te]

        results = []
        for params in strat_def["grid"]:
            try:
                sig = strat_def["gen"](train_p, params)
                if sig is None or sig.isna().all():
                    continue
                sig = sig.fillna(0)
                net = apply_costs(sig, train_r, TX, SLIP)
                net = net.dropna()
                if len(net) < 20:
                    continue
                sr = sharpe_ratio(net, RF)
                results.append({**params, "sharpe": sr})
            except Exception:
                continue

        if len(results) < 2:
            continue

        best_params = _select_median(results, strat_def["keys"])

        full_p = prices.iloc[ts:os_e]
        full_r = returns.iloc[ts:os_e]
        try:
            full_signal = strat_def["gen"](full_p, best_params)
        except Exception:
            continue
        if full_signal is None:
            continue

        test_r = returns.iloc[os_s:os_e]
        test_signal = full_signal.loc[test_r.index].fillna(0)
        test_net = apply_costs(test_signal, test_r, TX, SLIP)
        oos_parts.append(test_net)

    if len(oos_parts) < 4:
        return None

    oos_ret = pd.concat(oos_parts).sort_index()
    oos_ret = oos_ret[~oos_ret.index.duplicated(keep="first")]
    return oos_ret if len(oos_ret) >= 100 else None


def best_timing_per_instrument(data: Dict[str, pd.DataFrame]):
    """
    For EACH instrument, run all strategies and keep the best OOS returns.
    Returns dict: ticker → OOS returns series.
    """
    best = {}
    for ticker, df in data.items():
        p = df["Close"]
        r = df["Returns"]
        best_sr = -999
        best_ret = None
        for sname, sdef in STRATEGY_DEFS.items():
            oos = wf_timing_single_instrument(p, r, sdef)
            if oos is not None:
                sr = sharpe_ratio(oos, RF)
                if sr > best_sr:
                    best_sr = sr
                    best_ret = oos
        if best_ret is not None:
            best[ticker] = best_ret
            print(f"    {ticker}: best Sharpe = {best_sr:.2f}")
        else:
            print(f"    {ticker}: no valid timing signal")
    return best


# ══════════════════════════════════════════════════════════════════════════════
#  SWITCHING ACROSS INSTRUMENTS
# ══════════════════════════════════════════════════════════════════════════════

def instrument_switching(timed_returns: Dict[str, pd.Series],
                         bh_returns: Dict[str, pd.Series],
                         switch_window: int = 63):
    """
    Switching across multiple instruments using rolling Sharpe.
    Also builds equal-weight timed and equal-weight B&H portfolios.
    """
    # Align all timed returns
    timed_names = list(timed_returns.keys())
    if len(timed_names) < 2:
        return {}

    all_starts = [s.first_valid_index() for s in timed_returns.values()]
    all_ends = [s.last_valid_index() for s in timed_returns.values()]
    start = max(all_starts)
    end = min(all_ends)

    timed_df = pd.DataFrame({n: r.loc[start:end] for n, r in timed_returns.items()}).fillna(0)
    bh_df = pd.DataFrame({n: r.loc[start:end] for n, r in bh_returns.items()
                          if n in timed_names}).fillna(0)

    n_inst = len(timed_names)
    result = {}

    # 1. Equal-weight Buy & Hold
    ew_bh = bh_df.mean(axis=1)
    eq = _equity(ew_bh.dropna())
    result["EW_BuyHold"] = {
        "returns": ew_bh,
        **_metrics(eq, ew_bh.dropna()),
    }

    # 2. Equal-weight timed (each instrument individually timed, equal weight)
    ew_timed = timed_df.mean(axis=1)
    eq = _equity(ew_timed.dropna())
    result["EW_Timed"] = {
        "returns": ew_timed,
        **_metrics(eq, ew_timed.dropna()),
    }

    # 3. Hard switch (pick best-performing timed instrument by rolling Sharpe)
    roll = pd.DataFrame({
        n: rolling_sharpe(timed_df[n], switch_window, RF)
        for n in timed_names
    })

    hard_ret = pd.Series(0.0, index=timed_df.index)
    for idx in timed_df.index:
        if idx not in roll.index:
            continue
        row = roll.loc[idx]
        eligible = row[row > 0]
        if eligible.empty:
            # Fallback to equal-weight B&H
            hard_ret.loc[idx] = bh_df.loc[idx].mean() if idx in bh_df.index else 0
            continue
        best = eligible.idxmax()
        hard_ret.loc[idx] = timed_df.loc[idx, best]

    eq = _equity(hard_ret.dropna())
    result["Hard_Switch"] = {
        "returns": hard_ret,
        **_metrics(eq, hard_ret.dropna()),
    }

    # 4. Soft switch (weighted by rolling Sharpe)
    soft_ret = pd.Series(0.0, index=timed_df.index)
    for idx in timed_df.index:
        if idx not in roll.index:
            continue
        row = roll.loc[idx]
        eligible = row[row > 0]
        if eligible.empty:
            soft_ret.loc[idx] = bh_df.loc[idx].mean() if idx in bh_df.index else 0
            continue
        weights = eligible / eligible.sum()
        for name in weights.index:
            soft_ret.loc[idx] += weights[name] * timed_df.loc[idx, name]

    eq = _equity(soft_ret.dropna())
    result["Soft_Switch"] = {
        "returns": soft_ret,
        **_metrics(eq, soft_ret.dropna()),
    }

    # 5. Top-3 rotation (hold top 3 by rolling Sharpe, equal-weight)
    if n_inst >= 4:
        top3_ret = pd.Series(0.0, index=timed_df.index)
        for idx in timed_df.index:
            if idx not in roll.index:
                continue
            row = roll.loc[idx]
            eligible = row[row > 0].sort_values(ascending=False)
            if len(eligible) == 0:
                top3_ret.loc[idx] = bh_df.loc[idx].mean() if idx in bh_df.index else 0
                continue
            top = eligible.head(3)
            for name in top.index:
                top3_ret.loc[idx] += timed_df.loc[idx, name] / len(top)

        eq = _equity(top3_ret.dropna())
        result["Top3_Rotation"] = {
            "returns": top3_ret,
            **_metrics(eq, top3_ret.dropna()),
        }

    return result


# ══════════════════════════════════════════════════════════════════════════════
#  SPY-LEVEL SWITCHING (baseline: same methodology on single ETF)
# ══════════════════════════════════════════════════════════════════════════════

def spy_strategy_switching(data: Dict[str, pd.DataFrame],
                           switch_window: int = 63):
    """
    Run the ORIGINAL methodology: multiple strategies on SPY,
    then switch between them.
    """
    spy_df = list(data.values())[0]
    prices = spy_df["Close"]
    returns = spy_df["Returns"]

    strat_returns = {}
    for sname, sdef in STRATEGY_DEFS.items():
        oos = wf_timing_single_instrument(prices, returns, sdef)
        if oos is not None:
            strat_returns[sname] = oos

    if len(strat_returns) < 2:
        return {}

    # Align
    start = max(s.first_valid_index() for s in strat_returns.values())
    end = min(s.last_valid_index() for s in strat_returns.values())
    aligned = {n: r.loc[start:end] for n, r in strat_returns.items()}
    bench_ret = returns.loc[start:end]

    df = pd.DataFrame(aligned)
    names = df.columns.tolist()

    roll = pd.DataFrame({
        n: rolling_sharpe(df[n], switch_window, RF)
        for n in names
    })

    hard_ret = pd.Series(0.0, index=df.index)
    for idx in df.index:
        if idx not in roll.index:
            continue
        row = roll.loc[idx]
        eligible = row[row > 0]
        if eligible.empty:
            continue
        best = eligible.idxmax()
        hard_ret.loc[idx] = df.loc[idx, best]

    result = {}

    # B&H
    eq = _equity(bench_ret.dropna())
    result["SPY_BuyHold"] = {
        "returns": bench_ret,
        **_metrics(eq, bench_ret.dropna()),
    }

    # Hard switch
    eq = _equity(hard_ret.dropna())
    result["SPY_Strategy_Switch"] = {
        "returns": hard_ret,
        **_metrics(eq, hard_ret.dropna()),
    }

    return result


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")
    os.makedirs(out_dir, exist_ok=True)

    print("=" * 80)
    print("  STOCK-LEVEL ANALYSIS: Does Switching Work on Individual Instruments?")
    print("=" * 80)

    all_universe_results = {}

    for uni_name, uni_cfg in UNIVERSES.items():
        print(f"\n{'═' * 80}")
        print(f"  UNIVERSE: {uni_name}")
        print(f"  {uni_cfg['description']}")
        print(f"  Tickers: {', '.join(uni_cfg['tickers'])}")
        print(f"{'═' * 80}")

        # Download
        data = download_universe(uni_cfg["tickers"])
        if len(data) < 1:
            print(f"  Skipping {uni_name}: no data")
            continue

        if len(data) == 1 and uni_name == "MSCI World (SPY)":
            # Special case: SPY — use strategy-switching (original method)
            print(f"\n  Running strategy-level switching on SPY …")
            results = spy_strategy_switching(data)
        else:
            # Multi-instrument: per-instrument timing + instrument switching
            print(f"\n  Running per-instrument WF timing …")
            t0 = time.time()
            timed = best_timing_per_instrument(data)
            print(f"  Timing done in {time.time()-t0:.0f}s "
                  f"({len(timed)}/{len(data)} instruments timed)")

            if len(timed) < 1:
                print(f"  Skipping {uni_name}: no timed instruments")
                continue

            # B&H returns per instrument
            bh_returns = {t: df["Returns"] for t, df in data.items()}

            print(f"\n  Running switching across {len(timed)} instruments …")
            results = instrument_switching(timed, bh_returns, switch_window=63)

        all_universe_results[uni_name] = results

        # Print results
        print(f"\n  RESULTS for {uni_name}:")
        print(f"  {'Approach':>25s} {'Sharpe':>7s} {'CAGR':>8s} "
              f"{'MaxDD':>8s} {'Sortino':>8s} {'Calmar':>8s}")
        print(f"  {'-'*70}")
        for name, r in sorted(results.items(), key=lambda x: -x[1]["Sharpe"]):
            print(f"  {name:>25s} {r['Sharpe']:>7.2f} {r['CAGR']:>7.2%} "
                  f"{r['MaxDD']:>7.2%} {r['Sortino']:>7.2f} "
                  f"{r['Calmar']:>7.2f}")

    # ══════════════════════════════════════════════════════════════════════════
    #  CROSS-UNIVERSE COMPARISON
    # ══════════════════════════════════════════════════════════════════════════
    print(f"\n\n{'=' * 80}")
    print(f"  CROSS-UNIVERSE COMPARISON")
    print(f"{'=' * 80}")

    comparison = []

    for uni_name, results in all_universe_results.items():
        # Find B&H and best tactical approach
        bh_key = [k for k in results if "BuyHold" in k or "Buy" in k]
        tac_keys = [k for k in results if "Switch" in k or "Rotation" in k]

        if bh_key:
            bh = results[bh_key[0]]
            comparison.append({
                "universe": uni_name,
                "approach": "Buy & Hold",
                **{k: v for k, v in bh.items() if k != "returns"},
            })

        for tk in tac_keys:
            tac = results[tk]
            comparison.append({
                "universe": uni_name,
                "approach": tk,
                **{k: v for k, v in tac.items() if k != "returns"},
            })

        if "EW_Timed" in results:
            comparison.append({
                "universe": uni_name,
                "approach": "EW_Timed",
                **{k: v for k, v in results["EW_Timed"].items() if k != "returns"},
            })

    cdf = pd.DataFrame(comparison)
    print(f"\n  {'Universe':>25s} {'Approach':>25s} {'Sharpe':>7s} "
          f"{'CAGR':>8s} {'MaxDD':>8s} {'Sortino':>8s}")
    print(f"  {'─'*100}")
    for _, row in cdf.sort_values(["universe", "Sharpe"], ascending=[True, False]).iterrows():
        print(f"  {row['universe']:>25s} {row['approach']:>25s} "
              f"{row['Sharpe']:>7.2f} {row['CAGR']:>7.2%} "
              f"{row['MaxDD']:>7.2%} {row['Sortino']:>7.2f}")

    # ══════════════════════════════════════════════════════════════════════════
    #  ANALYSIS: KEY STATISTICS
    # ══════════════════════════════════════════════════════════════════════════
    print(f"\n\n{'=' * 80}")
    print(f"  KEY FINDINGS")
    print(f"{'=' * 80}")

    # Per universe: does switching beat B&H?
    for uni_name, results in all_universe_results.items():
        bh_key = [k for k in results if "BuyHold" in k or "Buy" in k]
        if not bh_key:
            continue
        bh_sharpe = results[bh_key[0]]["Sharpe"]
        bh_cagr = results[bh_key[0]]["CAGR"]
        bh_mdd = results[bh_key[0]]["MaxDD"]

        print(f"\n  {uni_name}:")
        print(f"    B&H: Sharpe={bh_sharpe:.2f}, CAGR={bh_cagr:.2%}, MaxDD={bh_mdd:.2%}")

        for name, r in results.items():
            if name == bh_key[0]:
                continue
            delta_s = r["Sharpe"] - bh_sharpe
            delta_c = r["CAGR"] - bh_cagr
            mdd_imp = (1 - abs(r["MaxDD"]) / abs(bh_mdd)) * 100 if bh_mdd != 0 else 0
            marker = "★" if r["Sharpe"] > bh_sharpe else "✗"
            print(f"    {marker} {name}: Sharpe={r['Sharpe']:.2f} (Δ{delta_s:+.2f}), "
                  f"CAGR={r['CAGR']:.2%} (Δ{delta_c:+.2%}), "
                  f"MaxDD={r['MaxDD']:.2%} ({mdd_imp:+.0f}% besser)")

    # ══════════════════════════════════════════════════════════════════════════
    #  SURVIVORSHIP BIAS WARNING
    # ══════════════════════════════════════════════════════════════════════════
    print(f"\n\n{'=' * 80}")
    print(f"  ⚠ SURVIVORSHIP BIAS ANALYSE")
    print(f"{'=' * 80}")
    print("""
  Die Mega-Cap-Ergebnisse sind STARK durch Survivorship Bias verzerrt:
    - Wir haben die 10 größten Aktien VON HEUTE ausgewählt
    - Diese sind DEFINITIONSGEMÄSS die Gewinner der letzten 15 Jahre
    - 2010 waren NVDA, AMZN, GOOGL noch keine Mega-Caps
    - Fallen gelassene Aktien (GE, IBM, XOM#2016) fehlen komplett
  
  Die Sektor-ETF-Ergebnisse sind DEUTLICH weniger verzerrt:
    - Sektoren existieren dauerhaft (kein Einzeltitel-Risiko)
    - Automatische Rebalancierung innerhalb jedes Sektors
    - Vergleichbare Methodologie, aber fairer Test
  
  EMPFEHLUNG: Sektor-ETFs als Instrument-Universum verwenden,
  NICHT Einzelaktien, wenn man die Switching-Methodik skalieren will.
""")

    # ══════════════════════════════════════════════════════════════════════════
    #  CHARTS
    # ══════════════════════════════════════════════════════════════════════════
    print("  Generating charts …")

    # Chart 1: Cross-universe comparison bar chart
    fig, axes = plt.subplots(1, 3, figsize=(18, 7))

    for ax_idx, (metric, label) in enumerate([
        ("Sharpe", "Sharpe Ratio"),
        ("CAGR", "CAGR"),
        ("MaxDD", "Max Drawdown"),
    ]):
        uni_names = cdf["universe"].unique()
        x = np.arange(len(uni_names))
        approaches = cdf["approach"].unique()
        n_approaches = len(approaches)
        width = 0.8 / n_approaches
        colors = plt.cm.Set2(np.linspace(0, 1, n_approaches))

        for i, approach in enumerate(approaches):
            vals = []
            for uni in uni_names:
                subset = cdf[(cdf["universe"] == uni) & (cdf["approach"] == approach)]
                if not subset.empty:
                    v = subset[metric].iloc[0]
                    vals.append(v * 100 if metric == "CAGR" else v * 100 if metric == "MaxDD" else v)
                else:
                    vals.append(0)
            offset = (i - n_approaches / 2 + 0.5) * width
            axes[ax_idx].bar(x + offset, vals, width, label=approach[:20],
                             color=colors[i])

        axes[ax_idx].set_xticks(x)
        axes[ax_idx].set_xticklabels([u[:15] for u in uni_names],
                                      rotation=20, fontsize=8)
        axes[ax_idx].set_ylabel(label)
        axes[ax_idx].set_title(label)
        if ax_idx == 0:
            axes[ax_idx].legend(fontsize=7, loc="upper left")

    fig.suptitle("Cross-Universe Comparison: Stock-Level vs. ETF Switching",
                 fontsize=14, fontweight="bold")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "stocks_cross_universe.png"), dpi=150)
    plt.close(fig)

    # Chart 2: Equity curves per universe
    n_uni = len(all_universe_results)
    fig, axes = plt.subplots(1, n_uni, figsize=(7 * n_uni, 6))
    if n_uni == 1:
        axes = [axes]

    for ax, (uni_name, results) in zip(axes, all_universe_results.items()):
        for name, r in sorted(results.items(), key=lambda x: -x[1]["Sharpe"]):
            ret = r["returns"]
            eq = _equity(ret.dropna())
            lw = 2.5 if "BuyHold" in name or "Buy" in name else 1.2
            ls = "-" if "BuyHold" in name or "Buy" in name else "--"
            ax.plot(eq.index, eq.values, label=f"{name} (S={r['Sharpe']:.2f})",
                    linewidth=lw, linestyle=ls)
        ax.set_title(uni_name, fontsize=11)
        ax.set_ylabel("Growth of $1")
        ax.legend(fontsize=7, loc="upper left")
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

    fig.suptitle("Equity Curves by Universe", fontsize=14, fontweight="bold")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "stocks_equity_curves.png"), dpi=150)
    plt.close(fig)

    # ══════════════════════════════════════════════════════════════════════════
    #  SAVE REPORT
    # ══════════════════════════════════════════════════════════════════════════
    lines = []
    lines.append("=" * 80)
    lines.append("  STOCK-LEVEL ANALYSIS REPORT")
    lines.append("=" * 80)

    for uni_name, results in all_universe_results.items():
        lines.append(f"\n\n{'─' * 60}")
        lines.append(f"  {uni_name}")
        lines.append(f"{'─' * 60}")
        for name, r in sorted(results.items(), key=lambda x: -x[1]["Sharpe"]):
            lines.append(f"  {name:>25s}  Sharpe={r['Sharpe']:.2f}  "
                         f"CAGR={r['CAGR']:.2%}  MaxDD={r['MaxDD']:.2%}  "
                         f"Sortino={r['Sortino']:.2f}  Calmar={r['Calmar']:.2f}")

    lines.append(f"\n\n{'=' * 80}")
    lines.append("  FAZIT")
    lines.append("=" * 80)

    report_path = os.path.join(out_dir, "stocks_analysis_report.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print(f"\n  Report: {report_path}")
    print(f"  Charts: {out_dir}/stocks_*.png")
    print("  DONE.")


if __name__ == "__main__":
    main()
