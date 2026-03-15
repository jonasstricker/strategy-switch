#!/usr/bin/env python3
"""
Multi-ETF Evaluation
=====================
Evaluiert die Walk-Forward Strategy-Switch Methodik auf:
  - SPY  (S&P 500)
  - URTH (MSCI World)
  - EEM  (Emerging Markets)
  - VGK  (FTSE Europe)

Prüft ob der Sharpe-Vorteil des Switching auch außerhalb von SPY besteht.
"""

from __future__ import annotations
import sys, os, warnings
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import yfinance as yf

from wf_backtest.strategies import momentum_signal, ma_signal, rsi_signal, apply_costs
from wf_backtest.strategies_ext import dual_momentum_signal, double_ma_signal
from wf_backtest.metrics import (
    sharpe_ratio, sortino_ratio, max_drawdown, cagr, calmar_ratio,
    rolling_sharpe, time_under_water,
)
from wf_backtest.walk_forward import generate_windows
from wf_backtest.switching import apply_switching

# ── Constants ────────────────────────────────────────────────────────────────
RF = 0.02
TX = 0.001
SLIP = 0.0005

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
                 for f in [20, 50] for s in [150, 200] if f < s],
        "gen": lambda p, params: double_ma_signal(p, params["fast"], params["slow"]),
        "keys": ["fast", "slow"],
    },
    "Dual_Momentum": {
        "grid": [{"abs_lookback": al, "trend_period": tp}
                 for al in [120, 200] for tp in [150, 252]],
        "gen": lambda p, params: dual_momentum_signal(p, params["abs_lookback"], params["trend_period"]),
        "keys": ["abs_lookback", "trend_period"],
    },
}

WF_CFG = {"train": 756, "test": 21, "step": 21}
SW = 63  # rolling-sharpe switching window

ETFS = {
    "SPY":  {"name": "S&P 500",          "ticker": "SPY",  "start": "2000-01-01"},
    "URTH": {"name": "MSCI World",        "ticker": "URTH", "start": "2012-01-01"},
    "EEM":  {"name": "Emerging Markets",  "ticker": "EEM",  "start": "2004-01-01"},
    "VGK":  {"name": "FTSE Europe",       "ticker": "VGK",  "start": "2005-01-01"},
}


# ── Helpers ──────────────────────────────────────────────────────────────────

def _equity(ret):
    return (1 + ret).cumprod()


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


def wf_single(prices, returns, sdef):
    windows = generate_windows(prices.index, WF_CFG["train"],
                               WF_CFG["test"], WF_CFG["step"])
    if len(windows) < 4:
        return None
    oos = []
    for ts, te, os_s, os_e in windows:
        train_p = prices.iloc[ts:te]
        train_r = returns.iloc[ts:te]
        results = []
        for params in sdef["grid"]:
            try:
                sig = sdef["gen"](train_p, params)
                if sig is None or sig.isna().all():
                    continue
                net = apply_costs(sig.fillna(0), train_r, TX, SLIP).dropna()
                if len(net) < 20:
                    continue
                results.append({**params, "sharpe": sharpe_ratio(net, RF)})
            except Exception:
                continue
        if len(results) < 2:
            continue
        bp = _select_median(results, sdef["keys"])
        try:
            full_sig = sdef["gen"](prices.iloc[ts:os_e], bp)
        except Exception:
            continue
        if full_sig is None:
            continue
        test_r = returns.iloc[os_s:os_e]
        test_sig = full_sig.loc[test_r.index].fillna(0)
        oos.append(apply_costs(test_sig, test_r, TX, SLIP))
    if len(oos) < 4:
        return None
    r = pd.concat(oos).sort_index()
    return r[~r.index.duplicated(keep="first")]


def evaluate_etf(ticker: str, start: str):
    """Run full WF evaluation for a single ETF. Returns metrics dict."""
    print(f"\n{'='*60}")
    print(f"  Evaluiere: {ticker} (ab {start})")
    print(f"{'='*60}")

    raw = yf.download(ticker, start=start, auto_adjust=True, progress=False)
    if isinstance(raw.columns, pd.MultiIndex):
        close_col = [c for c in raw.columns if c[0] == "Close"]
        close = raw[close_col[0]].dropna()
    else:
        close = raw["Close"].dropna()

    returns = close.pct_change().dropna()
    prices = close.loc[returns.index]

    print(f"  Daten: {len(prices)} Tage ({prices.index[0].date()} bis {prices.index[-1].date()})")

    # Per-strategy WF
    strat_oos = {}
    for sname, sdef in STRATEGY_DEFS.items():
        print(f"  WF: {sname} ...", end=" ", flush=True)
        r = wf_single(prices, returns, sdef)
        if r is not None:
            strat_oos[sname] = r
            sh = sharpe_ratio(r, RF)
            print(f"OK (OOS Sharpe: {sh:.2f})")
        else:
            print("SKIP (zu wenige Daten)")

    if not strat_oos:
        print("  ⚠ Keine Strategie hat genug OOS-Daten!")
        return None

    # Align
    start_dt = max(s.first_valid_index() for s in strat_oos.values())
    end_dt = min(s.last_valid_index() for s in strat_oos.values())
    aligned = {n: r.loc[start_dt:end_dt] for n, r in strat_oos.items()}
    bench_ret = returns.loc[start_dt:end_dt]
    df_strats = pd.DataFrame(aligned)
    names = df_strats.columns.tolist()

    # Rolling Sharpe switching (mit Meta-Kosten + Hysterese)
    roll = pd.DataFrame({n: rolling_sharpe(df_strats[n], SW, RF) for n in names})

    hard_ret, active_strat = apply_switching(
        df_strats, roll, tx=TX, slip=SLIP, min_hold=5,
    )

    # Metrics
    bh_eq = _equity(bench_ret)
    sw_eq = _equity(hard_ret)
    tuw_sw = time_under_water(sw_eq)
    tuw_bh = time_under_water(bh_eq)

    years = (bench_ret.index[-1] - bench_ret.index[0]).days / 365.25

    # Per-strategy metrics
    strat_metrics = {}
    for n in names:
        eq = _equity(aligned[n])
        strat_metrics[n] = {
            "Sharpe": sharpe_ratio(aligned[n], RF),
            "CAGR": cagr(eq),
            "MaxDD": max_drawdown(eq),
        }

    # Time in market
    pct_invested = float((active_strat != "Cash").mean())

    result = {
        "ticker": ticker,
        "period": f"{prices.index[0].date()} – {prices.index[-1].date()}",
        "years": years,
        "n_strategies": len(names),
        "strategies": names,
        "switch": {
            "CAGR": cagr(sw_eq),
            "Sharpe": sharpe_ratio(hard_ret, RF),
            "Sortino": sortino_ratio(hard_ret, RF),
            "Calmar": calmar_ratio(sw_eq),
            "MaxDD": max_drawdown(sw_eq),
            "Vola": float(hard_ret.std() * np.sqrt(252)),
            "MaxUW": tuw_sw["max_days"],
            "PctInvested": pct_invested,
        },
        "bh": {
            "CAGR": cagr(bh_eq),
            "Sharpe": sharpe_ratio(bench_ret, RF),
            "Sortino": sortino_ratio(bench_ret, RF),
            "Calmar": calmar_ratio(bh_eq),
            "MaxDD": max_drawdown(bh_eq),
            "Vola": float(bench_ret.std() * np.sqrt(252)),
            "MaxUW": tuw_bh["max_days"],
        },
        "strat_metrics": strat_metrics,
    }
    return result


def print_results(results: dict):
    """Print a nice comparison table."""
    print("\n")
    print("=" * 80)
    print("  ERGEBNIS: Walk-Forward Strategy Switch — Multi-ETF Evaluation")
    print("=" * 80)

    # Summary table
    header = f"{'':>20} | {'SPY':>12} | {'URTH':>12} | {'EEM':>12} | {'VGK':>12}"
    sep = "-" * len(header)

    print(f"\n{header}")
    print(sep)

    metrics = [
        ("Zeitraum (Jahre)",  lambda r: f"{r['years']:.1f}", None),
        ("Switch CAGR",       lambda r: f"{r['switch']['CAGR']:.1%}", None),
        ("Switch Sharpe",     lambda r: f"{r['switch']['Sharpe']:.2f}", None),
        ("Switch Sortino",    lambda r: f"{r['switch']['Sortino']:.2f}", None),
        ("Switch Calmar",     lambda r: f"{r['switch']['Calmar']:.2f}", None),
        ("Switch MaxDD",      lambda r: f"{r['switch']['MaxDD']:.1%}", None),
        ("Switch Vola",       lambda r: f"{r['switch']['Vola']:.1%}", None),
        ("Switch MaxUW (T)",  lambda r: f"{r['switch']['MaxUW']}", None),
        ("% Investiert",      lambda r: f"{r['switch']['PctInvested']:.0%}", None),
        ("",                  None, None),
        ("B&H CAGR",          lambda r: f"{r['bh']['CAGR']:.1%}", None),
        ("B&H Sharpe",        lambda r: f"{r['bh']['Sharpe']:.2f}", None),
        ("B&H Sortino",       lambda r: f"{r['bh']['Sortino']:.2f}", None),
        ("B&H Calmar",        lambda r: f"{r['bh']['Calmar']:.2f}", None),
        ("B&H MaxDD",         lambda r: f"{r['bh']['MaxDD']:.1%}", None),
        ("B&H Vola",          lambda r: f"{r['bh']['Vola']:.1%}", None),
        ("B&H MaxUW (T)",     lambda r: f"{r['bh']['MaxUW']}", None),
        ("",                  None, None),
        ("Δ Sharpe (Sw-BH)",  lambda r: f"{r['switch']['Sharpe'] - r['bh']['Sharpe']:+.2f}", None),
        ("Δ MaxDD (Sw-BH)",   lambda r: f"{r['switch']['MaxDD'] - r['bh']['MaxDD']:+.1%}", None),
    ]

    tickers = ["SPY", "URTH", "EEM", "VGK"]
    for label, fmt, _ in metrics:
        if fmt is None:
            print(sep)
            continue
        vals = []
        for t in tickers:
            if t in results and results[t] is not None:
                try:
                    vals.append(fmt(results[t]))
                except Exception:
                    vals.append("n/a")
            else:
                vals.append("n/a")
        line = f"{label:>20} | {vals[0]:>12} | {vals[1]:>12} | {vals[2]:>12} | {vals[3]:>12}"
        print(line)

    print(sep)

    # Per-strategy breakdown
    for t in tickers:
        if t not in results or results[t] is None:
            continue
        r = results[t]
        print(f"\n  {t} — Einzelstrategie-Sharpes:")
        for sname, sm in r["strat_metrics"].items():
            print(f"    {sname:>15}: Sharpe {sm['Sharpe']:.2f} | CAGR {sm['CAGR']:.1%} | MaxDD {sm['MaxDD']:.1%}")

    # Conclusion
    print("\n" + "=" * 80)
    print("  FAZIT")
    print("=" * 80)
    for t in tickers:
        if t not in results or results[t] is None:
            print(f"  {t}: ⚠ Nicht genug Daten für Evaluation.")
            continue
        r = results[t]
        sw_sh = r["switch"]["Sharpe"]
        bh_sh = r["bh"]["Sharpe"]
        delta = sw_sh - bh_sh
        sw_dd = r["switch"]["MaxDD"]
        bh_dd = r["bh"]["MaxDD"]
        name = ETFS[t]["name"]

        if delta > 0.1:
            verdict = f"✅ {name}: Switch klar besser (Sharpe +{delta:.2f})"
        elif delta > -0.1:
            verdict = f"🟡 {name}: Switch ≈ B&H (Sharpe {delta:+.2f})"
        else:
            verdict = f"❌ {name}: B&H war besser (Sharpe {delta:+.2f})"

        dd_better = sw_dd > bh_dd  # dd negative, less negative = better
        dd_note = f"MaxDD um {abs(sw_dd - bh_dd):.1%} {'besser' if dd_better else 'schlechter'}"
        print(f"  {verdict} — {dd_note}")


def main():
    results = {}
    for key, info in ETFS.items():
        try:
            r = evaluate_etf(info["ticker"], info["start"])
            results[key] = r
        except Exception as e:
            print(f"  ⚠ FEHLER bei {key}: {e}")
            results[key] = None

    print_results(results)


if __name__ == "__main__":
    main()
