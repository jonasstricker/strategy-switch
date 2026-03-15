#!/usr/bin/env python3
"""
Diagnose-Skript: Vergleich ALT (ohne Kosten/Hysterese) vs NEU (mit)
=====================================================================
"""

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
from wf_backtest.metrics import sharpe_ratio, rolling_sharpe
from wf_backtest.walk_forward import generate_windows
from wf_backtest.switching import apply_switching

RF = 0.02
TX = 0.001
SLIP = 0.0005
WF_CFG = {"train": 756, "test": 21, "step": 21}

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

ETF_DEFS = {
    "SPY":  {"start": "2000-01-01"},
    "URTH": {"start": "2012-01-01"},
    "EEM":  {"start": "2004-01-01"},
    "VGK":  {"start": "2005-01-01"},
}


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


def test_etf(ticker, start_date):
    print(f"\n{'='*70}")
    print(f"  {ticker}")
    print(f"{'='*70}")

    raw = yf.download(ticker, start=start_date, auto_adjust=True, progress=False)
    if isinstance(raw.columns, pd.MultiIndex):
        close_col = [c for c in raw.columns if c[0] == "Close"]
        close = raw[close_col[0]].dropna()
    else:
        close = raw["Close"].dropna()

    returns = close.pct_change().dropna()
    prices = close.loc[returns.index]

    strat_oos = {}
    for sname, sdef in STRATEGY_DEFS.items():
        print(f"  WF: {sname} ...", end=" ", flush=True)
        r = wf_single(prices, returns, sdef)
        if r is not None:
            strat_oos[sname] = r
            print(f"OK")
        else:
            print("SKIP")

    start = max(s.first_valid_index() for s in strat_oos.values())
    end = min(s.last_valid_index() for s in strat_oos.values())
    aligned = {n: r.loc[start:end] for n, r in strat_oos.items()}
    bench_ret = returns.loc[start:end]
    df_strats = pd.DataFrame(aligned)
    names = df_strats.columns.tolist()

    sw = 63
    roll = pd.DataFrame({n: rolling_sharpe(df_strats[n], sw, RF) for n in names})

    # ── ALT: Ohne Meta-Kosten, ohne Hysterese ──
    old_ret = pd.Series(0.0, index=df_strats.index)
    old_strat = pd.Series("Cash", index=df_strats.index)
    for idx in df_strats.index:
        if idx not in roll.index:
            continue
        row = roll.loc[idx]
        eligible = row[row > 0]
        if eligible.empty:
            continue
        best = eligible.idxmax()
        old_ret.loc[idx] = df_strats.loc[idx, best]
        old_strat.loc[idx] = best

    # ── NEU: Mit Meta-Kosten + Hysterese ──
    new_ret, new_strat = apply_switching(df_strats, roll, tx=TX, slip=SLIP, min_hold=5)

    # ── Vergleich ──
    old_sharpe = sharpe_ratio(old_ret, RF)
    new_sharpe = sharpe_ratio(new_ret, RF)
    bh_sharpe  = sharpe_ratio(bench_ret, RF)

    old_trades_meta = (old_strat != "Cash").astype(float).diff().abs().fillna(0).sum()
    new_trades_meta = (new_strat != "Cash").astype(float).diff().abs().fillna(0).sum()

    old_cagr = (1 + old_ret).cumprod().iloc[-1] ** (252 / len(old_ret)) - 1
    new_cagr = (1 + new_ret).cumprod().iloc[-1] ** (252 / len(new_ret)) - 1
    bh_cagr  = (1 + bench_ret).cumprod().iloc[-1] ** (252 / len(bench_ret)) - 1

    # Kurze Cash-Phasen zählen
    def count_short_cash(strat_series, max_days=3):
        states = strat_series.values
        cash_start = None
        count = 0
        for i in range(len(states)):
            if states[i] == "Cash" and cash_start is None:
                cash_start = i
            elif states[i] != "Cash" and cash_start is not None:
                if (i - cash_start) <= max_days:
                    count += 1
                cash_start = None
        return count

    old_short = count_short_cash(old_strat)
    new_short = count_short_cash(new_strat)

    print(f"\n  {'Metrik':<30} | {'Buy&Hold':>10} | {'ALT':>10} | {'NEU':>10}")
    print(f"  {'-'*30}-+-{'-'*10}-+-{'-'*10}-+-{'-'*10}")
    print(f"  {'Sharpe Ratio':<30} | {bh_sharpe:>10.3f} | {old_sharpe:>10.3f} | {new_sharpe:>10.3f}")
    print(f"  {'CAGR':<30} | {bh_cagr:>10.1%} | {old_cagr:>10.1%} | {new_cagr:>10.1%}")
    print(f"  {'Meta-Trades (Cash↔Long)':<30} | {'—':>10} | {old_trades_meta:>10.0f} | {new_trades_meta:>10.0f}")
    print(f"  {'Kurze Cash-Phasen (≤3T)':<30} | {'—':>10} | {old_short:>10} | {new_short:>10}")

    # EEM Detail um 14.03.2025
    if ticker == "EEM":
        window_start = "2025-03-05"
        window_end = "2025-03-25"
        mask = (df_strats.index >= window_start) & (df_strats.index <= window_end)

        print(f"\n  Detail um 14.03.2025:")
        print(f"  {'Datum':<12} | {'Kurs':>8} | {'ALT Strat':>12} | {'NEU Strat':>12} | {'ALT ret':>9} | {'NEU ret':>9}")
        print(f"  {'-'*12}-+-{'-'*8}-+-{'-'*12}-+-{'-'*12}-+-{'-'*9}-+-{'-'*9}")
        for idx in df_strats.index[mask]:
            print(f"  {idx.date()} | ${close.loc[idx]:>7.2f} | "
                  f"{old_strat.loc[idx]:>12} | {new_strat.loc[idx]:>12} | "
                  f"{old_ret.loc[idx]:>+8.3%} | {new_ret.loc[idx]:>+8.3%}")

    return {"ticker": ticker, "bh": bh_sharpe, "old": old_sharpe, "new": new_sharpe}


def main():
    print("=" * 70)
    print("  VERGLEICH: Alt (ohne Kosten/Hysterese) vs Neu (mit)")
    print("=" * 70)

    results = []
    for ticker, info in ETF_DEFS.items():
        r = test_etf(ticker, info["start"])
        results.append(r)

    print(f"\n{'='*70}")
    print(f"  ZUSAMMENFASSUNG")
    print(f"{'='*70}")
    print(f"\n  {'ETF':<8} | {'B&H Sharpe':>12} | {'ALT Sharpe':>12} | {'NEU Sharpe':>12} | {'NEU > B&H?':>12}")
    print(f"  {'-'*8}-+-{'-'*12}-+-{'-'*12}-+-{'-'*12}-+-{'-'*12}")
    for r in results:
        better = "✅ JA" if r["new"] > r["bh"] else "❌ NEIN"
        print(f"  {r['ticker']:<8} | {r['bh']:>12.3f} | {r['old']:>12.3f} | {r['new']:>12.3f} | {better:>12}")


if __name__ == "__main__":
    main()
