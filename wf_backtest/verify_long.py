#!/usr/bin/env python3
"""
Long-History Verification
==========================
Verifies whether the switching methodology still outperforms
MSCI World over a LONGER time horizon (~24 years).

Uses a synthetic MSCI World proxy: 70% SPY + 30% EFA (MSCI EAFE).
EFA available since Aug 2001 → ~24 years of test data.

Also runs sub-period analysis (2001-2009, 2009-2017, 2017-2026)
to check regime robustness.
"""

from __future__ import annotations

import sys, os, warnings, time, itertools
from typing import Dict, List, Tuple

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
    macd_signal, bollinger_breakout_signal,
    dual_momentum_signal, double_ma_signal, donchian_signal,
    partial_position_signal,
)
from .metrics import (
    sharpe_ratio, sortino_ratio, max_drawdown, cagr, calmar_ratio,
    drawdown_series, rolling_sharpe, time_under_water,
)
from .walk_forward import generate_windows

FIGSIZE = (16, 7)


# ══════════════════════════════════════════════════════════════════════════════
#  DATA: Synthetic MSCI World (70% SPY + 30% EFA)
# ══════════════════════════════════════════════════════════════════════════════

def download_synthetic_world(start="2000-01-01", end="2026-02-28"):
    """
    Download SPY + EFA, align, compute synthetic MSCI World returns.
    Also downloads IWDA.AS to validate the proxy correlation.
    """
    print("[data] Downloading SPY, EFA for synthetic MSCI World …")
    tickers = ["SPY", "EFA"]
    raw = yf.download(tickers, start=start, end=end,
                      auto_adjust=True, progress=False)

    if isinstance(raw.columns, pd.MultiIndex):
        close = raw["Close"]
    else:
        close = raw[["Close"]]

    close = close.dropna()
    print(f"[data]   SPY+EFA aligned: {len(close)} days "
          f"({close.index[0].date()} → {close.index[-1].date()})")

    # Daily returns
    spy_ret = close["SPY"].pct_change()
    efa_ret = close["EFA"].pct_change()

    # Synthetic MSCI World: 70% US + 30% International
    synth_ret = 0.70 * spy_ret + 0.30 * efa_ret
    synth_ret = synth_ret.dropna()

    # Build price series from returns
    synth_price = (1 + synth_ret).cumprod() * 100  # start at 100
    synth_price.name = "Close"

    df = pd.DataFrame({"Close": synth_price, "Returns": synth_ret})
    n_years = len(df) / 252
    print(f"[data]   Synthetic World: {len(df)} days ({n_years:.1f} years), "
          f"{df.index[0].date()} → {df.index[-1].date()}")

    # Validate vs IWDA.AS (overlap period)
    try:
        iwda = yf.download("IWDA.AS", start="2009-01-01", end=end,
                           auto_adjust=True, progress=False)
        if isinstance(iwda.columns, pd.MultiIndex):
            iwda.columns = iwda.columns.get_level_values(0)
        iwda_ret = iwda["Close"].pct_change().dropna()
        overlap = synth_ret.loc[synth_ret.index.intersection(iwda_ret.index)]
        iwda_ovl = iwda_ret.loc[overlap.index]
        corr = overlap.corr(iwda_ovl)
        print(f"[data]   Proxy correlation with IWDA.AS (overlap): {corr:.3f}")
    except Exception:
        print("[data]   Could not validate proxy (IWDA.AS download failed)")

    return df


# ══════════════════════════════════════════════════════════════════════════════
#  STRATEGY DEFS (same as meta_optimizer, kept lean)
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
        "grid": [{"lookback": lb} for lb in [40, 60, 90, 120, 160, 200, 252]],
        "gen": lambda p, params: momentum_signal(p, params["lookback"]),
        "keys": ["lookback"],
    },
    "MA": {
        "grid": [{"period": p} for p in [30, 50, 75, 100, 150, 200, 252]],
        "gen": lambda p, params: ma_signal(p, params["period"]),
        "keys": ["period"],
    },
    "MACD": {
        "grid": [{"fast": f, "slow": s, "signal": sig}
                 for f in [8, 12]
                 for s in [20, 26, 34]
                 for sig in [9, 12]
                 if f < s],
        "gen": lambda p, params: macd_signal(p, params["fast"], params["slow"], params["signal"]),
        "keys": ["fast", "slow", "signal"],
    },
    "Dual_Momentum": {
        "grid": [{"abs_lookback": al, "trend_period": tp}
                 for al in [60, 120, 200, 252]
                 for tp in [100, 200, 252]],
        "gen": lambda p, params: dual_momentum_signal(p, params["abs_lookback"], params["trend_period"]),
        "keys": ["abs_lookback", "trend_period"],
    },
    "Double_MA": {
        "grid": [{"fast": f, "slow": s}
                 for f in [20, 50, 100]
                 for s in [150, 200, 300]
                 if f < s],
        "gen": lambda p, params: double_ma_signal(p, params["fast"], params["slow"]),
        "keys": ["fast", "slow"],
    },
    "Donchian": {
        "grid": [{"period": per} for per in [20, 40, 55, 90, 120]],
        "gen": lambda p, params: donchian_signal(p, params["period"]),
        "keys": ["period"],
    },
}

WF_CONFIGS = [
    {"name": "3Y_1M", "train": 756,  "test": 21, "step": 21},
    {"name": "3Y_3M", "train": 756,  "test": 63, "step": 63},
    {"name": "5Y_3M", "train": 1260, "test": 63, "step": 63},
]

VOL_CONFIGS = [
    {"name": "no_vol", "use": False, "target": 0.15, "lookback": 63},
    {"name": "vol_15", "use": True,  "target": 0.15, "lookback": 63},
]

SWITCH_WINDOWS = [63, 126, 252]

RF = 0.02
TX = 0.001
SLIP = 0.0005


# ══════════════════════════════════════════════════════════════════════════════
#  WF ENGINE (same as meta_optimizer)
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


def _select_median(results, keys, top_pct):
    df = pd.DataFrame(results).sort_values("sharpe", ascending=False)
    n_top = max(1, int(len(df) * top_pct))
    top = df.head(n_top)
    out = {}
    for key in keys:
        med = top[key].median()
        all_vals = sorted(df[key].unique())
        out[key] = min(all_vals, key=lambda v: abs(v - med))
    return out


def run_single_wf(prices, returns, strat_def, wf_cfg, vol_cfg, top_pct=0.20):
    windows = generate_windows(prices.index, wf_cfg["train"], wf_cfg["test"], wf_cfg["step"])
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

        if len(results) < 3:
            continue

        best_params = _select_median(results, strat_def["keys"], top_pct)

        full_p = prices.iloc[ts:os_e]
        full_r = returns.iloc[ts:os_e]
        try:
            full_signal = strat_def["gen"](full_p, best_params)
        except Exception:
            continue
        if full_signal is None:
            continue

        if vol_cfg["use"]:
            vt = vol_target_scaler(full_r, vol_cfg["target"], vol_cfg["lookback"])
            full_signal = (full_signal * vt).clip(0, 1)

        test_r = returns.iloc[os_s:os_e]
        test_signal = full_signal.loc[test_r.index].fillna(0)
        test_net = apply_costs(test_signal, test_r, TX, SLIP)
        oos_parts.append(test_net)

    if len(oos_parts) < 4:
        return None

    oos_ret = pd.concat(oos_parts).sort_index()
    oos_ret = oos_ret[~oos_ret.index.duplicated(keep="first")]
    if len(oos_ret) < 100:
        return None

    eq = _equity(oos_ret)
    return {
        "oos_returns": oos_ret,
        "sharpe": sharpe_ratio(oos_ret, RF),
        "cagr": cagr(eq),
        "maxdd": max_drawdown(eq),
        "sortino": sortino_ratio(oos_ret, RF),
        "calmar": calmar_ratio(eq),
    }


def run_switching(strat_returns, bench_ret, switch_window):
    all_starts = [s.first_valid_index() for s in strat_returns.values()]
    all_ends = [s.last_valid_index() for s in strat_returns.values()]
    start = max(all_starts)
    end = min(all_ends)

    aligned = {n: r.loc[start:end] for n, r in strat_returns.items()}
    bench = bench_ret.loc[start:end]
    df = pd.DataFrame(aligned)
    names = df.columns.tolist()

    roll = pd.DataFrame({
        n: rolling_sharpe(df[n], switch_window, RF) for n in names
    })

    # Hard switch
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

    # Soft switch
    soft_ret = pd.Series(0.0, index=df.index)
    for idx in df.index:
        if idx not in roll.index:
            continue
        row = roll.loc[idx]
        eligible = row[row > 0]
        if eligible.empty:
            continue
        weights = eligible / eligible.sum()
        for name in weights.index:
            soft_ret.loc[idx] += weights[name] * df.loc[idx, name]

    blends = {}
    for bh_pct in [0.0, 0.1, 0.2, 0.3, 0.5]:
        tac_pct = 1 - bh_pct
        for mode, mode_ret in [("hard", hard_ret), ("soft", soft_ret)]:
            blend = bh_pct * bench + tac_pct * mode_ret
            name = f"{mode}_{int(bh_pct*100)}bh_{int(tac_pct*100)}tac_sw{switch_window}"
            eq = _equity(blend.dropna())
            if len(eq) < 100:
                continue
            blends[name] = {
                "returns": blend,
                "sharpe": sharpe_ratio(blend.dropna(), RF),
                "cagr": cagr(eq),
                "maxdd": max_drawdown(eq),
                "sortino": sortino_ratio(blend.dropna(), RF),
                "calmar": calmar_ratio(eq),
                "mode": mode, "bh_pct": bh_pct,
                "tac_pct": tac_pct, "sw_window": switch_window,
            }

    return blends


# ══════════════════════════════════════════════════════════════════════════════
#  SUB-PERIOD ANALYSIS
# ══════════════════════════════════════════════════════════════════════════════

def analyse_subperiod(label, prices, returns, bench_ret):
    """Run full sweep on a sub-period and return summary."""
    print(f"\n  --- Sub-period: {label} ---")
    print(f"      {prices.index[0].date()} → {prices.index[-1].date()} "
          f"({len(prices)/252:.1f} years)")

    # Phase 1: single strategies
    all_results = []
    for sname, sdef in STRATEGY_DEFS.items():
        for wf in WF_CONFIGS:
            for vc in VOL_CONFIGS:
                res = run_single_wf(prices, returns, sdef, wf, vc)
                if res is not None:
                    all_results.append({
                        "strategy": sname,
                        "wf": wf["name"],
                        "vol": vc["name"],
                        **{k: v for k, v in res.items() if k != "oos_returns"},
                        "_oos_returns": res["oos_returns"],
                    })

    if len(all_results) == 0:
        print("      No valid single-strategy results")
        return None

    # Phase 2: switching
    best_per_strat = {}
    for r in all_results:
        sn = r["strategy"]
        if sn not in best_per_strat or r["sharpe"] > best_per_strat[sn]["sharpe"]:
            best_per_strat[sn] = r

    top_strats = {k: v["_oos_returns"] for k, v in best_per_strat.items()
                  if v["sharpe"] > -0.5}

    all_switching = {}
    if len(top_strats) >= 2:
        for sw in SWITCH_WINDOWS:
            try:
                blends = run_switching(top_strats, bench_ret, sw)
                all_switching.update(blends)
            except Exception:
                pass

    # Bench metrics
    # Get common OOS period
    oos_starts = [r["_oos_returns"].index[0] for r in all_results]
    oos_ends = [r["_oos_returns"].index[-1] for r in all_results]
    common_start = min(oos_starts)
    common_end = max(oos_ends)
    b_ret = bench_ret.loc[common_start:common_end].dropna()
    b_eq = _equity(b_ret)
    b_met = _metrics(b_eq, b_ret)

    # Best single
    best_single = max(all_results, key=lambda x: x["sharpe"])
    # Best switching
    best_sw = max(all_switching.values(), key=lambda x: x["sharpe"]) if all_switching else None

    summary = {
        "label": label,
        "n_days": len(prices),
        "n_single": len(all_results),
        "n_switching": len(all_switching),
        "bench": b_met,
        "best_single": {
            "name": f"{best_single['strategy']}_{best_single['wf']}_{best_single['vol']}",
            **{k: v for k, v in best_single.items()
               if k not in ("strategy", "wf", "vol", "_oos_returns")},
        },
    }
    if best_sw:
        sw_eq = _equity(best_sw["returns"].dropna())
        summary["best_switching"] = {
            "name": f"{best_sw['mode']}_sw{best_sw['sw_window']}_"
                    f"{int(best_sw['bh_pct']*100)}bh",
            **_metrics(sw_eq, best_sw["returns"].dropna()),
            "_returns": best_sw["returns"],
        }
    else:
        summary["best_switching"] = None

    # Also save all single & switching for equity curves
    summary["_all_results"] = all_results
    summary["_all_switching"] = all_switching
    summary["_bench_ret"] = b_ret

    return summary


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")
    os.makedirs(out_dir, exist_ok=True)

    print("=" * 75)
    print("  LONG-HISTORY VERIFICATION: Synthetic MSCI World (70% SPY + 30% EFA)")
    print("=" * 75)

    # ── 1. Download extended data ────────────────────────────────────────────
    df = download_synthetic_world(start="2000-01-01", end="2026-02-28")
    prices = df["Close"]
    returns = df["Returns"]

    # ── 2. Full-period sweep ─────────────────────────────────────────────────
    print("\n" + "=" * 75)
    print("  FULL PERIOD SWEEP")
    print("=" * 75)

    t0 = time.time()
    all_results = []
    for si, (sname, sdef) in enumerate(STRATEGY_DEFS.items()):
        print(f"  [{si+1}/{len(STRATEGY_DEFS)}] {sname}...", end="", flush=True)
        t1 = time.time()
        for wf in WF_CONFIGS:
            for vc in VOL_CONFIGS:
                res = run_single_wf(prices, returns, sdef, wf, vc)
                if res is not None:
                    all_results.append({
                        "strategy": sname,
                        "wf": wf["name"],
                        "vol": vc["name"],
                        **{k: v for k, v in res.items() if k != "oos_returns"},
                        "_oos_returns": res["oos_returns"],
                    })
        print(f" done ({time.time()-t1:.0f}s)")

    print(f"\n  Phase 1: {len(all_results)} valid in {time.time()-t0:.0f}s")

    # Bench
    oos_starts = [r["_oos_returns"].index[0] for r in all_results]
    oos_ends = [r["_oos_returns"].index[-1] for r in all_results]
    common_start = min(oos_starts)
    common_end = max(oos_ends)
    bench_ret = returns.loc[common_start:common_end].dropna()
    bench_eq = _equity(bench_ret)
    bench_m = _metrics(bench_eq, bench_ret)

    # Switching
    best_per_strat = {}
    for r in all_results:
        sn = r["strategy"]
        if sn not in best_per_strat or r["sharpe"] > best_per_strat[sn]["sharpe"]:
            best_per_strat[sn] = r

    top_strats = {k: v["_oos_returns"] for k, v in best_per_strat.items()
                  if v["sharpe"] > -0.5}

    all_switching = {}
    for sw in SWITCH_WINDOWS:
        try:
            blends = run_switching(top_strats, bench_ret, sw)
            all_switching.update(blends)
        except Exception:
            pass

    # Grand ranking
    grand = []
    grand.append({"name": "Buy & Hold", "type": "benchmark",
                  **bench_m, "_returns": bench_ret})

    for r in sorted(all_results, key=lambda x: -x["sharpe"])[:8]:
        grand.append({
            "name": f"{r['strategy']}_{r['wf']}_{r['vol']}",
            "type": "single",
            "CAGR": r["cagr"], "Sharpe": r["sharpe"],
            "Sortino": r["sortino"], "Calmar": r["calmar"],
            "MaxDD": r["maxdd"], "MaxUW_d": 0,
            "_returns": r["_oos_returns"],
        })

    for name, data in sorted(all_switching.items(),
                              key=lambda x: -x[1]["sharpe"])[:5]:
        eq = _equity(data["returns"].dropna())
        grand.append({
            "name": name, "type": "switching",
            **_metrics(eq, data["returns"].dropna()),
            "_returns": data["returns"],
        })

    grand.sort(key=lambda x: -x["Sharpe"])

    print(f"\n  Benchmark: Sharpe={bench_m['Sharpe']:.2f}, "
          f"CAGR={bench_m['CAGR']:.2%}, MaxDD={bench_m['MaxDD']:.2%}")
    print(f"\n  GRAND RANKING — FULL PERIOD ({common_start.date()} → {common_end.date()}):")
    print(f"  {'#':>3s} {'Type':>10s} {'Name':>45s} {'Sharpe':>7s} "
          f"{'CAGR':>8s} {'MaxDD':>8s} {'Sortino':>8s} {'Calmar':>8s}")
    print("  " + "-" * 100)
    for i, g in enumerate(grand):
        mk = " ◀ BM" if g["type"] == "benchmark" else ""
        beat = " ★" if g["type"] != "benchmark" and g["Sharpe"] > bench_m["Sharpe"] else ""
        print(f"  {i+1:>3d} {g['type']:>10s} {g['name']:>45s} "
              f"{g['Sharpe']:>7.2f} {g['CAGR']:>7.2%} {g['MaxDD']:>7.2%} "
              f"{g['Sortino']:>7.2f} {g['Calmar']:>7.2f}{mk}{beat}")

    # Count beaters
    n_beat_single = sum(1 for r in all_results if r["sharpe"] > bench_m["Sharpe"])
    n_beat_sw = sum(1 for d in all_switching.values() if d["sharpe"] > bench_m["Sharpe"])
    print(f"\n  Beat B&H: {n_beat_single}/{len(all_results)} single, "
          f"{n_beat_sw}/{len(all_switching)} switching")

    # ── 3. Sub-period analysis ───────────────────────────────────────────────
    print("\n" + "=" * 75)
    print("  SUB-PERIOD ANALYSIS")
    print("=" * 75)

    # Define sub-periods
    periods = [
        ("2001–2009 (Dot-com + GFC)", "2001-01-01", "2009-12-31"),
        ("2010–2017 (Recovery/Bull)", "2010-01-01", "2017-12-31"),
        ("2018–2026 (Late Cycle+Covid)", "2018-01-01", "2026-12-31"),
    ]

    sub_results = []
    for label, s, e in periods:
        mask = (prices.index >= s) & (prices.index <= e)
        if mask.sum() < 504:  # need at least 2 years
            print(f"\n  Skipping {label}: insufficient data ({mask.sum()} days)")
            continue
        sub_p = prices.loc[mask]
        sub_r = returns.loc[mask]
        sub_br = returns.loc[mask]
        summary = analyse_subperiod(label, sub_p, sub_r, sub_br)
        if summary:
            sub_results.append(summary)

    # ── 4. Print comparison table ────────────────────────────────────────────
    print("\n\n" + "=" * 75)
    print("  COMPARISON: DOES SWITCHING BEAT BUY & HOLD ACROSS ALL PERIODS?")
    print("=" * 75)

    header = (f"  {'Period':>35s} │ {'B&H Sharpe':>10s} {'B&H CAGR':>9s} "
              f"{'B&H MaxDD':>9s} │ {'Best SW Sharpe':>13s} {'SW CAGR':>9s} "
              f"{'SW MaxDD':>9s} │ {'Winner':>8s}")
    print(header)
    print("  " + "─" * 115)

    comparison_rows = []

    # Full period
    best_sw_full = max(all_switching.values(), key=lambda x: x["sharpe"]) if all_switching else None
    if best_sw_full:
        sw_eq = _equity(best_sw_full["returns"].dropna())
        sw_m = _metrics(sw_eq, best_sw_full["returns"].dropna())
        winner = "Switch ★" if sw_m["Sharpe"] > bench_m["Sharpe"] else "B&H"
        print(f"  {'FULL PERIOD':>35s} │ {bench_m['Sharpe']:>10.2f} "
              f"{bench_m['CAGR']:>8.2%} {bench_m['MaxDD']:>8.2%} │ "
              f"{sw_m['Sharpe']:>13.2f} {sw_m['CAGR']:>8.2%} "
              f"{sw_m['MaxDD']:>8.2%} │ {winner:>8s}")
        comparison_rows.append({
            "period": "FULL",
            "bh_sharpe": bench_m["Sharpe"], "bh_cagr": bench_m["CAGR"],
            "bh_maxdd": bench_m["MaxDD"],
            "sw_sharpe": sw_m["Sharpe"], "sw_cagr": sw_m["CAGR"],
            "sw_maxdd": sw_m["MaxDD"],
            "winner": winner,
        })

    for sr in sub_results:
        bm = sr["bench"]
        if sr["best_switching"]:
            sw = sr["best_switching"]
            winner = "Switch ★" if sw["Sharpe"] > bm["Sharpe"] else "B&H"
            print(f"  {sr['label']:>35s} │ {bm['Sharpe']:>10.2f} "
                  f"{bm['CAGR']:>8.2%} {bm['MaxDD']:>8.2%} │ "
                  f"{sw['Sharpe']:>13.2f} {sw['CAGR']:>8.2%} "
                  f"{sw['MaxDD']:>8.2%} │ {winner:>8s}")
            comparison_rows.append({
                "period": sr["label"],
                "bh_sharpe": bm["Sharpe"], "bh_cagr": bm["CAGR"],
                "bh_maxdd": bm["MaxDD"],
                "sw_sharpe": sw["Sharpe"], "sw_cagr": sw["CAGR"],
                "sw_maxdd": sw["MaxDD"],
                "winner": winner,
            })
        else:
            print(f"  {sr['label']:>35s} │ {bm['Sharpe']:>10.2f} "
                  f"{bm['CAGR']:>8.2%} {bm['MaxDD']:>8.2%} │ "
                  f"{'N/A':>13s} {'N/A':>9s} {'N/A':>9s} │ {'N/A':>8s}")

    # Best single in each sub-period
    print(f"\n  BEST SINGLE STRATEGY PER PERIOD:")
    print(f"  {'Period':>35s} │ {'Strategy':>35s} │ {'Sharpe':>7s} "
          f"{'CAGR':>8s} {'MaxDD':>8s}")
    print("  " + "─" * 100)
    for sr in sub_results:
        bs = sr["best_single"]
        print(f"  {sr['label']:>35s} │ {bs['name']:>35s} │ "
              f"{bs['sharpe']:>7.2f} {bs['cagr']:>7.2%} {bs['maxdd']:>7.2%}")

    # ── 5. Generate charts ───────────────────────────────────────────────────
    print("\n  Generating charts …")

    # 5a) Full period equity curves
    fig, ax = plt.subplots(figsize=FIGSIZE)
    for g in grand[:8]:
        ret = g["_returns"]
        eq = _equity(ret.dropna())
        lw = 2.5 if g["type"] == "benchmark" else 1.2
        ls = "-" if g["type"] == "benchmark" else "--"
        ax.plot(eq.index, eq.values, label=g["name"][:35],
                linewidth=lw, linestyle=ls)
    ax.set_title(f"FULL PERIOD: Top Approaches vs Buy & Hold\n"
                 f"(Synthetic MSCI World = 70% SPY + 30% EFA)")
    ax.set_ylabel("Growth of $1")
    ax.set_yscale("log")
    ax.legend(loc="upper left", fontsize=8)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "verify_long_equity.png"), dpi=150)
    plt.close(fig)

    # 5b) Drawdowns
    fig, ax = plt.subplots(figsize=FIGSIZE)
    for g in grand[:6]:
        ret = g["_returns"]
        eq = _equity(ret.dropna())
        dd = drawdown_series(eq)
        ax.fill_between(dd.index, dd.values, alpha=0.2)
        ax.plot(dd.index, dd.values, label=g["name"][:35], linewidth=0.8)
    ax.set_title("Drawdowns: Full Period")
    ax.set_ylabel("Drawdown")
    ax.legend(loc="lower left", fontsize=8)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "verify_long_drawdowns.png"), dpi=150)
    plt.close(fig)

    # 5c) Sub-period comparison bar chart
    if comparison_rows:
        cdf = pd.DataFrame(comparison_rows)
        x = np.arange(len(cdf))
        width = 0.35
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))

        # Sharpe
        axes[0].bar(x - width/2, cdf["bh_sharpe"], width, label="Buy & Hold",
                    color="#2196F3")
        axes[0].bar(x + width/2, cdf["sw_sharpe"], width, label="Best Switching",
                    color="#4CAF50")
        axes[0].set_ylabel("Sharpe Ratio")
        axes[0].set_xticks(x)
        axes[0].set_xticklabels(cdf["period"].str[:15], rotation=30, fontsize=8)
        axes[0].legend()
        axes[0].set_title("Sharpe by Period")

        # CAGR
        axes[1].bar(x - width/2, cdf["bh_cagr"] * 100, width, label="Buy & Hold",
                    color="#2196F3")
        axes[1].bar(x + width/2, cdf["sw_cagr"] * 100, width, label="Best Switching",
                    color="#4CAF50")
        axes[1].set_ylabel("CAGR (%)")
        axes[1].set_xticks(x)
        axes[1].set_xticklabels(cdf["period"].str[:15], rotation=30, fontsize=8)
        axes[1].legend()
        axes[1].set_title("CAGR by Period")

        # MaxDD
        axes[2].bar(x - width/2, cdf["bh_maxdd"] * 100, width, label="Buy & Hold",
                    color="#2196F3")
        axes[2].bar(x + width/2, cdf["sw_maxdd"] * 100, width, label="Best Switching",
                    color="#4CAF50")
        axes[2].set_ylabel("Max Drawdown (%)")
        axes[2].set_xticks(x)
        axes[2].set_xticklabels(cdf["period"].str[:15], rotation=30, fontsize=8)
        axes[2].legend()
        axes[2].set_title("Max Drawdown by Period")

        fig.suptitle("Sub-Period Comparison: Buy & Hold vs. Best Switching",
                     fontsize=14, fontweight="bold")
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, "verify_long_subperiods.png"), dpi=150)
        plt.close(fig)

    # ── 6. Save report ───────────────────────────────────────────────────────
    lines = []
    lines.append("=" * 80)
    lines.append("  LONG-HISTORY VERIFICATION REPORT")
    lines.append("  Synthetic MSCI World = 70% SPY + 30% EFA")
    lines.append(f"  Data: {prices.index[0].date()} → {prices.index[-1].date()} "
                 f"({len(prices)/252:.1f} years)")
    lines.append("=" * 80)

    lines.append(f"\nFull-period OOS: {common_start.date()} → {common_end.date()}")
    lines.append(f"Single configs: {len(all_results)}")
    lines.append(f"Switching configs: {len(all_switching)}")
    lines.append(f"Beat B&H (single): {n_beat_single}/{len(all_results)}")
    lines.append(f"Beat B&H (switching): {n_beat_sw}/{len(all_switching)}")

    lines.append("\n\nGRAND RANKING:")
    lines.append("-" * 100)
    for i, g in enumerate(grand):
        mk = " ◀ BENCHMARK" if g["type"] == "benchmark" else ""
        beat = " ★" if g["type"] != "benchmark" and g["Sharpe"] > bench_m["Sharpe"] else ""
        lines.append(
            f"  {i+1:>3d} {g['type']:>10s} {g['name']:>45s} "
            f"Sharpe={g['Sharpe']:>6.2f}  CAGR={g['CAGR']:>7.2%}  "
            f"MaxDD={g['MaxDD']:>7.2%}  Sortino={g['Sortino']:>6.2f}  "
            f"Calmar={g['Calmar']:>6.2f}{mk}{beat}")

    lines.append("\n\nSUB-PERIOD COMPARISON:")
    lines.append("-" * 100)
    for cr in comparison_rows:
        lines.append(
            f"  {cr['period']:>35s}  "
            f"B&H: Sharpe={cr['bh_sharpe']:.2f} CAGR={cr['bh_cagr']:.2%} MaxDD={cr['bh_maxdd']:.2%}  "
            f"SW: Sharpe={cr['sw_sharpe']:.2f} CAGR={cr['sw_cagr']:.2%} MaxDD={cr['sw_maxdd']:.2%}  "
            f"→ {cr['winner']}")

    lines.append("\n\nBEST SINGLE PER PERIOD:")
    lines.append("-" * 100)
    for sr in sub_results:
        bs = sr["best_single"]
        lines.append(f"  {sr['label']:>35s}  {bs['name']:>35s}  "
                     f"Sharpe={bs['sharpe']:.2f}  CAGR={bs['cagr']:.2%}  MaxDD={bs['maxdd']:.2%}")

    # Verdict
    lines.append("\n" + "=" * 80)
    n_sw_wins = sum(1 for cr in comparison_rows if "Switch" in cr["winner"])
    n_total = len(comparison_rows)
    lines.append(f"  VERDICT: Switching wins {n_sw_wins}/{n_total} periods")
    if n_sw_wins == n_total:
        lines.append("  ★★★ SWITCHING OUTPERFORMS IN ALL TESTED PERIODS ★★★")
    elif n_sw_wins > n_total / 2:
        lines.append("  ★★ SWITCHING OUTPERFORMS IN MAJORITY OF PERIODS ★★")
    else:
        lines.append("  ✗ SWITCHING DOES NOT CONSISTENTLY OUTPERFORM")
    lines.append("=" * 80)

    report_path = os.path.join(out_dir, "verify_long_report.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print(f"\n  Report: {report_path}")
    print(f"  Charts: {out_dir}/verify_long_*.png")
    print("  DONE.")


if __name__ == "__main__":
    main()
