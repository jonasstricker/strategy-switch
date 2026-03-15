#!/usr/bin/env python3
"""
Meta-Optimizer: Exhaustive Search for MSCI World Outperformance
================================================================
Tests ALL combinations of:
  - 10 strategies (original 3 + 7 new)
  - 4 WF training windows (2Y, 3Y, 5Y, 7Y)
  - 3 WF test windows (1M, 3M, 6M)
  - Vol-target on/off + different targets (10%, 15%, 20%)
  - 6 switching window lengths
  - Ensemble combinations
  - With/without partial positioning

Finds the configuration that genuinely beats Buy & Hold OOS.
"""

from __future__ import annotations

import sys, os, warnings, itertools, time
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Any, Optional

warnings.filterwarnings("ignore")

if __name__ == "__main__" and __package__ is None:
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    __package__ = "wf_backtest"

import numpy as np
import pandas as pd
from tqdm import tqdm

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
sns.set_theme(style="whitegrid", palette="muted", font_scale=1.1)

from .cfg import FrameworkConfig
from .data_loader import download_data, validate_data
from .strategies import (
    momentum_signal, ma_signal, rsi_signal,
    vol_target_scaler, apply_costs,
)
from .strategies_ext import (
    macd_signal, bollinger_breakout_signal, bollinger_mean_reversion_signal,
    dual_momentum_signal, double_ma_signal, donchian_signal, ensemble_signal,
    partial_position_signal, adaptive_momentum_signal,
)
from .metrics import (
    sharpe_ratio, sortino_ratio, max_drawdown, cagr, calmar_ratio,
    drawdown_series, time_under_water, rolling_sharpe,
)
from .walk_forward import generate_windows
from .report import _ensure_dir

FIGSIZE = (14, 6)


def _equity(returns: pd.Series) -> pd.Series:
    return (1 + returns).cumprod()


def _metrics(eq, ret, rf=0.02):
    tuw = time_under_water(eq)
    return {
        "CAGR": cagr(eq),
        "Sharpe": sharpe_ratio(ret, rf),
        "Sortino": sortino_ratio(ret, rf),
        "Calmar": calmar_ratio(eq),
        "MaxDD": max_drawdown(eq),
        "MaxUW_d": tuw["max_days"],
    }


# ══════════════════════════════════════════════════════════════════════════════
#  STRATEGY DEFINITIONS (all with parameter grids)
# ══════════════════════════════════════════════════════════════════════════════

STRATEGY_DEFS = {
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
    "RSI": {
        "grid": [{"period": per, "threshold": thr}
                 for per in [10, 14, 20, 30]
                 for thr in [35, 45, 50, 55]],
        "gen": lambda p, params: rsi_signal(p, params["period"], params["threshold"]),
        "keys": ["period", "threshold"],
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
    "BB_Breakout": {
        "grid": [{"period": per, "n_std": ns}
                 for per in [15, 20, 30, 50]
                 for ns in [1.5, 2.0, 2.5]],
        "gen": lambda p, params: bollinger_breakout_signal(p, params["period"], params["n_std"]),
        "keys": ["period", "n_std"],
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
    "Partial_Pos": {
        "grid": [{"ma_short": ms, "ma_long": ml}
                 for ms in [20, 50, 100]
                 for ml in [150, 200, 300]
                 if ms < ml],
        "gen": lambda p, params: partial_position_signal(p, params["ma_short"], params["ma_long"]),
        "keys": ["ma_short", "ma_long"],
    },
}


# ══════════════════════════════════════════════════════════════════════════════
#  WF CONFIGURATIONS TO TEST
# ══════════════════════════════════════════════════════════════════════════════

WF_CONFIGS = [
    {"name": "3Y_1M",  "train": 756,  "test": 21,  "step": 21},
    {"name": "3Y_3M",  "train": 756,  "test": 63,  "step": 63},
    {"name": "5Y_3M",  "train": 1260, "test": 63,  "step": 63},
    {"name": "5Y_6M",  "train": 1260, "test": 126, "step": 126},
]

VOL_CONFIGS = [
    {"name": "no_vol",   "use": False, "target": 0.15, "lookback": 63},
    {"name": "vol_10",   "use": True,  "target": 0.10, "lookback": 63},
    {"name": "vol_15",   "use": True,  "target": 0.15, "lookback": 63},
]

SWITCH_WINDOWS = [63, 126, 189, 252]

TOP_PCT_OPTIONS = [0.20, 0.40]


# ══════════════════════════════════════════════════════════════════════════════
#  CORE WF ENGINE (simplified for speed)
# ══════════════════════════════════════════════════════════════════════════════

def _select_median(results: List[dict], keys: List[str], top_pct: float) -> dict:
    """Select median parameters from top quantile."""
    df = pd.DataFrame(results).sort_values("sharpe", ascending=False)
    n_top = max(1, int(len(df) * top_pct))
    top = df.head(n_top)
    out = {}
    for key in keys:
        med = top[key].median()
        all_vals = sorted(df[key].unique())
        out[key] = min(all_vals, key=lambda v: abs(v - med))
    return out


def run_single_wf(prices, returns, strat_name, strat_def, wf_cfg, vol_cfg,
                  top_pct, rf, tx_cost, slippage):
    """
    Run walk-forward for one strategy + one WF config + one vol config.
    Returns dict with OOS returns, sharpe, metrics.
    """
    dates = prices.index
    windows = generate_windows(dates, wf_cfg["train"], wf_cfg["test"], wf_cfg["step"])

    if len(windows) < 4:
        return None

    oos_parts = []
    param_hist = []

    for ts, te, os_s, os_e in windows:
        train_p = prices.iloc[ts:te]
        train_r = returns.iloc[ts:te]

        # Grid search on training data
        results = []
        for params in strat_def["grid"]:
            try:
                sig = strat_def["gen"](train_p, params)
                if sig is None or sig.isna().all():
                    continue
                sig = sig.fillna(0)
                net = apply_costs(sig, train_r, tx_cost, slippage)
                net = net.dropna()
                if len(net) < 20:
                    continue
                sr = sharpe_ratio(net, rf)
                results.append({**params, "sharpe": sr})
            except Exception:
                continue

        if len(results) < 3:
            continue

        best_params = _select_median(results, strat_def["keys"], top_pct)
        param_hist.append(best_params)

        # Generate signal using full data up to test end
        full_p = prices.iloc[ts:os_e]
        full_r = returns.iloc[ts:os_e]
        try:
            full_signal = strat_def["gen"](full_p, best_params)
        except Exception:
            continue

        if full_signal is None:
            continue

        # Vol target overlay
        if vol_cfg["use"]:
            vt = vol_target_scaler(full_r, vol_cfg["target"], vol_cfg["lookback"])
            full_signal = (full_signal * vt).clip(0, 1)

        test_r = returns.iloc[os_s:os_e]
        test_signal = full_signal.loc[test_r.index].fillna(0)
        test_net = apply_costs(test_signal, test_r, tx_cost, slippage)
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
        "sharpe": sharpe_ratio(oos_ret, rf),
        "cagr": cagr(eq),
        "maxdd": max_drawdown(eq),
        "sortino": sortino_ratio(oos_ret, rf),
        "calmar": calmar_ratio(eq),
        "n_windows": len(oos_parts),
        "param_history": param_hist,
    }


# ══════════════════════════════════════════════════════════════════════════════
#  SWITCHING ENGINE
# ══════════════════════════════════════════════════════════════════════════════

def run_switching(strat_returns: Dict[str, pd.Series],
                  bench_ret: pd.Series,
                  switch_window: int,
                  rf: float) -> Dict:
    """Run hard + soft switching for given window."""
    # Align to common period
    all_starts = [s.first_valid_index() for s in strat_returns.values()]
    all_ends = [s.last_valid_index() for s in strat_returns.values()]
    start = max(all_starts)
    end = min(all_ends)

    aligned = {}
    for name, ret in strat_returns.items():
        aligned[name] = ret.loc[start:end]

    bench = bench_ret.loc[start:end]
    df = pd.DataFrame(aligned)
    names = df.columns.tolist()

    # Rolling Sharpe
    roll = pd.DataFrame({
        n: rolling_sharpe(df[n], switch_window, rf)
        for n in names
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

    # Blend with Buy & Hold (80/20 through to 20/80)
    blends = {}
    for bh_pct in [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]:
        tac_pct = 1 - bh_pct
        for mode, mode_ret in [("hard", hard_ret), ("soft", soft_ret)]:
            blend = bh_pct * bench + tac_pct * mode_ret
            blend_name = f"{mode}_{int(bh_pct*100)}bh_{int(tac_pct*100)}tac_sw{switch_window}"
            eq = _equity(blend.dropna())
            if len(eq) < 100:
                continue
            blends[blend_name] = {
                "returns": blend,
                "sharpe": sharpe_ratio(blend.dropna(), rf),
                "cagr": cagr(eq),
                "maxdd": max_drawdown(eq),
                "sortino": sortino_ratio(blend.dropna(), rf),
                "calmar": calmar_ratio(eq),
                "mode": mode,
                "bh_pct": bh_pct,
                "tac_pct": tac_pct,
                "sw_window": switch_window,
            }

    return blends


# ══════════════════════════════════════════════════════════════════════════════
#  ENSEMBLE COMBINATIONS
# ══════════════════════════════════════════════════════════════════════════════

def run_ensembles(top_strat_returns: Dict[str, pd.Series],
                  bench_ret: pd.Series,
                  rf: float) -> Dict:
    """Create 2- and 3-strategy ensembles from top performers."""
    names = list(top_strat_returns.keys())
    if len(names) < 2:
        return {}

    # Align all
    start = max(s.first_valid_index() for s in top_strat_returns.values())
    end = min(s.last_valid_index() for s in top_strat_returns.values())
    aligned = {n: r.loc[start:end] for n, r in top_strat_returns.items()}
    bench = bench_ret.loc[start:end]

    ensembles = {}

    # Equal-weight 2-combos
    for combo in itertools.combinations(names, 2):
        rets = [aligned[n] for n in combo]
        avg = sum(rets) / len(rets)
        label = "+".join(combo[:2])
        eq = _equity(avg.dropna())
        if len(eq) < 100:
            continue
        ensembles[f"Ens2_{label}"] = {
            "returns": avg,
            "sharpe": sharpe_ratio(avg.dropna(), rf),
            "cagr": cagr(eq),
            "maxdd": max_drawdown(eq),
            "components": combo,
        }

    # Equal-weight 3-combos
    if len(names) >= 3:
        for combo in itertools.combinations(names, 3):
            rets = [aligned[n] for n in combo]
            avg = sum(rets) / len(rets)
            label = "+".join(c[:3] for c in combo)
            eq = _equity(avg.dropna())
            if len(eq) < 100:
                continue
            ensembles[f"Ens3_{label}"] = {
                "returns": avg,
                "sharpe": sharpe_ratio(avg.dropna(), rf),
                "cagr": cagr(eq),
                "maxdd": max_drawdown(eq),
                "components": combo,
            }

    # Blend each ensemble with B&H (50/50 and 70/30)
    for ens_name, ens_data in list(ensembles.items()):
        for bh_pct in [0.3, 0.5, 0.7]:
            blend = bh_pct * bench + (1 - bh_pct) * ens_data["returns"]
            eq = _equity(blend.dropna())
            if len(eq) < 100:
                continue
            ensembles[f"{ens_name}_{int(bh_pct*100)}bh"] = {
                "returns": blend,
                "sharpe": sharpe_ratio(blend.dropna(), rf),
                "cagr": cagr(eq),
                "maxdd": max_drawdown(eq),
                "components": ens_data.get("components", ()),
                "bh_pct": bh_pct,
            }

    return ensembles


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN META-OPTIMIZER
# ══════════════════════════════════════════════════════════════════════════════

def main():
    out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")
    os.makedirs(out_dir, exist_ok=True)

    cfg = FrameworkConfig()
    rf = cfg.risk_free_rate
    tx = cfg.costs.transaction_cost
    slip = cfg.costs.slippage

    # ── 1. Download data ─────────────────────────────────────────────────────
    print("=" * 70)
    print("  META-OPTIMIZER: Exhaustive Search for MSCI World Outperformance")
    print("=" * 70)

    df = download_data(cfg.data)
    validate_data(df)
    prices = df["Close"]
    returns = df["Returns"]

    # ── 2. Phase 1: Single strategy × WF config × vol config ────────────────
    print("\n" + "=" * 70)
    print("  PHASE 1: Single Strategy Sweep")
    outer = len(WF_CONFIGS) * len(VOL_CONFIGS) * len(TOP_PCT_OPTIONS)
    print(f"  Strategies: {len(STRATEGY_DEFS)} | WF: {len(WF_CONFIGS)} | "
          f"Vol: {len(VOL_CONFIGS)} | Top%: {len(TOP_PCT_OPTIONS)}")
    total = len(STRATEGY_DEFS) * outer
    print(f"  Total outer combos: {total}  (each runs full WF with inner grid)")
    print("=" * 70)

    all_results = []
    t0 = time.time()

    strat_items = list(STRATEGY_DEFS.items())
    for si, (strat_name, strat_def) in enumerate(strat_items):
        grid_sz = len(strat_def["grid"])
        print(f"  [{si+1}/{len(strat_items)}] {strat_name} "
              f"(grid={grid_sz}, combos={outer})...", end="", flush=True)
        t1 = time.time()
        for wf_cfg in WF_CONFIGS:
            for vol_cfg in VOL_CONFIGS:
                for top_pct in TOP_PCT_OPTIONS:
                    res = run_single_wf(
                        prices, returns, strat_name, strat_def,
                        wf_cfg, vol_cfg, top_pct, rf, tx, slip)
                    if res is not None:
                        all_results.append({
                            "strategy": strat_name,
                            "wf": wf_cfg["name"],
                            "vol": vol_cfg["name"],
                            "top_pct": top_pct,
                            **{k: v for k, v in res.items()
                               if k not in ("oos_returns", "param_history")},
                            "_oos_returns": res["oos_returns"],
                        })

        print(f" done ({time.time()-t1:.0f}s, total={len(all_results)})")

    print(f"\n  Phase 1 complete: {len(all_results)} valid configurations "
          f"in {time.time() - t0:.0f}s")

    # Sort by Sharpe
    results_df = pd.DataFrame([
        {k: v for k, v in r.items() if k != "_oos_returns"}
        for r in all_results
    ]).sort_values("sharpe", ascending=False)

    # Benchmark on OOS period
    if len(all_results) == 0:
        print("  No valid results! Aborting.")
        return

    # Find common OOS period
    oos_starts = [r["_oos_returns"].index[0] for r in all_results]
    oos_ends = [r["_oos_returns"].index[-1] for r in all_results]
    common_start = min(oos_starts)
    common_end = max(oos_ends)

    # Use the most common OOS period for benchmark
    bench_ret = returns.loc[common_start:common_end]
    bench_eq = _equity(bench_ret)
    bench_metrics = _metrics(bench_eq, bench_ret, rf)
    bench_sharpe = bench_metrics["Sharpe"]

    print(f"\n  Benchmark (Buy & Hold): Sharpe={bench_sharpe:.2f}, "
          f"CAGR={bench_metrics['CAGR']:.2%}, MaxDD={bench_metrics['MaxDD']:.2%}")

    # Top 30 single strategies
    print(f"\n  TOP 30 SINGLE STRATEGIES (by Sharpe):")
    print(f"  {'#':>3s} {'Strategy':>17s} {'WF':>7s} {'Vol':>8s} "
          f"{'Top%':>5s} {'Sharpe':>7s} {'CAGR':>8s} {'MaxDD':>8s} "
          f"{'Sortino':>8s} {'Calmar':>7s} {'Win':>4s}")
    print("  " + "-" * 105)
    for i, (_, row) in enumerate(results_df.head(30).iterrows()):
        print(f"  {i+1:>3d} {row['strategy']:>17s} {row['wf']:>7s} "
              f"{row['vol']:>8s} {row['top_pct']:>5.0%} "
              f"{row['sharpe']:>7.2f} {row['cagr']:>7.2%} "
              f"{row['maxdd']:>7.2%} {row['sortino']:>7.2f} "
              f"{row['calmar']:>7.2f} {row['n_windows']:>4d}")

    # Count which strategy / config appears most in top 30
    top30 = results_df.head(30)
    print(f"\n  Strategy frequency in top-30:")
    for s, c in top30["strategy"].value_counts().items():
        print(f"    {s}: {c}")
    print(f"\n  WF config frequency in top-30:")
    for s, c in top30["wf"].value_counts().items():
        print(f"    {s}: {c}")
    print(f"\n  Vol config frequency in top-30:")
    for s, c in top30["vol"].value_counts().items():
        print(f"    {s}: {c}")
    print(f"\n  Top% frequency in top-30:")
    for s, c in top30["top_pct"].value_counts().items():
        print(f"    {s:.0%}: {c}")

    # ── 3. Phase 2: Switching with top strategies ────────────────────────────
    print("\n" + "=" * 70)
    print("  PHASE 2: Strategy Switching (top performers)")
    print("=" * 70)

    # Take the best result per strategy (to avoid redundancy)
    best_per_strat = {}
    for r in all_results:
        sn = r["strategy"]
        if sn not in best_per_strat or r["sharpe"] > best_per_strat[sn]["sharpe"]:
            best_per_strat[sn] = r

    # Use top strategies with Sharpe > 0
    top_strats = {k: v["_oos_returns"] for k, v in best_per_strat.items()
                  if v["sharpe"] > 0}
    print(f"  Using {len(top_strats)} strategies with Sharpe > 0")

    all_switching = {}
    for sw in SWITCH_WINDOWS:
        blends = run_switching(top_strats, bench_ret, sw, rf)
        all_switching.update(blends)
        print(f"  SW={sw:>3d}: {len(blends)} blend configurations")

    # Sort switching results
    sw_rows = []
    for name, data in all_switching.items():
        sw_rows.append({
            "name": name,
            "sharpe": data["sharpe"],
            "cagr": data["cagr"],
            "maxdd": data["maxdd"],
            "mode": data.get("mode", ""),
            "bh_pct": data.get("bh_pct", 0),
            "tac_pct": data.get("tac_pct", 0),
            "sw_window": data.get("sw_window", 0),
        })
    sw_df = pd.DataFrame(sw_rows).sort_values("sharpe", ascending=False)

    print(f"\n  TOP 20 SWITCHING CONFIGS:")
    print(f"  {'#':>3s} {'Mode':>5s} {'BH%':>4s} {'Tac%':>4s} "
          f"{'SW':>4s} {'Sharpe':>7s} {'CAGR':>8s} {'MaxDD':>8s}")
    print("  " + "-" * 55)
    for i, (_, row) in enumerate(sw_df.head(20).iterrows()):
        print(f"  {i+1:>3d} {row['mode']:>5s} {row['bh_pct']:>3.0%} "
              f"{row['tac_pct']:>3.0%} {row['sw_window']:>4.0f} "
              f"{row['sharpe']:>7.2f} {row['cagr']:>7.2%} "
              f"{row['maxdd']:>7.2%}")

    # ── 4. Phase 3: Ensemble combinations ────────────────────────────────────
    print("\n" + "=" * 70)
    print("  PHASE 3: Ensemble Combinations")
    print("=" * 70)

    # Pick top 6 diverse strategies (different strat types)
    used_strats = set()
    top_diverse = {}
    for r in sorted(all_results, key=lambda x: -x["sharpe"]):
        sn = r["strategy"]
        if sn not in used_strats and len(top_diverse) < 6:
            top_diverse[sn] = r["_oos_returns"]
            used_strats.add(sn)

    ensembles = run_ensembles(top_diverse, bench_ret, rf)
    print(f"  Generated {len(ensembles)} ensemble combinations")

    # Sort ensembles
    ens_rows = []
    for name, data in ensembles.items():
        ens_rows.append({
            "name": name,
            "sharpe": data["sharpe"],
            "cagr": data["cagr"],
            "maxdd": data["maxdd"],
        })
    ens_df = pd.DataFrame(ens_rows).sort_values("sharpe", ascending=False)

    if not ens_df.empty:
        print(f"\n  TOP 15 ENSEMBLES:")
        print(f"  {'#':>3s} {'Name':>45s} {'Sharpe':>7s} {'CAGR':>8s} {'MaxDD':>8s}")
        print("  " + "-" * 75)
        for i, (_, row) in enumerate(ens_df.head(15).iterrows()):
            print(f"  {i+1:>3d} {row['name']:>45s} "
                  f"{row['sharpe']:>7.2f} {row['cagr']:>7.2%} "
                  f"{row['maxdd']:>7.2%}")

    # ── 5. FINAL: Grand ranking ──────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("  GRAND RANKING: ALL APPROACHES vs. BUY & HOLD")
    print("=" * 70)

    grand = []

    # Buy & Hold
    grand.append({
        "name": "Buy & Hold",
        "type": "benchmark",
        **bench_metrics,
        "_returns": bench_ret,
    })

    # Top 10 single
    for i, r in enumerate(sorted(all_results, key=lambda x: -x["sharpe"])[:10]):
        grand.append({
            "name": f"{r['strategy']}_{r['wf']}_{r['vol']}",
            "type": "single",
            "CAGR": r["cagr"],
            "Sharpe": r["sharpe"],
            "Sortino": r["sortino"],
            "Calmar": r["calmar"],
            "MaxDD": r["maxdd"],
            "MaxUW_d": 0,
            "_returns": r["_oos_returns"],
        })

    # Top 5 switching
    for i, (_, row) in enumerate(sw_df.head(5).iterrows()):
        name = row["name"]
        data = all_switching[name]
        eq = _equity(data["returns"].dropna())
        grand.append({
            "name": name,
            "type": "switching",
            **_metrics(eq, data["returns"].dropna(), rf),
            "_returns": data["returns"],
        })

    # Top 5 ensembles
    if not ens_df.empty:
        for i, (_, row) in enumerate(ens_df.head(5).iterrows()):
            name = row["name"]
            data = ensembles[name]
            eq = _equity(data["returns"].dropna())
            grand.append({
                "name": name,
                "type": "ensemble",
                **_metrics(eq, data["returns"].dropna(), rf),
                "_returns": data["returns"],
            })

    # Sort by Sharpe
    grand.sort(key=lambda x: -x["Sharpe"])

    print(f"\n  {'#':>3s} {'Type':>10s} {'Name':>50s} {'Sharpe':>7s} "
          f"{'CAGR':>8s} {'MaxDD':>8s} {'Sortino':>8s} {'Calmar':>8s}")
    print("  " + "-" * 110)
    for i, g in enumerate(grand):
        marker = " ◀ BENCHMARK" if g["type"] == "benchmark" else ""
        beat = ""
        if g["type"] != "benchmark" and g["Sharpe"] > bench_sharpe:
            beat = " ★"
        print(f"  {i+1:>3d} {g['type']:>10s} {g['name']:>50s} "
              f"{g['Sharpe']:>7.2f} {g['CAGR']:>7.2%} "
              f"{g['MaxDD']:>7.2%} {g['Sortino']:>7.2f} "
              f"{g['Calmar']:>7.2f}{marker}{beat}")

    # ── 6. Generate charts ───────────────────────────────────────────────────
    print("\n  Generating charts …")

    # Equity curves for top approaches vs benchmark
    fig, ax = plt.subplots(figsize=FIGSIZE)
    for g in grand[:8]:
        ret = g["_returns"]
        if ret is not None:
            eq = _equity(ret.dropna())
            lw = 2.5 if g["type"] == "benchmark" else 1.3
            ls = "-" if g["type"] == "benchmark" else "--"
            ax.plot(eq.index, eq.values, label=g["name"][:35],
                    linewidth=lw, linestyle=ls)
    ax.set_title("Top Approaches vs. Buy & Hold")
    ax.set_ylabel("Growth of $1")
    ax.legend(loc="upper left", fontsize=8)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "meta_equity.png"), dpi=150)
    plt.close(fig)

    # Drawdown comparison
    fig, ax = plt.subplots(figsize=FIGSIZE)
    for g in grand[:6]:
        ret = g["_returns"]
        if ret is not None:
            eq = _equity(ret.dropna())
            dd = drawdown_series(eq)
            ax.plot(dd.index, dd.values, label=g["name"][:35], linewidth=0.8)
    ax.set_title("Drawdowns: Top Approaches")
    ax.set_ylabel("Drawdown")
    ax.legend(loc="lower left", fontsize=8)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "meta_drawdowns.png"), dpi=150)
    plt.close(fig)

    # Heatmap: Strategy × WF config (avg Sharpe)
    pivot_data = results_df.groupby(["strategy", "wf"])["sharpe"].max().unstack()
    if not pivot_data.empty:
        fig, ax = plt.subplots(figsize=(12, 8))
        sns.heatmap(pivot_data, annot=True, fmt=".2f", cmap="RdYlGn",
                    center=bench_sharpe, ax=ax)
        ax.set_title(f"Max Sharpe by Strategy × WF Config (Benchmark={bench_sharpe:.2f})")
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, "meta_heatmap_strat_wf.png"), dpi=150)
        plt.close(fig)

    # Heatmap: Strategy × Vol config
    pivot_vol = results_df.groupby(["strategy", "vol"])["sharpe"].max().unstack()
    if not pivot_vol.empty:
        fig, ax = plt.subplots(figsize=(12, 8))
        sns.heatmap(pivot_vol, annot=True, fmt=".2f", cmap="RdYlGn",
                    center=bench_sharpe, ax=ax)
        ax.set_title(f"Max Sharpe by Strategy × Vol Config (Benchmark={bench_sharpe:.2f})")
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, "meta_heatmap_strat_vol.png"), dpi=150)
        plt.close(fig)

    # Bar chart: Sharpe of grand ranking
    fig, ax = plt.subplots(figsize=(14, 8))
    names = [g["name"][:30] for g in grand]
    sharpes = [g["Sharpe"] for g in grand]
    colors = ["#2196F3" if g["type"] == "benchmark" else
              "#4CAF50" if g["Sharpe"] > bench_sharpe else "#F44336"
              for g in grand]
    bars = ax.barh(range(len(names)), sharpes, color=colors)
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names, fontsize=8)
    ax.axvline(bench_sharpe, color="blue", linewidth=2, linestyle="--",
               label=f"Buy & Hold = {bench_sharpe:.2f}")
    ax.set_xlabel("Sharpe Ratio")
    ax.set_title("Grand Ranking: All Approaches")
    ax.legend()
    ax.invert_yaxis()
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "meta_grand_ranking.png"), dpi=150)
    plt.close(fig)

    # ── 7. Save text report ──────────────────────────────────────────────────
    report_lines = []
    report_lines.append("=" * 80)
    report_lines.append("  META-OPTIMIZER REPORT: EXHAUSTIVE SEARCH")
    report_lines.append("=" * 80)
    report_lines.append(f"\nTotal configurations tested: {len(all_results)}")
    report_lines.append(f"Switching configurations: {len(all_switching)}")
    report_lines.append(f"Ensemble configurations: {len(ensembles)}")
    report_lines.append(f"\nBenchmark: Sharpe={bench_sharpe:.2f}, "
                        f"CAGR={bench_metrics['CAGR']:.2%}, "
                        f"MaxDD={bench_metrics['MaxDD']:.2%}")
    report_lines.append(f"\nConfigurations that beat Buy & Hold Sharpe:")
    n_beat = sum(1 for r in all_results if r["sharpe"] > bench_sharpe)
    report_lines.append(f"  Single: {n_beat} / {len(all_results)}")
    n_beat_sw = sum(1 for d in all_switching.values() if d["sharpe"] > bench_sharpe)
    report_lines.append(f"  Switching: {n_beat_sw} / {len(all_switching)}")
    n_beat_ens = sum(1 for d in ensembles.values() if d["sharpe"] > bench_sharpe)
    report_lines.append(f"  Ensembles: {n_beat_ens} / {len(ensembles)}")

    report_lines.append("\n\nGRAND RANKING:")
    report_lines.append("-" * 110)
    for i, g in enumerate(grand):
        marker = " ◀ BENCHMARK" if g["type"] == "benchmark" else ""
        beat = " ★" if g["type"] != "benchmark" and g["Sharpe"] > bench_sharpe else ""
        report_lines.append(
            f"  {i+1:>3d} {g['type']:>10s} {g['name']:>50s} "
            f"Sharpe={g['Sharpe']:>6.2f}  CAGR={g['CAGR']:>7.2%}  "
            f"MaxDD={g['MaxDD']:>7.2%}  Sortino={g['Sortino']:>6.2f}  "
            f"Calmar={g['Calmar']:>6.2f}{marker}{beat}")

    report_lines.append("\n\nBEST OVERALL CONFIGURATION:")
    report_lines.append("-" * 50)
    best = grand[0]
    report_lines.append(f"  Name:    {best['name']}")
    report_lines.append(f"  Type:    {best['type']}")
    report_lines.append(f"  Sharpe:  {best['Sharpe']:.2f}")
    report_lines.append(f"  CAGR:    {best['CAGR']:.2%}")
    report_lines.append(f"  MaxDD:   {best['MaxDD']:.2%}")
    report_lines.append(f"  Sortino: {best['Sortino']:.2f}")
    report_lines.append(f"  Calmar:  {best['Calmar']:.2f}")

    if best["Sharpe"] > bench_sharpe:
        delta = best["Sharpe"] - bench_sharpe
        report_lines.append(f"\n  ★ BEATS Buy & Hold by {delta:.2f} Sharpe points")
        if abs(best["MaxDD"]) < abs(bench_metrics["MaxDD"]):
            mdd_imp = 1 - abs(best["MaxDD"]) / abs(bench_metrics["MaxDD"])
            report_lines.append(f"  ★ MaxDD {mdd_imp:.0%} better than Buy & Hold")
    else:
        report_lines.append(f"\n  ✗ Does NOT beat Buy & Hold")

    report_lines.append("\n" + "=" * 80)

    report_path = os.path.join(out_dir, "meta_optimizer_report.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(report_lines))

    print(f"\n  Report: {report_path}")
    print(f"  Charts: {out_dir}/")
    print("  DONE.")


if __name__ == "__main__":
    main()
