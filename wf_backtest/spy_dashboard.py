#!/usr/bin/env python3
"""
SPY Strategy-Switching Performance Dashboard
==============================================
Detailed performance analysis of the recommended approach:
  - Hard switching across WF-optimized strategies on SPY
  - 63-day rolling Sharpe selection window
  - 3Y training, 1M test WF configuration
"""

from __future__ import annotations
import sys, os, warnings
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

RF = 0.02; TX = 0.001; SLIP = 0.0005

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
        train_p, train_r = prices.iloc[ts:te], returns.iloc[ts:te]
        results = []
        for params in sdef["grid"]:
            try:
                sig = sdef["gen"](train_p, params)
                if sig is None or sig.isna().all(): continue
                net = apply_costs(sig.fillna(0), train_r, TX, SLIP).dropna()
                if len(net) < 20: continue
                results.append({**params, "sharpe": sharpe_ratio(net, RF)})
            except: continue
        if len(results) < 2: continue
        bp = _select_median(results, sdef["keys"])
        try:
            full_sig = sdef["gen"](prices.iloc[ts:os_e], bp)
        except: continue
        if full_sig is None: continue
        test_r = returns.iloc[os_s:os_e]
        test_sig = full_sig.loc[test_r.index].fillna(0)
        oos.append(apply_costs(test_sig, test_r, TX, SLIP))
    if len(oos) < 4: return None
    r = pd.concat(oos).sort_index()
    return r[~r.index.duplicated(keep="first")]


def main():
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")
    os.makedirs(out, exist_ok=True)

    print("=" * 70)
    print("  SPY STRATEGY-SWITCHING: DETAILED PERFORMANCE DASHBOARD")
    print("=" * 70)

    # Download SPY
    raw = yf.download("SPY", start="2010-01-01", end="2026-02-28",
                      auto_adjust=True, progress=False)
    if isinstance(raw.columns, pd.MultiIndex):
        close_col = [c for c in raw.columns if c[0] == "Close"]
        close = raw[close_col[0]].dropna() if close_col else None
    else:
        close = raw["Close"].dropna()
    returns = close.pct_change().dropna()
    prices = close.loc[returns.index]

    # ── 1. Run per-strategy WF timing ────────────────────────────────────────
    print("\n  Running WF timing per strategy …")
    strat_oos = {}
    for sname, sdef in STRATEGY_DEFS.items():
        r = wf_single(prices, returns, sdef)
        if r is not None:
            sr = sharpe_ratio(r, RF)
            strat_oos[sname] = r
            print(f"    {sname:>15s}: Sharpe={sr:.2f}, n={len(r)}")

    # ── 2. Build switching ────────────────────────────────────────────────────
    start = max(s.first_valid_index() for s in strat_oos.values())
    end = min(s.last_valid_index() for s in strat_oos.values())
    aligned = {n: r.loc[start:end] for n, r in strat_oos.items()}
    bench_ret = returns.loc[start:end]
    df_strats = pd.DataFrame(aligned)
    names = df_strats.columns.tolist()

    sw = 63
    roll = pd.DataFrame({n: rolling_sharpe(df_strats[n], sw, RF) for n in names})

    hard_ret = pd.Series(0.0, index=df_strats.index)
    active_strat = pd.Series("Cash", index=df_strats.index)
    for idx in df_strats.index:
        if idx not in roll.index: continue
        row = roll.loc[idx]
        eligible = row[row > 0]
        if eligible.empty: continue
        best = eligible.idxmax()
        hard_ret.loc[idx] = df_strats.loc[idx, best]
        active_strat.loc[idx] = best

    # ── 2b. Build trade log ──────────────────────────────────────────────────
    # Full switching log (internal)
    all_switches = []
    prev = "Cash"
    for idx in active_strat.index:
        cur = active_strat.loc[idx]
        if cur != prev:
            all_switches.append({
                "Datum": idx.strftime("%Y-%m-%d"),
                "Von": prev,
                "Nach": cur,
                "SPY_Kurs": float(close.loc[idx]),
            })
            prev = cur

    # ── ACTIONABLE trade log (nur echte Käufe/Verkäufe) ──────────────────
    # From the user's perspective, only CASH↔LONG matters:
    #   Cash → [any strategy] = KAUF SPY
    #   [any strategy] → Cash = VERKAUF SPY
    #   [strategy A] → [strategy B] = KEIN TRADE (beide = Long SPY)
    real_trades = []
    position = "Cash"  # tracks actual SPY position: "Cash" or "Long"
    for sw in all_switches:
        was_cash = (sw["Von"] == "Cash")
        now_cash = (sw["Nach"] == "Cash")
        if was_cash and not now_cash:
            # BUY SPY
            real_trades.append({
                "Datum": sw["Datum"],
                "Aktion": "KAUF SPY",
                "Grund": f"Strategie {sw['Nach']} aktiv",
                "SPY_Kurs": sw["SPY_Kurs"],
                "Kosten": TX + SLIP,
                "Position_danach": "LONG SPY",
            })
            position = "Long"
        elif not was_cash and now_cash:
            # SELL SPY
            real_trades.append({
                "Datum": sw["Datum"],
                "Aktion": "VERKAUF SPY",
                "Grund": f"Keine Strategie positiv → Cash",
                "SPY_Kurs": sw["SPY_Kurs"],
                "Kosten": TX + SLIP,
                "Position_danach": "CASH",
            })
            position = "Cash"
        # else: strategy→strategy = no action needed (stay in SPY)

    trade_df = pd.DataFrame(real_trades)
    trade_csv = os.path.join(out, "spy_trade_log.csv")
    trade_df.to_csv(trade_csv, index=False, sep=";")

    # Also save the full switching log
    full_csv = os.path.join(out, "spy_full_switch_log.csv")
    pd.DataFrame(all_switches).to_csv(full_csv, index=False, sep=";")

    # ── Print actionable trade log ────────────────────────────────────────
    print(f"\n  ═══ ECHTE TRADES (nur KAUF/VERKAUF SPY) ═══")
    print(f"  Strategie-Wechsel (z.B. RSI→Momentum) sind KEIN Trade,")
    print(f"  weil du in beiden Fällen SPY hältst!")
    print(f"\n  {'─'*85}")
    print(f"  {'#':>4s} │ {'Datum':>12s} │ {'Aktion':>14s} │ {'SPY Kurs':>10s} │ {'Kosten':>8s} │ Grund")
    print(f"  {'─'*85}")
    for i, t in enumerate(real_trades, 1):
        print(f"  {i:>4d} │ {t['Datum']:>12s} │ {t['Aktion']:>14s} │ "
              f"${t['SPY_Kurs']:>8.2f} │ {t['Kosten']:>7.2%} │ {t['Grund']}")
    print(f"  {'─'*85}")

    n_buys = sum(1 for t in real_trades if "KAUF" in t["Aktion"])
    n_sells = sum(1 for t in real_trades if "VERKAUF" in t["Aktion"])
    n_roundtrips = min(n_buys, n_sells)
    total_cost_pct = len(real_trades) * (TX + SLIP)
    cost_pa = total_cost_pct / (len(bench_ret) / 252)

    print(f"\n  Trade-Statistik:")
    print(f"    Echte Käufe:            {n_buys}")
    print(f"    Echte Verkäufe:         {n_sells}")
    print(f"    Round-Trips:            {n_roundtrips}")
    print(f"    Ø Trades pro Jahr:      {len(real_trades) / (len(bench_ret)/252):.1f}")
    print(f"    Kosten pro Trade:       {TX+SLIP:.2%} (0.10% Gebühren + 0.05% Slippage)")
    print(f"    Kosten Round-Trip:      {2*(TX+SLIP):.2%}")
    print(f"    Gesamtkosten (kum.):    {total_cost_pct:.2%}")
    print(f"    Kosten pro Jahr:        {cost_pa:.2%}")
    print(f"\n  Strategie-Interne Wechsel (kein Trade nötig): "
          f"{len(all_switches) - len(real_trades)}")
    print(f"  Trade-Log CSV:            {trade_csv}")
    print(f"  Vollst. Switch-Log CSV:   {full_csv}")

    # ── Holding periods ────────────────────────────────────────────────────
    print(f"\n  HALTEPERIODEN:")
    holding_days = []
    for i in range(0, len(real_trades)-1, 2):
        if i+1 < len(real_trades):
            buy_date = pd.Timestamp(real_trades[i]["Datum"])
            sell_date = pd.Timestamp(real_trades[i+1]["Datum"])
            days = (sell_date - buy_date).days
            buy_p = real_trades[i]["SPY_Kurs"]
            sell_p = real_trades[i+1]["SPY_Kurs"]
            ret = (sell_p / buy_p - 1) * 100
            holding_days.append(days)
            if i < 20 or i >= len(real_trades)-6:  # show first 10 and last 3 round-trips
                print(f"    {real_trades[i]['Datum']} → {real_trades[i+1]['Datum']}:  "
                      f"{days:>4d} Tage  |  ${buy_p:.0f}→${sell_p:.0f}  |  {ret:>+6.1f}%")
            elif i == 20:
                print(f"    ... ({n_roundtrips - 13} weitere Round-Trips) ...")

    if holding_days:
        print(f"\n    Ø Haltedauer:     {np.mean(holding_days):.0f} Tage")
        print(f"    Median Halted.:   {np.median(holding_days):.0f} Tage")
        print(f"    Längste Pos.:     {max(holding_days)} Tage")
        print(f"    Kürzeste Pos.:    {min(holding_days)} Tage")

    # ── 3. Compute metrics ────────────────────────────────────────────────────
    bh_eq = _equity(bench_ret)
    sw_eq = _equity(hard_ret)

    tuw_bh = time_under_water(bh_eq)
    tuw_sw = time_under_water(sw_eq)

    print(f"\n  OOS Period: {start.date()} → {end.date()} "
          f"({len(bench_ret)/252:.1f} years)")

    metrics = {
        "Buy & Hold SPY": {
            "CAGR": cagr(bh_eq),
            "Sharpe": sharpe_ratio(bench_ret, RF),
            "Sortino": sortino_ratio(bench_ret, RF),
            "Calmar": calmar_ratio(bh_eq),
            "MaxDD": max_drawdown(bh_eq),
            "MaxUW": tuw_bh["max_days"],
            "AvgUW": tuw_bh["avg_days"],
            "Vol_ann": bench_ret.std() * np.sqrt(252),
            "Best_Year": bench_ret.groupby(bench_ret.index.year).sum().max(),
            "Worst_Year": bench_ret.groupby(bench_ret.index.year).sum().min(),
            "Win_Rate": (bench_ret > 0).mean(),
            "Avg_Win": bench_ret[bench_ret > 0].mean(),
            "Avg_Loss": bench_ret[bench_ret < 0].mean(),
            "Skew": bench_ret.skew(),
            "Kurtosis": bench_ret.kurtosis(),
        },
        "Strategy Switch": {
            "CAGR": cagr(sw_eq),
            "Sharpe": sharpe_ratio(hard_ret, RF),
            "Sortino": sortino_ratio(hard_ret, RF),
            "Calmar": calmar_ratio(sw_eq),
            "MaxDD": max_drawdown(sw_eq),
            "MaxUW": tuw_sw["max_days"],
            "AvgUW": tuw_sw["avg_days"],
            "Vol_ann": hard_ret.std() * np.sqrt(252),
            "Best_Year": hard_ret.groupby(hard_ret.index.year).sum().max(),
            "Worst_Year": hard_ret.groupby(hard_ret.index.year).sum().min(),
            "Win_Rate": (hard_ret > 0).mean(),
            "Avg_Win": hard_ret[hard_ret > 0].mean(),
            "Avg_Loss": hard_ret[hard_ret < 0].mean(),
            "Skew": hard_ret.skew(),
            "Kurtosis": hard_ret.kurtosis(),
        },
    }

    # ── 4. Print performance table ────────────────────────────────────────────
    print(f"\n  {'Metric':>25s} │ {'Buy & Hold SPY':>16s} │ {'Strategy Switch':>16s} │ {'Δ':>12s}")
    print(f"  {'─'*75}")
    m_bh = metrics["Buy & Hold SPY"]
    m_sw = metrics["Strategy Switch"]
    rows = [
        ("CAGR", f"{m_bh['CAGR']:.2%}", f"{m_sw['CAGR']:.2%}", f"{m_sw['CAGR']-m_bh['CAGR']:+.2%}"),
        ("Sharpe Ratio", f"{m_bh['Sharpe']:.2f}", f"{m_sw['Sharpe']:.2f}", f"{m_sw['Sharpe']-m_bh['Sharpe']:+.2f}"),
        ("Sortino Ratio", f"{m_bh['Sortino']:.2f}", f"{m_sw['Sortino']:.2f}", f"{m_sw['Sortino']-m_bh['Sortino']:+.2f}"),
        ("Calmar Ratio", f"{m_bh['Calmar']:.2f}", f"{m_sw['Calmar']:.2f}", f"{m_sw['Calmar']-m_bh['Calmar']:+.2f}"),
        ("Max Drawdown", f"{m_bh['MaxDD']:.2%}", f"{m_sw['MaxDD']:.2%}", f"{(1-abs(m_sw['MaxDD'])/abs(m_bh['MaxDD']))*100:+.0f}% besser"),
        ("Max Underwater (Tage)", f"{m_bh['MaxUW']}", f"{m_sw['MaxUW']}", f"{m_sw['MaxUW']-m_bh['MaxUW']:+d}"),
        ("Avg Underwater (Tage)", f"{m_bh['AvgUW']:.0f}", f"{m_sw['AvgUW']:.0f}", ""),
        ("Ann. Volatilität", f"{m_bh['Vol_ann']:.2%}", f"{m_sw['Vol_ann']:.2%}", f"{m_sw['Vol_ann']-m_bh['Vol_ann']:+.2%}"),
        ("Bestes Jahr", f"{m_bh['Best_Year']:.2%}", f"{m_sw['Best_Year']:.2%}", ""),
        ("Schlechtestes Jahr", f"{m_bh['Worst_Year']:.2%}", f"{m_sw['Worst_Year']:.2%}", ""),
        ("Win Rate (täglich)", f"{m_bh['Win_Rate']:.1%}", f"{m_sw['Win_Rate']:.1%}", ""),
        ("Avg Win", f"{m_bh['Avg_Win']:.3%}", f"{m_sw['Avg_Win']:.3%}", ""),
        ("Avg Loss", f"{m_bh['Avg_Loss']:.3%}", f"{m_sw['Avg_Loss']:.3%}", ""),
        ("Schiefe", f"{m_bh['Skew']:.2f}", f"{m_sw['Skew']:.2f}", ""),
        ("Kurtosis", f"{m_bh['Kurtosis']:.2f}", f"{m_sw['Kurtosis']:.2f}", ""),
    ]
    for label, v1, v2, delta in rows:
        print(f"  {label:>25s} │ {v1:>16s} │ {v2:>16s} │ {delta:>12s}")

    # ── 5. Yearly returns ─────────────────────────────────────────────────────
    bh_yearly = bench_ret.groupby(bench_ret.index.year).sum()
    sw_yearly = hard_ret.groupby(hard_ret.index.year).sum()
    years = sorted(set(bh_yearly.index) & set(sw_yearly.index))

    print(f"\n  {'Jahr':>6s} │ {'B&H':>8s} │ {'Switch':>8s} │ {'Δ':>8s} │ Gewinner")
    print(f"  {'─'*55}")
    bh_wins = 0; sw_wins = 0
    for y in years:
        b = bh_yearly.get(y, 0)
        s = sw_yearly.get(y, 0)
        winner = "Switch ★" if s > b else "B&H"
        if s > b: sw_wins += 1
        else: bh_wins += 1
        print(f"  {y:>6d} │ {b:>7.2%} │ {s:>7.2%} │ {s-b:>+7.2%} │ {winner}")
    print(f"  {'─'*55}")
    print(f"  Switch gewinnt {sw_wins}/{len(years)} Jahre ({sw_wins/len(years):.0%})")

    # ── 6. Strategy allocation over time ──────────────────────────────────────
    alloc_counts = active_strat.value_counts()
    print(f"\n  STRATEGIE-ALLOKATION (% der Zeit):")
    for s, c in alloc_counts.sort_values(ascending=False).items():
        pct = c / len(active_strat)
        print(f"    {s:>15s}: {pct:>6.1%} ({c} Tage)")

    # ── 7. Per-strategy standalone metrics ────────────────────────────────────
    print(f"\n  EINZELSTRATEGIE-PERFORMANCE:")
    print(f"  {'Strategie':>15s} │ {'Sharpe':>7s} │ {'CAGR':>8s} │ {'MaxDD':>8s}")
    print(f"  {'─'*50}")
    for sn in sorted(strat_oos, key=lambda x: -sharpe_ratio(strat_oos[x].loc[start:end], RF)):
        r = strat_oos[sn].loc[start:end]
        eq = _equity(r)
        print(f"  {sn:>15s} │ {sharpe_ratio(r, RF):>7.2f} │ {cagr(eq):>7.2%} │ {max_drawdown(eq):>7.2%}")

    # ══════════════════════════════════════════════════════════════════════════
    #  CHARTS (8 panels)
    # ══════════════════════════════════════════════════════════════════════════
    print("\n  Generating dashboard …")

    fig = plt.figure(figsize=(20, 24))
    fig.suptitle("SPY Strategy-Switching — Performance Dashboard",
                 fontsize=18, fontweight="bold", y=0.98)

    # 1) Equity curves
    ax1 = fig.add_subplot(4, 2, 1)
    ax1.plot(bh_eq.index, bh_eq.values, label="Buy & Hold SPY",
             linewidth=2.5, color="#2196F3")
    ax1.plot(sw_eq.index, sw_eq.values, label="Strategy Switch",
             linewidth=2, color="#4CAF50", linestyle="--")
    ax1.set_title("Equity Curves")
    ax1.set_ylabel("Growth of $1")
    ax1.legend(fontsize=9)
    ax1.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

    # 2) Equity curves (log)
    ax2 = fig.add_subplot(4, 2, 2)
    ax2.plot(bh_eq.index, bh_eq.values, label="Buy & Hold SPY",
             linewidth=2.5, color="#2196F3")
    ax2.plot(sw_eq.index, sw_eq.values, label="Strategy Switch",
             linewidth=2, color="#4CAF50", linestyle="--")
    ax2.set_yscale("log")
    ax2.set_title("Equity Curves (Log Scale)")
    ax2.set_ylabel("Growth of $1 (log)")
    ax2.legend(fontsize=9)
    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

    # 3) Drawdowns
    ax3 = fig.add_subplot(4, 2, 3)
    dd_bh = drawdown_series(bh_eq)
    dd_sw = drawdown_series(sw_eq)
    ax3.fill_between(dd_bh.index, dd_bh.values, alpha=0.3, color="#2196F3")
    ax3.plot(dd_bh.index, dd_bh.values, color="#2196F3", linewidth=0.8,
             label="Buy & Hold")
    ax3.fill_between(dd_sw.index, dd_sw.values, alpha=0.3, color="#4CAF50")
    ax3.plot(dd_sw.index, dd_sw.values, color="#4CAF50", linewidth=0.8,
             label="Strategy Switch")
    ax3.set_title("Drawdowns")
    ax3.set_ylabel("Drawdown")
    ax3.legend(fontsize=9)
    ax3.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

    # 4) Rolling 1Y Sharpe
    ax4 = fig.add_subplot(4, 2, 4)
    rs_bh = rolling_sharpe(bench_ret, 252, RF)
    rs_sw = rolling_sharpe(hard_ret, 252, RF)
    ax4.plot(rs_bh.index, rs_bh.values, label="Buy & Hold",
             color="#2196F3", linewidth=0.8)
    ax4.plot(rs_sw.index, rs_sw.values, label="Strategy Switch",
             color="#4CAF50", linewidth=0.8)
    ax4.axhline(0, color="gray", linewidth=0.5, linestyle=":")
    ax4.set_title("Rolling 1-Year Sharpe Ratio")
    ax4.set_ylabel("Sharpe")
    ax4.legend(fontsize=9)
    ax4.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

    # 5) Yearly returns comparison
    ax5 = fig.add_subplot(4, 2, 5)
    x = np.arange(len(years))
    w = 0.35
    ax5.bar(x - w/2, [bh_yearly[y]*100 for y in years], w,
            label="Buy & Hold", color="#2196F3")
    ax5.bar(x + w/2, [sw_yearly[y]*100 for y in years], w,
            label="Switch", color="#4CAF50")
    ax5.set_xticks(x)
    ax5.set_xticklabels(years, rotation=45, fontsize=8)
    ax5.set_ylabel("Return (%)")
    ax5.set_title("Annual Returns")
    ax5.legend(fontsize=9)
    ax5.axhline(0, color="gray", linewidth=0.5)

    # 6) Cumulative outperformance
    ax6 = fig.add_subplot(4, 2, 6)
    cum_out = (sw_eq / bh_eq)
    ax6.plot(cum_out.index, cum_out.values, color="#FF9800", linewidth=1.5)
    ax6.axhline(1.0, color="gray", linewidth=0.5, linestyle=":")
    ax6.fill_between(cum_out.index, 1, cum_out.values,
                     where=cum_out.values >= 1, alpha=0.2, color="#4CAF50")
    ax6.fill_between(cum_out.index, 1, cum_out.values,
                     where=cum_out.values < 1, alpha=0.2, color="#F44336")
    ax6.set_title("Relative Performance (Switch / B&H)")
    ax6.set_ylabel("Ratio")
    ax6.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

    # 7) Strategy allocation stacked area
    ax7 = fig.add_subplot(4, 2, 7)
    # Resample monthly for clarity
    monthly_alloc = active_strat.resample("ME").agg(lambda x: x.mode()[0] if len(x) > 0 else "Cash")
    unique_strats = [s for s in STRATEGY_DEFS.keys() if s in active_strat.values] + ["Cash"]
    colors_map = {"RSI": "#4CAF50", "Momentum": "#2196F3", "MA": "#FF9800",
                  "Double_MA": "#9C27B0", "Dual_Momentum": "#F44336",
                  "MACD": "#00BCD4", "Cash": "#BDBDBD"}
    # Build binary monthly arrays
    for sn in unique_strats:
        mask = (monthly_alloc == sn).astype(float)
        bottom = pd.Series(0.0, index=monthly_alloc.index)
        for prev in unique_strats[:unique_strats.index(sn)]:
            bottom += (monthly_alloc == prev).astype(float)
        ax7.fill_between(mask.index, bottom, bottom + mask,
                         label=sn, color=colors_map.get(sn, "#888"),
                         alpha=0.7, step="post")
    ax7.set_title("Active Strategy Over Time")
    ax7.set_ylabel("Allocation")
    ax7.set_ylim(0, 1.1)
    ax7.legend(fontsize=7, ncol=3, loc="upper left")
    ax7.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

    # 8) Monthly returns heatmap
    ax8 = fig.add_subplot(4, 2, 8)
    sw_monthly = hard_ret.resample("ME").sum()
    pivot = pd.DataFrame({
        "Year": sw_monthly.index.year,
        "Month": sw_monthly.index.month,
        "Return": sw_monthly.values * 100,
    })
    heatmap_data = pivot.pivot_table(index="Year", columns="Month",
                                      values="Return", aggfunc="mean")
    heatmap_data.columns = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                            "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"][:len(heatmap_data.columns)]
    sns.heatmap(heatmap_data, annot=True, fmt=".1f", cmap="RdYlGn",
                center=0, ax=ax8, cbar_kws={"label": "Return %"})
    ax8.set_title("Monthly Returns Heatmap (Strategy Switch)")

    fig.tight_layout(rect=[0, 0, 1, 0.96])
    path = os.path.join(out, "spy_performance_dashboard.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"\n  Dashboard: {path}")

    # ── Summary ending value ──────────────────────────────────────────────────
    print(f"\n  $10.000 investiert am {start.date()}:")
    print(f"    Buy & Hold SPY: ${10000 * bh_eq.iloc[-1]:>12,.0f}")
    print(f"    Strategy Switch: ${10000 * sw_eq.iloc[-1]:>12,.0f}")
    print(f"    Mehrergebnis:    ${10000 * (sw_eq.iloc[-1] - bh_eq.iloc[-1]):>12,.0f}")

    # ── Usage / Anleitung ─────────────────────────────────────────────────────
    print(f"""
  ════════════════════════════════════════════════════════════════════════
  ANLEITUNG: SO WENDEST DU DAS SYSTEM AN
  ════════════════════════════════════════════════════════════════════════

  ► KONZEPT (Nur Long oder Cash, KEINE Shorts):
    Das System nutzt 5 Strategien (RSI, Momentum, MA, Double-MA,
    Dual-Momentum). Jede erzeugt täglich ein Signal:
      1 = SPY kaufen/halten (Long)
      0 = SPY verkaufen / in Cash gehen

    Alle 63 Tage (~3 Monate) wird per Walk-Forward-Optimierung die
    beste Strategie gewählt (höchster Rolling-Sharpe der letzten 63 Tage).
    Wenn KEINE Strategie positiven Sharpe hat → Cash.

  ► KOSTEN PRO TRADE:
    • Kaufen: 0.15% (0.10% Transaktionskosten + 0.05% Slippage)
    • Verkaufen: 0.15%
    • Round-Trip: 0.30%
    → In {len(bench_ret)/252:.0f} Jahren gab es {len(real_trades)} echte Trades
      ({n_buys} Käufe + {n_sells} Verkäufe)
      = ca. {len(real_trades)/(len(bench_ret)/252):.0f} Trades/Jahr
      = Gesamtkosten ca. {total_cost_pct:.2%} über die gesamte Laufzeit
    ⚠ Strategie→Strategie-Wechsel (z.B. RSI→Momentum) kosten NICHTS
      weil du in beiden Fällen einfach SPY hältst!

  ► PRAKTISCHE UMSETZUNG:
    1. Investiere in SPY (z.B. über IBKR, Trade Republic, etc.)
    2. Check alle 1-3 Monate welche Strategie aktuell aktiv ist
    3. Aktuelle Position laut System: {prev} (Stand: {end.date()})
    4. Wenn das Signal „Cash" zeigt → SPY verkaufen
       Wenn das Signal eine Strategie zeigt → SPY kaufen/halten
    5. Trade-Log steht in: {trade_csv}

  ► AKTUELLES SIGNAL:
    Aktive Strategie am {end.date()}: ** {active_strat.iloc[-1]} **
    {"→ SPY HALTEN (Long)" if active_strat.iloc[-1] != "Cash" else "→ CASH (nicht investiert)"}

  ► DATEIEN:
    Dashboard PNG:  {os.path.join(out, 'spy_performance_dashboard.png')}
    Trade-Log CSV:  {trade_csv}

  ► SKRIPT NEU AUSFÜHREN:
    cd "{os.path.dirname(os.path.dirname(os.path.abspath(__file__)))}"
    python -m wf_backtest.spy_dashboard

  ════════════════════════════════════════════════════════════════════════
""")


if __name__ == "__main__":
    main()
