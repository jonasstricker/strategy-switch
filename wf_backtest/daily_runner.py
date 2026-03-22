#!/usr/bin/env python3
"""
Daily Runner — Multi-ETF Strategy Switch
==========================================
Wird täglich per Windows Task Scheduler ausgeführt.

Analysiert 4 ETFs: SPY, URTH, EEM, VGK
Sendet E-Mail NUR wenn mind. 1 ETF sein Signal ändert (Cash↔Long).

State-Datei: wf_backtest/state.json
Log-Datei:   wf_backtest/daily_runner.log
"""

from __future__ import annotations
import sys, os, json, logging
from datetime import datetime
from pathlib import Path

# ── Fix imports ──────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

import warnings
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
from wf_backtest.notifier import send_email, build_multi_etf_email

# ── Paths ────────────────────────────────────────────────────────────────────
PKG_DIR = Path(__file__).resolve().parent
STATE_FILE = PKG_DIR / "state.json"
MOBILE_SIGNALS_FILE = PKG_DIR / "mobile_signals.json"
LOG_FILE = PKG_DIR / "daily_runner.log"

# ── Logging ──────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(LOG_FILE, encoding="utf-8"),
        logging.StreamHandler(sys.stdout),
    ],
)
log = logging.getLogger("daily_runner")

# ── Constants ────────────────────────────────────────────────────────────────
RF = 0.02
# Trade Republic Kosten: 1€ Fremdkostenpauschale pro Order
# Bei €10.000 Portfolio → 0.01% pro Trade; Spread ca. 0.02% bei liquiden ETFs
TX = 0.0001    # 1 bp (1€ auf 10k)
SLIP = 0.0002  # 2 bp Spread (liquide EU-ETFs)

# Lite-Modus: kleinere Grids für ressourcenbegrenzte Umgebungen (z.B. Render Free)
# Oracle Cloud (24GB RAM, 4 Kerne) braucht das NICHT
IS_CLOUD = bool(os.environ.get("LITE_MODE"))

ETF_DEFS = {
    "SXR8.DE":  {"name": "S&P100_EU", "start": "2010-06-01"},
    "URTH": {"name": "MSCI World",        "start": "2012-01-01"},
    "EEM":  {"name": "Emerging Markets",  "start": "2004-01-01"},
    "VGK":  {"name": "FTSE Europe",       "start": "2005-01-01"},
}

if IS_CLOUD:
    STRATEGY_DEFS = {
        "RSI": {
            "grid": [{"period": per, "threshold": thr}
                     for per in [14, 20]
                     for thr in [40, 50]],
            "gen": lambda p, params: rsi_signal(p, params["period"], params["threshold"]),
            "keys": ["period", "threshold"],
        },
        "Momentum": {
            "grid": [{"lookback": lb} for lb in [90, 200]],
            "gen": lambda p, params: momentum_signal(p, params["lookback"]),
            "keys": ["lookback"],
        },
        "MA": {
            "grid": [{"period": p} for p in [50, 200]],
            "gen": lambda p, params: ma_signal(p, params["period"]),
            "keys": ["period"],
        },
        "Double_MA": {
            "grid": [{"fast": 20, "slow": 200}],
            "gen": lambda p, params: double_ma_signal(p, params["fast"], params["slow"]),
            "keys": ["fast", "slow"],
        },
    }
    WF_CFG = {"train": 504, "test": 21, "step": 42}
else:
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


# ── Helper functions ─────────────────────────────────────────────────────────

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


def compute_etf(ticker: str, start_date: str) -> dict:
    """Run full WF analysis for a single ETF. Returns data dict."""
    log.info(f"  Lade {ticker}-Daten …")
    raw = yf.download(ticker, start=start_date, auto_adjust=True, progress=False)
    if isinstance(raw.columns, pd.MultiIndex):
        close_col = [c for c in raw.columns if c[0] == "Close"]
        close = raw[close_col[0]].dropna()
    else:
        close = raw["Close"].dropna()

    returns = close.pct_change().dropna()
    prices = close.loc[returns.index]

    # Run per-strategy WF
    strat_oos = {}
    for sname, sdef in STRATEGY_DEFS.items():
        log.info(f"    WF: {sname} …")
        r = wf_single(prices, returns, sdef)
        if r is not None:
            strat_oos[sname] = r

    if not strat_oos:
        raise RuntimeError(f"{ticker}: Keine Strategie hat genug OOS-Daten.")

    # Align
    start = max(s.first_valid_index() for s in strat_oos.values())
    end = min(s.last_valid_index() for s in strat_oos.values())
    aligned = {n: r.loc[start:end] for n, r in strat_oos.items()}
    bench_ret = returns.loc[start:end]
    df_strats = pd.DataFrame(aligned)
    names = df_strats.columns.tolist()

    # Rolling Sharpe & switching (mit Meta-Kosten + Hysterese)
    sw = 63
    roll = pd.DataFrame({n: rolling_sharpe(df_strats[n], sw, RF) for n in names})

    hard_ret, active_strat = apply_switching(
        df_strats, roll, tx=TX, slip=SLIP, min_hold=5,
    )

    # Trade log
    trades = []
    prev = "Cash"
    buy_price = None
    for idx in active_strat.index:
        cur = active_strat.loc[idx]
        if cur != prev:
            was_cash = (prev == "Cash")
            now_cash = (cur == "Cash")
            if was_cash and not now_cash:
                buy_price = float(close.loc[idx])
                trades.append({
                    "Datum": idx.strftime("%d.%m.%Y"),
                    "Aktion": f"KAUF {ticker}",
                    "Grund": f"Strategie {cur} aktiv",
                    "Kurs": f"${buy_price:,.2f}",
                    "Rendite": None,
                })
            elif not was_cash and now_cash:
                sell_price = float(close.loc[idx])
                pnl = round((sell_price / buy_price - 1) * 100, 1) if buy_price and buy_price > 0 else None
                trades.append({
                    "Datum": idx.strftime("%d.%m.%Y"),
                    "Aktion": f"VERKAUF {ticker}",
                    "Grund": "Keine Strategie positiv → Cash",
                    "Kurs": f"${sell_price:,.2f}",
                    "Rendite": pnl,
                })
                buy_price = None
            prev = cur

    # Current state
    current_strat = active_strat.iloc[-1]
    is_invested = current_strat != "Cash"
    price = float(close.iloc[-1])

    # Performance Metrics
    bh_eq = _equity(bench_ret)
    sw_eq = _equity(hard_ret)
    tuw_sw = time_under_water(sw_eq)
    tuw_bh = time_under_water(bh_eq)

    perf = {
        "sw_cagr": cagr(sw_eq),
        "sw_sharpe": sharpe_ratio(hard_ret, RF),
        "sw_sortino": sortino_ratio(hard_ret, RF),
        "sw_calmar": calmar_ratio(sw_eq),
        "sw_dd": max_drawdown(sw_eq),
        "sw_vol": float(hard_ret.std() * np.sqrt(252)),
        "sw_max_uw": tuw_sw["max_days"],
        "sw_best_year": float(hard_ret.groupby(hard_ret.index.year).sum().max()),
        "sw_worst_year": float(hard_ret.groupby(hard_ret.index.year).sum().min()),
        "sw_win_rate": float((hard_ret > 0).mean()),
        "bh_cagr": cagr(bh_eq),
        "bh_sharpe": sharpe_ratio(bench_ret, RF),
        "bh_sortino": sortino_ratio(bench_ret, RF),
        "bh_calmar": calmar_ratio(bh_eq),
        "bh_dd": max_drawdown(bh_eq),
        "bh_vol": float(bench_ret.std() * np.sqrt(252)),
        "bh_max_uw": tuw_bh["max_days"],
        "bh_best_year": float(bench_ret.groupby(bench_ret.index.year).sum().max()),
        "bh_worst_year": float(bench_ret.groupby(bench_ret.index.year).sum().min()),
        "bh_win_rate": float((bench_ret > 0).mean()),
    }

    # Strategy Check: Sharpe over time windows
    strat_sharpes = {}
    for n in names:
        if n in roll.columns and len(roll[n].dropna()) > 0:
            strat_sharpes[n] = float(roll[n].dropna().iloc[-1])

    strat_check = {}
    windows_def = [("3 Monate", 63), ("6 Monate", 126), ("1 Jahr", 252), ("Gesamt", len(bench_ret))]
    for wname, wdays in windows_def:
        d = min(wdays, len(bench_ret))
        strat_check[wname] = {}
        for n in names:
            if n in df_strats.columns:
                sr = df_strats[n].iloc[-d:]
                strat_check[wname][n] = sharpe_ratio(sr, RF) if len(sr) > 20 else 0.0
        strat_check[wname]["_switch"] = sharpe_ratio(hard_ret.iloc[-d:], RF) if d > 20 else 0.0
        strat_check[wname]["_bh"] = sharpe_ratio(bench_ret.iloc[-d:], RF) if d > 20 else 0.0

    return {
        "current_strat": current_strat,
        "is_invested": is_invested,
        "price": price,
        "perf": perf,
        "strat_sharpes": strat_sharpes,
        "strat_check": strat_check,
        "trades": trades,
        # Raw series for mobile 1Y data
        "_hard_ret": hard_ret,
        "_bench_ret": bench_ret,
        "_active_strat": active_strat,
        "_roll_sharpe": roll,       # Rolling Sharpe DF für Margin-Anzeige
    }


# ── State management ─────────────────────────────────────────────────────────

def load_state() -> dict | None:
    if STATE_FILE.exists():
        with open(STATE_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return None


def save_state(state: dict):
    with open(STATE_FILE, "w", encoding="utf-8") as f:
        json.dump(state, f, indent=2, ensure_ascii=False)


def _build_category_json(result: dict, category: str,
                         names_map: dict | None = None) -> dict:
    """
    Baut JSON-Daten für eine Aktien-Kategorie (Swarm, Value, Turnaround).
    Arbeitet mit dem neuen per-stock result-Format.
    """
    from wf_backtest.swarm_wf import MEGA_CAP_UNIVERSE
    if names_map is None:
        names_map = {}

    stock_results = result["stock_results"]
    sw_eq = result["sw_eq"]
    bh_eq = result["bh_eq"]

    # Per-stock data for individual signals
    stocks = []
    stocks_1y = {}
    for t, sr in stock_results.items():
        name = names_map.get(t) or sr.get("name") or MEGA_CAP_UNIVERSE.get(t, t)
        stocks.append({
            "ticker": t,
            "name": name,
            "signal": sr["signal"],
            "strategy": sr["strategy"],
            "signal_margin": sr["signal_margin"],
            "sharpe": round(sr["sw_sharpe"], 2),
            "bh_sharpe": round(sr["bh_sharpe"], 2),
            "pct_invested": round(sr["pct_invested"], 3),
            "n_trades": sr["n_trades"],
            # Fundamental data (Value/Turnaround only)
            "pe": sr.get("pe"),
            "pb": sr.get("pb"),
            "div_yield": round(sr["div_yield"] * 100, 2) if sr.get("div_yield") else None,
            "pct_from_high": round(sr["pct_from_52w_high"] * 100, 1) if sr.get("pct_from_52w_high") else None,
        })
        # 1Y normalized chart per stock
        if "close" in sr and sr["close"] is not None:
            s = sr["close"]
        elif "close_df" in result and t in result["close_df"].columns:
            s = result["close_df"][t].dropna()
        else:
            s = None
        if s is not None and len(s) > 60:
            n_1y = min(252, len(s))
            s_1y = s.iloc[-n_1y:]
            norm = s_1y / s_1y.iloc[0]
            step_st = max(1, len(norm) // 60)
            pts = []
            for ii in range(0, len(norm), step_st):
                pts.append({
                    "date": norm.index[ii].strftime("%Y-%m-%d"),
                    "value": round(float(norm.iloc[ii]), 4),
                })
            if pts[-1]["date"] != norm.index[-1].strftime("%Y-%m-%d"):
                pts.append({
                    "date": norm.index[-1].strftime("%Y-%m-%d"),
                    "value": round(float(norm.iloc[-1]), 4),
                })
            stocks_1y[t] = pts

    # Aggregated equity (downsampled ~200 points)
    step_s = max(1, len(sw_eq) // 200)
    equity = []
    for i in range(0, len(sw_eq), step_s):
        idx = sw_eq.index[i]
        equity.append({
            "date": idx.strftime("%Y-%m-%d"),
            "switch": round(float(sw_eq.iloc[i]), 4),
            "bh": round(float(bh_eq.iloc[i]), 4),
        })
    if equity and equity[-1]["date"] != sw_eq.index[-1].strftime("%Y-%m-%d"):
        equity.append({
            "date": sw_eq.index[-1].strftime("%Y-%m-%d"),
            "switch": round(float(sw_eq.iloc[-1]), 4),
            "bh": round(float(bh_eq.iloc[-1]), 4),
        })

    # Monthly heatmap (Switch + B&H)
    sw_ret = result["switch_ret"]
    bh_ret = result["bench_ret"]
    monthly_sw = sw_ret.resample("ME").sum() * 100
    monthly_bh = bh_ret.resample("ME").sum() * 100
    months_switch = {}
    for idx_m, val in monthly_sw.items():
        yr, mo = int(idx_m.year), int(idx_m.month)
        if yr not in months_switch:
            months_switch[yr] = {}
        months_switch[yr][mo] = round(float(val), 1)
    months_bh = {}
    for idx_m, val in monthly_bh.items():
        yr, mo = int(idx_m.year), int(idx_m.month)
        if yr not in months_bh:
            months_bh[yr] = {}
        months_bh[yr][mo] = round(float(val), 1)

    # Yearly returns
    yr_sw = sw_ret.groupby(sw_ret.index.year).sum() * 100
    yr_bh = bh_ret.groupby(bh_ret.index.year).sum() * 100
    yearly = []
    for yr in sorted(set(yr_sw.index) | set(yr_bh.index)):
        yearly.append({
            "year": int(yr),
            "switch": round(float(yr_sw.get(yr, 0)), 1),
            "bh": round(float(yr_bh.get(yr, 0)), 1),
        })

    # Trade history per stock
    trades = []
    for t, sr in stock_results.items():
        active = sr.get("active_strat")
        close_s = sr.get("close")
        if active is None or close_s is None:
            continue
        prev = "Cash"
        buy_price = None
        for idx_t in active.index:
            cur = active.loc[idx_t]
            if cur != prev:
                price_val = float(close_s.loc[idx_t]) if idx_t in close_s.index else 0
                if prev == "Cash" and cur != "Cash":
                    buy_price = price_val
                    trades.append({
                        "Datum": idx_t.strftime("%d.%m.%Y"),
                        "Aktion": f"KAUF {t}",
                        "Kurs": f"${price_val:,.2f}",
                        "Rendite": None,
                    })
                elif prev != "Cash" and cur == "Cash":
                    pnl = None
                    if buy_price and buy_price > 0:
                        pnl = round((price_val / buy_price - 1) * 100, 1)
                    trades.append({
                        "Datum": idx_t.strftime("%d.%m.%Y"),
                        "Aktion": f"VERKAUF {t}",
                        "Kurs": f"${price_val:,.2f}",
                        "Rendite": pnl,
                    })
                    buy_price = None
                prev = cur
    # Sort by date, keep last 40
    trades.sort(key=lambda x: x["Datum"].split(".")[::-1])
    trades = trades[-40:]

    # Detail metrics
    sw_sortino = round(sortino_ratio(sw_ret, RF), 2)
    bh_sortino = round(sortino_ratio(bh_ret, RF), 2)
    sw_calmar = round(calmar_ratio(sw_eq), 2)
    bh_calmar = round(calmar_ratio(bh_eq), 2)
    sw_vol = round(float(sw_ret.std() * np.sqrt(252)), 4)
    bh_vol = round(float(bh_ret.std() * np.sqrt(252)), 4)
    sw_uw = time_under_water(sw_eq)
    bh_uw = time_under_water(bh_eq)

    n_long = sum(1 for s in stocks if s["signal"] == "LONG")

    return {
        "category": category,
        "stocks": stocks,
        "n_long": n_long,
        "n_total": len(stocks),
        "sharpe_switch": round(result["sw_sharpe"], 2),
        "sharpe_bh": round(result["bh_sharpe"], 2),
        "cagr_switch": round(float(cagr(sw_eq)), 4),
        "cagr_bh": round(float(cagr(bh_eq)), 4),
        "max_dd_switch": round(float(max_drawdown(sw_eq)), 4),
        "max_dd_bh": round(float(max_drawdown(bh_eq)), 4),
        "sortino_switch": sw_sortino,
        "sortino_bh": bh_sortino,
        "calmar_switch": sw_calmar,
        "calmar_bh": bh_calmar,
        "vol_switch": sw_vol,
        "vol_bh": bh_vol,
        "max_uw_switch": sw_uw.get("max_days", 0),
        "max_uw_bh": bh_uw.get("max_days", 0),
        "pct_invested": round(result["pct_invested"], 4),
        "n_trades": result["n_trades"],
        "equity": equity,
        "stocks_1y": stocks_1y,
        "months_switch": months_switch,
        "months_bh": months_bh,
        "yearly_returns": yearly,
        "trades": trades,
        "start": result["start"].strftime("%Y-%m-%d") if hasattr(result["start"], "strftime") else str(result["start"]),
        "end": result["end"].strftime("%Y-%m-%d") if hasattr(result["end"], "strftime") else str(result["end"]),
        # Historische Rebalance-Verschiebungen (nur Swarm)
        "recent_rebalances": result.get("recent_rebalances"),
    }


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    log.info("=" * 60)
    log.info("Multi-ETF Daily Runner gestartet")
    log.info("=" * 60)

    try:
        # 1. Compute all ETFs
        all_data = {}
        for ticker, info in ETF_DEFS.items():
            log.info(f"Berechne {ticker} ({info['name']}) …")
            try:
                all_data[ticker] = compute_etf(ticker, info["start"])
                d = all_data[ticker]
                log.info(f"  → {'LONG' if d['is_invested'] else 'CASH'} "
                         f"({d['current_strat']}, ${d['price']:,.2f}, "
                         f"Sharpe: {d['perf']['sw_sharpe']:.2f})")
            except Exception as e:
                log.error(f"  ⚠ FEHLER bei {ticker}: {e}")

        if not all_data:
            raise RuntimeError("Kein einziger ETF konnte berechnet werden.")

        # 2. Load previous state
        prev_state = load_state()

        # 3. Determine signal changes
        signal_changes = {}
        send_mail = False

        for ticker, d in all_data.items():
            if prev_state is None:
                signal_changes[ticker] = "INITIAL"
                send_mail = True
            else:
                prev_invested = prev_state.get("etfs", {}).get(ticker, {}).get("is_invested")
                now_invested = d["is_invested"]

                if prev_invested is None:
                    signal_changes[ticker] = "INITIAL"
                    send_mail = True
                elif prev_invested != now_invested:
                    signal_changes[ticker] = "KAUF" if now_invested else "VERKAUF"
                    send_mail = True
                    log.info(f"  Signal-Wechsel {ticker}: "
                             f"{'Cash → LONG' if now_invested else 'LONG → Cash'}")
                else:
                    signal_changes[ticker] = None

        if not send_mail:
            log.info("Kein Signal-Wechsel bei keinem ETF. Keine E-Mail nötig.")
        else:
            changes_str = ", ".join(f"{t}={s}" for t, s in signal_changes.items() if s)
            log.info(f"Änderungen: {changes_str}")

        # 4. Send email if needed
        if send_mail and not os.environ.get("SKIP_EMAIL"):
            try:
                subject, html = build_multi_etf_email(all_data, signal_changes)
                success = send_email(subject, html)
                if success:
                    log.info(f"✅ E-Mail gesendet: {subject}")
                else:
                    log.error("❌ E-Mail-Versand fehlgeschlagen!")
            except Exception as e:
                log.warning(f"E-Mail übersprungen: {e}")
        elif send_mail:
            log.info("E-Mail übersprungen (SKIP_EMAIL gesetzt)")

        # 5. Save new state
        new_state = {
            "date": datetime.now().isoformat(),
            "etfs": {},
        }
        for ticker, d in all_data.items():
            new_state["etfs"][ticker] = {
                "is_invested": d["is_invested"],
                "current_strat": d["current_strat"],
                "price": d["price"],
                "sw_sharpe": d["perf"]["sw_sharpe"],
            }

        save_state(new_state)
        log.info(f"State gespeichert: {STATE_FILE}")

        # 6. Save mobile signals JSON (für iPhone-App)
        mobile_data = {
            "updated": datetime.now().strftime("%d.%m.%Y %H:%M"),
            "etfs": [],
            "swarm": None,
            "value": None,
            "turnaround": None,
        }
        for ticker, d in all_data.items():
            p = d["perf"]
            hard_ret = d["_hard_ret"]
            bench_ret = d["_bench_ret"]
            active_strat = d["_active_strat"]

            # ── 1Y Equity-Daten berechnen ──
            n_1y = min(252, len(hard_ret))
            hr_1y = hard_ret.iloc[-n_1y:]
            br_1y = bench_ret.iloc[-n_1y:]
            as_1y = active_strat.iloc[-n_1y:]

            sw_eq_1y = (1 + hr_1y).cumprod()
            bh_eq_1y = (1 + br_1y).cumprod()

            return_1y_switch = float(sw_eq_1y.iloc[-1] / sw_eq_1y.iloc[0] - 1) if len(sw_eq_1y) > 1 else 0.0
            return_1y_bh = float(bh_eq_1y.iloc[-1] / bh_eq_1y.iloc[0] - 1) if len(bh_eq_1y) > 1 else 0.0
            pct_invested_1y = float((as_1y != "Cash").mean())

            # Equity array for chart (every Nth point to keep JSON small)
            step = max(1, len(sw_eq_1y) // 120)
            equity_1y = []
            for i in range(0, len(sw_eq_1y), step):
                idx = sw_eq_1y.index[i]
                equity_1y.append({
                    "date": idx.strftime("%Y-%m-%d"),
                    "switch": round(float(sw_eq_1y.iloc[i]), 4),
                    "bh": round(float(bh_eq_1y.iloc[i]), 4),
                    "cash": bool(as_1y.iloc[i] == "Cash"),
                })
            if equity_1y and equity_1y[-1]["date"] != sw_eq_1y.index[-1].strftime("%Y-%m-%d"):
                equity_1y.append({
                    "date": sw_eq_1y.index[-1].strftime("%Y-%m-%d"),
                    "switch": round(float(sw_eq_1y.iloc[-1]), 4),
                    "bh": round(float(bh_eq_1y.iloc[-1]), 4),
                    "cash": bool(as_1y.iloc[-1] == "Cash"),
                })

            # ── Full equity history (downsampled ~250 points) ──
            sw_eq_full = _equity(hard_ret)
            bh_eq_full = _equity(bench_ret)
            step_full = max(1, len(sw_eq_full) // 250)
            equity_full = []
            for i in range(0, len(sw_eq_full), step_full):
                idx = sw_eq_full.index[i]
                equity_full.append({
                    "date": idx.strftime("%Y-%m-%d"),
                    "switch": round(float(sw_eq_full.iloc[i]), 4),
                    "bh": round(float(bh_eq_full.iloc[i]), 4),
                })
            if equity_full and equity_full[-1]["date"] != sw_eq_full.index[-1].strftime("%Y-%m-%d"):
                equity_full.append({
                    "date": sw_eq_full.index[-1].strftime("%Y-%m-%d"),
                    "switch": round(float(sw_eq_full.iloc[-1]), 4),
                    "bh": round(float(bh_eq_full.iloc[-1]), 4),
                })

            # ── Monthly returns heatmap (Switch + B&H) ──
            sw_monthly = hard_ret.resample("ME").sum() * 100
            bh_monthly = bench_ret.resample("ME").sum() * 100
            months_sw = {}  # {year: {month: value}}
            months_bh = {}
            for idx_m, val in sw_monthly.items():
                yr, mo = int(idx_m.year), int(idx_m.month)
                if yr not in months_sw:
                    months_sw[yr] = {}
                months_sw[yr][mo] = round(float(val), 1)
            for idx_m, val in bh_monthly.items():
                yr, mo = int(idx_m.year), int(idx_m.month)
                if yr not in months_bh:
                    months_bh[yr] = {}
                months_bh[yr][mo] = round(float(val), 1)

            # ── Yearly returns ──
            yr_sw = hard_ret.groupby(hard_ret.index.year).sum() * 100
            yr_bh = bench_ret.groupby(bench_ret.index.year).sum() * 100
            yearly_returns = []
            for yr in sorted(set(yr_sw.index) | set(yr_bh.index)):
                yearly_returns.append({
                    "year": int(yr),
                    "switch": round(float(yr_sw.get(yr, 0)), 1),
                    "bh": round(float(yr_bh.get(yr, 0)), 1),
                })

            # ── Trade log (last 20) ──
            trade_log = d["trades"][-20:] if d["trades"] else []

            last_date = hard_ret.index[-1].strftime("%d.%m.%Y") if len(hard_ret) > 0 else ""
            pct_invested_total = float((active_strat != "Cash").mean())

            # ── Detailed metrics ──
            tuw_sw = time_under_water(_equity(hard_ret))
            tuw_bh = time_under_water(_equity(bench_ret))

            # ── Signal-Marge: bester Rolling Sharpe (Abstand zur Schwelle 0) ──
            roll_sharpe = d["_roll_sharpe"]
            if len(roll_sharpe) > 0:
                last_sharpes = roll_sharpe.iloc[-1].dropna()
                best_sharpe = float(last_sharpes.max()) if len(last_sharpes) > 0 else 0.0
                strat_margins = {
                    str(n): round(float(v), 3)
                    for n, v in last_sharpes.items()
                }
            else:
                best_sharpe = 0.0
                strat_margins = {}

            etf_mobile = {
                "ticker": ticker,
                "name": ETF_DEFS[ticker]["name"],
                "signal": "LONG" if d["is_invested"] else "CASH",
                "strategy": d["current_strat"],
                "price": round(d["price"], 2),
                "last_date": last_date,
                # Core metrics
                "sharpe_switch": round(p["sw_sharpe"], 2),
                "sharpe_bh": round(p["bh_sharpe"], 2),
                "cagr_switch": round(p["sw_cagr"], 4),
                "cagr_bh": round(p["bh_cagr"], 4),
                "max_dd_switch": round(p["sw_dd"], 4),
                "max_dd_bh": round(p["bh_dd"], 4),
                "vol_switch": round(p["sw_vol"], 4),
                "vol_bh": round(p["bh_vol"], 4),
                "pct_invested": round(pct_invested_total, 4),
                # Extended metrics
                "sortino_switch": round(p["sw_sortino"], 2),
                "sortino_bh": round(p["bh_sortino"], 2),
                "calmar_switch": round(p["sw_calmar"], 2),
                "calmar_bh": round(p["bh_calmar"], 2),
                "max_uw_switch": int(tuw_sw["max_days"]),
                "max_uw_bh": int(tuw_bh["max_days"]),
                "best_year_switch": round(p["sw_best_year"] * 100, 1),
                "best_year_bh": round(p["bh_best_year"] * 100, 1),
                "worst_year_switch": round(p["sw_worst_year"] * 100, 1),
                "worst_year_bh": round(p["bh_worst_year"] * 100, 1),
                "win_rate_switch": round(p["sw_win_rate"] * 100, 1),
                "win_rate_bh": round(p["bh_win_rate"] * 100, 1),
                # 1Y data
                "return_1y_switch": round(return_1y_switch, 4),
                "return_1y_bh": round(return_1y_bh, 4),
                "pct_invested_1y": round(pct_invested_1y, 4),
                "equity_1y": equity_1y,
                # Full history
                "equity_full": equity_full,
                # Monthly matrix
                "months_switch": months_sw,
                "months_bh": months_bh,
                # Yearly returns
                "yearly_returns": yearly_returns,
                # Trade log
                "trades": trade_log,
                # Signal margin (Abstand zur Schwelle)
                "signal_margin": round(best_sharpe, 3),
                "strat_margins": strat_margins,
            }
            mobile_data["etfs"].append(etf_mobile)

        # ── 7. Swarm data (Aktien-Schwarm WF — Einzelaktien-Signale) ──
        try:
            log.info("Berechne Aktien-Schwarm (WF) …")
            from wf_backtest.swarm_wf import run_swarm_wf, MEGA_CAP_UNIVERSE
            swarm = run_swarm_wf(top_n=5 if IS_CLOUD else 10,
                                 cloud_mode=IS_CLOUD)
            if swarm is not None:
                mobile_data["swarm"] = _build_category_json(
                    swarm, "swarm", MEGA_CAP_UNIVERSE,
                )
                n_long = sum(1 for sr in swarm["stock_results"].values() if sr["signal"] == "LONG")
                log.info(f"  Swarm: {n_long}/{len(swarm['stock_results'])} LONG "
                         f"(Sharpe {swarm['sw_sharpe']:.2f})")
        except Exception as e:
            log.warning(f"  Swarm-Berechnung übersprungen: {e}")

        # ── 8. Value-Aktien ──
        try:
            log.info("Berechne Value-Aktien …")
            from wf_backtest.stock_screener import run_category_wf
            value = run_category_wf("value", cloud_mode=IS_CLOUD)
            if value is not None:
                mobile_data["value"] = _build_category_json(value, "value")
                n_long = sum(1 for sr in value["stock_results"].values() if sr["signal"] == "LONG")
                log.info(f"  Value: {n_long}/{len(value['stock_results'])} LONG "
                         f"(Sharpe {value['sw_sharpe']:.2f})")
        except Exception as e:
            log.warning(f"  Value-Berechnung übersprungen: {e}")

        # ── 9. Turnaround-Aktien ──
        try:
            log.info("Berechne Turnaround-Aktien …")
            from wf_backtest.stock_screener import run_category_wf as run_cat
            turnaround = run_cat("turnaround", cloud_mode=IS_CLOUD)
            if turnaround is not None:
                mobile_data["turnaround"] = _build_category_json(turnaround, "turnaround")
                n_long = sum(1 for sr in turnaround["stock_results"].values() if sr["signal"] == "LONG")
                log.info(f"  Turnaround: {n_long}/{len(turnaround['stock_results'])} LONG "
                         f"(Sharpe {turnaround['sw_sharpe']:.2f})")
        except Exception as e:
            log.warning(f"  Turnaround-Berechnung übersprungen: {e}")

        with open(MOBILE_SIGNALS_FILE, "w", encoding="utf-8") as f:
            json.dump(mobile_data, f, indent=2, ensure_ascii=False)
        log.info(f"Mobile-JSON gespeichert: {MOBILE_SIGNALS_FILE}")

    except Exception as e:
        log.exception(f"FEHLER: {e}")
        try:
            from wf_backtest.notifier import send_email as _err_send
            _err_send(
                f"❌ Multi-ETF Runner Fehler {datetime.now().strftime('%d.%m.%Y')}",
                f"<html><body><h2>Fehler im Daily Runner</h2><pre>{e}</pre></body></html>",
            )
        except Exception:
            pass

    log.info("Daily Runner beendet.\n")


if __name__ == "__main__":
    main()
