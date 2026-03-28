#!/usr/bin/env python3
"""
stock_screener.py — Value & Turnaround Aktien-Screening + WF-Switching
========================================================================

Zwei Kategorien:
  1. VALUE: Niedrige Bewertung (P/E, P/B), hohe Dividende, stabile Gewinne
  2. TURNAROUND: Stark gefallen, aber Erholung & Fundamentals deuten auf Wende

Jede Aktie bekommt eigenes Kauf/Verkauf-Signal via Walk-Forward Switching.
"""

from __future__ import annotations
import sys, os, warnings, logging
warnings.filterwarnings("ignore")

if __package__ is None:
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    __package__ = "wf_backtest"

import numpy as np
import pandas as pd
import yfinance as yf

from .strategies import momentum_signal, ma_signal, rsi_signal, apply_costs
from .strategies_ext import dual_momentum_signal, double_ma_signal
from .metrics import sharpe_ratio, rolling_sharpe
from .walk_forward import generate_windows
from .switching import apply_switching

log = logging.getLogger("stock_screener")

# ── Breites Screening-Universum (~100 liquide US/EU Aktien) ──────────────────
SCREENING_UNIVERSE = {
    # US Large / Mid Caps
    "AAPL": "Apple", "MSFT": "Microsoft", "GOOGL": "Alphabet",
    "AMZN": "Amazon", "NVDA": "NVIDIA", "META": "Meta",
    "BRK-B": "Berkshire Hathaway", "TSLA": "Tesla",
    "JPM": "JPMorgan", "V": "Visa", "UNH": "UnitedHealth",
    "JNJ": "Johnson & Johnson", "XOM": "ExxonMobil",
    "MA": "Mastercard", "PG": "Procter & Gamble",
    "HD": "Home Depot", "LLY": "Eli Lilly",
    "AVGO": "Broadcom", "ABBV": "AbbVie", "MRK": "Merck",
    "KO": "Coca-Cola", "PEP": "PepsiCo", "COST": "Costco",
    "CVX": "Chevron", "WMT": "Walmart", "BAC": "Bank of America",
    "CRM": "Salesforce", "ORCL": "Oracle", "ADBE": "Adobe",
    "MCD": "McDonald's", "NFLX": "Netflix", "AMD": "AMD",
    "TMO": "Thermo Fisher", "DIS": "Disney",
    "INTC": "Intel", "GS": "Goldman Sachs", "CAT": "Caterpillar",
    "GE": "GE Aerospace", "RTX": "RTX", "HON": "Honeywell",
    "NEE": "NextEra Energy", "LOW": "Lowe's",
    "TXN": "Texas Instruments", "QCOM": "Qualcomm",
    "NOW": "ServiceNow", "ISRG": "Intuitive Surgical",
    "BLK": "BlackRock", "LMT": "Lockheed Martin",
    "UBER": "Uber", "SBUX": "Starbucks",
    # More Value/Turnaround candidates
    "IBM": "IBM", "CSCO": "Cisco", "VZ": "Verizon",
    "T": "AT&T", "BMY": "Bristol-Myers", "GILD": "Gilead",
    "MO": "Altria", "PM": "Philip Morris", "SO": "Southern Company",
    "DUK": "Duke Energy", "MMM": "3M", "WBA": "Walgreens",
    "PFE": "Pfizer", "PYPL": "PayPal", "BA": "Boeing",
    "F": "Ford", "GM": "General Motors", "NKE": "Nike",
    "TMUS": "T-Mobile", "COP": "ConocoPhillips",
    "CI": "Cigna", "USB": "US Bancorp", "MS": "Morgan Stanley",
    "SCHW": "Schwab", "AXP": "American Express",
    "DE": "Deere", "CVS": "CVS Health", "MDT": "Medtronic",
    "SYK": "Stryker", "SLB": "Schlumberger",
    "FDX": "FedEx", "UPS": "UPS",
}

# ── Konstanten ───────────────────────────────────────────────────────────────
RF = 0.02
TX = 0.0000    # Alpaca: $0 Kommission
SLIP = 0.0005  # Spread Einzelaktien (konservativ)
TOP_N = 10

WF_CFG = {"train": 504, "test": 21, "step": 21}
ROLL_SHARPE_WINDOW = 42
MIN_HOLD = 5

STRATEGY_DEFS = {
    "RSI": {
        "grid": [{"period": per, "threshold": thr}
                 for per in [10, 14, 20]
                 for thr in [40, 50, 60]],
        "gen": lambda p, params: rsi_signal(p, params["period"], params["threshold"]),
        "keys": ["period", "threshold"],
    },
    "Momentum": {
        "grid": [{"lookback": lb} for lb in [30, 60, 90, 160]],
        "gen": lambda p, params: momentum_signal(p, params["lookback"]),
        "keys": ["lookback"],
    },
    "MA": {
        "grid": [{"period": p} for p in [50, 100, 150]],
        "gen": lambda p, params: ma_signal(p, params["period"]),
        "keys": ["period"],
    },
    "Double_MA": {
        "grid": [{"fast": f, "slow": s}
                 for f in [20, 50] for s in [100, 150, 200] if f < s],
        "gen": lambda p, params: double_ma_signal(p, params["fast"], params["slow"]),
        "keys": ["fast", "slow"],
    },
}

# Cloud-optimierte Versionen
CLOUD_WF_CFG = {"train": 252, "test": 21, "step": 42}
CLOUD_TOP_N = 5
CLOUD_STRATEGY_DEFS = {
    "RSI": {
        "grid": [{"period": per, "threshold": thr}
                 for per in [14, 20]
                 for thr in [40, 50]],
        "gen": lambda p, params: rsi_signal(p, params["period"], params["threshold"]),
        "keys": ["period", "threshold"],
    },
    "Momentum": {
        "grid": [{"lookback": lb} for lb in [60, 160]],
        "gen": lambda p, params: momentum_signal(p, params["lookback"]),
        "keys": ["lookback"],
    },
    "MA": {
        "grid": [{"period": p} for p in [50, 150]],
        "gen": lambda p, params: ma_signal(p, params["period"]),
        "keys": ["period"],
    },
}

# Reduziertes Universum für Cloud (nur die liquidesten → schnelleres Screening)
CLOUD_SCREENING_UNIVERSE = {
    "AAPL": "Apple", "MSFT": "Microsoft", "GOOGL": "Alphabet",
    "AMZN": "Amazon", "NVDA": "NVIDIA", "META": "Meta",
    "TSLA": "Tesla", "JPM": "JPMorgan", "V": "Visa",
    "JNJ": "Johnson & Johnson", "XOM": "ExxonMobil",
    "PG": "Procter & Gamble", "HD": "Home Depot",
    "ABBV": "AbbVie", "MRK": "Merck", "KO": "Coca-Cola",
    "PEP": "PepsiCo", "BAC": "Bank of America",
    "NFLX": "Netflix", "AMD": "AMD", "DIS": "Disney",
    "INTC": "Intel", "VZ": "Verizon", "T": "AT&T",
    "PFE": "Pfizer", "BA": "Boeing", "F": "Ford",
    "NKE": "Nike", "UPS": "UPS", "CI": "Cigna",
    "USB": "US Bancorp", "DUK": "Duke Energy",
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


# ── Fundamental Screening ────────────────────────────────────────────────────

def screen_fundamentals(tickers: list[str]) -> dict:
    """
    Holt Fundamental-Daten für alle Ticker.
    Returns: {ticker: {pe, pb, div_yield, market_cap, 52w_change, ...}}
    """
    fundamentals = {}
    for ticker in tickers:
        try:
            info = yf.Ticker(ticker).info
            pe = info.get("trailingPE") or info.get("forwardPE")
            pb = info.get("priceToBook")
            div_yield = info.get("dividendYield", 0) or 0
            market_cap = info.get("marketCap", 0)
            w52_high = info.get("fiftyTwoWeekHigh", 0)
            w52_low = info.get("fiftyTwoWeekLow", 0)
            current = info.get("currentPrice") or info.get("regularMarketPrice", 0)
            w52_change = info.get("52WeekChange")
            profit_margin = info.get("profitMargins")
            revenue_growth = info.get("revenueGrowth")
            name = info.get("shortName") or SCREENING_UNIVERSE.get(ticker, ticker)

            if current and w52_high and w52_high > 0:
                pct_from_high = (current - w52_high) / w52_high
            else:
                pct_from_high = None

            fundamentals[ticker] = {
                "name": name,
                "pe": pe,
                "pb": pb,
                "div_yield": div_yield,
                "market_cap": market_cap,
                "current_price": current,
                "pct_from_52w_high": pct_from_high,
                "52w_change": w52_change,
                "profit_margin": profit_margin,
                "revenue_growth": revenue_growth,
            }
        except Exception:
            continue
    return fundamentals


def select_value_stocks(fundamentals: dict, top_n: int = TOP_N) -> list[str]:
    """
    Wählt die Top-N Value-Aktien basierend auf:
    - Niedriges P/E (Gewicht 40%)
    - Niedriges P/B (Gewicht 30%)
    - Hohe Dividendenrendite (Gewicht 30%)
    Nur Aktien mit positivem P/E und P/B werden berücksichtigt.
    """
    candidates = []
    for ticker, f in fundamentals.items():
        pe = f.get("pe")
        pb = f.get("pb")
        if pe is None or pb is None or pe <= 0 or pb <= 0:
            continue
        if pe > 100 or f.get("market_cap", 0) < 5e9:
            continue
        candidates.append({
            "ticker": ticker,
            "pe": pe,
            "pb": pb,
            "div_yield": f.get("div_yield", 0),
        })

    if len(candidates) < top_n:
        return [c["ticker"] for c in candidates]

    df = pd.DataFrame(candidates).set_index("ticker")
    # Rank: lower PE/PB = better, higher div_yield = better
    df["pe_rank"] = df["pe"].rank(ascending=True)
    df["pb_rank"] = df["pb"].rank(ascending=True)
    df["div_rank"] = df["div_yield"].rank(ascending=False)
    df["score"] = df["pe_rank"] * 0.4 + df["pb_rank"] * 0.3 + df["div_rank"] * 0.3
    df = df.sort_values("score")
    return df.head(top_n).index.tolist()


def select_turnaround_stocks(fundamentals: dict, top_n: int = TOP_N) -> list[str]:
    """
    Wählt die Top-N Turnaround-Aktien basierend auf:
    - Stark gefallen von 52W-Hoch (aber nicht zu stark: -10% bis -50%)
    - Positives Revenue-Wachstum (Wende-Signal)
    - Akzeptable Marktkapitalisierung (>$5B)
    - Bonus: positive Profitmarge (Unternehmen verdient trotz Kurseinbruch)
    """
    candidates = []
    for ticker, f in fundamentals.items():
        pct_from_high = f.get("pct_from_52w_high")
        if pct_from_high is None:
            continue
        if f.get("market_cap", 0) < 5e9:
            continue
        # Muss zwischen -10% und -50% vom Hoch sein (echter Einbruch, kein Crash)
        if not (-0.50 <= pct_from_high <= -0.10):
            continue

        rev_growth = f.get("revenue_growth") or 0
        profit = f.get("profit_margin") or 0

        candidates.append({
            "ticker": ticker,
            "drop": abs(pct_from_high),     # Stärke des Einbruchs
            "rev_growth": rev_growth,        # Wachstum als Wende-Signal
            "profit": max(0, profit),        # Profitabilität
        })

    if len(candidates) < top_n:
        # Relax criteria if not enough: also accept -5% to -55%
        for ticker, f in fundamentals.items():
            pct_from_high = f.get("pct_from_52w_high")
            if pct_from_high is None or f.get("market_cap", 0) < 3e9:
                continue
            if not (-0.55 <= pct_from_high <= -0.05):
                continue
            if ticker not in [c["ticker"] for c in candidates]:
                candidates.append({
                    "ticker": ticker,
                    "drop": abs(pct_from_high),
                    "rev_growth": (f.get("revenue_growth") or 0),
                    "profit": max(0, f.get("profit_margin") or 0),
                })

    if not candidates:
        return []

    df = pd.DataFrame(candidates).set_index("ticker")
    # Score: bigger drop + positive growth + profit = better turnaround
    df["drop_rank"] = df["drop"].rank(ascending=False)       # Bigger drop = higher rank
    df["growth_rank"] = df["rev_growth"].rank(ascending=False)
    df["profit_rank"] = df["profit"].rank(ascending=False)
    df["score"] = df["drop_rank"] * 0.4 + df["growth_rank"] * 0.35 + df["profit_rank"] * 0.25
    df = df.sort_values("score")
    return df.head(top_n).index.tolist()


# ── WF Pipeline per Stock ────────────────────────────────────────────────────

def _wf_single_stock(prices, returns, sdef, wf_cfg=None):
    """Walk-Forward für eine einzelne Strategie auf Einzelaktie."""
    cfg = wf_cfg or WF_CFG
    windows = generate_windows(prices.index, cfg["train"],
                               cfg["test"], cfg["step"])
    if len(windows) < 3:
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
    if len(oos) < 3:
        return None
    r = pd.concat(oos).sort_index()
    return r[~r.index.duplicated(keep="first")]


def run_category_wf(category: str,
                    progress_callback=None,
                    cloud_mode: bool = False) -> dict | None:
    """
    Pipeline für Value- oder Turnaround-Kategorie.
    
    Parameters
    ----------
    category : str
        "value" oder "turnaround"
    cloud_mode : bool
        Reduziertes Universum und kleinere Grids für Cloud
    
    Returns dict mit Resultaten pro Aktie + Aggregat.
    """
    use_universe = CLOUD_SCREENING_UNIVERSE if cloud_mode else SCREENING_UNIVERSE
    use_top_n = CLOUD_TOP_N if cloud_mode else TOP_N
    use_strats = CLOUD_STRATEGY_DEFS if cloud_mode else STRATEGY_DEFS
    use_wf_cfg = CLOUD_WF_CFG if cloud_mode else WF_CFG
    min_data = 400 if cloud_mode else 600

    tickers = list(use_universe.keys())

    if progress_callback:
        progress_callback(0.05, f"{category.title()}: Lade Fundamentaldaten …")

    # 1. Screen fundamentals
    fundamentals = screen_fundamentals(tickers)
    if len(fundamentals) < 5:
        log.warning(f"  {category}: Zu wenig Fundamental-Daten ({len(fundamentals)})")
        return None

    # 2. Select top stocks
    if category == "value":
        selected = select_value_stocks(fundamentals, use_top_n)
    elif category == "turnaround":
        selected = select_turnaround_stocks(fundamentals, use_top_n)
    else:
        return None

    if len(selected) < 3:
        log.warning(f"  {category}: Nur {len(selected)} Aktien gefunden")
        return None

    log.info(f"  {category}: {len(selected)} Aktien ausgewählt: {', '.join(selected)}")

    if progress_callback:
        progress_callback(0.20, f"Lade Kursdaten für {len(selected)} Aktien …")

    # 3. Download price data (10+ Jahre für Turnaround, Value)
    raw = yf.download(selected, start="2012-01-01", auto_adjust=True, progress=False)
    if raw.empty:
        return None

    # 4. Per-stock WF switching
    stock_results = {}
    for si, ticker in enumerate(selected):
        if progress_callback:
            progress_callback(0.25 + 0.65 * (si / len(selected)),
                              f"WF: {ticker} ({si+1}/{len(selected)}) …")

        try:
            if isinstance(raw.columns, pd.MultiIndex):
                col = ("Close", ticker)
                if col not in raw.columns:
                    continue
                stock_close = raw[col].dropna()
            else:
                stock_close = raw["Close"].dropna()

            if len(stock_close) < min_data:
                continue
            stock_ret = stock_close.pct_change().dropna()
            stock_close = stock_close.loc[stock_ret.index]  # Align

            # WF per strategy
            strat_oos = {}
            for sname, sdef in use_strats.items():
                r = _wf_single_stock(stock_close, stock_ret, sdef, wf_cfg=use_wf_cfg)
                if r is not None:
                    strat_oos[sname] = r

            if len(strat_oos) < 2:
                continue

            # Align
            s_start = max(s.first_valid_index() for s in strat_oos.values())
            s_end = min(s.last_valid_index() for s in strat_oos.values())
            aligned = {n: r.loc[s_start:s_end] for n, r in strat_oos.items()}
            df_s = pd.DataFrame(aligned)
            names_s = df_s.columns.tolist()

            # Rolling Sharpe & Switching
            roll_s = pd.DataFrame({
                n: rolling_sharpe(df_s[n], ROLL_SHARPE_WINDOW, RF)
                for n in names_s
            })
            sw_ret_s, active_s = apply_switching(
                df_s, roll_s, tx=TX, slip=SLIP, min_hold=MIN_HOLD,
            )

            # Signal margin
            if len(roll_s) > 0:
                last_sharpes = roll_s.iloc[-1].dropna()
                margin = float(last_sharpes.max()) if len(last_sharpes) > 0 else 0.0
            else:
                margin = 0.0

            bench_s = stock_ret.loc[s_start:s_end]

            fund = fundamentals.get(ticker, {})
            stock_results[ticker] = {
                "switch_ret": sw_ret_s,
                "bench_ret": bench_s,
                "active_strat": active_s,
                "signal": "LONG" if active_s.iloc[-1] != "Cash" else "CASH",
                "strategy": active_s.iloc[-1] if active_s.iloc[-1] != "Cash" else "—",
                "signal_margin": round(margin, 3),
                "sw_sharpe": sharpe_ratio(sw_ret_s, RF),
                "bh_sharpe": sharpe_ratio(bench_s, RF),
                "pct_invested": float((active_s != "Cash").mean()),
                "n_trades": int((active_s != "Cash").astype(float).diff().abs().fillna(0).sum()),
                "name": fund.get("name", use_universe.get(ticker, ticker)),
                "pe": fund.get("pe"),
                "pb": fund.get("pb"),
                "div_yield": fund.get("div_yield"),
                "pct_from_52w_high": fund.get("pct_from_52w_high"),
                "revenue_growth": fund.get("revenue_growth"),
                "close": stock_close,
            }
        except Exception as e:
            log.warning(f"  {category}/{ticker}: {e}")
            continue

    if not stock_results:
        return None

    if progress_callback:
        progress_callback(0.92, "Aggregation …")

    # 5. Aggregate portfolio
    common_start = max(r["switch_ret"].index[0] for r in stock_results.values())
    common_end = min(r["switch_ret"].index[-1] for r in stock_results.values())

    agg_switch = pd.Series(0.0, index=pd.date_range(common_start, common_end, freq="B"))
    agg_bench = pd.Series(0.0, index=agg_switch.index)
    count = 0

    for ticker, sr in stock_results.items():
        sw = sr["switch_ret"].reindex(agg_switch.index, fill_value=0.0)
        bh = sr["bench_ret"].reindex(agg_bench.index, fill_value=0.0)
        agg_switch += sw
        agg_bench += bh
        count += 1

    if count > 0:
        agg_switch /= count
        agg_bench /= count

    agg_switch = agg_switch.loc[common_start:common_end].dropna()
    agg_bench = agg_bench.loc[common_start:common_end].dropna()
    common_idx = agg_switch.index.intersection(agg_bench.index)
    agg_switch = agg_switch.loc[common_idx]
    agg_bench = agg_bench.loc[common_idx]

    sw_eq = _equity(agg_switch)
    bh_eq = _equity(agg_bench)

    if progress_callback:
        progress_callback(1.0, "Fertig!")

    return {
        "category": category,
        "switch_ret": agg_switch,
        "bench_ret": agg_bench,
        "sw_eq": sw_eq,
        "bh_eq": bh_eq,
        "stock_results": stock_results,
        "sw_sharpe": sharpe_ratio(agg_switch, RF),
        "bh_sharpe": sharpe_ratio(agg_bench, RF),
        "pct_invested": np.mean([r["pct_invested"] for r in stock_results.values()]),
        "n_trades": sum(r["n_trades"] for r in stock_results.values()),
        "start": common_start,
        "end": common_end,
        "selected_tickers": list(stock_results.keys()),
    }
