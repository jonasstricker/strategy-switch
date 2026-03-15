#!/usr/bin/env python3
"""
Aktien-Schwarm (Stock Swarm)
==============================
Dynamische Aktienauswahl basierend auf:
  - Fundamentaldaten (Earnings Growth, Revenue Growth, ROE, P/E)
  - Momentum (6M & 12M Kursperformance)
  - Qualität (Profitmargen, Verschuldung)

System:
  1. Starte mit breitem Universum (~60 liquide Large Caps)
  2. Score jede Aktie auf Composite-Metrik
  3. Wähle Top-N Aktien (Standard: 10)
  4. Rebalance alle 63 Tage (quartalsweise)
  5. Overlay: WF-Switch Cash/Long Signal → wenn Cash, alles verkaufen
"""

from __future__ import annotations
import sys, os, warnings
warnings.filterwarnings("ignore")

if __package__ is None:
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    __package__ = "wf_backtest"

import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime

# ── Universum: Große, liquide US-Aktien quer über Sektoren ─────────────────
UNIVERSE = {
    # Tech
    "AAPL": "Apple", "MSFT": "Microsoft", "GOOGL": "Alphabet",
    "AMZN": "Amazon", "NVDA": "NVIDIA", "META": "Meta",
    "AVGO": "Broadcom", "CRM": "Salesforce", "ADBE": "Adobe",
    "ORCL": "Oracle", "AMD": "AMD", "INTC": "Intel",
    # Finance
    "JPM": "JPMorgan", "V": "Visa", "MA": "Mastercard",
    "BAC": "Bank of America", "GS": "Goldman Sachs", "BLK": "BlackRock",
    # Healthcare
    "UNH": "UnitedHealth", "JNJ": "Johnson & Johnson", "LLY": "Eli Lilly",
    "PFE": "Pfizer", "ABBV": "AbbVie", "MRK": "Merck",
    "TMO": "Thermo Fisher",
    # Consumer
    "PG": "Procter & Gamble", "KO": "Coca-Cola", "PEP": "PepsiCo",
    "COST": "Costco", "WMT": "Walmart", "MCD": "McDonald's",
    "NKE": "Nike", "SBUX": "Starbucks",
    # Industrie
    "CAT": "Caterpillar", "HON": "Honeywell", "UPS": "UPS",
    "GE": "GE Aerospace", "RTX": "RTX", "LMT": "Lockheed Martin",
    "DE": "John Deere",
    # Energie
    "XOM": "ExxonMobil", "CVX": "Chevron", "COP": "ConocoPhillips",
    # Kommunikation
    "DIS": "Disney", "NFLX": "Netflix", "CMCSA": "Comcast",
    # Versorger / REITs
    "NEE": "NextEra Energy", "AMT": "American Tower",
    # Weitere
    "TSLA": "Tesla", "BRK-B": "Berkshire Hathaway",
    "HD": "Home Depot", "LOW": "Lowe's",
    "TXN": "Texas Instruments", "QCOM": "Qualcomm",
    "AMAT": "Applied Materials", "NOW": "ServiceNow",
    "ISRG": "Intuitive Surgical", "UBER": "Uber",
}

# ── Scoring-Gewichte ─────────────────────────────────────────────────────────
SCORE_WEIGHTS = {
    "momentum_6m": 0.25,      # 6-Monats-Momentum
    "momentum_12m": 0.15,     # 12-Monats-Momentum
    "earnings_growth": 0.20,  # Gewinnwachstum
    "revenue_growth": 0.15,   # Umsatzwachstum
    "roe": 0.10,              # Eigenkapitalrendite
    "profit_margin": 0.10,    # Gewinnmarge
    "pe_inverse": 0.05,       # Niedriger P/E = besser (invertiert)
}


def _safe_float(val, default=np.nan):
    """Safely convert a value to float."""
    try:
        if val is None:
            return default
        return float(val)
    except (ValueError, TypeError):
        return default


def fetch_fundamentals(tickers: list[str]) -> pd.DataFrame:
    """
    Fetch fundamental data for all tickers using yfinance.
    Returns DataFrame with one row per ticker.
    """
    rows = []
    for ticker in tickers:
        try:
            info = yf.Ticker(ticker).info
            rows.append({
                "ticker": ticker,
                "name": UNIVERSE.get(ticker, ticker),
                "sector": info.get("sector", "Unknown"),
                "market_cap": _safe_float(info.get("marketCap")),
                "trailing_pe": _safe_float(info.get("trailingPE")),
                "forward_pe": _safe_float(info.get("forwardPE")),
                "earnings_growth": _safe_float(info.get("earningsGrowth")),
                "revenue_growth": _safe_float(info.get("revenueGrowth")),
                "roe": _safe_float(info.get("returnOnEquity")),
                "profit_margin": _safe_float(info.get("profitMargins")),
                "debt_to_equity": _safe_float(info.get("debtToEquity")),
                "current_price": _safe_float(info.get("currentPrice")),
                "target_price": _safe_float(info.get("targetMeanPrice")),
                "recommendation": info.get("recommendationKey", "none"),
            })
        except Exception:
            continue
    return pd.DataFrame(rows)


def fetch_prices(tickers: list[str], start: str = "2020-01-01") -> dict[str, pd.Series]:
    """Download close prices for all tickers. Returns dict ticker → Series."""
    raw = yf.download(tickers, start=start, auto_adjust=True, progress=False)
    if raw.empty:
        return {}

    prices = {}
    if isinstance(raw.columns, pd.MultiIndex):
        for ticker in tickers:
            try:
                col = ("Close", ticker)
                if col in raw.columns:
                    s = raw[col].dropna()
                    if len(s) > 100:
                        prices[ticker] = s
            except Exception:
                continue
    else:
        # Single ticker
        if "Close" in raw.columns:
            prices[tickers[0]] = raw["Close"].dropna()
    return prices


def compute_momentum(prices: dict[str, pd.Series]) -> pd.DataFrame:
    """Compute 6M and 12M momentum for each ticker."""
    rows = []
    for ticker, p in prices.items():
        if len(p) < 252:
            continue
        mom_6m = (p.iloc[-1] / p.iloc[-126] - 1) if len(p) >= 126 else np.nan
        mom_12m = (p.iloc[-1] / p.iloc[-252] - 1) if len(p) >= 252 else np.nan
        # Volatility (annualized)
        vol = p.pct_change().dropna().iloc[-252:].std() * np.sqrt(252)
        rows.append({
            "ticker": ticker,
            "momentum_6m": mom_6m,
            "momentum_12m": mom_12m,
            "volatility": vol,
            "price_latest": float(p.iloc[-1]),
        })
    return pd.DataFrame(rows)


def _rank_percentile(series: pd.Series, ascending: bool = True) -> pd.Series:
    """Rank as percentile 0-1. ascending=True means higher rank = higher value."""
    if ascending:
        return series.rank(pct=True, na_option="bottom")
    else:
        return (1 - series.rank(pct=True, na_option="top"))


def score_stocks(fundamentals: pd.DataFrame,
                 momentum: pd.DataFrame,
                 weights: dict = None) -> pd.DataFrame:
    """
    Compute composite score for each stock.
    Higher = better. Returns merged DataFrame with 'composite_score'.
    """
    if weights is None:
        weights = SCORE_WEIGHTS

    df = fundamentals.merge(momentum, on="ticker", how="inner")

    # Rank each factor as percentile (0-1)
    df["r_mom6"] = _rank_percentile(df["momentum_6m"], ascending=True)
    df["r_mom12"] = _rank_percentile(df["momentum_12m"], ascending=True)
    df["r_earn"] = _rank_percentile(df["earnings_growth"], ascending=True)
    df["r_rev"] = _rank_percentile(df["revenue_growth"], ascending=True)
    df["r_roe"] = _rank_percentile(df["roe"], ascending=True)
    df["r_margin"] = _rank_percentile(df["profit_margin"], ascending=True)
    # Lower P/E = better → invert
    df["r_pe_inv"] = _rank_percentile(df["trailing_pe"], ascending=False)

    # Composite
    df["composite_score"] = (
        weights["momentum_6m"] * df["r_mom6"] +
        weights["momentum_12m"] * df["r_mom12"] +
        weights["earnings_growth"] * df["r_earn"] +
        weights["revenue_growth"] * df["r_rev"] +
        weights["roe"] * df["r_roe"] +
        weights["profit_margin"] * df["r_margin"] +
        weights["pe_inverse"] * df["r_pe_inv"]
    )

    return df.sort_values("composite_score", ascending=False)


def select_swarm(scored: pd.DataFrame, top_n: int = 10,
                 max_per_sector: int = 3) -> pd.DataFrame:
    """
    Select top N stocks with sector diversification.
    Max 3 per sector to avoid concentration.
    """
    selected = []
    sector_count = {}
    for _, row in scored.iterrows():
        sector = row.get("sector", "Unknown")
        if sector_count.get(sector, 0) >= max_per_sector:
            continue
        selected.append(row)
        sector_count[sector] = sector_count.get(sector, 0) + 1
        if len(selected) >= top_n:
            break
    return pd.DataFrame(selected)


def backtest_swarm(prices: dict[str, pd.Series],
                   scored_history: list[tuple[pd.Timestamp, list[str]]],
                   rebalance_days: int = 63,
                   tx_cost: float = 0.001,
                   slippage: float = 0.0005) -> pd.DataFrame:
    """
    Backtest a swarm portfolio with periodic rebalancing.

    scored_history: list of (date, [ticker1, ticker2, ...]) tuples
                    defining which stocks to hold at each rebalance.

    Returns DataFrame with columns: date, swarm_return, bh_return (SPY)
    """
    # Build returns for each ticker
    all_returns = {}
    for ticker, p in prices.items():
        all_returns[ticker] = p.pct_change().dropna()

    if not all_returns:
        return pd.DataFrame()

    # Common date range
    common_idx = None
    for r in all_returns.values():
        if common_idx is None:
            common_idx = r.index
        else:
            common_idx = common_idx.intersection(r.index)

    if common_idx is None or len(common_idx) < 100:
        return pd.DataFrame()

    common_idx = common_idx.sort_values()

    # Build daily swarm returns
    swarm_ret = pd.Series(0.0, index=common_idx)

    # Simple approach: equal-weight top stocks, rebalance at each scored_history entry
    current_stocks = []
    history_idx = 0

    for date in common_idx:
        # Check if rebalance needed
        while history_idx < len(scored_history) and scored_history[history_idx][0] <= date:
            new_stocks = scored_history[history_idx][1]
            # Apply cost for turnover
            old_set = set(current_stocks)
            new_set = set(new_stocks)
            turnover = len(old_set.symmetric_difference(new_set)) / max(len(new_set), 1)
            cost = turnover * (tx_cost + slippage)
            swarm_ret.loc[date] -= cost
            current_stocks = new_stocks
            history_idx += 1

        # Equal-weight daily return
        if current_stocks:
            daily_rets = []
            for t in current_stocks:
                if t in all_returns and date in all_returns[t].index:
                    daily_rets.append(all_returns[t].loc[date])
            if daily_rets:
                swarm_ret.loc[date] += np.mean(daily_rets)

    return swarm_ret


def run_swarm_analysis(top_n: int = 10, progress_callback=None):
    """
    Full swarm analysis pipeline.
    Returns dict with all results.
    """
    tickers = list(UNIVERSE.keys())

    if progress_callback:
        progress_callback(0.1, "Lade Fundamentaldaten …")

    # 1. Fetch fundamentals
    fundamentals = fetch_fundamentals(tickers)
    if fundamentals.empty:
        return None

    if progress_callback:
        progress_callback(0.3, "Lade Kursdaten …")

    # 2. Fetch prices
    prices = fetch_prices(tickers, start="2020-01-01")
    if not prices:
        return None

    if progress_callback:
        progress_callback(0.5, "Berechne Momentum & Scoring …")

    # 3. Compute momentum
    momentum = compute_momentum(prices)

    # 4. Score & select
    scored = score_stocks(fundamentals, momentum)
    swarm = select_swarm(scored, top_n=top_n)

    if progress_callback:
        progress_callback(0.6, "Backteste historisch …")

    # 5. Historical backtest with quarterly rebalancing
    # We'll simulate past rebalancing using momentum only (fundamentals are point-in-time)
    # Use rolling 6M/12M momentum + simple quality proxy
    swarm_tickers = swarm["ticker"].tolist()

    # Build quarterly rebalance points based on momentum ranking
    all_dates = None
    for p in prices.values():
        if all_dates is None:
            all_dates = p.index
        else:
            all_dates = all_dates.union(p.index)
    all_dates = all_dates.sort_values()

    rebalance_history = []
    rebalance_days = 63

    for i in range(252, len(all_dates), rebalance_days):
        date = all_dates[i]
        # Score based on historical momentum at that point
        scores = []
        for ticker, p in prices.items():
            p_to_date = p.loc[:date]
            if len(p_to_date) < 252:
                continue
            mom6 = p_to_date.iloc[-1] / p_to_date.iloc[-126] - 1 if len(p_to_date) >= 126 else 0
            mom12 = p_to_date.iloc[-1] / p_to_date.iloc[-252] - 1 if len(p_to_date) >= 252 else 0
            # Use momentum + volatility-adjusted return as proxy
            vol = p_to_date.pct_change().dropna().iloc[-126:].std() * np.sqrt(252)
            risk_adj = mom6 / max(vol, 0.01)
            scores.append((ticker, 0.5 * mom6 + 0.3 * mom12 + 0.2 * risk_adj))

        scores.sort(key=lambda x: -x[1])
        top = [s[0] for s in scores[:top_n]]
        rebalance_history.append((date, top))

    if progress_callback:
        progress_callback(0.8, "Berechne Portfolio-Returns …")

    # 6. Run backtest
    swarm_returns = backtest_swarm(prices, rebalance_history)

    # 7. SPY benchmark
    spy_prices = prices.get("SPY", None)
    if spy_prices is None:
        try:
            raw = yf.download("SPY", start="2020-01-01", auto_adjust=True, progress=False)
            if isinstance(raw.columns, pd.MultiIndex):
                spy_prices = raw[("Close", "SPY")].dropna()
            else:
                spy_prices = raw["Close"].dropna()
        except Exception:
            spy_prices = None

    spy_ret = spy_prices.pct_change().dropna() if spy_prices is not None else None

    if progress_callback:
        progress_callback(1.0, "Fertig!")

    return {
        "fundamentals": fundamentals,
        "momentum": momentum,
        "scored": scored,
        "swarm": swarm,
        "swarm_tickers": swarm_tickers,
        "swarm_returns": swarm_returns,
        "spy_returns": spy_ret,
        "prices": prices,
        "rebalance_history": rebalance_history,
        "top_n": top_n,
    }
