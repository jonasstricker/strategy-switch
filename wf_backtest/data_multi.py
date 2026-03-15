"""
Enhanced Data Loader
=====================
Download multiple ETFs for comparison. Returns dict of DataFrames.
"""

import sys
from typing import Dict, List, Optional

import pandas as pd
import yfinance as yf


# ── Regional ETF tickers ────────────────────────────────────────────────────

REGIONAL_ETFS = {
    "MSCI World":  ["IWDA.AS", "URTH", "ACWI"],
    "Brazil":      ["EWZ"],
    "China":       ["MCHI", "FXI", "GXC"],
    "Europe":      ["VGK", "IEUR.AS", "EZU"],
    "Japan":       ["EWJ", "DXJ"],
}


def download_single(ticker: str, start: str, end: str) -> Optional[pd.DataFrame]:
    """Download a single ticker, return DataFrame with Close/Returns or None."""
    try:
        raw = yf.download(ticker, start=start, end=end,
                          auto_adjust=True, progress=False)
    except Exception:
        return None
    if raw is None or raw.empty:
        return None
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.get_level_values(0)
    if "Close" not in raw.columns:
        return None
    close = raw["Close"].dropna()
    if len(close) < 252:  # need at least 1 year
        return None
    df = pd.DataFrame({"Close": close})
    df["Returns"] = df["Close"].pct_change()
    df = df.dropna()
    return df


def download_regional_etfs(start: str = "2005-01-01",
                           end: str = "2026-02-28",
                           min_years: float = 8.0
                           ) -> Dict[str, pd.DataFrame]:
    """
    Download all regional ETFs. For each region, try tickers in order
    and use the first one that has enough history.
    """
    result = {}
    for region, tickers in REGIONAL_ETFS.items():
        for ticker in tickers:
            print(f"[data] {region}: trying {ticker} …")
            df = download_single(ticker, start, end)
            if df is not None:
                n_years = len(df) / 252
                print(f"[data]   → {len(df)} days ({n_years:.1f} years)")
                if n_years >= min_years:
                    df.attrs["ticker"] = ticker
                    result[region] = df
                    print(f"[data]   ✓ using {ticker}")
                    break
                else:
                    print(f"[data]   ✗ only {n_years:.1f} years (need {min_years})")
            else:
                print(f"[data]   ✗ no data")
    return result
