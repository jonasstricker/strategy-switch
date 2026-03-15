"""
Data Loader
============
Download MSCI World ETF proxy data via yfinance.
Falls through a list of tickers until one satisfies the minimum-history
requirement.
"""

import sys
from typing import Optional

import pandas as pd
import yfinance as yf

from .cfg import DataConfig


def download_data(cfg: DataConfig) -> pd.DataFrame:
    """
    Try each ticker in order; return the first one that has
    >= cfg.min_years of daily data.
    Returns a DataFrame with columns: Close, Returns.
    """
    for ticker in cfg.tickers:
        print(f"[data] Trying {ticker} …")
        try:
            raw = yf.download(
                ticker,
                start=cfg.start_date,
                end=cfg.end_date,
                auto_adjust=True,
                progress=False,
            )
        except Exception as exc:
            print(f"[data]   ✗ download failed: {exc}")
            continue

        if raw is None or raw.empty:
            print(f"[data]   ✗ no data returned")
            continue

        # Handle multi-level columns from yfinance
        if isinstance(raw.columns, pd.MultiIndex):
            raw.columns = raw.columns.get_level_values(0)

        if "Close" not in raw.columns:
            print(f"[data]   ✗ no 'Close' column")
            continue

        close: pd.Series = raw["Close"].dropna()
        n_years = len(close) / 252
        print(f"[data]   → {len(close)} days ({n_years:.1f} years)")

        if n_years < cfg.min_years:
            print(f"[data]   ✗ need ≥ {cfg.min_years} years")
            continue

        df = pd.DataFrame({"Close": close})
        df["Returns"] = df["Close"].pct_change()
        df = df.dropna()
        print(f"[data]   ✓ using {ticker} "
              f"({df.index[0].date()} → {df.index[-1].date()})")
        return df

    sys.exit("[data] ERROR – no ticker with sufficient history. "
             "Check internet connection or adjust tickers / min_years.")


def validate_data(df: pd.DataFrame) -> None:
    """Basic sanity checks (print warnings, do not raise)."""
    pct_nan = df["Close"].isna().mean()
    if pct_nan > 0.01:
        print(f"[data] WARNING: {pct_nan:.1%} NaN in Close")

    # Check for suspicious gaps
    gaps = df.index.to_series().diff().dt.days
    big_gaps = gaps[gaps > 5]
    if len(big_gaps) > 0:
        print(f"[data] WARNING: {len(big_gaps)} gaps > 5 calendar days")
