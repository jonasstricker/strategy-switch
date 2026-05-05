#!/usr/bin/env python3
"""
Historical Data Loader with Local Caching
============================================
Downloads long-term historical daily close prices from Yahoo Finance (2003+).
Caches data locally as CSV files to avoid re-downloading.

Features:
  - Local CSV cache: downloads once, then reads from disk
  - Auto-refresh: updates cached files if older than max_age_days
  - First 3 years excluded from output (algo warmup for WF)
  - Batch download for efficiency

Usage:
  from wf_backtest.hist_loader import load_prices_cached
  prices = load_prices_cached(["AAPL", "MSFT", ...])
"""

from __future__ import annotations
import logging, os
from datetime import datetime
from pathlib import Path

import pandas as pd
import yfinance as yf

log = logging.getLogger("hist_loader")

# ── Configuration ────────────────────────────────────────────────────────────
CACHE_DIR = Path(__file__).resolve().parent / "cache" / "hist_prices"
DEFAULT_START = "2003-01-01"
WARMUP_YEARS = 3


def _cache_path(ticker: str) -> Path:
    """Path to cached CSV for a ticker."""
    return CACHE_DIR / f"{ticker.replace('.', '_').replace('-', '_')}.csv"


def _save_cache(ticker: str, series: pd.Series):
    """Save price series to CSV cache."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    path = _cache_path(ticker)
    df = pd.DataFrame({"date": series.index.strftime("%Y-%m-%d"),
                        "close": series.values})
    df.to_csv(path, index=False)


def _load_cache(ticker: str) -> pd.Series | None:
    """Load cached price series. Returns None if not cached."""
    path = _cache_path(ticker)
    if not path.exists():
        return None
    try:
        df = pd.read_csv(path, parse_dates=["date"])
        s = pd.Series(df["close"].values,
                      index=pd.DatetimeIndex(df["date"]),
                      name=ticker)
        return s
    except Exception:
        return None


def _cache_is_fresh(ticker: str, max_age_days: int = 7) -> bool:
    """Check if cache file exists and is recent enough."""
    path = _cache_path(ticker)
    if not path.exists():
        return False
    mtime = datetime.fromtimestamp(path.stat().st_mtime)
    age = (datetime.now() - mtime).days
    return age <= max_age_days


def _update_cache(ticker: str, cached: pd.Series) -> pd.Series:
    """Update a cached series with the latest data from Yahoo."""
    last_date = cached.index[-1]
    start = (last_date + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    try:
        raw = yf.download(ticker, start=start, auto_adjust=True, progress=False)
        if raw.empty:
            return cached
        if isinstance(raw.columns, pd.MultiIndex):
            raw.columns = raw.columns.get_level_values(0)
        if "Close" not in raw.columns:
            return cached
        new_data = raw["Close"].dropna()
        if len(new_data) == 0:
            return cached
        combined = pd.concat([cached, new_data])
        combined = combined[~combined.index.duplicated(keep="last")]
        combined = combined.sort_index()
        _save_cache(ticker, combined)
        return combined
    except Exception:
        return cached


def load_prices_cached(tickers: list[str],
                       start: str = DEFAULT_START,
                       skip_warmup: bool = True,
                       warmup_years: int = WARMUP_YEARS,
                       max_cache_age_days: int = 7,
                       progress_callback=None,
                       force_refresh: bool = False) -> dict[str, pd.Series]:
    """
    Load historical close prices with CSV caching.

    First call: batch-downloads from Yahoo and caches as CSV.
    Subsequent calls: reads from cache, only updates if stale.

    Parameters
    ----------
    tickers : list of str
        Stock tickers to load
    start : str
        Earliest date (YYYY-MM-DD), default 2003-01-01
    skip_warmup : bool
        Remove first warmup_years from output
    warmup_years : int
        Years to skip (default 3) — so usable data starts ~2006
    max_cache_age_days : int
        Re-download if cache older than this (default 7)
    progress_callback : callable
        Optional (pct, msg) callback
    force_refresh : bool
        Force full re-download

    Returns
    -------
    dict of {ticker: pd.Series} with daily close prices
    """
    result = {}
    to_download = []

    # Phase 1: Load from cache where possible
    for ticker in tickers:
        if not force_refresh and _cache_is_fresh(ticker, max_cache_age_days):
            cached = _load_cache(ticker)
            if cached is not None and len(cached) > 100:
                # Update with latest data if cache is >1 day old
                if not _cache_is_fresh(ticker, max_age_days=1):
                    cached = _update_cache(ticker, cached)
                result[ticker] = cached
                continue
        to_download.append(ticker)

    cached_count = len(result)
    if to_download:
        log.info(f"Hist-Loader: {cached_count} aus Cache, "
                 f"{len(to_download)} zum Download")
    else:
        log.info(f"Hist-Loader: Alle {cached_count} Ticker aus Cache")

    # Phase 2: Batch download missing tickers
    if to_download:
        if progress_callback:
            progress_callback(0.05, f"Yahoo: Batch-Download {len(to_download)} Ticker …")

        raw = yf.download(to_download, start=start,
                          auto_adjust=True, progress=False)

        if not raw.empty:
            if isinstance(raw.columns, pd.MultiIndex):
                for i, ticker in enumerate(to_download):
                    if progress_callback and i % 10 == 0:
                        progress_callback(
                            0.05 + 0.80 * (i / len(to_download)),
                            f"Verarbeite {ticker} ({i+1}/{len(to_download)}) …"
                        )
                    try:
                        col = ("Close", ticker)
                        if col in raw.columns:
                            s = raw[col].dropna()
                            if len(s) > 100:
                                _save_cache(ticker, s)
                                result[ticker] = s
                                log.info(f"  ✓ {ticker}: {len(s)} Tage "
                                         f"({s.index[0].date()} → {s.index[-1].date()})")
                    except Exception:
                        continue
            elif len(to_download) == 1:
                if "Close" in raw.columns:
                    s = raw["Close"].dropna()
                    if len(s) > 100:
                        ticker = to_download[0]
                        _save_cache(ticker, s)
                        result[ticker] = s

    # Phase 3: Skip warmup years
    if skip_warmup and warmup_years > 0:
        for ticker in list(result.keys()):
            s = result[ticker]
            if len(s) > 0:
                warmup_end = s.index[0] + pd.DateOffset(years=warmup_years)
                trimmed = s[s.index >= warmup_end]
                if len(trimmed) > 252:
                    result[ticker] = trimmed
                else:
                    log.warning(f"  {ticker}: Zu wenig Daten nach Warmup-Trim")
                    del result[ticker]

    # Phase 4: Trim to requested start date (cache may have older data)
    start_ts = pd.Timestamp(start)
    for ticker in list(result.keys()):
        s = result[ticker]
        trimmed = s[s.index >= start_ts]
        if len(trimmed) > 100:
            result[ticker] = trimmed
        else:
            del result[ticker]

    if progress_callback:
        progress_callback(1.0, f"Fertig: {len(result)} Ticker geladen")

    log.info(f"Ergebnis: {len(result)} Ticker mit Daten")
    for t in sorted(result.keys())[:5]:
        s = result[t]
        log.info(f"  {t}: {s.index[0].date()} → {s.index[-1].date()} "
                 f"({len(s)} Tage, {len(s)/252:.1f} Jahre)")
    if len(result) > 5:
        log.info(f"  … und {len(result) - 5} weitere")

    return result


def clear_cache():
    """Remove all cached price files."""
    if CACHE_DIR.exists():
        for f in CACHE_DIR.glob("*.csv"):
            f.unlink()
        log.info(f"Cache gelöscht: {CACHE_DIR}")


def cache_info() -> dict:
    """Return info about cached files."""
    if not CACHE_DIR.exists():
        return {"count": 0, "size_mb": 0, "tickers": []}
    files = list(CACHE_DIR.glob("*.csv"))
    total_size = sum(f.stat().st_size for f in files)
    tickers = [f.stem.replace("_", ".") for f in files]
    return {
        "count": len(files),
        "size_mb": round(total_size / 1024 / 1024, 2),
        "tickers": sorted(tickers),
    }


# ── CLI for testing ──────────────────────────────────────────────────────────
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(message)s")

    test_tickers = ["AAPL", "MSFT", "GOOGL", "JPM", "XOM",
                    "VZ", "T", "PFE", "BA", "KO"]

    print("=== Download + Cache Test ===")
    prices = load_prices_cached(test_tickers)

    print(f"\nErgebnis: {len(prices)} Ticker")
    for t, s in sorted(prices.items()):
        print(f"  {t}: {s.index[0].date()} → {s.index[-1].date()} "
              f"({len(s)} Tage, {len(s)/252:.1f} Jahre)")

    print(f"\nCache: {cache_info()}")
