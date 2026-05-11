#!/usr/bin/env python3
"""
Polygon.io Data Loader with Local Caching
============================================
Downloads historical daily OHLCV data from Polygon.io REST API.
Merges with Yahoo Finance for the most recent data.

Features:
  - Local CSV cache: only downloads once, then reads from disk
  - Rate-limit aware: 5 calls/min on free tier, auto-throttle
  - Merge: Polygon (2003–2016) + Yahoo (2016–today)
  - First 3 years excluded from backtest output (algo warmup)

Usage:
  from wf_backtest.polygon_loader import load_prices
  prices = load_prices(["AAPL", "MSFT", ...], start="2003-01-01")
"""

from __future__ import annotations
import json, logging, os, time
from datetime import datetime, date
from pathlib import Path
from urllib.request import Request, urlopen
from urllib.error import HTTPError

import pandas as pd

log = logging.getLogger("polygon_loader")

# ── Configuration ────────────────────────────────────────────────────────────
CACHE_DIR = Path(__file__).resolve().parent / "cache" / "polygon"
POLYGON_BASE = "https://api.polygon.io"
CALLS_PER_MIN = 5          # Free tier limit
CALL_INTERVAL = 60.0 / CALLS_PER_MIN + 0.5  # ~12.5s between calls

# Polygon start date (data available from ~2003 for most US stocks)
DEFAULT_START = "2003-01-01"
# Where Yahoo takes over (Polygon free tier has 2-year delay for some data)
YAHOO_HANDOFF = "2016-01-01"
# First N years to skip for algo warmup
WARMUP_YEARS = 3


def _get_api_key() -> str:
    """Get Polygon API key from env or config file."""
    key = os.environ.get("POLYGON_API_KEY", "")
    if key:
        return key
    # Try config file
    cfg_path = Path(__file__).resolve().parent / "polygon_config.json"
    if cfg_path.exists():
        with open(cfg_path, "r") as f:
            return json.load(f).get("api_key", "")
    return ""


def _polygon_get(path: str, params: dict, api_key: str) -> dict:
    """Single Polygon REST API call."""
    params["apiKey"] = api_key
    query = "&".join(f"{k}={v}" for k, v in params.items())
    url = f"{POLYGON_BASE}{path}?{query}"
    req = Request(url)
    req.add_header("Accept", "application/json")
    try:
        with urlopen(req) as resp:
            return json.loads(resp.read().decode())
    except HTTPError as e:
        body = e.read().decode() if e.fp else ""
        log.error(f"Polygon API {e.code}: {body}")
        raise


def download_ticker_polygon(ticker: str, start: str, end: str,
                            api_key: str) -> pd.Series | None:
    """
    Download daily close prices for a single ticker from Polygon.io.
    Returns a pd.Series indexed by date, or None on failure.
    """
    try:
        data = _polygon_get(
            f"/v2/aggs/ticker/{ticker}/range/1/day/{start}/{end}",
            {"adjusted": "true", "sort": "asc", "limit": "50000"},
            api_key,
        )
    except Exception as e:
        log.warning(f"  {ticker}: Polygon download failed: {e}")
        return None

    results = data.get("results")
    if not results:
        log.warning(f"  {ticker}: No data from Polygon")
        return None

    dates = [pd.Timestamp(r["t"], unit="ms").normalize() for r in results]
    closes = [r["c"] for r in results]
    s = pd.Series(closes, index=pd.DatetimeIndex(dates), name=ticker)
    s = s[~s.index.duplicated(keep="last")]
    s = s.sort_index()
    return s


def _cache_path(ticker: str) -> Path:
    """Path to cached CSV for a ticker."""
    return CACHE_DIR / f"{ticker.replace('.', '_')}.csv"


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


def _cache_is_fresh(ticker: str, max_age_days: int = 30) -> bool:
    """Check if cache file exists and is recent enough."""
    path = _cache_path(ticker)
    if not path.exists():
        return False
    mtime = datetime.fromtimestamp(path.stat().st_mtime)
    age = (datetime.now() - mtime).days
    return age <= max_age_days


def download_with_cache(tickers: list[str],
                        start: str = DEFAULT_START,
                        end: str = None,
                        api_key: str = None,
                        progress_callback=None,
                        force_refresh: bool = False) -> dict[str, pd.Series]:
    """
    Download historical prices for multiple tickers from Polygon.io.
    Uses local CSV cache to avoid re-downloading.

    Parameters
    ----------
    tickers : list of str
        Stock tickers to download
    start : str
        Start date (YYYY-MM-DD), default 2003-01-01
    end : str
        End date (YYYY-MM-DD), default today
    api_key : str
        Polygon API key (or reads from env/config)
    progress_callback : callable
        Optional (pct, msg) callback for progress updates
    force_refresh : bool
        Force re-download even if cached

    Returns
    -------
    dict of {ticker: pd.Series} with daily close prices
    """
    if api_key is None:
        api_key = _get_api_key()
    if not api_key:
        log.error("No Polygon API key! Set POLYGON_API_KEY env var or "
                  "create wf_backtest/polygon_config.json")
        return {}

    if end is None:
        end = date.today().strftime("%Y-%m-%d")

    result = {}
    to_download = []

    # Phase 1: Load from cache where possible
    for ticker in tickers:
        if not force_refresh and _cache_is_fresh(ticker):
            cached = _load_cache(ticker)
            if cached is not None and len(cached) > 100:
                result[ticker] = cached
                continue
        to_download.append(ticker)

    if to_download:
        log.info(f"Polygon: {len(result)} aus Cache, "
                 f"{len(to_download)} zum Download")
    else:
        log.info(f"Polygon: Alle {len(result)} Ticker aus Cache geladen")
        return result

    # Phase 2: Download missing tickers with rate limiting
    total = len(to_download)
    call_count = 0
    for i, ticker in enumerate(to_download):
        if progress_callback:
            pct = (i + 1) / total
            eta_min = (total - i - 1) * CALL_INTERVAL / 60
            progress_callback(
                pct,
                f"Polygon: {ticker} ({i+1}/{total}), "
                f"~{eta_min:.0f} min verbleibend"
            )

        # Rate limiting
        if call_count > 0 and call_count % CALLS_PER_MIN == 0:
            log.info(f"  Rate limit: Warte {CALL_INTERVAL:.0f}s "
                     f"({call_count} calls bisher) …")
        if call_count > 0:
            time.sleep(CALL_INTERVAL)

        series = download_ticker_polygon(ticker, start, end, api_key)
        call_count += 1

        if series is not None and len(series) > 100:
            _save_cache(ticker, series)
            result[ticker] = series
            log.info(f"  ✓ {ticker}: {len(series)} Tage "
                     f"({series.index[0].date()} → {series.index[-1].date()})")
        else:
            log.warning(f"  ✗ {ticker}: Nicht genug Daten")

    return result


def merge_polygon_yahoo(polygon_prices: dict[str, pd.Series],
                        yahoo_prices: dict[str, pd.Series],
                        handoff_date: str = YAHOO_HANDOFF) -> dict[str, pd.Series]:
    """
    Merge Polygon historical data with Yahoo recent data.

    Strategy:
      - Use Polygon data up to handoff_date
      - Use Yahoo data from handoff_date onwards
      - Normalize at overlap point to handle any price differences

    Parameters
    ----------
    polygon_prices : dict
        {ticker: Series} from Polygon (2003-2016+)
    yahoo_prices : dict
        {ticker: Series} from Yahoo (2016-today)
    handoff_date : str
        Date to switch from Polygon to Yahoo

    Returns
    -------
    dict of {ticker: pd.Series} merged daily close prices
    """
    cutoff = pd.Timestamp(handoff_date)
    merged = {}

    all_tickers = set(polygon_prices.keys()) | set(yahoo_prices.keys())

    for ticker in all_tickers:
        poly = polygon_prices.get(ticker)
        yahoo = yahoo_prices.get(ticker)

        if poly is not None and yahoo is not None:
            # Both available → merge at handoff point
            poly_part = poly[poly.index < cutoff]
            yahoo_part = yahoo[yahoo.index >= cutoff]

            if len(poly_part) > 0 and len(yahoo_part) > 0:
                # Normalize: find the ratio at the overlap point
                # Use the first Yahoo price and last Polygon price
                last_poly = float(poly_part.iloc[-1])
                first_yahoo = float(yahoo_part.iloc[0])

                if last_poly > 0 and first_yahoo > 0:
                    ratio = first_yahoo / last_poly
                    poly_adjusted = poly_part * ratio
                    combined = pd.concat([poly_adjusted, yahoo_part])
                    combined = combined[~combined.index.duplicated(keep="last")]
                    combined = combined.sort_index()
                    merged[ticker] = combined
                else:
                    merged[ticker] = yahoo
            elif len(yahoo_part) > 0:
                merged[ticker] = yahoo
            else:
                merged[ticker] = poly
        elif yahoo is not None:
            merged[ticker] = yahoo
        elif poly is not None:
            merged[ticker] = poly

    return merged


def load_prices(tickers: list[str],
                start: str = DEFAULT_START,
                api_key: str = None,
                skip_warmup: bool = True,
                warmup_years: int = WARMUP_YEARS,
                progress_callback=None,
                force_refresh: bool = False) -> dict[str, pd.Series]:
    """
    Main entry point: Load historical prices with Polygon + Yahoo merge.

    1. Download from Polygon (with cache) for 2003–2016
    2. Download from Yahoo for 2016–today
    3. Merge at handoff date
    4. Optionally skip first N years (warmup)

    Parameters
    ----------
    tickers : list of str
    start : str
        Earliest date to request from Polygon
    api_key : str
        Polygon API key
    skip_warmup : bool
        If True, remove first warmup_years from output
    warmup_years : int
        Number of years to skip (default 3)
    progress_callback : callable
    force_refresh : bool

    Returns
    -------
    dict of {ticker: pd.Series} with daily close prices
    """
    import yfinance as yf

    if api_key is None:
        api_key = _get_api_key()

    # 1. Polygon historical data
    log.info("Phase 1: Polygon.io historische Daten laden …")
    poly_prices = download_with_cache(
        tickers, start=start, end=YAHOO_HANDOFF,
        api_key=api_key,
        progress_callback=progress_callback,
        force_refresh=force_refresh,
    )

    # 2. Yahoo recent data
    log.info("Phase 2: Yahoo Finance aktuelle Daten laden …")
    if progress_callback:
        progress_callback(0.0, "Yahoo: Batch-Download …")

    raw = yf.download(tickers, start=YAHOO_HANDOFF,
                      auto_adjust=True, progress=False)
    yahoo_prices = {}
    if not raw.empty:
        if isinstance(raw.columns, pd.MultiIndex):
            for ticker in tickers:
                try:
                    col = ("Close", ticker)
                    if col in raw.columns:
                        s = raw[col].dropna()
                        if len(s) > 50:
                            yahoo_prices[ticker] = s
                except Exception:
                    continue
        else:
            # Single ticker
            if "Close" in raw.columns:
                s = raw["Close"].dropna()
                if len(s) > 50:
                    yahoo_prices[tickers[0]] = s

    log.info(f"  Yahoo: {len(yahoo_prices)} Ticker geladen")

    # 3. Merge
    log.info("Phase 3: Merge Polygon + Yahoo …")
    merged = merge_polygon_yahoo(poly_prices, yahoo_prices,
                                 handoff_date=YAHOO_HANDOFF)

    # 4. Skip warmup years
    if skip_warmup and warmup_years > 0:
        for ticker in list(merged.keys()):
            s = merged[ticker]
            if len(s) > 0:
                warmup_end = s.index[0] + pd.DateOffset(years=warmup_years)
                trimmed = s[s.index >= warmup_end]
                if len(trimmed) > 252:  # At least 1 year after warmup
                    merged[ticker] = trimmed
                else:
                    log.warning(f"  {ticker}: Zu wenig Daten nach Warmup-Trim")
                    del merged[ticker]

    log.info(f"Ergebnis: {len(merged)} Ticker mit Daten")
    for t in sorted(merged.keys())[:5]:
        s = merged[t]
        log.info(f"  {t}: {s.index[0].date()} → {s.index[-1].date()} "
                 f"({len(s)} Tage)")
    if len(merged) > 5:
        log.info(f"  … und {len(merged) - 5} weitere")

    return merged


# ── CLI for testing ──────────────────────────────────────────────────────────
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(message)s")

    test_tickers = ["AAPL", "MSFT", "GOOGL", "JPM", "XOM"]
    prices = load_prices(test_tickers, api_key=_get_api_key())

    print(f"\n{'='*50}")
    print(f"Ergebnis: {len(prices)} Ticker")
    for t, s in sorted(prices.items()):
        print(f"  {t}: {s.index[0].date()} → {s.index[-1].date()} "
              f"({len(s)} Tage, {len(s)/252:.1f} Jahre)")
