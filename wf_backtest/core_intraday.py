"""
Live estimate of the index close shortly before the US close (same-day trading).

The core rule needs the index close of day t. Shortly before the bell the live
price decides the rule exactly like the final close on all but ~0.3-0.8 % of the
days (signal 10-60 min before the close, backtest section B), so the decision is
taken on the live price and the order is sent right away.

Sources (the live price is never taken from a single unchecked number):
  1. Alpaca snapshot of the ETF proxy (SPY/QQQ):
       index_now = last_complete_index_close * ETF_last_trade / ETF_previous_close
  2. yfinance 1-minute bars of the index itself (cross-check, may be delayed).
Both must agree within CROSSCHECK_TOL, otherwise no signal is produced.
"""
from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timezone
from urllib.request import Request, urlopen
from zoneinfo import ZoneInfo

import pandas as pd

log = logging.getLogger("core_intraday")

NY = ZoneInfo("America/New_York")
DATA_URL = "https://data.alpaca.markets"
MAX_QUOTE_AGE_S = 300          # latest ETF trade must be younger than 5 min
MAX_YF_AGE_S = 1800            # yfinance index bar (delayed quotes) younger than 30 min
CROSSCHECK_TOL = 0.01          # sources may differ by max. 1 % (delay of the cross-check source)


class LiveQuoteError(RuntimeError):
    """No trustworthy live price -> no signal, no trading."""


def session_is_running(now_ny: datetime) -> bool:
    """Regular US session Mon-Fri 9:30-16:00 ET (holidays are caught by the quote age check)."""
    if now_ny.weekday() >= 5:
        return False
    return (9, 30) <= (now_ny.hour, now_ny.minute) < (16, 0)


def _parse_ts(s: str) -> datetime:
    return datetime.fromisoformat(s.replace("Z", "+00:00"))


def _alpaca_keys() -> tuple[str, str]:
    key = os.environ.get("ALPACA_KEY_ID") or os.environ.get("ALPACA_LIVE_KEY_ID")
    secret = os.environ.get("ALPACA_SECRET_KEY") or os.environ.get("ALPACA_LIVE_SECRET_KEY")
    if not key or not secret:
        raise LiveQuoteError("no Alpaca keys for the live quote")
    return key, secret


def alpaca_snapshot(symbol: str, timeout: int = 20) -> dict:
    key, secret = _alpaca_keys()
    req = Request(f"{DATA_URL}/v2/stocks/{symbol}/snapshot",
                  headers={"APCA-API-KEY-ID": key, "APCA-API-SECRET-KEY": secret,
                           "Accept": "application/json"})
    try:
        with urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode())
    except Exception as e:
        raise LiveQuoteError(f"Alpaca snapshot {symbol} failed: {e}") from e


def proxy_estimate(snapshot: dict, last_close: float, last_close_date, now_utc: datetime) -> float:
    """Index level now = last complete index close scaled by the ETF's move since its previous close."""
    try:
        trade, prev = snapshot["latestTrade"], snapshot["prevDailyBar"]
        px, prev_close = float(trade["p"]), float(prev["c"])
        trade_t, prev_date = _parse_ts(trade["t"]), _parse_ts(prev["t"]).astimezone(NY).date()
    except (KeyError, TypeError, ValueError) as e:
        raise LiveQuoteError(f"unexpected snapshot format: {e!r}") from e
    age = (now_utc - trade_t).total_seconds()
    if age > MAX_QUOTE_AGE_S:
        raise LiveQuoteError(f"ETF quote is {age:.0f}s old (market closed or feed stale)")
    if prev_date != last_close_date:
        raise LiveQuoteError(f"ETF previous close is {prev_date}, index history ends {last_close_date}")
    if px <= 0 or prev_close <= 0:
        raise LiveQuoteError("non-positive ETF price")
    return float(last_close) * px / prev_close


def yfinance_index_price(ticker: str, now_utc: datetime) -> float | None:
    """Last 1-minute bar of the index itself; None if unavailable or stale (cross-check only)."""
    try:
        import yfinance as yf
        df = yf.download(ticker, period="1d", interval="1m", progress=False, threads=False)
        col = df["Close"]
        s = (col[ticker] if isinstance(col, pd.DataFrame) and ticker in col.columns else
             col.iloc[:, 0] if isinstance(col, pd.DataFrame) else col).dropna()
        if s.empty:
            return None
        ts = pd.Timestamp(s.index[-1])
        ts = ts.tz_localize("UTC") if ts.tzinfo is None else ts.tz_convert("UTC")
        if (now_utc - ts.to_pydatetime()).total_seconds() > MAX_YF_AGE_S:
            return None
        return float(s.iloc[-1])
    except Exception as e:  # cross-check must never break the run by itself
        log.warning(f"yfinance live price {ticker} unavailable: {e}")
        return None


def combine_estimates(primary: float | None, secondary: float | None, tol: float = CROSSCHECK_TOL) -> tuple[float, str]:
    """Pick the live price; refuse it if two sources disagree. Returns (price, source description)."""
    if primary is None and secondary is None:
        raise LiveQuoteError("no live quote available")
    if primary is not None and secondary is not None:
        dev = primary / secondary - 1
        if abs(dev) > tol:
            raise LiveQuoteError(f"live sources disagree by {dev:+.2%} (ETF proxy {primary:.2f}, index {secondary:.2f})")
        return primary, f"ETF proxy, cross-checked with index ({dev:+.2%})"
    if primary is not None:
        return primary, "ETF proxy only (no cross-check)"
    return secondary, "index only (no ETF proxy)"


def estimate_index_close(sleeve: dict, last_close: float, last_close_date, now_utc: datetime) -> dict:
    """Live index level for one sleeve: {"price", "source", "quote_utc"}."""
    primary = None
    try:
        primary = proxy_estimate(alpaca_snapshot(sleeve["fallback_symbol"]), last_close, last_close_date, now_utc)
    except LiveQuoteError as e:
        log.warning(f"{sleeve['name']}: ETF proxy failed: {e}")
    secondary = yfinance_index_price(sleeve["signal_ticker"], now_utc)
    price, how = combine_estimates(primary, secondary)
    return {"price": round(price, 2), "source": how, "quote_utc": now_utc.strftime("%Y-%m-%d %H:%M UTC")}


def splice_live_bar(close: pd.Series, price: float, now_ny: datetime) -> pd.Series:
    """Append today's live price as the last bar (today's incomplete daily bar must already be removed)."""
    today = pd.Timestamp(now_ny.date())
    if len(close) and close.index[-1] >= today:
        raise ValueError("today's bar is already in the series")
    out = close.copy()
    out.loc[today] = float(price)
    return out
