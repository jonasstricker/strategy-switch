"""
Core strategy (live): S&P 500 + Nasdaq-100 trend filter, leveraged only in uptrends.

Two sleeves with a fixed share of equity (50/50); each follows only its own index.
Rule (fixed a priori, validated 1988-2026 and with real ETFs 2007-2026,
see research/evaluate_core.py and research/evaluate_universe.py):
  - Signal series: index close (^GSPC resp. ^NDX) and its 200-day SMA.
  - RISK_ON  when close > SMA * (1 + band_in)   (re-entry needs +1 % confirmation)
  - RISK_OFF when close < SMA * (1 - band_out)  (exit as soon as close < SMA)
  - otherwise the previous state is kept.
  - Decided with data up to the close of day t only; traded on day t+1.
  - RISK_ON  -> sleeve exposure = leverage via 1x (SPY/QQQ) / 2x daily (SSO/QLD)
  - RISK_OFF -> sleeve in the T-bill ETF (SGOV)
  - Portfolio signal: RISK_ON (all sleeves on), PARTIAL, RISK_OFF (all off).

Run as module to write docs/data/core_signal.json (exit code 1 if the data is
stale or invalid, so the workflow stops before any order is sent):
    python -m wf_backtest.core_strategy
"""
from __future__ import annotations

import json
import logging
import os
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from urllib.request import Request, urlopen
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

log = logging.getLogger("core_strategy")

ROOT = Path(__file__).resolve().parent.parent
CORE_SIGNAL_FILE = ROOT / "docs" / "data" / "core_signal.json"
NY = ZoneInfo("America/New_York")

CONFIG = {
    # fallback_symbol: Alpaca daily bars if yfinance fails; weights must sum to 1
    "sleeves": [
        {"name": "S&P 500", "signal_ticker": "^GSPC", "fallback_symbol": "SPY",
         "etf_1x": "SPY", "etf_2x": "SSO", "weight": 0.5},
        {"name": "Nasdaq-100", "signal_ticker": "^NDX", "fallback_symbol": "QQQ",
         "etf_1x": "QQQ", "etf_2x": "QLD", "weight": 0.5},
    ],
    "sma_days": 200,
    "band_in": 0.01,
    "band_out": 0.0,
    "leverage": 2.0,               # per sleeve: 1.0 = 1x ETF, 1.5 = 1x/2x 50/50, 2.0 = 2x ETF only
    "etf_cash": "SGOV",
    "etf_cash_hist": "BIL",        # backtest proxy (SGOV exists only since 2020)
    "benchmark": "SPY",            # buy & hold comparison in the backtest
    "cash_buffer": 0.01,           # keep 1 % uninvested for price moves / fees
    "max_data_age_days": 4,        # weekend + one holiday
    "history_start": "2000-01-01",
    "backtest_start": "2007-06-01",
    "cost_per_turnover": 0.0005,
}


# ── Rule ────────────────────────────────────────────────────────────────────

def trend_state(close: pd.Series, sma_days: int, band_in: float, band_out: float) -> pd.Series:
    """1.0 = RISK_ON, 0.0 = RISK_OFF at each close (uses data up to that close only)."""
    sma = close.rolling(sma_days).mean().to_numpy()
    px = close.to_numpy(dtype=float)
    out = np.zeros(len(px))
    st = 0.0
    for i in range(len(px)):
        s = sma[i]
        if not np.isnan(s):
            if st == 0.0 and px[i] > s * (1 + band_in):
                st = 1.0
            elif st == 1.0 and px[i] < s * (1 - band_out):
                st = 0.0
        out[i] = st
    return pd.Series(out, index=close.index)


def target_weights(states: dict, cfg: dict = CONFIG, cash_symbol: str | None = None) -> dict:
    """Portfolio weights (fractions of equity) for {sleeve name: risk_on}."""
    invest = 1.0 - cfg["cash_buffer"]
    cash_sym = cash_symbol or cfg["etf_cash"]
    lev = float(np.clip(cfg["leverage"], 0.0, 2.0))
    w: dict[str, float] = {}
    for sl in cfg["sleeves"]:
        if not states[sl["name"]]:
            parts = {cash_sym: 1.0}
        elif lev <= 1.0:
            parts = {sl["etf_1x"]: lev, cash_sym: 1.0 - lev}
        else:
            parts = {sl["etf_1x"]: 2.0 - lev, sl["etf_2x"]: lev - 1.0}
        for k, v in parts.items():
            w[k] = w.get(k, 0.0) + v * sl["weight"]
    return {k: round(v * invest, 6) for k, v in w.items() if v > 1e-9}


# ── Data ────────────────────────────────────────────────────────────────────

def _drop_incomplete_bar(close: pd.Series, now_ny: datetime) -> pd.Series:
    """Remove today's bar while the US session is still running."""
    if len(close) and close.index[-1].date() == now_ny.date() and \
            (now_ny.hour, now_ny.minute) < (16, 10):
        return close.iloc[:-1]
    return close


def _yf_ohlc(tickers, start: str, auto_adjust: bool, retries: int = 3) -> pd.DataFrame:
    import yfinance as yf
    last = None
    for attempt in range(1, retries + 1):
        try:
            df = yf.download(tickers, start=start, auto_adjust=auto_adjust,
                             progress=False, threads=False)
            if df is not None and len(df):
                return df
            last = RuntimeError("empty download")
        except Exception as e:  # network / API changes
            last = e
        log.warning(f"yfinance {tickers} attempt {attempt}/{retries} failed: {last}")
        time.sleep(5 * attempt)
    raise RuntimeError(f"yfinance download failed for {tickers}: {last}")


def _close_series(df: pd.DataFrame, ticker: str) -> pd.Series:
    col = df["Close"]
    if isinstance(col, pd.DataFrame):
        col = col[ticker] if ticker in col.columns else col.iloc[:, 0]
    s = col.dropna().astype(float)
    s.index = pd.DatetimeIndex(s.index).tz_localize(None).normalize()
    return s


def _alpaca_daily_close(symbol: str, start: str) -> pd.Series:
    key = os.environ.get("ALPACA_KEY_ID") or os.environ.get("ALPACA_LIVE_KEY_ID")
    secret = os.environ.get("ALPACA_SECRET_KEY") or os.environ.get("ALPACA_LIVE_SECRET_KEY")
    if not key or not secret:
        raise RuntimeError("no Alpaca keys for fallback data")
    end = (datetime.now(timezone.utc) - timedelta(minutes=20)).strftime("%Y-%m-%dT%H:%M:%SZ")
    rows, token = [], None
    while True:
        url = (f"https://data.alpaca.markets/v2/stocks/{symbol}/bars?timeframe=1Day"
               f"&start={start}T00:00:00Z&end={end}&adjustment=raw&feed=sip&limit=10000")
        if token:
            url += f"&page_token={token}"
        req = Request(url, headers={"APCA-API-KEY-ID": key, "APCA-API-SECRET-KEY": secret,
                                    "Accept": "application/json"})
        with urlopen(req, timeout=30) as resp:
            data = json.loads(resp.read().decode())
        rows += data.get("bars") or []
        token = data.get("next_page_token")
        if not token:
            break
    if not rows:
        raise RuntimeError("Alpaca returned no bars")
    idx = pd.to_datetime([b["t"] for b in rows], utc=True).tz_convert(NY).tz_localize(None).normalize()
    return pd.Series([float(b["c"]) for b in rows], index=idx).sort_index()


def load_signal_close(sleeve: dict, cfg: dict = CONFIG,
                      now_ny: datetime | None = None) -> tuple[pd.Series, str]:
    now_ny = now_ny or datetime.now(NY)
    ticker, fallback = sleeve["signal_ticker"], sleeve["fallback_symbol"]
    try:
        df = _yf_ohlc(ticker, cfg["history_start"], auto_adjust=False)
        close, source = _close_series(df, ticker), f"{ticker} (yfinance)"
    except Exception as e:
        log.warning(f"Primary data failed ({e}) -> fallback Alpaca {fallback}")
        close = _alpaca_daily_close(fallback, cfg["history_start"])
        source = f"{fallback} (Alpaca fallback)"
    return _drop_incomplete_bar(close, now_ny), source


def validate_close(close: pd.Series, cfg: dict, now_ny: datetime) -> str | None:
    """Return an error message if the data must not be traded on."""
    if len(close) < cfg["sma_days"] + 50:
        return f"too little history ({len(close)} bars)"
    if not np.isfinite(close.iloc[-cfg["sma_days"]:]).all():
        return "NaN/inf in recent closes"
    age = (now_ny.date() - close.index[-1].date()).days
    if age > cfg["max_data_age_days"]:
        return f"stale data: last close {close.index[-1].date()} is {age} days old"
    jumps = close.pct_change().iloc[-cfg["sma_days"]:].abs()
    if (jumps > 0.25).any():
        return f"implausible daily move {jumps.max():.1%} on {jumps.idxmax().date()}"
    return None


# ── Backtest with the traded ETFs (decide at close t, trade at open t+1) ──────

def backtest_real(states: dict, cfg: dict = CONFIG) -> dict | None:
    """states: {sleeve name: trend_state series}; rebalanced daily to the target weights."""
    syms = list(dict.fromkeys([cfg["etf_cash_hist"], cfg["benchmark"]] +
                              [sl[k] for sl in cfg["sleeves"] for k in ("etf_1x", "etf_2x")]))
    try:
        df = _yf_ohlc(syms, cfg["backtest_start"], auto_adjust=True)
    except Exception as e:
        log.warning(f"Backtest data unavailable: {e}")
        return None
    o, c = df["Open"][syms].dropna(), df["Close"][syms].dropna()
    idx = o.index.intersection(c.index)
    idx = idx[idx >= pd.Timestamp(cfg["backtest_start"])]
    o, c = o.loc[idx], c.loc[idx]
    gap = (o / c.shift(1) - 1).fillna(0.0).to_numpy()
    intra = (c / o - 1).fillna(0.0).to_numpy()

    names = [sl["name"] for sl in cfg["sleeves"]]
    st = np.column_stack([states[n].reindex(idx, method="ffill").fillna(0.0).to_numpy() > 0 for n in names])
    keys = [tuple(map(bool, row)) for row in st]
    bt_cfg = {**cfg, "cash_buffer": 0.0}
    combos = {k: [target_weights(dict(zip(names, k)), bt_cfg, cfg["etf_cash_hist"]).get(s, 0.0) for s in syms]
              for k in set(keys)}
    W_dec = np.array([combos[k] for k in keys])
    W_new = np.vstack([np.zeros((1, len(syms))), W_dec[:-1]])     # held from open t+1
    W_old = np.vstack([np.zeros((1, len(syms))), W_new[:-1]])
    turn = np.abs(np.diff(W_new, axis=0, prepend=W_new[:1])).sum(1)
    strat = (1 + (W_old * gap).sum(1)) * (1 + (W_new * intra).sum(1)) - 1 - turn * cfg["cost_per_turnover"]
    strat = pd.Series(strat, index=idx).iloc[1:]
    bh = c[cfg["benchmark"]].pct_change().loc[strat.index].fillna(0.0)

    def _m(x):
        eq = (1 + x).cumprod()
        yrs = len(x) / 252
        return {"cagr": float(eq.iloc[-1] ** (1 / yrs) - 1),
                "vol": float(x.std() * np.sqrt(252)),
                "max_dd": float((eq / eq.cummax() - 1).min()),
                "worst_year": float(((1 + x).groupby(x.index.year).prod() - 1).min())}

    eq_s, eq_b = (1 + strat).cumprod(), (1 + bh).cumprod()
    step = max(1, len(eq_s) // 250)
    pts = list(range(0, len(eq_s), step)) + list(range(max(0, len(eq_s) - 60), len(eq_s)))
    equity = [{"date": eq_s.index[i].strftime("%Y-%m-%d"), "switch": round(float(eq_s.iloc[i]), 4),
               "bh": round(float(eq_b.iloc[i]), 4), "cash": round(float(W_new[i + 1][0]), 2)}
              for i in sorted(set(pts))]
    ys = (1 + strat).groupby(strat.index.year).prod() - 1
    yb = (1 + bh).groupby(bh.index.year).prod() - 1
    return {
        "period": f"{strat.index[0].date()} .. {strat.index[-1].date()}",
        "benchmark": cfg["benchmark"],
        "strategy": _m(strat),
        "buy_hold": _m(bh),
        "switches_per_year": float((np.diff(st.astype(int), axis=0) != 0).sum() / (len(st) / 252)),
        "equity": equity,
        "yearly": [{"year": int(y), "switch": round(float(ys[y]) * 100, 1), "bh": round(float(yb[y]) * 100, 1)}
                   for y in ys.index],
    }


# ── Signal ──────────────────────────────────────────────────────────────────

def compute_core_signal(cfg: dict = CONFIG, now_ny: datetime | None = None,
                        with_backtest: bool = True) -> dict:
    now_ny = now_ny or datetime.now(NY)
    out = {"status": "error", "generated_utc": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
           "config": cfg}
    sleeves, states, history = [], {}, []
    for sl in cfg["sleeves"]:
        try:
            close, source = load_signal_close(sl, cfg, now_ny)
        except Exception as e:
            out["error"] = f"{sl['name']}: data load failed: {e}"
            return out
        err = validate_close(close, cfg, now_ny)
        if err:
            out["error"] = f"{sl['name']}: {err}"
            out["as_of"] = close.index[-1].strftime("%Y-%m-%d") if len(close) else None
            return out

        state = trend_state(close, cfg["sma_days"], cfg["band_in"], cfg["band_out"])
        sma = close.rolling(cfg["sma_days"]).mean()
        changes = state[state.diff().fillna(0) != 0]
        last_switch = changes.index[-1] if len(changes) else state.index[0]
        states[sl["name"]] = state
        history += [{"date": t.strftime("%Y-%m-%d"), "sleeve": sl["name"],
                     "signal": "RISK_ON" if v else "RISK_OFF"} for t, v in changes.iloc[-12:].items()]
        sleeves.append({
            "name": sl["name"],
            "weight": sl["weight"],
            "signal": "RISK_ON" if state.iloc[-1] > 0 else "RISK_OFF",
            "as_of": close.index[-1].strftime("%Y-%m-%d"),
            "source": source,
            "index": {
                "close": round(float(close.iloc[-1]), 2),
                "sma": round(float(sma.iloc[-1]), 2),
                "dist_pct": round(float(close.iloc[-1] / sma.iloc[-1] - 1) * 100, 2),
                "entry_level": round(float(sma.iloc[-1] * (1 + cfg["band_in"])), 2),
                "exit_level": round(float(sma.iloc[-1] * (1 - cfg["band_out"])), 2),
            },
            "last_switch": last_switch.strftime("%Y-%m-%d"),
            "days_in_state": int((state.index > last_switch).sum()),
        })

    on = {s["name"]: s["signal"] == "RISK_ON" for s in sleeves}
    dates = sorted({s["as_of"] for s in sleeves})
    if len(dates) > 1:
        log.warning(f"Last closes differ between sleeves: {dates} -> as_of = oldest")
    out.update({
        "status": "ok",
        "error": None,
        "as_of": dates[0],
        "source": ", ".join(s["source"] for s in sleeves),
        "signal": "RISK_ON" if all(on.values()) else "RISK_OFF" if not any(on.values()) else "PARTIAL",
        "exposure": round(float(cfg["leverage"]) * sum(sl["weight"] for sl in cfg["sleeves"] if on[sl["name"]]), 4),
        "weights": target_weights(on, cfg),
        "sleeves": sleeves,
        "switch_history": sorted(history, key=lambda h: h["date"])[-12:],
    })
    if with_backtest:
        out["backtest"] = backtest_real(states, cfg)
    return out


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    sig = compute_core_signal()
    CORE_SIGNAL_FILE.parent.mkdir(parents=True, exist_ok=True)
    if sig["status"] != "ok" and CORE_SIGNAL_FILE.exists():
        # keep the last good signal for the dashboard, flag the failure
        try:
            prev = json.loads(CORE_SIGNAL_FILE.read_text(encoding="utf-8"))
            prev.update({"status": "error", "error": sig.get("error"), "generated_utc": sig["generated_utc"]})
            sig = prev
        except Exception:
            pass
    CORE_SIGNAL_FILE.write_text(json.dumps(sig, indent=1, ensure_ascii=False), encoding="utf-8")
    if sig["status"] != "ok":
        log.error(f"CORE SIGNAL INVALID: {sig.get('error')}")
        return 1
    log.info(f"Core signal {sig['signal']} as of {sig['as_of']} -> weights {sig['weights']}")
    for s in sig["sleeves"]:
        ix = s["index"]
        log.info(f"  {s['name']}: {s['signal']} ({s['source']}) close {ix['close']} vs SMA {ix['sma']} "
                 f"({ix['dist_pct']:+.2f}%)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
