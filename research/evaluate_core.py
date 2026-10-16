"""
Reproducible evaluation of the live core rule (wf_backtest/core_strategy.py).

    python research/evaluate_core.py            # S&P 500 1960-today, real ETFs, bootstrap
    python research/evaluate_core.py --world    # + 10 other equity indices

Index model: S&P 500 total return (^SP500TR from 1988, ^GSPC + 4 %/yr dividends before),
cash = 3M T-bill (^IRX) - 0.10 %/yr, SPY fee 0.09 %/yr, 2x ETF = 2*SPY - T-bill - 2.0 %/yr
(drag fitted to SSO 2006-2026), 5 bp cost per unit of turnover.
lag 1 = trade at the deciding close (optimistic), lag 2 = one day later (conservative);
the live system trades at the next open, which backtest_real() simulates with the real ETFs.
"""
from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from wf_backtest.core_strategy import CONFIG, _close_series, _yf_ohlc, backtest_real, trend_state  # noqa: E402

warnings.filterwarnings("ignore")

SPY_FEE, TBILL_FEE, DRAG_2X, DIV_PRE_1988 = 0.0009, 0.0010, 0.020, 0.04
COST = CONFIG["cost_per_turnover"]
RULE = dict(sma_days=CONFIG["sma_days"], band_in=CONFIG["band_in"], band_out=CONFIG["band_out"])
LEVELS = (1.0, 1.5, 2.0)
WORLD = {"^NDX": ("Nasdaq 100", 0.010), "^DJI": ("Dow Jones", 0.023), "^RUT": ("Russell 2000", 0.015),
         "^GDAXI": ("DAX (TR)", 0.0), "^N225": ("Nikkei 225", 0.015), "^FTSE": ("FTSE 100", 0.037),
         "^HSI": ("Hang Seng", 0.034), "^GSPTSE": ("TSX", 0.028), "^STOXX50E": ("Euro Stoxx 50", 0.034),
         "^AXJO": ("ASX 200", 0.043)}


def _close(ticker: str, start: str) -> pd.Series:
    return _close_series(_yf_ohlc(ticker, start, auto_adjust=True), ticker)


def net_returns(E_dec, r: np.ndarray, rf: np.ndarray, lag: int = 1) -> np.ndarray:
    """Daily net returns when the exposure decided at close t is held from close t+lag."""
    E = np.concatenate([np.zeros(lag), np.asarray(E_dec, dtype=float)[:-lag]])
    w_2x = np.clip(E - 1, 0, None)
    w_spy = np.minimum(E, 1) - w_2x
    w_cash = np.clip(1 - E, 0, None)
    r_spy = r - SPY_FEE / 252
    gross = w_cash * (rf - TBILL_FEE / 252) + w_spy * r_spy + w_2x * (2 * r_spy - rf - DRAG_2X / 252)
    W = np.vstack([w_cash, w_spy, w_2x])
    return gross - np.abs(np.diff(W, axis=1, prepend=W[:, :1])).sum(axis=0) * COST


def cagr_dd(x: np.ndarray) -> tuple[float, float]:
    eq = np.cumprod(1 + x)
    return eq[-1] ** (252 / len(x)) - 1, (eq / np.maximum.accumulate(eq) - 1).min()


def load_index() -> tuple[pd.Series, pd.Series, pd.Series]:
    px = _close("^GSPC", "1955-01-01")
    tr = _close("^SP500TR", "1987-12-01").pct_change().reindex(px.index)
    rf = (_close("^IRX", "1959-01-01") / 100).reindex(px.index).ffill().shift(1).fillna(0.0) / 252
    r = px.pct_change().fillna(0.0) + DIV_PRE_1988 / 252
    use_tr = tr.notna() & (px.index >= pd.Timestamp("1988-01-05"))
    r[use_tr] = tr[use_tr]
    return px, r, rf


def report_index(px: pd.Series, r: pd.Series, rf: pd.Series) -> None:
    st = trend_state(px, **RULE).to_numpy()
    variants = {"Buy & Hold": (np.ones(len(px)), 1), **{f"Trend {L:g}x": (L * st, 1) for L in LEVELS},
                "Trend 2x lag2": (2 * st, 2)}
    nets = {k: pd.Series(net_returns(E, r.to_numpy(), rf.to_numpy(), lag), index=px.index)
            for k, (E, lag) in variants.items()}
    print("\n== S&P 500 index model: CAGR / MaxDD / worst calendar year ==")
    for a, b in (("1960-01-05", "1987-12-31"), ("1988-01-01", None), ("1960-01-05", None)):
        print(f"  {a[:4]}-{px.index[-1].year if b is None else b[:4]}")
        for k, x in nets.items():
            s = x.loc[a:b]
            c, dd = cagr_dd(s.to_numpy())
            wy = ((1 + s).groupby(s.index.year).prod() - 1).min()
            print(f"    {k:14s} {c * 100:6.2f}% {dd * 100:7.1f}% {wy * 100:7.1f}%")

    print("\n== Rolling 10y windows since 1960 (monthly starts): excess CAGR vs Buy & Hold ==")
    bh = nets["Buy & Hold"].loc["1960-01-05":]
    starts = bh.resample("MS").first().index
    starts = starts[starts + pd.DateOffset(years=10) <= bh.index[-1]]
    for k in [f"Trend {L:g}x" for L in LEVELS]:
        x = nets[k]
        ex = np.array([cagr_dd(x.loc[s:s + pd.DateOffset(years=10)].to_numpy())[0]
                       - cagr_dd(bh.loc[s:s + pd.DateOffset(years=10)].to_numpy())[0] for s in starts]) * 100
        print(f"  {k:14s} wins {np.mean(ex > 0) * 100:4.0f}% of {len(ex)}  median {np.median(ex):+5.2f}  "
              f"p10 {np.percentile(ex, 10):+5.2f}  p90 {np.percentile(ex, 90):+5.2f} %-points/yr")


def report_real(px: pd.Series) -> None:
    print("\n== Real ETFs (BIL/SPY/SSO, decide at close, trade at next open, 5 bp) ==")
    st = trend_state(px, **RULE)
    sp = {**CONFIG["sleeves"][0], "weight": 1.0}
    for L in LEVELS:
        bt = backtest_real({sp["name"]: st}, {**CONFIG, "sleeves": [sp], "leverage": L})
        if bt is None:
            print("  ETF data unavailable")
            return
        s = bt["strategy"]
        print(f"  Trend {L:g}x       CAGR {s['cagr'] * 100:6.2f}%  MaxDD {s['max_dd'] * 100:6.1f}%  "
              f"Vol {s['vol'] * 100:5.1f}%  worst year {s['worst_year'] * 100:6.1f}%")
    b = bt["buy_hold"]
    print(f"  Buy & Hold SPY  CAGR {b['cagr'] * 100:6.2f}%  MaxDD {b['max_dd'] * 100:6.1f}%  "
          f"Vol {b['vol'] * 100:5.1f}%  worst year {b['worst_year'] * 100:6.1f}%   ({bt['period']})")


def report_bootstrap(px: pd.Series, r: pd.Series, rf: pd.Series, paths: int, years: int = 10,
                     block: int = 250, warm: int = 260, seed: int = 42) -> None:
    rng = np.random.default_rng(seed)
    base = pd.DataFrame({"r": r, "rpx": px.pct_change(), "rf": rf}).loc["1960-01-05":].dropna()
    R, RPX, RF = (base[c].to_numpy() for c in ("r", "rpx", "rf"))
    n = years * 252 + warm

    def sample() -> np.ndarray:
        out, i = np.empty(n, dtype=int), 0
        while i < n:
            s = rng.integers(0, len(R))
            k = min(rng.geometric(1 / block), n - i)
            out[i:i + k] = (s + np.arange(k)) % len(R)
            i += k
        return out

    for label, delta in (("historical equity premium", 0.0), ("equity premium -3 %/yr", 0.03)):
        res = {k: [] for k in ["Buy & Hold", *[f"Trend {L:g}x" for L in LEVELS]]}
        for _ in range(paths):
            ix = sample()
            rr, rff = R[ix] - delta / 252, RF[ix]
            st = trend_state(pd.Series(np.cumprod(1 + RPX[ix] - delta / 252)), **RULE).to_numpy()
            res["Buy & Hold"].append(cagr_dd(net_returns(np.ones(n), rr, rff)[warm:]))
            for L in LEVELS:
                res[f"Trend {L:g}x"].append(cagr_dd(net_returns(L * st, rr, rff)[warm:]))
        bh = np.array(res["Buy & Hold"])
        print(f"\n== {years}y block bootstrap, {label} ({paths} paths) ==")
        print("                  CAGR p10    p50    p90 | MaxDD p50 | P(beat B&H) P(DD<-50%)")
        for k, v in res.items():
            v = np.array(v)
            beat = "" if k == "Buy & Hold" else f"{(v[:, 0] > bh[:, 0]).mean() * 100:.0f}%"
            print(f"  {k:14s} {np.percentile(v[:, 0], 10) * 100:7.1f}% {np.median(v[:, 0]) * 100:5.1f}% "
                  f"{np.percentile(v[:, 0], 90) * 100:5.1f}% | {np.median(v[:, 1]) * 100:8.1f}% | "
                  f"{beat:>11s} {(v[:, 1] < -0.5).mean() * 100:9.0f}%")


def report_world(rf_us: pd.Series) -> None:
    print("\n== Other markets, same rule, price index + dividend estimate, US T-bill as cash ==")
    raw = _yf_ohlc(list(WORLD), "1960-01-01", auto_adjust=True)["Close"]
    raw.index = pd.DatetimeIndex(raw.index).tz_localize(None).normalize()
    ex = {L: [] for L in LEVELS}
    for t, (name, div) in WORLD.items():
        px = raw[t].dropna()
        r = (px.pct_change().fillna(0.0) + div / 252).to_numpy()
        rf = rf_us.reindex(px.index).ffill().fillna(0.0).to_numpy()
        st = trend_state(px, **RULE).to_numpy()
        a = max(px.index[260], pd.Timestamp("1990-01-01"))
        sl = px.index >= a
        c_bh, dd_bh = cagr_dd(net_returns(np.ones(len(px)), r, rf)[sl])
        line = f"  {name:14s} {a.year}-  B&H {c_bh * 100:5.1f}%/{dd_bh * 100:4.0f}%"
        for L in LEVELS:
            c, dd = cagr_dd(net_returns(L * st, r, rf)[sl])
            ex[L].append((c - c_bh) * 100)
            line += f" | {L:g}x {(c - c_bh) * 100:+5.1f}/{dd * 100:4.0f}%"
        print(line)
    for L, v in ex.items():
        print(f"  Trend {L:g}x: mean excess {np.mean(v):+.2f} %-points/yr, wins {sum(x > 0 for x in v)}/{len(v)}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--world", action="store_true", help="also test 10 other equity indices")
    ap.add_argument("--paths", type=int, default=2000, help="bootstrap paths (0 = skip)")
    args = ap.parse_args()
    print(f"Rule: SMA{RULE['sma_days']}, enter above SMA*(1+{RULE['band_in']:g}), "
          f"exit below SMA*(1-{RULE['band_out']:g}); live leverage {CONFIG['leverage']:g}x")
    px, r, rf = load_index()
    report_index(px, r, rf)
    report_real(px)
    if args.paths:
        report_bootstrap(px, r, rf, args.paths)
    if args.world:
        report_world(rf)


if __name__ == "__main__":
    main()
