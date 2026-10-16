"""
Bias-free comparison of exposure rules: Sharpe, CAGR, drawdown, robustness.

    python research/compare_variants.py [--paths 1000] [--no-world]

All variants are fixed in advance with textbook parameters (no parameter search):
  B&H Lx        constant exposure L (SPY / SSO mix; B&H 2x = holding SSO)
  Trend Lx      live core rule (SMA200, +1 % entry band), exposure L while RISK_ON
  VolT s        exposure = min(2, s / realized 20-day vol), always invested
  Trend+VolT s  core trend state * min(2, s / realized 20-day vol)
  Ens+VolT s    share of 5 trend rules (price > SMA 50/100/150/200/250) * min(2, s / vol)
Continuous exposures are re-traded only when they move by more than 0.1.
Cost model and data as in evaluate_core.py.
"""
from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE.parent)]
from evaluate_core import COST, RULE, WORLD, load_index, net_returns  # noqa: E402
from wf_backtest.core_strategy import _yf_ohlc, trend_state  # noqa: E402

warnings.filterwarnings("ignore")
SQ, BAND, VOL_DAYS, CAP = np.sqrt(252), 0.1, 20, 2.0
VARIANTS = {
    "B&H 1x": ("bh", 1.0), "B&H 1.5x": ("bh", 1.5), "B&H 2x": ("bh", 2.0),
    "Trend 1x": ("trend", 1.0), "Trend 1.5x": ("trend", 1.5), "Trend 2x": ("trend", 2.0),
    "VolT 16%": ("volt", 0.16),
    "Trend+VolT 12%": ("trend_volt", 0.12), "Trend+VolT 16%": ("trend_volt", 0.16),
    "Trend+VolT 20%": ("trend_volt", 0.20),
    "Ens+VolT 16%": ("ens_volt", 0.16), "Ens+VolT 20%": ("ens_volt", 0.20),
}
HDR = f"    {'':16s} {'CAGR':>7s} {'Vol':>6s} {'Sharpe':>6s} {'MaxDD':>7s} {'Calmar':>6s} {'AvgExp':>6s} {'Trades/y':>8s}"


def banded(E: np.ndarray) -> np.ndarray:
    """Re-trade only if the exposure moves by more than BAND (entries/exits always)."""
    out, cur = np.empty(len(E)), 0.0
    for i, e in enumerate(E):
        if abs(e - cur) > BAND or (e == 0.0) != (cur == 0.0):
            cur = e
        out[i] = cur
    return out


def exposures(px: pd.Series, names=tuple(VARIANTS)) -> dict[str, np.ndarray]:
    """Exposure (0..2) decided at each close, using data up to that close only."""
    one = np.ones(len(px))
    st = trend_state(px, **RULE).to_numpy()
    vol = px.pct_change().rolling(VOL_DAYS).std().to_numpy() * SQ
    ens = np.mean([(px > px.rolling(n).mean()).to_numpy() for n in (50, 100, 150, 200, 250)], axis=0)
    base = {"volt": one, "trend_volt": st, "ens_volt": ens}
    out = {}
    for k in names:
        kind, p = VARIANTS[k]
        if kind == "bh":
            out[k] = p * one
        elif kind == "trend":
            out[k] = p * st
        else:
            out[k] = banded(base[kind] * np.nan_to_num(np.clip(p / vol, 0, CAP)))
    return out


def stats(x: np.ndarray, rf: np.ndarray) -> dict:
    eq = np.cumprod(1 + x)
    ex = x - rf
    return {"cagr": eq[-1] ** (252 / len(x)) - 1, "vol": x.std() * SQ,
            "sharpe": ex.mean() / ex.std() * SQ, "dd": (eq / np.maximum.accumulate(eq) - 1).min()}


def row(name: str, x: np.ndarray, rf: np.ndarray, E: np.ndarray) -> str:
    s = stats(x, rf)
    trades = (np.abs(np.diff(E)) > 1e-12).sum() / (len(x) / 252)
    return (f"    {name:16s} {s['cagr'] * 100:6.2f}% {s['vol'] * 100:5.1f}% {s['sharpe']:6.2f} "
            f"{s['dd'] * 100:6.1f}% {s['cagr'] / -s['dd']:6.2f} {E.mean():6.2f} {trades:8.1f}")


def block_index(n_out: int, n_pool: int, rng, block: int = 250) -> np.ndarray:
    out, i = np.empty(n_out, dtype=int), 0
    while i < n_out:
        s = rng.integers(0, n_pool)
        k = min(rng.geometric(1 / block), n_out - i)
        out[i:i + k] = (s + np.arange(k)) % n_pool
        i += k
    return out


def report_index(px: pd.Series, r: pd.Series, rf: pd.Series, E: dict) -> dict:
    R, RF = r.to_numpy(), rf.to_numpy()
    nets = {k: net_returns(v, R, RF) for k, v in E.items()}
    for a, b, label in (("1960-01-05", "1989-12-31", "design period"), ("1990-01-01", None, "holdout"),
                        ("1960-01-05", None, "full")):
        m = (px.index >= a) & ((px.index <= b) if b else True)
        print(f"\n== S&P 500 index model {a[:4]}-{b[:4] if b else px.index[-1].year} ({label}) ==")
        print(HDR)
        for k in E:
            print(row(k, nets[k][m], RF[m], E[k][m]))
    m = px.index >= "1960-01-05"
    print("\n  one day later execution (lag 2), full period Sharpe / CAGR / MaxDD:")
    for k in ("Trend 1.5x", "Trend 2x", "Trend+VolT 16%", "Trend+VolT 20%", "Ens+VolT 16%"):
        s = stats(net_returns(E[k], R, RF, lag=2)[m], RF[m])
        print(f"    {k:16s} {s['sharpe']:.2f} / {s['cagr'] * 100:.2f}% / {s['dd'] * 100:.1f}%")
    return nets


def report_sharpe_significance(px: pd.Series, rf: pd.Series, nets: dict, reps: int = 1000) -> None:
    m = px.index >= "1960-01-05"
    X = {k: v[m] - rf.to_numpy()[m] for k, v in nets.items()}
    rng = np.random.default_rng(7)
    d = {k: [] for k in X if k != "B&H 1x"}
    for _ in range(reps):
        ix = block_index(m.sum(), m.sum(), rng)
        xb = X["B&H 1x"][ix]
        b = xb.mean() / xb.std()
        for k in d:
            xk = X[k][ix]
            d[k].append((xk.mean() / xk.std() - b) * SQ)
    print(f"\n== Sharpe difference vs B&H 1x, 1960-today, paired block bootstrap ({reps}x) ==")
    for k, v in d.items():
        v = np.array(v)
        print(f"    {k:16s} {np.median(v):+.2f}  90% CI [{np.percentile(v, 5):+.2f}, {np.percentile(v, 95):+.2f}]"
              f"  P(>0) {np.mean(v > 0) * 100:3.0f}%")


def report_etf(px: pd.Series, rf: pd.Series, E: dict) -> None:
    syms = ["BIL", "SPY", "SSO"]
    df = _yf_ohlc(syms, "2007-06-01", auto_adjust=True)
    o, c = df["Open"][syms].dropna(), df["Close"][syms].dropna()
    idx = o.index.intersection(c.index)
    o, c = o.loc[idx], c.loc[idx]
    gap, intra = (o / c.shift(1) - 1).fillna(0.0).to_numpy(), (c / o - 1).fillna(0.0).to_numpy()
    rf_e = rf.reindex(idx).ffill().fillna(0.0).to_numpy()
    print(f"\n== Real ETFs BIL/SPY/SSO {idx[0].date()} .. {idx[-1].date()}, decided at close, traded next open ==")
    print(HDR)
    for k, v in E.items():
        e = pd.Series(v, index=px.index).reindex(idx).ffill().fillna(0.0).to_numpy()
        lev = np.clip(e - 1, 0, None)
        W = np.column_stack([np.clip(1 - e, 0, None), np.minimum(e, 1) - lev, lev])
        W_new = np.vstack([np.zeros((1, 3)), W[:-1]])
        W_old = np.vstack([np.zeros((1, 3)), W_new[:-1]])
        turn = np.abs(np.diff(W_new, axis=0, prepend=W_new[:1])).sum(1)
        x = (1 + (W_old * gap).sum(1)) * (1 + (W_new * intra).sum(1)) - 1 - turn * COST
        print(row(k, x[2:], rf_e[2:], e[2:]))


def report_world(rf_us: pd.Series) -> None:
    names = ["B&H 1x", "Trend 1x", "Trend 2x", "VolT 16%", "Trend+VolT 16%", "Ens+VolT 16%"]
    raw = _yf_ohlc(list(WORLD), "1960-01-01", auto_adjust=True)["Close"]
    raw.index = pd.DatetimeIndex(raw.index).tz_localize(None).normalize()
    print("\n== Other markets (from 1990 or first year), Sharpe, US T-bill as cash ==")
    print(f"    {'':14s}" + "".join(f"{n:>15s}" for n in names))
    sh = {n: [] for n in names}
    for t, (label, div) in WORLD.items():
        px = raw[t].dropna()
        r = (px.pct_change().fillna(0.0) + div / 252).to_numpy()
        rf = rf_us.reindex(px.index).ffill().fillna(0.0).to_numpy()
        E = exposures(px, names)
        m = px.index >= max(px.index[260], pd.Timestamp("1990-01-01"))
        for n in names:
            sh[n].append(stats(net_returns(E[n], r, rf)[m], rf[m])["sharpe"])
        print(f"    {label:14s}" + "".join(f"{sh[n][-1]:15.2f}" for n in names))
    b = np.array(sh["B&H 1x"])
    print(f"    {'mean':14s}" + "".join(f"{np.mean(v):15.2f}" for v in sh.values()))
    print(f"    {'beats B&H':14s}" + "".join(f"{f'{int((np.array(v) > b).sum())}/{len(b)}':>15s}" for v in sh.values()))


def report_paths(px: pd.Series, r: pd.Series, rf: pd.Series, paths: int, years: int = 10, warm: int = 260) -> None:
    names = ["B&H 1x", "B&H 2x", "Trend 1x", "Trend 1.5x", "Trend 2x", "Trend+VolT 16%", "Trend+VolT 20%",
             "Ens+VolT 16%"]
    base = pd.DataFrame({"r": r, "rpx": px.pct_change(), "rf": rf}).loc["1960-01-05":].dropna()
    R, RPX, RF = (base[c].to_numpy() for c in ("r", "rpx", "rf"))
    n = years * 252 + warm
    rng = np.random.default_rng(42)
    for label, delta in (("historical equity premium", 0.0), ("equity premium -3 %/yr", 0.03)):
        res = {k: [] for k in names}
        for _ in range(paths):
            ix = block_index(n, len(R), rng)
            rr, rff = R[ix] - delta / 252, RF[ix]
            E = exposures(pd.Series(np.cumprod(1 + RPX[ix] - delta / 252)), names)
            for k in names:
                s = stats(net_returns(E[k], rr, rff)[warm:], rff[warm:])
                res[k].append((s["cagr"], s["dd"], s["sharpe"]))
        b = np.array(res["B&H 1x"])
        print(f"\n== {years}y block bootstrap, {label} ({paths} paths) ==")
        print("                     CAGR p10   p50   p90 | MaxDD p50 P(<-50%) | Sharpe p50 P(>B&H) | P(CAGR>B&H)")
        for k in names:
            v = np.array(res[k])
            print(f"    {k:16s} {np.percentile(v[:, 0], 10) * 100:8.1f}% {np.median(v[:, 0]) * 100:5.1f}% "
                  f"{np.percentile(v[:, 0], 90) * 100:5.1f}% | {np.median(v[:, 1]) * 100:8.1f}% "
                  f"{(v[:, 1] < -0.5).mean() * 100:7.0f}% | {np.median(v[:, 2]):10.2f} "
                  f"{(v[:, 2] > b[:, 2]).mean() * 100:7.0f}% | {(v[:, 0] > b[:, 0]).mean() * 100:10.0f}%")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--paths", type=int, default=1000, help="bootstrap paths (0 = skip)")
    ap.add_argument("--no-world", action="store_true", help="skip the 10 other markets")
    args = ap.parse_args()
    px, r, rf = load_index()
    E = exposures(px)
    nets = report_index(px, r, rf, E)
    report_sharpe_significance(px, rf, nets)
    report_etf(px, rf, E)
    if not args.no_world:
        report_world(rf)
    if args.paths:
        report_paths(px, r, rf, args.paths)


if __name__ == "__main__":
    main()
