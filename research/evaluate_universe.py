"""
Research: can a mix of 2x ETFs or other equity types lift the core strategy's return?

    python research/evaluate_universe.py [--paths 600]

Pre-registered, no parameter search: the live rule (SMA200, enter > SMA*1.01, exit < SMA) is
applied unchanged to every index. Index model = price index + dividend estimate (S&P 500: total
return); kx ETF = k*index - (k-1)*T-bill - drag, the drag fitted to the real US kx ETF of that
index. Checks: two sub-periods, execution one day later (lag 2), real US 2x ETFs (2007-today),
rolling 10y windows, joint 10y block bootstrap (historical premium and premium -3 %/yr), a simple
German tax model (26.375 %, 30 % Teilfreistellung for equity funds) and the UCITS 2x ETFs.
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
from evaluate_core import COST, RULE, TBILL_FEE, load_index  # noqa: E402
from wf_backtest.core_strategy import _yf_ohlc, trend_state  # noqa: E402

warnings.filterwarnings("ignore")
SQ = np.sqrt(252)
TYPES = {"^GSPC": ("S&P 500", None), "^NDX": ("Nasdaq-100", 0.010),
         "^RUT": ("Russell 2000", 0.015), "^MID": ("MidCap 400", 0.015)}
FEE_1X = {"^GSPC": 0.0009, "^NDX": 0.0020, "^RUT": 0.0019, "^MID": 0.0023}
ETF_1X = {"^GSPC": "SPY", "^NDX": "QQQ", "^RUT": "IWM", "^MID": "MDY"}
ETF_2X = {"^GSPC": "SSO", "^NDX": "QLD", "^RUT": "UWM", "^MID": "MVV"}
ETF_3X = {"^GSPC": "UPRO", "^NDX": "TQQQ"}
UCITS = {"DBPG.DE": ("Xtrackers S&P 500 2x  LU0411078552", "SSO", "^GSPC"),
         "CL2.PA": ("Amundi MSCI USA 2x    FR0010755611", "SSO", "^GSPC"),
         "LQQ.PA": ("Amundi Nasdaq-100 2x  FR0010342592", "QLD", "^NDX")}
TAX, TF = 0.26375, 0.30
SPLIT = "2008-01-01"


# ----------------------------------------------------------------------------- data
def _closes(tickers: list[str], start: str) -> pd.DataFrame:
    df = _yf_ohlc(tickers, start, auto_adjust=True)["Close"]
    if isinstance(df, pd.Series):
        df = df.to_frame(tickers[0])
    df.index = pd.DatetimeIndex(df.index).tz_localize(None).normalize()
    return df.astype(float)


def load_universe():
    px_s, r_s, rf = load_index()
    idx = px_s.index[px_s.index >= pd.Timestamp("1980-01-01")]
    raw = _closes([t for t in TYPES if t != "^GSPC"] + ["^IXIC"], "1970-01-01")
    PX, R = {"^GSPC": px_s.loc[idx]}, {"^GSPC": r_s.loc[idx]}
    for t, (name, div) in TYPES.items():
        if t == "^GSPC":
            continue
        s = raw[t].dropna() if t in raw else pd.Series(dtype=float)
        if len(s) < 2500:
            print(f"  ! {name} ({t}) not available -> skipped")
            continue
        s = s.reindex(idx).ffill()
        PX[t], R[t] = s, s.pct_change() + div / 252
    PX, R = pd.DataFrame(PX), pd.DataFrame(R)
    start = max(PX[t].first_valid_index() for t in PX)
    start = PX.index[PX.index.get_loc(start) + 300]
    return PX, R, rf.loc[idx], start, (px_s, r_s, rf, raw.get("^IXIC"))


def states(PX: pd.DataFrame) -> dict[str, np.ndarray]:
    return {t: trend_state(PX[t].dropna(), **RULE).reindex(PX.index).fillna(0.0).to_numpy() for t in PX}


# ----------------------------------------------------------------------------- returns
def _held(a, lag: int) -> np.ndarray:
    a = np.asarray(a, dtype=float)
    return np.concatenate([np.zeros(lag), a[:-lag]])


def mix_returns(E_dec, rc, r1, r2, r3, lag: int = 1, switch=None):
    """Net daily returns and T-bill part of exposure E (0..3) built from T-bills, a 1x fund,
    a 2x ETF and (E > 2) a 3x ETF. Exposure decided at close t is held from close t+lag."""
    E = _held(E_dec, lag)
    w3 = np.clip(E - 2, 0, 1)
    w2 = np.clip(E - 1, 0, 1) - w3
    w1 = np.minimum(E, 1) - np.clip(E - 1, 0, 1)
    wc = np.clip(1 - E, 0, None)
    cash = wc * rc
    W = np.vstack([wc, w1, w2, w3])
    turn = np.abs(np.diff(W, axis=1, prepend=W[:, :1])).sum(axis=0)
    if switch is not None:  # rotation: sell the old fund, buy the new one
        inv = (E > 0).astype(float)
        turn = turn + 2 * (1 - wc) * _held(switch, lag) * inv * np.r_[0.0, inv[:-1]]
    return cash + w1 * r1 + w2 * r2 + w3 * r3 - turn * COST, cash


def model_legs(R: pd.DataFrame, rf: pd.Series, d2: dict, d3: dict):
    f = rf.to_numpy()
    legs = {}
    for t in R:
        r = np.nan_to_num(R[t].to_numpy())
        legs[t] = (r - FEE_1X[t] / 252, 2 * r - f - d2.get(t, 0.02) / 252,
                   3 * r - 2 * f - d3.get(t, 0.03) / 252)
    return f - TBILL_FEE / 252, legs


def blend(parts):
    return sum(w * p[0] for w, p in parts), sum(w * p[1] for w, p in parts)


def rotation(idx, R, st, legs, rc, universe, L, lag, lookback=252):
    """At each month end pick the index with the best 12-month return; its own trend decides."""
    tr = (1 + R[universe].fillna(0.0)).cumprod()
    mom = ((tr / tr.shift(lookback) - 1).where(R[universe].notna().cumsum() > lookback)).to_numpy()
    m = idx.month
    is_me = np.r_[m[1:] != m[:-1], True]
    code, cur = np.zeros(len(idx), dtype=int), 0
    for i in range(len(idx)):
        if is_me[i] and np.isfinite(mom[i]).any():
            cur = int(np.nanargmax(mom[i]))
        code[i] = cur
    cols = np.arange(len(idx))
    E_dec = L * np.vstack([st[u] for u in universe])[code, cols]
    hold = np.r_[np.full(lag, code[0]), code[:-lag]]
    r1, r2, r3 = (np.vstack([legs[u][k] for u in universe])[hold, cols] for k in range(3))
    switch = np.r_[0.0, (code[1:] != code[:-1]).astype(float)]
    return mix_returns(E_dec, rc, r1, r2, r3, lag=lag, switch=switch)


def variants(idx, R, st, legs, rc, lag: int = 1, with3x: bool = True) -> dict:
    S, N = "^GSPC", "^NDX"
    have = [t for t in TYPES if t in st]
    one = np.ones(len(idx))

    def mix(E, t):
        return mix_returns(E, rc, *legs[t], lag=lag)

    V = {"B&H S&P 1x": mix(one, S), "B&H S&P 2x": mix(2 * one, S),
         "S&P trend 1.5x": mix(1.5 * st[S], S), "S&P trend 2x": mix(2 * st[S], S)}
    for t in have[1:]:
        V[f"{TYPES[t][0]} trend 2x"] = mix(2 * st[t], t)
    if N in st:
        V["S&P+NDX trend 1.5x"] = blend([(0.5, V["S&P trend 1.5x"]), (0.5, mix(1.5 * st[N], N))])
        V["S&P+NDX trend 2x"] = blend([(0.5, V["S&P trend 2x"]), (0.5, V["Nasdaq-100 trend 2x"])])
        V["Rot S&P/NDX trend 2x"] = rotation(idx, R, st, legs, rc, [S, N], 2.0, lag)
    V[f"{len(have)} types trend 2x"] = blend([(1 / len(have), mix(2 * st[t], t)) for t in have])
    V[f"Rot {len(have)} types trend 2x"] = rotation(idx, R, st, legs, rc, have, 2.0, lag)
    if with3x:
        V["S&P trend 3x"] = mix(3 * st[S], S)
    return V


# ----------------------------------------------------------------------------- metrics
def stats(x: np.ndarray, rfa: np.ndarray) -> dict:
    eq = np.cumprod(1 + x)
    ex = x - rfa
    return dict(cagr=eq[-1] ** (252 / len(x)) - 1, vol=x.std() * SQ, sharpe=ex.mean() / ex.std() * SQ,
                dd=(eq / np.maximum.accumulate(eq) - 1).min())


def worst_year(x: pd.Series) -> float:
    return ((1 + x).groupby(x.index.year).prod() - 1).min()


def rolling10(x: pd.Series) -> np.ndarray:
    starts = x.resample("MS").first().index
    starts = starts[starts + pd.DateOffset(years=10) <= x.index[-1]]
    out = []
    for s in starts:
        seg = x.loc[s:s + pd.DateOffset(years=10)].to_numpy()
        out.append(np.prod(1 + seg) ** (252 / len(seg)) - 1)
    return np.array(out)


# ----------------------------------------------------------------------------- reports
def fit_drag(etf: pd.Series, r: pd.Series, rf: pd.Series, k: int):
    a = etf.dropna().pct_change().dropna()
    idx = a.index.intersection(r.dropna().index)
    a, rr, rff = a.loc[idx].to_numpy(), r.loc[idx].to_numpy(), rf.loc[idx].to_numpy()
    target, base = np.log1p(a).sum(), k * rr - (k - 1) * rff
    lo, hi = -0.10, 0.20
    for _ in range(50):
        d = (lo + hi) / 2
        lo, hi = (d, hi) if np.log1p(base - d / 252).sum() > target else (lo, d)
    return (lo + hi) / 2, f"{idx[0].year}-{idx[-1].year}"


def report_costs(R, rf, E):
    print("\n== Real US leveraged ETFs: annual drag vs k*index - (k-1)*T-bill (fitted on total return) ==")
    d2, d3 = {}, {}
    for k, table, out in ((2, ETF_2X, d2), (3, ETF_3X, d3)):
        for t, e in table.items():
            if t in R and e in E and E[e].notna().sum() > 1000:
                out[t], per = fit_drag(E[e], R[t], rf, k)
                print(f"  {e:5s} {k}x {TYPES[t][0]:13s} drag {out[t] * 100:5.2f} %/yr  ({per})")
    return d2, d3


def report_types(PX, R, st, legs, rc, rfa, start):
    print(f"\n== Same rule on each equity type, index model, lag 1 ==")
    print(f"  {'':14s} {'period':9s} | {'Buy&Hold 1x':^20s} | {'Trend 1x':^20s} | {'Trend 2x':^20s}")
    print(f"  {'':14s} {'':9s} | {'CAGR Sharpe MaxDD':^20s} | {'CAGR Sharpe MaxDD':^20s} | {'CAGR Sharpe MaxDD':^20s}")
    idx = PX.index
    for t in PX:
        for a, b in ((start, SPLIT), (SPLIT, None), (start, None)):
            sl = (idx >= pd.Timestamp(a)) & ((idx < pd.Timestamp(b)) if b else True)
            line = f"  {TYPES[t][0] if a == start and b == SPLIT else '':14s} {idx[sl][0].year}-{idx[sl][-1].year % 100:02d}  |"
            for E in (np.ones(len(idx)), st[t], 2 * st[t]):
                s = stats(mix_returns(E, rc, *legs[t])[0][sl], rfa[sl])
                line += f" {s['cagr'] * 100:5.1f}% {s['sharpe']:5.2f} {s['dd'] * 100:5.0f}% |"
            print(line)


def report_candidates(idx, R, st, legs, rc, rfa, start):
    V1 = variants(idx, R, st, legs, rc, lag=1)
    V2 = variants(idx, R, st, legs, rc, lag=2)
    sl0 = idx >= start
    nets = {k: pd.Series(v[0][sl0], index=idx[sl0]) for k, v in V1.items()}
    rf_s = pd.Series(rfa[sl0], index=idx[sl0])
    bh10 = rolling10(nets["B&H S&P 1x"])
    print(f"\n== Candidates, index model ({idx[sl0][0].date()} - {idx[-1].date()}) ==")
    print(f"  {'':22s} {'<2008':>13s} {'>=2008':>13s} | {'full: CAGR':>10s} {'Vol':>5s} {'Shrp':>5s} {'MaxDD':>6s} "
          f"{'worstY':>6s} | {'lag2':>12s} | 10y: {'>B&H':>4s} {'>=11%':>5s} {'>=15%':>5s} {'med':>5s}")
    for k, x in nets.items():
        a, b = x[x.index < SPLIT], x[x.index >= SPLIT]
        sa, sb = stats(a.to_numpy(), rf_s.loc[a.index].to_numpy()), stats(b.to_numpy(), rf_s.loc[b.index].to_numpy())
        s = stats(x.to_numpy(), rf_s.to_numpy())
        s2 = stats(V2[k][0][sl0], rfa[sl0])
        r10 = rolling10(x)
        print(f"  {k:22s} {sa['cagr'] * 100:5.1f}/{sa['dd'] * 100:4.0f}% {sb['cagr'] * 100:5.1f}/{sb['dd'] * 100:4.0f}% | "
              f"{s['cagr'] * 100:9.2f}% {s['vol'] * 100:4.0f}% {s['sharpe']:5.2f} {s['dd'] * 100:5.1f}% "
              f"{worst_year(x) * 100:5.0f}% | {s2['cagr'] * 100:5.1f}/{s2['dd'] * 100:4.0f}% | "
              f"{np.mean(r10 > bh10) * 100:8.0f}% {np.mean(r10 >= 0.11) * 100:4.0f}% {np.mean(r10 >= 0.15) * 100:4.0f}% "
              f"{np.median(r10) * 100:4.1f}%")
    print(f"  (10y windows: {len(bh10)} monthly starts; lag2 = trade one close later)")
    return V1


def etf_variants(PX, R, st, E):
    """Variants on real US ETFs, traded one close later: (V, period mask, T-bill ETF returns, legs)."""
    have = [t for t in PX if ETF_1X[t] in E and ETF_2X[t] in E]
    cols = [ETF_1X[t] for t in have] + [ETF_2X[t] for t in have] + ["BIL"]
    first = max(E[cols].dropna().index[0], pd.Timestamp("2007-06-01"))
    ret = E[cols].pct_change().reindex(PX.index).fillna(0.0)
    legs = {t: (ret[ETF_1X[t]].to_numpy(), ret[ETF_2X[t]].to_numpy(), np.zeros(len(PX))) for t in have}
    rc = ret["BIL"].to_numpy()
    V = variants(PX.index, R[have], {t: st[t] for t in have}, legs, rc, lag=2, with3x=False)
    sl = (PX.index > first) & PX.index.isin(E[cols].dropna().index)
    return V, sl, rc, legs


def run_etf_validation(PX, R, st, E):
    V, sl, rc, _ = etf_variants(PX, R, st, E)
    idx = PX.index[sl]
    print(f"\n== Real US ETFs (SPY/QQQ/IWM/MDY, SSO/QLD/UWM/MVV, BIL), trade one close later, "
          f"{idx[0].date()} - {idx[-1].date()} ==")
    print(f"  {'':22s} {'CAGR':>7s} {'Vol':>5s} {'Sharpe':>6s} {'MaxDD':>7s} {'worstY':>7s}")
    for k, (x, _) in V.items():
        s = stats(x[sl], rc[sl])
        print(f"  {k:22s} {s['cagr'] * 100:6.2f}% {s['vol'] * 100:4.0f}% {s['sharpe']:6.2f} {s['dd'] * 100:6.1f}% "
              f"{worst_year(pd.Series(x[sl], index=idx)) * 100:6.1f}%")


def ndx_excess(rn: np.ndarray, rs: np.ndarray, mode: str) -> float:
    """Daily drift to subtract from the Nasdaq-100 so that its mean ('mean') or its
    compound growth ('cagr') equals the S&P 500's over the sample."""
    if mode == "mean":
        return float(rn.mean() - rs.mean())
    lo, hi, target = -0.01, 0.01, np.log1p(rs).sum()
    for _ in range(60):
        c = (lo + hi) / 2
        lo, hi = (c, hi) if np.log1p(rn - c).sum() > target else (lo, c)
    return (lo + hi) / 2


def neutralize_ndx(PX: pd.DataFrame, R: pd.DataFrame, start, mode: str):
    """Remove the Nasdaq-100's historical excess return over the S&P 500 (hindsight check)."""
    sl = R.index >= start
    ex = ndx_excess(R["^NDX"][sl].to_numpy(), R["^GSPC"][sl].to_numpy(), mode)
    R2, PX2 = R.copy(), PX.copy()
    R2["^NDX"] = R["^NDX"] - ex
    p = PX["^NDX"].pct_change() - ex
    PX2["^NDX"] = (1 + p.fillna(0.0)).cumprod().where(PX["^NDX"].notna())
    return PX2, R2, ex * 252


def report_neutral(PX, R, rf, d2, d3, start):
    if "^NDX" not in PX:
        return
    for mode, text in (("cagr", "same CAGR as S&P (fair)"), ("mean", "same mean daily return as S&P (harsh)")):
        PXn, Rn, ex = neutralize_ndx(PX, R, start, mode)
        report_neutral_block(PX, Rn, PXn, rf, d2, d3, start, f"{text}: -{ex * 100:.1f} %/yr")


def report_neutral_block(PX, Rn, PXn, rf, d2, d3, start, title):
    rc, legs = model_legs(Rn, rf, d2, d3)
    V = variants(PX.index, Rn, states(PXn), legs, rc, lag=1, with3x=False)
    sl = PX.index >= start
    rfa = rf.to_numpy()
    bh10 = None
    print(f"\n== Hindsight check: Nasdaq-100 without its excess return, {title} ==")
    for k in ("B&H S&P 1x", "S&P trend 2x", "Nasdaq-100 trend 2x", "S&P+NDX trend 1.5x", "S&P+NDX trend 2x",
              "Rot S&P/NDX trend 2x"):
        x = pd.Series(V[k][0][sl], index=PX.index[sl])
        s = stats(x.to_numpy(), rfa[sl])
        r10 = rolling10(x)
        bh10 = r10 if bh10 is None else bh10
        a, b = x[x.index < SPLIT], x[x.index >= SPLIT]
        print(f"  {k:22s} CAGR {s['cagr'] * 100:5.2f}%  Sharpe {s['sharpe']:.2f}  MaxDD {s['dd'] * 100:5.1f}%  "
              f"<2008 {stats(a.to_numpy(), rfa[sl][:len(a)])['cagr'] * 100:5.1f}%  "
              f">=2008 {stats(b.to_numpy(), rfa[sl][len(a):])['cagr'] * 100:5.1f}%  | 10y >B&H "
              f"{np.mean(r10 > bh10) * 100:3.0f}%  >=11% {np.mean(r10 >= 0.11) * 100:3.0f}%  "
              f">=15% {np.mean(r10 >= 0.15) * 100:3.0f}%  med {np.median(r10) * 100:4.1f}%")


def report_bootstrap(PX, R, rf, d2, d3, start, paths, years=10, block=250, warm=300, seed=7):
    rng = np.random.default_rng(seed)
    RP = PX.pct_change()
    base = pd.concat([R.add_prefix("r"), RP.add_prefix("p"), rf.rename("rf")], axis=1).loc[start:].dropna()
    Rv, Pv, Fv = base[[f"r{t}" for t in PX]].to_numpy(), base[[f"p{t}" for t in PX]].to_numpy(), base["rf"].to_numpy()
    n = years * 252 + warm
    idx = pd.bdate_range("2000-01-03", periods=n)
    keep = ["B&H S&P 1x", "B&H S&P 2x", "S&P trend 1.5x", "S&P trend 2x", "Nasdaq-100 trend 2x", "S&P+NDX trend 1.5x",
            "S&P+NDX trend 2x", "Rot S&P/NDX trend 2x", f"{len(PX.columns)} types trend 2x",
            f"Rot {len(PX.columns)} types trend 2x"]
    iN, iS = list(PX.columns).index("^NDX"), list(PX.columns).index("^GSPC")
    out = {}

    def sample() -> np.ndarray:
        out, i = np.empty(n, dtype=int), 0
        while i < n:
            s = rng.integers(0, len(Rv))
            k = min(rng.geometric(1 / block), n - i)
            out[i:i + k] = (s + np.arange(k)) % len(Rv)
            i += k
        return out

    for label, delta, neutral in (("historical premium", 0.0, False), ("premium -3 %/yr", 0.03, False),
                                  ("premium -3 %/yr AND Nasdaq-100 CAGR = S&P CAGR", 0.03, True)):
        Rs, Ps = Rv.copy(), Pv.copy()
        if neutral:
            ex = ndx_excess(Rv[:, iN], Rv[:, iS], "cagr")
            Rs[:, iN] -= ex
            Ps[:, iN] -= ex
        res = {k: [] for k in keep}
        for _ in range(paths):
            ix = sample()
            Rp = pd.DataFrame(Rs[ix] - delta / 252, index=idx, columns=PX.columns)
            Pp = pd.DataFrame(np.cumprod(1 + Ps[ix] - delta / 252, axis=0), index=idx, columns=PX.columns)
            rfp = pd.Series(Fv[ix], index=idx)
            rc, legs = model_legs(Rp, rfp, d2, d3)
            V = variants(idx, Rp, states(Pp), legs, rc, lag=1, with3x=False)
            for k in keep:
                x = V[k][0][warm:]
                eq = np.cumprod(1 + x)
                res[k].append((eq[-1] ** (252 / len(x)) - 1, (eq / np.maximum.accumulate(eq) - 1).min()))
        bh = np.array(res["B&H S&P 1x"])[:, 0]
        out[label] = {k: np.array(v) for k, v in res.items()}
        print(f"\n== {years}y joint block bootstrap since {start.year}, {label} ({paths} paths) ==")
        print(f"  {'':22s} {'CAGR p10':>8s} {'p50':>6s} {'p90':>6s} | {'>=11%':>5s} {'>=15%':>5s} {'>B&H':>5s} | "
              f"{'MaxDD p50':>9s} {'DD<-50%':>7s}")
        for k, v in res.items():
            v = np.array(v)
            print(f"  {k:22s} {np.percentile(v[:, 0], 10) * 100:7.1f}% {np.median(v[:, 0]) * 100:5.1f}% "
                  f"{np.percentile(v[:, 0], 90) * 100:5.1f}% | {np.mean(v[:, 0] >= 0.11) * 100:4.0f}% "
                  f"{np.mean(v[:, 0] >= 0.15) * 100:4.0f}% {np.mean(v[:, 0] > bh) * 100:4.0f}% | "
                  f"{np.median(v[:, 1]) * 100:8.1f}% {np.mean(v[:, 1] < -0.5) * 100:6.0f}%")
    return out


def after_tax_wealth(net: np.ndarray, cash: np.ndarray, years: np.ndarray, tf: float, deferred: bool) -> float:
    """Terminal wealth (start 1) after German capital gains tax (26.375 %).
    deferred: Buy & Hold, taxed once at the end. Otherwise gains are taxed every year
    (conservative: as if realised), equity part with Teilfreistellung tf, T-bill part fully,
    losses carried forward."""
    if deferred:
        g = float(np.prod(1 + net))
        return g - TAX * (1 - tf) * max(0.0, g - 1)
    W, carry = 1.0, 0.0
    cut = np.flatnonzero(np.r_[True, years[1:] != years[:-1], True])
    for a, b in zip(cut[:-1], cut[1:]):
        g = np.cumprod(1 + net[a:b])
        cash_pnl = W * float((np.r_[1.0, g[:-1]] * cash[a:b]).sum())
        pnl = W * (g[-1] - 1)
        taxable = (1 - tf) * (pnl - cash_pnl) + cash_pnl - carry
        W *= g[-1]
        if taxable > 0:
            W, carry = W - TAX * taxable, 0.0
        else:
            carry = -taxable
    return W


def report_tax(V, idx, start):
    sl = idx >= start
    years = idx[sl].year.to_numpy()
    rows = [("B&H S&P 1x", TF, True, "UCITS ETF, held"), ("B&H S&P 2x", TF, True, "UCITS 2x ETF, held"),
            ("S&P trend 1.5x", 0.0, False, "US ETFs (Alpaca), no TF"),
            ("S&P trend 2x", 0.0, False, "US ETFs (Alpaca), no TF"),
            ("S&P trend 2x", TF, False, "UCITS 2x, TF 30 %"),
            ("S&P trend 3x", 0.0, False, "US ETFs (Alpaca), no TF"),
            ("Nasdaq-100 trend 2x", 0.0, False, "US ETFs (Alpaca), no TF"),
            ("Nasdaq-100 trend 2x", TF, False, "UCITS 2x, TF 30 %"),
            ("S&P+NDX trend 1.5x", 0.0, False, "US ETFs (Alpaca), no TF"),
            ("S&P+NDX trend 1.5x", TF, False, "UCITS 1x+2x, TF 30 %"),
            ("S&P+NDX trend 2x", TF, False, "UCITS 2x, TF 30 %"),
            ("S&P+NDX trend 2x", 0.0, False, "US ETFs (Alpaca), no TF"),
            ("Rot S&P/NDX trend 2x", TF, False, "UCITS 2x, TF 30 %")]
    dates = idx[sl]
    starts = [pd.Timestamp(s) for s in pd.Series(dates).groupby(dates.to_period("M")).first()]
    starts = [s for s in starts if s + pd.DateOffset(years=10) <= dates[-1]]
    print(f"\n== After German tax (26.375 %; strategies taxed yearly as if realised = conservative) ==")
    print(f"  {'':22s} {'setup':26s} {'pre-tax':>8s} {'after tax':>9s} | 10y median pre / after")
    out = []
    for k, tf, dfr, label in rows:
        if k not in V:
            continue
        net, cash = V[k][0][sl], V[k][1][sl]
        yrs = len(net) / 252
        pre = np.prod(1 + net) ** (1 / yrs) - 1
        post = after_tax_wealth(net, cash, years, tf, dfr) ** (1 / yrs) - 1
        p10, a10 = [], []
        for s in starts:
            m = (dates >= s) & (dates < s + pd.DateOffset(years=10))
            y10 = m.sum() / 252
            p10.append(np.prod(1 + net[m]) ** (1 / y10) - 1)
            a10.append(after_tax_wealth(net[m], cash[m], years[m], tf, dfr) ** (1 / y10) - 1)
        print(f"  {k:22s} {label:26s} {pre * 100:7.2f}% {post * 100:8.2f}% | {np.median(p10) * 100:5.1f}% / "
              f"{np.median(a10) * 100:4.1f}%")
        out.append(dict(k=k, tf=tf, deferred=dfr, pre=pre, post=post))
    return out


def report_ucits(R):
    data = _closes(list(UCITS) + ["SSO", "QLD", "EURUSD=X"], "2006-01-01")
    fx = data["EURUSD=X"].dropna()
    print("\n== UCITS 2x ETFs (Teilfreistellung) vs US 2x ETFs, both in EUR ==")
    for tic, (name, us, t) in UCITS.items():
        s = data[tic].dropna() if tic in data else pd.Series(dtype=float)
        if len(s) < 500:
            print(f"  {name}: no data ({tic})")
            continue
        jumps = s.pct_change().abs() > 0.35
        if jumps.any():
            print(f"  {name}: suspicious jumps on {[d.date() for d in s.index[jumps]][:3]} -> using data after last jump")
            s = s.loc[s.index[jumps][-1]:]
        f = fx.reindex(s.index).ffill()
        u = (data[us].dropna() / fx.reindex(data[us].dropna().index).ffill()).reindex(s.index).ffill()
        ok = f.notna() & u.notna()
        s, u = s[ok], u[ok]
        yrs = (s.index[-1] - s.index[0]).days / 365.25
        c_s, c_u = (s.iloc[-1] / s.iloc[0]) ** (1 / yrs) - 1, (u.iloc[-1] / u.iloc[0]) ** (1 / yrs) - 1
        wk = pd.concat([s, (1 + R[t].fillna(0)).cumprod(), 1 / fx], axis=1, keys=["f", "i", "x"]).ffill()
        wk = wk.loc[s.index[0]:].resample("W-FRI").last().pct_change().dropna()
        X = np.column_stack([np.ones(len(wk)), wk["i"], wk["x"]])
        beta = np.linalg.lstsq(X, wk["f"].to_numpy(), rcond=None)[0]
        print(f"  {name} {s.index[0].year}-{s.index[-1].year}: {c_s * 100:5.1f} %/yr vs {us} {c_u * 100:5.1f} %/yr "
              f"(EUR) | weekly beta index {beta[1]:.2f}, USD/EUR {beta[2]:.2f}")


def report_long_growth(px_s, r_s, rf_all, ixic, d2):
    if ixic is None or ixic.dropna().empty:
        return
    idx = px_s.index[px_s.index >= ixic.dropna().index[0]]
    px_n = ixic.reindex(idx).ffill()
    r_n = np.nan_to_num((px_n.pct_change() + 0.010 / 252).to_numpy())
    r_sp, f = r_s.loc[idx].to_numpy(), rf_all.loc[idx].to_numpy()
    rc = f - TBILL_FEE / 252
    st_s = trend_state(px_s, **RULE).reindex(idx).to_numpy()
    st_n = trend_state(px_n, **RULE).to_numpy()
    leg_s = (r_sp - 0.0009 / 252, 2 * r_sp - f - d2.get("^GSPC", 0.02) / 252, np.zeros(len(idx)))
    leg_n = (r_n - 0.002 / 252, 2 * r_n - f - d2.get("^NDX", 0.02) / 252, np.zeros(len(idx)))
    print("\n== Longer history: Nasdaq Composite (growth/tech) vs S&P 500, same rule, 2x drag as SSO/QLD ==")
    for a, b in (("1972-01-01", "1990-01-01"), ("1990-01-01", None), ("1972-01-01", None)):
        sl = (idx >= pd.Timestamp(a)) & ((idx < pd.Timestamp(b)) if b else True)
        line = f"  {a[:4]}-{idx[sl][-1].year}"
        for nm, E, lg in (("S&P B&H", np.ones(len(idx)), leg_s), ("S&P T2x", 2 * st_s, leg_s),
                          ("Comp B&H", np.ones(len(idx)), leg_n), ("Comp T2x", 2 * st_n, leg_n),
                          ("50/50 T2x", None, None)):
            if E is None:
                x = 0.5 * mix_returns(2 * st_s, rc, *leg_s)[0] + 0.5 * mix_returns(2 * st_n, rc, *leg_n)[0]
            else:
                x = mix_returns(E, rc, *lg)[0]
            s = stats(x[sl], f[sl])
            line += f" | {nm} {s['cagr'] * 100:5.1f}%/{s['dd'] * 100:4.0f}% S{s['sharpe']:.2f}"
        print(line)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--paths", type=int, default=600, help="bootstrap paths per scenario (0 = skip)")
    args = ap.parse_args()
    PX, R, rf, start, (px_s, r_s, rf_all, ixic) = load_universe()
    print(f"Types: {', '.join(TYPES[t][0] for t in PX)}; common evaluation start {start.date()}")
    E = _closes([ETF_1X[t] for t in PX] + [ETF_2X[t] for t in PX] + ["BIL"], "2006-01-01")
    E3 = _closes(list(ETF_3X.values()), "2009-01-01")
    d2, d3 = report_costs(R, rf, pd.concat([E, E3], axis=1))
    st = states(PX)
    rc, legs = model_legs(R, rf, d2, d3)
    rfa = rf.to_numpy()
    report_types(PX, R, st, legs, rc, rfa, start)
    V = report_candidates(PX.index, R, st, legs, rc, rfa, start)
    report_neutral(PX, R, rf, d2, d3, start)
    report_long_growth(px_s, r_s, rf_all, ixic, d2)
    run_etf_validation(PX, R, st, E)
    report_tax(V, PX.index, start)
    report_ucits(R)
    if args.paths:
        report_bootstrap(PX, R, rf, d2, d3, start, args.paths)


if __name__ == "__main__":
    main()
