#!/usr/bin/env python3
"""
Swarm-Diagnose: Warum schlaegt "Top-10 + Switching" Buy & Hold nicht?

Zerlegung (gleicher Zeitraum wie der Dashboard-Swarm):
  A) Auswahl : rollierende Top-10 nach Market Cap (EW wie Produktion / cap-gewichtet)
               vs SPY / QQQ
  B) Timing  : Produktions-Switching (WF + Rolling-Sharpe, 1:1 nachgebaut) vs
               einfache Alternativen (Einzelaktien-SMA200, Index-Trendfilter)
               + Timing-Skill: Rendite der Aktien an investierten vs. Cash-Tagen
  C) "Kopflastig" langfristig: survivorship-freie Mega-Cap-ETFs vs SPY

Ausgabe: research/swarm_report.txt
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from wf_backtest import swarm_wf as S  # noqa: E402
from wf_backtest.core_strategy import CONFIG as CORE, trend_state  # noqa: E402
from wf_backtest.metrics import rolling_sharpe  # noqa: E402
from wf_backtest.switching import apply_switching, held_exposure  # noqa: E402
from wf_backtest.walk_forward import wf_strategy_oos  # noqa: E402

OUT = Path(__file__).resolve().parent / "swarm_report.txt"
LINES: list[str] = []


def out(s: str = "") -> None:
    print(s)
    LINES.append(s)


def stats(r: pd.Series, rf: float = 0.02) -> dict:
    r = r.dropna()
    eq = (1 + r).cumprod()
    yrs = len(r) / 252
    vol = r.std() * np.sqrt(252)
    return {
        "cagr": eq.iloc[-1] ** (1 / yrs) - 1,
        "vol": vol,
        "dd": (eq / eq.cummax() - 1).min(),
        "sharpe": (r.mean() * 252 - rf) / vol if vol > 0 else np.nan,
    }


def row(name: str, r: pd.Series) -> dict:
    s = stats(r)
    out(f"  {name:<46} {s['cagr']*100:6.2f}%  {s['vol']*100:5.1f}%  {s['dd']*100:6.1f}%  {s['sharpe']:5.2f}")
    return s


# -- Daten --------------------------------------------------------------------

def load():
    uni = S.download_universe(start="2016-01-01", use_polygon=False)
    close = pd.DataFrame({t: d["close"] for t, d in uni.items()}).sort_index()
    shares = pd.DataFrame({t: d["shares"] for t, d in uni.items()}).reindex(close.index).ffill()
    keep = close.notna().sum(axis=1) >= 20          # wie Produktion
    close, shares = close[keep], shares[keep]
    ext = yf.download(["SPY", "QQQ", "BIL", "^GSPC", "^NDX"], start="2015-01-01",
                      auto_adjust=True, progress=False)["Close"]
    ext = ext.reindex(close.index).ffill()
    return close, shares, ext


def top10_schedule(close, shares, n=S.TOP_N, every=S.REBALANCE_DAYS, warm=252):
    """Auswahl am Schluss des Rebalance-Tags (wie Produktion)."""
    mcap = close * shares
    sched = []
    for d in close.index[warm::every]:
        m = mcap.loc[d].dropna()
        if len(m) >= n:
            sched.append((d, m.sort_values(ascending=False).iloc[:n]))
    return sched


def ew_weights(close, sched):
    """Gleichgewicht, gueltig ab dem Handelstag NACH der Auswahl (kein Look-Ahead)."""
    idx = close.index
    w = pd.DataFrame(0.0, index=idx, columns=close.columns)
    for k, (d, top) in enumerate(sched):
        i0 = idx.get_loc(d) + 1
        i1 = idx.get_loc(sched[k + 1][0]) + 1 if k + 1 < len(sched) else len(idx)
        cols = [close.columns.get_loc(t) for t in top.index]
        w.iloc[i0:i1, cols] = 1.0 / len(top)
    return w


def turnover_cost(w):
    return 0.5 * w.diff().abs().sum(axis=1).fillna(0) * S.SLIP


def cap_weighted(rets, sched, idx):
    """Cap-gewichtet ab Auswahl, innerhalb des Quartals driftend (= echt 'kopflastig')."""
    starts = {}
    for d, top in sched:
        i0 = idx.get_loc(d) + 1
        if i0 < len(idx):
            starts[idx[i0]] = top / top.sum()
    r = pd.Series(0.0, index=idx)
    w = None
    for d in idx:
        if d in starts:
            new = starts[d]
            if w is not None:
                r[d] -= 0.5 * new.sub(w, fill_value=0).abs().sum() * S.SLIP
            w = new.copy()
        if w is None:
            continue
        rr = rets.loc[d, w.index].fillna(0.0)
        pr = float((w * rr).sum())
        r[d] += pr
        w = w * (1 + rr) / (1 + pr)
    return r


def production_switching(close, sched):
    """Per-Aktie WF + Rolling-Sharpe-Switching exakt wie run_swarm_wf (nicht-Cloud)."""
    ever = sorted({t for _, top in sched for t in top.index})
    res = {}
    for i, t in enumerate(ever, 1):
        c = close[t].dropna()
        if len(c) < 600:
            continue
        r = c.pct_change().dropna()
        c = c.loc[r.index]
        oos, pos = {}, {}
        for name, sdef in S.STRATEGY_DEFS.items():
            x = wf_strategy_oos(c, r, sdef, S.WF_CFG, S.TX, S.SLIP, S.RF)
            if x is not None:
                oos[name], pos[name], _ = x
        if len(oos) < 2:
            continue
        s0 = max(s.first_valid_index() for s in oos.values())
        s1 = min(s.last_valid_index() for s in oos.values())
        df = pd.DataFrame({n: s.loc[s0:s1] for n, s in oos.items()})
        roll = pd.DataFrame({n: rolling_sharpe(df[n], S.ROLL_SHARPE_WINDOW, S.RF) for n in df.columns})
        sw, active = apply_switching(df, roll, tx=S.TX, slip=S.SLIP, min_hold=S.MIN_HOLD)
        res[t] = (sw, held_exposure(active, pos))
        print(f"    WF {i}/{len(ever)} {t}: investiert {float((res[t][1] > 0).mean())*100:.0f}%")
    return res


# -- Teil C: lange Historie mit echten ETFs ----------------------------------

def etf_long_history():
    tick = ["SPY", "OEF", "XLG", "MGC", "QQQ", "RSP"]
    names = {"SPY": "S&P 500", "OEF": "S&P 100 (Top 100)", "XLG": "S&P Top 50",
             "MGC": "Vanguard Mega Cap", "QQQ": "Nasdaq-100", "RSP": "S&P 500 Equal Wt"}
    px = yf.download(tick, start="2000-01-01", auto_adjust=True, progress=False)["Close"]
    wins = [("2001-01-01", "2007-12-31"), ("2008-01-01", "2014-12-31"),
            ("2015-01-01", str(px.index[-1].date()))]

    def cagr(s, a, b):
        s = s.loc[a:b].dropna()
        if s.empty or (s.index[0] - pd.Timestamp(a)).days > 10:
            return None
        return (s.iloc[-1] / s.iloc[0]) ** (252 / len(s)) - 1

    out("")
    out("C) 'Kopflastig' langfristig - echte ETFs (keine Survivorship), CAGR p.a. (Diff. zu SPY)")
    out("  " + " " * 24 + "".join(f"{a[:4]}-{b[:4]:<15}" for a, b in wins))
    for t in tick:
        cells = []
        for a, b in wins:
            v, base = cagr(px[t], a, b), cagr(px["SPY"], a, b)
            cells.append(f"{'n/a':<20}" if v is None else f"{v*100:5.1f}% ({(v-base)*100:+5.1f}){'':<6}")
        out(f"  {t:<5}{names[t]:<19}" + "".join(cells))


def main():
    out("Swarm-Diagnose  (Top-10 Market Cap, quartalsweise, Universum = swarm_wf.MEGA_CAP_UNIVERSE)")
    close, shares, ext = load()
    idx = close.index
    rets = close.pct_change(fill_method=None)
    cash = ext["BIL"].pct_change().fillna(0)
    sched = top10_schedule(close, shares)
    out(f"  Universum: {close.shape[1]} Aktien, Daten {idx[0].date()} .. {idx[-1].date()}, "
        f"{len(sched)} Rebalances")
    for d, top in sched[-2:]:
        out(f"  Top-10 per {d.date()}: {', '.join(top.index)}")

    w = ew_weights(close, sched)
    ew = (rets.fillna(0) * w).sum(axis=1) - turnover_cost(w)
    cw = cap_weighted(rets, sched, idx)

    # Einzelaktien-SMA200 (Signal am Vortagesschluss), Cash = BIL
    above = (close > close.rolling(200).mean()).astype(float).shift(1).fillna(0.0)
    slot = rets.fillna(0).mul(above).add((1 - above).mul(cash, axis=0))
    sma_cost = (above.diff().abs().fillna(0) * w).sum(axis=1) * S.SLIP
    ew_sma = (slot * w).sum(axis=1) - turnover_cost(w) - sma_cost

    # Index-Trendfilter wie Core-Strategie (1x), Signal am Vortagesschluss
    def idx_filter(sig_close, port):
        st = trend_state(sig_close.dropna(), CORE["sma_days"], CORE["band_in"], CORE["band_out"])
        st = st.reindex(idx).ffill().shift(1).fillna(0.0)
        return st * port + (1 - st) * cash - st.diff().abs().fillna(0) * S.SLIP

    ew_trend = idx_filter(ext["^GSPC"], ew)
    qqq_trend = idx_filter(ext["^NDX"], ext["QQQ"].pct_change().fillna(0))

    print("  Produktions-Switching nachbauen (dauert ein paar Minuten) ...")
    res = production_switching(close, sched)
    sw_df = pd.DataFrame({t: v[0] for t, v in res.items()}).reindex(idx)
    ex_df = pd.DataFrame({t: v[1] for t, v in res.items()}).reindex(idx)
    mem = (w[sw_df.columns] > 0) & sw_df.notna()
    wsw = mem.div(mem.sum(axis=1).replace(0, np.nan), axis=0).fillna(0)
    swarm = (sw_df.fillna(0) * wsw).sum(axis=1)

    p0 = max(v[0].index[0] for v in res.values())
    p1 = idx[-1]
    per = slice(p0, p1)
    out("")
    out(f"A+B) Zeitraum {p0.date()} .. {p1.date()}  (= Dashboard-Swarm-Zeitraum)")
    out(f"  {'Variante':<46} {'CAGR':>7}  {'Vol':>5}  {'MaxDD':>6}  {'Sharpe':>5}")
    row("SPY Buy & Hold", ext["SPY"].pct_change().loc[per])
    row("QQQ Buy & Hold", ext["QQQ"].pct_change().loc[per])
    s_ew = row("Top-10 EW Buy & Hold (Swarm-Benchmark)", ew.loc[per])
    row("Top-10 cap-gewichtet ('kopflastig')", cw.loc[per])
    s_sw = row("Top-10 + Produktions-Switching (= Swarm)", swarm.loc[per])
    row("Top-10 + Einzelaktien-SMA200", ew_sma.loc[per])
    row("Top-10 + Index-Trendfilter S&P SMA200 (1x)", ew_trend.loc[per])
    row("QQQ + Index-Trendfilter NDX SMA200 (1x)", qqq_trend.loc[per])

    # Timing-Skill: Rendite der Aktie an Tagen, an denen der Swarm investiert vs. in Cash war
    m = mem.loc[per]
    stock_r = rets[sw_df.columns].loc[per]
    inv = (ex_df.loc[per] > 0) & m
    outm = (ex_df.loc[per] <= 0) & m
    r_in = float(np.nanmean(stock_r.where(inv).to_numpy())) * 252
    r_out = float(np.nanmean(stock_r.where(outm).to_numpy())) * 252
    frac = float(inv.values.sum() / m.values.sum())
    naive = frac * s_ew["cagr"] + (1 - frac) * float(cash.loc[per].mean() * 252)
    out("")
    out("B) Timing-Skill des Produktions-Switchings (nur Tage mit Top-10-Mitgliedschaft)")
    out(f"  Anteil investiert                         {frac*100:5.1f}%")
    out(f"  Aktienrendite an investierten Tagen        {r_in*100:6.1f}% p.a. (arithm.)")
    out(f"  Aktienrendite an Cash-Tagen (verpasst)     {r_out*100:6.1f}% p.a. (arithm.)")
    out(f"  Erwartung ohne Skill (Quote x B&H + Cash)  {naive*100:6.1f}% CAGR  vs. real {s_sw['cagr']*100:.1f}%")
    n_flips = int((ex_df.loc[per] > 0).astype(float).diff().abs().where(m).sum().sum())
    yrs = (p1 - p0).days / 365.25
    out(f"  Ein/Aus-Wechsel je Aktie und Jahr          {n_flips / max(1, len(res)) / yrs:5.1f}")

    yr = pd.DataFrame({"SPY": ext["SPY"].pct_change(), "Top10_EW": ew, "Swarm": swarm}).loc[per]
    yr = (1 + yr).groupby(yr.index.year).prod() - 1
    out("")
    out("  Jahr     SPY   Top10-EW   Swarm")
    for y, rr in yr.iterrows():
        out(f"  {y}  {rr['SPY']*100:6.1f}%  {rr['Top10_EW']*100:7.1f}%  {rr['Swarm']*100:6.1f}%")

    etf_long_history()
    OUT.write_text("\n".join(LINES) + "\n", encoding="utf-8")
    print(f"\n-> {OUT}")


if __name__ == "__main__":
    main()
