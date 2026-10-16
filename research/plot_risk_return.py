"""
Risk/return map of all variants from evaluate_universe.py: CAGR (x) vs. maximum drawdown (y).

    python research/plot_risk_return.py [--paths 600]

Panels: index model 1988-2026, real US ETFs 2007-2026, after German tax, 10y bootstrap medians.
Writes research/risk_return_map.png.
"""
from __future__ import annotations

import argparse
import contextlib
import io
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from adjustText import adjust_text  # noqa: E402
from matplotlib.colors import to_rgba  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Patch, Rectangle  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import evaluate_universe as eu  # noqa: E402

OUT = HERE / "risk_return_map.png"
S = "^GSPC"
STYLE = {  # variant -> (label, colour, marker)
    "B&H S&P 1x": ("B&H S&P 500", "black", "s"),
    "B&H S&P 2x": ("B&H S&P 500 2x", "#636363", "s"),
    "S&P trend 1x": ("S&P Trend 1x", "#9ecae1", "o"),
    "S&P trend 1.5x": ("S&P Trend 1,5x (aktuell)", "#4292c6", "o"),
    "S&P trend 2x": ("S&P Trend 2x", "#08519c", "o"),
    "S&P trend 3x": ("S&P Trend 3x", "#08306b", "o"),
    "B&H Nasdaq-100": ("B&H Nasdaq-100", "#fdae6b", "s"),
    "Nasdaq-100 trend 2x": ("Nasdaq-100 Trend 2x", "#e6550d", "o"),
    "B&H Russell 2000": ("B&H Russell 2000", "#c994c7", "s"),
    "Russell 2000 trend 2x": ("Russell 2000 Trend 2x", "#7a0177", "o"),
    "B&H MidCap 400": ("B&H MidCap 400", "#d9b38c", "s"),
    "MidCap 400 trend 2x": ("MidCap 400 Trend 2x", "#8c510a", "o"),
    "S&P+NDX trend 1.5x": ("S&P+NDX Trend 1,5x", "#74c476", "D"),
    "S&P+NDX trend 2x": ("S&P+NDX Trend 2x", "#006d2c", "D"),
    "Rot S&P/NDX trend 2x": ("Rotation S&P/NDX 2x", "#e7298a", "^"),
    "4 types trend 2x": ("4 Typen Trend 2x", "#969696", "D"),
    "Rot 4 types trend 2x": ("Rotation 4 Typen 2x", "#bdbdbd", "^"),
}
OTHER = {"^NDX": "Nasdaq-100", "^RUT": "Russell 2000", "^MID": "MidCap 400"}
HINDSIGHT = ("Nasdaq-100 trend 2x", "S&P+NDX trend 1.5x", "S&P+NDX trend 2x", "Rot S&P/NDX trend 2x")
SCENARIOS = ("wie 1988–2026", "Aktienrendite −3 %/J.", "−3 %/J. und Nasdaq = S&P")


def de(v: float) -> str:
    return f"{v:.1f}".replace(".", ",").replace("-", "−")


def tag(k: str, x: float, y: float) -> str:
    dd = f"{y:.0f}".replace("-", "−")
    return f"{STYLE[k][0]} ({de(x)} % | {dd} %)"


def extras(n: int, st, legs, rc, lag: int) -> dict:
    """Variants not built by eu.variants(): S&P trend 1x and Buy & Hold of the other indices."""
    out = {"S&P trend 1x": eu.mix_returns(st[S], rc, *legs[S], lag=lag)}
    for t, name in OTHER.items():
        if t in legs:
            out[f"B&H {name}"] = eu.mix_returns(np.ones(n), rc, *legs[t], lag=lag)
    return out


def points(V: dict, sl, rfa) -> dict:
    out = {}
    for k, v in V.items():
        s = eu.stats(v[0][sl], rfa[sl])
        out[k] = (s["cagr"] * 100, s["dd"] * 100)
    return out


def dot(ax, x, y, k, fill="full", size=75):
    color, marker = STYLE[k][1], STYLE[k][2]
    face = {"full": color, "half": to_rgba(color, 0.35), "none": "white"}[fill]
    ax.scatter(x, y, s=size, marker=marker, facecolors=[face], edgecolors=["black" if fill == "full" else color],
               linewidths=0.6 if fill == "full" else 1.6, zorder=3)


def labelled(ax, P: dict, texts, xs, ys):
    for k in STYLE:
        if k in P:
            x, y = P[k]
            dot(ax, x, y, k)
            texts.append(ax.text(x, y, tag(k, x, y), fontsize=8.5))
            xs.append(x)
            ys.append(y)


def finish(ax, title, texts, xs, ys, bh=None):
    ax.set_title(title, fontsize=11.5, fontweight="bold", loc="left")
    ax.set_xlabel("Rendite p.a. (CAGR) in %")
    ax.set_ylabel("Maximaler Drawdown in %")
    ax.grid(alpha=0.3)
    lo, hi = min(xs) - 1.5, max(xs) + 3.0
    ax.set_xlim(lo, hi)
    ax.set_ylim(min(ys) - 6, 0)
    for x, txt, ls in ((11, "11 %", ":"), (15, "15 % Ziel", "--")):
        if lo < x < hi:
            ax.axvline(x, color="firebrick", ls=ls, lw=1.2, alpha=0.8, zorder=1)
            ax.text(x, 0.99, f" {txt}", color="firebrick", fontsize=9, va="top", transform=ax.get_xaxis_transform())
    if bh is not None:
        ax.add_patch(Rectangle(bh, hi - bh[0], -bh[1], color="green", alpha=0.08, lw=0, zorder=0))
    adjust_text(texts, x=xs, y=ys, ax=ax, arrowprops=dict(arrowstyle="-", color="0.55", lw=0.6))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--paths", type=int, default=600, help="bootstrap paths per scenario")
    args = ap.parse_args()
    with contextlib.redirect_stdout(io.StringIO()):
        PX, R, rf, start, _ = eu.load_universe()
        E = eu._closes([eu.ETF_1X[t] for t in PX] + [eu.ETF_2X[t] for t in PX] + ["BIL"], "2006-01-01")
        E3 = eu._closes(list(eu.ETF_3X.values()), "2009-01-01")
        d2, d3 = eu.report_costs(R, rf, pd.concat([E, E3], axis=1))
        st, n = eu.states(PX), len(PX)
        rc, legs = eu.model_legs(R, rf, d2, d3)
        V = eu.variants(PX.index, R, st, legs, rc, lag=1) | extras(n, st, legs, rc, lag=1)
        tax = eu.report_tax(V, PX.index, start)
        PXn, Rn, _ = eu.neutralize_ndx(PX, R, start, "cagr")
        rcn, legsn = eu.model_legs(Rn, rf, d2, d3)
        Vn = eu.variants(PX.index, Rn, eu.states(PXn), legsn, rcn, lag=1, with3x=False)
        Ve, sle, rce, legse = eu.etf_variants(PX, R, st, E)
        Ve |= extras(n, st, legse, rce, lag=2)
        boot = eu.report_bootstrap(PX, R, rf, d2, d3, start, args.paths)
    sl, rfa = PX.index >= start, rf.to_numpy()
    model, etf = points(V, sl, rfa), points(Ve, sle, rce)
    neutral = {k: v for k, v in points(Vn, sl, rfa).items() if k in HINDSIGHT}

    fig, axes = plt.subplots(2, 2, figsize=(22, 15.5))
    ax = axes[0, 0]
    texts, xs, ys = [], [], []
    labelled(ax, model, texts, xs, ys)
    for k, (x1, y1) in neutral.items():
        dot(ax, x1, y1, k, fill="none")
        ax.annotate("", (x1, y1), model[k], arrowprops=dict(arrowstyle="->", color=STYLE[k][1], ls="--", lw=1.2))
        xs.append(x1)
        ys.append(y1)
    finish(ax, f"(a) Vergangenheit {PX.index[sl][0].year}–{PX.index[-1].year}: Index-Modell vor Steuern, nach Kosten",
           texts, xs, ys, bh=model["B&H S&P 1x"])

    ax = axes[0, 1]
    texts, xs, ys = [], [], []
    labelled(ax, etf, texts, xs, ys)
    idx_e = PX.index[sle]
    finish(ax, f"(b) Echte US-ETFs {idx_e[0].year}–{idx_e[-1].year} (SPY/QQQ/IWM/MDY, SSO/QLD/UWM/MVV), "
               "1 Tag später gehandelt", texts, xs, ys, bh=etf["B&H S&P 1x"])

    ax = axes[1, 0]
    texts, xs, ys = [], [], []
    bh_tax = None
    for row in tax:
        k, x0, x1 = row["k"], row["pre"] * 100, row["post"] * 100
        y = model[k][1]
        setup = "UCITS, gehalten" if row["deferred"] else ("UCITS" if row["tf"] else "Alpaca")
        dot(ax, x0, y, k, fill="half", size=40)
        ax.annotate("", (x1, y), (x0, y), arrowprops=dict(arrowstyle="->", color="0.55", lw=1))
        dot(ax, x1, y, k, fill="full" if row["tf"] else "none")
        texts.append(ax.text(x1, y + (2.5 if row["tf"] else -4.0), f"{STYLE[k][0]} ({setup}): {de(x1)} %",
                             fontsize=8.5))
        xs += [x0, x1]
        ys += [y, y]
        if k == "B&H S&P 1x":
            bh_tax = (x1, y)
    finish(ax, f"(c) Nach deutscher Steuer {PX.index[sl][0].year}–{PX.index[-1].year} (Pfeil = Steuerabzug)",
           texts, xs, ys, bh=bh_tax)

    ax = axes[1, 1]
    texts, xs, ys = [], [], []
    scen = list(boot)
    for k in STYLE:
        if k not in boot[scen[0]]:
            continue
        pts = [(np.median(boot[s][k][:, 0]) * 100, np.median(boot[s][k][:, 1]) * 100) for s in scen]
        ax.plot(*zip(*pts), color=STYLE[k][1], lw=1.2, alpha=0.6, zorder=2)
        for (x, y), fill in zip(pts, ("full", "half", "none")):
            dot(ax, x, y, k, fill=fill)
            xs.append(x)
            ys.append(y)
        texts.append(ax.text(*pts[0], tag(k, *pts[0]), fontsize=8.5))
    spread = "\n".join(f"   {STYLE[k][0]}: {de(np.percentile(boot[scen[0]][k][:, 0], 10) * 100)} bis "
                       f"{de(np.percentile(boot[scen[0]][k][:, 0], 90) * 100)} %"
                       for k in ("B&H S&P 1x", "S&P trend 2x", "S&P+NDX trend 2x"))
    ax.text(0.01, 0.93, f"Median aus {args.paths} simulierten 10-Jahres-Pfaden,\nWerte im Label = Szenario "
                        f"wie 1988–2026.\nStreuung groß (10.–90. Perzentil):\n{spread}", transform=ax.transAxes,
            fontsize=8.5, va="top", bbox=dict(boxstyle="round", fc="white", ec="0.7"))
    finish(ax, "(d) Zukunft: 10-Jahres-Simulation, Median je Szenario", texts, xs, ys)

    grey = "0.35"
    handles = [Line2D([], [], marker="s", ls="", color=grey, label="Buy & Hold"),
               Line2D([], [], marker="o", ls="", color=grey, label="Trendregel auf einem Index"),
               Line2D([], [], marker="D", ls="", color=grey, label="Mix 50/50 bzw. 4×25 %, jeder Teil mit eigenem Trend"),
               Line2D([], [], marker="^", ls="", color=grey, label="Rotation in den stärksten Index (12 Monate)"),
               Line2D([], [], marker="D", ls="--", color=grey, markerfacecolor="white",
                      label="(a) hohl + Pfeil: Nasdaq ohne Mehrrendite ggü. S&P (fairer Test)"),
               Line2D([], [], marker="o", ls="", color=grey, markerfacecolor="white",
                      label="(c) hohl = Alpaca/US-ETFs ohne Teilfreistellung, gefüllt = UCITS mit"),
               Line2D([], [], marker="o", ls="", color=grey, label=f"(d) gefüllt = {SCENARIOS[0]}"),
               Line2D([], [], marker="o", ls="", color=grey, markerfacecolor=to_rgba(grey, 0.35),
                      label=f"(d) halb = {SCENARIOS[1]}"),
               Line2D([], [], marker="o", ls="", color=grey, markerfacecolor="white", label=f"(d) hohl = {SCENARIOS[2]}"),
               Patch(color="green", alpha=0.15, label="mehr Rendite UND weniger Drawdown als B&H S&P 500"),
               Line2D([], [], color="firebrick", ls="--", label="Ziel 15 % / Untergrenze 11 % (gepunktet)")]
    fig.legend(handles=handles, loc="lower center", ncol=4, fontsize=9.5, frameon=False)
    fig.suptitle("Rendite vs. maximaler Drawdown – alle Varianten (gleiche Regel: SMA200, Einstieg +1 %, "
                 "Ausstieg 0 %; rechts oben = besser)", fontsize=14, fontweight="bold")
    fig.tight_layout(rect=(0, 0.06, 1, 0.97))
    fig.savefig(OUT, dpi=110)
    print(f"saved {OUT}")


if __name__ == "__main__":
    main()
