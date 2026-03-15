#!/usr/bin/env python3
"""
SPY Strategy-Switching — Live Dashboard App
==============================================
Streamlit app showing:
  1. Current BUY / SELL / HOLD signal
  2. Full performance history
  3. Strategy health monitoring
  4. Trade log with dates
"""

from __future__ import annotations
import sys, os, warnings
warnings.filterwarnings("ignore")

# Fix package imports
if __package__ is None:
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    __package__ = "wf_backtest"

import numpy as np
import pandas as pd
import yfinance as yf
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from datetime import datetime, timedelta

from .strategies import (
    momentum_signal, ma_signal, rsi_signal, apply_costs,
)
from .strategies_ext import (
    dual_momentum_signal, double_ma_signal,
)
from .metrics import (
    sharpe_ratio, sortino_ratio, max_drawdown, cagr, calmar_ratio,
    drawdown_series, rolling_sharpe, time_under_water,
)
from .walk_forward import generate_windows
from .switching import apply_switching
from .swarm import (
    UNIVERSE, SCORE_WEIGHTS, fetch_fundamentals, fetch_prices,
    compute_momentum, score_stocks, select_swarm, backtest_swarm,
    run_swarm_analysis,
)
from .swarm_wf import (
    run_swarm_wf, MEGA_CAP_UNIVERSE,
)

# ── Constants ────────────────────────────────────────────────────────────────
RF = 0.02
TX = 0.001
SLIP = 0.0005

STRATEGY_DEFS = {
    "RSI": {
        "grid": [{"period": per, "threshold": thr}
                 for per in [10, 14, 20, 30]
                 for thr in [35, 45, 50, 55]],
        "gen": lambda p, params: rsi_signal(p, params["period"], params["threshold"]),
        "keys": ["period", "threshold"],
    },
    "Momentum": {
        "grid": [{"lookback": lb} for lb in [40, 90, 160, 252]],
        "gen": lambda p, params: momentum_signal(p, params["lookback"]),
        "keys": ["lookback"],
    },
    "MA": {
        "grid": [{"period": p} for p in [50, 100, 200]],
        "gen": lambda p, params: ma_signal(p, params["period"]),
        "keys": ["period"],
    },
    "Double_MA": {
        "grid": [{"fast": f, "slow": s}
                 for f in [20, 50] for s in [150, 200] if f < s],
        "gen": lambda p, params: double_ma_signal(p, params["fast"], params["slow"]),
        "keys": ["fast", "slow"],
    },
    "Dual_Momentum": {
        "grid": [{"abs_lookback": al, "trend_period": tp}
                 for al in [120, 200] for tp in [150, 252]],
        "gen": lambda p, params: dual_momentum_signal(p, params["abs_lookback"], params["trend_period"]),
        "keys": ["abs_lookback", "trend_period"],
    },
}

WF_CFG = {"train": 756, "test": 21, "step": 21}

STRAT_COLORS = {
    "RSI": "#4CAF50",
    "Momentum": "#2196F3",
    "MA": "#FF9800",
    "Double_MA": "#9C27B0",
    "Dual_Momentum": "#F44336",
    "Cash": "#9E9E9E",
}

ETF_DEFS = {
    "SPY":  {"name": "S&P 500",          "ticker": "SPY",  "start": "2000-01-01"},
    "URTH": {"name": "MSCI World",        "ticker": "URTH", "start": "2012-01-01"},
    "EEM":  {"name": "Emerging Markets",  "ticker": "EEM",  "start": "2004-01-01"},
    "VGK":  {"name": "FTSE Europe",       "ticker": "VGK",  "start": "2005-01-01"},
}


# ── Core computation ─────────────────────────────────────────────────────────

def _equity(ret):
    return (1 + ret).cumprod()


def _select_median(results, keys, top_pct=0.20):
    df = pd.DataFrame(results).sort_values("sharpe", ascending=False)
    n_top = max(1, int(len(df) * top_pct))
    top = df.head(n_top)
    out = {}
    for key in keys:
        med = top[key].median()
        all_vals = sorted(df[key].unique())
        out[key] = min(all_vals, key=lambda v: abs(v - med))
    return out


def wf_single(prices, returns, sdef):
    windows = generate_windows(prices.index, WF_CFG["train"],
                               WF_CFG["test"], WF_CFG["step"])
    if len(windows) < 4:
        return None
    oos = []
    for ts, te, os_s, os_e in windows:
        train_p = prices.iloc[ts:te]
        train_r = returns.iloc[ts:te]
        results = []
        for params in sdef["grid"]:
            try:
                sig = sdef["gen"](train_p, params)
                if sig is None or sig.isna().all():
                    continue
                net = apply_costs(sig.fillna(0), train_r, TX, SLIP).dropna()
                if len(net) < 20:
                    continue
                results.append({**params, "sharpe": sharpe_ratio(net, RF)})
            except Exception:
                continue
        if len(results) < 2:
            continue
        bp = _select_median(results, sdef["keys"])
        try:
            full_sig = sdef["gen"](prices.iloc[ts:os_e], bp)
        except Exception:
            continue
        if full_sig is None:
            continue
        test_r = returns.iloc[os_s:os_e]
        test_sig = full_sig.loc[test_r.index].fillna(0)
        oos.append(apply_costs(test_sig, test_r, TX, SLIP))
    if len(oos) < 4:
        return None
    r = pd.concat(oos).sort_index()
    return r[~r.index.duplicated(keep="first")]


@st.cache_data(ttl=3600, show_spinner=False)
def load_and_compute(ticker="SPY", start_date="2000-01-01"):
    """Download data, run WF, build switching. Cached for 1 hour."""
    raw = yf.download(ticker, start=start_date,
                      auto_adjust=True, progress=False)
    if isinstance(raw.columns, pd.MultiIndex):
        close_col = [c for c in raw.columns if c[0] == "Close"]
        close = raw[close_col[0]].dropna()
    else:
        close = raw["Close"].dropna()

    returns = close.pct_change().dropna()
    prices = close.loc[returns.index]

    # Run per-strategy WF
    strat_oos = {}
    for sname, sdef in STRATEGY_DEFS.items():
        r = wf_single(prices, returns, sdef)
        if r is not None:
            strat_oos[sname] = r

    # Align
    start = max(s.first_valid_index() for s in strat_oos.values())
    end = min(s.last_valid_index() for s in strat_oos.values())
    aligned = {n: r.loc[start:end] for n, r in strat_oos.items()}
    bench_ret = returns.loc[start:end]
    df_strats = pd.DataFrame(aligned)
    names = df_strats.columns.tolist()

    # Rolling Sharpe & switching (mit Meta-Kosten + Hysterese)
    sw = 63
    roll = pd.DataFrame({n: rolling_sharpe(df_strats[n], sw, RF) for n in names})

    hard_ret, active_strat = apply_switching(
        df_strats, roll, tx=TX, slip=SLIP, min_hold=5,
    )

    # Build trade log
    trades = []
    prev = "Cash"
    for idx in active_strat.index:
        cur = active_strat.loc[idx]
        if cur != prev:
            was_cash = (prev == "Cash")
            now_cash = (cur == "Cash")
            if was_cash and not now_cash:
                trades.append({
                    "Datum": idx, "Aktion": f"KAUF {ticker}",
                    "Grund": f"Strategie {cur} aktiv",
                    "Kurs": float(close.loc[idx]),
                    "Position": "LONG",
                })
            elif not was_cash and now_cash:
                trades.append({
                    "Datum": idx, "Aktion": f"VERKAUF {ticker}",
                    "Grund": "Keine Strategie positiv → Cash",
                    "Kurs": float(close.loc[idx]),
                    "Position": "CASH",
                })
            prev = cur

    return {
        "ticker": ticker,
        "close": close,
        "returns": returns,
        "prices": prices,
        "bench_ret": bench_ret,
        "hard_ret": hard_ret,
        "active_strat": active_strat,
        "strat_oos": strat_oos,
        "df_strats": df_strats,
        "roll_sharpe": roll,
        "trades": trades,
        "start": start,
        "end": end,
    }


# ══════════════════════════════════════════════════════════════════════════════
#  STREAMLIT APP
# ══════════════════════════════════════════════════════════════════════════════

st.set_page_config(
    page_title="SPY Strategy Switch",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Sidebar ──────────────────────────────────────────────────────────────────
st.sidebar.title("📊 Strategy Switch")
st.sidebar.markdown("---")
selected_etf = st.sidebar.selectbox(
    "📌 ETF wählen",
    list(ETF_DEFS.keys()),
    format_func=lambda x: f"{x} — {ETF_DEFS[x]['name']}",
    index=0,
)
etf_info = ETF_DEFS[selected_etf]

st.sidebar.markdown("---")
page = st.sidebar.radio(
    "Navigation",
    ["🎯 Aktuelles Signal", "📈 Performance", "🔍 Strategie-Check",
     "📋 Trade-Log", "🐝 Aktien-Schwarm", "ℹ️ Anleitung"],
    index=0,
)

st.sidebar.markdown("---")
st.sidebar.markdown(
    f"**ETF:** {selected_etf} ({etf_info['name']})  \n"
    "**Methode:** Walk-Forward Optimierung  \n"
    "**Strategien:** RSI, Momentum, MA, Double-MA, Dual-Momentum  \n"
    "**Kosten:** 0.15% pro Trade (0.10% + 0.05% Slippage)  \n"
    "**Shorts:** ❌ Nur Long oder Cash"
)

# ── Load data ────────────────────────────────────────────────────────────────
with st.spinner(f"⏳ Lade {selected_etf}-Daten und berechne Strategien …"):
    data = load_and_compute(etf_info["ticker"], etf_info["start"])

etf_ticker = selected_etf
etf_label = etf_info["name"]

close = data["close"]
bench_ret = data["bench_ret"]
hard_ret = data["hard_ret"]
active_strat = data["active_strat"]
trades = data["trades"]
strat_oos = data["strat_oos"]
roll_sharpe = data["roll_sharpe"]
start = data["start"]
end = data["end"]

bh_eq = _equity(bench_ret)
sw_eq = _equity(hard_ret)

current_strat = active_strat.iloc[-1]
current_price = float(close.iloc[-1])
current_date = close.index[-1]
is_invested = current_strat != "Cash"
last_trade = trades[-1] if trades else None


# ══════════════════════════════════════════════════════════════════════════════
#  PAGE 1: AKTUELLES SIGNAL
# ══════════════════════════════════════════════════════════════════════════════

if page == "🎯 Aktuelles Signal":
    st.title(f"🎯 Aktuelles Signal — {etf_ticker}")
    st.markdown(f"**Stand:** {current_date.strftime('%d.%m.%Y')} — **{etf_ticker} Kurs:** ${current_price:,.2f}")

    # Signal box
    col1, col2, col3 = st.columns([2, 1, 1])

    with col1:
        if is_invested:
            st.success(
                f"## ✅ HALTEN — {etf_ticker} Long\n"
                f"Aktive Strategie: **{current_strat}**\n\n"
                f"{etf_ticker} bleibt im Depot. Kein Handlungsbedarf."
            )
        else:
            st.error(
                f"## 🔴 CASH — Nicht investiert\n"
                f"Keine Strategie hat positiven Sharpe.\n\n"
                f"{etf_ticker} verkauft halten / nicht kaufen."
            )

    with col2:
        st.metric("Aktive Strategie", current_strat)
        st.metric(f"{etf_ticker} Kurs", f"${current_price:,.2f}")

    with col3:
        # Performance YTD
        ytd_start = bench_ret.index[bench_ret.index.year == current_date.year][0]
        bh_ytd = (1 + bench_ret.loc[ytd_start:]).prod() - 1
        sw_ytd = (1 + hard_ret.loc[ytd_start:]).prod() - 1
        st.metric("B&H YTD", f"{bh_ytd:.1%}")
        st.metric("Switch YTD", f"{sw_ytd:.1%}", delta=f"{sw_ytd - bh_ytd:+.1%}")

    st.markdown("---")

    # ── What to do next ──────────────────────────────────────────────────
    st.subheader("📋 Was du jetzt tun solltest")

    if is_invested:
        st.info(
            f"**Aktion: NICHTS TUN** — {etf_ticker} halten.\n\n"
            f"Die Strategie **{current_strat}** ist aktiv und {etf_ticker} bleibt im Depot. "
            f"Prüfe das Signal in 1-2 Wochen erneut."
        )
    else:
        st.warning(
            f"**Aktion: NICHT KAUFEN** — In Cash bleiben.\n\n"
            f"Keine der 5 Strategien zeigt aktuell ein positives Signal (Rolling Sharpe < 0). "
            f"Warte bis das System wieder ein Kaufsignal gibt."
        )

    # Recent trades
    st.subheader("🕐 Letzte Trades")
    if trades:
        recent = trades[-min(10, len(trades)):]
        df_recent = pd.DataFrame(recent)
        df_recent["Datum"] = pd.to_datetime(df_recent["Datum"]).dt.strftime("%d.%m.%Y")
        df_recent["Kurs"] = df_recent["Kurs"].apply(lambda x: f"${x:,.2f}")
        df_recent = df_recent[["Datum", "Aktion", "Position", "Kurs", "Grund"]]
        df_recent.columns = ["Datum", "Aktion", "Position danach", f"{etf_ticker} Kurs", "Grund"]
        st.dataframe(df_recent.iloc[::-1], width="stretch", hide_index=True)

    # ── Next strategies' status ──────────────────────────────────────────
    st.subheader("📊 Strategie-Signale (aktuell)")
    last_roll = roll_sharpe.iloc[-1] if len(roll_sharpe) > 0 else None
    if last_roll is not None:
        cols = st.columns(len(STRATEGY_DEFS))
        for i, sname in enumerate(STRATEGY_DEFS.keys()):
            with cols[i]:
                val = last_roll.get(sname, 0)
                is_active = (sname == current_strat)
                color = "🟢" if val > 0 else "🔴"
                active_marker = " ⭐" if is_active else ""
                st.metric(
                    f"{color} {sname}{active_marker}",
                    f"{val:.2f}",
                    help=f"63-Tage Rolling Sharpe. >0 = Kaufsignal"
                )


    # ── Letztes Jahr: Investiert vs Cash + Performance ──────────────────
    st.markdown("---")
    st.subheader("📅 Performance der letzten 12 Monate")

    # 1-Jahres-Fenster
    n_1y = min(252, len(hard_ret))
    hr_1y = hard_ret.iloc[-n_1y:]
    br_1y = bench_ret.iloc[-n_1y:]
    as_1y = active_strat.iloc[-n_1y:]
    cl_1y = close.loc[hr_1y.index]

    bh_eq_1y = (1 + br_1y).cumprod()
    sw_eq_1y = (1 + hr_1y).cumprod()

    ret_bh_1y = bh_eq_1y.iloc[-1] - 1
    ret_sw_1y = sw_eq_1y.iloc[-1] - 1
    dd_sw_1y = max_drawdown(sw_eq_1y)
    dd_bh_1y = max_drawdown(bh_eq_1y)
    pct_inv = float((as_1y != "Cash").mean())

    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("Switch Return", f"{ret_sw_1y:+.1%}",
              delta=f"{ret_sw_1y - ret_bh_1y:+.1%} vs B&H")
    m2.metric("B&H Return", f"{ret_bh_1y:+.1%}")
    m3.metric("Investiert", f"{pct_inv:.0%} der Zeit")
    m4.metric("Max DD Switch", f"{dd_sw_1y:.1%}")
    m5.metric("Max DD B&H", f"{dd_bh_1y:.1%}")

    # Equity + Investitions-Phasen Chart
    fig_1y = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        row_heights=[0.7, 0.3],
        vertical_spacing=0.05,
        subplot_titles=("Equity Curve (letztes Jahr)", "Long / Cash"),
    )

    fig_1y.add_trace(go.Scatter(
        x=bh_eq_1y.index, y=bh_eq_1y.values,
        name=f"Buy & Hold {etf_ticker}",
        line=dict(color="#2196F3", width=2),
    ), row=1, col=1)

    fig_1y.add_trace(go.Scatter(
        x=sw_eq_1y.index, y=sw_eq_1y.values,
        name="Strategy Switch",
        line=dict(color="#4CAF50", width=2.5),
    ), row=1, col=1)

    # Cash-Phasen als rote Flächen hinterlegen
    is_cash_1y = (as_1y == "Cash")
    cash_starts = []
    in_cash = False
    for i_cs, idx_cs in enumerate(is_cash_1y.index):
        if is_cash_1y.iloc[i_cs] and not in_cash:
            cash_starts.append(idx_cs)
            in_cash = True
        elif not is_cash_1y.iloc[i_cs] and in_cash:
            fig_1y.add_vrect(
                x0=cash_starts[-1], x1=idx_cs,
                fillcolor="rgba(255,0,0,0.08)", line_width=0,
                row=1, col=1,
            )
            in_cash = False
    if in_cash:
        fig_1y.add_vrect(
            x0=cash_starts[-1], x1=is_cash_1y.index[-1],
            fillcolor="rgba(255,0,0,0.08)", line_width=0,
            row=1, col=1,
        )

    # Long/Cash Balken
    long_mask = (as_1y != "Cash").astype(int)
    colors_1y = ["#4CAF50" if v == 1 else "#F44336" for v in long_mask.values]
    fig_1y.add_trace(go.Bar(
        x=long_mask.index, y=long_mask.values,
        marker_color=colors_1y, showlegend=False,
    ), row=2, col=1)

    fig_1y.update_yaxes(title_text="Wachstum", row=1, col=1)
    fig_1y.update_yaxes(
        title_text="Position", tickvals=[0, 1],
        ticktext=["Cash", "Long"], row=2, col=1,
    )
    fig_1y.update_layout(
        height=500, template="plotly_white", showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=-0.18),
    )
    st.plotly_chart(fig_1y, use_container_width=True)

    # Aktive Strategie in den letzten 12 Monaten
    strat_count_1y = as_1y.value_counts()
    strat_pcts = (strat_count_1y / len(as_1y) * 100).round(1)
    st.markdown("**Strategie-Verteilung (letzte 12M):** " +
                " · ".join(f"**{n}** {v:.0f}%" for n, v in strat_pcts.items()))


# ══════════════════════════════════════════════════════════════════════════════
#  PAGE 2: PERFORMANCE
# ══════════════════════════════════════════════════════════════════════════════

elif page == "📈 Performance":
    st.title("📈 Performance-Dashboard")

    # ── Mode toggle ──────────────────────────────────────────────────────
    perf_mode = st.radio(
        "Modus", ["📊 SPY Strategy Switch", "🐝 Aktien-Schwarm"],
        horizontal=True, index=0,
    )

    if perf_mode == "📊 SPY Strategy Switch":
        # ════════════════════════════════════════════════════════════════
        #  SPY Strategy Switch Performance (original)
        # ════════════════════════════════════════════════════════════════
        st.markdown(f"OOS Zeitraum: **{start.strftime('%d.%m.%Y')}** → **{end.strftime('%d.%m.%Y')}** "
                    f"({len(bench_ret)/252:.1f} Jahre)")

        # ── Key metrics ──────────────────────────────────────────────
        c1, c2, c3, c4, c5, c6 = st.columns(6)
        tuw_sw = time_under_water(sw_eq)

        with c1:
            sw_cagr = cagr(sw_eq)
            bh_cagr = cagr(bh_eq)
            st.metric("CAGR (Switch)", f"{sw_cagr:.1%}", delta=f"{sw_cagr-bh_cagr:+.1%} vs B&H")
        with c2:
            sw_sh = sharpe_ratio(hard_ret, RF)
            bh_sh = sharpe_ratio(bench_ret, RF)
            st.metric("Sharpe", f"{sw_sh:.2f}", delta=f"{sw_sh-bh_sh:+.2f} vs B&H")
        with c3:
            sw_sort = sortino_ratio(hard_ret, RF)
            st.metric("Sortino", f"{sw_sort:.2f}")
        with c4:
            sw_dd = max_drawdown(sw_eq)
            bh_dd = max_drawdown(bh_eq)
            st.metric("Max Drawdown", f"{sw_dd:.1%}", delta=f"B&H: {bh_dd:.1%}", delta_color="off")
        with c5:
            st.metric("Max Underwater", f"{tuw_sw['max_days']} Tage")
        with c6:
            final_bh = 10000 * bh_eq.iloc[-1]
            final_sw = 10000 * sw_eq.iloc[-1]
            st.metric("$10K → Switch", f"${final_sw:,.0f}", delta=f"B&H: ${final_bh:,.0f}")

        st.markdown("---")

        # ── Equity curves ────────────────────────────────────────────
        fig = make_subplots(rows=2, cols=2,
                            subplot_titles=("Equity Curves", "Drawdowns",
                                            "Rolling 1J Sharpe", "Relative Outperformance"),
                            vertical_spacing=0.12, horizontal_spacing=0.08)

        fig.add_trace(go.Scatter(x=bh_eq.index, y=bh_eq.values, name="Buy & Hold",
                                 line=dict(color="#2196F3", width=2)), row=1, col=1)
        fig.add_trace(go.Scatter(x=sw_eq.index, y=sw_eq.values, name="Strategy Switch",
                                 line=dict(color="#4CAF50", width=2, dash="dash")), row=1, col=1)

        dd_bh = drawdown_series(bh_eq)
        dd_sw = drawdown_series(sw_eq)
        fig.add_trace(go.Scatter(x=dd_bh.index, y=dd_bh.values, name="DD B&H",
                                 fill="tozeroy", line=dict(color="#2196F3", width=0.5),
                                 fillcolor="rgba(33,150,243,0.2)"), row=1, col=2)
        fig.add_trace(go.Scatter(x=dd_sw.index, y=dd_sw.values, name="DD Switch",
                                 fill="tozeroy", line=dict(color="#4CAF50", width=0.5),
                                 fillcolor="rgba(76,175,80,0.2)"), row=1, col=2)

        rs_bh = rolling_sharpe(bench_ret, 252, RF)
        rs_sw = rolling_sharpe(hard_ret, 252, RF)
        fig.add_trace(go.Scatter(x=rs_bh.index, y=rs_bh.values, name="Sharpe B&H",
                                 line=dict(color="#2196F3", width=1)), row=2, col=1)
        fig.add_trace(go.Scatter(x=rs_sw.index, y=rs_sw.values, name="Sharpe Switch",
                                 line=dict(color="#4CAF50", width=1)), row=2, col=1)
        fig.add_hline(y=0, line_dash="dot", line_color="gray", row=2, col=1)

        rel = sw_eq / bh_eq
        fig.add_trace(go.Scatter(x=rel.index, y=rel.values, name="Switch / B&H",
                                 line=dict(color="#FF9800", width=1.5)), row=2, col=2)
        fig.add_hline(y=1.0, line_dash="dot", line_color="gray", row=2, col=2)

        fig.update_layout(height=700, showlegend=True, template="plotly_white",
                          legend=dict(orientation="h", yanchor="bottom", y=-0.15))
        st.plotly_chart(fig, use_container_width=True)

        # ── Yearly comparison ────────────────────────────────────────
        st.subheader("📅 Jahresrenditen")
        bh_yearly = bench_ret.groupby(bench_ret.index.year).sum()
        sw_yearly = hard_ret.groupby(hard_ret.index.year).sum()
        years = sorted(set(bh_yearly.index) & set(sw_yearly.index))

        fig_yr = go.Figure()
        fig_yr.add_trace(go.Bar(x=years, y=[bh_yearly[y]*100 for y in years],
                                name="Buy & Hold", marker_color="#2196F3"))
        fig_yr.add_trace(go.Bar(x=years, y=[sw_yearly[y]*100 for y in years],
                                name="Switch", marker_color="#4CAF50"))
        fig_yr.update_layout(barmode="group", template="plotly_white",
                             yaxis_title="Return (%)", height=350,
                             legend=dict(orientation="h"))
        st.plotly_chart(fig_yr, use_container_width=True)

        # Heatmap
        st.subheader("🗓️ Monatsrenditen (Strategy Switch)")
        sw_monthly = hard_ret.resample("ME").sum() * 100
        pivot = pd.DataFrame({
            "Jahr": sw_monthly.index.year,
            "Monat": sw_monthly.index.month,
            "Return": sw_monthly.values,
        })
        hm = pivot.pivot_table(index="Jahr", columns="Monat", values="Return", aggfunc="mean")
        month_names = ["Jan", "Feb", "Mär", "Apr", "Mai", "Jun",
                       "Jul", "Aug", "Sep", "Okt", "Nov", "Dez"]
        hm.columns = [month_names[m-1] for m in hm.columns]

        fig_hm = px.imshow(hm, text_auto=".1f", color_continuous_scale="RdYlGn",
                           color_continuous_midpoint=0, aspect="auto")
        fig_hm.update_layout(height=400, coloraxis_colorbar=dict(title="Return %"))
        st.plotly_chart(fig_hm, use_container_width=True)

        # ── Full metrics table ───────────────────────────────────────
        st.subheader("📊 Detaillierte Kennzahlen")
        tuw_bh = time_under_water(bh_eq)
        metrics_df = pd.DataFrame({
            "Kennzahl": [
                "CAGR", "Sharpe Ratio", "Sortino Ratio", "Calmar Ratio",
                "Max Drawdown", "Max Underwater (Tage)", "Ø Underwater (Tage)",
                "Ann. Volatilität", "Bestes Jahr", "Schlechtestes Jahr",
                "Win Rate (täglich)", "Avg Win", "Avg Loss",
            ],
            f"Buy & Hold {etf_ticker}": [
                f"{cagr(bh_eq):.2%}", f"{sharpe_ratio(bench_ret, RF):.2f}",
                f"{sortino_ratio(bench_ret, RF):.2f}", f"{calmar_ratio(bh_eq):.2f}",
                f"{max_drawdown(bh_eq):.2%}", f"{tuw_bh['max_days']}",
                f"{tuw_bh['avg_days']:.0f}",
                f"{bench_ret.std()*np.sqrt(252):.2%}",
                f"{bench_ret.groupby(bench_ret.index.year).sum().max():.2%}",
                f"{bench_ret.groupby(bench_ret.index.year).sum().min():.2%}",
                f"{(bench_ret > 0).mean():.1%}",
                f"{bench_ret[bench_ret > 0].mean():.3%}",
                f"{bench_ret[bench_ret < 0].mean():.3%}",
            ],
            "Strategy Switch": [
                f"{cagr(sw_eq):.2%}", f"{sharpe_ratio(hard_ret, RF):.2f}",
                f"{sortino_ratio(hard_ret, RF):.2f}", f"{calmar_ratio(sw_eq):.2f}",
                f"{max_drawdown(sw_eq):.2%}", f"{tuw_sw['max_days']}",
                f"{tuw_sw['avg_days']:.0f}",
                f"{hard_ret.std()*np.sqrt(252):.2%}",
                f"{hard_ret.groupby(hard_ret.index.year).sum().max():.2%}",
                f"{hard_ret.groupby(hard_ret.index.year).sum().min():.2%}",
                f"{(hard_ret > 0).mean():.1%}",
                f"{hard_ret[hard_ret > 0].mean():.3%}",
                f"{hard_ret[hard_ret < 0].mean():.3%}",
            ],
        })
        st.dataframe(metrics_df, width="stretch", hide_index=True)

    else:
        # ════════════════════════════════════════════════════════════════
        #  🐝 Aktien-Schwarm Performance
        # ════════════════════════════════════════════════════════════════

        @st.cache_data(ttl=3600, show_spinner=False)
        def _load_swarm_perf(n):
            return run_swarm_analysis(top_n=n)

        sw_top_n = st.slider("Anzahl Aktien im Schwarm", 5, 20, 10,
                             key="perf_swarm_top_n")

        with st.spinner("⏳ Lade Schwarm-Daten …"):
            sdata = _load_swarm_perf(sw_top_n)

        if sdata is None:
            st.error("❌ Daten konnten nicht geladen werden.")
        else:
            swarm_ret = sdata["swarm_returns"]
            spy_sw_ret = sdata["spy_returns"]
            sw_prices = sdata["prices"]
            rebal_hist = sdata["rebalance_history"]
            sw_scored = sdata["scored"]

            # Align common dates
            if swarm_ret is not None and len(swarm_ret) > 50:
                common_idx = swarm_ret.index
                if spy_sw_ret is not None:
                    common_idx = swarm_ret.index.intersection(spy_sw_ret.index)

                swarm_eq = (1 + swarm_ret.loc[common_idx]).cumprod()
                spy_bh_eq = (1 + spy_sw_ret.loc[common_idx]).cumprod() if spy_sw_ret is not None else None

                n_years = len(common_idx) / 252
                st.markdown(
                    f"Backtest: **{common_idx[0].strftime('%d.%m.%Y')}** → "
                    f"**{common_idx[-1].strftime('%d.%m.%Y')}** ({n_years:.1f} Jahre) · "
                    f"**{sw_top_n}** Aktien · Quartalsweises Rebalancing"
                )

                # ── Key metrics ──────────────────────────────────────
                swm_cagr = cagr(swarm_eq)
                swm_sharpe = sharpe_ratio(swarm_ret.loc[common_idx], RF)
                swm_sortino = sortino_ratio(swarm_ret.loc[common_idx], RF)
                swm_dd = max_drawdown(swarm_eq)
                swm_vol = swarm_ret.loc[common_idx].std() * np.sqrt(252)
                tuw_swm = time_under_water(swarm_eq)

                bh_vals = {}
                if spy_bh_eq is not None:
                    bh_vals["cagr"] = cagr(spy_bh_eq)
                    bh_vals["sharpe"] = sharpe_ratio(spy_sw_ret.loc[common_idx], RF)
                    bh_vals["dd"] = max_drawdown(spy_bh_eq)

                mc1, mc2, mc3, mc4, mc5, mc6 = st.columns(6)
                with mc1:
                    delta_c = f"{swm_cagr - bh_vals.get('cagr', 0):+.1%} vs B&H" if bh_vals else ""
                    st.metric("CAGR (Schwarm)", f"{swm_cagr:.1%}", delta=delta_c)
                with mc2:
                    delta_s = f"{swm_sharpe - bh_vals.get('sharpe', 0):+.2f} vs B&H" if bh_vals else ""
                    st.metric("Sharpe", f"{swm_sharpe:.2f}", delta=delta_s)
                with mc3:
                    st.metric("Sortino", f"{swm_sortino:.2f}")
                with mc4:
                    delta_dd = f"B&H: {bh_vals.get('dd', 0):.1%}" if bh_vals else ""
                    st.metric("Max Drawdown", f"{swm_dd:.1%}",
                              delta=delta_dd, delta_color="off")
                with mc5:
                    st.metric("Max Underwater", f"{tuw_swm['max_days']} Tage")
                with mc6:
                    final_swm = 10000 * swarm_eq.iloc[-1]
                    final_bh_s = 10000 * spy_bh_eq.iloc[-1] if spy_bh_eq is not None else 0
                    st.metric("$10K → Schwarm", f"${final_swm:,.0f}",
                              delta=f"B&H: ${final_bh_s:,.0f}")

                st.markdown("---")

                # ── 4-panel chart ────────────────────────────────────
                fig = make_subplots(
                    rows=2, cols=2,
                    subplot_titles=("Equity Curves", "Drawdowns",
                                    "Rolling 1J Sharpe", "Relative Outperformance"),
                    vertical_spacing=0.12, horizontal_spacing=0.08,
                )
                # Equity
                if spy_bh_eq is not None:
                    fig.add_trace(go.Scatter(
                        x=spy_bh_eq.index, y=spy_bh_eq.values,
                        name="SPY Buy & Hold",
                        line=dict(color="#2196F3", width=2),
                    ), row=1, col=1)
                fig.add_trace(go.Scatter(
                    x=swarm_eq.index, y=swarm_eq.values,
                    name="🐝 Aktien-Schwarm",
                    line=dict(color="#FF6F00", width=2.5),
                ), row=1, col=1)

                # Drawdowns
                dd_swm = drawdown_series(swarm_eq)
                fig.add_trace(go.Scatter(
                    x=dd_swm.index, y=dd_swm.values, name="DD Schwarm",
                    fill="tozeroy", line=dict(color="#FF6F00", width=0.5),
                    fillcolor="rgba(255,111,0,0.2)"), row=1, col=2)
                if spy_bh_eq is not None:
                    dd_spy = drawdown_series(spy_bh_eq)
                    fig.add_trace(go.Scatter(
                        x=dd_spy.index, y=dd_spy.values, name="DD SPY B&H",
                        fill="tozeroy", line=dict(color="#2196F3", width=0.5),
                        fillcolor="rgba(33,150,243,0.15)"), row=1, col=2)

                # Rolling Sharpe
                rs_swm = rolling_sharpe(swarm_ret.loc[common_idx], 252, RF)
                fig.add_trace(go.Scatter(
                    x=rs_swm.index, y=rs_swm.values, name="Sharpe Schwarm",
                    line=dict(color="#FF6F00", width=1.5)), row=2, col=1)
                if spy_sw_ret is not None:
                    rs_spy = rolling_sharpe(spy_sw_ret.loc[common_idx], 252, RF)
                    fig.add_trace(go.Scatter(
                        x=rs_spy.index, y=rs_spy.values, name="Sharpe SPY",
                        line=dict(color="#2196F3", width=1)), row=2, col=1)
                fig.add_hline(y=0, line_dash="dot", line_color="gray", row=2, col=1)

                # Relative outperformance
                if spy_bh_eq is not None:
                    rel_swm = swarm_eq / spy_bh_eq
                    fig.add_trace(go.Scatter(
                        x=rel_swm.index, y=rel_swm.values,
                        name="Schwarm / SPY",
                        line=dict(color="#FF6F00", width=1.5)), row=2, col=2)
                    fig.add_hline(y=1.0, line_dash="dot", line_color="gray",
                                  row=2, col=2)

                fig.update_layout(height=700, showlegend=True,
                                  template="plotly_white",
                                  legend=dict(orientation="h",
                                              yanchor="bottom", y=-0.15))
                st.plotly_chart(fig, use_container_width=True)

                # ── Yearly comparison ────────────────────────────────
                st.subheader("📅 Jahresrenditen")
                swm_yearly = swarm_ret.loc[common_idx].groupby(
                    swarm_ret.loc[common_idx].index.year).sum()
                fig_yr = go.Figure()
                if spy_sw_ret is not None:
                    spy_yearly = spy_sw_ret.loc[common_idx].groupby(
                        spy_sw_ret.loc[common_idx].index.year).sum()
                    yrs = sorted(set(swm_yearly.index) & set(spy_yearly.index))
                    fig_yr.add_trace(go.Bar(
                        x=yrs, y=[spy_yearly[y] * 100 for y in yrs],
                        name="SPY Buy & Hold", marker_color="#2196F3"))
                else:
                    yrs = sorted(swm_yearly.index)
                fig_yr.add_trace(go.Bar(
                    x=yrs, y=[swm_yearly[y] * 100 for y in yrs],
                    name="🐝 Schwarm", marker_color="#FF6F00"))
                fig_yr.update_layout(barmode="group", template="plotly_white",
                                     yaxis_title="Return (%)", height=350,
                                     legend=dict(orientation="h"))
                st.plotly_chart(fig_yr, use_container_width=True)

                # ── Monthly heatmap ──────────────────────────────────
                st.subheader("🗓️ Monatsrenditen (Aktien-Schwarm)")
                swm_monthly = swarm_ret.loc[common_idx].resample("ME").sum() * 100
                pivot_s = pd.DataFrame({
                    "Jahr": swm_monthly.index.year,
                    "Monat": swm_monthly.index.month,
                    "Return": swm_monthly.values,
                })
                hm_s = pivot_s.pivot_table(index="Jahr", columns="Monat",
                                           values="Return", aggfunc="mean")
                m_names = ["Jan", "Feb", "Mär", "Apr", "Mai", "Jun",
                           "Jul", "Aug", "Sep", "Okt", "Nov", "Dez"]
                hm_s.columns = [m_names[m - 1] for m in hm_s.columns]
                fig_hm_s = px.imshow(hm_s, text_auto=".1f",
                                     color_continuous_scale="RdYlGn",
                                     color_continuous_midpoint=0, aspect="auto")
                fig_hm_s.update_layout(height=400,
                                       coloraxis_colorbar=dict(title="Return %"))
                st.plotly_chart(fig_hm_s, use_container_width=True)

                # ── Full metrics table ───────────────────────────────
                st.subheader("📊 Detaillierte Kennzahlen")
                tuw_bh_s = time_under_water(spy_bh_eq) if spy_bh_eq is not None else {"max_days": "–", "avg_days": 0}

                def _fmt_bh_spy(spy_bh_eq_local, spy_sw_ret_local, common_local, tuw_local):
                    """Build B&H column for metrics table."""
                    br = spy_sw_ret_local.loc[common_local]
                    return [
                        f"{cagr(spy_bh_eq_local):.2%}",
                        f"{sharpe_ratio(br, RF):.2f}",
                        f"{sortino_ratio(br, RF):.2f}",
                        f"{calmar_ratio(spy_bh_eq_local):.2f}",
                        f"{max_drawdown(spy_bh_eq_local):.2%}",
                        f"{tuw_local['max_days']}",
                        f"{tuw_local['avg_days']:.0f}",
                        f"{br.std() * np.sqrt(252):.2%}",
                        f"{br.groupby(br.index.year).sum().max():.2%}",
                        f"{br.groupby(br.index.year).sum().min():.2%}",
                        f"{(br > 0).mean():.1%}",
                        f"{br[br > 0].mean():.3%}",
                        f"{br[br < 0].mean():.3%}",
                    ]

                sr = swarm_ret.loc[common_idx]
                swarm_col = [
                    f"{swm_cagr:.2%}", f"{swm_sharpe:.2f}",
                    f"{swm_sortino:.2f}", f"{calmar_ratio(swarm_eq):.2f}",
                    f"{swm_dd:.2%}", f"{tuw_swm['max_days']}",
                    f"{tuw_swm['avg_days']:.0f}",
                    f"{swm_vol:.2%}",
                    f"{sr.groupby(sr.index.year).sum().max():.2%}",
                    f"{sr.groupby(sr.index.year).sum().min():.2%}",
                    f"{(sr > 0).mean():.1%}",
                    f"{sr[sr > 0].mean():.3%}",
                    f"{sr[sr < 0].mean():.3%}",
                ]

                tbl = {"Kennzahl": [
                    "CAGR", "Sharpe Ratio", "Sortino Ratio", "Calmar Ratio",
                    "Max Drawdown", "Max Underwater (Tage)", "Ø Underwater (Tage)",
                    "Ann. Volatilität", "Bestes Jahr", "Schlechtestes Jahr",
                    "Win Rate (täglich)", "Avg Win", "Avg Loss",
                ]}
                if spy_bh_eq is not None and spy_sw_ret is not None:
                    tbl["Buy & Hold SPY"] = _fmt_bh_spy(
                        spy_bh_eq, spy_sw_ret, common_idx, tuw_bh_s)
                tbl["🐝 Aktien-Schwarm"] = swarm_col
                st.dataframe(pd.DataFrame(tbl), width="stretch", hide_index=True)

                # ── Individual stock curves ──────────────────────────
                st.markdown("---")
                st.subheader("📈 Einzelaktien im letzten Schwarm")
                if rebal_hist:
                    last_picks = rebal_hist[-1][1]
                    fig_ind = go.Figure()
                    for tk in last_picks:
                        if tk in sw_prices:
                            p = sw_prices[tk].loc[common_idx[0]:]
                            p_norm = p / p.iloc[0]
                            fig_ind.add_trace(go.Scatter(
                                x=p_norm.index, y=p_norm.values,
                                name=f"{tk} ({UNIVERSE.get(tk, '')})",
                                line=dict(width=1.2),
                            ))
                    if spy_bh_eq is not None:
                        fig_ind.add_trace(go.Scatter(
                            x=spy_bh_eq.index, y=spy_bh_eq.values,
                            name="SPY", line=dict(color="black", width=2, dash="dash"),
                        ))
                    fig_ind.update_layout(
                        template="plotly_white", height=450,
                        yaxis_title="Normiert (1 = Start)",
                        legend=dict(orientation="h", yanchor="bottom", y=-0.3),
                    )
                    st.plotly_chart(fig_ind, use_container_width=True)
            else:
                st.warning("Nicht genügend Daten für Performance-Berechnung.")


# ══════════════════════════════════════════════════════════════════════════════
#  PAGE 3: STRATEGIE-CHECK
# ══════════════════════════════════════════════════════════════════════════════

elif page == "🔍 Strategie-Check":
    st.title(f"🔍 Strategie-Check — {etf_ticker}")
    st.markdown("Läuft die Strategie noch gut? Hier die Diagnose:")

    # ── Health metrics ───────────────────────────────────────────────────
    # 1. Recent rolling Sharpe
    recent_252 = hard_ret.iloc[-252:] if len(hard_ret) >= 252 else hard_ret
    recent_126 = hard_ret.iloc[-126:] if len(hard_ret) >= 126 else hard_ret
    recent_63 = hard_ret.iloc[-63:] if len(hard_ret) >= 63 else hard_ret

    sh_1y = sharpe_ratio(recent_252, RF)
    sh_6m = sharpe_ratio(recent_126, RF)
    sh_3m = sharpe_ratio(recent_63, RF)
    sh_full = sharpe_ratio(hard_ret, RF)

    # 2. Drawdown status
    dd = drawdown_series(sw_eq)
    current_dd = dd.iloc[-1]
    tuw = time_under_water(sw_eq)

    # 3. Outperformance check
    bh_1y = sharpe_ratio(bench_ret.iloc[-252:], RF) if len(bench_ret) >= 252 else 0
    outperforms_1y = sh_1y > bh_1y

    # ── Traffic light ────────────────────────────────────────────────────
    issues = []
    if sh_3m < 0:
        issues.append("⚠️ 3-Monats-Sharpe negativ")
    if sh_6m < 0:
        issues.append("⚠️ 6-Monats-Sharpe negativ")
    if sh_1y < 0:
        issues.append("🔴 1-Jahres-Sharpe negativ")
    if current_dd < -0.10:
        issues.append(f"🔴 Aktueller Drawdown: {current_dd:.1%}")
    if tuw["current_days"] > 180:
        issues.append(f"⚠️ Bereits {tuw['current_days']} Tage unter Wasser")
    if not outperforms_1y:
        issues.append("⚠️ Switch underperformt B&H über 1 Jahr")

    if len(issues) == 0:
        health = "🟢 GESUND"
        health_color = "green"
        health_msg = "Alle Indikatoren positiv. Strategie läuft normal."
    elif len(issues) <= 2 and "🔴" not in str(issues):
        health = "🟡 ACHTUNG"
        health_color = "orange"
        health_msg = "Einige Warnsignale. Strategie beobachten."
    else:
        health = "🔴 KRITISCH"
        health_color = "red"
        health_msg = "Mehrere negative Signale. Strategie überprüfen!"

    st.markdown(f"### Status: {health}")
    st.markdown(f"*{health_msg}*")

    if issues:
        st.markdown("**Erkannte Probleme:**")
        for issue in issues:
            st.markdown(f"- {issue}")

    st.markdown("---")

    # ── Sharpe over time windows ─────────────────────────────────────────
    st.subheader("📊 Sharpe-Ratio über verschiedene Zeiträume")

    time_windows = {
        "3 Monate": sh_3m,
        "6 Monate": sh_6m,
        "1 Jahr": sh_1y,
        "Gesamt": sh_full,
    }
    bh_windows = {
        "3 Monate": sharpe_ratio(bench_ret.iloc[-63:], RF) if len(bench_ret) >= 63 else 0,
        "6 Monate": sharpe_ratio(bench_ret.iloc[-126:], RF) if len(bench_ret) >= 126 else 0,
        "1 Jahr": bh_1y,
        "Gesamt": sharpe_ratio(bench_ret, RF),
    }

    fig_health = go.Figure()
    fig_health.add_trace(go.Bar(
        x=list(time_windows.keys()),
        y=list(time_windows.values()),
        name="Strategy Switch",
        marker_color=["#4CAF50" if v > 0 else "#F44336" for v in time_windows.values()],
    ))
    fig_health.add_trace(go.Bar(
        x=list(bh_windows.keys()),
        y=list(bh_windows.values()),
        name="Buy & Hold",
        marker_color="#2196F3",
        opacity=0.5,
    ))
    fig_health.update_layout(barmode="group", template="plotly_white",
                             yaxis_title="Sharpe Ratio", height=350)
    fig_health.add_hline(y=0, line_dash="dot", line_color="gray")
    st.plotly_chart(fig_health, use_container_width=True)

    # ── Rolling Sharpe detail ────────────────────────────────────────────
    st.subheader("📉 Rolling Sharpe (1 Jahr)")
    rs_sw = rolling_sharpe(hard_ret, 252, RF)
    rs_bh = rolling_sharpe(bench_ret, 252, RF)

    fig_rs = go.Figure()
    fig_rs.add_trace(go.Scatter(x=rs_sw.index, y=rs_sw.values, name="Switch",
                                line=dict(color="#4CAF50", width=1.5)))
    fig_rs.add_trace(go.Scatter(x=rs_bh.index, y=rs_bh.values, name="B&H",
                                line=dict(color="#2196F3", width=1)))
    fig_rs.add_hline(y=0, line_dash="dot", line_color="red")
    fig_rs.update_layout(template="plotly_white", height=350,
                         yaxis_title="Rolling 1Y Sharpe")
    st.plotly_chart(fig_rs, use_container_width=True)

    # ── Strategy allocation timeline ─────────────────────────────────────
    st.subheader("🔄 Strategie-Allokation über Zeit")

    # Build allocation data for stacked area
    strat_names = list(STRATEGY_DEFS.keys()) + ["Cash"]
    monthly_alloc = active_strat.resample("ME").agg(
        lambda x: x.mode()[0] if len(x) > 0 else "Cash")

    fig_alloc = go.Figure()
    for sn in strat_names:
        mask = (monthly_alloc == sn).astype(int)
        fig_alloc.add_trace(go.Bar(
            x=monthly_alloc.index, y=mask.values, name=sn,
            marker_color=STRAT_COLORS.get(sn, "#888"),
        ))
    fig_alloc.update_layout(barmode="stack", template="plotly_white",
                            height=300, yaxis_title="Aktiv",
                            showlegend=True, bargap=0)
    st.plotly_chart(fig_alloc, use_container_width=True)

    # Allocation %
    st.subheader("📊 Strategie-Verteilung (Gesamtzeit)")
    alloc = active_strat.value_counts()
    fig_pie = px.pie(values=alloc.values, names=alloc.index,
                     color=alloc.index,
                     color_discrete_map=STRAT_COLORS)
    fig_pie.update_layout(height=350)
    st.plotly_chart(fig_pie, use_container_width=True)

    # ── Individual strategy performance ──────────────────────────────────
    st.subheader("🏆 Einzelstrategie-Performance")
    strat_data = []
    for sn in sorted(strat_oos, key=lambda x: -sharpe_ratio(strat_oos[x].loc[start:end], RF)):
        r = strat_oos[sn].loc[start:end]
        eq = _equity(r)
        strat_data.append({
            "Strategie": sn,
            "Sharpe": f"{sharpe_ratio(r, RF):.2f}",
            "CAGR": f"{cagr(eq):.2%}",
            "Max DD": f"{max_drawdown(eq):.2%}",
            "Sortino": f"{sortino_ratio(r, RF):.2f}",
        })
    st.dataframe(pd.DataFrame(strat_data), width="stretch", hide_index=True)


# ══════════════════════════════════════════════════════════════════════════════
#  PAGE 4: TRADE LOG
# ══════════════════════════════════════════════════════════════════════════════

elif page == "📋 Trade-Log":
    st.title(f"📋 Trade-Log — {etf_ticker}")
    st.markdown(
        f"Nur **echte Trades** (KAUF/VERKAUF {etf_ticker}). "
        f"Strategie-Wechsel (z.B. RSI→Momentum) kosten nichts — du hältst in beiden Fällen {etf_ticker}."
    )

    # Summary
    n_buys = sum(1 for t in trades if t["Aktion"].startswith("KAUF"))
    n_sells = sum(1 for t in trades if t["Aktion"].startswith("VERKAUF"))
    n_years = len(bench_ret) / 252

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Gesamte Trades", len(trades))
    c2.metric("Käufe", n_buys)
    c3.metric("Verkäufe", n_sells)
    c4.metric("Ø Trades/Jahr", f"{len(trades)/n_years:.1f}")
    c5.metric("Kosten/Trade", "0.15%")

    st.markdown("---")

    # Round-trip analysis
    st.subheader("🔄 Round-Trips")
    roundtrips = []
    buy_dates = [t for t in trades if t["Aktion"].startswith("KAUF")]
    sell_dates = [t for t in trades if t["Aktion"].startswith("VERKAUF")]

    for i, (buy, sell) in enumerate(zip(buy_dates, sell_dates)):
        days = (sell["Datum"] - buy["Datum"]).days
        ret = (sell["Kurs"] / buy["Kurs"] - 1)
        cost = 2 * (TX + SLIP)
        net_ret = ret - cost
        roundtrips.append({
            "#": i + 1,
            "Kauf": buy["Datum"].strftime("%d.%m.%Y"),
            "Verkauf": sell["Datum"].strftime("%d.%m.%Y"),
            "Tage": days,
            "Kauf $": f"${buy['Kurs']:,.2f}",
            "Verkauf $": f"${sell['Kurs']:,.2f}",
            "Brutto": f"{ret:+.2%}",
            "Netto (−0.30%)": f"{net_ret:+.2%}",
            "Gewinn?": "✅" if net_ret > 0 else "❌",
        })

    if roundtrips:
        df_rt = pd.DataFrame(roundtrips)
        wins = sum(1 for r in roundtrips if "✅" in r["Gewinn?"])
        st.markdown(f"**{wins} von {len(roundtrips)} Round-Trips profitabel ({wins/len(roundtrips):.0%})**")

        # Holding period stats
        hold_days = [(sell["Datum"] - buy["Datum"]).days
                     for buy, sell in zip(buy_dates, sell_dates)]
        sc1, sc2, sc3, sc4 = st.columns(4)
        sc1.metric("Ø Haltedauer", f"{np.mean(hold_days):.0f} Tage")
        sc2.metric("Median", f"{np.median(hold_days):.0f} Tage")
        sc3.metric("Längste", f"{max(hold_days)} Tage")
        sc4.metric("Kürzeste", f"{min(hold_days)} Tage")

        st.dataframe(df_rt, width="stretch", hide_index=True)

    # Full trade log
    st.subheader("📜 Alle Trades (chronologisch)")
    if trades:
        df_trades = pd.DataFrame(trades)
        df_trades["Datum"] = pd.to_datetime(df_trades["Datum"]).dt.strftime("%d.%m.%Y")
        df_trades["Kurs"] = df_trades["Kurs"].apply(lambda x: f"${x:,.2f}")
        df_trades["Kosten"] = "0.15%"
        df_trades = df_trades[["Datum", "Aktion", "Kurs", "Kosten", "Position", "Grund"]]
        st.dataframe(df_trades.iloc[::-1], width="stretch", hide_index=True)

    # Download button
    if trades:
        csv = pd.DataFrame(trades).to_csv(index=False, sep=";")
        st.download_button("📥 Trade-Log als CSV herunterladen", csv,
                           "spy_trade_log.csv", "text/csv")


# ══════════════════════════════════════════════════════════════════════════════
#  PAGE 5: AKTIEN-SCHWARM (WF-optimiert, Rolling Top-10 nach Market Cap)
# ══════════════════════════════════════════════════════════════════════════════

elif page == "🐝 Aktien-Schwarm":
    st.title("🐝 Aktien-Schwarm — WF Strategy-Switching auf Top-10")
    st.markdown(
        "Historisch rollierende **Top-10 US-Aktien** nach Market Cap. "
        "Quartalsweise Rotation (63 Tage) + Walk-Forward-optimiertes "
        "Strategy-Switching (Long/Cash) mit Hysterese."
    )

    # ── Settings ─────────────────────────────────────────────────────────
    sc1, sc2 = st.columns(2)
    with sc1:
        swarm_top_n = st.slider("Anzahl Aktien im Portfolio", 5, 20, 10,
                                key="swarm_topn")
    with sc2:
        st.markdown(f"**Universum:** {len(MEGA_CAP_UNIVERSE)} Mega-Cap-Aktien")
        st.markdown("**Methode:** WF-optimiertes Long/Cash-Switching")

    # ── Cached WF-swarm computation ──────────────────────────────────────
    @st.cache_data(ttl=3600, show_spinner=False)
    def load_swarm_wf(top_n_param):
        return run_swarm_wf(top_n=top_n_param, start="2015-01-01")

    prog_bar = st.progress(0, text="⏳ Starte Analyse …")

    with st.spinner("⏳ Lade ~50 Aktien & berechne WF-Switching …"):
        swarm_wf_data = load_swarm_wf(swarm_top_n)

    prog_bar.empty()

    if swarm_wf_data is None:
        st.error("❌ Konnte keine Daten laden. Bitte später erneut versuchen.")
    else:
        sw_ret    = swarm_wf_data["switch_ret"]
        bh_ret_sw = swarm_wf_data["bench_ret"]
        fixed_bh  = swarm_wf_data["bh_fixed_ret"]
        sw_eq_sw  = swarm_wf_data["sw_eq"]
        bh_eq_sw  = swarm_wf_data["bh_eq"]
        fixed_eq  = swarm_wf_data["fixed_bh_eq"]
        active_sw = swarm_wf_data["active_strat"]
        rebalance_hist_sw = swarm_wf_data["rebalance_history"]
        strat_oos_sw = swarm_wf_data["strat_oos"]
        roll_sharpe_sw = swarm_wf_data["roll_sharpe"]

        # ── Tab layout ───────────────────────────────────────────────────
        tab_sig, tab_perf, tab_rebal, tab_wf = st.tabs([
            "🎯 Signal", "📈 Performance", "🔄 Rebalancing", "🔬 WF-Details"
        ])

        # ══════════════ TAB 1: Aktuelles Signal ═════════════════════════
        with tab_sig:
            curr_strat = active_sw.iloc[-1] if len(active_sw) else "?"
            is_invested = curr_strat != "Cash"

            st.markdown("### Aktuelles Signal")
            if is_invested:
                st.success(f"🟢 **LONG** — Aktive Strategie: {curr_strat}")
            else:
                st.error("🔴 **CASH** — Schwarm-Portfolio nicht investiert")

            # Aktuelle Top-10
            if rebalance_hist_sw:
                last_date, last_tickers = rebalance_hist_sw[-1]
                st.markdown(f"**Aktuelle Top-{swarm_top_n} (seit {last_date.strftime('%d.%m.%Y')}):**")
                cols_cur = st.columns(min(5, len(last_tickers)))
                for j, t in enumerate(last_tickers):
                    nm = MEGA_CAP_UNIVERSE.get(t, t)
                    cols_cur[j % min(5, len(last_tickers))].markdown(f"**{t}**\n{nm}")

            # Key metrics
            st.markdown("---")
            st.markdown("### Kennzahlen (gesamter Zeitraum)")
            m1, m2, m3, m4, m5 = st.columns(5)
            m1.metric("Sharpe Switch", f"{swarm_wf_data['sw_sharpe']:.2f}")
            m2.metric("Sharpe B&H", f"{swarm_wf_data['bh_sharpe']:.2f}")
            m3.metric("% Investiert", f"{swarm_wf_data['pct_invested']:.0%}")
            m4.metric("Meta-Trades", f"{swarm_wf_data['n_trades']}")
            m5.metric("CAGR Switch", f"{cagr(sw_eq_sw):.1%}")

            # Sharpe delta
            delta = swarm_wf_data["sw_sharpe"] - swarm_wf_data["bh_sharpe"]
            if delta > 0:
                st.success(f"✅ Switching schlägt B&H um **{delta:+.2f} Sharpe-Punkte**")
            else:
                st.warning(f"⚠️ B&H besser um {abs(delta):.2f} Sharpe-Punkte — Parameter prüfen")

        # ══════════════ TAB 2: Performance ══════════════════════════════
        with tab_perf:
            st.subheader("WF-Switching vs. Buy & Hold")

            # Equity Curves
            fig_eq = go.Figure()
            fig_eq.add_trace(go.Scatter(
                x=sw_eq_sw.index, y=sw_eq_sw.values,
                name="🔀 WF-Switching", line=dict(color="#4CAF50", width=2.5),
            ))
            fig_eq.add_trace(go.Scatter(
                x=bh_eq_sw.index, y=bh_eq_sw.values,
                name="📊 Rolling Top-10 B&H", line=dict(color="#2196F3", width=1.5, dash="dash"),
            ))
            fig_eq.add_trace(go.Scatter(
                x=fixed_eq.index, y=fixed_eq.values,
                name="📌 Fixe Top-10 B&H", line=dict(color="#FF9800", width=1.5, dash="dot"),
            ))

            # Cash-Phasen als rote Bereiche
            cash_mask = (active_sw == "Cash")
            if cash_mask.any():
                changes = cash_mask.astype(int).diff().fillna(0)
                starts = active_sw.index[changes == 1]
                ends = active_sw.index[changes == -1]
                if cash_mask.iloc[0]:
                    starts = starts.insert(0, active_sw.index[0])
                if cash_mask.iloc[-1]:
                    ends = ends.append(pd.DatetimeIndex([active_sw.index[-1]]))
                for s, e in zip(starts, ends):
                    fig_eq.add_vrect(
                        x0=s, x1=e, fillcolor="red", opacity=0.08,
                        layer="below", line_width=0,
                    )

            fig_eq.update_layout(
                template="plotly_white", height=500,
                yaxis_title="Wachstum (1€)",
                legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01),
                title="Equity Curves (rote Bereiche = Cash)",
            )
            st.plotly_chart(fig_eq, use_container_width=True)

            # Metrics table
            st.markdown("---")
            st.subheader("Vergleich Kennzahlen")

            sw_cagr_v = cagr(sw_eq_sw)
            bh_cagr_v = cagr(bh_eq_sw)
            fx_cagr_v = cagr(fixed_eq)
            sw_dd_v = max_drawdown(sw_eq_sw)
            bh_dd_v = max_drawdown(bh_eq_sw)
            fx_dd_v = max_drawdown(fixed_eq)
            sw_vol = sw_ret.std() * np.sqrt(252)
            bh_vol = bh_ret_sw.std() * np.sqrt(252)
            fx_vol = fixed_bh.std() * np.sqrt(252)

            comp_df = pd.DataFrame({
                "Methode": ["🔀 WF-Switching", "📊 Rolling Top-10 B&H", "📌 Fixe Top-10 B&H"],
                "CAGR": [f"{sw_cagr_v:.1%}", f"{bh_cagr_v:.1%}", f"{fx_cagr_v:.1%}"],
                "Sharpe": [f"{swarm_wf_data['sw_sharpe']:.2f}",
                           f"{swarm_wf_data['bh_sharpe']:.2f}",
                           f"{swarm_wf_data['fixed_bh_sharpe']:.2f}"],
                "Max DD": [f"{sw_dd_v:.1%}", f"{bh_dd_v:.1%}", f"{fx_dd_v:.1%}"],
                "Volatilität": [f"{sw_vol:.1%}", f"{bh_vol:.1%}", f"{fx_vol:.1%}"],
            })
            st.dataframe(comp_df, hide_index=True, width="stretch")

            # Yearly comparison
            st.markdown("---")
            st.subheader("Jahres-Vergleich")
            sw_yr = sw_ret.resample("YE").apply(lambda x: (1 + x).prod() - 1)
            bh_yr = bh_ret_sw.resample("YE").apply(lambda x: (1 + x).prod() - 1)
            yr_df = pd.DataFrame({
                "Jahr": sw_yr.index.year,
                "Switch": sw_yr.values,
                "Top-10 B&H": bh_yr.reindex(sw_yr.index).values,
            })
            yr_df["Diff"] = yr_df["Switch"] - yr_df["Top-10 B&H"]
            for col in ["Switch", "Top-10 B&H", "Diff"]:
                yr_df[col] = yr_df[col].apply(lambda v: f"{v:+.1%}" if not np.isnan(v) else "–")
            st.dataframe(yr_df, hide_index=True, width="stretch")

            # Strategy distribution
            st.markdown("---")
            st.subheader("Strategie-Verteilung")
            strat_dist = active_sw.value_counts(normalize=True).sort_values(ascending=False)
            fig_bar = px.bar(
                x=strat_dist.index, y=strat_dist.values * 100,
                labels={"x": "Strategie", "y": "Anteil (%)"},
                title="Wie oft wurde welche Strategie genutzt?",
                color=strat_dist.index,
            )
            fig_bar.update_layout(template="plotly_white", height=350, showlegend=False)
            st.plotly_chart(fig_bar, use_container_width=True)

        # ══════════════ TAB 3: Rebalancing ═════════════════════════════
        with tab_rebal:
            st.subheader("🔄 Rebalancing-Verlauf (nach Market Cap)")
            st.markdown(
                f"**{len(rebalance_hist_sw)}** Rebalancing-Punkte seit Start. "
                f"Alle ~63 Handelstage werden die **Top-{swarm_top_n}** "
                f"nach Market Cap neu bestimmt."
            )

            if rebalance_hist_sw:
                recent = rebalance_hist_sw[-8:]
                for rb_date, rb_tickers in reversed(recent):
                    rb_names = [MEGA_CAP_UNIVERSE.get(t, t) for t in rb_tickers]
                    with st.expander(
                        f"📅 {rb_date.strftime('%d.%m.%Y')} — {len(rb_tickers)} Aktien",
                        expanded=(rb_date == recent[-1][0]),
                    ):
                        cols_rb = st.columns(min(5, len(rb_tickers)))
                        for j, (t, n) in enumerate(zip(rb_tickers, rb_names)):
                            cols_rb[j % min(5, len(rb_tickers))].markdown(
                                f"**{t}**  \n{n}"
                            )

                # Turnover
                st.markdown("---")
                st.subheader("Umschlag (Turnover)")
                turnovers = []
                for i_rb in range(1, len(rebalance_hist_sw)):
                    old_s = set(rebalance_hist_sw[i_rb - 1][1])
                    new_s = set(rebalance_hist_sw[i_rb][1])
                    turnovers.append({
                        "Datum": rebalance_hist_sw[i_rb][0].strftime("%d.%m.%Y"),
                        "Neu": len(new_s - old_s),
                        "Raus": len(old_s - new_s),
                        "Turnover": f"{(len(new_s - old_s) + len(old_s - new_s)) / (2 * swarm_top_n):.0%}",
                    })
                if turnovers:
                    df_to = pd.DataFrame(turnovers)
                    st.dataframe(df_to.iloc[::-1].head(20), width="stretch", hide_index=True)

                # Holding frequency
                st.markdown("---")
                st.subheader("Häufigste Aktien im Portfolio")
                all_held = [t for _, tl in rebalance_hist_sw for t in tl]
                from collections import Counter
                freq = Counter(all_held).most_common(20)
                freq_df = pd.DataFrame(freq, columns=["Ticker", "Mal im Portfolio"])
                freq_df["Name"] = freq_df["Ticker"].map(
                    lambda t: MEGA_CAP_UNIVERSE.get(t, t))
                freq_df["Anteil"] = freq_df["Mal im Portfolio"].apply(
                    lambda x: f"{x/len(rebalance_hist_sw):.0%}")
                st.dataframe(freq_df[["Ticker", "Name", "Mal im Portfolio", "Anteil"]],
                            hide_index=True, width="stretch")

        # ══════════════ TAB 4: WF-Details ══════════════════════════════
        with tab_wf:
            st.subheader("🔬 Walk-Forward Details")

            st.markdown(
                f"**WF-Parameter:** Train={504}d, Test={21}d, Step={21}d  \n"
                f"**Rolling Sharpe Fenster:** {42}d  \n"
                f"**Min. Haltedauer (Hysterese):** {5}d  \n"
                f"**Strategien:** {', '.join(swarm_wf_data['names'])}"
            )

            # Rolling Sharpe per strategy
            st.markdown("---")
            st.subheader("Rolling Sharpe je Strategie")
            fig_rs = go.Figure()
            colors = ["#4CAF50", "#2196F3", "#FF9800", "#E91E63", "#9C27B0"]
            for i, sn in enumerate(roll_sharpe_sw.columns):
                fig_rs.add_trace(go.Scatter(
                    x=roll_sharpe_sw.index, y=roll_sharpe_sw[sn],
                    name=sn, line=dict(color=colors[i % len(colors)], width=1.2),
                ))
            fig_rs.add_hline(y=0, line_dash="dash", line_color="gray")
            fig_rs.update_layout(
                template="plotly_white", height=400,
                yaxis_title="Sharpe (rollierend)",
                title="Rolling Sharpe — Beste Strategie wird automatisch gewählt",
            )
            st.plotly_chart(fig_rs, use_container_width=True)

            # Active strategy timeline
            st.markdown("---")
            st.subheader("Aktive Strategie über Zeit")
            all_strats = sorted(active_sw.unique())
            strat_map = {s: i for i, s in enumerate(all_strats)}
            strat_num = active_sw.map(strat_map)
            fig_at = go.Figure()
            fig_at.add_trace(go.Scatter(
                x=strat_num.index, y=strat_num.values,
                mode="lines", line=dict(color="#4CAF50", width=1),
                fill="tozeroy", fillcolor="rgba(76,175,80,0.15)",
            ))
            fig_at.update_layout(
                template="plotly_white", height=300,
                yaxis=dict(
                    tickvals=list(strat_map.values()),
                    ticktext=list(strat_map.keys()),
                ),
                title="Strategie-Wechsel im Zeitverlauf",
            )
            st.plotly_chart(fig_at, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
#  PAGE 6: ANLEITUNG
# ══════════════════════════════════════════════════════════════════════════════

elif page == "ℹ️ Anleitung":
    st.title("ℹ️ Anleitung")

    st.markdown("""
    ## Wie funktioniert das System?

    ### Konzept
    Das System ist ein **taktisches Long/Cash-System** für den SPY ETF (S&P 500).
    Es werden **keine Shorts** eingegangen — du bist entweder in SPY investiert
    oder in Cash.

    ### Die 5 Strategien
    | Strategie | Beschreibung |
    |---|---|
    | **RSI** | Kauft wenn RSI-Indikator über Schwelle (Momentum-Filter) |
    | **Momentum** | Kauft wenn kumulative Rendite über Lookback > 0 |
    | **MA** | Kauft wenn Kurs über gleitendem Durchschnitt |
    | **Double MA** | Kauft wenn schneller MA über langsamem MA |
    | **Dual Momentum** | Kombination aus absolutem und Trend-Momentum |

    ### Walk-Forward Optimierung
    - **Training:** 3 Jahre (~756 Tage)
    - **Test:** 1 Monat (~21 Tage)
    - Für jede Strategie werden die besten Parameter auf den Trainingsdaten gesucht
    - Der **Median** der Top-20% wird verwendet (kein Overfitting)
    - Nur Out-of-Sample Renditen zählen

    ### Switching-Logik
    - Alle 63 Tage wird der **Rolling Sharpe** jeder Strategie berechnet
    - Die Strategie mit dem **höchsten positiven Sharpe** wird gewählt
    - Wenn **keine** Strategie positiven Sharpe hat → **Cash**
    - Strategie-Wechsel (z.B. RSI→Momentum) erfordern **keinen Trade** (beides = Long SPY)

    ### Kosten
    - **0.10%** Transaktionskosten pro Trade
    - **0.05%** Slippage pro Trade
    - **= 0.15%** pro Kauf oder Verkauf
    - **= 0.30%** Round-Trip (Kauf + Verkauf)
    - Kosten sind **bereits in der Performance eingerechnet**

    ### So nutzt du das System
    1. **Öffne die App** (oder führe das Skript aus)
    2. **Prüfe das aktuelle Signal** auf der Startseite
    3. **Handle nur bei echten KAUF/VERKAUF-Signalen:**
       - Signal zeigt Strategie → SPY kaufen/halten
       - Signal zeigt Cash → SPY verkaufen
    4. **Prüfe alle 1-2 Wochen** ob sich das Signal geändert hat
    5. **Nutze den Strategie-Check** um zu sehen ob das System noch gut läuft

    ### App starten
    ```bash
    cd "C:\\Users\\strickej\\Product Spice model UI\\Privat_test"
    .venv\\Scripts\\activate
    streamlit run wf_backtest/app.py
    ```

    ### Wichtige Hinweise
    - ⚠️ **Vergangene Performance garantiert keine zukünftige Rendite**
    - ⚠️ Das System ist ein **Werkzeug**, keine Anlageberatung
    - ⚠️ Immer eigene Due Diligence machen
    """)
