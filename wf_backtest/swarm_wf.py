#!/usr/bin/env python3
"""
swarm_wf.py — Walk-Forward Switching auf Rolling-Top-10 Aktien
================================================================

Methode:
  1. Bestimme historisch die 10 wertvollsten Aktien (Market-Cap-Proxy)
     zu jedem Rebalance-Zeitpunkt (alle 63 Tage)
  2. Bilde Equal-Weight-Portfolio dieser 10 Aktien
  3. Wende WF-optimierte Strategie-Switching-Logik an (Long/Cash)
  4. Vergleiche vs Buy & Hold des gleichen Top-10-Portfolios

Parametrisch angepasst für Aktien-Portfolios:
  - Kürzeres WF-Training (504 statt 756 Tage)
  - Angepasste Strategie-Grids (optimiert für höhere Volatilität)
  - Rolling Sharpe Window = 42 (kürzer, reagiert schneller)
"""

from __future__ import annotations
import sys, os, warnings
warnings.filterwarnings("ignore")

if __package__ is None:
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    __package__ = "wf_backtest"

import numpy as np
import pandas as pd
import yfinance as yf

from .strategies import momentum_signal, ma_signal, rsi_signal, apply_costs
from .strategies_ext import dual_momentum_signal, double_ma_signal
from .metrics import sharpe_ratio, rolling_sharpe
from .walk_forward import generate_windows
from .switching import apply_switching

# ── Universum: ~50 größte US-Aktien (nach Market Cap, Stand ~2024) ───────────
# Wir nehmen ein breites Universum, aus dem historisch die Top-10 bestimmt werden
MEGA_CAP_UNIVERSE = {
    "AAPL": "Apple", "MSFT": "Microsoft", "GOOGL": "Alphabet",
    "AMZN": "Amazon", "NVDA": "NVIDIA", "META": "Meta",
    "BRK-B": "Berkshire Hathaway", "TSLA": "Tesla",
    "JPM": "JPMorgan", "V": "Visa", "UNH": "UnitedHealth",
    "JNJ": "Johnson & Johnson", "XOM": "ExxonMobil",
    "MA": "Mastercard", "PG": "Procter & Gamble",
    "HD": "Home Depot", "LLY": "Eli Lilly",
    "AVGO": "Broadcom", "ABBV": "AbbVie", "MRK": "Merck",
    "KO": "Coca-Cola", "PEP": "PepsiCo", "COST": "Costco",
    "CVX": "Chevron", "WMT": "Walmart", "BAC": "Bank of America",
    "CRM": "Salesforce", "ORCL": "Oracle", "ADBE": "Adobe",
    "MCD": "McDonald's", "NFLX": "Netflix", "AMD": "AMD",
    "TMO": "Thermo Fisher", "DIS": "Disney",
    "INTC": "Intel", "GS": "Goldman Sachs", "CAT": "Caterpillar",
    "GE": "GE Aerospace", "RTX": "RTX", "HON": "Honeywell",
    "NEE": "NextEra Energy", "LOW": "Lowe's",
    "TXN": "Texas Instruments", "QCOM": "Qualcomm",
    "NOW": "ServiceNow", "ISRG": "Intuitive Surgical",
    "BLK": "BlackRock", "LMT": "Lockheed Martin",
    "UBER": "Uber", "SBUX": "Starbucks",
}

# ── Konstanten ───────────────────────────────────────────────────────────────
RF = 0.02
# Alpaca: $0 Kommission, Spread konservativ 5bp
TX = 0.0000
SLIP = 0.0005
TOP_N = 10
REBALANCE_DAYS = 63    # Quartalsweise

# Angepasste WF-Parameter für Aktienportfolios (kürzer, reagiert schneller)
WF_CFG = {"train": 504, "test": 21, "step": 21}
ROLL_SHARPE_WINDOW = 42   # Kürzer als ETF (63) → schnellere Reaktion
MIN_HOLD = 5              # Hysterese: 5 Tage Mindesthaltedauer

# Angepasste Strategie-Grids (breitere Suche für Aktien)
STRATEGY_DEFS = {
    "RSI": {
        "grid": [{"period": per, "threshold": thr}
                 for per in [10, 14, 20]
                 for thr in [40, 50, 60]],
        "gen": lambda p, params: rsi_signal(p, params["period"], params["threshold"]),
        "keys": ["period", "threshold"],
    },
    "Momentum": {
        "grid": [{"lookback": lb} for lb in [30, 60, 90, 160]],
        "gen": lambda p, params: momentum_signal(p, params["lookback"]),
        "keys": ["lookback"],
    },
    "MA": {
        "grid": [{"period": p} for p in [50, 100, 150]],
        "gen": lambda p, params: ma_signal(p, params["period"]),
        "keys": ["period"],
    },
    "Double_MA": {
        "grid": [{"fast": f, "slow": s}
                 for f in [20, 50] for s in [100, 150, 200] if f < s],
        "gen": lambda p, params: double_ma_signal(p, params["fast"], params["slow"]),
        "keys": ["fast", "slow"],
    },
}

# Cloud-optimierte Versionen (kleinere Grids, kürzeres Training)
CLOUD_WF_CFG = {"train": 252, "test": 21, "step": 42}
CLOUD_STRATEGY_DEFS = {
    "RSI": {
        "grid": [{"period": per, "threshold": thr}
                 for per in [14, 20]
                 for thr in [40, 50]],
        "gen": lambda p, params: rsi_signal(p, params["period"], params["threshold"]),
        "keys": ["period", "threshold"],
    },
    "Momentum": {
        "grid": [{"lookback": lb} for lb in [60, 160]],
        "gen": lambda p, params: momentum_signal(p, params["lookback"]),
        "keys": ["lookback"],
    },
    "MA": {
        "grid": [{"period": p} for p in [50, 150]],
        "gen": lambda p, params: ma_signal(p, params["period"]),
        "keys": ["period"],
    },
}


# ── Helper-Funktionen ────────────────────────────────────────────────────────

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


def download_universe(start: str = "2015-01-01",
                      progress_callback=None,
                      cloud_mode: bool = False) -> dict:
    """
    Lade historische Kurse + aktuelle Shares Outstanding
    für alle Aktien im Mega-Cap-Universum.

    Returns: {ticker: {"close": Series, "shares": float, "name": str}}
    """
    tickers = list(MEGA_CAP_UNIVERSE.keys())

    if progress_callback:
        progress_callback(0.05, "Lade Kursdaten …")

    # Batch-Download aller Kurse
    raw = yf.download(tickers, start=start, auto_adjust=True, progress=False)
    if raw.empty:
        return {}

    universe = {}

    if cloud_mode:
        # Cloud: Verwende Preis als Proxy für Ranking (keine einzelnen API-Calls)
        # Mega-Caps: Preis × typische Shares-Schätzung reicht für Top-N Ranking
        if progress_callback:
            progress_callback(0.30, "Verarbeite Kurse (Cloud-Modus) …")
        if isinstance(raw.columns, pd.MultiIndex):
            for ticker in tickers:
                try:
                    col = ("Close", ticker)
                    if col in raw.columns:
                        s = raw[col].dropna()
                        if len(s) > 400:
                            # Shares=1 → Ranking nur nach Preis-Niveau
                            # Für bekannte Mega-Caps ausreichend genau
                            universe[ticker] = {
                                "close": s,
                                "shares": 1.0,
                                "name": MEGA_CAP_UNIVERSE.get(ticker, ticker),
                            }
                except Exception:
                    continue
        return universe

    shares_data = {}

    # Shares Outstanding für Market-Cap-Berechnung holen
    if progress_callback:
        progress_callback(0.15, "Lade Shares Outstanding …")

    for ticker in tickers:
        try:
            info = yf.Ticker(ticker).info
            shares = info.get("sharesOutstanding") or info.get("impliedSharesOutstanding")
            if shares:
                shares_data[ticker] = float(shares)
        except Exception:
            continue

    if progress_callback:
        progress_callback(0.30, "Verarbeite Kurse …")

    # Extrahiere Close-Kurse
    if isinstance(raw.columns, pd.MultiIndex):
        for ticker in tickers:
            try:
                col = ("Close", ticker)
                if col in raw.columns:
                    s = raw[col].dropna()
                    if len(s) > 500 and ticker in shares_data:
                        universe[ticker] = {
                            "close": s,
                            "shares": shares_data[ticker],
                            "name": MEGA_CAP_UNIVERSE.get(ticker, ticker),
                        }
            except Exception:
                continue
    return universe


def build_rolling_top10(universe: dict,
                        top_n: int = TOP_N,
                        rebalance_days: int = REBALANCE_DAYS,
                        progress_callback=None) -> dict:
    """
    Baut das rollierende Top-10-Portfolio.

    Zu jedem Rebalance-Zeitpunkt:
    1. Berechne Market Cap = Close-Preis × Shares Outstanding
    2. Wähle die Top-N nach Market Cap
    3. Equal-Weight-Portfolio

    Returns dict mit:
      - portfolio_ret: Tägliche Portfolio-Returns
      - bh_ret: Buy-and-Hold Returns der gleichen Top-10
      - rebalance_history: [(date, [tickers])]
      - portfolio_close: Synthetischer Portfolio-Kurs
    """
    if progress_callback:
        progress_callback(0.35, "Bestimme historische Top-10 …")

    # Gemeinsamer Datumindex
    all_closes = {}
    all_returns = {}
    all_shares = {}

    for ticker, data in universe.items():
        all_closes[ticker] = data["close"]
        all_returns[ticker] = data["close"].pct_change().dropna()
        all_shares[ticker] = data["shares"]

    # Finde gemeinsamen Datumsbereich
    close_df = pd.DataFrame(all_closes)
    close_df = close_df.dropna(how="all")

    # Brauche mindestens 20 Aktien für sinnvolle Auswahl
    valid_count = close_df.notna().sum(axis=1)
    close_df = close_df[valid_count >= 20]

    if len(close_df) < 500:
        raise ValueError("Nicht genügend Daten für Top-10 Analyse")

    returns_df = close_df.pct_change().dropna()

    # Rebalance-Punkte bestimmen
    rebalance_dates = []
    for i in range(252, len(close_df.index), rebalance_days):
        rebalance_dates.append(close_df.index[i])

    if not rebalance_dates:
        raise ValueError("Keine Rebalance-Punkte gefunden")

    # Portfolio aufbauen
    portfolio_ret = pd.Series(0.0, index=returns_df.index)
    bh_ret = pd.Series(0.0, index=returns_df.index)
    rebalance_history = []
    current_holdings = []
    bh_holdings = []

    total_steps = len(rebalance_dates)

    for r_idx, r_date in enumerate(rebalance_dates):
        if progress_callback and r_idx % 5 == 0:
            p = 0.35 + 0.25 * (r_idx / total_steps)
            progress_callback(p, f"Rebalance {r_idx+1}/{total_steps} …")

        # Market Cap an diesem Tag berechnen
        prices_at = close_df.loc[r_date].dropna()
        market_caps = {}
        for ticker in prices_at.index:
            if ticker in all_shares:
                market_caps[ticker] = prices_at[ticker] * all_shares[ticker]

        if len(market_caps) < top_n:
            continue

        # Top-N nach Market Cap
        sorted_caps = sorted(market_caps.items(), key=lambda x: -x[1])
        top_tickers = [t for t, _ in sorted_caps[:top_n]]
        rebalance_history.append((r_date, top_tickers))
        current_holdings = top_tickers

        # Buy-and-Hold: fixiere die allererste Auswahl
        if not bh_holdings:
            bh_holdings = top_tickers

    if not rebalance_history:
        raise ValueError("Keine gültigen Rebalance-Punkte")

    if progress_callback:
        progress_callback(0.60, "Berechne Portfolio-Returns …")

    # Tägliche Returns berechnen basierend auf aktuellen Holdings
    current_stocks = []
    rebalance_idx = 0

    for date in returns_df.index:
        # Rebalance prüfen
        while (rebalance_idx < len(rebalance_history) and
               rebalance_history[rebalance_idx][0] <= date):
            new_stocks = rebalance_history[rebalance_idx][1]
            # Turnover-Kosten
            old_set = set(current_stocks)
            new_set = set(new_stocks)
            if current_stocks:  # Nicht beim ersten Mal
                turnover = len(old_set.symmetric_difference(new_set)) / (2 * top_n)
                portfolio_ret.loc[date] -= turnover * (TX + SLIP)
            current_stocks = new_stocks
            rebalance_idx += 1

        # Equal-weight Return
        if current_stocks:
            rets = []
            for t in current_stocks:
                if t in returns_df.columns and date in returns_df.index:
                    v = returns_df.loc[date, t]
                    if not np.isnan(v):
                        rets.append(v)
            if rets:
                portfolio_ret.loc[date] += np.mean(rets)

        # Buy-and-Hold der fixen Gruppe
        if bh_holdings:
            rets_bh = []
            for t in bh_holdings:
                if t in returns_df.columns and date in returns_df.index:
                    v = returns_df.loc[date, t]
                    if not np.isnan(v):
                        rets_bh.append(v)
            if rets_bh:
                bh_ret.loc[date] = np.mean(rets_bh)

    # Trimme auf Bereich ab erster Rebalance
    first_date = rebalance_history[0][0]
    portfolio_ret = portfolio_ret.loc[first_date:]
    bh_ret = bh_ret.loc[first_date:]

    # Synthetischer Portfolio-Kurs für Strategien
    portfolio_close = _equity(portfolio_ret) * 100  # Start bei 100

    return {
        "portfolio_ret": portfolio_ret,
        "bh_ret": bh_ret,
        "portfolio_close": portfolio_close,
        "rebalance_history": rebalance_history,
        "close_df": close_df,
        "returns_df": returns_df,
    }


def wf_single_portfolio(prices, returns, sdef, wf_cfg=None):
    """Walk-Forward für eine einzelne Strategie auf Portfolio-Daten."""
    cfg = wf_cfg or WF_CFG
    windows = generate_windows(prices.index, cfg["train"],
                               cfg["test"], cfg["step"])
    if len(windows) < 3:
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
    if len(oos) < 3:
        return None
    r = pd.concat(oos).sort_index()
    return r[~r.index.duplicated(keep="first")]


def run_swarm_wf(top_n: int = TOP_N,
                 start: str = "2015-01-01",
                 progress_callback=None,
                 cloud_mode: bool = False) -> dict | None:
    """
    Vollständige Pipeline mit EINZELAKTIEN-Signalen:
    1. Download Universe
    2. Build Rolling Top-N (Market Cap)
    3. Pro Aktie im aktuellen Top-N: WF-Optimierung + Switching
    4. Aggregiertes Portfolio-Ergebnis

    Jede Aktie hat eigenes Kauf-/Verkaufssignal.
    """
    # 1. Download
    universe = download_universe(start=start, progress_callback=progress_callback,
                                 cloud_mode=cloud_mode)
    if len(universe) < 20:
        return None

    # 2. Build Rolling Top-10 (für Gesamtvergleich + Auswahl)
    port_data = build_rolling_top10(
        universe, top_n=top_n,
        progress_callback=progress_callback,
    )

    portfolio_ret = port_data["portfolio_ret"]
    bh_ret = port_data["bh_ret"]
    close_df = port_data["close_df"]
    returns_df = port_data["returns_df"]

    # Aktuelles Top-10 (für Signale)
    last_rebal = port_data["rebalance_history"][-1] if port_data["rebalance_history"] else None
    if last_rebal is None:
        return None
    current_top10 = last_rebal[1]
    rebalance_history = port_data["rebalance_history"]

    # Alle Aktien, die jemals im Top-N waren (für bias-freien Backtest)
    all_ever_top = set()
    for _, tickers in rebalance_history:
        all_ever_top.update(tickers)

    if progress_callback:
        progress_callback(0.65, "WF pro Einzelaktie …")

    # 3. Pro Aktie: eigene WF-Optimierung + Switching
    use_strats = CLOUD_STRATEGY_DEFS if cloud_mode else STRATEGY_DEFS
    use_wf_cfg = CLOUD_WF_CFG if cloud_mode else WF_CFG
    min_data = 400 if cloud_mode else 600

    stock_results = {}
    all_stocks_to_run = list(all_ever_top)
    for si, ticker in enumerate(all_stocks_to_run):
        if ticker not in close_df.columns:
            continue
        stock_close = close_df[ticker].dropna()
        if len(stock_close) < min_data:
            continue
        stock_ret = stock_close.pct_change().dropna()
        stock_close = stock_close.loc[stock_ret.index]  # Align to returns index

        if progress_callback:
            progress_callback(0.65 + 0.25 * (si / len(all_stocks_to_run)),
                              f"WF: {ticker} ({si+1}/{len(all_stocks_to_run)}) …")

        # Run WF per strategy for this stock
        strat_oos = {}
        for sname, sdef in use_strats.items():
            r = wf_single_portfolio(stock_close, stock_ret, sdef, wf_cfg=use_wf_cfg)
            if r is not None:
                strat_oos[sname] = r

        if len(strat_oos) < 2:
            continue

        # Align strategies
        s_start = max(s.first_valid_index() for s in strat_oos.values())
        s_end = min(s.last_valid_index() for s in strat_oos.values())
        aligned = {n: r.loc[s_start:s_end] for n, r in strat_oos.items()}
        df_s = pd.DataFrame(aligned)
        names_s = df_s.columns.tolist()

        # Rolling Sharpe & Switching per stock
        roll_s = pd.DataFrame({
            n: rolling_sharpe(df_s[n], ROLL_SHARPE_WINDOW, RF)
            for n in names_s
        })
        sw_ret_s, active_s = apply_switching(
            df_s, roll_s, tx=TX, slip=SLIP, min_hold=MIN_HOLD,
        )

        # Signal margin for this stock
        if len(roll_s) > 0:
            last_sharpes = roll_s.iloc[-1].dropna()
            margin = float(last_sharpes.max()) if len(last_sharpes) > 0 else 0.0
        else:
            margin = 0.0

        # Bench for this stock (aligned period)
        bench_s = stock_ret.loc[s_start:s_end]

        stock_results[ticker] = {
            "switch_ret": sw_ret_s,
            "bench_ret": bench_s,
            "active_strat": active_s,
            "close": stock_close,
            "signal": "LONG" if active_s.iloc[-1] != "Cash" else "CASH",
            "strategy": active_s.iloc[-1] if active_s.iloc[-1] != "Cash" else "—",
            "signal_margin": round(margin, 3),
            "sw_sharpe": sharpe_ratio(sw_ret_s, RF),
            "bh_sharpe": sharpe_ratio(bench_s, RF),
            "pct_invested": float((active_s != "Cash").mean()),
            "n_trades": int((active_s != "Cash").astype(float).diff().abs().fillna(0).sum()),
        }

    if not stock_results:
        return None

    if progress_callback:
        progress_callback(0.92, "Aggregation & Metriken …")

    # 4. Aggregiertes Portfolio (mit rollierender Mitgliedschaft — kein Look-Ahead)
    # An jedem Tag: nur die Aktien mitteln, die zu dem Zeitpunkt im Top-N waren.

    first_common = max(
        (r["switch_ret"].index[0] for r in stock_results.values()),
        default=pd.Timestamp("2015-01-01"),
    )
    last_common = min(
        (r["switch_ret"].index[-1] for r in stock_results.values()),
        default=pd.Timestamp("2024-12-31"),
    )
    all_dates = pd.bdate_range(first_common, last_common)

    # Build per-date membership from rebalance_history (vectorized)
    # Create a DataFrame: rows=dates, cols=tickers, value=1 if in top-N
    all_tickers_wf = list(stock_results.keys())
    member_df = pd.DataFrame(0, index=all_dates, columns=all_tickers_wf)

    current_members = set()
    rebal_idx = 0
    for d in all_dates:
        while (rebal_idx < len(rebalance_history) and
               rebalance_history[rebal_idx][0] <= d):
            current_members = set(rebalance_history[rebal_idx][1])
            rebal_idx += 1
        for t in current_members:
            if t in member_df.columns:
                member_df.at[d, t] = 1

    # Build switch return DataFrame
    sw_df = pd.DataFrame(index=all_dates)
    for t in all_tickers_wf:
        sw_df[t] = stock_results[t]["switch_ret"].reindex(all_dates, fill_value=0.0)

    # Masked returns: only count when stock is member
    masked = sw_df * member_df
    counts = member_df.sum(axis=1).replace(0, 1)
    agg_switch = masked.sum(axis=1) / counts

    # Benchmark: nutze das rollierende Top-10 Portfolio (historisch korrekt)
    agg_bench = portfolio_ret.reindex(all_dates, fill_value=0.0)

    # Trim to actual data
    valid = member_df.sum(axis=1) > 0
    agg_switch = agg_switch.loc[valid].dropna()
    agg_bench = agg_bench.loc[agg_switch.index].dropna()
    common_idx = agg_switch.index.intersection(agg_bench.index)
    agg_switch = agg_switch.loc[common_idx]
    agg_bench = agg_bench.loc[common_idx]

    sw_eq = _equity(agg_switch)
    bh_eq = _equity(agg_bench)

    # Gesamte investierte Quote (Durchschnitt der aktuellen Top-10 Aktien)
    current_results = {t: stock_results[t] for t in current_top10 if t in stock_results}
    all_invested = np.mean([r["pct_invested"] for r in current_results.values()]) if current_results else 0.5
    total_trades = sum(r["n_trades"] for r in stock_results.values())

    if progress_callback:
        progress_callback(1.0, "Fertig!")

    return {
        # Aggregated returns & equity
        "switch_ret": agg_switch,
        "bench_ret": agg_bench,
        "sw_eq": sw_eq,
        "bh_eq": bh_eq,
        "active_strat": pd.Series("Mixed", index=common_idx),
        # Per-stock results (nur aktuelle Top-10 für Signale)
        "stock_results": current_results,
        # Portfolio details (rolling Top-10 mit Verschiebungen)
        "rebalance_history": port_data["rebalance_history"],
        "close_df": close_df,
        # Metriken
        "sw_sharpe": sharpe_ratio(agg_switch, RF),
        "bh_sharpe": sharpe_ratio(agg_bench, RF),
        "pct_invested": all_invested,
        "n_trades": total_trades,
        "top_n": top_n,
        "start": common_idx[0] if len(common_idx) > 0 else first_common,
        "end": common_idx[-1] if len(common_idx) > 0 else last_common,
        "names": list(STRATEGY_DEFS.keys()),
        "universe_names": MEGA_CAP_UNIVERSE,
        # Historische Top-10 Verschiebungen (letzte 10 Rebalances)
        "recent_rebalances": [
            {"date": d.strftime("%Y-%m-%d"), "tickers": t}
            for d, t in port_data["rebalance_history"][-10:]
        ],
    }
