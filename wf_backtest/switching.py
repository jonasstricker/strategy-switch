"""
Strategy Switching
===================
Once we have OOS returns from each individual strategy, we can
dynamically allocate between them.

Three modes:
  Hard switching – 100 % to the best strategy (highest rolling 6M Sharpe > 0).
  Soft switching – weight proportional to rolling Sharpe (if > 0).
  apply_switching – Hard switching MIT Meta-Level-Kosten + Hysterese.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .metrics import sharpe_ratio as sr_func, rolling_sharpe
from .strategies import apply_costs

# ── Default Parameters (Meta-Level) ─────────────────────────────────────────
MIN_HOLD_DAYS = 5          # Mindest-Haltedauer in Handelstagen
# Trade Republic: 1€ pro Order ≈ 1bp auf 10k Portfolio
SWITCH_TX     = 0.0001     # 1 bp Transaktionskosten
SWITCH_SLIP   = 0.0002     # 2 bp Slippage (liquide EU-ETFs)


def apply_switching(
    df_strats: pd.DataFrame,
    roll_sharpe_df: pd.DataFrame,
    tx: float = SWITCH_TX,
    slip: float = SWITCH_SLIP,
    min_hold: int = MIN_HOLD_DAYS,
    *,
    return_next: bool = False,
):
    """
    Meta-Level Switching mit Kosten und Hysterese.

    Timing (kein Look-Ahead): Die Entscheidung am Schluss von Tag t (Rolling
    Sharpe inkl. Tag t) bestimmt die Strategie, die an Tag t+1 gehalten wird.
    Der Return von Tag t wird also nie zur Auswahl der Strategie für Tag t
    verwendet.

    Parameters
    ----------
    df_strats : DataFrame
        Strategie-OOS-Returns (Spalten = Strategie-Namen).
    roll_sharpe_df : DataFrame
        Rolling Sharpe pro Strategie (gleicher Index/Spalten wie df_strats).
    tx : float
        Transaktionskosten pro Trade (einfach).
    slip : float
        Slippage pro Trade (einfach).
    min_hold : int
        Mindest-Haltedauer (Handelstage) bevor Cash ↔ Long wechseln darf.
    return_next : bool
        Zusätzlich die Entscheidung am letzten Schlusskurs (= Strategie für
        den nächsten Handelstag, Live-Signal) zurückgeben.

    Returns
    -------
    hard_ret : Series      – Netto-Returns nach Switching + Meta-Kosten
    active_strat : Series  – An diesem Tag gehaltene Strategie ("Cash" oder Name)
    next_strat : str       – nur mit return_next=True
    """
    cost_per_trade = tx + slip
    roll = roll_sharpe_df.reindex(df_strats.index)

    decision = pd.Series("Cash", index=df_strats.index, dtype=object)

    prev_position = "Cash"   # "Cash" oder "Long"
    hold_counter  = min_hold  # Startet erfüllt, damit 1. Trade sofort möglich

    for i, idx in enumerate(df_strats.index):
        hold_counter += 1

        row      = roll.loc[idx]
        eligible = row[row > 0]

        raw_best     = None if eligible.empty else eligible.idxmax()
        raw_position = "Cash" if raw_best is None else "Long"

        # ── Hysterese: Mindest-Haltedauer prüfen ────────────────────────
        if raw_position != prev_position and hold_counter < min_hold:
            if prev_position == "Cash":
                decision.iloc[i] = "Cash"
            elif raw_best is not None:
                decision.iloc[i] = raw_best
            else:
                # Alle Strategien negativ, ABER Hysterese erzwingt Long
                all_sharpes = row.dropna()
                if not all_sharpes.empty:
                    decision.iloc[i] = all_sharpes.idxmax()
            continue

        decision.iloc[i] = raw_best if raw_best is not None else "Cash"
        if raw_position != prev_position:
            hold_counter = 0
            prev_position = raw_position

    active_strat = decision.shift(1).fillna("Cash")

    cols = {c: j for j, c in enumerate(df_strats.columns)}
    values = df_strats.to_numpy(dtype=float)
    held_idx = active_strat.map(cols).to_numpy()
    hard = np.zeros(len(df_strats))
    for i, j in enumerate(held_idx):
        if not pd.isna(j):
            v = values[i, int(j)]
            hard[i] = 0.0 if np.isnan(v) else v
    hard_ret = pd.Series(hard, index=df_strats.index)

    # Meta-Level Kosten bei Cash ↔ Long (am ersten Tag der neuen Position)
    is_long = (active_strat != "Cash").astype(int)
    switched = is_long.diff().fillna(is_long.iloc[0] if len(is_long) else 0).abs()
    hard_ret = hard_ret - switched * cost_per_trade

    if return_next:
        next_strat = decision.iloc[-1] if len(decision) else "Cash"
        return hard_ret, active_strat, next_strat
    return hard_ret, active_strat


def held_exposure(active_strat: pd.Series, strat_pos: dict) -> pd.Series:
    """Actual market exposure per day: position of the held strategy (0 = Cash)."""
    exp = pd.Series(0.0, index=active_strat.index)
    for name, pos in strat_pos.items():
        mask = active_strat == name
        if mask.any():
            exp[mask] = pos.reindex(active_strat.index).fillna(0.0)[mask]
    return exp


def _align_returns(strat_returns: dict[str, pd.Series]) -> pd.DataFrame:
    """Align all OOS return series to a common index."""
    df = pd.DataFrame(strat_returns)
    df = df.sort_index()
    return df


def hard_switch(strat_returns: dict[str, pd.Series],
                rolling_window: int = 126,
                min_sharpe: float = 0.0,
                rf_annual: float = 0.02) -> pd.Series:
    """
    Hard switching: each day, pick the single strategy with the
    highest rolling 6M Sharpe > min_sharpe.
    If none qualifies, go to cash.

    Returns combined daily return series.
    """
    df = _align_returns(strat_returns)
    names = df.columns.tolist()

    # Rolling Sharpe up to the previous close selects today's strategy
    roll = pd.DataFrame({
        name: rolling_sharpe(df[name], rolling_window, rf_annual)
        for name in names
    }).shift(1)

    combined = pd.Series(0.0, index=df.index)
    allocation = pd.DataFrame(0.0, index=df.index, columns=names)

    for idx in df.index:
        if idx not in roll.index:
            continue
        row = roll.loc[idx]
        eligible = row[row > min_sharpe]
        if eligible.empty:
            continue
        best = eligible.idxmax()
        combined.loc[idx] = df.loc[idx, best]
        allocation.loc[idx, best] = 1.0

    return combined, allocation


def soft_switch(strat_returns: dict[str, pd.Series],
                rolling_window: int = 126,
                min_sharpe: float = 0.0,
                rf_annual: float = 0.02) -> pd.Series:
    """
    Soft switching: weight strategies proportional to their
    rolling 6M Sharpe (if > min_sharpe). Weights sum to 1.
    If none qualifies, go to cash.

    Returns combined daily return series.
    """
    df = _align_returns(strat_returns)
    names = df.columns.tolist()

    roll = pd.DataFrame({
        name: rolling_sharpe(df[name], rolling_window, rf_annual)
        for name in names
    }).shift(1)

    combined = pd.Series(0.0, index=df.index)
    allocation = pd.DataFrame(0.0, index=df.index, columns=names)

    for idx in df.index:
        if idx not in roll.index:
            continue
        row = roll.loc[idx]
        eligible = row[row > min_sharpe]
        if eligible.empty:
            continue
        weights = eligible / eligible.sum()
        for name in weights.index:
            combined.loc[idx] += weights[name] * df.loc[idx, name]
            allocation.loc[idx, name] = weights[name]

    return combined, allocation


def switching_summary(strat_returns: dict[str, pd.Series],
                      rolling_window: int = 126,
                      min_sharpe: float = 0.0,
                      rf_annual: float = 0.02) -> dict:
    """
    Run both switching modes and return summary stats.
    """
    hard_ret, hard_alloc = hard_switch(
        strat_returns, rolling_window, min_sharpe, rf_annual)
    soft_ret, soft_alloc = soft_switch(
        strat_returns, rolling_window, min_sharpe, rf_annual)

    return {
        "hard": {
            "returns": hard_ret,
            "allocation": hard_alloc,
            "sharpe": sr_func(hard_ret.dropna(), rf_annual),
        },
        "soft": {
            "returns": soft_ret,
            "allocation": soft_alloc,
            "sharpe": sr_func(soft_ret.dropna(), rf_annual),
        },
    }
