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
) -> tuple[pd.Series, pd.Series]:
    """
    Meta-Level Switching mit Kosten und Hysterese.

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

    Returns
    -------
    hard_ret : Series      – Netto-Returns nach Switching + Meta-Kosten
    active_strat : Series  – Aktive Strategie pro Tag ("Cash" oder Name)
    """
    cost_per_trade = tx + slip

    hard_ret     = pd.Series(0.0, index=df_strats.index)
    active_strat = pd.Series("Cash", index=df_strats.index)

    prev_position = "Cash"   # "Cash" oder "Long"
    hold_counter  = min_hold  # Startet erfüllt, damit 1. Trade sofort möglich

    for i, idx in enumerate(df_strats.index):
        hold_counter += 1

        # ── Rohsignal bestimmen ──────────────────────────────────────────
        if idx in roll_sharpe_df.index:
            row      = roll_sharpe_df.loc[idx]
            eligible = row[row > 0]
        else:
            eligible = pd.Series(dtype=float)

        raw_best     = None if eligible.empty else eligible.idxmax()
        raw_position = "Cash" if raw_best is None else "Long"

        # ── Hysterese: Mindest-Haltedauer prüfen ────────────────────────
        if raw_position != prev_position and hold_counter < min_hold:
            # Wechsel blockiert → bleibe bei alter Position
            if prev_position == "Cash":
                active_strat.iloc[i] = "Cash"
                hard_ret.iloc[i]     = 0.0
            else:
                # Bleibe Long → nutze beste verfügbare Strategie
                if raw_best is not None:
                    active_strat.iloc[i] = raw_best
                    hard_ret.iloc[i]     = df_strats.loc[idx, raw_best]
                else:
                    # Alle Strategien negativ, ABER Hysterese erzwingt Long
                    if idx in roll_sharpe_df.index:
                        all_sharpes = roll_sharpe_df.loc[idx].dropna()
                        if not all_sharpes.empty:
                            least_bad = all_sharpes.idxmax()
                            active_strat.iloc[i] = least_bad
                            hard_ret.iloc[i]     = df_strats.loc[idx, least_bad]
            continue

        # ── Kein Hysterese-Block → normaler Ablauf ──────────────────────
        if raw_best is not None:
            active_strat.iloc[i] = raw_best
            hard_ret.iloc[i]     = df_strats.loc[idx, raw_best]
            new_position = "Long"
        else:
            active_strat.iloc[i] = "Cash"
            hard_ret.iloc[i]     = 0.0
            new_position = "Cash"

        # ── Meta-Level Kosten bei Cash ↔ Long ───────────────────────────
        if new_position != prev_position:
            hard_ret.iloc[i] -= cost_per_trade
            hold_counter = 0   # Reset Haltedauer-Zähler
            prev_position = new_position

    return hard_ret, active_strat


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

    # Rolling Sharpe for each strategy
    roll = pd.DataFrame({
        name: rolling_sharpe(df[name], rolling_window, rf_annual)
        for name in names
    })

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
    })

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
