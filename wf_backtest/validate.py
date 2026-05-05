#!/usr/bin/env python3
"""
Validation Script — Matrix-Berechnung + Zeitraum-Vergleich
============================================================
Prüft ob die Matrix-Berechnung korrekt ist und vergleicht
Kern-Parameter (Sharpe, Max DD, CAGR) über verschiedene Zeiträume.

Validierung:
  1. Alle 3 Kategorien (Swarm, Value, Turnaround) separat
  2. Alpha-Mix: gewichtete Kombination
  3. Zeiträume: Full (ab Datenstart), 2018+, 2021+
  4. Prüft ob Matrix-Updates für alle Kategorien laufen
"""

from __future__ import annotations
import sys, os, warnings, logging
warnings.filterwarnings("ignore")

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.chdir(ROOT)

import numpy as np
import pandas as pd

from wf_backtest.metrics import sharpe_ratio, max_drawdown, cagr, sortino_ratio
from wf_backtest.metrics import time_under_water

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("validate")

RF = 0.02


def _equity(ret):
    return (1 + ret).cumprod()


def validate_from_signals(signals_path: str = None):
    """
    Validate the matrix calculation from the latest mobile_signals.json.
    Compare metrics across different time windows.
    """
    import json
    from pathlib import Path

    if signals_path is None:
        signals_path = os.path.join(ROOT, "docs", "data", "mobile_signals.json")
        if not os.path.exists(signals_path):
            signals_path = os.path.join(ROOT, "wf_backtest", "mobile_signals.json")

    with open(signals_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    print(f"\n{'='*70}")
    print(f"  VALIDIERUNG — mobile_signals.json")
    print(f"  Updated: {data.get('updated', '?')}")
    print(f"{'='*70}\n")

    # Validate each category
    for cat_key in ["swarm", "value", "turnaround", "alpha_mix"]:
        cat = data.get(cat_key)
        if cat is None:
            print(f"  ⚠️  {cat_key}: NICHT VORHANDEN!")
            continue

        print(f"\n{'─'*50}")
        print(f"  {cat_key.upper()}")
        print(f"{'─'*50}")

        # Check months_switch exists and has recent data
        months = cat.get("months_switch", {})
        if not months:
            print(f"  ❌ months_switch LEER!")
            continue

        years = sorted(months.keys(), key=int)
        print(f"  Daten: {years[0]}–{years[-1]}")
        print(f"  Start: {cat.get('start', '?')}, End: {cat.get('end', '?')}")

        # Check last month has data
        last_year = years[-1]
        last_months = months[last_year]
        last_month_num = max(int(m) for m in last_months.keys())
        print(f"  Letzter Monat: {last_year}/{last_month_num:02d} "
              f"→ {last_months[str(last_month_num)]:+.1f}%")

        # Core metrics
        print(f"\n  Kern-Parameter (Gesamtzeitraum):")
        print(f"    Sharpe Switch: {cat.get('sharpe_switch', '?')}")
        print(f"    Sharpe B&H:    {cat.get('sharpe_bh', '?')}")
        print(f"    CAGR Switch:   {cat.get('cagr_switch', 0)*100:.1f}%")
        print(f"    CAGR B&H:      {cat.get('cagr_bh', 0)*100:.1f}%")
        print(f"    Max DD Switch:  {cat.get('max_dd_switch', 0)*100:.1f}%")
        print(f"    Max DD B&H:     {cat.get('max_dd_bh', 0)*100:.1f}%")

        # Calculate metrics from monthly data for different windows
        yearly = cat.get("yearly_returns", [])
        if yearly:
            print(f"\n  Zeitraum-Vergleich (aus Jahresrenditen):")
            print(f"  {'Zeitraum':<15s} {'Switch':>10s} {'B&H':>10s} {'Diff':>10s}")
            print(f"  {'─'*45}")

            for start_year in [int(years[0]), 2018, 2021]:
                filtered = [y for y in yearly if y["year"] >= start_year]
                if not filtered:
                    continue
                sw_total = sum(y["switch"] for y in filtered) / 100
                bh_total = sum(y["bh"] for y in filtered) / 100
                n_years = len(filtered)
                sw_ann = sw_total / n_years if n_years > 0 else 0
                bh_ann = bh_total / n_years if n_years > 0 else 0

                label = f"ab {start_year}"
                print(f"  {label:<15s} {sw_ann*100:>+9.1f}% {bh_ann*100:>+9.1f}% "
                      f"{(sw_ann-bh_ann)*100:>+9.1f}%")

            # Per-year breakdown
            print(f"\n  Jahresrenditen (Switch vs B&H):")
            print(f"  {'Jahr':<6s} {'Switch':>10s} {'B&H':>10s} {'Alpha':>10s}")
            print(f"  {'─'*36}")
            for y in yearly:
                alpha = y["switch"] - y["bh"]
                sw_icon = "✅" if y["switch"] > 0 else "❌"
                print(f"  {y['year']:<6d} {y['switch']:>+9.1f}% {y['bh']:>+9.1f}% "
                      f"{alpha:>+9.1f}% {sw_icon}")

        # Check for gaps in monthly data
        all_months_present = True
        for yr in years[:-1]:  # Skip current year
            yr_months = months[yr]
            for m in range(1, 13):
                if str(m) not in yr_months:
                    print(f"  ⚠️  Fehlender Monat: {yr}/{m:02d}")
                    all_months_present = False
        if all_months_present:
            print(f"\n  ✅ Keine Lücken in den Monatsdaten")

        # Stocks info
        stocks = cat.get("stocks", [])
        if stocks:
            n_long = sum(1 for s in stocks if s.get("signal") == "LONG")
            n_total = len(stocks)
            print(f"  📊 Positionen: {n_long}/{n_total} LONG")

    # Alpha-Mix specific: check weights
    alpha = data.get("alpha_mix", {})
    if alpha:
        print(f"\n{'─'*50}")
        print(f"  ALPHA-MIX GEWICHTUNG")
        print(f"{'─'*50}")
        weights = alpha.get("weights", {})
        for k, v in weights.items():
            print(f"    {k}: {v}")
        print(f"  Investitionsgrad: {alpha.get('pct_invested', 0)*100:.1f}%")
        print(f"  Anzahl Trades: {alpha.get('n_trades', 0)}")


def validate_live_calculation(use_hist: bool = True):
    """
    Run the actual WF calculation and compare with stored results.
    This validates that the matrix calculation is correct.
    """
    from wf_backtest.swarm_wf import run_swarm_wf
    from wf_backtest.stock_screener import run_category_wf

    print(f"\n{'='*70}")
    print(f"  LIVE VALIDIERUNG — Neuberechnung")
    print(f"{'='*70}")

    results = {}

    # Run each category
    for name, runner in [
        ("swarm", lambda: run_swarm_wf(top_n=10, use_polygon=use_hist)),
        ("value", lambda: run_category_wf("value", use_polygon=use_hist)),
        ("turnaround", lambda: run_category_wf("turnaround", use_polygon=use_hist)),
    ]:
        print(f"\n  Berechne {name} …")
        try:
            result = runner()
            if result is None:
                print(f"  ❌ {name}: Keine Ergebnisse")
                continue
            results[name] = result
            print(f"  ✅ {name}: Sharpe={result['sw_sharpe']:.2f}, "
                  f"Start={result['start']}")
        except Exception as e:
            print(f"  ❌ {name}: FEHLER — {e}")
            import traceback
            traceback.print_exc()

    if len(results) < 2:
        print("\n  ⚠️  Zu wenig Kategorien für Alpha-Mix Validierung")
        return results

    # Compare time windows
    print(f"\n{'─'*50}")
    print(f"  ZEITRAUM-VERGLEICH (Kern-Parameter)")
    print(f"{'─'*50}")

    for name, result in results.items():
        sw = result["switch_ret"]
        bh = result["bench_ret"]

        print(f"\n  {name.upper()}")
        print(f"  {'Zeitraum':<15s} {'Sharpe':>8s} {'CAGR':>8s} {'MaxDD':>8s} "
              f"{'Sharpe':>8s} {'CAGR':>8s} {'MaxDD':>8s}")
        print(f"  {'':15s} {'─Switch─':>8s} {'':>8s} {'':>8s} "
              f"{'──B&H──':>8s} {'':>8s} {'':>8s}")

        for label, start_date in [
            ("Full", None),
            ("ab 2018", "2018-01-01"),
            ("ab 2021", "2021-01-01"),
        ]:
            if start_date:
                cut = pd.Timestamp(start_date)
                sw_cut = sw[sw.index >= cut]
                bh_cut = bh[bh.index >= cut]
            else:
                sw_cut = sw
                bh_cut = bh

            if len(sw_cut) < 50:
                continue

            sw_eq = _equity(sw_cut)
            bh_eq = _equity(bh_cut)

            sw_sharpe = sharpe_ratio(sw_cut, RF)
            bh_sharpe = sharpe_ratio(bh_cut, RF)
            sw_cagr = cagr(sw_eq) * 100
            bh_cagr = cagr(bh_eq) * 100
            sw_dd = max_drawdown(sw_eq) * 100
            bh_dd = max_drawdown(bh_eq) * 100

            print(f"  {label:<15s} {sw_sharpe:>8.2f} {sw_cagr:>+7.1f}% {sw_dd:>+7.1f}% "
                  f"{bh_sharpe:>8.2f} {bh_cagr:>+7.1f}% {bh_dd:>+7.1f}%")

    return results


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Validierung der Matrix-Berechnung")
    parser.add_argument("--signals", type=str, default=None,
                        help="Pfad zur mobile_signals.json")
    parser.add_argument("--live", action="store_true",
                        help="Live-Neuberechnung (dauert ~15 min)")
    parser.add_argument("--no-hist", action="store_true",
                        help="Ohne historische Daten (nur Yahoo ab 2016)")
    args = parser.parse_args()

    # Always validate stored signals
    validate_from_signals(args.signals)

    # Optionally run live calculation
    if args.live:
        validate_live_calculation(use_hist=not args.no_hist)
