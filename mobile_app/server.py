#!/usr/bin/env python3
"""
mobile_app/server.py — Strategy-Switch PWA (Cloud-First)
=========================================================
Runs on Render.com (or locally). Computes signals automatically:
  - On first startup if no data exists
  - When data is older than STALE_HOURS (auto-refresh on API request)
  - On manual "Recalculate" button press
No PC needed — everything runs in the cloud.
"""

from __future__ import annotations
import json
import os
import subprocess
import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

from flask import Flask, render_template, jsonify

# ── Paths ────────────────────────────────────────────────────────────────────
APP_DIR = Path(__file__).resolve().parent
PROJECT_DIR = APP_DIR.parent
SIGNALS_FILE = PROJECT_DIR / "wf_backtest" / "mobile_signals.json"

# Detect environment: local Windows or cloud (Render/Linux)
IS_CLOUD = os.environ.get("RENDER") or not (PROJECT_DIR / ".venv").exists()
if IS_CLOUD:
    PYTHON_EXE = sys.executable
else:
    PYTHON_EXE = str(PROJECT_DIR / ".venv" / "Scripts" / "python.exe")

# Auto-refresh: recompute if signals are older than this (hours)
STALE_HOURS = 20

# ── Flask App ─────────────────────────────────────────────────────────────
app = Flask(__name__, static_folder="static", template_folder="templates")

# ── Recalculation state ──────────────────────────────────────────────────
_recalc_lock = threading.Lock()
_recalc_running = False


def load_signals() -> dict:
    """Read pre-computed signals from JSON file."""
    if SIGNALS_FILE.exists():
        with open(SIGNALS_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return {"updated": "Noch keine Daten — Berechnung läuft …", "etfs": []}


def _signals_are_stale() -> bool:
    """Check if signals file is missing or older than STALE_HOURS."""
    if not SIGNALS_FILE.exists():
        return True
    age_seconds = time.time() - SIGNALS_FILE.stat().st_mtime
    return age_seconds > STALE_HOURS * 3600


def _run_daily_runner():
    """Run the daily runner in a subprocess (blocking)."""
    global _recalc_running
    try:
        print(f"  [Auto-Compute] Starte Berechnung …")
        result = subprocess.run(
            [str(PYTHON_EXE), "-m", "wf_backtest.daily_runner"],
            cwd=str(PROJECT_DIR),
            capture_output=True,
            text=True,
            timeout=900,  # 15 min max (cloud is slower)
        )
        if result.returncode == 0:
            print(f"  [Auto-Compute] ✅ Berechnung fertig")
        else:
            print(f"  [Auto-Compute] ❌ Fehler: {result.stderr[-500:]}")
    except Exception as e:
        print(f"  [Auto-Compute] ❌ Exception: {e}")
    finally:
        with _recalc_lock:
            _recalc_running = False


def _trigger_recalc_if_needed():
    """Start background recalculation if signals are stale and not already running."""
    global _recalc_running
    with _recalc_lock:
        if _recalc_running:
            return False
        if not _signals_are_stale():
            return False
        _recalc_running = True
    t = threading.Thread(target=_run_daily_runner, daemon=True)
    t.start()
    return True


@app.route("/")
def index():
    # Auto-trigger computation on page load if stale
    _trigger_recalc_if_needed()
    return render_template("index.html")


@app.route("/api/signals")
def api_signals():
    """Return all pre-computed ETF signals. Auto-refreshes if stale."""
    _trigger_recalc_if_needed()
    data = load_signals()
    with _recalc_lock:
        data["recalculating"] = _recalc_running
    return jsonify(data)


@app.route("/api/recalculate", methods=["POST"])
def api_recalculate():
    """Trigger a manual recalculation in the background."""
    global _recalc_running
    with _recalc_lock:
        if _recalc_running:
            return jsonify({"status": "already_running",
                            "message": "Berechnung läuft bereits …"})
        _recalc_running = True

    t = threading.Thread(target=_run_daily_runner, daemon=True)
    t.start()
    return jsonify({"status": "started",
                    "message": "Neuberechnung gestartet … (ca. 2–5 Min.)"})


@app.route("/manifest.json")
def manifest():
    return app.send_static_file("manifest.json")


@app.route("/sw.js")
def service_worker():
    return app.send_static_file("sw.js")


if __name__ == "__main__":
    print("=" * 60)
    print("  Strategy-Switch PWA — Cloud Dashboard")
    print("=" * 60)

    if _signals_are_stale():
        print(f"\n  Signale veraltet oder fehlend → Auto-Compute beim ersten Request")
    elif SIGNALS_FILE.exists():
        data = load_signals()
        print(f"\n  Signale: {data.get('updated', '?')}")
        for etf in data.get("etfs", []):
            sig = "🟢 LONG" if etf["signal"] == "LONG" else "🔴 CASH"
            print(f"    {etf['ticker']:5s} {sig}  ({etf['strategy']})")

    port = int(os.environ.get("PORT", 8511))
    print(f"\n  http://localhost:{port}\n")
    app.run(host="0.0.0.0", port=port, debug=False)
