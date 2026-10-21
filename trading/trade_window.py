"""
Gate für den Core-Workflow (nur Standardbibliothek, läuft ohne pip install).

Entscheidet je Workflow-Lauf, ob gehandelt wird und mit welchem Timing, und schreibt
`go`, `timing`, `reason` nach $GITHUB_OUTPUT. Nicht-Handeln ist kein Fehler (Exit 0);
Exit 1 nur, wenn die Entscheidung nicht möglich ist (z. B. Alpaca-Uhr nicht erreichbar).

Umschalten ohne Code-Änderung: Repository-Variable TRADE_TIMING (Settings → Secrets and
variables → Actions → Variables):
  leer / "sameday"  Signal und Handel 7-35 min vor Handelsschluss (Standard; mehrere Cron-Läufe,
                    das Fenster aus der Alpaca-Uhr deckt Sommer-/Winterzeit und frühen Schluss ab)
  "nextday"         Signal aus dem Vortagesschluss, Handel 14:45 UTC (Folgetag)

Env: EVENT_NAME, EVENT_SCHEDULE, TRADE_TIMING, INPUT_TIMING, ALPACA_KEY_ID/_SECRET_KEY
     bzw. ALPACA_LIVE_KEY_ID/_SECRET_KEY (nur für die Marktuhr)
"""
from __future__ import annotations

import json
import os
import sys
from datetime import datetime
from typing import NamedTuple
from urllib.request import Request, urlopen

LEGACY_CRON = "45 14 * * 1-5"        # Handel am Folgetag (muss zu daily.yml passen)
MIN_MINUTES_TO_CLOSE = 7
MAX_MINUTES_TO_CLOSE = 35
TIMINGS = ("nextday", "sameday")
DEFAULT_TIMING = "sameday"


class Decision(NamedTuple):
    go: bool
    timing: str
    reason: str


def seconds_to_close(clock: dict) -> float | None:
    if not clock.get("is_open"):
        return None
    return (datetime.fromisoformat(clock["next_close"]) - datetime.fromisoformat(clock["timestamp"])).total_seconds()


def decide(event: str, schedule: str, repo_timing: str, input_timing: str, secs_to_close) -> Decision:
    """secs_to_close: Callable[[], float | None] (wird nur bei Same-Day-Cron aufgerufen)."""
    repo_timing = (repo_timing or DEFAULT_TIMING).strip().lower()
    if repo_timing not in TIMINGS:
        raise ValueError(f"TRADE_TIMING {repo_timing!r} ungültig (erlaubt: {TIMINGS})")
    if event == "workflow_dispatch":
        timing = (input_timing or "auto").strip().lower()
        timing = repo_timing if timing == "auto" else timing
        if timing not in TIMINGS:
            raise ValueError(f"Timing {timing!r} ungültig (erlaubt: {TIMINGS})")
        return Decision(True, timing, f"manueller Lauf ({timing})")
    if event != "schedule":
        return Decision(False, repo_timing, f"Event {event!r} handelt nicht")
    if schedule == LEGACY_CRON:
        if repo_timing == "sameday":
            return Decision(False, "nextday", "TRADE_TIMING=sameday: Folgetag-Lauf übersprungen")
        return Decision(True, "nextday", "Folgetag-Lauf")
    if repo_timing != "sameday":
        return Decision(False, "sameday", "TRADE_TIMING != sameday: Same-Day-Cron übersprungen")
    secs = secs_to_close()
    if secs is None:
        return Decision(False, "sameday", "Markt geschlossen (Feiertag, früher Schluss oder falsche Sommer-/Winterzeit-Cron)")
    minutes = secs / 60
    if minutes > MAX_MINUTES_TO_CLOSE:
        return Decision(False, "sameday", f"zu früh: noch {minutes:.0f} min bis Handelsschluss (Fenster {MIN_MINUTES_TO_CLOSE}-{MAX_MINUTES_TO_CLOSE})")
    if minutes < MIN_MINUTES_TO_CLOSE:
        return Decision(False, "sameday", f"zu spät: nur noch {minutes:.1f} min bis Handelsschluss")
    return Decision(True, "sameday", f"Same-Day-Lauf, {minutes:.0f} min bis Handelsschluss")


def fetch_clock() -> dict:
    live_key, live_secret = os.environ.get("ALPACA_LIVE_KEY_ID"), os.environ.get("ALPACA_LIVE_SECRET_KEY")
    key, secret = os.environ.get("ALPACA_KEY_ID"), os.environ.get("ALPACA_SECRET_KEY")
    if key and secret:
        base = "https://paper-api.alpaca.markets"
    elif live_key and live_secret:
        key, secret, base = live_key, live_secret, "https://api.alpaca.markets"
    else:
        raise RuntimeError("keine Alpaca-Keys für die Marktuhr")
    req = Request(f"{base}/v2/clock", headers={"APCA-API-KEY-ID": key, "APCA-API-SECRET-KEY": secret,
                                               "Accept": "application/json"})
    with urlopen(req, timeout=20) as resp:
        return json.loads(resp.read().decode())


def main() -> int:
    try:
        d = decide(os.environ.get("EVENT_NAME", ""), os.environ.get("EVENT_SCHEDULE", ""),
                   os.environ.get("TRADE_TIMING", ""), os.environ.get("INPUT_TIMING", ""),
                   lambda: seconds_to_close(fetch_clock()))
    except Exception as e:
        print(f"::error::Gate fehlgeschlagen: {e}")
        return 1
    print(f"go={d.go} timing={d.timing} — {d.reason}")
    out = os.environ.get("GITHUB_OUTPUT")
    if out:
        with open(out, "a", encoding="utf-8") as f:
            f.write(f"go={'true' if d.go else 'false'}\ntiming={d.timing}\nreason={d.reason}\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
