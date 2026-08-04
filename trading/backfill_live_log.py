"""
Einmaliges Skript: Fehlenden Log-Eintrag (27.07.) nachtragen.

Usage:
  1) Keys in .env ablegen (siehe .env.example)
  2) python trading\backfill_live_log.py
"""
import json, os, sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from trading.alpaca_trader import AlpacaClient, TRADE_LOG_LIVE  # lädt .env automatisch

key = os.environ.get("ALPACA_LIVE_KEY_ID", "")
secret = os.environ.get("ALPACA_LIVE_SECRET_KEY", "")
if not key or not secret:
    print("FEHLER: ALPACA_LIVE_KEY_ID / ALPACA_LIVE_SECRET_KEY nicht gesetzt (env oder .env)!")
    sys.exit(1)

client = AlpacaClient(key, secret, live=True)
account = client.get_account()
positions = client.get_positions()

equity = float(account["equity"])
cash = float(account["cash"])

pos_snap = {}
for p in positions:
    qty = float(p["qty"])
    price = float(p["current_price"])
    pos_snap[p["symbol"]] = {"qty": qty, "price": round(price, 2), "value": round(qty * price, 2)}

entry = {
    "date": "2026-07-27 20:53 UTC (nachgetragen)",
    "equity": round(equity, 2),
    "cash": round(cash, 2),
    "invested": round(equity - cash, 2),
    "pct_invested": round((equity - cash) / equity * 100, 1) if equity else 0,
    "n_positions": len(positions),
    "positions": pos_snap,
    "targets": {},
    "orders": [],
    "note": "Nachgetragen: git push scheiterte am 27.07 wegen Race-Condition; Order wurde trotzdem korrekt ausgefuehrt.",
}

log_data = []
if TRADE_LOG_LIVE.exists():
    log_data = json.loads(TRADE_LOG_LIVE.read_text())

log_data.append(entry)
TRADE_LOG_LIVE.write_text(json.dumps(log_data, indent=2, ensure_ascii=False))

print(f"Nachgetragen: Equity=${equity:,.2f}, {len(positions)} Positionen")
print(f"Gespeichert in: {TRADE_LOG_LIVE}")
