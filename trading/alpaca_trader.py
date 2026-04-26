"""
Alpaca Strategy-Switch Auto-Trader
===================================
Liest Signale aus mobile_signals.json und platziert Orders über Alpaca REST API.
Läuft komplett ohne PC — direkt aus GitHub Actions.

Usage (lokal):
  python -m trading.alpaca_trader                    # Dry-Run
  python -m trading.alpaca_trader --execute          # Paper-Trading
  python -m trading.alpaca_trader --execute --live   # Live (echtes Geld!)

Env-Variablen (GitHub Secrets):
  ALPACA_KEY_ID        API Key ID
  ALPACA_SECRET_KEY    API Secret Key
  ALPACA_LIVE          "1" für Live-Konto (Default: Paper)
  TRADING_ENABLED      "1" um Orders zu senden (Default: nur Dry-Run)
"""

import json, os, logging, argparse
from datetime import datetime, timezone
from pathlib import Path
from urllib.request import Request, urlopen
from urllib.error import HTTPError

TRADE_LOG = Path(__file__).parent / "trade_log.json"

log = logging.getLogger("alpaca_trader")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

SIGNALS_FILE = Path(__file__).parent.parent / "docs" / "data" / "mobile_signals.json"

# ── Alpaca API endpoints ─────────────────────────────────────────────────────
PAPER_URL = "https://paper-api.alpaca.markets"
LIVE_URL  = "https://api.alpaca.markets"

# Yahoo ticker → Alpaca symbol (Alpaca only trades US-listed securities)
TICKER_MAP = {
    "SXR8.DE": "SPY",   # S&P 500 EU ETF → US equivalent
    "URTH":    "URTH",
    "EEM":     "EEM",
    "VGK":     "VGK",
}


def _api(base_url: str, key: str, secret: str, method: str, path: str,
         body: dict = None) -> dict:
    """Make an Alpaca API call."""
    url = f"{base_url}{path}"
    data = json.dumps(body).encode() if body else None
    req = Request(url, data=data, method=method)
    req.add_header("APCA-API-KEY-ID", key)
    req.add_header("APCA-API-SECRET-KEY", secret)
    req.add_header("Content-Type", "application/json")
    req.add_header("Accept", "application/json")
    try:
        with urlopen(req) as resp:
            return json.loads(resp.read().decode()) if resp.status != 204 else {}
    except HTTPError as e:
        error_body = e.read().decode() if e.fp else ""
        log.error(f"API Error {e.code}: {error_body}")
        raise


class AlpacaClient:
    def __init__(self, key: str, secret: str, live: bool = False):
        self.key = key
        self.secret = secret
        self.base = LIVE_URL if live else PAPER_URL
        self.mode = "LIVE" if live else "PAPER"

    def _call(self, method, path, body=None):
        return _api(self.base, self.key, self.secret, method, path, body)

    def get_account(self) -> dict:
        return self._call("GET", "/v2/account")

    def get_positions(self) -> list:
        return self._call("GET", "/v2/positions")

    def get_position(self, symbol: str) -> dict | None:
        try:
            return self._call("GET", f"/v2/positions/{symbol}")
        except HTTPError as e:
            if e.code == 404:
                return None
            raise

    def place_order(self, symbol: str, qty: int, side: str,
                    order_type: str = "market", time_in_force: str = "day",
                    limit_price: float = None) -> dict:
        """Place order. Default: Market day order."""
        body = {
            "symbol": symbol,
            "qty": str(qty),
            "side": side,
            "type": order_type,
            "time_in_force": time_in_force,
        }
        if order_type == "limit" and limit_price is not None:
            body["limit_price"] = str(round(limit_price, 2))
        return self._call("POST", "/v2/orders", body)

    def close_position(self, symbol: str) -> dict:
        return self._call("DELETE", f"/v2/positions/{symbol}")


def load_signals(path: Path = SIGNALS_FILE) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"Signaldatei nicht gefunden: {path}")
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def map_ticker(yahoo_ticker: str) -> str | None:
    """Convert Yahoo ticker to Alpaca symbol. Returns None if not tradeable."""
    if yahoo_ticker in TICKER_MAP:
        return TICKER_MAP[yahoo_ticker]
    # US stocks: strip any suffix
    clean = yahoo_ticker.split(".")[0]
    return clean


def compute_targets(signals: dict, equity: float) -> dict:
    """
    Compute target dollar allocation per symbol.
    Alpha-Mix: Category-weighted (e.g. 40/40/20), then equal-weight within category.
    Stocks appearing in multiple categories get combined allocations.
    Returns: {alpaca_symbol: {"dollars": float, "signal": str, "reason": str}}
    """
    targets = {}

    alpha = signals.get("alpha_mix")
    if alpha and alpha.get("stocks"):
        raw_weights = alpha.get("weights", {})
        cat_weights = {}
        for k, v in raw_weights.items():
            cat_weights[k] = float(str(v).replace("%", "")) / 100.0

        # Group LONG stocks by category
        long_by_cat = {}
        for s in alpha["stocks"]:
            cat = s.get("category", "unknown")
            if s["signal"] == "LONG":
                long_by_cat.setdefault(cat, []).append(s)

        log.info(f"Allocation: weights={raw_weights}, "
                 f"LONG per cat: {{{', '.join(k+':'+str(len(v)) for k,v in long_by_cat.items())}}}")

        for s in alpha["stocks"]:
            symbol = map_ticker(s["ticker"])
            if not symbol:
                continue
            cat = s.get("category", "unknown")
            w = cat_weights.get(cat, 0)
            cat_budget = equity * w
            n_long = len(long_by_cat.get(cat, []))

            if s["signal"] == "LONG" and n_long > 0:
                alloc = cat_budget / n_long
                if symbol in targets and targets[symbol]["signal"] == "LONG":
                    targets[symbol]["dollars"] += alloc
                    targets[symbol]["reason"] += f" + {cat}"
                else:
                    targets[symbol] = {
                        "dollars": alloc,
                        "signal": "LONG",
                        "reason": f"{cat} {s['ticker']} LONG",
                    }
            else:
                if symbol not in targets or targets[symbol]["signal"] != "LONG":
                    targets[symbol] = {
                        "dollars": 0,
                        "signal": "CASH",
                        "reason": f"{cat} {s['ticker']} CASH",
                    }

        # Log consolidated targets
        for sym, t in sorted(targets.items()):
            if t["signal"] == "LONG":
                log.info(f"  Target: {sym} ${t['dollars']:,.0f} ({t['reason']})")
    else:
        log.warning("Kein alpha_mix vorhanden — keine Targets!")

    return targets


def compute_orders(targets: dict, current_positions: dict,
                   prices: dict, min_order_value: float = 50) -> list:
    """
    Compare target vs current positions.
    Returns list of order dicts.
    """
    orders = []
    all_symbols = set(targets.keys()) | set(current_positions.keys())

    for symbol in sorted(all_symbols):
        target = targets.get(symbol, {"dollars": 0, "reason": "Nicht mehr im Portfolio"})
        target_dollars = target["dollars"]
        price = prices.get(symbol, 0)

        if price <= 0:
            log.warning(f"  Kein Preis für {symbol} — übersprungen")
            continue

        target_qty = int(target_dollars / price)
        current_qty = current_positions.get(symbol, 0)
        diff = target_qty - current_qty

        if abs(diff * price) < min_order_value:
            continue

        if diff > 0:
            orders.append({
                "symbol": symbol,
                "side": "buy",
                "qty": diff,
                "reason": target.get("reason", ""),
                "value": round(diff * price, 2),
            })
        elif diff < 0:
            orders.append({
                "symbol": symbol,
                "side": "sell",
                "qty": abs(diff),
                "reason": target.get("reason", ""),
                "value": round(abs(diff) * price, 2),
            })

    return orders


def run(execute: bool = False, live: bool = False, signals_path: Path = None):
    """Main trading logic."""
    key = os.environ.get("ALPACA_KEY_ID", "")
    secret = os.environ.get("ALPACA_SECRET_KEY", "")

    if not key or not secret:
        log.error("❌ ALPACA_KEY_ID und ALPACA_SECRET_KEY müssen gesetzt sein!")
        log.error("   Lokal:   set ALPACA_KEY_ID=... && set ALPACA_SECRET_KEY=...")
        log.error("   GitHub:  Settings → Secrets → Actions → New Secret")
        return False

    client = AlpacaClient(key, secret, live=live)

    # Diagnostics
    log.info(f"Key-ID: {key[:8]}...{key[-4:]} (len={len(key)})")
    log.info(f"Secret: {secret[:4]}...{secret[-4:]} (len={len(secret)})")
    log.info(f"Base URL: {client.base}")

    print(f"\n{'='*60}")
    print(f"  Alpaca Strategy-Switch Trader")
    print(f"  Modus: {'🔴 LIVE' if live else '🟢 PAPER'}  |  "
          f"{'⚠️  EXECUTE' if execute else 'DRY-RUN'}")
    print(f"{'='*60}\n")

    # Load signals
    path = signals_path or SIGNALS_FILE
    signals = load_signals(path)
    log.info(f"Signale geladen: {len(signals.get('etfs', []))} ETFs")

    # Print signals
    print("── Aktuelle Signale ──")
    for etf in signals.get("etfs", []):
        icon = "🟢" if etf["signal"] == "LONG" else "🔴"
        print(f"  {icon} {etf['ticker']:12s} {etf['signal']:6s}  ({etf['strategy']})")
    for cat_key in ("swarm", "value", "turnaround"):
        cat = signals.get(cat_key)
        if not cat:
            continue
        n_l = sum(1 for s in cat["stocks"] if s["signal"] == "LONG")
        print(f"  {cat_key.upper()}: {n_l}/{len(cat['stocks'])} LONG")

    # Account info
    try:
        account = client.get_account()
    except Exception as e:
        log.error(f"❌ API-Verbindung fehlgeschlagen: {e}")
        if isinstance(e, HTTPError):
            log.error(f"   Response: {e.read().decode() if e.fp else 'n/a'}")
        return False

    equity = float(account["equity"])
    cash = float(account["cash"])
    log.info(f"Konto: ${equity:,.2f} Equity  |  ${cash:,.2f} Cash")
    log.info(f"Account Status: {account.get('status')} | Trading blocked: {account.get('trading_blocked')} | Transfers blocked: {account.get('transfers_blocked')}")

    # Current positions
    positions = client.get_positions()
    current = {}
    prices = {}
    if positions:
        print("\n── Aktuelle Positionen ──")
        for p in positions:
            sym = p["symbol"]
            qty = int(float(p["qty"]))
            price = float(p["current_price"])
            value = float(p["market_value"])
            current[sym] = qty
            prices[sym] = price
            pnl = float(p["unrealized_pl"])
            pnl_pct = float(p["unrealized_plpc"]) * 100
            icon = "📈" if pnl >= 0 else "📉"
            print(f"  {icon} {sym:8s} {qty:4d} × ${price:.2f} = ${value:,.0f}  "
                  f"({'+' if pnl>=0 else ''}{pnl_pct:.1f}%)")
    else:
        print("\n  Keine offenen Positionen.")

    # Fetch prices for symbols not in portfolio
    targets = compute_targets(signals, equity)

    # Safety: if alpha_mix is missing and we have positions, do NOT liquidate
    if not targets and current:
        log.warning("⚠️  Keine Targets berechnet aber Positionen vorhanden — "
                     "überspringe Trading um versehentliche Liquidation zu vermeiden!")
        _save_trade_log(equity, cash, current, prices, targets, [], execute)
        return True

    for sym in targets:
        if sym not in prices:
            # Use Alpaca snapshot for price
            try:
                snap = _api("https://data.alpaca.markets", key, secret,
                            "GET", f"/v2/stocks/{sym}/snapshot")
                prices[sym] = float(snap["latestTrade"]["p"])
            except Exception:
                log.warning(f"  Preis für {sym} nicht verfügbar")
                prices[sym] = 0

    # Compute orders
    orders = compute_orders(targets, current, prices)

    if not orders:
        print("\n✅ Keine Orders nötig — Portfolio ist aktuell.")
        _save_trade_log(equity, cash, current, prices, targets, [], execute)
        return True

    # Execute or dry-run
    print(f"\n── {'⚠️  MOC Orders' if execute else 'DRY-RUN'} ──")
    for o in orders:
        icon = "🟢" if o["side"] == "buy" else "🔴"
        print(f"  {icon} {o['side'].upper():4s} {o['qty']:4d}× {o['symbol']:8s}  "
              f"MOC  (~${o['value']:,.0f})  — {o['reason']}")

    executed_orders = []
    if execute:
        print("\n── Sende Market Orders ──")
        for o in orders:
            try:
                result = client.place_order(
                    o["symbol"], o["qty"], o["side"],
                    order_type="market",
                    time_in_force="day",
                )
                o["status"] = result.get("status", "?")
                o["order_id"] = result.get("id", "?")
                executed_orders.append(o)
                log.info(f"  ✅ {o['side'].upper()} {o['qty']}× {o['symbol']} "
                         f"MOC → Order {o['order_id']} ({o['status']})")
            except Exception as e:
                err_detail = str(e)
                if isinstance(e, HTTPError):
                    err_body = e.read().decode() if e.fp else ""
                    err_detail = f"HTTP {e.code}: {err_body}"
                    log.error(f"  ❌ {o['symbol']}: {err_detail}")
                    log.error(f"     URL: {e.url}")
                    log.error(f"     Key-ID (first 8): {key[:8]}...")
                else:
                    log.error(f"  ❌ {o['symbol']}: {err_detail}")
                o["status"] = f"ERROR: {err_detail}"
                executed_orders.append(o)
    else:
        print(f"\n  ℹ️  DRY-RUN — keine Orders gesendet.")
        print(f"  Für echte Orders: --execute (oder TRADING_ENABLED=1)")

    # ── Save daily trade log ──
    _save_trade_log(equity, cash, current, prices, targets, executed_orders, execute)

    return True


def _save_trade_log(equity, cash, positions, prices, targets, orders, executed):
    """Append daily entry to trade_log.json for performance tracking."""
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    # Build position snapshot
    pos_snap = {}
    for sym, qty in positions.items():
        price = prices.get(sym, 0)
        pos_snap[sym] = {"qty": qty, "price": round(price, 2), "value": round(qty * price, 2)}

    # Build target snapshot
    tgt_snap = {}
    for sym, t in targets.items():
        if t["signal"] == "LONG":
            tgt_snap[sym] = {"dollars": round(t["dollars"], 2), "reason": t["reason"]}

    entry = {
        "date": now,
        "equity": round(equity, 2),
        "cash": round(cash, 2),
        "invested": round(equity - cash, 2),
        "pct_invested": round((equity - cash) / equity * 100, 1) if equity else 0,
        "n_positions": len(positions),
        "positions": pos_snap,
        "targets": tgt_snap,
        "orders": [{"symbol": o["symbol"], "side": o["side"], "qty": o["qty"],
                     "value": o["value"], "status": o.get("status", "dry-run")}
                    for o in orders],
        "executed": executed,
    }

    # Load existing log
    log_data = []
    if TRADE_LOG.exists():
        try:
            with open(TRADE_LOG, "r") as f:
                log_data = json.load(f)
        except (json.JSONDecodeError, IOError):
            log_data = []

    log_data.append(entry)

    with open(TRADE_LOG, "w") as f:
        json.dump(log_data, f, indent=2, ensure_ascii=False)

    log.info(f"Trade-Log gespeichert: {TRADE_LOG} ({len(log_data)} Einträge)")


def main():
    parser = argparse.ArgumentParser(description="Alpaca Strategy-Switch Trader")
    parser.add_argument("--execute", action="store_true",
                        help="Orders wirklich senden")
    parser.add_argument("--live", action="store_true",
                        help="Live-Konto statt Paper")
    parser.add_argument("--signals", type=str, default=None,
                        help="Pfad zur Signaldatei")
    args = parser.parse_args()

    # Also check env vars (for GitHub Actions)
    execute = args.execute or os.environ.get("TRADING_ENABLED") == "1"
    live = args.live or os.environ.get("ALPACA_LIVE") == "1"

    if execute and live:
        if not os.environ.get("CI"):  # Not in CI → ask for confirmation
            confirm = input("⚠️  LIVE-ORDERS! Fortfahren? (ja/nein): ")
            if confirm.strip().lower() != "ja":
                print("Abgebrochen.")
                return

    sig_path = Path(args.signals) if args.signals else None
    run(execute=execute, live=live, signals_path=sig_path)


if __name__ == "__main__":
    main()
