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
from pathlib import Path
from urllib.request import Request, urlopen
from urllib.error import HTTPError

log = logging.getLogger("alpaca_trader")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

SIGNALS_FILE = Path(__file__).parent.parent / "wf_backtest" / "mobile_signals.json"

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
                    order_type: str = "market", time_in_force: str = "day") -> dict:
        return self._call("POST", "/v2/orders", {
            "symbol": symbol,
            "qty": str(qty),
            "side": side,
            "type": order_type,
            "time_in_force": time_in_force,
        })

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
    Returns: {alpaca_symbol: {"dollars": float, "signal": str, "reason": str}}
    """
    targets = {}

    # ETFs: 15% each
    for etf in signals.get("etfs", []):
        symbol = map_ticker(etf["ticker"])
        if not symbol:
            continue
        if etf["signal"] == "LONG":
            targets[symbol] = {
                "dollars": equity * 0.15,
                "signal": "LONG",
                "reason": f"ETF {etf['ticker']} LONG ({etf['strategy']})",
            }
        else:
            targets[symbol] = {
                "dollars": 0,
                "signal": "CASH",
                "reason": f"ETF {etf['ticker']} CASH",
            }

    # Categories
    cat_weights = {"swarm": 0.18, "value": 0.12, "turnaround": 0.10}
    for cat_key, weight in cat_weights.items():
        cat = signals.get(cat_key)
        if not cat:
            continue
        stocks = cat.get("stocks", [])
        n_long = sum(1 for s in stocks if s["signal"] == "LONG")

        for s in stocks:
            symbol = map_ticker(s["ticker"])
            if not symbol:
                continue
            if s["signal"] == "LONG" and n_long > 0:
                alloc = (equity * weight) / n_long
                targets[symbol] = {
                    "dollars": alloc,
                    "signal": "LONG",
                    "reason": f"{cat_key} {s['ticker']} LONG ({s.get('strategy', '?')})",
                }
            else:
                # Only set to 0 if not already set by another category with LONG
                if symbol not in targets or targets[symbol]["signal"] != "LONG":
                    targets[symbol] = {
                        "dollars": 0,
                        "signal": "CASH",
                        "reason": f"{cat_key} {s['ticker']} CASH",
                    }

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
        return False

    equity = float(account["equity"])
    cash = float(account["cash"])
    log.info(f"Konto: ${equity:,.2f} Equity  |  ${cash:,.2f} Cash")

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
        return True

    # Execute or dry-run
    print(f"\n── {'⚠️  Orders' if execute else 'DRY-RUN'} ──")
    for o in orders:
        icon = "🟢" if o["side"] == "buy" else "🔴"
        print(f"  {icon} {o['side'].upper():4s} {o['qty']:4d}× {o['symbol']:8s}  "
              f"(~${o['value']:,.0f})  — {o['reason']}")

    if execute:
        print("\n── Sende Orders ──")
        for o in orders:
            try:
                result = client.place_order(o["symbol"], o["qty"], o["side"])
                log.info(f"  ✅ {o['side'].upper()} {o['qty']}× {o['symbol']} → "
                         f"Order {result.get('id', '?')} ({result.get('status', '?')})")
            except Exception as e:
                log.error(f"  ❌ {o['symbol']}: {e}")
    else:
        print(f"\n  ℹ️  DRY-RUN — keine Orders gesendet.")
        print(f"  Für echte Orders: --execute (oder TRADING_ENABLED=1)")

    return True


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
