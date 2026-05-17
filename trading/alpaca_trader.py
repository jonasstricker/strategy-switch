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

import json, os, logging, argparse, time
from datetime import datetime, timezone
from pathlib import Path
from urllib.request import Request, urlopen
from urllib.error import HTTPError

TRADE_LOG = Path(__file__).parent / "trade_log.json"
FAILED_ORDERS_FILE = Path(__file__).parent / "failed_orders.json"
MAX_RETRIES = 3
RETRY_DELAY_SEC = 10

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
         body: dict = None, retries: int = MAX_RETRIES) -> dict:
    """Make an Alpaca API call with retry logic."""
    url = f"{base_url}{path}"
    data = json.dumps(body).encode() if body else None
    last_error = None
    for attempt in range(1, retries + 1):
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
            last_error = e
            # Don't retry on auth errors (403) or client errors (4xx except 429)
            if e.code == 429:  # Rate limit — retry
                log.warning(f"Rate limited (attempt {attempt}/{retries}), retrying...")
                time.sleep(RETRY_DELAY_SEC * attempt)
                continue
            elif 400 <= e.code < 500:
                log.error(f"API Error {e.code}: {error_body}")
                raise
            else:  # 5xx — retry
                log.warning(f"Server error {e.code} (attempt {attempt}/{retries}): {error_body}")
                if attempt < retries:
                    time.sleep(RETRY_DELAY_SEC * attempt)
                    continue
                raise
        except Exception as e:
            last_error = e
            log.warning(f"Connection error (attempt {attempt}/{retries}): {e}")
            if attempt < retries:
                time.sleep(RETRY_DELAY_SEC * attempt)
                continue
            raise
    raise last_error


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
    Alpha-Mix: Category-weighted (e.g. 45/40/15), then equal-weight within category.

    IMPORTANT: Only the LONG-proportion of each category budget is invested.
    If a category has 4/10 stocks LONG, only 4/10 of that category's budget
    is invested — the rest stays as cash. This ensures the portfolio's
    investment level reflects the actual signal strength.

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

        # Count LONG and total stocks per category
        long_by_cat = {}
        total_by_cat = {}
        for s in alpha["stocks"]:
            cat = s.get("category", "unknown")
            total_by_cat[cat] = total_by_cat.get(cat, 0) + 1
            if s["signal"] == "LONG":
                long_by_cat.setdefault(cat, []).append(s)

        # Log allocation ratios
        total_invested_pct = 0
        for cat, w in cat_weights.items():
            n_long = len(long_by_cat.get(cat, []))
            n_total = total_by_cat.get(cat, 1)
            invest_ratio = n_long / n_total if n_total > 0 else 0
            invested_pct = w * invest_ratio * 100
            total_invested_pct += invested_pct
            log.info(f"  {cat}: {n_long}/{n_total} LONG "
                     f"→ {invest_ratio:.0%} × {w:.0%} = {invested_pct:.1f}% investiert")
        log.info(f"  Gesamt: {total_invested_pct:.1f}% investiert, "
                 f"{100-total_invested_pct:.1f}% Cash")

        for s in alpha["stocks"]:
            symbol = map_ticker(s["ticker"])
            if not symbol:
                continue
            cat = s.get("category", "unknown")
            w = cat_weights.get(cat, 0)
            n_long = len(long_by_cat.get(cat, []))
            n_total = total_by_cat.get(cat, 1)

            if s["signal"] == "LONG" and n_long > 0:
                # Each LONG stock gets: equity × cat_weight / total_stocks_in_cat
                # This means the category budget is proportionally reduced
                # when fewer stocks are LONG
                alloc = equity * w / n_total
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
        total_alloc = sum(t["dollars"] for t in targets.values() if t["signal"] == "LONG")
        log.info(f"  Investiert: ${total_alloc:,.0f} / ${equity:,.0f} "
                 f"= {total_alloc/equity*100:.1f}%")
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

        target_qty = round(target_dollars / price, 3)  # fractional shares
        current_qty = current_positions.get(symbol, 0)
        diff = round(target_qty - current_qty, 3)

        if abs(diff * price) < min_order_value:
            continue

        if diff > 0:
            orders.append({
                "symbol": symbol,
                "side": "buy",
                "qty": round(diff, 3),
                "reason": target.get("reason", ""),
                "value": round(diff * price, 2),
            })
        elif diff < 0:
            orders.append({
                "symbol": symbol,
                "side": "sell",
                "qty": round(abs(diff), 3),
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
        err_msg = str(e)
        if isinstance(e, HTTPError):
            err_body = e.read().decode() if e.fp else "n/a"
            err_msg = f"HTTP {e.code}: {err_body}"
        log.error(f"❌ API-Verbindung fehlgeschlagen: {err_msg}")
        _send_api_down_alert(err_msg, key)
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
            qty = float(p["qty"])
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
    failed_orders = []
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
                failed_orders.append(o)

        # ── Send alert email if any orders failed ──
        if failed_orders:
            _send_failure_alert(failed_orders, equity, targets)
            _save_failed_orders(failed_orders, targets)
    else:
        print(f"\n  ℹ️  DRY-RUN — keine Orders gesendet.")
        print(f"  Für echte Orders: --execute (oder TRADING_ENABLED=1)")

    # ── Save daily trade log ──
    _save_trade_log(equity, cash, current, prices, targets, executed_orders, execute)

    return True


def _send_email_alert(subject: str, html_body: str):
    """Send alert email using the shared email config."""
    try:
        import sys
        sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
        from wf_backtest.notifier import send_email
        success = send_email(subject, html_body)
        if success:
            log.info(f"✅ Alert-Email gesendet: {subject}")
        else:
            log.error(f"❌ Alert-Email fehlgeschlagen: {subject}")
    except Exception as e:
        log.error(f"❌ Email-System nicht verfügbar: {e}")


def _send_api_down_alert(error_msg: str, key_prefix: str):
    """Send email alert when Alpaca API is unreachable."""
    now = datetime.now(timezone.utc).strftime("%d.%m.%Y %H:%M UTC")
    html = f"""
    <html><body style="font-family: Arial, sans-serif;">
    <h2 style="color: #C62828;">🚨 Alpaca API DOWN — Keine Orders möglich!</h2>
    <p><strong>Zeitpunkt:</strong> {now}</p>
    <p><strong>Fehler:</strong></p>
    <pre style="background: #FFEBEE; padding: 12px; border-radius: 8px;">{error_msg}</pre>
    <p><strong>API Key (Prefix):</strong> {key_prefix[:8]}...</p>
    <h3>Mögliche Ursachen:</h3>
    <ul>
        <li>API-Keys abgelaufen oder deaktiviert</li>
        <li>Paper Trading Account inaktiv</li>
        <li>Alpaca-Server Wartung</li>
        <li>Account gesperrt (trading_blocked)</li>
    </ul>
    <h3>Sofort-Maßnahmen:</h3>
    <ol>
        <li>Prüfe <a href="https://app.alpaca.markets">Alpaca Dashboard</a></li>
        <li>Generiere ggf. neue API-Keys</li>
        <li>Prüfe Account-Status (aktiv / gesperrt)</li>
    </ol>
    <p style="color: #999; font-size: 12px;">
        Pending Orders wurden in <code>failed_orders.json</code> gespeichert.
        Beim nächsten erfolgreichen Lauf werden diese automatisch nachgeholt.
    </p>
    </body></html>
    """
    _send_email_alert(f"🚨 ALPACA API FEHLER — {now}", html)


def _send_failure_alert(failed_orders: list, equity: float, targets: dict):
    """Send email alert when orders fail to execute."""
    now = datetime.now(timezone.utc).strftime("%d.%m.%Y %H:%M UTC")

    order_rows = ""
    for o in failed_orders:
        order_rows += f"""
        <tr>
            <td>{o['symbol']}</td>
            <td>{o['side'].upper()}</td>
            <td>{o['qty']}</td>
            <td>${o['value']:,.2f}</td>
            <td style="color: #C62828;">{o.get('status', 'ERROR')}</td>
        </tr>"""

    target_rows = ""
    for sym, t in sorted(targets.items()):
        if t["signal"] == "LONG":
            target_rows += f"""
            <tr>
                <td>{sym}</td>
                <td>${t['dollars']:,.2f}</td>
                <td>{t['reason']}</td>
            </tr>"""

    html = f"""
    <html><body style="font-family: Arial, sans-serif;">
    <h2 style="color: #E65100;">⚠️ Trading-Fehler — {len(failed_orders)} Orders gescheitert!</h2>
    <p><strong>Zeitpunkt:</strong> {now}</p>
    <p><strong>Equity:</strong> ${equity:,.2f}</p>

    <h3>Gescheiterte Orders:</h3>
    <table border="1" cellpadding="6" cellspacing="0" style="border-collapse: collapse;">
        <tr style="background: #FFEBEE;">
            <th>Symbol</th><th>Seite</th><th>Stück</th><th>Wert</th><th>Fehler</th>
        </tr>
        {order_rows}
    </table>

    <h3>Geplante Ziel-Allokation:</h3>
    <table border="1" cellpadding="6" cellspacing="0" style="border-collapse: collapse;">
        <tr style="background: #E3F2FD;">
            <th>Symbol</th><th>Ziel $</th><th>Grund</th>
        </tr>
        {target_rows}
    </table>

    <p style="color: #999; font-size: 12px;">
        Orders werden beim nächsten Lauf automatisch wiederholt.
        Prüfe <a href="https://app.alpaca.markets">Alpaca Dashboard</a>.
    </p>
    </body></html>
    """
    _send_email_alert(f"⚠️ {len(failed_orders)} ORDERS GESCHEITERT — {now}", html)


def _save_failed_orders(failed_orders: list, targets: dict):
    """Save failed orders to file for retry on next run."""
    now = datetime.now(timezone.utc).isoformat()
    entry = {
        "date": now,
        "orders": [{"symbol": o["symbol"], "side": o["side"], "qty": o["qty"],
                     "value": o["value"], "status": o.get("status", "ERROR")}
                    for o in failed_orders],
        "targets": {sym: {"dollars": round(t["dollars"], 2), "reason": t["reason"]}
                    for sym, t in targets.items() if t["signal"] == "LONG"},
    }

    data = []
    if FAILED_ORDERS_FILE.exists():
        try:
            with open(FAILED_ORDERS_FILE, "r") as f:
                data = json.load(f)
        except (json.JSONDecodeError, IOError):
            data = []

    data.append(entry)
    # Keep only last 30 entries
    data = data[-30:]

    with open(FAILED_ORDERS_FILE, "w") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    log.info(f"Failed orders gespeichert: {FAILED_ORDERS_FILE}")


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
