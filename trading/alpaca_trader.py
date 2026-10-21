"""
Alpaca Core-Strategy Auto-Trader
================================
Liest das Core-Signal (docs/data/core_signal.json, erzeugt von
`python -m wf_backtest.core_strategy`) und bringt das Konto auf die
Ziel-Gewichte (je 50 % S&P 500 / Nasdaq-100: SSO bzw. QLD im Aufwärtstrend,
sonst SGOV). Alle anderen Positionen werden verkauft. Läuft komplett ohne PC — direkt aus GitHub Actions.

Sicherheit: kein Trading bei veraltetem/ungültigem Signal, erst Verkäufe,
dann Käufe nur mit vorhandenem Cash (nie Margin), offene Alt-Orders werden
storniert. Bei jedem Problem Exit-Code 1 → GitHub schickt eine Fehler-Mail.
Same-Day-Signale (timing "sameday", siehe README) werden nur mit frischem Live-Kurs
und kurz vor Handelsschluss gehandelt.

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

import json, os, logging, argparse, time, math, sys
from datetime import datetime, timezone
from pathlib import Path
from urllib.request import Request, urlopen
from urllib.error import HTTPError
from zoneinfo import ZoneInfo

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from wf_backtest.core_strategy import CONFIG as CORE_CFG, CORE_SIGNAL_FILE  # noqa: E402

TRADE_LOG_PAPER = Path(__file__).parent / "trade_log.json"
TRADE_LOG_LIVE   = Path(__file__).parent / "trade_log_live.json"
FAILED_ORDERS_FILE = Path(__file__).parent / "failed_orders.json"
MAX_RETRIES = 3
RETRY_DELAY_SEC = 10

log = logging.getLogger("alpaca_trader")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

SIGNALS_FILE = CORE_SIGNAL_FILE
ALLOWED_SYMBOLS = {s[k] for s in CORE_CFG["sleeves"] for k in ("etf_1x", "etf_2x")} | {CORE_CFG["etf_cash"]}
REBALANCE_TOLERANCE = 0.03   # Ziel-Positionen nur handeln, wenn > 3 % der Equity daneben
MIN_ORDER_VALUE = 50.0
FILL_WAIT_SEC = 120
SAMEDAY_MAX_SIGNAL_AGE_S = 900     # Live-Signal darf beim Handeln max. 15 min alt sein
SAMEDAY_MIN_SECONDS_TO_CLOSE = 90  # Same-Day: darunter keine neuen Orders (Day-Orders würden sonst auf morgen rutschen)


def _load_dotenv(path: Path = None):
    """Minimal .env loader (keine externe Abhaengigkeit noetig). Lokal only —
    in GitHub Actions kommen die Werte bereits aus den Secrets als Env-Vars."""
    path = path or (Path(__file__).parent.parent / ".env")
    if not path.exists():
        return
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, _, v = line.partition("=")
        k, v = k.strip(), v.strip().strip('"').strip("'")
        if v:
            os.environ.setdefault(k, v)


_load_dotenv()

# ── Alpaca API endpoints ─────────────────────────────────────────────────────
PAPER_URL = "https://paper-api.alpaca.markets"
LIVE_URL  = "https://api.alpaca.markets"


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
            e.body = error_body  # stream is consumed — keep the text for callers
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

    def get_cumulative_deposits(self) -> float:
        """Sum of all executed cash deposits/withdrawals (net contributions).
        Used to separate strategy performance from capital in/outflows."""
        try:
            acts = self._call("GET", "/v2/account/activities?activity_types=CSD,CSW")
        except Exception:
            return 0.0
        total = 0.0
        for a in acts or []:
            try:
                total += float(a.get("net_amount", 0))
            except (TypeError, ValueError):
                continue
        return round(total, 2)

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

    def get_clock(self) -> dict:
        return self._call("GET", "/v2/clock")

    def get_asset(self, symbol: str) -> dict:
        return self._call("GET", f"/v2/assets/{symbol}")

    def get_open_orders(self) -> list:
        return self._call("GET", "/v2/orders?status=open&limit=500")

    def cancel_all_orders(self):
        return self._call("DELETE", "/v2/orders")

    def get_order(self, order_id: str) -> dict:
        return self._call("GET", f"/v2/orders/{order_id}")


def load_signals(path: Path = SIGNALS_FILE) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"Signaldatei nicht gefunden: {path}")
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def validate_core_signal(sig: dict, now_ny: datetime | None = None) -> str | None:
    """Fehlermeldung, wenn auf diesem Signal NICHT gehandelt werden darf."""
    if sig.get("status") != "ok":
        return f"Signal-Status '{sig.get('status')}': {sig.get('error')}"
    if sig.get("signal") not in ("RISK_ON", "PARTIAL", "RISK_OFF"):
        return f"unbekanntes Signal {sig.get('signal')!r}"
    try:
        as_of = datetime.strptime(sig["as_of"], "%Y-%m-%d").date()
    except (KeyError, TypeError, ValueError):
        return "as_of fehlt oder ungültig"
    now_ny = now_ny or datetime.now(ZoneInfo("America/New_York"))
    age = (now_ny.date() - as_of).days
    if not 0 <= age <= CORE_CFG["max_data_age_days"]:
        return f"Signal veraltet: Schlusskurs vom {as_of} ({age} Tage alt)"
    if sig.get("timing") == "sameday":
        if not sig.get("intraday"):
            return "Same-Day-Signal ohne Live-Kurs (Markt nicht offen oder Kursquelle ausgefallen)"
        if as_of != now_ny.date():
            return f"Same-Day-Signal gilt für {as_of}, heute ist {now_ny.date()}"
        try:
            gen = datetime.strptime(sig["generated_utc"], "%Y-%m-%d %H:%M UTC").replace(tzinfo=timezone.utc)
        except (KeyError, TypeError, ValueError):
            return "generated_utc fehlt oder ungültig"
        age_s = (now_ny.astimezone(timezone.utc) - gen).total_seconds()
        if not -120 <= age_s <= SAMEDAY_MAX_SIGNAL_AGE_S:
            return f"Live-Signal ist {age_s / 60:.0f} min alt (max. {SAMEDAY_MAX_SIGNAL_AGE_S // 60} min)"
    w = sig.get("weights")
    if not isinstance(w, dict) or not w or set(w) - ALLOWED_SYMBOLS:
        return f"ungültige Ziel-Gewichte {w!r} (erlaubt: {sorted(ALLOWED_SYMBOLS)})"
    try:
        vals = [float(v) for v in w.values()]
    except (TypeError, ValueError):
        return f"ungültige Ziel-Gewichte {w!r}"
    if any(not 0 <= v <= 1 for v in vals) or sum(vals) > 1.0001:
        return f"Gewichte außerhalb [0,1] oder Summe > 100 %: {w}"
    return None


def seconds_to_close(clock: dict) -> float | None:
    """Sekunden bis zum Handelsschluss (None, wenn Markt geschlossen oder Antwort unlesbar)."""
    if not clock.get("is_open"):
        return None
    try:
        return (datetime.fromisoformat(clock["next_close"]) - datetime.fromisoformat(clock["timestamp"])).total_seconds()
    except (KeyError, TypeError, ValueError):
        return None


def compute_core_targets(sig: dict, equity: float) -> dict:
    """Ziel-Dollarbetrag je Symbol. Returns {symbol: {"dollars", "signal", "reason"}}."""
    return {sym: {"dollars": equity * float(w), "signal": "LONG",
                  "reason": f"Core {sig['signal']} {float(w):.0%}"}
            for sym, w in sig["weights"].items() if float(w) > 0}


def _fmt_qty(q: float) -> str:
    return f"{q:.9f}".rstrip("0").rstrip(".")


def _floor_qty(q: float, fractionable: bool) -> float:
    return math.floor(q * 1000 + 1e-9) / 1000 if fractionable else float(math.floor(q + 1e-9))


def compute_orders(targets: dict, current_positions: dict, prices: dict, equity: float,
                   fractionable: dict | None = None, min_order_value: float = MIN_ORDER_VALUE,
                   tolerance: float = REBALANCE_TOLERANCE) -> list:
    """
    Orders, um vom aktuellen Bestand zum Ziel zu kommen (Verkäufe zuerst).
    - Positionen, die nicht im Ziel sind, werden immer komplett verkauft.
    - Ziel-Positionen werden nur gehandelt, wenn sie mehr als
      max(min_order_value, tolerance × Equity) vom Ziel abweichen
      (verhindert Mini-Trades und unnötige Steuer-Realisierungen).
    """
    fractionable = fractionable or {}
    threshold = max(min_order_value, tolerance * equity)
    orders = []
    for symbol in sorted(set(targets) | set(current_positions)):
        cur = float(current_positions.get(symbol, 0.0))
        price = float(prices.get(symbol) or 0.0)
        tgt = targets.get(symbol)
        if not tgt or tgt["dollars"] <= 0:
            if cur > 0:
                orders.append({"symbol": symbol, "side": "sell", "qty": cur,
                               "reason": "nicht im Ziel-Portfolio (komplett verkaufen)",
                               "value": round(cur * price, 2)})
            elif cur < 0:
                log.warning(f"  Short-Position {symbol} ({cur}) wird nicht automatisch geschlossen")
            continue
        if price <= 0:
            raise ValueError(f"kein Preis für Ziel-Symbol {symbol}")
        diff_val = tgt["dollars"] - cur * price
        if abs(diff_val) < threshold:
            continue
        qty = _floor_qty(abs(diff_val) / price, fractionable.get(symbol, True))
        if diff_val < 0:
            qty = min(qty, cur)
        if qty <= 0:
            continue
        orders.append({"symbol": symbol, "side": "buy" if diff_val > 0 else "sell", "qty": qty,
                       "reason": tgt.get("reason", ""), "value": round(qty * price, 2)})
    orders.sort(key=lambda o: (o["side"] != "sell", o["symbol"]))
    return orders


def _submit(client: "AlpacaClient", o: dict, executed: list, failed: list):
    try:
        result = client.place_order(o["symbol"], _fmt_qty(o["qty"]), o["side"],
                                    order_type="market", time_in_force="day")
        o["status"] = result.get("status", "?")
        o["order_id"] = result.get("id")
        log.info(f"  ✅ {o['side'].upper()} {_fmt_qty(o['qty'])}× {o['symbol']} → Order {o['order_id']} ({o['status']})")
    except Exception as e:
        err = str(e)
        if isinstance(e, HTTPError):
            err = f"HTTP {e.code}: {getattr(e, 'body', '')}"
        log.error(f"  ❌ {o['side'].upper()} {o['symbol']}: {err}")
        o["status"] = f"ERROR: {err}"
        failed.append(o)
    executed.append(o)


def _wait_for_fills(client: "AlpacaClient", orders: list, failed: list, timeout: int = FILL_WAIT_SEC):
    """Wartet (bei offenem Markt) auf Ausführung; abgelehnte/stornierte Orders → failed."""
    pending = [o for o in orders if o.get("order_id")]
    deadline = time.time() + timeout
    while pending and time.time() < deadline:
        time.sleep(3)
        for o in list(pending):
            try:
                st = client.get_order(o["order_id"]).get("status", "?")
            except Exception as e:
                log.warning(f"  Order-Status {o['symbol']} nicht abrufbar: {e}")
                continue
            o["status"] = st
            if st == "filled":
                pending.remove(o)
            elif st in ("canceled", "expired", "rejected", "suspended"):
                log.error(f"  ❌ Order {o['symbol']} {st}")
                failed.append(o)
                pending.remove(o)
    for o in pending:
        log.warning(f"  ⏳ Order {o['symbol']} nach {timeout}s noch '{o['status']}' (wird beim nächsten Lauf geprüft)")


def run(execute: bool = False, live: bool = False, signals_path: Path = None) -> bool:
    """Main trading logic. Returns False on any problem (→ Exit-Code 1 → GitHub-Fehlermail)."""
    if live:
        key    = os.environ.get("ALPACA_LIVE_KEY_ID")    or os.environ.get("ALPACA_KEY_ID", "")
        secret = os.environ.get("ALPACA_LIVE_SECRET_KEY") or os.environ.get("ALPACA_SECRET_KEY", "")
    else:
        key    = os.environ.get("ALPACA_KEY_ID", "")
        secret = os.environ.get("ALPACA_SECRET_KEY", "")

    if not key or not secret:
        log.error("❌ API-Keys fehlen! Live: ALPACA_LIVE_KEY_ID/ALPACA_LIVE_SECRET_KEY, Paper: ALPACA_KEY_ID/ALPACA_SECRET_KEY")
        return False

    client = AlpacaClient(key, secret, live=live)

    # Diagnostics (never log any part of the secret — Actions logs of public repos are public)
    log.info(f"Key-ID: {key[:4]}... (len={len(key)}) | Base URL: {client.base}")

    print(f"\n{'='*60}")
    print(f"  Alpaca Core-Strategy Trader")
    print(f"  Modus: {'🔴 LIVE' if live else '🟢 PAPER'}  |  "
          f"{'⚠️  EXECUTE' if execute else 'DRY-RUN'}")
    print(f"{'='*60}\n")

    # Core-Signal laden und prüfen — bei Problemen KEIN Trading
    path = signals_path or SIGNALS_FILE
    try:
        sig = load_signals(path)
    except Exception as e:
        log.error(f"❌ Core-Signal nicht lesbar ({path}): {e}")
        return False
    err = validate_core_signal(sig)
    if err:
        log.error(f"❌ Core-Signal ungültig — kein Trading: {err}")
        return False
    sameday = sig.get("timing") == "sameday"
    print(f"── Core-Signal ({'Live-Kurs vor Handelsschluss' if sameday else 'Schlusskurs'} {sig['as_of']}) ──")
    print(f"  {({'RISK_ON': '🟢', 'PARTIAL': '🟡'}).get(sig['signal'], '🔴')} {sig['signal']}  |  "
          f"Exposure {float(sig.get('exposure') or 0):.1f}x")
    for sl in sig.get("sleeves", []):
        ix = sl.get("index", {})
        print(f"    {'🟢' if sl.get('signal') == 'RISK_ON' else '🔴'} {sl.get('name')}: {ix.get('close')} "
              f"vs SMA{CORE_CFG['sma_days']} {ix.get('sma')} ({ix.get('dist_pct', 0):+.2f}%)")
    print("  Ziel: " + ", ".join(f"{s} {float(w):.1%}" for s, w in sig["weights"].items()))

    # Account info
    try:
        account = client.get_account()
    except Exception as e:
        err_msg = str(e)
        if isinstance(e, HTTPError):
            err_body = getattr(e, "body", "") or "n/a"
            err_msg = f"HTTP {e.code}: {err_body}"
        log.error(f"❌ API-Verbindung fehlgeschlagen: {err_msg}")
        _send_api_down_alert(err_msg, key)
        return False

    if account.get("trading_blocked") or account.get("account_blocked"):
        log.error("❌ Konto gesperrt (trading_blocked/account_blocked) — kein Trading")
        return False

    equity = float(account["equity"])
    cash = float(account["cash"])
    log.info(f"Konto: ${equity:,.2f} Equity  |  ${cash:,.2f} Cash  |  Buying Power "
             f"${float(account.get('buying_power') or 0):,.2f}  |  Multiplier {account.get('multiplier')}")
    log.info(f"Account Status: {account.get('status')} | Trading blocked: {account.get('trading_blocked')} | Transfers blocked: {account.get('transfers_blocked')}")
    if equity <= 0:
        log.error("❌ Equity <= 0 — kein Trading")
        return False

    if sameday:
        secs = seconds_to_close(client.get_clock())
        if secs is None or secs < SAMEDAY_MIN_SECONDS_TO_CLOSE:
            log.error(f"❌ Same-Day-Modus, aber Markt zu bzw. nur noch {secs}s bis Handelsschluss — kein Trading")
            return False
        log.info(f"Same-Day-Modus: noch {secs / 60:.1f} min bis Handelsschluss")
    # Net capital contributions (deposits − withdrawals) — used to separate
    # strategy performance from capital in/outflows in the trade log.
    cum_deposits = client.get_cumulative_deposits()
    if cum_deposits:
        log.info(f"Kumulierte Einzahlungen: ${cum_deposits:,.2f}")

    # Offene Orders früherer Läufe stornieren — jeder Lauf rechnet das Ziel komplett neu
    open_orders = client.get_open_orders() or []
    if open_orders:
        log.info(f"{len(open_orders)} offene Order(s): " + ", ".join(
            f"{o.get('side')} {o.get('qty') or o.get('notional')} {o.get('symbol')}" for o in open_orders))
        if execute:
            client.cancel_all_orders()
            time.sleep(3)
            log.info("  → storniert")

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
            print(f"  {icon} {sym:8s} {qty:7.3f} × ${price:.2f} = ${value:,.0f}  "
                  f"({'+' if pnl>=0 else ''}{pnl_pct:.1f}%)")
    else:
        print("\n  Keine offenen Positionen.")

    # Ziele, Handelbarkeit, Preise
    targets = compute_core_targets(sig, equity)
    fractionable = {}
    for sym in targets:
        try:
            asset = client.get_asset(sym)
        except Exception as e:
            log.error(f"❌ Asset-Info {sym} nicht abrufbar: {e}")
            return False
        if not asset.get("tradable") or asset.get("status") != "active":
            log.error(f"❌ {sym} ist bei Alpaca nicht handelbar ({asset.get('status')})")
            return False
        fractionable[sym] = bool(asset.get("fractionable"))
        if sym not in prices:
            try:
                snap = _api("https://data.alpaca.markets", key, secret,
                            "GET", f"/v2/stocks/{sym}/snapshot")
                prices[sym] = float(snap["latestTrade"]["p"])
            except Exception as e:
                log.error(f"❌ Kein Preis für {sym}: {e}")
                return False
    log.info("Ziel: " + ", ".join(f"{s} ${t['dollars']:,.0f}" + ("" if fractionable[s] else " (nur ganze Stück)")
                                  for s, t in targets.items()))

    try:
        orders = compute_orders(targets, current, prices, equity, fractionable)
    except ValueError as e:
        log.error(f"❌ {e}")
        return False

    extra = {"core_signal": sig["signal"], "signal_as_of": sig["as_of"], "timing": sig.get("timing", "nextday")}
    if not orders:
        print(f"\n✅ Keine Orders nötig — Portfolio entspricht dem Ziel (Toleranz {REBALANCE_TOLERANCE:.0%}).")
        if execute:
            _save_trade_log(equity, cash, current, prices, targets, [], execute, live, cum_deposits, extra)
        return True

    print(f"\n── {'⚠️  Orders' if execute else 'DRY-RUN'} (erst Verkäufe, dann Käufe) ──")
    for o in orders:
        icon = "🟢" if o["side"] == "buy" else "🔴"
        print(f"  {icon} {o['side'].upper():4s} {o['qty']:10.3f}× {o['symbol']:6s} (~${o['value']:,.0f})  — {o['reason']}")

    if not execute:
        print(f"\n  ℹ️  DRY-RUN — keine Orders gesendet, kein Log geschrieben.")
        print(f"  Für echte Orders: --execute (oder TRADING_ENABLED=1)")
        return True

    market_open = bool(client.get_clock().get("is_open"))
    log.info("Markt offen" if market_open else
             "Markt geschlossen — Orders werden zur nächsten Eröffnung ausgeführt")

    executed_orders, failed_orders = [], []
    sells = [o for o in orders if o["side"] == "sell"]
    buys = [o for o in orders if o["side"] == "buy"]
    for o in sells:
        _submit(client, o, executed_orders, failed_orders)
    if market_open and sells:
        _wait_for_fills(client, sells, failed_orders)

    if buys and sameday:
        secs = seconds_to_close(client.get_clock())
        if secs is None or secs < SAMEDAY_MIN_SECONDS_TO_CLOSE:
            log.error(f"❌ Nur noch {secs}s bis Handelsschluss — Käufe nicht mehr gesendet "
                      f"(Verkaufserlös bleibt Cash, der nächste Lauf holt es nach)")
            for o in buys:
                o["status"] = "SKIPPED: zu spät vor Handelsschluss"
                executed_orders.append(o)
                failed_orders.append(o)
            buys = []

    if buys:
        # Käufe nur mit vorhandenem Cash — nie Margin. Rest folgt beim nächsten Lauf.
        acct = client.get_account()
        budget = min(float(acct["cash"]), float(acct.get("buying_power") or 0)) * 0.995
        planned = sum(o["value"] for o in buys)
        if planned > budget:
            scale = max(budget, 0.0) / planned
            log.warning(f"  Kaufbudget ${budget:,.0f} < geplant ${planned:,.0f} → Käufe auf {scale:.0%} "
                        f"skaliert, Rest beim nächsten Lauf")
            for o in buys:
                o["qty"] = _floor_qty(o["qty"] * scale, fractionable.get(o["symbol"], True))
                o["value"] = round(o["qty"] * prices[o["symbol"]], 2)
            buys = [o for o in buys if o["qty"] > 0 and o["value"] >= MIN_ORDER_VALUE]
        for o in buys:
            _submit(client, o, executed_orders, failed_orders)
        if market_open and buys:
            _wait_for_fills(client, buys, failed_orders)

    if failed_orders:
        _send_failure_alert(failed_orders, equity, targets)
        _save_failed_orders(failed_orders, targets)

    _save_trade_log(equity, cash, current, prices, targets, executed_orders, execute, live, cum_deposits, extra)
    return not failed_orders


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


def _save_trade_log(equity, cash, positions, prices, targets, orders, executed, live: bool = False, cum_deposits: float = 0.0,
                    extra: dict | None = None):
    """Append daily entry to trade_log.json (paper) or trade_log_live.json (live)."""
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
    # Net capital contributions (deposits − withdrawals). Lets the return
    # calculation stay honest across deposits: real profit = equity − cum_deposits.
    if cum_deposits:
        entry["cum_deposits"] = round(cum_deposits, 2)
    if extra:
        entry.update(extra)

    # Load existing log
    log_data = []
    trade_log_path = TRADE_LOG_LIVE if live else TRADE_LOG_PAPER
    if trade_log_path.exists():
        try:
            with open(trade_log_path, "r") as f:
                log_data = json.load(f)
        except (json.JSONDecodeError, IOError):
            log_data = []

    if log_data and extra and log_data[-1].get("date", "")[:10] == now[:10] and \
            log_data[-1].get("signal_as_of") == extra.get("signal_as_of"):
        if not entry["orders"] and log_data[-1].get("orders"):
            entry["orders"] = log_data[-1]["orders"]
        log_data[-1] = entry
    else:
        log_data.append(entry)

    with open(trade_log_path, "w") as f:
        json.dump(log_data, f, indent=2, ensure_ascii=False)

    log.info(f"Trade-Log gespeichert: {trade_log_path} ({len(log_data)} Einträge)")


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
    ok = run(execute=execute, live=live, signals_path=sig_path)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
