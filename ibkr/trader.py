"""
IBKR Strategy-Switch Auto-Trader
================================
Reads signals from mobile_signals.json and places orders via Interactive Brokers.

Usage:
  python -m ibkr.trader                    # Dry-run (nur anzeigen)
  python -m ibkr.trader --execute          # Orders wirklich senden
  python -m ibkr.trader --execute --live   # LIVE-Konto (Vorsicht!)

Requirements:
  pip install ib_insync

Ports:
  TWS Paper:     7497
  TWS Live:      7496
  Gateway Paper:  4002
  Gateway Live:   4001
"""

import json
import argparse
import logging
from pathlib import Path
from dataclasses import dataclass

try:
    from ib_insync import IB, Stock, MarketOrder, util
except ImportError:
    print("❌ ib_insync nicht installiert. Bitte: pip install ib_insync")
    raise SystemExit(1)

log = logging.getLogger("ibkr_trader")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

SIGNALS_FILE = Path(__file__).parent.parent / "wf_backtest" / "mobile_signals.json"

# ── Ticker → IBKR Contract mapping ──────────────────────────────────────────
# IBKR uses different symbols/exchanges than Yahoo Finance
TICKER_MAP = {
    # ETFs
    "SXR8.DE": {"symbol": "SXR8", "exchange": "IBIS", "currency": "EUR", "type": "STK"},
    "URTH":    {"symbol": "URTH", "exchange": "ARCA", "currency": "USD", "type": "STK"},
    "EEM":     {"symbol": "EEM",  "exchange": "ARCA", "currency": "USD", "type": "STK"},
    "VGK":     {"symbol": "VGK",  "exchange": "ARCA", "currency": "USD", "type": "STK"},
}

# Mega-cap stocks (Schwarm/Value/Turnaround) — IBKR erkennt die meisten direkt
# Für US-Aktien: exchange=SMART, currency=USD
# Für EU-Aktien: anpassen nach Bedarf


@dataclass
class TradeOrder:
    """Represents a single order to place."""
    ticker: str
    action: str          # "BUY" or "SELL"
    quantity: int
    reason: str
    ibkr_symbol: str
    exchange: str
    currency: str


def load_signals(path: Path = SIGNALS_FILE) -> dict:
    """Load the latest computed signals."""
    if not path.exists():
        log.error(f"Signaldatei nicht gefunden: {path}")
        raise FileNotFoundError(path)
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def build_ibkr_contract(ticker: str) -> Stock:
    """Create an IBKR Stock contract for a given ticker."""
    if ticker in TICKER_MAP:
        m = TICKER_MAP[ticker]
        return Stock(m["symbol"], m["exchange"], m["currency"])
    # Default: assume US stock on SMART
    clean = ticker.replace(".DE", "").replace(".L", "").replace(".PA", "")
    return Stock(clean, "SMART", "USD")


def compute_target_positions(signals: dict, capital: float,
                             etf_weight: float = 0.15,
                             category_weight: float = 0.18) -> dict[str, dict]:
    """
    Compute target positions based on signals and capital.

    Allocation:
      - 4 ETFs:        4 × 15% = 60% max
      - Schwarm:        18% (split across LONG stocks)
      - Value:          12% (split across LONG stocks)
      - Turnaround:     10% (split across LONG stocks)
      Total:           100%

    Returns dict: {ticker: {"target_shares": int, "signal": str, "price": float}}
    """
    targets = {}

    # ── ETFs ──
    for etf in signals.get("etfs", []):
        ticker = etf["ticker"]
        signal = etf["signal"]
        price = etf["price"]
        if signal == "LONG" and price > 0:
            alloc = capital * etf_weight
            shares = int(alloc / price)
        else:
            shares = 0
        targets[ticker] = {
            "target_shares": shares,
            "signal": signal,
            "price": price,
            "reason": f"ETF {signal} ({etf['strategy']})",
        }

    # ── Categories (Schwarm / Value / Turnaround) ──
    cat_weights = {"swarm": 0.18, "value": 0.12, "turnaround": 0.10}
    for cat_key, weight in cat_weights.items():
        cat = signals.get(cat_key)
        if not cat:
            continue
        stocks = cat.get("stocks", [])
        n_long = sum(1 for s in stocks if s["signal"] == "LONG")
        if n_long == 0:
            for s in stocks:
                targets[s["ticker"]] = {
                    "target_shares": 0,
                    "signal": "CASH",
                    "price": 0,
                    "reason": f"{cat_key} CASH",
                }
            continue

        alloc_per_stock = (capital * weight) / n_long
        for s in stocks:
            ticker = s["ticker"]
            signal = s["signal"]
            # Get approximate price from IBKR later; use 0 as placeholder
            if signal == "LONG":
                targets[ticker] = {
                    "target_shares": None,  # Will be resolved with live prices
                    "alloc": alloc_per_stock,
                    "signal": signal,
                    "price": None,
                    "reason": f"{cat_key} LONG ({s.get('strategy', '?')})",
                }
            else:
                targets[ticker] = {
                    "target_shares": 0,
                    "signal": "CASH",
                    "price": 0,
                    "reason": f"{cat_key} CASH",
                }

    return targets


def compute_orders(targets: dict, current_positions: dict) -> list[TradeOrder]:
    """
    Compare target vs current positions and generate orders.

    current_positions: {symbol: quantity} from IBKR
    """
    orders = []

    all_tickers = set(targets.keys()) | set(current_positions.keys())

    for ticker in sorted(all_tickers):
        target = targets.get(ticker, {"target_shares": 0, "reason": "Nicht mehr im Portfolio"})
        target_qty = target.get("target_shares", 0) or 0
        current_qty = current_positions.get(ticker, 0)

        diff = target_qty - current_qty

        if diff == 0:
            continue

        m = TICKER_MAP.get(ticker, {})
        ibkr_symbol = m.get("symbol", ticker.replace(".DE", ""))
        exchange = m.get("exchange", "SMART")
        currency = m.get("currency", "USD")

        if diff > 0:
            orders.append(TradeOrder(
                ticker=ticker,
                action="BUY",
                quantity=diff,
                reason=target.get("reason", "Signal LONG"),
                ibkr_symbol=ibkr_symbol,
                exchange=exchange,
                currency=currency,
            ))
        else:
            orders.append(TradeOrder(
                ticker=ticker,
                action="SELL",
                quantity=abs(diff),
                reason=target.get("reason", "Signal CASH"),
                ibkr_symbol=ibkr_symbol,
                exchange=exchange,
                currency=currency,
            ))

    return orders


def resolve_stock_prices(ib: IB, targets: dict) -> dict:
    """Fetch live prices for stocks where target_shares is None (needs price)."""
    for ticker, info in targets.items():
        if info.get("target_shares") is None and info.get("alloc"):
            contract = build_ibkr_contract(ticker)
            ib.qualifyContracts(contract)
            ticker_data = ib.reqMktData(contract, '', False, False)
            ib.sleep(2)  # Wait for data
            price = ticker_data.marketPrice()
            if price and price > 0:
                info["price"] = price
                info["target_shares"] = int(info["alloc"] / price)
            else:
                log.warning(f"  Kein Preis für {ticker} — übersprungen")
                info["target_shares"] = 0
            ib.cancelMktData(contract)
    return targets


def execute_orders(ib: IB, orders: list[TradeOrder], dry_run: bool = True) -> list[dict]:
    """Place orders via IBKR. Returns list of order results."""
    results = []

    for order in orders:
        contract = Stock(order.ibkr_symbol, order.exchange, order.currency)
        ib.qualifyContracts(contract)

        mkt_order = MarketOrder(order.action, order.quantity)

        log.info(f"  {'🔵 DRY-RUN' if dry_run else '🟢 EXECUTE'}: "
                 f"{order.action} {order.quantity}x {order.ticker} "
                 f"({order.ibkr_symbol}@{order.exchange}) — {order.reason}")

        if dry_run:
            results.append({
                "ticker": order.ticker,
                "action": order.action,
                "quantity": order.quantity,
                "status": "DRY_RUN",
                "reason": order.reason,
            })
        else:
            trade = ib.placeOrder(contract, mkt_order)
            ib.sleep(1)
            results.append({
                "ticker": order.ticker,
                "action": order.action,
                "quantity": order.quantity,
                "status": str(trade.orderStatus.status),
                "order_id": trade.order.orderId,
                "reason": order.reason,
            })

    return results


def get_current_positions(ib: IB) -> dict[str, int]:
    """Get current portfolio positions as {ticker: quantity}."""
    positions = {}
    for pos in ib.positions():
        symbol = pos.contract.symbol
        # Try to reconstruct the Yahoo ticker
        if pos.contract.exchange == "IBIS":
            symbol += ".DE"
        positions[symbol] = int(pos.position)
    return positions


def get_account_summary(ib: IB) -> dict:
    """Get key account metrics."""
    values = ib.accountSummary()
    result = {}
    for v in values:
        if v.tag in ("NetLiquidation", "TotalCashValue", "GrossPositionValue"):
            if v.currency in ("USD", "EUR", "BASE"):
                result[f"{v.tag}_{v.currency}"] = float(v.value)
    return result


def main():
    parser = argparse.ArgumentParser(description="IBKR Strategy-Switch Trader")
    parser.add_argument("--execute", action="store_true",
                        help="Orders wirklich senden (sonst nur Dry-Run)")
    parser.add_argument("--live", action="store_true",
                        help="Live-Konto statt Paper Trading")
    parser.add_argument("--host", default="127.0.0.1",
                        help="IBKR Gateway Host (default: 127.0.0.1)")
    parser.add_argument("--capital", type=float, default=None,
                        help="Kapital in USD (default: automatisch vom Konto)")
    parser.add_argument("--signals", type=str, default=None,
                        help="Pfad zur Signaldatei (default: mobile_signals.json)")
    args = parser.parse_args()

    # Port selection
    port = 7496 if args.live else 7497  # TWS
    mode = "🔴 LIVE" if args.live else "🟢 PAPER"
    dry = not args.execute

    print(f"\n{'='*60}")
    print(f"  IBKR Strategy-Switch Trader")
    print(f"  Modus: {mode}  |  {'DRY-RUN' if dry else '⚠️  EXECUTE'}")
    print(f"  Port: {port}  |  Host: {args.host}")
    print(f"{'='*60}\n")

    if args.execute and args.live:
        confirm = input("⚠️  LIVE-ORDERS WERDEN GESENDET! Fortfahren? (ja/nein): ")
        if confirm.strip().lower() != "ja":
            print("Abgebrochen.")
            return

    # Load signals
    sig_path = Path(args.signals) if args.signals else SIGNALS_FILE
    signals = load_signals(sig_path)
    log.info(f"Signale geladen: {len(signals.get('etfs', []))} ETFs, "
             f"Schwarm: {'✓' if signals.get('swarm') else '✗'}, "
             f"Value: {'✓' if signals.get('value') else '✗'}, "
             f"Turnaround: {'✓' if signals.get('turnaround') else '✗'}")

    # Print current signals
    print("\n── Aktuelle Signale ──")
    for etf in signals.get("etfs", []):
        sig = etf["signal"]
        icon = "🟢" if sig == "LONG" else "🔴"
        print(f"  {icon} {etf['ticker']:12s} {sig:6s}  ({etf['strategy']})  ${etf['price']:.2f}")

    for cat_key in ("swarm", "value", "turnaround"):
        cat = signals.get(cat_key)
        if not cat:
            continue
        print(f"\n  {cat_key.upper()}:")
        for s in cat["stocks"]:
            sig = s["signal"]
            icon = "🟢" if sig == "LONG" else "🔴"
            print(f"    {icon} {s['ticker']:10s} {sig:6s}  ({s.get('strategy', '?')})")

    # Connect to IBKR
    ib = IB()
    try:
        log.info(f"Verbinde mit IBKR ({args.host}:{port}) ...")
        ib.connect(args.host, port, clientId=1)
        log.info("✅ Verbunden!")
    except Exception as e:
        log.error(f"❌ Verbindung fehlgeschlagen: {e}")
        log.error("Stelle sicher, dass TWS oder IB Gateway läuft und API aktiviert ist.")
        log.error("TWS → Edit → Global Configuration → API → Settings → Enable ActiveX and Socket Clients")
        return

    try:
        # Account info
        summary = get_account_summary(ib)
        capital = args.capital
        if capital is None:
            capital = summary.get("NetLiquidation_USD") or summary.get("NetLiquidation_BASE", 0)
        log.info(f"Kontostand: ${capital:,.2f}")

        # Current positions
        current_pos = get_current_positions(ib)
        if current_pos:
            print("\n── Aktuelle Positionen ──")
            for t, q in sorted(current_pos.items()):
                print(f"  {t}: {q} Stück")
        else:
            print("\n  Keine offenen Positionen.")

        # Compute targets
        targets = compute_target_positions(signals, capital)

        # Resolve prices for stocks without price info
        targets = resolve_stock_prices(ib, targets)

        # Compute orders
        orders = compute_orders(targets, current_pos)

        if not orders:
            print("\n✅ Keine Orders nötig — Portfolio ist aktuell.")
            return

        # Print & execute orders
        print(f"\n── {'DRY-RUN Orders' if dry else '⚠️  Echte Orders'} ──")
        results = execute_orders(ib, orders, dry_run=dry)

        # Summary
        print(f"\n── Zusammenfassung ──")
        buys = [r for r in results if r["action"] == "BUY"]
        sells = [r for r in results if r["action"] == "SELL"]
        print(f"  Käufe:   {len(buys)}")
        print(f"  Verkäufe: {len(sells)}")
        if dry:
            print(f"\n  ℹ️  Dies war ein DRY-RUN. Für echte Orders: python -m ibkr.trader --execute")

    finally:
        ib.disconnect()
        log.info("Verbindung getrennt.")


if __name__ == "__main__":
    main()
