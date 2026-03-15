#!/usr/bin/env python3
"""
E-Mail Notification Module — Multi-ETF
=========================================
Sendet Performance-Reports per Gmail SMTP für bis zu 4 ETFs.
HTML-Mail mit Tab-ähnlichen Sektionen:
  Tab 1: Aktuelles Signal (alle ETFs)
  Tab 2: Performance (Switch vs B&H)
  Tab 3: Strategie-Check (Ampel + Zeitfenster)
  Tab 4: Letzte Trades
"""

from __future__ import annotations
import json, smtplib, ssl, os
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from datetime import datetime

CONFIG_PATH = os.path.join(os.path.dirname(__file__), "email_config.json")

ETF_LABELS = {
    "SPY": "S&P 500",
    "URTH": "MSCI World",
    "EEM": "Emerging Markets",
    "VGK": "FTSE Europe",
}


def _load_config() -> dict:
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def send_email(subject: str, html_body: str, text_body: str = "") -> bool:
    """Send an email via Gmail SMTP. Returns True on success."""
    cfg = _load_config()
    msg = MIMEMultipart("alternative")
    msg["Subject"] = subject
    msg["From"] = cfg["sender_email"]
    msg["To"] = cfg["recipient_email"]

    if text_body:
        msg.attach(MIMEText(text_body, "plain", "utf-8"))
    msg.attach(MIMEText(html_body, "html", "utf-8"))

    try:
        ctx = ssl.create_default_context()
        with smtplib.SMTP(cfg["smtp_server"], cfg["smtp_port"]) as server:
            server.ehlo()
            server.starttls(context=ctx)
            server.ehlo()
            server.login(cfg["sender_email"], cfg["sender_password"])
            server.sendmail(cfg["sender_email"], cfg["recipient_email"],
                            msg.as_string())
        return True
    except Exception as e:
        print(f"[EMAIL ERROR] {e}")
        return False


# ── HTML helpers ─────────────────────────────────────────────────────────────

_STYLES = """
<style>
    body { font-family: Segoe UI, Helvetica, Arial, sans-serif; max-width: 780px;
           margin: 0 auto; color: #333; }
    .header { background: linear-gradient(135deg, #1a237e, #283593); color: white;
              padding: 24px; border-radius: 12px 12px 0 0; }
    .tab-bar { display: flex; background: #e8eaf6; }
    .tab { flex: 1; text-align: center; padding: 12px 8px; font-weight: bold;
           font-size: 14px; border-bottom: 3px solid transparent; color: #555; }
    .tab-active { border-bottom: 3px solid #1a237e; color: #1a237e;
                  background: white; }
    .section { padding: 16px; margin: 0; border: 1px solid #e0e0e0;
               border-top: none; }
    .etf-card { display: inline-block; width: 23%; min-width: 160px;
                vertical-align: top; margin: 4px; padding: 12px;
                border-radius: 8px; border: 1px solid #e0e0e0; text-align: center; }
    .signal-long { background: #E8F5E9; border-left: 4px solid #4CAF50; }
    .signal-cash { background: #FFEBEE; border-left: 4px solid #F44336; }
    table { width: 100%; border-collapse: collapse; font-size: 13px; }
    th { padding: 8px; text-align: left; background: #f5f5f5; }
    td { padding: 6px 8px; }
    .g { color: #2E7D32; background: #E8F5E9; }
    .y { color: #F57F17; background: #FFF8E1; }
    .r { color: #C62828; background: #FFEBEE; }
    .b { color: #1565C0; background: #E3F2FD; }
    .bold { font-weight: bold; }
    .footer { margin: 16px 0; padding: 12px; background: #f5f5f5;
              border-radius: 8px; font-size: 12px; color: #999; }
</style>
"""


def _sharpe_class(val):
    if val > 0.5:
        return "g"
    elif val > 0:
        return "y"
    return "r"


def _sharpe_icon(val):
    if val > 0.5:
        return "\U0001f7e2"
    elif val > 0:
        return "\U0001f7e1"
    return "\U0001f534"


def _pf(val, fmt=".1%"):
    if val is None:
        return "\u2013"
    return f"{val:{fmt}}"


# ── Main builder ─────────────────────────────────────────────────────────────

def build_multi_etf_email(
    all_data: dict,           # {ticker: {data dict from compute_all}}
    signal_changes: dict,     # {ticker: "INITIAL"|"KAUF"|"VERKAUF"|None}
) -> tuple[str, str]:
    """Build subject + HTML body for multi-ETF email with tab sections."""

    now = datetime.now().strftime("%d.%m.%Y %H:%M")
    date_short = datetime.now().strftime("%d.%m.%Y")

    # Determine subject from signal changes
    changes = {t: s for t, s in signal_changes.items() if s is not None}
    if not changes:
        subject_detail = "Keine Änderung"
    else:
        parts = []
        for t, sig in changes.items():
            if sig == "KAUF":
                parts.append(f"{t} \U0001f7e2 LONG")
            elif sig == "VERKAUF":
                parts.append(f"{t} \U0001f534 CASH")
            else:
                parts.append(f"{t} \U0001f680 Start")
        subject_detail = " | ".join(parts)

    subject = f"\U0001f4ca Strategy Switch {date_short}: {subject_detail}"

    # ╔══════════════════════════════════════════════════════════════════════╗
    # ║  TAB 1: AKTUELLES SIGNAL                                           ║
    # ╚══════════════════════════════════════════════════════════════════════╝

    signal_cards = ""
    for ticker, d in all_data.items():
        label = ETF_LABELS.get(ticker, ticker)
        price = d["price"]
        strat = d["current_strat"]
        invested = d["is_invested"]
        changed = signal_changes.get(ticker)

        if invested:
            cls = "signal-long"
            icon = "\u2705"
            status = f"LONG — {strat}"
        else:
            cls = "signal-cash"
            icon = "\U0001f534"
            status = "CASH"

        badge = ""
        if changed == "KAUF":
            badge = '<span style="background:#4CAF50;color:white;padding:2px 8px;border-radius:4px;font-size:11px;">NEU: KAUF</span>'
        elif changed == "VERKAUF":
            badge = '<span style="background:#F44336;color:white;padding:2px 8px;border-radius:4px;font-size:11px;">NEU: VERKAUF</span>'
        elif changed == "INITIAL":
            badge = '<span style="background:#1a237e;color:white;padding:2px 8px;border-radius:4px;font-size:11px;">INITIAL</span>'

        signal_cards += f"""
        <div class="etf-card {cls}">
            <div style="font-size:16px;font-weight:bold;">{ticker}</div>
            <div style="font-size:12px;color:#666;">{label}</div>
            <div style="font-size:18px;margin:8px 0;">{icon} {status}</div>
            <div style="font-size:14px;">${price:,.2f}</div>
            {f'<div style="margin-top:6px;">{badge}</div>' if badge else ''}
        </div>"""

    # Quick action summary
    actions = []
    for ticker, d in all_data.items():
        if d["is_invested"]:
            actions.append(f"<b>{ticker}</b>: \u2705 Halten ({d['current_strat']})")
        else:
            actions.append(f"<b>{ticker}</b>: \U0001f534 Cash")

    action_html = " &nbsp;|&nbsp; ".join(actions)

    # ╔══════════════════════════════════════════════════════════════════════╗
    # ║  TAB 2: PERFORMANCE                                                ║
    # ╚══════════════════════════════════════════════════════════════════════╝

    # Comparison table: all 4 ETFs
    perf_metrics = [
        ("CAGR", "sw_cagr", "bh_cagr", ".1%"),
        ("Sharpe", "sw_sharpe", "bh_sharpe", ".2f"),
        ("Sortino", "sw_sortino", "bh_sortino", ".2f"),
        ("Calmar", "sw_calmar", "bh_calmar", ".2f"),
        ("Max Drawdown", "sw_dd", "bh_dd", ".1%"),
        ("Volatilität", "sw_vol", "bh_vol", ".1%"),
        ("Max Underwater", "sw_max_uw", "bh_max_uw", "d"),
        ("Bestes Jahr", "sw_best_year", "bh_best_year", ".1%"),
        ("Schlechtestes Jahr", "sw_worst_year", "bh_worst_year", ".1%"),
        ("Win Rate", "sw_win_rate", "bh_win_rate", ".1%"),
    ]

    tickers = list(all_data.keys())

    # Header row
    perf_header = '<th style="padding:8px;text-align:left;">Kennzahl</th>'
    for t in tickers:
        perf_header += f'<th style="padding:8px;text-align:center;" colspan="2">{t}</th>'

    perf_subheader = '<td></td>'
    for t in tickers:
        perf_subheader += '<td style="padding:4px;text-align:center;font-size:11px;color:#666;">Switch</td>'
        perf_subheader += '<td style="padding:4px;text-align:center;font-size:11px;color:#666;">B&H</td>'

    perf_rows = ""
    for i, (name, sw_key, bh_key, fmt) in enumerate(perf_metrics):
        bg = ' style="background:#f8f9fa;"' if i % 2 else ""
        row = f'<tr{bg}><td style="padding:6px 8px;font-weight:bold;">{name}</td>'
        for t in tickers:
            perf = all_data[t]["perf"]
            sw_val = perf.get(sw_key)
            bh_val = perf.get(bh_key)
            if fmt == "d":
                sw_str = str(sw_val) if sw_val is not None else "\u2013"
                bh_str = str(bh_val) if bh_val is not None else "\u2013"
                if name == "Max Underwater":
                    sw_str += "T"
                    bh_str += "T"
            else:
                sw_str = _pf(sw_val, fmt)
                bh_str = _pf(bh_val, fmt)
            row += f'<td style="padding:6px 4px;text-align:center;font-weight:bold;">{sw_str}</td>'
            row += f'<td style="padding:6px 4px;text-align:center;color:#666;">{bh_str}</td>'
        row += '</tr>'
        perf_rows += row

    # ╔══════════════════════════════════════════════════════════════════════╗
    # ║  TAB 3: STRATEGIE-CHECK                                            ║
    # ╚══════════════════════════════════════════════════════════════════════╝

    strat_check_blocks = ""
    for ticker in tickers:
        d = all_data[ticker]
        strat_check = d.get("strat_check", {})
        strat_sharpes = d.get("strat_sharpes", {})
        label = ETF_LABELS.get(ticker, ticker)

        windows = list(strat_check.keys())
        strategies = sorted({s for w in strat_check.values() for s in w
                             if not s.startswith("_")})

        # Time-window table
        check_header = "".join(
            f'<th style="padding:6px 8px;text-align:center;">{w}</th>'
            for w in windows)

        check_rows = ""
        for s in strategies:
            cells = ""
            for w in windows:
                val = strat_check.get(w, {}).get(s, 0.0)
                cls = _sharpe_class(val)
                cells += f'<td class="{cls} bold" style="padding:6px 8px;text-align:center;">{val:.2f}</td>'
            check_rows += f'<tr><td style="padding:6px 8px;font-weight:bold;">{s}</td>{cells}</tr>\n'

        # Switch row
        sw_cells = ""
        for w in windows:
            val = strat_check.get(w, {}).get("_switch", 0.0)
            cls = _sharpe_class(val)
            sw_cells += f'<td class="{cls} bold" style="padding:6px 8px;text-align:center;">{val:.2f}</td>'
        check_rows += f'<tr style="border-top:2px solid #333;"><td style="padding:6px 8px;font-weight:bold;">\U0001f4ca Switch</td>{sw_cells}</tr>\n'

        # B&H row
        bh_cells = ""
        for w in windows:
            val = strat_check.get(w, {}).get("_bh", 0.0)
            bh_cells += f'<td class="b" style="padding:6px 8px;text-align:center;">{val:.2f}</td>'
        check_rows += f'<tr><td style="padding:6px 8px;color:#1565C0;">\U0001f4c9 Buy & Hold</td>{bh_cells}</tr>\n'

        # Ampel
        ampel_rows = ""
        for sname, sh in strat_sharpes.items():
            icon = _sharpe_icon(sh)
            cls = _sharpe_class(sh)
            ampel_rows += f'<tr><td style="padding:4px 8px;">{icon} {sname}</td><td class="{cls} bold" style="padding:4px 8px;">{sh:.2f}</td></tr>'

        strat_check_blocks += f"""
        <div style="margin-bottom:16px;padding:12px;border:1px solid #e0e0e0;border-radius:8px;">
            <h4 style="margin:0 0 8px 0;">{ticker} — {label}</h4>
            <table>
                <tr style="background:#f5f5f5;">
                    <th style="padding:6px 8px;text-align:left;">Strategie</th>
                    {check_header}
                </tr>
                {check_rows}
            </table>
            <div style="margin-top:12px;">
                <b>\U0001f6a6 Ampel (63T Rolling Sharpe)</b>
                <table style="margin-top:4px;">{ampel_rows}</table>
            </div>
        </div>"""

    # ╔══════════════════════════════════════════════════════════════════════╗
    # ║  TAB 4: LETZTE TRADES                                              ║
    # ╚══════════════════════════════════════════════════════════════════════╝

    trades_blocks = ""
    for ticker in tickers:
        d = all_data[ticker]
        recent = d.get("trades", [])[-8:]
        if not recent:
            continue
        label = ETF_LABELS.get(ticker, ticker)
        trade_rows = ""
        for t in recent:
            color = "#4CAF50" if t.get("Aktion", "").startswith("KAUF") else "#F44336"
            trade_rows += f"""
            <tr>
                <td style="padding:4px 8px;">{t.get('Datum','')}</td>
                <td style="padding:4px 8px;color:{color};font-weight:bold;">{t.get('Aktion','')}</td>
                <td style="padding:4px 8px;">{t.get('Kurs','')}</td>
                <td style="padding:4px 8px;">{t.get('Grund','')}</td>
            </tr>"""

        trades_blocks += f"""
        <div style="margin-bottom:12px;">
            <b>{ticker} — {label}</b>
            <table style="margin-top:4px;">
                <tr style="background:#f5f5f5;">
                    <th style="padding:4px 8px;">Datum</th>
                    <th style="padding:4px 8px;">Aktion</th>
                    <th style="padding:4px 8px;">Kurs</th>
                    <th style="padding:4px 8px;">Grund</th>
                </tr>
                {trade_rows}
            </table>
        </div>"""

    # ╔══════════════════════════════════════════════════════════════════════╗
    # ║  ASSEMBLE HTML                                                     ║
    # ╚══════════════════════════════════════════════════════════════════════╝

    html = f"""
    <html>
    <head>{_STYLES}</head>
    <body>

        <!-- Header -->
        <div class="header">
            <h1 style="margin:0;font-size:22px;">\U0001f4ca Strategy Switch \u2014 Multi-ETF Daily Report</h1>
            <p style="margin:8px 0 0 0;opacity:0.9;">{now}</p>
        </div>

        <!-- Tab Bar -->
        <div class="tab-bar">
            <div class="tab tab-active">\U0001f3af Signal</div>
            <div class="tab">\U0001f4c8 Performance</div>
            <div class="tab">\U0001f50d Strategie-Check</div>
            <div class="tab">\U0001f4cb Trades</div>
        </div>

        <!-- ══════════════ TAB 1: SIGNAL ══════════════ -->
        <div class="section" style="background:#fafafa;">
            <h3 style="margin:0 0 6px 0;">\U0001f3af Aktuelles Signal</h3>
            <p style="margin:0 0 12px 0;font-size:13px;color:#666;">
                {action_html}
            </p>
            <div style="text-align:center;">
                {signal_cards}
            </div>
        </div>

        <!-- ══════════════ TAB 2: PERFORMANCE ══════════════ -->
        <div class="section">
            <h3 style="margin:0 0 12px 0;">\U0001f4c8 Performance-Vergleich</h3>
            <p style="margin:0 0 8px 0;font-size:12px;color:#666;">
                Switch = Strategy-Switch Methodik  |  B&H = Buy & Hold
            </p>
            <div style="overflow-x:auto;">
                <table>
                    <tr style="background:#BBDEFB;">{perf_header}</tr>
                    <tr style="background:#E3F2FD;">{perf_subheader}</tr>
                    {perf_rows}
                </table>
            </div>
        </div>

        <!-- ══════════════ TAB 3: STRATEGIE-CHECK ══════════════ -->
        <div class="section">
            <h3 style="margin:0 0 6px 0;">\U0001f50d Strategie-Gesundheitscheck</h3>
            <p style="margin:0 0 12px 0;font-size:12px;color:#666;">
                Sharpe Ratio \u00fcber verschiedene Zeitfenster.
                \U0001f7e2 &gt; 0.5 \u00b7 \U0001f7e1 0\u20130.5 \u00b7 \U0001f534 &lt; 0
            </p>
            {strat_check_blocks}
        </div>

        <!-- ══════════════ TAB 4: TRADES ══════════════ -->
        <div class="section">
            <h3 style="margin:0 0 12px 0;">\U0001f4cb Letzte Trades</h3>
            {trades_blocks if trades_blocks else '<p style="color:#999;">Keine Trades in der letzten Zeit.</p>'}
        </div>

        <!-- Footer -->
        <div class="footer">
            <p style="margin:0;">
                \u26a0\ufe0f Keine Anlageberatung. Vergangene Performance garantiert keine zuk\u00fcnftige Rendite.<br>
                Automatisch generiert vom Strategy Switch System.<br>
                N\u00e4chste Mail nur bei Signal-\u00c4nderung (Cash \u2194 Long) bei mindestens einem ETF.
            </p>
        </div>

    </body>
    </html>
    """

    # Plain text fallback
    text_lines = [f"Strategy Switch Report {date_short}", "=" * 40, ""]
    for ticker, d in all_data.items():
        label = ETF_LABELS.get(ticker, ticker)
        pos = "LONG" if d["is_invested"] else "CASH"
        perf = d["perf"]
        text_lines.append(f"{ticker} ({label}): {pos} | {d['current_strat']}")
        text_lines.append(f"  Kurs: ${d['price']:,.2f} | "
                          f"Switch Sharpe: {_pf(perf.get('sw_sharpe'), '.2f')} | "
                          f"B&H Sharpe: {_pf(perf.get('bh_sharpe'), '.2f')}")
        text_lines.append("")

    return subject, html
