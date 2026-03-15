#!/usr/bin/env python3
"""
start_tunnel.py — Start cloudflared tunnel, capture URL, email it, keep running.
"""

import json
import re
import smtplib
import subprocess
import sys
import time
from email.mime.text import MIMEText
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent.parent
CLOUDFLARED = PROJECT_DIR / "cloudflared.exe"
EMAIL_CFG   = PROJECT_DIR / "wf_backtest" / "email_config.json"
URL_FILE    = PROJECT_DIR / "tunnel_url.txt"

def send_url_email(url: str) -> None:
    """Send the tunnel URL via Gmail."""
    try:
        with open(EMAIL_CFG, "r", encoding="utf-8") as f:
            cfg = json.load(f)

        body = (
            f"Dein Strategy-Switch Dashboard ist erreichbar unter:\n\n"
            f"  {url}\n\n"
            f"Öffne den Link auf dem iPhone und füge ihn "
            f"zum Home-Bildschirm hinzu (Safari → Teilen → Zum Home-Bildschirm).\n\n"
            f"Die URL ändert sich bei jedem PC-Neustart.\n"
        )
        msg = MIMEText(body, "plain", "utf-8")
        msg["Subject"] = "📱 Strategy-Switch — Neue Dashboard-URL"
        msg["From"]    = cfg["sender_email"]
        msg["To"]      = cfg["recipient_email"]

        with smtplib.SMTP(cfg["smtp_server"], cfg["smtp_port"]) as s:
            s.starttls()
            s.login(cfg["sender_email"], cfg["sender_password"])
            s.send_message(msg)
        print(f"  ✅ URL per E-Mail gesendet an {cfg['recipient_email']}")
    except Exception as e:
        print(f"  ⚠️  E-Mail fehlgeschlagen: {e}")


def main():
    if not CLOUDFLARED.exists():
        print(f"  ❌ cloudflared.exe nicht gefunden: {CLOUDFLARED}")
        sys.exit(1)

    print("  Starte cloudflared tunnel …")
    proc = subprocess.Popen(
        [str(CLOUDFLARED), "tunnel", "--url", "http://localhost:8511"],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )

    url_found = False
    url_pattern = re.compile(r"(https://[a-z0-9\-]+\.trycloudflare\.com)")

    for line in proc.stdout:
        line = line.strip()
        if not line:
            continue

        # Look for the URL
        m = url_pattern.search(line)
        if m and not url_found:
            url = m.group(1)
            url_found = True

            print(f"\n  {'=' * 55}")
            print(f"  📱 Dashboard-URL:")
            print(f"  {url}")
            print(f"  {'=' * 55}\n")

            # Save to file
            URL_FILE.write_text(url, encoding="utf-8")
            print(f"  URL gespeichert: {URL_FILE}")

            # Email it
            send_url_email(url)
            print(f"\n  Tunnel läuft. CTRL+C zum Beenden.\n")

    proc.wait()


if __name__ == "__main__":
    main()
