#!/bin/bash
# ============================================================================
# Strategy-Switch — Oracle Cloud VM Setup
# ============================================================================
# Dieses Script richtet alles auf einer frischen Oracle Cloud Ubuntu VM ein:
#   - Python 3.12, pip, venv
#   - Projekt-Abhängigkeiten
#   - Gunicorn als Systemd-Service (startet automatisch, always-on)
#   - Nginx als Reverse-Proxy (Port 80 → 8511)
#   - Cron-Job für tägliche Neuberechnung (22:30 UTC)
#
# Aufruf: bash setup.sh
# ============================================================================

set -e

APP_DIR="$HOME/strategy-switch"
VENV="$APP_DIR/.venv"

echo ""
echo "  =================================================="
echo "  Strategy-Switch — Oracle Cloud Setup"
echo "  =================================================="
echo ""

# ── 1. System-Pakete ─────────────────────────────────────────────────────────
echo "[1/6] Installiere System-Pakete …"
sudo apt-get update -qq
sudo apt-get install -y -qq python3 python3-pip python3-venv nginx git > /dev/null

# ── 2. Python venv + Abhängigkeiten ──────────────────────────────────────────
echo "[2/6] Erstelle Python-Umgebung …"
cd "$APP_DIR"
python3 -m venv "$VENV"
"$VENV/bin/pip" install --upgrade pip -q
"$VENV/bin/pip" install -r requirements.txt -q
"$VENV/bin/pip" install gunicorn -q
echo "  ✅ Python-Umgebung bereit"

# ── 3. Erste Berechnung ─────────────────────────────────────────────────────
echo "[3/6] Starte erste Berechnung (kann 2-5 Min. dauern) …"
cd "$APP_DIR"
"$VENV/bin/python" -m wf_backtest.daily_runner || echo "  ⚠️  Erste Berechnung fehlgeschlagen (wird beim nächsten Request erneut versucht)"

# ── 4. Systemd-Service ──────────────────────────────────────────────────────
echo "[4/6] Erstelle Systemd-Service …"
sudo tee /etc/systemd/system/strategy-switch.service > /dev/null << UNIT
[Unit]
Description=Strategy-Switch PWA
After=network.target

[Service]
Type=simple
User=$USER
WorkingDirectory=$APP_DIR
ExecStart=$VENV/bin/gunicorn mobile_app.server:app --bind 127.0.0.1:8511 --timeout 900 --workers 2 --threads 2
Restart=always
RestartSec=5
Environment="PATH=$VENV/bin:/usr/bin"

[Install]
WantedBy=multi-user.target
UNIT

sudo systemctl daemon-reload
sudo systemctl enable strategy-switch
sudo systemctl start strategy-switch
echo "  ✅ Service gestartet"

# ── 5. Nginx Reverse Proxy ──────────────────────────────────────────────────
echo "[5/6] Konfiguriere Nginx …"
sudo tee /etc/nginx/sites-available/strategy-switch > /dev/null << 'NGINX'
server {
    listen 80;
    server_name _;

    location / {
        proxy_pass http://127.0.0.1:8511;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_read_timeout 900s;
        proxy_connect_timeout 60s;
    }
}
NGINX

sudo ln -sf /etc/nginx/sites-available/strategy-switch /etc/nginx/sites-enabled/
sudo rm -f /etc/nginx/sites-enabled/default
sudo nginx -t && sudo systemctl reload nginx
echo "  ✅ Nginx konfiguriert"

# ── 6. Cron-Job für tägliche Berechnung ─────────────────────────────────────
echo "[6/6] Erstelle Cron-Job (täglich 22:30 UTC) …"
CRON_CMD="30 22 * * 1-5 cd $APP_DIR && $VENV/bin/python -m wf_backtest.daily_runner >> $APP_DIR/cron.log 2>&1"
(crontab -l 2>/dev/null | grep -v "daily_runner"; echo "$CRON_CMD") | crontab -
echo "  ✅ Cron-Job eingerichtet"

# ── Fertig ───────────────────────────────────────────────────────────────────
PUBLIC_IP=$(curl -s ifconfig.me 2>/dev/null || echo "DEINE_IP")
echo ""
echo "  =================================================="
echo "  ✅ Setup abgeschlossen!"
echo "  =================================================="
echo ""
echo "  Dashboard erreichbar unter:"
echo "  http://$PUBLIC_IP"
echo ""
echo "  Tägliche Neuberechnung: Mo-Fr 22:30 UTC"
echo "  Service-Status: sudo systemctl status strategy-switch"
echo "  Logs: sudo journalctl -u strategy-switch -f"
echo ""
