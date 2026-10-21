# Auto-Trading Setup

## Vergleich: Alpaca vs IBKR

| Feature              | Alpaca (empfohlen)     | IBKR                  |
|----------------------|------------------------|-----------------------|
| **PC nötig?**        | ❌ Nein                | ✅ Ja (TWS muss laufen)|
| **GitHub Actions?**  | ✅ Direkt integriert   | ❌ Nicht möglich       |
| **REST API?**        | ✅ Ja                  | ⚠️ Nur mit Gateway    |
| **Paper Trading?**   | ✅ Kostenlos           | ✅ Kostenlos           |
| **EU-ETFs?**         | ❌ Nur US-ETFs         | ✅ Alle                |
| **US-Aktien?**       | ✅ Ja                  | ✅ Ja                  |
| **Kosten?**          | $0 Kommission          | $0 (Tiered)           |

**→ Empfehlung: Alpaca** (kein PC nötig, läuft automatisch in GitHub Actions)

> SXR8.DE (EU) wird automatisch durch SPY (US-Äquivalent) ersetzt.

---

# Option A: Alpaca (kein PC nötig)

## 1. Alpaca-Konto erstellen

1. Gehe zu **[alpaca.markets](https://alpaca.markets)**
2. **Sign Up** → Paper-Trading-Konto ist sofort aktiv
3. Im Dashboard: **API Keys** → Neuen Key generieren
4. **Key ID** und **Secret Key** aufschreiben

## 2. GitHub Secrets setzen

In deinem GitHub Repo: **Settings → Secrets and variables → Actions → New repository secret**

| Secret Name         | Wert                        |
|--------------------|-----------------------------|
| `ALPACA_KEY_ID`    | Dein API Key ID             |
| `ALPACA_SECRET_KEY`| Dein API Secret Key         |
| `TRADING_ENABLED`  | `1` (Orders wirklich senden)|
| `ALPACA_LIVE`      | Leer lassen (= Paper)      |

## 3. Fertig!

Der Ablauf ist jetzt vollautomatisch:

```
Mo-Fr 22:30 UTC → GitHub Actions startet
  ↓
Signale berechnen → mobile_signals.json
  ↓
JSON auf GitHub Pages → PWA aktualisiert
  ↓
Alpaca Trader → Orders automatisch platziert
  ↓
Du bekommst die Signale auf dem Handy + Portfolio ist aktuell
```

**Kein PC nötig. Alles läuft in der Cloud.**

## 4. Stufenweise aktivieren

| Schritt | GitHub Secrets | Was passiert |
|---------|---------------|--------------|
| 1. Test | Nur Keys setzen | Dry-Run: zeigt Orders im Log, sendet nichts |
| 2. Paper | + `TRADING_ENABLED=1` | Paper-Trading: echte Orders, Spielgeld |
| 3. Live | + `ALPACA_LIVE=1` | **Echtes Geld!** |

## 5. Lokal testen (optional)

```bash
# Env-Variablen setzen
set ALPACA_KEY_ID=dein_key
set ALPACA_SECRET_KEY=dein_secret

# Dry-Run
python -m trading.alpaca_trader

# Paper-Trading
python -m trading.alpaca_trader --execute

# Live (Vorsicht!)
python -m trading.alpaca_trader --execute --live
```

## 6. Same-Day-Handel (Core-Strategie)

Standard: Signal und Orders 7–35 min vor US-Handelsschluss am Signaltag (`TRADE_TIMING` leer oder `sameday`).

Zurückschalten auf Handel am Folgetag (Signal aus dem Vortagesschluss, Orders um 14:45 UTC) ohne Code-Deploy:
**Settings → Secrets and variables → Actions → Variables → `TRADE_TIMING` = `nextday`**.

| Baustein | Aufgabe |
|----------|---------|
| `daily.yml` → Job `gate` + `trading/trade_window.py` | wählt pro Lauf Timing und prüft mit der Alpaca-Marktuhr das Zeitfenster (Sommer-/Winterzeit, Feiertage, früher Schluss) |
| 4 Cron-Läufe (19:28/19:41 und 20:28/20:41 UTC) | der zweite Lauf je Zeitzone ist das Backup bei verspätetem GitHub-Cron; er handelt nur, wenn sich das Signal geändert hat |
| `wf_backtest/core_intraday.py` | Live-Indexstand: Alpaca-Snapshot SPY/QQQ skaliert auf den letzten Index-Schluss, Gegenprüfung mit yfinance (>1 % Abweichung = kein Signal) |
| `alpaca_trader.py` | handelt ein Live-Signal nur, wenn es <15 min alt ist, vom heutigen Tag stammt und noch ≥90 s bis Handelsschluss bleiben |

Bekannte Grenzen: An frühen Handelsschlüssen (13:00 ET) wird übersprungen, der Folgetag holt es nach. Fällt ein Lauf
aus, bleibt die Position bis zum nächsten Lauf unverändert. An Ex-Dividenden-Tagen weicht der ETF-Proxy um die
Dividende (SPY ~0,3 %, QQQ ~0,15 %) vom Index ab.

Tests: `python -m unittest test_same_day -v`

## Portfolio-Aufteilung

| Kategorie   | Gewicht | Alpaca Symbol |
|------------|---------|---------------|
| S&P 500    | 15%     | SPY           |
| MSCI World | 15%     | URTH          |
| EM         | 15%     | EEM           |
| Europa     | 15%     | VGK           |
| Schwarm    | 18%     | Einzelaktien  |
| Value      | 12%     | Einzelaktien  |
| Turnaround | 10%     | Einzelaktien  |

---

# Option B: IBKR (braucht laufenden PC)

Siehe `ibkr/trader.py` — erfordert TWS oder IB Gateway auf dem PC.

```bash
pip install ib_insync
python -m ibkr.trader              # Dry-Run
python -m ibkr.trader --execute    # Paper
```
