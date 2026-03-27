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
