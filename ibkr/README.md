# IBKR Auto-Trading Setup

## 1. IBKR Paper-Konto erstellen (kostenlos)

1. **[interactivebrokers.com](https://www.interactivebrokers.com)** → Konto eröffnen
2. **Paper Trading** aktivieren (unter Kontoeinstellungen)
3. **TWS (Trader Workstation)** herunterladen und installieren

## 2. TWS konfigurieren

1. TWS starten und mit **Paper-Konto** einloggen
2. **Edit → Global Configuration → API → Settings:**
   - ✅ Enable ActiveX and Socket Clients
   - ✅ Read-Only API = **NEIN** (deaktivieren für echte Orders)
   - Socket Port: **7497** (Paper) / **7496** (Live)
   - Trusted IPs: 127.0.0.1

## 3. Python-Bibliothek installieren

```bash
pip install ib_insync
```

## 4. Trader starten

```bash
# Dry-Run (zeigt was passieren würde, keine echten Orders)
python -m ibkr.trader

# Paper-Trading (echte Orders, aber Spielgeld)
python -m ibkr.trader --execute

# LIVE (echtes Geld — nur wenn Paper-Test erfolgreich!)
python -m ibkr.trader --execute --live
```

## 5. Wie es funktioniert

```
GitHub Actions (22:30 UTC)
    ↓
Berechnet Signale → mobile_signals.json
    ↓
Du siehst Signale auf dem Handy (PWA)
    ↓
Auf dem PC: python -m ibkr.trader
    ↓
Liest Signale, vergleicht mit IBKR-Portfolio
    ↓
Zeigt/platziert nötige Orders
```

## Portfolio-Aufteilung

| Kategorie   | Gewicht | Beschreibung                    |
|------------|---------|----------------------------------|
| SXR8.DE    | 15%     | S&P 500 Europa                  |
| URTH       | 15%     | MSCI World                      |
| EEM        | 15%     | Emerging Markets                |
| VGK        | 15%     | Europa                          |
| Schwarm    | 18%     | Top-10 Mega-Caps                |
| Value      | 12%     | Top-10 unterbewertete Aktien    |
| Turnaround | 10%     | Gefallene Aktien mit Potenzial  |

## Sicherheit

- **Paper-Trading zuerst** — kein Risiko mit echtem Geld
- **Dry-Run Standard** — ohne `--execute` passiert nichts
- **Doppelte Bestätigung** — bei `--live --execute` muss "ja" getippt werden
- **Lokal auf deinem PC** — TWS muss laufen, keine Cloud-Credentials nötig
- **Keine Credentials im Code** — IBKR-Login nur in TWS selbst
