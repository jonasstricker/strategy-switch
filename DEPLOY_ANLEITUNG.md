# Strategy-Switch App — GitHub Actions + Pages Deployment

Die App läuft komplett kostenlos auf GitHub:
- **GitHub Actions** berechnet täglich alle Signale (Mo–Fr 22:30 UTC)
- **GitHub Pages** hostet die PWA als statische Seite
- Kein Server, kein PC, keine Kreditkarte nötig
- 2.000 Minuten/Monat kostenlos (wir brauchen ~220 Min.)

---

## Schritt 1: GitHub Pages aktivieren

1. Gehe zu deinem Repo: **https://github.com/jonasstricker/strategy-switch**
2. Klicke auf **Settings** (Zahnrad-Tab oben)
3. In der linken Sidebar: **Pages**
4. Unter **Source**: Wähle **Deploy from a branch**
5. Unter **Branch**: Wähle **main** und Ordner **`/docs`**
6. Klicke **Save**
7. Warte 1–2 Minuten → Deine App ist erreichbar unter:
   **https://jonasstricker.github.io/strategy-switch/**

---

## Schritt 2: Erste Berechnung manuell starten

Die erste Berechnung muss manuell ausgelöst werden:

1. Gehe zu **https://github.com/jonasstricker/strategy-switch/actions**
2. Links klicke auf **Daily Strategy Calculation**
3. Rechts klicke **Run workflow** → **Run workflow** (grüner Button)
4. Warte ca. 10 Minuten bis der Job grün wird ✅
5. Danach ist die App unter der GitHub Pages URL mit Daten befüllt

---

## Schritt 3: App auf dem iPhone speichern

1. Öffne **https://jonasstricker.github.io/strategy-switch/** in **Safari**
2. Tippe auf das **Teilen-Symbol** (Quadrat mit Pfeil ↑)
3. Scrolle und tippe **Zum Home-Bildschirm**
4. Bestätige mit **Hinzufügen**
5. Fertig! Die App erscheint als permanentes Icon

---

## So funktioniert es

- **Automatisch**: GitHub Actions Cron-Job läuft Mo–Fr um 22:30 UTC
- **Berechnung**: ~10 Min. pro Lauf, Ergebnis wird als JSON ins Repo committed
- **Anzeige**: GitHub Pages serviert die statische HTML-Seite + JSON instant
- **Manuell**: Du kannst jederzeit unter Actions → Run workflow neu berechnen
- **Kosten**: $0 (Free Tier: 2.000 Min./Monat, wir brauchen ~220)

---

## Spätere Code-Updates

Auf deinem PC:
```powershell
cd "c:\Users\strickej\Product Spice model UI\Privat_test"
git add -A
git commit -m "Update"
git push
```

Die nächste automatische Berechnung nutzt dann den neuen Code.
Für sofortige Neuberechnung: Actions → Run workflow.

---

## Nützliche Links

- **App**: https://jonasstricker.github.io/strategy-switch/
- **Repo**: https://github.com/jonasstricker/strategy-switch
- **Actions**: https://github.com/jonasstricker/strategy-switch/actions
- **Pages Settings**: https://github.com/jonasstricker/strategy-switch/settings/pages
