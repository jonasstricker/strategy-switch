# Strategy-Switch App — Cloud Deployment Anleitung

Die App läuft komplett in der Cloud (Render.com). Kein PC nötig.
Signale werden automatisch neu berechnet wenn du die App öffnest
und die Daten älter als 20 Stunden sind.

---

## Schritt 1: GitHub-Konto erstellen (falls noch keins vorhanden)

1. Gehe zu **https://github.com/signup**
2. Registriere dich mit deiner E-Mail (z.B. jonasstricker89@gmail.com)
3. Bestätige die E-Mail

---

## Schritt 2: Neues Repository auf GitHub erstellen

1. Gehe zu **https://github.com/new**
2. Repository name: `strategy-switch`
3. Visibility: **Private** (wichtig! Dein Code bleibt privat)
4. NICHT "Add a README file" ankreuzen
5. Klicke **Create repository**
6. Du siehst jetzt eine Seite mit Befehlen — kopiere die HTTPS-URL:
   `https://github.com/DEIN-USERNAME/strategy-switch.git`

---

## Schritt 3: Code hochladen (einmalig)

Öffne ein Terminal (PowerShell) im Projektordner und führe aus:

```powershell
cd "c:\Users\strickej\Product Spice model UI\Privat_test"
git remote add origin https://github.com/jonasstricker/strategy-switch.git
git branch -M main
git push -u origin main
```

> Bei der ersten Nutzung fragt Git nach deinem GitHub-Benutzernamen
> und Passwort. Als "Passwort" brauchst du ein **Personal Access Token**:
> GitHub → Settings → Developer settings → Personal access tokens → 
> Tokens (classic) → Generate new token → Haken bei "repo" → Generate.
> Dieses Token ist dein Passwort.

---

## Schritt 4: Render.com Account erstellen

1. Gehe zu **https://render.com**
2. Klicke **Get Started for Free**
3. Wähle **Sign in with GitHub** (nutzt deinen GitHub-Account)
4. Erlaube Render.com Zugriff auf dein GitHub

---

## Schritt 5: Web Service erstellen auf Render.com

1. Klicke **New** → **Web Service**
2. Wähle **Build and deploy from a Git repository** → Next
3. Verbinde dein `strategy-switch` Repository
4. Render erkennt automatisch die `render.yaml` — alles ist vorkonfiguriert:
   - Name: `strategy-switch`
   - Runtime: Python 3.13
   - Build Command: `pip install -r requirements.txt`
   - Start Command: `gunicorn mobile_app.server:app --bind 0.0.0.0:$PORT --timeout 900 --workers 1 --threads 2`
   - Plan: **Free**
5. Klicke **Create Web Service**
6. Warte bis der Build fertig ist (ca. 2-5 Minuten)

---

## Schritt 6: App-URL auf dem iPhone speichern

Nach dem Deployment bekommst du eine permanente URL:
**https://strategy-switch.onrender.com** (oder ähnlich)

So speicherst du sie als App auf dem iPhone:
1. Öffne die URL in **Safari** auf dem iPhone
2. Tippe auf das **Teilen-Symbol** (Quadrat mit Pfeil ↑)
3. Scrolle und tippe **Zum Home-Bildschirm**
4. Bestätige mit **Hinzufügen**
5. Fertig! Die App erscheint als Icon auf deinem Home-Bildschirm

---

## So funktioniert die Cloud-App

- **Beim ersten Öffnen**: Falls keine Daten vorhanden oder Daten > 20h alt
  → Berechnung startet automatisch im Hintergrund (2-5 Min.)
- **Danach**: Sofortige Anzeige der gespeicherten Signale
- **Manuell neu berechnen**: Button "🔄 Neu berechnen" in der App
- **Render Free Tier**: Die App "schläft" nach 15 Min. Inaktivität.
  Beim nächsten Öffnen ~30 Sek. Aufwachzeit, dann läuft sie normal.

---

## Spätere Updates

Wenn sich der Code ändert, einfach:
```powershell
cd "c:\Users\strickej\Product Spice model UI\Privat_test"
git add -A
git commit -m "Update"
git push
```
Render.com deployed automatisch bei jedem Push.
