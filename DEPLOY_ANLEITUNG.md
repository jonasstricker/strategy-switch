# Strategy-Switch App — Oracle Cloud Deployment

Die App läuft auf einer kostenlosen Oracle Cloud VM (4 CPU-Kerne, 24 GB RAM).
Always-on, kein PC nötig. Volle Rechenpower, keine Abstriche bei den Strategien.
Tägliche automatische Neuberechnung um 22:30 UTC (Mo-Fr).

---

## Schritt 1: GitHub-Repo erstellen + Code hochladen

### 1a. GitHub-Konto erstellen (falls nötig)
1. Gehe zu **https://github.com/signup**
2. Registriere dich mit deiner E-Mail
3. Bestätige die E-Mail

### 1b. Repository erstellen
1. Gehe zu **https://github.com/new**
2. Name: `strategy-switch`
3. Visibility: **Private**
4. NICHT "Add a README file" ankreuzen
5. Klicke **Create repository**

### 1c. Code hochladen (PowerShell im Projektordner)
```powershell
cd "c:\Users\strickej\Product Spice model UI\Privat_test"
git remote add origin https://github.com/DEIN-USERNAME/strategy-switch.git
git branch -M main
git push -u origin main
```

> **Passwort**: GitHub → Settings → Developer settings → Personal access tokens →
> Tokens (classic) → Generate new token → Haken bei "repo" → Generate.
> Dieses Token als Passwort eingeben.

---

## Schritt 2: Oracle Cloud Account erstellen

1. Gehe zu **https://cloud.oracle.com/free**
2. Klicke **Start for Free**
3. Registriere dich (E-Mail, Name, Land: Germany)
4. **Kreditkarte wird NICHT belastet** — nur zur Verifizierung
5. Wähle Home Region: **Germany Central (Frankfurt)**
6. Warte auf Bestätigungs-E-Mail (kann bis zu 30 Min. dauern)

---

## Schritt 3: VM erstellen (kostenlos, always-on)

1. Anmelden bei **https://cloud.oracle.com**
2. Klicke **Create a VM instance** (oder: Hamburger-Menü → Compute → Instances → Create Instance)
3. Konfiguration:
   - **Name**: `strategy-switch`
   - **Image**: **Ubuntu 22.04** (Canonical Ubuntu)
   - **Shape**: Klicke **Change shape** → **Ampere** → **VM.Standard.A1.Flex**
     - OCPUs: **4**
     - Memory: **24 GB**
     - (Das ist komplett kostenlos im Always-Free-Tier!)
   - **Networking**: Default (neues VCN erstellen lassen)
   - **SSH Key**: Klicke **Generate a key pair** → **Save Private Key** herunterladen
     - Speichere die `.key`-Datei sicher ab (z.B. `oracle-key.key`)
4. Klicke **Create**
5. Warte bis Status = **RUNNING** (1-2 Min.)
6. Notiere die **Public IP Address** (z.B. `129.159.xxx.xxx`)

---

## Schritt 4: Firewall öffnen (Port 80)

1. Im Oracle Dashboard: Klicke auf deine VM → **Subnet** → **Security List**
2. Klicke **Add Ingress Rules**:
   - Source CIDR: `0.0.0.0/0`
   - Destination Port: `80`
   - Description: `HTTP`
3. Klicke **Add Ingress Rules**

---

## Schritt 5: SSH-Verbindung + Setup

### 5a. SSH-Verbindung herstellen (PowerShell)
```powershell
# Pfad zu deinem heruntergeladenen SSH-Key anpassen!
ssh -i C:\Users\strickej\Downloads\oracle-key.key ubuntu@DEINE_IP
```

> Beim ersten Mal: `yes` eingeben wenn nach fingerprint gefragt wird.
> Falls "Permission denied": Rechtsklick auf die .key-Datei → Eigenschaften →
> Sicherheit → Erweitert → Vererbung deaktivieren → Nur eigenen Benutzer behalten.

### 5b. Projekt klonen + Setup ausführen
```bash
# Auf der VM (nach SSH-Login):
git clone https://github.com/DEIN-USERNAME/strategy-switch.git
cd strategy-switch

# Automatisches Setup (installiert alles, startet Service + Cron)
bash cloud/setup.sh
```

Das Script:
- Installiert Python, Nginx
- Erstellt Python-Umgebung + Abhängigkeiten
- Führt erste Berechnung aus
- Startet den Web-Service (Gunicorn, always-on)
- Konfiguriert Nginx (Port 80)
- Richtet Cron-Job ein (Mo-Fr 22:30 UTC)

### 5c. Ubuntu-Firewall öffnen
```bash
sudo iptables -I INPUT 6 -m state --state NEW -p tcp --dport 80 -j ACCEPT
sudo netfilter-persistent save
```

---

## Schritt 6: App auf dem iPhone speichern

Deine App ist jetzt erreichbar unter:
**http://DEINE_IP** (z.B. http://129.159.123.45)

So speicherst du sie als App:
1. Öffne die URL in **Safari** auf dem iPhone
2. Tippe auf das **Teilen-Symbol** (Quadrat mit Pfeil ↑)
3. Scrolle und tippe **Zum Home-Bildschirm**
4. Bestätige mit **Hinzufügen**
5. Fertig! Die App erscheint als permanentes Icon

---

## So funktioniert die Cloud-App

- **Always-On**: Die VM läuft 24/7, kein "Aufwachen" nötig
- **Tägliche Berechnung**: Cron-Job Mo-Fr 22:30 UTC (automatisch)
- **Sofortige Anzeige**: Daten werden vorberechnet, App lädt instant
- **Manuelle Berechnung**: Button "Neu berechnen" in der App
- **Volle Power**: 4! Kerne + 24 GB RAM → alle Strategien, volle Grids

---

## Spätere Code-Updates

Auf deinem PC:
```powershell
cd "c:\Users\strickej\Product Spice model UI\Privat_test"
git add -A
git commit -m "Update"
git push
```

Dann auf der VM (SSH):
```bash
cd ~/strategy-switch
git pull
sudo systemctl restart strategy-switch
```

---

## Nützliche Befehle (SSH auf der VM)

```bash
# Service-Status prüfen
sudo systemctl status strategy-switch

# Live-Logs ansehen
sudo journalctl -u strategy-switch -f

# Manuelle Neuberechnung
cd ~/strategy-switch && .venv/bin/python -m wf_backtest.daily_runner

# Service neustarten
sudo systemctl restart strategy-switch

# Cron-Log ansehen
tail -50 ~/strategy-switch/cron.log
```
