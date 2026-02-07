#!/bin/bash
# Script zum Starten des Web-Interfaces

cd ~/voxtral

# Virtuelle Umgebung aktivieren
source venv/bin/activate

echo "Starte Gradio Web-Interface auf Port 7634..."
echo "Erreichbar via SSH-Tunnel über http://localhost:7634"

# Startet dein Python-Script
python app.py
