#!/bin/bash
"""
AI Trading Bot — VPS Deployment Script (HARDENED)
Oracle Cloud Free Tier Ubuntu 22.04

Usage:
  chmod +x scripts/setup_vps.sh
  sudo ./scripts/setup_vps.sh
"""

set -e

echo "=========================================="
echo "AI Trading Bot — VPS Setup (HARDENED)"
echo "Oracle Cloud Free Tier Ubuntu 22.04"
echo "=========================================="

echo "[1/8] Installing system packages..."
sudo apt update && sudo apt upgrade -y
sudo apt install -y python3 python3-pip python3-venv git curl wget \
    wine64 wine32 winehq-stable xvfb xdotool supervisor

echo "[2/8] Installing MetaTrader 5 via Wine..."
MT5_DIR="$HOME/.wine/drive_c/Program Files/MetaTrader 5"
if [ ! -d "$MT5_DIR" ]; then
    export WINEDLLOVERRIDES="winemenubuilder.exe=d"
    export DISPLAY=:0
    wineboot -u 2>/dev/null || true
    wget -q -O /tmp/mt5setup.exe "https://download.mql5.com/cdn/web/metaquotes.software.corp/mt5/mt5setup.exe"
    echo "  Installing MT5 (this may take several minutes)..."
    xvfb-run wine /tmp/mt5setup.exe /auto 2>&1 | tail -5 || true
    sleep 30
    echo "  MT5 installation complete."
else
    echo "  MT5 already installed."
fi

echo "[3/8] Setting up project..."
PROJECT_DIR="$HOME/trading_bot"
if [ ! -d "$PROJECT_DIR" ]; then
    echo "  Upload project: scp -r trading_bot ubuntu@<VPS_IP>:~/"
    mkdir -p "$PROJECT_DIR"
fi
cd "$PROJECT_DIR"

echo "[4/8] Python virtual environment..."
if [ ! -d "venv" ]; then python3 -m venv venv; fi
source venv/bin/activate

echo "[5/8] Installing Python dependencies..."
pip install --upgrade pip
pip install -r trading_bot/requirements.txt 2>&1 | tail -3

echo "[6/8] Configuring .env..."
if [ ! -f "trading_bot/.env" ]; then
    cp trading_bot/.env.example trading_bot/.env
    echo "  EDIT .env: nano trading_bot/.env"
fi
chmod 600 trading_bot/.env

echo "[7/8] Creating systemd service (HARDENED)..."
SERVICE_FILE="/etc/systemd/system/trading-bot.service"
sudo tee "$SERVICE_FILE" > /dev/null << 'SERVICEEOF'
[Unit]
Description=AI Trading Bot (HARDENED)
After=network.target

[Service]
Type=simple
User=%u
WorkingDirectory=%h/trading_bot
Environment=DISPLAY=:0
Environment=WINEDLLOVERRIDES=winemenubuilder.exe=d
ExecStartPre=/bin/sleep 15
ExecStart=%h/trading_bot/venv/bin/python3 %h/trading_bot/trading_bot/main.py
Restart=always
RestartSec=10
StartLimitInterval=60
StartLimitBurst=5
StandardOutput=append:%h/trading_bot/bot.log
StandardError=append:%h/trading_bot/bot.log
TimeoutStopSec=30

[Install]
WantedBy=default.target
SERVICEEOF

sudo sed -i "s|%h|$HOME|g" "$SERVICE_FILE"
sudo sed -i "s|%u|$USER|g" "$SERVICE_FILE"
sudo systemctl daemon-reload

echo "[8/8] Setup complete."
echo ""
echo "=========================================="
echo "  SETUP COMPLETE — HARDENED"
echo "=========================================="
echo ""
echo "  EDIT CREDENTIALS:"
echo "    nano $PROJECT_DIR/trading_bot/.env"
echo ""
echo "  TEST:"
echo "    cd $PROJECT_DIR && source venv/bin/activate"
echo "    python3 trading_bot/main.py"
echo ""
echo "  START 24/7 (auto-recovery):"
echo "    sudo systemctl enable trading-bot"
echo "    sudo systemctl start trading-bot"
echo "    sudo systemctl status trading-bot"
echo ""
echo "  MONITOR:"
echo "    tail -f $PROJECT_DIR/bot.log"
echo "    tail -f $PROJECT_DIR/trading_bot/logs/trading_bot.log"
echo "    tail -f $PROJECT_DIR/trading_bot/logs/heartbeat.log"
echo ""
echo "  SAFETY FEATURES ENABLED:"
echo "    - MT5 Health Monitor (auto-reconnect)"
echo "    - Watchdog (heartbeat, max 5 restarts/hour)"
echo "    - Safe Execution Gate (4 checks before trade)"
echo "    - Degraded mode (AI/News failure = safe fallback)"
echo "    - Crash-proof outer loop"
echo "    - Log rotation (10MB per file)"
echo "    - Heartbeat log (critical events only)"
echo "=========================================="