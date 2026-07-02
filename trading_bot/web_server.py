"""
V22 Dashboard — Flask Web Server
=================================
Reads state.json (written by main_v22.py) and serves:
  - Account balance / equity
  - Daily P/L
  - Open positions
  - 24-hour trade history
  - Pause/Resume toggle

Zero external API cost — all self-hosted.
"""

import json, os, time
from datetime import datetime, timezone, timedelta
from flask import Flask, render_template, jsonify, request

app = Flask(__name__)
STATE_FILE = os.path.join(os.path.dirname(__file__), "..", "logs", "bot_state.json")
PAUSE_FILE = os.path.join(os.path.dirname(__file__), "..", "logs", "paused.flag")


def read_state():
    """Read current bot state."""
    try:
        with open(STATE_FILE, "r") as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return {"balance": 0, "equity": 0, "daily_pnl": 0,
                "positions": [], "trades": [], "status": "starting",
                "cycle": 0, "consec_losses": 0}


def get_paused():
    """Check if bot is paused."""
    return os.path.exists(PAUSE_FILE)


@app.route("/")
def dashboard():
    """Serve the dashboard HTML."""
    return render_template("dashboard.html")


@app.route("/api/state")
def api_state():
    """API: Return full bot state as JSON."""
    state = read_state()
    state["paused"] = get_paused()
    return jsonify(state)


@app.route("/api/pause", methods=["POST"])
def api_pause():
    """API: Toggle pause/resume."""
    if get_paused():
        os.remove(PAUSE_FILE)
        return jsonify({"paused": False, "message": "Bot resumed"})
    else:
        with open(PAUSE_FILE, "w") as f:
            f.write("paused")
        return jsonify({"paused": True, "message": "Bot paused"})


@app.route("/api/trades")
def api_trades():
    """API: Return 24-hour trade history."""
    state = read_state()
    now = datetime.now(timezone.utc)
    cutoff = (now - timedelta(hours=24)).isoformat()
    recent = [t for t in state.get("trades", [])
              if t.get("close_time", "") >= cutoff]
    return jsonify(recent)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)