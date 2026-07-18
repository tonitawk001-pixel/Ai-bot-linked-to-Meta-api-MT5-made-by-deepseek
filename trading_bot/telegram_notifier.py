"""
Telegram Notifier — shared module for both Contabo and MT5 bots.
"""

import requests
import time
from datetime import datetime, timezone

TOKEN = "8576199875:AAHYRdna3TxQwHP50cpd9isW128BzBuvFKM"
CHAT_ID = 5233262246
ACCOUNT_NAME = "Default"
_last_hb = 0
_HB_INTERVAL = 300  # 5 minutes

def send_msg(text: str):
    if not CHAT_ID: return
    try:
        requests.post(f"https://api.telegram.org/bot{TOKEN}/sendMessage",
            json={"chat_id": CHAT_ID, "text": text, "parse_mode": "HTML"}, timeout=10)
    except: pass

def set_name(name: str):
    global ACCOUNT_NAME
    ACCOUNT_NAME = name

def startup(balance=None):
    m = f"🤖 <b>{ACCOUNT_NAME}</b>\n🟢 <b>BOT STARTED</b>\n"
    if balance is not None: m += f"💰 <b>Balance:</b> ${balance:.2f}\n"
    m += f"⏰ {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S')} UTC"
    send_msg(m)

def shutdown():
    m = f"🤖 <b>{ACCOUNT_NAME}</b>\n🔴 <b>BOT STOPPED</b>\n"
    m += f"⏰ {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S')} UTC"
    send_msg(m)

def trade_opened(direction, symbol, price, lot, sl, tp, score, balance):
    m = (f"🤖 <b>{ACCOUNT_NAME}</b>\n📈 <b>TRADE OPENED</b>\n"
         f"━━━━━━━━━━━━━━━\n<b>Direction:</b> {direction}\n<b>Symbol:</b> {symbol}\n"
         f"<b>Price:</b> ${price:.2f}\n<b>Lot:</b> {lot}\n<b>SL:</b> ${sl:.2f}\n"
         f"<b>TP:</b> ${tp:.2f}\n<b>Score:</b> {score}\n<b>Balance:</b> ${balance:.2f}\n"
         f"⏰ {datetime.now(timezone.utc).strftime('%H:%M:%S')} UTC")
    send_msg(m)

def trade_closed(direction, symbol, entry, exit_price, pnl, reason, balance):
    em = "✅" if pnl > 0 else "❌"
    m = (f"🤖 <b>{ACCOUNT_NAME}</b>\n{em} <b>TRADE CLOSED</b>\n"
         f"━━━━━━━━━━━━━━━\n<b>Direction:</b> {direction}\n<b>Symbol:</b> {symbol}\n"
         f"<b>Entry:</b> ${entry:.2f}\n<b>Exit:</b> ${exit_price:.2f}\n"
         f"<b>P&L:</b> ${pnl:+.2f}\n<b>Reason:</b> {reason}\n<b>Balance:</b> ${balance:.2f}\n"
         f"⏰ {datetime.now(timezone.utc).strftime('%H:%M:%S')} UTC")
    send_msg(m)

def error(msg):
    m = (f"🤖 <b>{ACCOUNT_NAME}</b>\n⚠️ <b>ERROR</b>\n━━━━━━━━━━━━━━━\n{msg[:200]}\n"
         f"⏰ {datetime.now(timezone.utc).strftime('%H:%M:%S')} UTC")
    send_msg(m)

def heartbeat(balance, positions, trades):
    global _last_hb
    now = time.time()
    if now - _last_hb < _HB_INTERVAL: return
    _last_hb = now
    m = (f"🤖 <b>{ACCOUNT_NAME}</b>\n💚 <b>BOT ALIVE</b>\n"
         f"━━━━━━━━━━━━━━━\n<b>Balance:</b> ${balance:.2f}\n"
         f"<b>Open Positions:</b> {positions}\n<b>Total Trades:</b> {trades}\n"
         f"⏰ {datetime.now(timezone.utc).strftime('%H:%M:%S')} UTC")
    send_msg(m)