"""
Standalone MT5 test trade script.
Places a 0.01 lot XAUUSD market order directly via the MT5 Python API
to verify the connection and trade execution works end-to-end.

This bypasses the bot's EXECUTION_ENABLED gate — it's purely for testing.
"""

import os
import sys
import time
from dotenv import load_dotenv

# Load .env from the project root
load_dotenv(os.path.join(os.path.dirname(__file__), "..", "trading_bot", ".env"))

import MetaTrader5 as mt5

# Credentials from .env
MT5_LOGIN = int(os.getenv("MT5_LOGIN", "0"))
MT5_PASSWORD = os.getenv("MT5_PASSWORD", "")
MT5_SERVER = os.getenv("MT5_SERVER", "ICMarkets-Demo")
SYMBOL = "XAUUSD"
LOT = 0.01

print("=" * 60)
print("MT5 TEST TRADE — Place a real XAUUSD position")
print(f"Account: {MT5_LOGIN} | Server: {MT5_SERVER}")
print("=" * 60)

# Step 1: Initialize MT5
print("\n[1] Initializing MT5...")
if not mt5.initialize(login=MT5_LOGIN, password=MT5_PASSWORD, server=MT5_SERVER):
    print(f"    FAILED: mt5.initialize() error = {mt5.last_error()}")
    sys.exit(1)
print("    OK — terminal initialized.")

# Step 2: Verify login
print(f"\n[2] Verifying login to account {MT5_LOGIN}...")
if not mt5.login(MT5_LOGIN, MT5_PASSWORD, MT5_SERVER):
    print(f"    FAILED: mt5.login() error = {mt5.last_error()}")
    mt5.shutdown()
    sys.exit(1)

account = mt5.account_info()
if account is None:
    print("    FAILED: Could not retrieve account info.")
    mt5.shutdown()
    sys.exit(1)
print(f"    OK — Balance: ${account.balance:.2f} | Equity: ${account.equity:.2f} | Leverage: 1:{account.leverage}")

# Step 3: Ensure symbol is visible
print(f"\n[3] Checking symbol {SYMBOL}...")
info = mt5.symbol_info(SYMBOL)
if info is None:
    print(f"    FAILED: Symbol {SYMBOL} not found.")
    mt5.shutdown()
    sys.exit(1)
if not info.visible:
    print("    Symbol not visible — selecting...")
    if not mt5.symbol_select(SYMBOL, True):
        print(f"    FAILED: Could not select {SYMBOL}.")
        mt5.shutdown()
        sys.exit(1)
print(f"    OK — {SYMBOL} available | Digits: {info.digits} | Min lot: {info.volume_min} | Max lot: {info.volume_max}")

# Check market is open
if info.session_deals == 0:
    print(f"    WARNING: Market for {SYMBOL} appears CLOSED (session_deals=0). Trade may fail.")
else:
    print("    Market is open (session_deals > 0).")

# Step 4: Get current price
print("\n[4] Fetching current tick data...")
tick = mt5.symbol_info_tick(SYMBOL)
if tick is None:
    print("    FAILED: Could not get tick data.")
    mt5.shutdown()
    sys.exit(1)
print(f"    Bid: {tick.bid} | Ask: {tick.ask} | Spread: {tick.ask - tick.bid:.{info.digits}f}")

# Step 5: Place a BUY order
print(f"\n[5] Placing BUY {LOT} {SYMBOL}...")

# Compute SL/TP: ~50 points away (~$5 risk at 0.01 lot for XAUUSD)
# XAUUSD point value: 1 point = $0.01 per 0.01 lot, so 500 points = $5
sl_distance = 5.0   # ~500 points in price
tp_distance = 5.0   # ~500 points in price

price = tick.ask
sl = round(price - sl_distance, info.digits)
tp = round(price + tp_distance, info.digits)

request = {
    "action": mt5.TRADE_ACTION_DEAL,
    "symbol": SYMBOL,
    "volume": LOT,
    "type": mt5.ORDER_TYPE_BUY,
    "price": price,
    "sl": sl,
    "tp": tp,
    "deviation": 20,
    "magic": 999888,
    "comment": "TEST_TRADE",
    "type_time": mt5.ORDER_TIME_GTC,
    "type_filling": mt5.ORDER_FILLING_IOC,
}

result = mt5.order_send(request)
print(f"    Order result struct: {result}")
if result is None:
    print(f"    FAILED: mt5.order_send() returned None. Last error: {mt5.last_error()}")
    mt5.shutdown()
    sys.exit(1)

print(f"    Retcode: {result.retcode}")

if result.retcode == mt5.TRADE_RETCODE_DONE:
    print(f"    SUCCESS! Order ticket: {result.order}")
    print(f"    Entry: {price} | SL: {sl} | TP: {tp}")
    print(f"\n    The position should now be visible in your MT5 terminal!")
    print(f"    Ticket #{result.order} — check the 'Trade' tab in MT5.")
else:
    retcode_meanings = {
        10004: "TRADE_RETCODE_REQUOTE — price changed, retry",
        10006: "TRADE_RETCODE_REJECT — request rejected",
        10007: "TRADE_RETCODE_CANCEL — canceled by trader",
        10008: "TRADE_RETCODE_PLACED — order placed (pending)",
        10009: "TRADE_RETCODE_DONE — done ✓",
        10010: "TRADE_RETCODE_DONE_PARTIAL — partial fill",
        10011: "TRADE_RETCODE_ERROR — processing error",
        10012: "TRADE_RETCODE_TIMEOUT — timeout",
        10013: "TRADE_RETCODE_INVALID — invalid request",
        10014: "TRADE_RETCODE_INVALID_VOLUME — bad volume",
        10015: "TRADE_RETCODE_INVALID_PRICE — bad price",
        10016: "TRADE_RETCODE_INVALID_STOPS — bad stops",
        10017: "TRADE_RETCODE_TRADE_DISABLED — trading disabled",
        10018: "TRADE_RETCODE_MARKET_CLOSED — market closed",
        10019: "TRADE_RETCODE_NO_MONEY — insufficient funds",
        10020: "TRADE_RETCODE_PRICE_CHANGED — price moved",
        10021: "TRADE_RETCODE_PRICE_OFF — no quotes",
        10022: "TRADE_RETCODE_INVALID_EXPIRATION — bad expiration",
        10023: "TRADE_RETCODE_ORDER_CHANGED — order changed",
        10024: "TRADE_RETCODE_TOO_MANY_REQUESTS — rate limited",
        10025: "TRADE_RETCODE_NO_CHANGES — no changes in modify",
        10026: "TRADE_RETCODE_SERVER_DISABLES_AT — auto-trading off on server",
        10027: "TRADE_RETCODE_CLIENT_DISABLES_AT — auto-trading off on client",
        10028: "TRADE_RETCODE_LOCKED — request locked for processing",
        10029: "TRADE_RETCODE_FROZEN — order/modify frozen",
        10030: "TRADE_RETCODE_INVALID_FILL — bad fill type",
        10031: "TRADE_RETCODE_CONNECTION — no connection",
        10032: "TRADE_RETCODE_ONLY_REAL — only real accounts allowed",
        10033: "TRADE_RETCODE_LIMIT_ORDERS — max pending orders reached",
        10034: "TRADE_RETCODE_LIMIT_VOLUME — max volume reached",
        10035: "TRADE_RETCODE_INVALID_ORDER — invalid order type",
        10036: "TRADE_RETCODE_POSITION_CLOSED — position already closed",
    }
    meaning = retcode_meanings.get(result.retcode, f"Unknown retcode ({result.retcode})")
    print(f"    FAILED: {meaning}")
    print(f"    Full result comment: {result.comment if hasattr(result, 'comment') else 'N/A'}")

    if result.retcode == 10026 or result.retcode == 10027:
        print("\n    *** AUTO-TRADING IS DISABLED! ***")
        print("    In MT5, go to Tools → Options → Expert Advisors →")
        print("    Check 'Allow automated trading' and 'Allow Algo Trading'")

# Step 6: Show open positions
print("\n[6] Current open XAUUSD positions:")
positions = mt5.positions_get(symbol=SYMBOL)
if positions is None or len(positions) == 0:
    print("    No open XAUUSD positions.")
else:
    for p in positions:
        pnl = round(p.profit, 2)
        print(f"    Ticket #{p.ticket} | {p.type_str} | Lot: {p.volume} | "
              f"Entry: {p.price_open} | Current: {p.price_current} | P/L: ${pnl}")
        print(f"      SL: {p.sl} | TP: {p.tp} | Comment: {p.comment}")

print("\n[7] Shutting down MT5 connection...")
mt5.shutdown()
print("    Done.")
print("\n" + "=" * 60)
print("TEST COMPLETE")
print("=" * 60)