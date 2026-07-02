"""
V22 LIVE TRADER — Gold Scalping with V22 Engine + Web Dashboard
=================================================================
Exact same logic as the V22 backtest that delivered:
  - $300 → $70k+ across 3 windows (18 months out-of-sample)
  - Graduated risk: 0.5%/1.5%/2.5%/3%
  - Trailing SL, 3-loss halt, BUY EMA200 filter, 5 max positions

Writes state.json for web dashboard. Checks paused.flag for pause/resume.
"""

import sys, os, time, json
from datetime import datetime, timedelta, timezone
from collections import defaultdict

import pandas as pd
import numpy as np
import MetaTrader5 as mt5

from trading_bot.config import Config
from trading_bot.utils.logger import logger
from trading_bot.mt5.data_feed import get_candles, TIMEFRAME_MAP
from trading_bot.indicators.technical_indicators import compute_all_indicators
from trading_bot.strategy.gold_scalping_strategy import GoldScalpingStrategy
from trading_bot.strategy.gold_volatility_filter import GoldVolatilityFilter
from trading_bot.execution.mt5_executor import execute_trade

# === V22 CONFIG ===
SYMBOL = "XAUUSD"
MIN_SCORE = 50
MAX_POSITIONS = 5
HALT_AFTER_LOSSES = 3
HALT_HOURS = 2
TRAIL_SL = True

# File paths for web dashboard
STATE_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "logs", "bot_state.json")
PAUSE_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "logs", "paused.flag")

# V22 State (persists across cycles)
consecutive_losses = 0
halt_until = None
daily_pnl = 0.0
paper_positions = []
last_entry_time = None
ENTRY_COOLDOWN_MINUTES = 15
all_trades_history = []  # Full trade log for dashboard

def write_state(balance, equity, positions, status, cycle):
    """Export current state to JSON for web dashboard."""
    try:
        os.makedirs(os.path.dirname(STATE_FILE), exist_ok=True)
        pos_data = [{"direction": p["direction"], "entry": p["entry"],
                      "sl": p["sl"], "tp": p["tp"], "lot": p["lot"],
                      "be": p.get("be", False)} for p in positions]
        state = {
            "balance": round(balance, 2),
            "equity": round(equity, 2),
            "daily_pnl": round(daily_pnl, 2),
            "positions": pos_data,
            "trades": all_trades_history[-200:],  # Last 200 trades
            "status": status,
            "cycle": cycle,
            "consec_losses": consecutive_losses,
            "updated": datetime.now(timezone.utc).isoformat()
        }
        with open(STATE_FILE, "w") as f:
            json.dump(state, f, indent=2)
    except Exception:
        pass

def is_paused():
    return os.path.exists(PAUSE_FILE)

def get_risk_percent(balance):
    if balance < 250: return 0.5
    elif balance < 500: return 1.5
    elif balance < 1000: return 2.5
    return 3.0

def get_daily_loss_limit(balance):
    return -balance * 0.05

def can_trade(now_dt, balance):
    global halt_until, daily_pnl
    if halt_until and now_dt < halt_until: return False
    if daily_pnl <= get_daily_loss_limit(balance): return False
    return True

def record_sl():
    global consecutive_losses, halt_until
    consecutive_losses += 1
    if consecutive_losses >= HALT_AFTER_LOSSES:
        halt_until = datetime.now(timezone.utc) + timedelta(hours=HALT_HOURS)
        logger.warning(f"V22 HALT: {HALT_AFTER_LOSSES} consecutive losses — pausing {HALT_HOURS}h")

def record_win():
    global consecutive_losses
    consecutive_losses = 0

def reset_daily():
    global daily_pnl
    daily_pnl = 0.0

def update_paper_positions(current_price, atr_val):
    global paper_positions, daily_pnl
    remaining = []
    for pos in paper_positions:
        e, d, sl, tp, lot = pos["entry"], pos["direction"], pos["sl"], pos["tp"], pos["lot"]
        pv = lot * 100
        # Trailing SL
        if TRAIL_SL and not pos.get("be", False):
            pm = current_price - e if d == "BUY" else e - current_price
            if pm >= atr_val:
                pos["be"] = True; pos["sl"] = e
        elif TRAIL_SL and pos.get("be", False):
            if d == "BUY":
                ns = current_price - atr_val * 0.7
                if ns > sl + 0.5: pos["sl"] = round(ns, 2)
            else:
                ns = current_price + atr_val * 0.7
                if ns < sl - 0.5: pos["sl"] = round(ns, 2)
        sl, tp = pos["sl"], pos["tp"]
        hit = False; pnl = 0.0; reason = ""
        if d == "BUY":
            if tp and current_price >= tp: pnl = (tp - e) * pv; reason = "TP"; hit = True
            elif sl and current_price <= sl: pnl = (sl - e) * pv; reason = "SL" if sl <= e else "TRAIL"; hit = True
        else:
            if tp and current_price <= tp: pnl = (e - tp) * pv; reason = "TP"; hit = True
            elif sl and current_price >= sl: pnl = (e - sl) * pv; reason = "SL" if sl >= e else "TRAIL"; hit = True
        if hit:
            daily_pnl += pnl
            trade_entry = {**pos, "pnl": round(pnl, 2), "reason": reason,
                           "close_time": datetime.now(timezone.utc).isoformat()}
            all_trades_history.append(trade_entry)
            logger.info(f"  V22 PAPER CLOSE: {d} {lot} {SYMBOL} P/L=${pnl:+.2f} ({reason})")
            if reason == "SL": record_sl()
            else: record_win()
        else:
            remaining.append(pos)
    paper_positions = remaining

def fetch_live_data() -> dict:
    data = {}
    for tf in ["M1", "M5", "M15"]:
        df = get_candles(symbol=SYMBOL, timeframe=tf, count=500)
        if df is None or df.empty:
            if tf == "M1":
                data["M1"] = get_candles(symbol=SYMBOL, timeframe="M5", count=500)
            else:
                return None
        else:
            data[tf] = df
    return data

def v22_cycle():
    global paper_positions
    data = fetch_live_data()
    if data is None: return
    m1_df, m5_df, m15_df = data.get("M1"), data.get("M5"), data.get("M15")
    if m1_df is None or m5_df is None or m15_df is None: return
    if m1_df.empty or m5_df.empty or m15_df.empty: return
    account = mt5.account_info()
    if account is None: return
    balance, equity = account.balance, account.equity
    try:
        m1_ind = compute_all_indicators(m1_df)
        m5_ind = compute_all_indicators(m5_df)
        m15_ind = compute_all_indicators(m15_df)
    except: return
    try: atr_val = float(m5_ind["atr"].iloc[-1]) if not m5_ind.get("atr", pd.Series()).empty else 3.5
    except: atr_val = 3.5
    current_price = float(m15_df["close"].iloc[-1])
    now_dt = datetime.now(timezone.utc)
    update_paper_positions(current_price, atr_val)
    if not can_trade(now_dt, balance): return
    if len(paper_positions) >= MAX_POSITIONS: return

    # Entry cooldown
    global last_entry_time
    if last_entry_time is not None:
        if (now_dt - last_entry_time).total_seconds() / 60 < ENTRY_COOLDOWN_MINUTES: return

    # Pause check
    if is_paused():
        return

    strategy = GoldScalpingStrategy()
    strategy._max_trades_per_day = 200; strategy._max_open_positions = 10
    try:
        result = strategy.analyze(m1_indicators=m1_ind, m5_indicators=m5_ind, m15_indicators=m15_ind,
                                   m1_ohlcv=m1_df, m5_ohlcv=m5_df, m15_ohlcv=m15_df, news_context=None)
    except: return
    direction = result.get("direction", "NONE")
    score = result.get("setup_score", 0)
    if direction == "NONE" or score < MIN_SCORE: return

    if direction == "BUY":
        c = m15_df["close"].values
        if len(c) >= 200:
            e200 = pd.Series(c).ewm(200, adjust=False).mean().values
            if len(e200) >= 10 and float(e200[-1]) <= float(e200[-10]): return

    vol_filter = GoldVolatilityFilter()
    try:
        vo = vol_filter.analyze(m1_ohlcv=m1_df, m5_ohlcv=m5_df, m15_ohlcv=m15_df,
                                 m1_indicators=m1_ind, m5_indicators=m5_ind, m15_indicators=m15_ind)
        if not vo.get("trade_ok", False): return
    except: return

    sd, td = atr_val * 1.5, atr_val * 3.0
    sl = round(current_price - sd, 2) if direction == "BUY" else round(current_price + sd, 2)
    tp = round(current_price + td, 2) if direction == "BUY" else round(current_price - td, 2)
    rp = get_risk_percent(balance)
    lot = max(0.01, min(balance * (rp/100) / (sd * 100) if sd else 0.01, 50.0))
    lot = round(lot, 2)
    logger.info(f"V22 SIGNAL: {direction} score={score} lot={lot} SL={sl} TP={tp} risk={rp}% | {result.get('reason','')[:80]}")

    if Config.EXECUTION_ENABLED:
        try:
            exec_result = execute_trade(action=direction, symbol=SYMBOL, lot_size=lot, sl=sl, tp=tp,
                                         ohlcv=m15_df, risk_evaluation={"approved":True, "adjusted_lot_scale":1.0})
            logger.info(f"V22 EXECUTED: {exec_result}")
        except Exception as e:
            logger.error(f"V22 execution error: {e}")
    else:
        last_entry_time = now_dt
        paper_positions.append({"entry": current_price, "tp": tp, "sl": sl,
                                 "direction": direction, "lot": lot,
                                 "open_time": now_dt.isoformat(), "be": False})
        logger.info(f"V22 PAPER OPEN: {direction} {lot} {SYMBOL} @ {current_price} SL={sl} TP={tp}")

def run_v22():
    logger.info("=" * 60)
    logger.info("V22 GOLD SCALPING BOT — LIVE (Paper Trading)")
    logger.info("Graduated risk: 0.5%/1.5%/2.5%/3% | Trailing SL | BUY EMA200 filter")
    logger.info(f"Execution: {'ENABLED' if Config.EXECUTION_ENABLED else 'PAPER ONLY'}")
    logger.info(f"Max positions: {MAX_POSITIONS} | Halt: {HALT_AFTER_LOSSES} losses → {HALT_HOURS}h")
    logger.info("=" * 60)
    if not mt5.initialize(login=Config.MT5_LOGIN, password=Config.MT5_PASSWORD, server=Config.MT5_SERVER):
        logger.critical("MT5 init failed"); return
    logger.info("MT5 connected")
    cycle = 0
    last_daily_reset = datetime.now(timezone.utc).date()
    while True:
        cycle += 1
        now_utc = datetime.now(timezone.utc)
        if now_utc.date() != last_daily_reset:
            reset_daily(); last_daily_reset = now_utc.date()
        if not mt5.terminal_info():
            logger.warning("MT5 disconnected, reconnecting...")
            mt5.shutdown(); time.sleep(5)
            if not mt5.initialize(login=Config.MT5_LOGIN, password=Config.MT5_PASSWORD, server=Config.MT5_SERVER):
                logger.error("Reconnect failed, retrying in 30s..."); time.sleep(30); continue
        try:
            acc = mt5.account_info()
            if acc:
                status = "paused" if is_paused() else ("halted" if (halt_until and now_utc < halt_until) else "running")
                write_state(acc.balance, acc.equity, paper_positions, status, cycle)
                if not is_paused():
                    v22_cycle()
                logger.info(f"--- CYCLE #{cycle} | Bal:${acc.balance:.2f} | Daily:${daily_pnl:+.2f} | Open:{len(paper_positions)} | Losses:{consecutive_losses}")
        except KeyboardInterrupt: logger.info("Stopped by user"); break
        except Exception as e: logger.error(f"Cycle #{cycle} error: {e}")
        interval = getattr(Config, 'ANALYSIS_INTERVAL_MINUTES', 1)
        time.sleep(interval * 60)
    mt5.shutdown()

if __name__ == "__main__":
    run_v22()