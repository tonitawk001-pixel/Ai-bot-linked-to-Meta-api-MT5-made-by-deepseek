"""
V22 Gold Scalping Bot — MetaApi Cloud Edition (FINAL v4.3)
=======================================================
CRITICAL FIXES (v4.3):
  1. TRUE Dynamic Risk Sizing — 2% flat risk per trade, sent to MetaApi
  2. LIVE Breakeven via modify_position() — secures profits on MetaApi
  3. LIVE 5x ATR TP — ADX dynamic TP actually sent to broker orders

BUG FIX: v4.2 calculated SL/TP in memory but NEVER called execute_trade()
          for strategy trades. Only startup_test/heartbeat placed real orders.
          Now ALL strategy trades execute real MetaApi orders with proper SL/TP.
"""

import os
import sys
import time
import json
from datetime import datetime, timedelta, timezone

import pandas as pd
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
load_dotenv(os.path.join(os.path.dirname(__file__), "..", ".env"))

from trading_bot.config import Config
from trading_bot.utils.logger import logger
from trading_bot.metaapi_connection import MetaApiConnection
from trading_bot.metaapi.data_feed import get_candles
from trading_bot.indicators.technical_indicators import compute_all_indicators
from trading_bot.strategy.gold_scalping_strategy import GoldScalpingStrategy
from trading_bot.strategy.gold_volatility_filter import GoldVolatilityFilter
from trading_bot.metaapi.executor import execute_trade, close_position, modify_position
from trading_bot.telegram_notifier import startup as tg_startup, shutdown as tg_shutdown, trade_opened as tg_open, trade_closed as tg_close, heartbeat as tg_hb, error as tg_err, set_name as tg_setname


# === V22 CONFIG ===
SYMBOL = "XAUUSD"
MIN_SCORE = 45
MAX_POSITIONS = 1
MIN_ATR = 1.0
TRADE_HOURS_START = 8
TRADE_HOURS_END = 22

# Position management
TP_ATR_MULT = 3.5
TP_ATR_MULT_TREND = 5.0
SL_ATR_MULT = 1.5
BE_ATR_MULT = 2.0
TRAIL_ATR_MULT = 0.7

# v4.3: Flat 2% risk
RISK_PERCENT_FLAT = 2.0  # Risk 2% of balance per trade

# Risk management
HALT_AFTER_LOSSES = 3
HALT_HOURS = 6
ENTRY_COOLDOWN_MINUTES = 30
DAILY_LOSS_PCT = 0.03

# Graduated risk (v4.3 keeps this for reference but uses flat 2%)
RISK_PERCENT = {(0, 250): 0.5, (250, 500): 1.5, (500, 1000): 2.5, (1000, float('inf')): 3.0}

HEARTBEAT_INTERVAL_MINUTES = 60
HEARTBEAT_CLOSE_AFTER_SECONDS = 30
BACKTEST_SPREAD_PIPS = 0.50

# v4.2 config (kept)
MAX_SPREAD_POINTS = 45
NEWS_PAUSE_MINUTES_BEFORE = 30
NEWS_RESUME_MINUTES_AFTER = 15
NEWS_CACHE_HOURS = 6
ADX_TREND_THRESHOLD = 25

# v4.3: Breakeven buffer in points
BE_BUFFER_POINTS = 12  # 12 points buffer above entry for breakeven

STATE_FILE = os.path.join(os.path.dirname(__file__), "..", "logs", "bot_state.json")
PAUSE_FILE = os.path.join(os.path.dirname(__file__), "..", "logs", "paused.flag")

# === Bot State ===
consecutive_losses = 0
halt_until = None
daily_pnl = 0.0
positions = []          # v4.3: Now tracks LIVE MetaApi positions with real IDs
last_entry = None
trades_log = []
last_heartbeat_time = None
last_processed_m15_time = None
startup_test_done = False

# v4.2 State
last_spread_check = 0.0
cached_events = []
events_cache_time = None

# v4.3: Track which positions have been breakeven-modified
be_modified_ids = set()


def get_risk_pct(balance: float) -> float:
    """v4.3: Always returns 2% flat risk."""
    return RISK_PERCENT_FLAT / 100.0  # Returns 0.02 for 2%


def is_in_trading_session(dt_utc: datetime) -> bool:
    hour = dt_utc.hour
    return TRADE_HOURS_START <= hour < TRADE_HOURS_END


def is_paused() -> bool:
    return os.path.exists(PAUSE_FILE)


def can_trade(now_dt: datetime, balance: float) -> bool:
    if halt_until and now_dt < halt_until:
        return False
    if daily_pnl <= -balance * DAILY_LOSS_PCT:
        return False
    return True


def record_loss():
    global consecutive_losses, halt_until
    consecutive_losses += 1
    if consecutive_losses >= HALT_AFTER_LOSSES:
        halt_until = datetime.now(timezone.utc) + timedelta(hours=HALT_HOURS)
        logger.warning(f"[HALT] {consecutive_losses} consecutive losses -> {HALT_HOURS}h pause")


def record_win():
    global consecutive_losses
    consecutive_losses = 0


def reset_daily():
    global daily_pnl
    daily_pnl = 0.0


def save_state():
    try:
        os.makedirs(os.path.dirname(STATE_FILE), exist_ok=True)
        state = {
            "positions": positions,
            "consecutive_losses": consecutive_losses,
            "halt_until": halt_until.isoformat() if halt_until else None,
            "daily_pnl": daily_pnl,
            "last_entry": last_entry.isoformat() if last_entry else None,
            "trades_log": trades_log[-200:],
            "last_processed_m15_time": last_processed_m15_time.isoformat() if last_processed_m15_time else None,
            "be_modified_ids": list(be_modified_ids),
            "updated": datetime.now(timezone.utc).isoformat(),
        }
        with open(STATE_FILE, "w") as f:
            json.dump(state, f, indent=2, default=str)
    except Exception as e:
        logger.error(f"Could not save state: {e}")


def load_state():
    global positions, consecutive_losses, halt_until, daily_pnl, last_entry, last_processed_m15_time, trades_log, be_modified_ids
    try:
        if not os.path.exists(STATE_FILE):
            return
        with open(STATE_FILE, "r") as f:
            state = json.load(f)
        positions = state.get("positions", [])
        consecutive_losses = state.get("consecutive_losses", 0)
        hu = state.get("halt_until")
        halt_until = datetime.fromisoformat(hu) if hu else None
        daily_pnl = state.get("daily_pnl", 0.0)
        le = state.get("last_entry")
        last_entry = datetime.fromisoformat(le) if le else None
        trades_log = state.get("trades_log", [])
        lm = state.get("last_processed_m15_time")
        last_processed_m15_time = datetime.fromisoformat(lm) if lm else None
        be_modified_ids = set(state.get("be_modified_ids", []))
        logger.info(f"STATE RESTORED: positions={len(positions)}, BE_ids={len(be_modified_ids)}")
    except Exception as e:
        logger.warning(f"Could not load state: {e}")


def write_state_for_dashboard(balance: float, equity: float, status: str, cycle: int):
    try:
        os.makedirs(os.path.dirname(STATE_FILE), exist_ok=True)
        state = {
            "balance": round(balance, 2),
            "equity": round(equity, 2),
            "daily_pnl": round(daily_pnl, 2),
            "positions": positions[-50:],
            "trades": trades_log[-200:],
            "status": status,
            "cycle": cycle,
            "consec_losses": consecutive_losses,
            "updated": datetime.now(timezone.utc).isoformat(),
            "platform": "metaapi",
        }
        with open(STATE_FILE, "w") as f:
            json.dump(state, f, indent=2)
    except Exception:
        pass


# ============================================================
# ADX Calculation
# ============================================================

def compute_adx(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    if len(close) < period * 2:
        return pd.Series([np.nan] * len(close), index=close.index)
    high = high.astype(float); low = low.astype(float); close = close.astype(float)
    tr = pd.concat([(high - low).abs(), (high - close.shift()).abs(), (low - close.shift()).abs()], axis=1).max(axis=1)
    up_move = high - high.shift(); down_move = low.shift() - low
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    atr = tr.ewm(span=period, adjust=False).mean()
    plus_di = 100 * pd.Series(plus_dm, index=close.index).ewm(span=period, adjust=False).mean() / atr
    minus_di = 100 * pd.Series(minus_dm, index=close.index).ewm(span=period, adjust=False).mean() / atr
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
    return dx.ewm(span=period, adjust=False).mean()


# ============================================================
# Spread Check
# ============================================================

def check_spread(conn: MetaApiConnection) -> tuple:
    global last_spread_check
    try:
        price = conn.get_symbol_price(SYMBOL)
        if price and "bid" in price and "ask" in price:
            bid = price["bid"]
            ask = price["ask"]
            spread_points = (ask - bid) * 100
            last_spread_check = spread_points
            if spread_points > MAX_SPREAD_POINTS:
                return False, spread_points, f"[REJECTED] Spread too high: {spread_points:.1f} points (max: {MAX_SPREAD_POINTS})"
            return True, spread_points, f"Spread OK: {spread_points:.1f} points"
        return False, 0, "[REJECTED] Could not fetch spread data"
    except Exception as e:
        return False, 0, f"[REJECTED] Spread check error: {e}"


# ============================================================
# News Filter
# ============================================================

def fetch_high_impact_events(conn: MetaApiConnection) -> list:
    global cached_events, events_cache_time
    now = datetime.now(timezone.utc)
    if events_cache_time and (now - events_cache_time).total_seconds() < NEWS_CACHE_HOURS * 3600:
        return cached_events
    try:
        calendar = conn.api.metatrader_account_api
        future_events = conn._run(_async_get_calendar_events(conn, calendar))
        high_impact_usd = []
        for ev in future_events:
            country = str(getattr(ev, "country", "") or "").upper()
            impact = str(getattr(ev, "impact", "") or "").upper()
            event_time = getattr(ev, "time", None)
            title = str(getattr(ev, "title", "") or "")
            if country == "USD" and impact == "HIGH" and event_time:
                ev_dt = event_time if isinstance(event_time, datetime) else datetime.fromisoformat(str(event_time))
                if ev_dt.tzinfo is None:
                    ev_dt = ev_dt.replace(tzinfo=timezone.utc)
                high_impact_usd.append({"time": ev_dt, "title": title})
        cached_events = high_impact_usd
        events_cache_time = now
        logger.info(f"[NEWS] Cached {len(high_impact_usd)} high-impact USD events")
        return high_impact_usd
    except Exception as e:
        logger.debug(f"[NEWS] Calendar fetch not available: {e}")
        cached_events = []
        events_cache_time = now
        return []


async def _async_get_calendar_events(conn, calendar_api):
    try:
        from datetime import timedelta as td
        start = datetime.now(timezone.utc)
        end = start + td(days=3)
        events = await calendar_api.get_events(start, end)
        return events
    except Exception:
        return []


def check_news_filter(conn: MetaApiConnection) -> tuple:
    try:
        events = fetch_high_impact_events(conn)
    except Exception:
        events = []
    if not events:
        return True, 999, "[NEWS] No high-impact events in window"
    now = datetime.now(timezone.utc)
    for ev in events:
        ev_time = ev["time"]
        mins_until = (ev_time - now).total_seconds() / 60.0
        if -15 <= mins_until <= NEWS_PAUSE_MINUTES_BEFORE:
            if mins_until > 0:
                msg = f"[REJECTED] High-impact USD news in {mins_until:.0f} mins: {ev['title']}"
            else:
                msg = f"[REJECTED] High-impact USD news event active: {ev['title']} (resume in {NEWS_RESUME_MINUTES_AFTER}min)"
            return False, mins_until, msg
        if -NEWS_RESUME_MINUTES_AFTER <= mins_until <= 0:
            msg = f"[REJECTED] Post-news cooldown ({-mins_until:.0f}/{NEWS_RESUME_MINUTES_AFTER}min): {ev['title']}"
            return False, mins_until, msg
    return True, 999, "[NEWS] Clear — no event conflict"


# ============================================================
# v4.3: Calculate lot size for 2% flat risk
# ============================================================

def get_dynamic_max_lot_size(balance: float) -> float:
    """Safety cap: 0.1 lots per $100 of balance, clamped 0.01-10.0."""
    cap = (balance / 100.0) * 0.1
    return max(0.01, min(cap, 10.0))


def calculate_lot_size(balance: float, sl_distance: float, conn: MetaApiConnection) -> float:
    """
    Calculate lot size to risk exactly 2% of balance.
    Formula: Risk Amount = Balance * 0.02
             Risk Per Lot = SL_Distance * 100 (XAUUSD contract size)
             Lots = Risk Amount / Risk Per Lot
    """
    if sl_distance <= 0 or balance <= 0:
        return 0.01  # Minimum safety

    risk_amount = balance * 0.02  # 2% of balance
    risk_per_lot = sl_distance * 100  # XAUUSD = 100 units per lot
    raw_lot = risk_amount / risk_per_lot

    # Clamp to broker limits (get symbol info from MetaApi)
    try:
        spec = conn.get_symbol_specification(SYMBOL)
        vol_min = spec.get("volume_min", 0.01)
        vol_max = spec.get("volume_max", 10.0)
        vol_step = spec.get("volume_step", 0.01)
    except:
        vol_min, vol_max, vol_step = 0.01, 10.0, 0.01

    # Round to step size
    lot = round(raw_lot / vol_step) * vol_step
    lot = max(vol_min, min(lot, vol_max))
    lot = max(0.01, round(lot, 2))

    # NEW: Dynamic safety cap based on balance
    max_lot = get_dynamic_max_lot_size(balance)
    if lot > max_lot:
        logger.warning(f"[SAFETY] Capping trade size {lot:.4f} to {max_lot:.4f} for balance ${balance:.2f}")
        lot = max_lot
        lot = round(lot / vol_step) * vol_step

    logger.info(f"[RISK] Balance=${balance:.2f} RiskAmt=${risk_amount:.2f} SLdist=${sl_distance:.2f} RawLot={raw_lot:.4f} FinalLot={lot:.2f}")
    return lot


# ============================================================
# Startup Test
# ============================================================

def startup_test(conn: MetaApiConnection):
    global startup_test_done
    if startup_test_done:
        return
    startup_test_done = True
    logger.info("=== STARTUP TEST: Opening 0.01 BUY for 30 seconds ===")
    try:
        current_price = None
        df = get_candles(symbol=SYMBOL, timeframe="M1", count=1)
        if df is not None and not df.empty:
            current_price = float(df["close"].iloc[-1])
        else:
            df = get_candles(symbol=SYMBOL, timeframe="M5", count=1)
            if df is not None and not df.empty:
                current_price = float(df["close"].iloc[-1])
        if current_price is None:
            logger.error("STARTUP TEST FAILED: Cannot fetch current price")
            return
        test_sl = round(current_price - 50, 2)
        test_tp = round(current_price + 50, 2)
        exec_result = execute_trade(
            action="BUY", symbol=SYMBOL, lot_size=0.01,
            sl=test_sl, tp=test_tp,
            ohlcv=df if df is not None else pd.DataFrame(),
            risk_evaluation={"approved": True, "adjusted_lot_scale": 1.0},
        )
        if exec_result and exec_result[0].get("success"):
            order_id = exec_result[0].get("order_id", "")
            logger.info(f"STARTUP TEST: Position opened (ID: {order_id}), closing in 30s")
            time.sleep(30)
            close_result = close_position(position_id=order_id)
            if close_result and close_result.get("success"):
                logger.info("=== STARTUP TEST: Position CLOSED — Connection OK ===")
            else:
                logger.warning(f"STARTUP TEST: Opened but close result: {close_result}")
        else:
            logger.error(f"STARTUP TEST FAILED: {exec_result}")
    except Exception as e:
        logger.error(f"STARTUP TEST FAILED: Exception: {e}")


# ============================================================
# Heartbeat Test
# ============================================================

def heartbeat_test(conn: MetaApiConnection):
    global last_heartbeat_time
    now = datetime.now(timezone.utc)
    if last_heartbeat_time is not None:
        elapsed = (now - last_heartbeat_time).total_seconds() / 60
        if elapsed < HEARTBEAT_INTERVAL_MINUTES:
            return
    last_heartbeat_time = now
    logger.info("--- HEARTBEAT: placing test trade to verify connection ---")
    try:
        current_price = None
        df = get_candles(symbol=SYMBOL, timeframe="M1", count=1)
        if df is not None and not df.empty:
            current_price = float(df["close"].iloc[-1])
        else:
            df = get_candles(symbol=SYMBOL, timeframe="M5", count=1)
            if df is not None and not df.empty:
                current_price = float(df["close"].iloc[-1])
        if current_price is None:
            logger.error("HEARTBEAT FAILED: Cannot fetch current price")
            return
        test_sl = round(current_price - 50, 2)
        test_tp = round(current_price + 50, 2)
        exec_result = execute_trade(
            action="BUY", symbol=SYMBOL, lot_size=0.01,
            sl=test_sl, tp=test_tp,
            ohlcv=df if df is not None else pd.DataFrame(),
            risk_evaluation={"approved": True, "adjusted_lot_scale": 1.0},
        )
        if exec_result and exec_result[0].get("success"):
            order_id = exec_result[0].get("order_id", "")
            logger.info(f"HEARTBEAT: Test trade opened, closing in {HEARTBEAT_CLOSE_AFTER_SECONDS}s")
            time.sleep(HEARTBEAT_CLOSE_AFTER_SECONDS)
            close_result = close_position(position_id=order_id)
            if close_result and close_result.get("success"):
                logger.info(f"HEARTBEAT: Connection is ALIVE.")
            else:
                logger.warning(f"HEARTBEAT: opened but close result: {close_result}")
        else:
            logger.error(f"HEARTBEAT FAILED: {exec_result}")
    except Exception as e:
        logger.error(f"HEARTBEAT FAILED: Exception: {e}")


def fetch_live_data(conn: MetaApiConnection) -> dict:
    """Fetch M5 + M15 candles with error handling."""
    data = {}
    for tf in ["M5", "M15"]:
        try:
            df = get_candles(symbol=SYMBOL, timeframe=tf, count=500)
        except Exception as e:
            logger.warning(f"[DATA] {tf} fetch failed ({e}), retrying with M15 fallback...")
            df = None
        if df is None or df.empty:
            if tf == "M5":
                try:
                    df = get_candles(symbol=SYMBOL, timeframe="M15", count=500)
                except Exception as e:
                    logger.warning(f"[DATA] M15 fallback also failed: {e}")
                    return None
            else:
                return None
        try:
            if not df.empty:
                data[tf] = df
        except:
            return None
    return data


def safe_vol_filter(vf, m5_ohlcv, m15_ohlcv, m5_indicators, m15_indicators):
    from trading_bot.utils.logger import logger as lg
    import logging
    old = lg.level; lg.setLevel(logging.ERROR)
    try:
        em1 = {"rsi": pd.Series([50]), "emas": pd.DataFrame(), "macd": pd.Series([0])}
        empty_m1_ohlcv = m5_ohlcv.tail(20)
        result = vf.analyze(
            m1_ohlcv=empty_m1_ohlcv, m5_ohlcv=m5_ohlcv, m15_ohlcv=m15_ohlcv,
            m1_indicators=em1, m5_indicators=m5_indicators, m15_indicators=m15_indicators,
        )
        return result
    except Exception:
        return {"trade_ok": True, "lot_reduction_factor": 1.0}
    finally:
        lg.setLevel(old)


def update_paper_positions(current_price: float, atr_val: float):
    """v4.3: Now also calls modify_position() for LIVE breakeven on MetaApi."""
    global daily_pnl
    surviving = []
    for p in positions:
        e, d, sl, tp, lot = p["entry"], p["dir"], p["sl"], p["tp"], p["lot"]
        pv = lot * 100
        pos_id = p.get("position_id", "")

        # ======= v4.3: LIVE BREAKEVEN via modify_position() =======
        if pos_id and pos_id not in be_modified_ids:
            be_price = e + (atr_val * BE_ATR_MULT) if d == "BUY" else e - (atr_val * BE_ATR_MULT)
            be_hit = (d == "BUY" and current_price >= be_price) or (d == "SELL" and current_price <= be_price)
            if be_hit:
                buffer = BE_BUFFER_POINTS * 0.01  # Convert points to price
                new_sl = e + buffer if d == "BUY" else e - buffer
                logger.info(f"[BREAKEVEN] Moving SL to {new_sl:.2f} for position {pos_id}")
                try:
                    result = modify_position(position_id=pos_id, sl=round(new_sl, 2), tp=tp)
                    if result and result.get("success"):
                        p["sl"] = round(new_sl, 2)
                        p["be"] = True
                        be_modified_ids.add(pos_id)
                        logger.info(f"[BREAKEVEN] ✅ Position {pos_id} secured at {new_sl:.2f}")
                    else:
                        logger.warning(f"[BREAKEVEN] modify_position failed: {result}")
                except Exception as ex:
                    logger.warning(f"[BREAKEVEN] modify_position exception: {ex}")

        # Original in-memory trailing SL
        if not p.get("be", False) and p.get("be_target"):
            if d == "BUY" and current_price >= p["be_target"]:
                p["be"] = True; p["sl"] = e
            elif d == "SELL" and current_price <= p["be_target"]:
                p["be"] = True; p["sl"] = e
        if p.get("be"):
            ns = current_price - atr_val * TRAIL_ATR_MULT if d == "BUY" else current_price + atr_val * TRAIL_ATR_MULT
            if d == "BUY" and ns > sl + 0.5:
                p["sl"] = round(ns, 2)
                # Also update live SL on MetaApi
                if pos_id:
                    try:
                        modify_position(position_id=pos_id, sl=round(ns, 2), tp=tp)
                    except:
                        pass
            elif d == "SELL" and ns < sl - 0.5:
                p["sl"] = round(ns, 2)
                if pos_id:
                    try:
                        modify_position(position_id=pos_id, sl=round(ns, 2), tp=tp)
                    except:
                        pass

        sl, tp = p["sl"], p["tp"]
        hit, pnl, reason = False, 0.0, ""
        if d == "BUY":
            if tp and current_price >= tp:
                pnl = (tp - e) * pv; reason = "TP"; hit = True
            elif sl and current_price <= sl:
                pnl = (sl - e) * pv; reason = "TRAIL" if sl > e else "SL"; hit = True
        else:
            if tp and current_price <= tp:
                pnl = (e - tp) * pv; reason = "TP"; hit = True
            elif sl and current_price >= sl:
                pnl = (e - sl) * pv; reason = "TRAIL" if sl < e else "SL"; hit = True
        if hit:
            pnl -= BACKTEST_SPREAD_PIPS * lot * 100
            daily_pnl += pnl
            p["pnl"] = pnl; p["reason"] = reason; p["close_price"] = current_price
            p["close_time"] = datetime.now(timezone.utc)
            trades_log.append(p)
            # Close on MetaApi if we have a position ID
            if pos_id and reason in ("TP", "SL", "TRAIL"):
                try:
                    close_position(position_id=pos_id)
                    logger.info(f"[CLOSE] Position {pos_id} closed ({reason}) via MetaApi")
                except Exception as ex:
                    logger.warning(f"[CLOSE] close_position failed: {ex}")
            if reason == "SL":
                record_loss()
            else:
                record_win()
        else:
            surviving.append(p)
    return surviving


def sync_state_with_broker(conn: MetaApiConnection):
    """Fetch live positions from MetaApi and remove stale entries from bot_state.json."""
    global positions
    try:
        live_positions = conn.get_positions(symbol=SYMBOL)
        live_ids = set(p.get("id", "") for p in live_positions if p.get("id"))
        before = len(positions)
        positions = [p for p in positions if p.get("position_id", "") in live_ids]
        removed = before - len(positions)
        if removed > 0:
            logger.warning(f"[STATE SYNC] Removed {removed} stale/orphaned positions from state")
        if live_ids:
            logger.info(f"[STATE SYNC] Broker has {len(live_ids)} live positions, tracking {len(positions)}")
    except Exception as e:
        logger.warning(f"[STATE SYNC] Could not sync (non-fatal): {e}")


# Track consecutive connection failures for full re-init
_connection_failures = 0

def check_connection(conn: MetaApiConnection) -> bool:
    """Hardened connection check with retry loop and full re-init on persistent failure."""
    global _connection_failures
    max_retries = 3
    for attempt in range(1, max_retries + 1):
        try:
            acc = conn.get_account_info()
            if acc and "balance" in acc:
                if attempt > 1:
                    logger.info(f"[CONNECT] Reconnected on attempt {attempt}")
                _connection_failures = 0
                return True
        except Exception as e:
            logger.warning(f"[CONNECT] Attempt {attempt}/{max_retries} failed: {e}")
            if attempt < max_retries:
                conn._initialized = False
                time.sleep(5 * attempt)
                try:
                    if conn.initialize():
                        _connection_failures = 0
                        return True
                except:
                    pass
    
    # If all 3 attempts failed, increment counter
    _connection_failures += 1
    
    # After 3 consecutive failed attempts (9 total), force a full restart
    if _connection_failures >= 3:
        logger.critical(f"[CONNECT] {_connection_failures} consecutive failures — forcing full re-initialization")
        try:
            conn.shutdown()
            time.sleep(10)
            conn._initialized = False
            if conn.initialize():
                _connection_failures = 0
                logger.info("[CONNECT] Full re-initialization successful")
                return True
        except Exception as e:
            logger.error(f"[CONNECT] Full re-init also failed: {e}")
    
    logger.critical("[CONNECT] All reconnection attempts failed — will retry next cycle")
    return False


def v22_cycle(conn: MetaApiConnection):
    """v4.3: Now sends REAL MetaApi orders with execute_trade() for strategy trades."""
    global positions, last_entry, daily_pnl, last_processed_m15_time, last_date, halt_until, consecutive_losses

    if not check_connection(conn):
        logger.warning("Skipping cycle - no connection")
        return

    data = fetch_live_data(conn)
    if data is None:
        return
    m5_df, m15_df = data.get("M5"), data.get("M15")
    if m5_df is None or m15_df is None or m5_df.empty or m15_df.empty:
        return

    last_m15_time = m15_df.index[-1]
    if last_processed_m15_time is not None and last_m15_time <= last_processed_m15_time:
        return

    now_utc = datetime.now(timezone.utc)

    if last_date is None:
        last_date_today()
    if last_date is not None and last_date != now_utc.date():
        daily_pnl = 0.0
        last_date = now_utc.date()

    if halt_until and now_utc < halt_until:
        return
    if daily_pnl <= -STARTING_BALANCE * DAILY_LOSS_PCT:
        return

    if not is_in_trading_session(now_utc):
        last_processed_m15_time = last_m15_time
        return

    # FRIDAY ENTRY BLOCK: No new trades after 18:00 UTC on Friday
    if now_utc.weekday() == 4 and now_utc.hour >= 18:
        logger.info(f"[FRIDAY] Blocking entries - Friday {now_utc.hour}:00 UTC after 18:00 cutoff")
        last_processed_m15_time = last_m15_time
        return

    m15w = m15_df.tail(500).copy()
    m5u = m5_df[m5_df.index <= last_m15_time]
    m5w = m5u.tail(500).copy()

    if len(m15w) < 50 or len(m5w) < 50:
        last_processed_m15_time = last_m15_time
        return

    try:
        m5_ind = compute_all_indicators(m5w)
        m15_ind = compute_all_indicators(m15w)
    except Exception as exc:
        logger.error(f"Indicator compute failed: {exc}")
        last_processed_m15_time = last_m15_time
        return

    if m5_ind is None or m15_ind is None:
        last_processed_m15_time = last_m15_time
        return
    if m5_ind.get("atr") is None or len(m5_ind["atr"]) == 0:
        last_processed_m15_time = last_m15_time
        return

    atr_val = float(m5_ind["atr"].iloc[-1])
    if atr_val < MIN_ATR:
        last_processed_m15_time = last_m15_time
        return

    # ADX for dynamic TP
    try:
        adx_series = compute_adx(m5w["high"], m5w["low"], m5w["close"], period=14)
        adx_val = float(adx_series.iloc[-1]) if not pd.isna(adx_series.iloc[-1]) else 0
    except Exception:
        adx_val = 0

    tp_mult = TP_ATR_MULT_TREND if adx_val >= ADX_TREND_THRESHOLD else TP_ATR_MULT
    current_price = float(m15w["close"].iloc[-1])

    # UPDATE POSITIONS — includes LIVE breakeven + trailing via MetaApi
    positions[:] = update_paper_positions(current_price, atr_val)

    if consecutive_losses >= HALT_AFTER_LOSSES and halt_until is None:
        halt_until = now_utc + timedelta(hours=HALT_HOURS)
        logger.warning(f"[HALT] {consecutive_losses} consecutive losses -> {HALT_HOURS}h pause")
        last_processed_m15_time = last_m15_time
        return

    # Check if we already have a live position
    if len(positions) >= MAX_POSITIONS:
        last_processed_m15_time = last_m15_time
        save_state()
        return

    if last_entry and (now_utc - last_entry).total_seconds() / 60 < ENTRY_COOLDOWN_MINUTES:
        last_processed_m15_time = last_m15_time
        return

    if is_paused():
        last_processed_m15_time = last_m15_time
        return

    # SPREAD FILTER
    spread_ok, spread_pts, spread_msg = check_spread(conn)
    if not spread_ok:
        logger.warning(spread_msg)
        last_processed_m15_time = last_m15_time
        return

    # NEWS FILTER
    news_ok, news_mins, news_msg = check_news_filter(conn)
    if not news_ok:
        logger.warning(news_msg)
        last_processed_m15_time = last_m15_time
        return

    # RUN STRATEGY
    strategy = GoldScalpingStrategy()
    strategy._max_trades_per_day = 50
    strategy._max_open_positions = MAX_POSITIONS
    try:
        empty_m1 = {"rsi": pd.Series([50]), "emas": pd.DataFrame(), "macd": pd.Series([0])}
        empty_m1_ohlcv = m5w.tail(20)
        result = strategy.analyze(
            m1_indicators=empty_m1, m5_indicators=m5_ind, m15_indicators=m15_ind,
            m1_ohlcv=empty_m1_ohlcv, m5_ohlcv=m5w, m15_ohlcv=m15w, news_context=None,
        )
    except Exception:
        last_processed_m15_time = last_m15_time
        return

    direction = result.get("direction", "NONE")
    score = result.get("setup_score", 0)
    if direction == "NONE" or score < MIN_SCORE:
        logger.info(f"[REJECTED] Direction: {direction} Score: {score} (need ≥{MIN_SCORE})")
        last_processed_m15_time = last_m15_time
        return

    # RSI CONFLUENCE
    try:
        rsi_bullish = m5_ind["rsi"].iloc[-1] > 40 and m15_ind["rsi"].iloc[-1] > 40
        rsi_bearish = m5_ind["rsi"].iloc[-1] < 60 and m15_ind["rsi"].iloc[-1] < 60
        if direction == "BUY" and not rsi_bullish:
            logger.info(f"[REJECTED] RSI confluence: BUY requires RSI>40 on both M5 ({m5_ind['rsi'].iloc[-1]:.1f}) and M15 ({m15_ind['rsi'].iloc[-1]:.1f})")
            last_processed_m15_time = last_m15_time
            return
        if direction == "SELL" and not rsi_bearish:
            logger.info(f"[REJECTED] RSI confluence: SELL requires RSI<60 on both M5 ({m5_ind['rsi'].iloc[-1]:.1f}) and M15 ({m15_ind['rsi'].iloc[-1]:.1f})")
            last_processed_m15_time = last_m15_time
            return
    except Exception:
        pass

    # EMA200 FILTER
    closes = m15w["close"].values
    if len(closes) >= 200:
        ema200 = pd.Series(closes).ewm(200, adjust=False).mean().values
        if len(ema200) >= 10:
            rising = float(ema200[-1]) > float(ema200[-10])
            if direction == "BUY" and not rising:
                logger.info(f"[REJECTED] EMA200 trend: BUY requires EMA200 rising")
                last_processed_m15_time = last_m15_time
                return
            if direction == "SELL" and rising:
                logger.info(f"[REJECTED] EMA200 trend: SELL requires EMA200 falling")
                last_processed_m15_time = last_m15_time
                return

    # VOLATILITY FILTER
    vf = GoldVolatilityFilter()
    try:
        vo = vf.analyze(
            m1_ohlcv=empty_m1_ohlcv, m5_ohlcv=m5w, m15_ohlcv=m15w,
            m1_indicators=empty_m1, m5_indicators=m5_ind, m15_indicators=m15_ind,
        )
        if not vo.get("trade_ok", False):
            logger.info(f"[REJECTED] Volatility filter: trade_ok=False")
            last_processed_m15_time = last_m15_time
            return
    except Exception:
        pass

    # ======= v4.3: CALCULATE REAL SL/TP =======
    sl_dist = atr_val * SL_ATR_MULT
    tp_dist = atr_val * tp_mult

    if direction == "BUY":
        sl = round(current_price - sl_dist, 2)
        tp = round(current_price + tp_dist, 2)
    else:
        sl = round(current_price + sl_dist, 2)
        tp = round(current_price - tp_dist, 2)

    # Get balance for risk calculation
    try:
        balance = conn.get_account_info()["balance"]
    except Exception:
        balance = 304.99

    # ======= v4.3: CALCULATE LOT SIZE (2% flat risk) =======
    lot = calculate_lot_size(balance, sl_dist, conn)

    be_target = current_price + (atr_val * BE_ATR_MULT if direction == "BUY" else -atr_val * BE_ATR_MULT)

    # ======= v4.3: PLACE REAL METAPI ORDER =======
    logger.info(f"[EXECUTE] Sending {direction} {lot} lots to MetaApi: SL={sl} TP={tp} (ADX={adx_val:.1f} {'TREND TP' if adx_val >= ADX_TREND_THRESHOLD else ''})")

    exec_result = execute_trade(
        action=direction,
        symbol=SYMBOL,
        lot_size=lot,
        sl=sl,
        tp=tp,
        ohlcv=pd.DataFrame(),
        risk_evaluation={"approved": True, "adjusted_lot_scale": 1.0},
    )

    if exec_result and exec_result[0].get("success"):
        order_id = exec_result[0].get("order_id", "")
        logger.info(f"✅ V22 SIGNAL: {direction} score={score} lot={lot} SL={sl} TP={tp} atr={atr_val:.2f} adx={adx_val:.1f} spread={spread_pts:.1f}pts OrderID={order_id}")

        # Track position with real MetaApi order ID
        pos = {
            "entry": current_price,
            "sl": sl,
            "tp": tp,
            "lot": lot,
            "dir": direction,
            "open_time": last_m15_time.to_pydatetime() if hasattr(last_m15_time, 'to_pydatetime') else last_m15_time,
            "score": score,
            "be_target": be_target,
            "be": False,
            "position_id": order_id,  # v4.3: REAL MetaApi position ID
        }
        positions.append(pos)
        last_entry = now_utc
    else:
        reason = exec_result[0].get("reason", "Unknown") if exec_result else "No result"
        logger.error(f"[REJECTED] MetaApi order failed: {reason}")

    last_processed_m15_time = last_m15_time
    save_state()


# State variable for daily reset
last_date = None
STARTING_BALANCE = 304.99

def last_date_today():
    global last_date
    last_date = datetime.now(timezone.utc).date()


def run_v22():
    global last_date, STARTING_BALANCE
    logger.info("=" * 60)
    logger.info("V22 GOLD SCALPING BOT — MetaApi Cloud (v4.3)")
    logger.info("LIVE MetaApi orders | 2% Flat Risk | ADX Dynamic TP | Breakeven")
    logger.info("=" * 60)
    logger.info(f"Config: MIN_SCORE={MIN_SCORE}, MAX_POS={MAX_POSITIONS}, Risk=2% flat")
    logger.info(f"TP: {TP_ATR_MULT}x/{TP_ATR_MULT_TREND}x | SL: {SL_ATR_MULT}x")
    logger.info(f"Spread max: {MAX_SPREAD_POINTS}pts | ADX threshold: {ADX_TREND_THRESHOLD}")
    logger.info(f"Session: {TRADE_HOURS_START}:00-{TRADE_HOURS_END}:00 UTC")
    logger.info(f"Execution: {'ENABLED' if Config.EXECUTION_ENABLED else 'PAPER ONLY'}")
    logger.info("=" * 60)

    conn = MetaApiConnection()
    if not conn.initialize():
        logger.critical("MetaApi init failed.")
        return

    load_state()
    last_date = datetime.now(timezone.utc).date()

    # State sync: clean orphaned positions from bot_state.json
    sync_state_with_broker(conn)

    # Telegram startup
    try:
        acc_info = conn.get_account_info()
        bal = acc_info.get("balance", 0) if acc_info else 0
        tg_setname(f"Contabo_{acc_info.get('login', '?')}")
        tg_startup(balance=bal)
    except:
        pass

    if not is_paused():
        startup_test(conn)

    cycle = 0

    while True:
        try:
            cycle += 1
            now_utc = datetime.now(timezone.utc)

            if not is_paused():
                heartbeat_test(conn)

            # FRIDAY AUTO-CLOSE: Close all positions at 21:00 UTC
            if now_utc.weekday() == 4 and now_utc.hour >= 21:
                for p in list(positions):
                    pos_id = p.get("position_id", "")
                    if pos_id:
                        try:
                            close_position(position_id=pos_id)
                            logger.info(f"[FRIDAY] Auto-closing position {pos_id} at {now_utc.strftime('%H:%M UTC')}")
                        except Exception as ex:
                            logger.warning(f"[FRIDAY] close_position failed for {pos_id}: {ex}")
                positions.clear()

            if not is_paused():
                v22_cycle(conn)

            if cycle % 10 == 0:
                logger.info(f"Cycle #{cycle} | Open: {len(positions)} | Losses: {consecutive_losses} | BE_ids: {len(be_modified_ids)} | Trades: {len(trades_log)}")

            try:
                acc = conn.get_account_info()
                bal = acc["balance"]
                eq = acc["equity"]
                status = "paused" if is_paused() else ("halted" if (halt_until and now_utc < halt_until) else "running")
                write_state_for_dashboard(bal, eq, status, cycle)
            except Exception:
                pass

        except KeyboardInterrupt:
            logger.info("Stopped by user.")
            save_state()
            break
        except Exception as exc:
            logger.error(f"Cycle #{cycle} error: {exc}", exc_info=True)
            save_state()

        interval_min = getattr(Config, "ANALYSIS_INTERVAL_MINUTES", 1)
        time.sleep(interval_min * 60)


if __name__ == "__main__":
    run_v22()