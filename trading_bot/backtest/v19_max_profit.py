"""
V19 — MAXIMUM COMPOUNDING (Relaxed Filters + Higher Risk)
==========================================================

Changes from V17:
  - Remove Monday/Thursday blocks
  - Death zone only {10,11,12,16} UTC
  - 3% risk per trade
  - Max 5 concurrent positions
  - No single-SL cooldown (only 3-consecutive-loss → 2-hour halt)
"""

import sys, os, time, json
from datetime import datetime, timedelta, timezone
from collections import defaultdict
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from trading_bot.utils.logger import logger
from trading_bot.strategy.gold_scalping_strategy import GoldScalpingStrategy
from trading_bot.strategy.gold_volatility_filter import GoldVolatilityFilter
from trading_bot.risk.risk_manager import RiskManager
from trading_bot.indicators.technical_indicators import compute_all_indicators
from trading_bot.config import Config
from trading_bot.backtest.gold_backtest import MockDeepSeekClient, MockNewsAggregator

import MetaTrader5 as mt5

DAYS = 180
START_OFFSET_DAYS = 180  # Shift window back: fetch 180-360 days ago (Jul-Dec 2025)
INITIAL_BALANCE = 300.0
SYMBOL = "XAUUSD"
MT5_LOGIN = Config.MT5_LOGIN
MT5_PASSWORD = Config.MT5_PASSWORD
MT5_SERVER = Config.MT5_SERVER

# V19: Relaxed filters
DEATH_ZONE_HOURS = set()  # Removed — doesn't generalize across time windows
SKIP_THURSDAYS = False
SKIP_MONDAYS = False
MAX_POSITIONS = 5  # Up from 3
BUY_REQUIRES_RISING_EMA200 = True
TRAILING_SL_ENABLED = True
TRAIL_AFTER_ATR_MULT = 1.0
TRAIL_DISTANCE_ATR_MULT = 0.7
RISK_PERCENT = 3.0  # Up from 2%
MIN_SCORE_THRESHOLD = 50
HALT_AFTER_CONSEC_LOSSES = 3
HALT_DURATION_HOURS = 2  # Down from 4


class Tracker:
    def __init__(self, balance=300.0):
        self.balance = balance
        self.equity = balance
        self.equity_peak = balance
        self.equity_low = balance
        self.open_positions = []
        self.consecutive_losses = 0
        self.loss_streak_halt_until = None
        self.daily_pnl = 0.0
        self.trades = []
        self.daily_trades = defaultdict(int)
        self.daily_low_balance = balance  # Track lowest balance point

    def count_open(self):
        return len(self.open_positions)

    def open_pos(self, entry, tp, sl, dt, direction="BUY", lot=0.01):
        self.open_positions.append({
            "entry": entry, "tp": tp, "sl": sl, "initial_sl": sl,
            "direction": direction, "lot": lot,
            "open_time": dt, "breakeven_moved": False,
        })

    def update_all(self, price, dt, atr_val=None):
        closed, remaining = [], []
        atr = atr_val or 3.5

        for pos in self.open_positions:
            hit = False; pnl = 0.0; reason = ""
            direction = pos.get("direction", "BUY")
            lot = pos.get("lot", 0.01)
            pip_value = lot * 100
            entry = pos["entry"]
            current_sl = pos["sl"]

            if TRAILING_SL_ENABLED and not pos.get("breakeven_moved", False):
                profit_move = price - entry if direction == "BUY" else entry - price
                if profit_move >= atr * TRAIL_AFTER_ATR_MULT:
                    pos["breakeven_moved"] = True
                    pos["sl"] = entry
            elif TRAILING_SL_ENABLED and pos.get("breakeven_moved", False):
                if direction == "BUY":
                    new_sl = price - atr * TRAIL_DISTANCE_ATR_MULT
                    if new_sl > current_sl + 0.5: pos["sl"] = round(new_sl, 2)
                else:
                    new_sl = price + atr * TRAIL_DISTANCE_ATR_MULT
                    if new_sl < current_sl - 0.5: pos["sl"] = round(new_sl, 2)

            current_sl = pos["sl"]
            tp = pos["tp"]

            if direction == "BUY":
                if tp and price >= tp:
                    pnl = (tp - entry) * pip_value; reason = "TP"; hit = True
                elif current_sl and price <= current_sl:
                    pnl = (current_sl - entry) * pip_value
                    reason = "TRAIL" if current_sl > entry else "SL"
                    hit = True
            else:
                if tp and price <= tp:
                    pnl = (entry - tp) * pip_value; reason = "TP"; hit = True
                elif current_sl and price >= current_sl:
                    pnl = (entry - current_sl) * pip_value
                    reason = "TRAIL" if current_sl < entry else "SL"
                    hit = True

            if hit:
                self.balance += pnl
                self.daily_pnl += pnl
                closed.append({**pos, "pnl": round(pnl, 2), "reason": reason, "close_time": dt})
                self.trades.append({**pos, "pnl": round(pnl, 2), "reason": reason, "close_time": dt})
                if reason == "SL":
                    self.consecutive_losses += 1
                    if self.consecutive_losses >= HALT_AFTER_CONSEC_LOSSES:
                        self.loss_streak_halt_until = dt + timedelta(hours=HALT_DURATION_HOURS)
                else:
                    self.consecutive_losses = 0
            else:
                remaining.append(pos)

        self.open_positions = remaining
        self.daily_low_balance = min(self.daily_low_balance, self.balance)
        return closed

    def can_trade(self, dt):
        if self.loss_streak_halt_until and dt < self.loss_streak_halt_until:
            return False
        if self.daily_pnl <= -INITIAL_BALANCE * 0.05:
            return False
        return True

    def reset_daily(self):
        self.daily_pnl = 0.0
        self.daily_low_balance = self.balance

    def force_close_all(self, price, dt):
        for pos in self.open_positions:
            direction = pos.get("direction", "BUY"); lot = pos.get("lot", 0.01)
            pnl = ((price - pos["entry"]) if direction == "BUY" else (pos["entry"] - price)) * lot * 100
            self.trades.append({**pos, "pnl": round(pnl, 2), "reason": "EOD", "close_time": dt})
            self.balance += pnl
        self.open_positions = []

    def get_equity(self, price):
        unrealized = 0.0
        for pos in self.open_positions:
            lot = pos.get("lot", 0.01)
            unrealized += ((price - pos["entry"]) if pos.get("direction")=="BUY" else (pos["entry"]-price)) * lot * 100
        equity = self.balance + unrealized
        self.equity_peak = max(self.equity_peak, equity)
        self.equity_low = min(self.equity_low, equity)
        return equity


def fetch_mt5_data(symbol, days=180):
    if not mt5.initialize(login=MT5_LOGIN, password=MT5_PASSWORD, server=MT5_SERVER):
        return None
    if not mt5.symbol_select(symbol, True):
        mt5.shutdown(); return None

    tf_map = {"M1": mt5.TIMEFRAME_M1, "M5": mt5.TIMEFRAME_M5, "M15": mt5.TIMEFRAME_M15}
    data = {}
    end_dt = datetime.now(timezone.utc) - timedelta(days=START_OFFSET_DAYS)
    start_dt = end_dt - timedelta(days=days)

    for tf_name, tf_val in tf_map.items():
        rates = mt5.copy_rates_range(symbol, tf_val, start_dt, end_dt)
        if rates is None or len(rates) == 0:
            if tf_name == "M1":
                # M1 may not be available for old windows — just skip it
                logger.warning(f"  M1 unavailable for this window, will use M5 as substitute")
                data[tf_name] = pd.DataFrame()
                continue
            logger.error(f"  {tf_name} fetch failed: {mt5.last_error()}")
            if tf_name != "M1":
                mt5.shutdown(); return None
            data[tf_name] = pd.DataFrame(); continue
        df = pd.DataFrame(rates); df["time"] = pd.to_datetime(df["time"], unit="s", utc=True); df.set_index("time", inplace=True)
        df["session"] = df.index.map(lambda dt: "weekend" if dt.weekday()>=5 else ("london" if 8<=dt.hour<13 else ("overlap" if 13<=dt.hour<17 else ("new_york" if 17<=dt.hour<22 else "asian"))))
        data[tf_name] = df
    mt5.shutdown()
    return data


def _ema200_slope(window_15m) -> bool:
    try:
        close = window_15m["close"].values
        if len(close) < 200: return False
        ema200 = pd.Series(close).ewm(span=200, adjust=False).mean().values
        return len(ema200) >= 10 and float(ema200[-1]) > float(ema200[-10])
    except: return False


def run_v19():
    test_end = datetime.now(timezone.utc) - timedelta(days=START_OFFSET_DAYS)
    test_start = test_end - timedelta(days=DAYS)
    logger.info("=" * 70)
    logger.info(f"V19 — MAX COMPOUNDING: 3% risk, 5 positions, no Mon/Thu block")
    logger.info(f"WINDOW: {test_start.date()} to {test_end.date()} (offset={START_OFFSET_DAYS}d)")
    logger.info(f"Death hours: {DEATH_ZONE_HOURS} | Min score: {MIN_SCORE_THRESHOLD}")
    logger.info("=" * 70)

    data = fetch_mt5_data(SYMBOL, days=DAYS)
    if data is None: logger.critical("No data."); return

    m1_df, m5_df, m15_df = data["M1"], data["M5"], data["M15"]
    if len(m15_df) < 500: return

    strategy = GoldScalpingStrategy()
    strategy._max_trades_per_day = 200; strategy._max_open_positions = 10
    vol_filter = GoldVolatilityFilter()
    tracker = Tracker(balance=INITIAL_BALANCE)
    min_warmup = 200; total = len(m15_df)
    blocked = defaultdict(int)
    start_time = time.time()
    all_time_low_balance = INITIAL_BALANCE

    for idx in range(min_warmup, total):
        dt = m15_df.index[idx]
        price_m15 = float(m15_df["close"].iloc[idx])
        session = m15_df["session"].iloc[idx]

        pct = (idx - min_warmup) / (total - min_warmup) * 100
        if int(pct) % 10 == 0 and int(pct) > 0:
            elapsed = time.time() - start_time
            all_time_low_balance = min(all_time_low_balance, tracker.balance)
            logger.info(f"  {int(pct)}% | Bal:${tracker.balance:.0f} | Low:${all_time_low_balance:.0f} | T:{len(tracker.trades)} | Open:{tracker.count_open()}")

        if session == "weekend": tracker.force_close_all(price_m15, dt); continue
        if dt.hour == 0 and dt.minute < 15: tracker.reset_daily()
        if dt.hour in DEATH_ZONE_HOURS: blocked["death"] += 1; continue

        # No Monday/Thursday blocks

        if not m1_df.empty:
            m1_window = m1_df[m1_df.index <= dt].iloc[-500:] if len(m1_df[m1_df.index <= dt]) >= 500 else m1_df[m1_df.index <= dt]
        else:
            m1_window = pd.DataFrame()
        m5_window = m5_df[m5_df.index <= dt].iloc[-500:] if len(m5_df[m5_df.index <= dt]) >= 500 else m5_df[m5_df.index <= dt]
        m15_window = m15_df[m15_df.index <= dt].iloc[idx + 1 - min(idx + 1, 500):idx + 1]

        # Fallback: if M1 is unavailable, use M5 as substitute
        has_m1 = len(m1_window) >= 50
        if not has_m1 and len(m5_window) < 50: continue
        if has_m1 and (len(m5_window) < 50 or len(m15_window) < 50): continue

        try:
            m1_ind = compute_all_indicators(m1_window) if has_m1 else compute_all_indicators(m5_window)
            m5_ind = compute_all_indicators(m5_window)
            m15_ind = compute_all_indicators(m15_window)
            m1_ohlcv = m1_window if has_m1 else m5_window
        except: continue

        try:
            atr_series = m5_ind.get("atr", pd.Series(dtype=float))
            atr_val = float(atr_series.iloc[-1]) if not atr_series.empty else 3.5
        except: atr_val = 3.5

        tracker.update_all(price_m15, dt, atr_val=atr_val)
        all_time_low_balance = min(all_time_low_balance, tracker.balance)

        if not tracker.can_trade(dt): continue
        if tracker.count_open() >= MAX_POSITIONS: continue

        try:
            result = strategy.analyze(
                m1_indicators=m1_ind, m5_indicators=m5_ind, m15_indicators=m15_ind,
                m1_ohlcv=m1_ohlcv, m5_ohlcv=m5_window, m15_ohlcv=m15_window, news_context=None)
        except: continue

        direction = result.get("direction", "NONE")
        score = result.get("setup_score", 0)

        if direction == "NONE" or score < MIN_SCORE_THRESHOLD:
            blocked["low_score"] += 1; continue

        if BUY_REQUIRES_RISING_EMA200 and direction == "BUY":
            if not _ema200_slope(m15_window): blocked["buy_ema"] += 1; continue

        try:
            vol_result = vol_filter.analyze(
                m1_ohlcv=m1_window, m5_ohlcv=m5_window, m15_ohlcv=m15_window,
                m1_indicators=m1_ind, m5_indicators=m5_ind, m15_indicators=m15_ind)
            if not vol_result.get("trade_ok", False): blocked["vol"] += 1; continue
        except: continue

        entry = price_m15
        sl_dist = atr_val * 1.5
        tp_dist = atr_val * 3.0

        if direction == "BUY":
            sl = round(entry - sl_dist, 2); tp = round(entry + tp_dist, 2)
        else:
            sl = round(entry + sl_dist, 2); tp = round(entry - tp_dist, 2)

        current_balance = tracker.balance
        risk_amount = current_balance * (RISK_PERCENT / 100.0)
        lot = risk_amount / (sl_dist * 100) if sl_dist > 0 else 0.01
        lot = max(0.01, min(lot, 50.0))
        lot = round(lot, 2)

        tracker.open_pos(entry, tp, sl, dt, direction=direction, lot=lot)
        tracker.daily_trades[dt.date()] += 1

    final_price = float(m15_df["close"].iloc[-1])
    tracker.force_close_all(final_price, m15_df.index[-1])
    elapsed = time.time() - start_time
    all_time_low_balance = min(all_time_low_balance, tracker.balance)

    # REPORT
    trades = tracker.trades
    total = len(trades)
    wins = [t for t in trades if t["pnl"] > 0]
    losses = [t for t in trades if t["pnl"] < 0]
    wr = len(wins) / max(total, 1) * 100
    gp = sum(t["pnl"] for t in wins)
    gl = abs(sum(t["pnl"] for t in losses))
    pf = gp / max(gl, 0.01)
    total_pnl = sum(t["pnl"] for t in trades)
    fb = INITIAL_BALANCE + total_pnl
    buy_pnl = sum(t["pnl"] for t in trades if t.get("direction") == "BUY")
    sell_pnl = sum(t["pnl"] for t in trades if t.get("direction") == "SELL")
    buy_cnt = sum(1 for t in trades if t.get("direction") == "BUY")
    sell_cnt = sum(1 for t in trades if t.get("direction") == "SELL")
    active_days = len(tracker.daily_trades)
    atd = total / max(active_days, 1)
    max_dd = (tracker.equity_peak - tracker.equity_low) / max(tracker.equity_peak, 0.01) * 100

    max_consec = curr = 0
    for t in trades:
        if t["pnl"] < 0: curr += 1; max_consec = max(max_consec, curr)
        else: curr = 0

    tp_ct = len([t for t in trades if t["reason"]=="TP"])
    sl_ct = len([t for t in trades if t["reason"]=="SL"])
    tr_ct = len([t for t in trades if t["reason"]=="TRAIL"])

    logger.info("\n" + "=" * 70)
    logger.info("V19 MAX COMPOUNDING RESULTS")
    logger.info("=" * 70)
    logger.info(f"  Initial: ${INITIAL_BALANCE:.0f} → Final: ${fb:.0f} | P/L: ${total_pnl:+.0f} (+{total_pnl/INITIAL_BALANCE*100:.0f}%)")
    logger.info(f"  ALL-TIME LOW BALANCE: ${all_time_low_balance:.2f} (lowest point)")
    logger.info(f"  Max Drawdown from Peak: {max_dd:.1f}%")
    logger.info(f"  Trades: {total} (BUY:{buy_cnt} SELL:{sell_cnt}) | WR: {wr:.1f}% | PF: {pf:.2f}")
    logger.info(f"  Avg Win: ${gp/max(len(wins),1):.1f} | Avg Loss: ${gl/max(len(losses),1):.1f}")
    logger.info(f"  TP: {tp_ct} | SL: {sl_ct} | TRAIL saves: {tr_ct} | Max Cons Loss: {max_consec}")
    logger.info(f"  BUY P/L: ${buy_pnl:+.0f} | SELL P/L: ${sell_pnl:+.0f}")
    logger.info(f"  Days: {active_days} | Avg/day: {atd:.1f} | Runtime: {elapsed:.1f}s")
    logger.info(f"  Blocked: death={blocked.get('death',0)} low_score={blocked.get('low_score',0)} buy_ema={blocked.get('buy_ema',0)} vol={blocked.get('vol',0)}")
    logger.info("=" * 70)

    # COMPARISON
    logger.info("\n" + "=" * 85)
    logger.info("FULL COMPARISON: V15 → V16 → V17 → V19")
    logger.info("=" * 85)
    logger.info(f"{'Metric':<22s} {'V15':>10s} {'V16':>10s} {'V17':>10s} {'V19':>10s}")
    logger.info("-" * 62)
    logger.info(f"{'Trades':<22s} {'504':>10s} {'313':>10s} {'262':>10s} {total:>10d}")
    logger.info(f"{'Win Rate':<22s} {'42.3%':>10s} {'46.0%':>10s} {'49.2%':>10s} {wr:>9.1f}%")
    logger.info(f"{'Profit Factor':<22s} {'1.32':>10s} {'1.63':>10s} {'1.83':>10s} {pf:>10.2f}")
    logger.info(f"{'Final Balance':<22s} {'$1138':>10s} {'$1240':>10s} {'$1950':>10s} ${fb:>9.0f}")
    logger.info(f"{'Total P/L':<22s} {'+$838':>10s} {'+$940':>10s} {'+$1650':>10s} ${total_pnl:>+9.0f}")
    logger.info(f"{'Return %':<22s} {'+280%':>10s} {'+313%':>10s} {'+550%':>10s} {total_pnl/INITIAL_BALANCE*100:>+9.0f}%")
    logger.info(f"{'Lowest Balance':<22s} {'-$':>10s} {'-$':>10s} {'-$':>10s} ${all_time_low_balance:>9.0f}")
    logger.info(f"{'Max Drawdown':<22s} {'-':>10s} {'-':>10s} {'-':>10s} {max_dd:>9.1f}%")
    logger.info(f"{'Avg/day':<22s} {'12.3':>10s} {'10.1':>10s} {'10.5':>10s} {atd:>10.1f}")
    logger.info("=" * 85)


if __name__ == "__main__":
    run_v19()