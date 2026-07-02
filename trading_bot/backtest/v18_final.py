"""
V18 FINAL — Asymmetric Entries + Dynamic SL + M5 Confirmation
=============================================================

Improvements over V17:
  A. Asymmetric score thresholds: BUY >= 70, SELL >= 50
  B. BUY RSI zone: only enter BUY when M5 RSI < 45
  C. M5 candle confirmation: require M5 close in trade direction
  D. Dynamic SL width: 1.2x-2.0x ATR based on volatility
  E. Minimum ATR filter: skip trades when M5 ATR < $2.50
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
INITIAL_BALANCE = 300.0
SYMBOL = "XAUUSD"
MT5_LOGIN = Config.MT5_LOGIN
MT5_PASSWORD = Config.MT5_PASSWORD
MT5_SERVER = Config.MT5_SERVER

# --- V18 FILTERS ---
DEATH_ZONE_HOURS = {10, 11, 12, 13, 16, 17, 23}
SKIP_THURSDAYS = True
SKIP_MONDAYS = True
MAX_POSITIONS = 3
BUY_REQUIRES_RISING_EMA200 = True
TRAILING_SL_ENABLED = True
TRAIL_AFTER_ATR_MULT = 1.0
TRAIL_DISTANCE_ATR_MULT = 0.7
RISK_PERCENT = 2.0

# V18 NEW: Asymmetric thresholds
BUY_MIN_SCORE = 70
SELL_MIN_SCORE = 50
# V18 NEW: BUY RSI zone
BUY_MAX_RSI = 45
# V18 NEW: M5 candle confirmation
M5_CONFIRMATION_ENABLED = True
# V18 NEW: Dynamic SL
SL_MIN_ATR_MULT = 1.2
SL_MAX_ATR_MULT = 2.0
# V18 NEW: Min ATR
MIN_ATR_THRESHOLD = 2.50


class Tracker:
    def __init__(self, balance=300.0):
        self.balance = balance
        self.equity = balance
        self.equity_peak = balance
        self.equity_low = balance
        self.open_positions = []
        self.sl_cooldown_until = None
        self.consecutive_losses = 0
        self.loss_streak_halt_until = None
        self.daily_pnl = 0.0
        self.trades = []
        self.daily_trades = defaultdict(int)

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
                if direction == "BUY":
                    profit_move = price - entry
                else:
                    profit_move = entry - price
                if profit_move >= atr * TRAIL_AFTER_ATR_MULT:
                    pos["breakeven_moved"] = True
                    pos["sl"] = entry
            elif TRAILING_SL_ENABLED and pos.get("breakeven_moved", False):
                if direction == "BUY":
                    new_sl = price - atr * TRAIL_DISTANCE_ATR_MULT
                    if new_sl > current_sl + 0.5:
                        pos["sl"] = round(new_sl, 2)
                else:
                    new_sl = price + atr * TRAIL_DISTANCE_ATR_MULT
                    if new_sl < current_sl - 0.5:
                        pos["sl"] = round(new_sl, 2)

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
                    self.sl_cooldown_until = dt + timedelta(minutes=45)
                    self.consecutive_losses += 1
                    if self.consecutive_losses >= 3:
                        self.loss_streak_halt_until = dt + timedelta(hours=4)
                else:
                    self.consecutive_losses = 0
            else:
                remaining.append(pos)
        self.open_positions = remaining
        return closed

    def can_trade(self, dt):
        if self.sl_cooldown_until and dt < self.sl_cooldown_until: return False
        if self.loss_streak_halt_until and dt < self.loss_streak_halt_until: return False
        if self.daily_pnl <= -INITIAL_BALANCE * 0.05: return False
        return True

    def reset_daily(self):
        self.daily_pnl = 0.0

    def force_close_all(self, price, dt):
        for pos in self.open_positions:
            direction = pos.get("direction", "BUY"); lot = pos.get("lot", 0.01); pip_value = lot * 100
            pnl = (price - pos["entry"]) * pip_value if direction == "BUY" else (pos["entry"] - price) * pip_value
            self.trades.append({**pos, "pnl": round(pnl, 2), "reason": "EOD", "close_time": dt})
            self.balance += pnl
        self.open_positions = []

    def get_equity(self, price):
        unrealized = 0.0
        for pos in self.open_positions:
            lot = pos.get("lot", 0.01); pip_value = lot * 100
            unrealized += (price - pos["entry"]) * pip_value if pos.get("direction")=="BUY" else (pos["entry"]-price) * pip_value
        equity = self.balance + unrealized
        self.equity_peak = max(self.equity_peak, equity)
        self.equity_low = min(self.equity_low, equity)
        return equity


def fetch_mt5_data(symbol, days=180):
    logger.info(f"Fetching {days} days of {symbol} data from MT5...")
    if not mt5.initialize(login=MT5_LOGIN, password=MT5_PASSWORD, server=MT5_SERVER):
        logger.error(f"MT5 init failed: {mt5.last_error()}"); return None
    if not mt5.symbol_select(symbol, True):
        logger.error(f"Symbol {symbol} not available"); mt5.shutdown(); return None

    tf_map = {"M1": mt5.TIMEFRAME_M1, "M5": mt5.TIMEFRAME_M5, "M15": mt5.TIMEFRAME_M15}
    data = {}
    end_dt = datetime.now(timezone.utc)
    start_dt = end_dt - timedelta(days=days)

    for tf_name, tf_val in tf_map.items():
        logger.info(f"  Fetching {tf_name}...")
        rates = mt5.copy_rates_range(symbol, tf_val, start_dt, end_dt)
        if rates is None or len(rates) == 0:
            if tf_name == "M1":
                rates = mt5.copy_rates_range(symbol, tf_val, end_dt - timedelta(days=60), end_dt)
            if rates is None or len(rates) == 0: mt5.shutdown(); return None
        df = pd.DataFrame(rates); df["time"] = pd.to_datetime(df["time"], unit="s", utc=True); df.set_index("time", inplace=True)
        df["session"] = df.index.map(lambda dt: "weekend" if dt.weekday()>=5 else ("london" if 8<=dt.hour<13 else ("overlap" if 13<=dt.hour<17 else ("new_york" if 17<=dt.hour<22 else "asian"))))
        data[tf_name] = df
        logger.info(f"    Got {len(df)} candles")
    mt5.shutdown()
    return data


def _ema200_slope(window_15m) -> bool:
    try:
        close = window_15m["close"].values
        if len(close) < 200: return False
        ema200 = pd.Series(close).ewm(span=200, adjust=False).mean().values
        return len(ema200) >= 10 and float(ema200[-1]) > float(ema200[-10])
    except: return False


def _m5_confirm(m5_window, direction) -> bool:
    """Check if last 2 M5 candles closed in trade direction."""
    if len(m5_window) < 3: return False
    closes = m5_window["close"].values
    last_change = float(closes[-1]) - float(closes[-2])
    prev_change = float(closes[-2]) - float(closes[-3])
    if direction == "BUY":
        return last_change > 0 and prev_change > 0
    elif direction == "SELL":
        return last_change < 0 and prev_change < 0
    return False


def _get_m5_rsi(m5_indicators) -> float:
    try:
        rsi = m5_indicators.get("rsi", pd.Series(dtype=float))
        return float(rsi.iloc[-1]) if not rsi.empty else 50.0
    except: return 50.0


def run_v18():
    logger.info("=" * 70)
    logger.info("V18 FINAL — ASYMMETRIC + DYNAMIC SL + M5 CONFIRMATION")
    logger.info(f"BUY: score>={BUY_MIN_SCORE} RSI<{BUY_MAX_RSI} | SELL: score>={SELL_MIN_SCORE}")
    logger.info(f"Dynamic SL: {SL_MIN_ATR_MULT}-{SL_MAX_ATR_MULT}x ATR | Min ATR=${MIN_ATR_THRESHOLD}")
    logger.info("=" * 70)

    data = fetch_mt5_data(SYMBOL, days=DAYS)
    if data is None: logger.critical("No data."); return

    m1_df, m5_df, m15_df = data["M1"], data["M5"], data["M15"]
    if len(m15_df) < 500: logger.critical("Insufficient candles."); return

    strategy = GoldScalpingStrategy()
    strategy._max_trades_per_day = 100; strategy._max_open_positions = 10
    vol_filter = GoldVolatilityFilter()
    deepseek = MockDeepSeekClient()
    tracker = Tracker(balance=INITIAL_BALANCE)
    min_warmup = 200; total = len(m15_df)
    blocked = defaultdict(int)
    start_time = time.time()
    last_display_pct = 0

    for idx in range(min_warmup, total):
        dt = m15_df.index[idx]
        price_m15 = float(m15_df["close"].iloc[idx])
        session = m15_df["session"].iloc[idx]

        pct = (idx - min_warmup) / (total - min_warmup) * 100
        if int(pct) > last_display_pct and int(pct) % 10 == 0:
            elapsed = time.time() - start_time
            logger.info(f"  {int(pct)}% | Bal:${tracker.balance:.0f} | T:{len(tracker.trades)} | Open:{tracker.count_open()}")

        last_display_pct = int(pct)

        if session == "weekend": tracker.force_close_all(price_m15, dt); continue
        if dt.hour == 0 and dt.minute < 15: tracker.reset_daily()
        if dt.hour in DEATH_ZONE_HOURS: blocked["death"] += 1; continue
        if SKIP_THURSDAYS and dt.weekday() == 3: blocked["thu"] += 1; continue
        if SKIP_MONDAYS and dt.weekday() == 0: blocked["mon"] += 1; continue

        m1_window = m1_df[m1_df.index <= dt].iloc[-500:] if len(m1_df[m1_df.index <= dt]) >= 500 else m1_df[m1_df.index <= dt]
        m5_window = m5_df[m5_df.index <= dt].iloc[-500:] if len(m5_df[m5_df.index <= dt]) >= 500 else m5_df[m5_df.index <= dt]
        m15_window = m15_df[m15_df.index <= dt].iloc[idx + 1 - min(idx + 1, 500):idx + 1]

        if len(m1_window) < 50 or len(m5_window) < 50 or len(m15_window) < 50: continue

        try:
            m1_ind = compute_all_indicators(m1_window)
            m5_ind = compute_all_indicators(m5_window)
            m15_ind = compute_all_indicators(m15_window)
        except: continue

        try:
            atr_series = m5_ind.get("atr", pd.Series(dtype=float))
            atr_val = float(atr_series.iloc[-1]) if not atr_series.empty else 3.5
        except: atr_val = 3.5

        tracker.update_all(price_m15, dt, atr_val=atr_val)
        if not tracker.can_trade(dt): continue
        if tracker.count_open() >= MAX_POSITIONS: continue

        # V18 NEW: Min ATR filter
        if atr_val < MIN_ATR_THRESHOLD: blocked["min_atr"] += 1; continue

        try:
            result = strategy.analyze(
                m1_indicators=m1_ind, m5_indicators=m5_ind, m15_indicators=m15_ind,
                m1_ohlcv=m1_window, m5_ohlcv=m5_window, m15_ohlcv=m15_window, news_context=None)
        except: continue

        direction = result.get("direction", "NONE")
        score = result.get("setup_score", 0)

        # V18 NEW: Asymmetric score thresholds
        if direction == "NONE": continue
        if direction == "BUY" and score < BUY_MIN_SCORE: blocked["buy_score"] += 1; continue
        if direction == "SELL" and score < SELL_MIN_SCORE: blocked["sell_score"] += 1; continue

        # V18 NEW: BUY RSI zone
        if direction == "BUY":
            rsi_val = _get_m5_rsi(m5_ind)
            if rsi_val > BUY_MAX_RSI: blocked["buy_rsi"] += 1; continue

        # BUY requires rising EMA200
        if BUY_REQUIRES_RISING_EMA200 and direction == "BUY":
            if not _ema200_slope(m15_window): blocked["buy_ema"] += 1; continue

        # V18 NEW: M5 candle confirmation
        if M5_CONFIRMATION_ENABLED:
            if not _m5_confirm(m5_window, direction): blocked["m5_confirm"] += 1; continue

        try:
            vol_result = vol_filter.analyze(
                m1_ohlcv=m1_window, m5_ohlcv=m5_window, m15_ohlcv=m15_window,
                m1_indicators=m1_ind, m5_indicators=m5_ind, m15_indicators=m15_ind)
            if not vol_result.get("trade_ok", False): blocked["vol"] += 1; continue
        except: continue

        entry = price_m15

        # V18 NEW: Dynamic SL width based on ATR regime
        atr_ratio = vol_result.get("atr_ratio", 1.0)
        if atr_ratio > 1.5:
            sl_mult = SL_MIN_ATR_MULT  # High vol = tighter SL
        elif atr_ratio < 0.8:
            sl_mult = SL_MAX_ATR_MULT  # Low vol = wider SL
        else:
            sl_mult = 1.5  # Normal
        tp_mult = sl_mult * 2.0  # Keep R:R ratio

        sl_dist = atr_val * sl_mult
        tp_dist = atr_val * tp_mult

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

    # REPORT
    trades = tracker.trades
    total = len(trades)
    wins = [t for t in trades if t["pnl"] > 0]
    losses = [t for t in trades if t["pnl"] < 0]
    breakeven = [t for t in trades if t["pnl"] == 0]
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

    max_consec = curr = 0
    for t in trades:
        if t["pnl"] < 0: curr += 1; max_consec = max(max_consec, curr)
        else: curr = 0

    tp_ct = len([t for t in trades if t["reason"]=="TP"])
    sl_ct = len([t for t in trades if t["reason"]=="SL"])
    tr_ct = len([t for t in trades if t["reason"]=="TRAIL"])

    logger.info("\n" + "=" * 70)
    logger.info("V18 FINAL RESULTS")
    logger.info("=" * 70)
    logger.info(f"  Initial: ${INITIAL_BALANCE:.0f} → Final: ${fb:.0f} | P/L: ${total_pnl:+.0f} (+{total_pnl/INITIAL_BALANCE*100:.0f}%)")
    logger.info(f"  Trades: {total} (BUY:{buy_cnt} SELL:{sell_cnt}) | WR: {wr:.1f}% | PF: {pf:.2f}")
    logger.info(f"  Avg Win: ${gp/max(len(wins),1):.1f} | Avg Loss: ${gl/max(len(losses),1):.1f}")
    logger.info(f"  TP: {tp_ct}/{tr_ct} TRAIL | SL: {sl_ct} | Max Cons Loss: {max_consec}")
    logger.info(f"  BUY P/L: ${buy_pnl:+.0f} | SELL P/L: ${sell_pnl:+.0f}")
    logger.info(f"  Days: {active_days} | Avg/day: {atd:.1f} | Runtime: {elapsed:.1f}s")
    logger.info(f"  Blocked: death={blocked.get('death',0)} mon={blocked.get('mon',0)} thu={blocked.get('thu',0)}")
    logger.info(f"  Blocked: buy_score={blocked.get('buy_score',0)} sell_score={blocked.get('sell_score',0)}")
    logger.info(f"  Blocked: buy_rsi={blocked.get('buy_rsi',0)} buy_ema={blocked.get('buy_ema',0)}")
    logger.info(f"  Blocked: m5_confirm={blocked.get('m5_confirm',0)} min_atr={blocked.get('min_atr',0)} vol={blocked.get('vol',0)}")
    logger.info("=" * 70)

    # COMPARISON
    logger.info("\n" + "=" * 80)
    logger.info("V15 → V16 → V17 → V18 COMPARISON")
    logger.info("=" * 80)
    logger.info(f"{'Metric':<22s} {'V15':>10s} {'V16':>10s} {'V17':>10s} {'V18':>10s}")
    logger.info("-" * 62)
    logger.info(f"{'Trades':<22s} {'504':>10s} {'313':>10s} {'262':>10s} {total:>10d}")
    logger.info(f"{'Win Rate':<22s} {'42.3%':>10s} {'46.0%':>10s} {'49.2%':>10s} {wr:>9.1f}%")
    logger.info(f"{'Profit Factor':<22s} {'1.32':>10s} {'1.63':>10s} {'1.83':>10s} {pf:>10.2f}")
    logger.info(f"{'Final Balance':<22s} {'$1138':>10s} {'$1240':>10s} {'$1950':>10s} ${fb:>9.0f}")
    logger.info(f"{'Total P/L':<22s} {'+$838':>10s} {'+$940':>10s} {'+$1650':>10s} ${total_pnl:>+9.0f}")
    logger.info(f"{'Return %':<22s} {'+280%':>10s} {'+313%':>10s} {'+550%':>10s} {total_pnl/INITIAL_BALANCE*100:>+9.0f}%")
    logger.info(f"{'BUY P/L':<22s} {'+$121':>10s} {'+$62':>10s} {'+$109':>10s} ${buy_pnl:>+9.0f}")
    logger.info(f"{'SELL P/L':<22s} {'+$718':>10s} {'+$878':>10s} {'+$1541':>10s} ${sell_pnl:>+9.0f}")
    logger.info(f"{'Avg/day':<22s} {'12.3':>10s} {'10.1':>10s} {'10.5':>10s} {atd:>10.1f}")
    logger.info("=" * 80)

    # Save
    tf = os.path.join(os.path.dirname(__file__), "..", "..", "logs", "v18_trades.json")
    os.makedirs(os.path.dirname(tf), exist_ok=True)
    with open(tf, "w") as f:
        ser = []
        for t in trades:
            st = dict(t)
            for k in ["open_time", "close_time"]:
                if k in st and hasattr(st[k], "isoformat"): st[k] = st[k].isoformat()
            ser.append(st)
        json.dump(ser, f, indent=2, default=str)


if __name__ == "__main__":
    run_v18()