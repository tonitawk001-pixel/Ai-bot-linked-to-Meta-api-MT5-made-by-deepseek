"""
V15 — 6-Month Backtest Using REAL MT5 Historical Data
======================================================

Fetches real XAUUSD M1/M5/M15 data from MT5 for the last 180 days,
runs the full GoldScalpingStrategy pipeline candle-by-candle,
and reports: total trades, win rate, P/L, max drawdown, profit factor.

Uses mock AI/News (real API calls would be too slow for 6 months of candles).
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

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
DAYS = 180  # 6 months
INITIAL_BALANCE = 300.0
SYMBOL = "XAUUSD"
MT5_LOGIN = Config.MT5_LOGIN
MT5_PASSWORD = Config.MT5_PASSWORD
MT5_SERVER = Config.MT5_SERVER

# ---------------------------------------------------------------------------
# Tracker (same logic as v14)
# ---------------------------------------------------------------------------
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
            "entry": entry, "tp": tp, "sl": sl,
            "direction": direction, "lot": lot,
            "open_time": dt,
        })

    def update_all(self, price, dt):
        """Check SL/TP hits. Returns list of closed trades."""
        closed, remaining = [], []
        for pos in self.open_positions:
            hit = False; pnl = 0.0; reason = ""
            direction = pos.get("direction", "BUY")
            lot = pos.get("lot", 0.01)
            # XAUUSD: 1 lot = 100 oz, 1 point = $100 per lot
            # 0.01 lot = $1 per point
            pip_value = lot * 100  # $ per 1.0 price move

            if direction == "BUY":
                if pos["tp"] and price >= pos["tp"]:
                    pnl = (pos["tp"] - pos["entry"]) * pip_value
                    reason = "TP"; hit = True
                elif pos["sl"] and price <= pos["sl"]:
                    pnl = (pos["sl"] - pos["entry"]) * pip_value
                    reason = "SL"; hit = True
            else:  # SELL
                if pos["tp"] and price <= pos["tp"]:
                    pnl = (pos["entry"] - pos["tp"]) * pip_value
                    reason = "TP"; hit = True
                elif pos["sl"] and price >= pos["sl"]:
                    pnl = (pos["entry"] - pos["sl"]) * pip_value
                    reason = "SL"; hit = True

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
        """Check if trading is allowed (cooldowns, daily limits)."""
        if self.sl_cooldown_until and dt < self.sl_cooldown_until:
            return False
        if self.loss_streak_halt_until and dt < self.loss_streak_halt_until:
            return False
        # Daily loss limit: -5% of balance
        if self.daily_pnl <= -INITIAL_BALANCE * 0.05:
            return False
        return True

    def reset_daily(self):
        self.daily_pnl = 0.0

    def force_close_all(self, price, dt):
        """Close all positions at current price."""
        closed = []
        for pos in self.open_positions:
            direction = pos.get("direction", "BUY")
            lot = pos.get("lot", 0.01)
            pip_value = lot * 100
            if direction == "BUY":
                pnl = (price - pos["entry"]) * pip_value
            else:
                pnl = (pos["entry"] - price) * pip_value
            closed.append({**pos, "pnl": round(pnl, 2), "reason": "EOD", "close_time": dt})
            self.trades.append({**pos, "pnl": round(pnl, 2), "reason": "EOD", "close_time": dt})
            self.balance += pnl
        self.open_positions = []
        return closed

    def get_equity(self, price):
        """Calculate current equity including unrealized P/L."""
        unrealized = 0.0
        for pos in self.open_positions:
            direction = pos.get("direction", "BUY")
            lot = pos.get("lot", 0.01)
            pip_value = lot * 100
            if direction == "BUY":
                unrealized += (price - pos["entry"]) * pip_value
            else:
                unrealized += (pos["entry"] - price) * pip_value
        equity = self.balance + unrealized
        if equity > self.equity_peak:
            self.equity_peak = equity
        if equity < self.equity_low:
            self.equity_low = equity
        return equity


# ---------------------------------------------------------------------------
# Fetch MT5 Data
# ---------------------------------------------------------------------------
def fetch_mt5_data(symbol: str, days: int = 180) -> dict:
    """Fetch M1, M5, M15 data from MT5 for the specified number of days."""
    logger.info(f"Fetching {days} days of {symbol} data from MT5...")

    # Initialize MT5
    if not mt5.initialize(login=MT5_LOGIN, password=MT5_PASSWORD, server=MT5_SERVER):
        logger.error(f"MT5 init failed: {mt5.last_error()}")
        return None

    # Ensure symbol is visible
    if not mt5.symbol_select(symbol, True):
        logger.error(f"Symbol {symbol} not available")
        mt5.shutdown()
        return None

    # Timeframe mapping
    tf_map = {
        "M1": mt5.TIMEFRAME_M1,
        "M5": mt5.TIMEFRAME_M5,
        "M15": mt5.TIMEFRAME_M15,
    }

    data = {}
    end_dt = datetime.now(timezone.utc)
    start_dt = end_dt - timedelta(days=days)

    for tf_name, tf_val in tf_map.items():
        logger.info(f"  Fetching {tf_name} ({start_dt.date()} to {end_dt.date()})...")
        rates = mt5.copy_rates_range(symbol, tf_val, start_dt, end_dt)
        if rates is None or len(rates) == 0:
            error = mt5.last_error()
            logger.error(f"  Failed to fetch {tf_name}: {error}")
            # 180 days of M1 is ~260k candles — some brokers limit to ~100k
            # Fall back to fetching in chunks or just the available data
            if tf_name == "M1":
                logger.warning("  M1 data may exceed broker limit — retrying with shorter range...")
                rates = mt5.copy_rates_range(symbol, tf_val, end_dt - timedelta(days=60), end_dt)
                if rates is None or len(rates) == 0:
                    logger.error(f"  M1 fallback also failed: {mt5.last_error()}")
            if rates is None or len(rates) == 0:
                mt5.shutdown()
                return None

        df = pd.DataFrame(rates)
        df["time"] = pd.to_datetime(df["time"], unit="s", utc=True)
        df.set_index("time", inplace=True)

        # Add session labels
        df = _add_session_labels(df)

        data[tf_name] = df
        logger.info(f"    Got {len(df)} candles ({df.index[0]} to {df.index[-1]})")

    mt5.shutdown()
    logger.info("MT5 data fetch complete.")
    return data


def _add_session_labels(df: pd.DataFrame) -> pd.DataFrame:
    """Add session labels to dataframe."""
    def get_session(dt):
        h = dt.hour
        w = dt.weekday()
        if w >= 5:
            return "weekend"
        if 8 <= h < 13:
            return "london"
        elif 13 <= h < 17:
            return "overlap"
        elif 17 <= h < 22:
            return "new_york"
        elif 0 <= h < 8 or h >= 22:
            return "asian"
        return "transition"

    df["session"] = df.index.map(get_session)
    return df


# ---------------------------------------------------------------------------
# Backtest Engine
# ---------------------------------------------------------------------------
def run_v15_six_month():
    logger.info("=" * 70)
    logger.info("V15 — 6-MONTH XAUUSD BACKTEST WITH REAL MT5 DATA")
    logger.info(f"Account: {MT5_LOGIN} | Server: {MT5_SERVER}")
    logger.info(f"Initial balance: ${INITIAL_BALANCE:.2f} | Period: {DAYS} days")
    logger.info("=" * 70)

    # Fetch data
    data = fetch_mt5_data(SYMBOL, days=DAYS)
    if data is None:
        logger.critical("Failed to fetch MT5 data. Aborting.")
        return

    m1_df = data["M1"]
    m5_df = data["M5"]
    m15_df = data["M15"]

    logger.info(f"M15 candles: {len(m15_df)}")

    if len(m15_df) < 500:
        logger.critical(f"Only {len(m15_df)} M15 candles — insufficient for 6-month backtest.")
        return

    # Initialize components
    strategy = GoldScalpingStrategy()
    strategy._max_trades_per_day = 100
    strategy._max_open_positions = 10

    vol_filter = GoldVolatilityFilter()
    risk_manager = RiskManager(default_balance=INITIAL_BALANCE)
    deepseek = MockDeepSeekClient()

    # Generate mock news events aligned with data dates
    start_dt = m15_df.index[0]
    news_events = _generate_news_events(start_dt, DAYS)
    news_agg = MockNewsAggregator(news_events, start_dt)

    tracker = Tracker(balance=INITIAL_BALANCE)

    # Build M15 indicator cache (compute once for warmup, then rolling)
    # We'll compute indicators on a rolling window
    min_warmup = 200  # Need at least 200 candles for EMA200

    logger.info(f"Starting candle-by-candle walk-forward (M15, {len(m15_df)} candles)...")
    start_time = time.time()

    last_display_pct = 0
    total = len(m15_df)

    for idx in range(min_warmup, total):
        dt = m15_df.index[idx]
        price_m15 = float(m15_df["close"].iloc[idx])
        session = m15_df["session"].iloc[idx]

        # Progress indicator
        pct = (idx - min_warmup) / (total - min_warmup) * 100
        if int(pct) > last_display_pct and int(pct) % 10 == 0:
            elapsed = time.time() - start_time
            eta = elapsed / max((idx - min_warmup) / (total - min_warmup), 0.001) - elapsed
            logger.info(
                f"  {int(pct)}% | Candle {idx}/{total} | "
                f"Balance: ${tracker.balance:.2f} | "
                f"Trades: {len(tracker.trades)} | "
                f"Open: {tracker.count_open()} | "
                f"Elapsed: {elapsed:.0f}s | ETA: {eta:.0f}s"
            )
            last_display_pct = int(pct)

        # Skip weekends
        if session == "weekend":
            tracker.force_close_all(price_m15, dt)
            continue

        # Daily reset at midnight UTC
        if dt.hour == 0 and dt.minute < 15:
            tracker.reset_daily()

        # Get intraday timestamps for M1/M5 windows
        # Find the corresponding indices in M1 and M5 dataframes
        m1_window = m1_df[m1_df.index <= dt].iloc[-500:] if len(m1_df[m1_df.index <= dt]) >= 500 else m1_df[m1_df.index <= dt]
        m5_window = m5_df[m5_df.index <= dt].iloc[-500:] if len(m5_df[m5_df.index <= dt]) >= 500 else m5_df[m5_df.index <= dt]
        m15_window = m15_df[m15_df.index <= dt].iloc[idx + 1 - min(idx + 1, 500):idx + 1]

        if len(m1_window) < 50 or len(m5_window) < 50 or len(m15_window) < 50:
            continue

        # Compute indicators on all timeframes
        try:
            m1_ind = compute_all_indicators(m1_window)
            m5_ind = compute_all_indicators(m5_window)
            m15_ind = compute_all_indicators(m15_window)
        except Exception:
            continue

        # Check position closures
        closed = tracker.update_all(price_m15, dt)

        # Check if we can trade
        if not tracker.can_trade(dt):
            continue

        # Max positions check
        if tracker.count_open() >= 3:
            continue

        # Run strategy
        try:
            strategy_result = strategy.analyze(
                m1_indicators=m1_ind,
                m5_indicators=m5_ind,
                m15_indicators=m15_ind,
                m1_ohlcv=m1_window,
                m5_ohlcv=m5_window,
                m15_ohlcv=m15_window,
                news_context=None,  # Skip news for speed in backtest
            )
        except Exception:
            continue

        direction = strategy_result.get("direction", "NONE")
        setup_score = strategy_result.get("setup_score", 0)

        if direction == "NONE" or setup_score < 30:
            continue

        # Run volatility filter
        try:
            vol_result = vol_filter.analyze(
                m1_ohlcv=m1_window,
                m5_ohlcv=m5_window,
                m15_ohlcv=m15_window,
                m1_indicators=m1_ind,
                m5_indicators=m5_ind,
                m15_indicators=m15_ind,
            )
            if not vol_result.get("trade_ok", False):
                continue
        except Exception:
            continue

        # Compute ATR for SL/TP
        try:
            atr_series = m5_ind.get("atr", pd.Series(dtype=float))
            atr_val = float(atr_series.iloc[-1]) if not atr_series.empty else 3.5
        except Exception:
            atr_val = 3.5

        # Determine entry price: use current M15 close
        entry = price_m15

        # Compute SL/TP
        sl_dist = atr_val * 1.5
        tp_dist = atr_val * 3.0

        if direction == "BUY":
            sl = round(entry - sl_dist, 2)
            tp = round(entry + tp_dist, 2)
        else:
            sl = round(entry + sl_dist, 2)
            tp = round(entry - tp_dist, 2)

        # Lot size: 2% risk of $300 = $6 risk
        # risk_amount / (sl_dist * 100) = lot
        risk_amount = INITIAL_BALANCE * 0.02
        lot = risk_amount / (sl_dist * 100) if sl_dist > 0 else 0.01
        lot = max(0.01, min(lot, 10.0))
        lot = round(lot, 2)

        # Open position
        tracker.open_pos(entry, tp, sl, dt, direction=direction, lot=lot)
        tracker.daily_trades[dt.date()] += 1

    # Close any remaining positions at final price
    final_price = float(m15_df["close"].iloc[-1])
    final_dt = m15_df.index[-1]
    tracker.force_close_all(final_price, final_dt)

    elapsed = time.time() - start_time
    logger.info(f"Backtest complete in {elapsed:.1f}s")

    # -------------------------------------------------------------------
    # Report
    # -------------------------------------------------------------------
    trades = tracker.trades
    total_trades = len(trades)
    wins = [t for t in trades if t["pnl"] > 0]
    losses = [t for t in trades if t["pnl"] < 0]
    breakeven = [t for t in trades if t["pnl"] == 0]

    win_rate = len(wins) / max(total_trades, 1) * 100
    gross_profit = sum(t["pnl"] for t in wins)
    gross_loss = abs(sum(t["pnl"] for t in losses))
    profit_factor = gross_profit / max(gross_loss, 0.01)
    total_pnl = sum(t["pnl"] for t in trades)
    final_balance = INITIAL_BALANCE + total_pnl
    max_dd = (tracker.equity_peak - tracker.equity_low) / max(tracker.equity_peak, 0.01) * 100

    # P/L by reason
    tp_trades = [t for t in trades if t["reason"] == "TP"]
    sl_trades = [t for t in trades if t["reason"] == "SL"]
    eod_trades = [t for t in trades if t["reason"] == "EOD"]
    tp_total = sum(t["pnl"] for t in tp_trades)
    sl_total = sum(t["pnl"] for t in sl_trades)

    # Max consecutive losses
    max_consec = 0
    curr_streak = 0
    for t in trades:
        if t["pnl"] < 0:
            curr_streak += 1
            max_consec = max(max_consec, curr_streak)
        else:
            curr_streak = 0

    # P/L by direction
    buy_pnl = sum(t["pnl"] for t in trades if t.get("direction") == "BUY")
    sell_pnl = sum(t["pnl"] for t in trades if t.get("direction") == "SELL")

    # Active trading days
    active_days = len(tracker.daily_trades)

    # Average trades per active day
    avg_trades_per_day = total_trades / max(active_days, 1)

    logger.info("\n" + "=" * 70)
    logger.info("V15 — 6-MONTH XAUUSD BACKTEST RESULTS")
    logger.info("=" * 70)
    logger.info(f"  Period: {DAYS} days ({m15_df.index[0].date()} to {m15_df.index[-1].date()})")
    logger.info(f"  Total candles: {total}")
    logger.info(f"  ──────────────────────────────────────")
    logger.info(f"  Initial balance: ${INITIAL_BALANCE:.2f}")
    logger.info(f"  Final balance:   ${final_balance:.2f}")
    logger.info(f"  Total P/L:       ${total_pnl:+.2f}")
    logger.info(f"  Return:          {(total_pnl / INITIAL_BALANCE * 100):+.1f}%")
    logger.info(f"  ──────────────────────────────────────")
    logger.info(f"  Total trades:    {total_trades}")
    logger.info(f"  Wins:            {len(wins)}")
    logger.info(f"  Losses:          {len(losses)}")
    logger.info(f"  Breakeven:       {len(breakeven)}")
    logger.info(f"  Win rate:        {win_rate:.1f}%")
    logger.info(f"  ──────────────────────────────────────")
    logger.info(f"  Profit factor:   {profit_factor:.2f}")
    logger.info(f"  Average win:     ${(gross_profit / max(len(wins), 1)):.2f}")
    logger.info(f"  Average loss:    ${(gross_loss / max(len(losses), 1)):.2f}")
    logger.info(f"  Max consec losses: {max_consec}")
    logger.info(f"  ──────────────────────────────────────")
    logger.info(f"  Max drawdown:    {max_dd:.1f}%")
    logger.info(f"  Equity peak:     ${tracker.equity_peak:.2f}")
    logger.info(f"  Equity low:      ${tracker.equity_low:.2f}")
    logger.info(f"  ──────────────────────────────────────")
    logger.info(f"  TP trades:       {len(tp_trades)} (${tp_total:+.2f})")
    logger.info(f"  SL trades:       {len(sl_trades)} (${sl_total:+.2f})")
    logger.info(f"  EOD closes:      {len(eod_trades)}")
    logger.info(f"  ──────────────────────────────────────")
    logger.info(f"  BUY P/L:         ${buy_pnl:+.2f}")
    logger.info(f"  SELL P/L:        ${sell_pnl:+.2f}")
    logger.info(f"  ──────────────────────────────────────")
    logger.info(f"  Active days:     {active_days}")
    logger.info(f"  Avg trades/day:  {avg_trades_per_day:.1f}")
    logger.info(f"  Runtime:         {elapsed:.1f}s")
    logger.info("=" * 70)

    # Trade breakdown by reason
    if sl_trades:
        logger.info("\nLast 5 losing trades (SL):")
        for t in sl_trades[-5:]:
            logger.info(f"  {t['open_time']} {t['direction']} entry={t['entry']} sl={t['sl']} pnl=${t['pnl']} reason={t['reason']}")

    if tp_trades:
        logger.info("\nLast 5 winning trades (TP):")
        for t in tp_trades[-5:]:
            logger.info(f"  {t['open_time']} {t['direction']} entry={t['entry']} tp={t['tp']} pnl=${t['pnl']} reason={t['reason']}")

    # Save trades to JSON for later analysis
    trades_file = os.path.join(os.path.dirname(__file__), "..", "..", "logs", "v15_trades.json")
    os.makedirs(os.path.dirname(trades_file), exist_ok=True)
    with open(trades_file, "w") as f:
        # Convert datetime keys to strings
        serializable = []
        for t in trades:
            st = dict(t)
            for key in ["open_time", "close_time"]:
                if key in st and hasattr(st[key], "isoformat"):
                    st[key] = st[key].isoformat()
            serializable.append(st)
        json.dump(serializable, f, indent=2, default=str)
    logger.info(f"\nFull trade log saved to: {trades_file}")

    return {
        "total_trades": total_trades,
        "win_rate": round(win_rate, 1),
        "total_pnl": round(total_pnl, 2),
        "final_balance": round(final_balance, 2),
        "profit_factor": round(profit_factor, 2),
        "max_drawdown_pct": round(max_dd, 1),
        "max_consecutive_losses": max_consec,
    }


def _generate_news_events(start_dt, days):
    """Generate mock news events spread across the period."""
    events = {}
    event_specs = [
        (30, "NFP", 4.0),
        (60, "CPI", 3.5),
        (90, "FOMC", 5.0),
        (105, "NFP", 4.0),
        (135, "CPI", 3.5),
        (150, "NFP", 4.0),
        (165, "FOMC", 5.0),
    ]
    for off, name, mult in event_specs:
        if off < days:
            dt = start_dt + timedelta(days=off)
            events[dt.replace(hour=12, minute=30)] = {"name": name, "vol_mult": mult}
    return events


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    run_v15_six_month()