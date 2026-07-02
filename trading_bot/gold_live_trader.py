"""
Gold Scalping V13 — LIVE TRADER for MT5 Demo
==============================================
Runs the V13 strategy on live XAUUSD M15 data via MT5.
Opens trades on your demo account when signals trigger.

USAGE:
  1. Open MetaTrader 5 on your PC (log in to demo)
  2. Make sure XAUUSD is in Market Watch
  3. Close the "MetaTrader 5" welcome/news window if it opens
  4. Run: python trading_bot/gold_live_trader.py
"""

import sys
import os
import time
from datetime import datetime, timedelta, timezone

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import MetaTrader5 as mt5
import pandas as pd
import numpy as np

from trading_bot.utils.logger import logger
from trading_bot.strategy.gold_scalping_strategy import GoldScalpingStrategy
from trading_bot.strategy.gold_volatility_filter import GoldVolatilityFilter
from trading_bot.risk.risk_manager import RiskManager
from trading_bot.config import Config
from trading_bot.indicators.technical_indicators import compute_all_indicators
from trading_bot.mt5.data_feed import get_candles


class GoldLiveTrader:
    def __init__(self):
        self.strategy = GoldScalpingStrategy()
        self.vol_filter = GoldVolatilityFilter()
        self.risk_manager = RiskManager(default_balance=300.0)

        self.strategy._max_trades_per_day = 30
        self.strategy._max_open_positions = 3

        self.open_positions = {}
        self.sl_cooldown_until = None
        self.consecutive_losses = 0
        self.loss_streak_halt_until = None
        self.daily_pnl = 0.0
        self.last_m15_candle_time = None
        self.trades_today = 0
        self.current_day = None

        self.symbol = "XAUUSD"
        self.lot = 0.02
        self.sl_atr = 10.0
        self.tp_atr = 2.5

        logger.info("GoldLiveTrader initialized.")

    def connect_mt5(self):
        """
        Connect to the ALREADY RUNNING MT5 terminal.
        This does NOT re-login — it just attaches to the existing terminal.
        """
        logger.info("Connecting to MT5 (attach to running terminal)...")

        # Initialize without login/password — just attach to running terminal
        if not mt5.initialize():
            logger.error(f"MT5 initialize failed. Error: {mt5.last_error()}")
            logger.error("Make sure MetaTrader 5 is OPEN and you are logged in.")
            return False

        # Check if terminal is actually connected
        info = mt5.terminal_info()
        if info is None:
            logger.error("MT5 terminal is running but not connected.")
            logger.error("Please log in to your demo account in MT5 first.")
            mt5.shutdown()
            return False

        # Get account info
        account = mt5.account_info()
        if account is None:
            logger.warning("Account info not available. Is MT5 logged in?")
        else:
            logger.info(f"✅ Connected! Account: {account.login} | "
                        f"Server: {account.server} | "
                        f"Balance: ${account.balance:.2f}")
            self.risk_manager.set_balance(account.balance)

        # Select XAUUSD
        if not mt5.symbol_select(self.symbol, True):
            logger.error(f"Failed to select {self.symbol}. Open Market Watch in MT5 and add XAUUSD.")
            mt5.shutdown()
            return False

        logger.info(f"Monitoring {self.symbol} M15...")
        logger.info(f"Lot: {self.lot} | SL: {self.sl_atr}x ATR | TP: {self.tp_atr}x ATR")
        logger.info("Press Ctrl+C to stop.\n")
        return True

    def run(self):
        logger.info("=" * 60)
        logger.info("GOLD SCALPING LIVE TRADER")
        logger.info("=" * 60)

        if not self.connect_mt5():
            logger.error("\n💡 FIX: 1) Close this terminal")
            logger.error("     2) Close and reopen MetaTrader 5")
            logger.error("     3) Log in to your demo account")
            logger.error("     4) Wait for MT5 to fully load")
            logger.error("     5) Run this script again")
            return

        try:
            while True:
                self._process_candle()
                time.sleep(5)
        except KeyboardInterrupt:
            logger.info("\nStopping live trader...")
        finally:
            mt5.shutdown()
            logger.info("Live trader stopped.")

    def _process_candle(self):
        m15_data = get_candles(self.symbol, "M15", 5)
        if m15_data is None or len(m15_data) < 3:
            return

        latest_time = m15_data.index[-1]
        if self.last_m15_candle_time == latest_time:
            return

        self.last_m15_candle_time = latest_time
        current_dt = latest_time.to_pydatetime().replace(tzinfo=timezone.utc)

        logger.info(f"\n{'='*50}")
        logger.info(f"NEW M15 CANDLE: {current_dt}")
        logger.info(f"{'='*50}")

        day_str = current_dt.strftime("%Y-%m-%d")
        if self.current_day != day_str:
            self.current_day = day_str
            self.trades_today = 0
            self.daily_pnl = 0.0
            self.strategy.reset_daily()
            self.risk_manager.reset_daily()
            logger.info(f"--- NEW DAY: {day_str} ---")

        if self._is_on_cooldown(current_dt):
            return

        if self.daily_pnl <= -15.0:
            logger.warning(f"Daily loss limit hit (-${abs(self.daily_pnl):.2f}). Stopping for day.")
            return

        m1_data = get_candles(self.symbol, "M1", 200)
        m5_data = get_candles(self.symbol, "M5", 100)
        m15_data = get_candles(self.symbol, "M15", 50)

        if m1_data is None or m5_data is None or m15_data is None:
            logger.warning("Failed to fetch data")
            return

        try:
            m1i = compute_all_indicators(m1_data)
            m5i = compute_all_indicators(m5_data)
            m15i = compute_all_indicators(m15_data)
        except Exception as e:
            logger.warning(f"Indicator error: {e}")
            return

        atr_val = float(m15i["atr"].iloc[-1]) if not m15i["atr"].empty else 7.0
        if not m15i["atr"].empty and len(m15i["atr"].dropna()) > 20:
            cur = float(m15i["atr"].iloc[-1])
            avg = float(m15i["atr"].iloc[-21:-1].mean())
            if avg > 0:
                ratio = cur / avg
                if ratio > 1.8 or ratio < 0.3:
                    logger.info(f"ATR ratio {ratio:.2f} — extreme, skip")
                    return

        regime = self._get_regime(m15i)

        sr = self.strategy.analyze(
            m1_indicators=m1i, m5_indicators=m5i, m15_indicators=m15i,
            m1_ohlcv=m1_data, m5_ohlcv=m5_data, m15_ohlcv=m15_data,
            news_context=None,
        )

        direction = sr.get("direction", "NONE")
        score = sr.get("setup_score", 0)
        bias = sr.get("bias", "neutral")
        reason = sr.get("reason", "")

        logger.info(f"Signal: dir={direction} score={score} bias={bias} regime={regime}")

        if direction == "NONE":
            return
        if regime == "neutral":
            logger.info(f"No clear trend — skip")
            return
        if (regime == "bullish" and direction != "BUY") or \
           (regime == "bearish" and direction != "SELL"):
            logger.info(f"{regime} trend — skip {direction}")
            return
        if score < 80 or score >= 90:
            logger.info(f"Score {score} not in range — skip")
            return

        try:
            vfr = self.vol_filter.analyze(
                m1_ohlcv=m1_data, m5_ohlcv=m5_data, m15_ohlcv=m15_data,
                m1_indicators=m1i, m5_indicators=m5i, m15_indicators=m15i,
            )
            if not vfr.get("trade_ok", True):
                logger.info(f"Vol filter blocked: {vfr.get('reason', '')}")
                return
        except Exception:
            pass

        open_pos = len(mt5.positions_get(symbol=self.symbol) or [])
        if open_pos >= 3:
            logger.info(f"Already {open_pos} open — limit reached")
            return

        can_trade, limit_reason = self.strategy.can_trade(open_pos)
        if not can_trade:
            logger.info(f"Can't trade: {limit_reason}")
            return

        self._execute_trade(direction, atr_val, current_dt, reason, score)

    def _get_regime(self, m15i):
        emas = m15i.get("emas", pd.DataFrame())
        if emas.empty or "EMA_20" not in emas.columns or len(emas) < 10:
            return "neutral"
        try:
            vals = emas["EMA_20"].dropna().values
            if len(vals) < 5:
                return "neutral"
            diff = float(vals[-1]) - float(vals[-5])
            if diff > 0:
                return "bullish"
            elif diff < 0:
                return "bearish"
            return "neutral"
        except Exception:
            return "neutral"

    def _execute_trade(self, direction, atr_val, current_dt, reason, score):
        tick = mt5.symbol_info_tick(self.symbol)
        if tick is None:
            logger.error("Failed to get tick")
            return

        price = tick.ask if direction == "BUY" else tick.bid
        sl_distance = atr_val * self.sl_atr
        tp_distance = atr_val * self.tp_atr

        if direction == "BUY":
            sl = round(price - sl_distance, 2)
            tp = round(price + tp_distance, 2)
            order_type = mt5.ORDER_TYPE_BUY
        else:
            sl = round(price + sl_distance, 2)
            tp = round(price - tp_distance, 2)
            order_type = mt5.ORDER_TYPE_SELL

        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": self.symbol,
            "volume": self.lot,
            "type": order_type,
            "price": price,
            "sl": sl,
            "tp": tp,
            "deviation": 20,
            "magic": 202406,
            "comment": f"V13_{direction}_S{score}",
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_IOC,
        }

        result = mt5.order_send(request)

        if result.retcode != mt5.TRADE_RETCODE_DONE:
            logger.error(f"❌ Order failed: {result.retcode} — {result.comment}")
            return

        logger.info(f"✅ {direction} OPENED | ${price:.2f} | SL: ${sl:.2f} | "
                    f"TP: ${tp:.2f} | Score: {score} | Lot: {self.lot}")

        self.trades_today += 1
        self.strategy.record_trade()

    def _is_on_cooldown(self, current_dt):
        if self.sl_cooldown_until and current_dt < self.sl_cooldown_until:
            remaining = (self.sl_cooldown_until - current_dt).seconds // 60
            logger.info(f"Cooldown: {remaining}min left")
            return True
        if self.loss_streak_halt_until and current_dt < self.loss_streak_halt_until:
            remaining = (self.loss_streak_halt_until - current_dt).seconds // 60
            logger.info(f"Halt: {remaining}min left")
            return True
        return False


if __name__ == "__main__":
    trader = GoldLiveTrader()
    trader.run()