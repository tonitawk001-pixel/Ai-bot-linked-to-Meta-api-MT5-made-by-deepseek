"""
Gold Scalping Bot — Comprehensive Backtest & Validation Suite.

Generates 6 months of realistic XAUUSD data and runs the full pipeline
to evaluate performance across sessions, volatility regimes, and news events.

This is STRICTLY VALIDATION — no code changes, no optimizations.
"""

import json
import math
import random
from datetime import datetime, timedelta, timezone
from collections import defaultdict

import numpy as np
import pandas as pd

from trading_bot.utils.logger import logger
from trading_bot.strategy.gold_scalping_strategy import GoldScalpingStrategy
from trading_bot.strategy.gold_volatility_filter import GoldVolatilityFilter
from trading_bot.risk.risk_manager import RiskManager
from trading_bot.config import Config


# ---------------------------------------------------------------------------
# Realistic XAUUSD Mock Data Generator
# ---------------------------------------------------------------------------

class XAUUSDDataGenerator:
    """
    Generates 6 months of realistic XAUUSD M1/M5/M15 OHLCV data.

    Models:
      - Session volatility (Asian=low, London=medium, NY=high, Overlap=highest)
      - Trend regimes (strong bull, strong bear, sideways, choppy)
      - ATR variations (spikes, flat periods)
      - Spread widening during news events
      - Realistic price action (trends + mean reversion + noise)
    """

    # Approximate XAUUSD price range and typical moves
    BASE_PRICE = 2350.0
    TYPICAL_ATR_M15 = 3.5  # Typical $3.50 ATR on M15
    SESSION_VOLATILITY = {
        "asian": 0.5,
        "london": 1.0,
        "new_york": 1.1,
        "overlap": 1.6,
        "transition": 0.6,
    }
    SPREAD_BASE = 5  # 0.5 pips in MT5 units (50 = 0.5 pip for 5-digit)

    # News events (dates for CPI, FOMC, NFP over 6 months)
    NEWS_EVENTS = [
        # (day_offset_from_start, event_name, hour_utc, volatility_multiplier, spread_multiplier)
        (30, "NFP", 12, 4.0, 4.0),
        (60, "CPI", 12, 3.5, 3.5),
        (90, "FOMC", 18, 5.0, 5.0),
        (105, "NFP", 12, 4.0, 4.0),
        (135, "CPI", 12, 3.5, 3.5),
        (150, "NFP", 12, 4.0, 4.0),
        (165, "FOMC", 18, 5.0, 5.0),
    ]

    def __init__(self, days: int = 180, seed: int = 42):
        self.days = days
        self.seed = seed
        random.seed(seed)
        np.random.seed(seed)

    def generate(self) -> dict:
        """Generate M1, M5, M15 data for 6 months."""
        total_minutes = self.days * 24 * 60
        start_dt = datetime(2026, 1, 1, tzinfo=timezone.utc)

        # --- Regime generator (changes every 2-7 days) ---
        regimes = self._generate_regimes(total_minutes)

        # --- Generate M1 data ---
        m1_data = self._generate_tf_data(total_minutes, start_dt, regimes, aggregation=1)

        # --- Resample to M5 and M15 ---
        m5_data = self._resample_ohlcv(m1_data, 5)
        m15_data = self._resample_ohlcv(m1_data, 15)

        return {
            "M1": m1_data,
            "M5": m5_data,
            "M15": m15_data,
            "regimes": regimes,
            "news_events": self.NEWS_EVENTS,
        }

    def _generate_regimes(self, total_minutes: int) -> list:
        """Generate market regime timeline."""
        regimes = []
        pos = 0
        while pos < total_minutes:
            duration_minutes = random.randint(48 * 60, 168 * 60)  # 2-7 days
            regime = random.choices(
                ["bullish_trend", "bearish_trend", "sideways", "choppy"],
                weights=[0.30, 0.25, 0.25, 0.20],
                k=1,
            )[0]
            strength = random.uniform(0.3, 1.0)
            regimes.append({
                "start_minute": pos,
                "end_minute": min(pos + duration_minutes, total_minutes),
                "regime": regime,
                "strength": strength,
            })
            pos += duration_minutes
        return regimes

    def _get_session(self, dt: datetime) -> str:
        """Determine trading session."""
        hour = dt.hour
        if 8 <= hour < 17 and 13 <= hour < 22:
            return "overlap"
        elif 8 <= hour < 17:
            return "london"
        elif 13 <= hour < 22:
            return "new_york"
        elif hour >= 23 or hour < 8:
            return "asian"
        return "transition"

    def _generate_tf_data(self, total_minutes: int, start_dt: datetime,
                          regimes: list, aggregation: int = 1) -> pd.DataFrame:
        """Generate OHLCV data at specified aggregation."""
        records = []
        price = self.BASE_PRICE
        base_atr = self.TYPICAL_ATR_M15

        for minute_idx in range(0, total_minutes, aggregation):
            dt = start_dt + timedelta(minutes=minute_idx)
            session = self._get_session(dt)
            vol_factor = self.SESSION_VOLATILITY.get(session, 0.8)

            # Find current regime
            current_regime = "sideways"
            regime_strength = 0.5
            for r in regimes:
                if r["start_minute"] <= minute_idx < r["end_minute"]:
                    current_regime = r["regime"]
                    regime_strength = r["strength"]
                    break

            # Check for news events
            news_vol = 1.0
            news_spread = 1.0
            for day_offset, event_name, hour_utc, vol_mul, spread_mul in self.NEWS_EVENTS:
                event_dt = start_dt + timedelta(days=day_offset, hours=hour_utc - start_dt.hour)
                minutes_to_event = abs((dt - event_dt).total_seconds() / 60)
                if minutes_to_event < 30:
                    decay = max(0, 1 - minutes_to_event / 30)
                    news_vol = max(news_vol, 1 + (vol_mul - 1) * decay)
                    news_spread = max(news_spread, 1 + (spread_mul - 1) * decay)

            # Compute effective ATR
            effective_atr = base_atr * vol_factor * news_vol
            if current_regime == "bullish_trend":
                drift = base_atr * 0.03 * regime_strength * vol_factor
            elif current_regime == "bearish_trend":
                drift = -base_atr * 0.03 * regime_strength * vol_factor
            elif current_regime == "choppy":
                drift = 0
                effective_atr *= 1.5
            else:  # sideways
                drift = np.random.normal(0, base_atr * 0.01)
                effective_atr *= 0.6

            # OHLC generation with realistic candle patterns
            open_price = price
            noise = np.random.normal(drift, effective_atr)
            close_price = open_price + noise

            # --- Inject realistic candle patterns (25-35% of candles) ---
            pattern_roll = random.random()
            candle_pattern = None

            # Higher pattern frequency in active sessions
            pattern_chance = 0.25  # base
            if session in ("overlap",):
                pattern_chance = 0.35
            elif session in ("london", "new_york"):
                pattern_chance = 0.30

            if pattern_roll < pattern_chance:
                pattern_type = random.choices(
                    ["hammer", "shooting_star", "engulfing_bull", "engulfing_bear", "long_wick_lower", "long_wick_upper"],
                    weights=[0.20, 0.18, 0.18, 0.17, 0.15, 0.12],
                    k=1,
                )[0]

                total_range = effective_atr * random.uniform(0.5, 1.5)
                body_size = total_range * random.uniform(0.08, 0.30)

                if pattern_type == "hammer":
                    # Small body at top, long lower wick
                    body_start = open_price
                    body_end = body_start + body_size if random.random() > 0.5 else body_start - body_size
                    lower_wick = total_range * random.uniform(0.55, 0.75)
                    upper_wick = total_range - body_size - lower_wick
                    low_price = min(body_start, body_end) - lower_wick
                    high_price = max(body_start, body_end) + upper_wick
                    open_price, close_price = body_start, body_end
                elif pattern_type == "shooting_star":
                    # Small body at bottom, long upper wick
                    body_start = open_price
                    body_end = body_start + body_size if random.random() > 0.5 else body_start - body_size
                    upper_wick = total_range * random.uniform(0.55, 0.75)
                    lower_wick = total_range - body_size - upper_wick
                    high_price = max(body_start, body_end) + upper_wick
                    low_price = min(body_start, body_end) - lower_wick
                    open_price, close_price = body_start, body_end
                elif pattern_type == "engulfing_bull":
                    # Large bullish body (>60% of range)
                    body_size = total_range * random.uniform(0.6, 0.85)
                    body_end = open_price + body_size  # bullish
                    upper_wick = total_range * random.uniform(0.02, 0.15)
                    lower_wick = total_range - body_size - upper_wick
                    high_price = body_end + upper_wick
                    low_price = open_price - lower_wick
                    close_price = body_end
                elif pattern_type == "engulfing_bear":
                    # Large bearish body (>60% of range)
                    body_size = total_range * random.uniform(0.6, 0.85)
                    body_end = open_price - body_size  # bearish
                    upper_wick = total_range * random.uniform(0.02, 0.15)
                    lower_wick = total_range - body_size - upper_wick
                    high_price = open_price + upper_wick
                    low_price = body_end - lower_wick
                    close_price = body_end
                elif pattern_type == "long_wick_lower":
                    lower_wick = total_range * random.uniform(0.6, 0.8)
                    remaining = total_range - lower_wick
                    body_part = remaining * random.uniform(0.3, 0.6)
                    upper_wick = remaining - body_part
                    body_end = open_price + body_part if random.random() > 0.5 else open_price - body_part
                    low_price = min(open_price, body_end) - lower_wick
                    high_price = max(open_price, body_end) + upper_wick
                    close_price = body_end
                else:  # long_wick_upper
                    upper_wick = total_range * random.uniform(0.6, 0.8)
                    remaining = total_range - upper_wick
                    body_part = remaining * random.uniform(0.3, 0.6)
                    lower_wick = remaining - body_part
                    body_end = open_price + body_part if random.random() > 0.5 else open_price - body_part
                    high_price = max(open_price, body_end) + upper_wick
                    low_price = min(open_price, body_end) - lower_wick
                    close_price = body_end

                # Safety clamp
                close_price = max(price - effective_atr * 3, min(price + effective_atr * 3, close_price))
                high_price = max(open_price, close_price, high_price)
                low_price = min(open_price, close_price, low_price)
            else:
                # Normal candle
                high_price = max(open_price, close_price) + abs(np.random.normal(0, effective_atr * 0.3))
                low_price = min(open_price, close_price) - abs(np.random.normal(0, effective_atr * 0.3))

            # Volume
            base_vol = 50 + 20 * vol_factor * news_vol
            tick_volume = max(1, int(np.random.normal(base_vol, 10)))

            # Spread (wider during news and Asian)
            spread_val = self.SPREAD_BASE * (1 + 0.3 * (vol_factor - 1)) * news_spread
            spread_val = max(3, min(200, int(spread_val)))
            spread_val += random.randint(-2, 2)

            records.append({
                "time": dt,
                "open": round(open_price, 2),
                "high": round(high_price, 2),
                "low": round(low_price, 2),
                "close": round(close_price, 2),
                "tick_volume": tick_volume,
                "spread": spread_val,
                "session": session,
                "regime": current_regime,
            })

            price = close_price

        df = pd.DataFrame(records)
        df.set_index("time", inplace=True)
        return df

    @staticmethod
    def _resample_ohlcv(df: pd.DataFrame, factor: int) -> pd.DataFrame:
        """Resample M1 data to higher timeframes."""
        freq_str = f"{factor}min"
        resampled = df.resample(freq_str).agg({
            "open": "first",
            "high": "max",
            "low": "min",
            "close": "last",
            "tick_volume": "sum",
            "spread": "mean",
        }).dropna()

        # Carry over session and regime
        session_series = df["session"].resample(freq_str).agg(lambda x: x.mode().iloc[0] if len(x.mode()) > 0 else x.iloc[0])
        regime_series = df["regime"].resample(freq_str).agg(lambda x: x.mode().iloc[0] if len(x.mode()) > 0 else x.iloc[0])

        resampled["session"] = session_series
        resampled["regime"] = regime_series

        # Round spread
        resampled["spread"] = resampled["spread"].round(0).astype(int)
        resampled["tick_volume"] = resampled["tick_volume"].astype(int)

        return resampled


# ---------------------------------------------------------------------------
# Mock DeepSeek Client (for backtest — advisory only)
# ---------------------------------------------------------------------------

class MockDeepSeekClient:
    """Simulates DeepSeek AI responses for backtesting."""

    def __init__(self):
        self._initialized = True

    def initialize(self) -> bool:
        return True

    def analyze_market(self, payload: dict) -> dict:
        """Return semi-random but realistic AI analysis."""
        setup_score = payload.get("strategy_result", {}).get("setup_score", 0)
        direction = payload.get("strategy_result", {}).get("direction", "NONE")
        news_risk = payload.get("news_context", {}).get("global_risk_mode", "low")

        if setup_score >= 80:
            confidence = random.randint(70, 90)
            sentiment = "bullish" if direction == "BUY" else "bearish"
            risk_flag = random.choice(["low", "low", "medium"])
        elif setup_score >= 60:
            confidence = random.randint(50, 70)
            sentiment = "bullish" if direction == "BUY" else "bearish"
            risk_flag = random.choice(["low", "medium", "medium"])
        else:
            confidence = random.randint(30, 55)
            sentiment = "neutral"
            risk_flag = "medium"

        if news_risk in ("high", "news_blackout"):
            risk_flag = "high"

        return {
            "sentiment": sentiment,
            "confidence": confidence,
            "reasoning": f"Mock analysis for {direction} setup",
            "risk_flag": risk_flag,
            "conflicts_detected": random.random() < 0.1,
            "ai_unavailable": False,
        }

    def shutdown(self):
        pass


# ---------------------------------------------------------------------------
# Mock News Aggregator (for backtest)
# ---------------------------------------------------------------------------

class MockNewsAggregator:
    """Simulates news context for backtesting."""

    def __init__(self, news_events: list, start_dt: datetime):
        self.news_events = news_events
        self.start_dt = start_dt
        self._current_dt = None

    def set_current_time(self, dt: datetime):
        self._current_dt = dt

    def get_news_context(self) -> dict:
        """Return current news context."""
        if self._current_dt is None:
            return {"global_risk_mode": "low", "news_items_count": 0}

        risk_mode = "low"
        for day_offset, event_name, hour_utc, vol_mul, spread_mul in self.news_events:
            event_dt = self.start_dt + timedelta(days=day_offset, hours=hour_utc - self.start_dt.hour)
            dist = abs((self._current_dt - event_dt).total_seconds() / 60)
            if dist < 30:
                risk_mode = "news_blackout"
                break
            elif dist < 60:
                risk_mode = "high"
                break
            elif dist < 120:
                risk_mode = "medium"

        return {"global_risk_mode": risk_mode, "news_items_count": 1 if risk_mode != "low" else 0}

    def get_risk_overlay(self) -> dict:
        ctx = self.get_news_context()
        if ctx["global_risk_mode"] == "news_blackout":
            return {
                "news_block_all_trades": True,
                "reduce_lot_by_percent": 0.5,
                "increase_risk_score_by": 30,
                "reason": "News blackout active",
            }
        if ctx["global_risk_mode"] == "high":
            return {
                "news_block_all_trades": False,
                "reduce_lot_by_percent": 0.3,
                "increase_risk_score_by": 15,
                "reason": "High risk news environment",
            }
        return {
            "news_block_all_trades": False,
            "reduce_lot_by_percent": 0.0,
            "increase_risk_score_by": 0,
            "reason": "Normal news",
        }


# ---------------------------------------------------------------------------
# Simulated MT5 position counter
# ---------------------------------------------------------------------------

class MockMT5Tracker:
    """Tracks simulated positions for position limit checks."""

    def __init__(self):
        self.open_positions = []
        self._position_ages = []  # (idx, age_in_candles)
        self._max_hold_candles = 2  # Close after 2 M5 candles (10 min) for scalp turnover

    def count_open_xauusd(self) -> int:
        return len(self.open_positions)

    def open_position(self, action: str, price: float, sl: float, tp: float):
        self.open_positions.append({
            "action": action,
            "entry": price,
            "sl": sl,
            "tp": tp,
        })
        self._position_ages.append(0)

    def update_positions(self, current_price: float) -> list:
        """Check SL/TP and return closed positions. Also enforce max hold time."""
        closed = []
        remaining = []
        remaining_ages = []

        for i, pos in enumerate(self.open_positions):
            if i >= len(self._position_ages):
                self._position_ages.append(0)
            age = self._position_ages[i] + 1
            hit = False
            pnl = 0.0
            reason = ""

            # XAUUSD P&L: 0.01 lot = ~$0.10 per $1 move
            lot_size = 0.01
            pip_value = lot_size * 0.10  # $0.10 per point

            if pos["action"] == "BUY":
                if pos["sl"] and current_price <= pos["sl"]:
                    pnl = (pos["sl"] - pos["entry"]) * pip_value
                    reason = "SL"
                    hit = True
                elif pos["tp"] and current_price >= pos["tp"]:
                    pnl = (pos["tp"] - pos["entry"]) * pip_value
                    reason = "TP"
                    hit = True
                elif age >= self._max_hold_candles:
                    pnl = (current_price - pos["entry"]) * pip_value
                    reason = "MAX_HOLD"
                    hit = True
            else:
                if pos["sl"] and current_price >= pos["sl"]:
                    pnl = (pos["entry"] - pos["sl"]) * pip_value
                    reason = "SL"
                    hit = True
                elif pos["tp"] and current_price <= pos["tp"]:
                    pnl = (pos["entry"] - pos["tp"]) * pip_value
                    reason = "TP"
                    hit = True
                elif age >= self._max_hold_candles:
                    pnl = (pos["entry"] - current_price) * pip_value
                    reason = "MAX_HOLD"
                    hit = True

            if hit:
                closed.append({**pos, "pnl": round(pnl, 2), "reason": reason})
            else:
                remaining.append(pos)
                remaining_ages.append(age)

        self.open_positions = remaining
        self._position_ages = remaining_ages
        return closed


# ---------------------------------------------------------------------------
# Main Backtest Runner
# ---------------------------------------------------------------------------

def run_gold_backtest(days: int = 180, use_vol_filter: bool = True,
                      use_news_filter: bool = True, use_mtf: bool = True,
                      session_filter: bool = True) -> dict:
    """
    Run full system backtest on generated XAUUSD data.

    Returns comprehensive performance report as dict.
    """
    label_parts = []
    if use_vol_filter: label_parts.append("VOL")
    else: label_parts.append("noVOL")
    if use_news_filter: label_parts.append("NEWS")
    else: label_parts.append("noNEWS")
    if use_mtf: label_parts.append("MTF")
    else: label_parts.append("noMTF")
    if session_filter: label_parts.append("SESS")
    else: label_parts.append("noSESS")
    label = "_".join(label_parts)

    logger.info(f"\n{'='*60}")
    logger.info(f"GOLD BACKTEST: {label} | {days} days")
    logger.info(f"{'='*60}")

    # Generate data
    gen = XAUUSDDataGenerator(days=days, seed=42)
    data = gen.generate()
    start_dt = data["M1"].index[0]

    # Initialize components
    strategy = GoldScalpingStrategy()
    vol_filter = GoldVolatilityFilter()
    risk_manager = RiskManager(default_balance=10000.0)
    deepseek = MockDeepSeekClient()
    news_agg = MockNewsAggregator(data["news_events"], start_dt)
    mt5_tracker = MockMT5Tracker()

    # Compute all indicators upfront
    from trading_bot.indicators.technical_indicators import compute_all_indicators

    m1_df = data["M1"]
    m5_df = data["M5"]
    m15_df = data["M15"]

    m1_ind = compute_all_indicators(m1_df)
    m5_ind = compute_all_indicators(m5_df)
    m15_ind = compute_all_indicators(m15_df)

    # Walk through M5 candles (primary signal timeframe)
    trades = []
    blocked_signals = []
    daily_trades = defaultdict(int)
    session_trades = defaultdict(list)
    news_trades = defaultdict(list)
    regime_trades = defaultdict(list)

    # Process each M5 candle
    for idx in range(100, len(m5_df) - 1):  # Skip first 100 for warmup
        current_dt = m5_df.index[idx]
        current_price = float(m5_df["close"].iloc[idx])
        actual_session = m5_df["session"].iloc[idx]
        actual_regime = m5_df["regime"].iloc[idx]

        # Reset daily counters
        if current_dt.hour == 0 and current_dt.minute < 5:
            risk_manager.reset_daily()
            strategy.reset_daily()
            day_key = current_dt.strftime("%Y-%m-%d")
            daily_trades[day_key] = 0

        # News context
        news_agg.set_current_time(current_dt)
        news_context = news_agg.get_news_context()
        news_overlay = news_agg.get_risk_overlay()

        # Check open positions
        closed = mt5_tracker.update_positions(current_price)
        for c in closed:
            # Find the matching trade and record P&L
            for t in trades:
                if t.get("pnl_recorded"):
                    continue
                t["exit_price"] = current_price
                t["pnl"] = c["pnl"]
                t["exit_reason"] = c["reason"]
                t["exit_time"] = str(current_dt)
                t["pnl_recorded"] = True
                risk_manager.record_result(c["pnl"])
                break

        # --- Run strategy ---
        # Slice data up to current candle
        m1_slice = m1_df.iloc[:max(idx * 5, 100)]
        m5_slice = m5_df.iloc[:idx + 1]
        m15_slice = m15_df.iloc[:max(idx // 3, 100)]

        # Recompute indicators on slices
        try:
            m1_ind_slice = compute_all_indicators(m1_slice)
            m5_ind_slice = compute_all_indicators(m5_slice)
            m15_ind_slice = compute_all_indicators(m15_slice)
        except Exception:
            continue

        strategy_result = strategy.analyze(
            m1_indicators=m1_ind_slice,
            m5_indicators=m5_ind_slice,
            m15_indicators=m15_ind_slice,
            m1_ohlcv=m1_slice,
            m5_ohlcv=m5_slice,
            m15_ohlcv=m15_slice,
            news_context=news_context,
        )

        # --- Apply filters based on mode ---
        if not use_mtf:
            # Disable multi-timeframe: only use M5
            strategy_result = strategy.analyze(
                m1_indicators=m5_ind_slice,
                m5_indicators=m5_ind_slice,
                m15_indicators=m5_ind_slice,
                m1_ohlcv=m5_slice,
                m5_ohlcv=m5_slice,
                m15_ohlcv=m5_slice,
                news_context=news_context,
            )

        direction = strategy_result.get("direction", "NONE")
        setup_score = strategy_result.get("setup_score", 0)

        if direction == "NONE":
            continue

        # Volatility filter
        vol_filter_result = {"trade_ok": True, "lot_reduction_factor": 1.0, "reason": "ok",
                             "atr_ratio": 1.0, "spread_assessment": "normal",
                             "market_regime": "normal"}
        if use_vol_filter:
            try:
                vol_filter_result = vol_filter.analyze(
                    m1_ohlcv=m1_slice,
                    m5_ohlcv=m5_slice,
                    m15_ohlcv=m15_slice,
                    m1_indicators=m1_ind_slice,
                    m5_indicators=m5_ind_slice,
                    m15_indicators=m15_ind_slice,
                )
            except Exception:
                vol_filter_result = {"trade_ok": True, "lot_reduction_factor": 1.0,
                                     "reason": "filter error"}

        if not vol_filter_result.get("trade_ok", True):
            blocked_signals.append({
                "time": str(current_dt),
                "reason": "volatility",
                "score": setup_score,
                "session": actual_session,
            })
            continue

        # News filter
        if not use_news_filter:
            news_overlay = {"news_block_all_trades": False}
        if news_overlay.get("news_block_all_trades", False):
            blocked_signals.append({
                "time": str(current_dt),
                "reason": "news_blackout",
                "score": setup_score,
                "session": actual_session,
            })
            continue

        # Session filter
        if session_filter:
            if actual_session == "asian" and random.random() < 0.4:
                blocked_signals.append({
                    "time": str(current_dt),
                    "reason": "asian_reduced",
                    "score": setup_score,
                    "session": actual_session,
                })
                continue

        # Position limits — override _last_trade_time with simulated time
        open_count = mt5_tracker.count_open_xauusd()
        # Temporarily inject candle time for cooldown to work in simulation
        _saved_time = strategy._last_trade_time
        if strategy._last_trade_time is not None:
            # Convert real last_trade_time to simulated: use current_dt minus cooldown
            strategy._last_trade_time = current_dt - timedelta(minutes=5)
        can_trade, limit_reason = strategy.can_trade(open_count)
        strategy._last_trade_time = _saved_time  # restore
        if not can_trade:
            blocked_signals.append({
                "time": str(current_dt),
                "reason": limit_reason,
                "score": setup_score,
                "session": actual_session,
            })
            continue

        # AI analysis
        ai_payload = {
            "strategy_result": {"setup_score": setup_score, "direction": direction},
            "news_context": news_context,
        }
        ai_analysis = deepseek.analyze_market(ai_payload)

        # Risk Manager
        rule_decision = {
            "trend": strategy_result.get("bias", "neutral"),
            "setup_valid": setup_score >= 40,
            "setup_strength": setup_score,
            "atr_value": float(m5_ind_slice["atr"].iloc[-1]) if not m5_ind_slice["atr"].empty else 3.5,
            "volatility": "medium",
            "rsi_condition": "neutral",
        }
        base_risk = risk_manager.validate(
            rule_decision=rule_decision,
            ai_analysis=ai_analysis,
            ohlcv=m5_slice,
            news_overlay=news_overlay,
        )
        risk_eval = risk_manager.gold_specific_adjustments(
            base_result=base_risk,
            account_balance=10000.0 + sum(t.get("pnl", 0) for t in trades),
            volatility_info=vol_filter_result,
        )

        if not risk_eval.get("approved", False):
            blocked_signals.append({
                "time": str(current_dt),
                "reason": f"risk_{risk_eval.get('reason', 'unknown')[:50]}",
                "score": setup_score,
                "session": actual_session,
            })
            continue

        # Execute trade
        atr_val = float(m5_ind_slice["atr"].iloc[-1]) if not m5_ind_slice["atr"].empty else 3.5
        sl_distance = atr_val * 1.5
        tp_distance = atr_val * 3.0

        if direction == "BUY":
            sl = round(current_price - sl_distance, 2)
            tp = round(current_price + tp_distance, 2)
        else:
            sl = round(current_price + sl_distance, 2)
            tp = round(current_price - tp_distance, 2)

        lot = 0.01
        lot_scale = risk_eval.get("adjusted_lot_scale", 1.0)
        final_lot = round(lot * lot_scale, 2)

        trade = {
            "time": str(current_dt),
            "direction": direction,
            "entry_price": current_price,
            "sl": sl,
            "tp": tp,
            "lot": final_lot,
            "setup_score": setup_score,
            "session": actual_session,
            "regime": actual_regime,
            "news_mode": news_context.get("global_risk_mode", "low"),
            "vol_regime": vol_filter_result.get("market_regime", "normal"),
            "atr_ratio": vol_filter_result.get("atr_ratio", 1.0),
            "pnl": 0.0,
            "pnl_recorded": False,
        }

        trades.append(trade)
        mt5_tracker.open_position(direction, current_price, sl, tp)
        strategy.record_trade()

        day_key = current_dt.strftime("%Y-%m-%d")
        daily_trades[day_key] += 1

        # Track by session
        effective_session = actual_session
        session_trades[effective_session].append(trade)

        # Track by news mode
        news_mode = news_context.get("global_risk_mode", "low")
        news_trades[news_mode].append(trade)

        # Track by regime
        regime_trades[actual_regime].append(trade)

    # Close remaining positions at end
    final_price = float(m5_df["close"].iloc[-1])
    final_closed = mt5_tracker.update_positions(final_price)
    for c in final_closed:
        for t in trades:
            if not t.get("pnl_recorded"):
                t["exit_price"] = final_price
                t["pnl"] = c["pnl"]
                t["exit_reason"] = "EOD"
                t["pnl_recorded"] = True
                risk_manager.record_result(c["pnl"])
                break

    # --- Compute metrics ---
    total_trades = len(trades)
    profitable = [t for t in trades if t.get("pnl", 0) > 0]
    losing = [t for t in trades if t.get("pnl", 0) < 0]
    win_count = len(profitable)
    loss_count = len(losing)
    win_rate = (win_count / total_trades * 100) if total_trades > 0 else 0.0

    gross_profit = sum(t.get("pnl", 0) for t in profitable)
    gross_loss = abs(sum(t.get("pnl", 0) for t in losing))
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else float("inf")
    total_pnl = sum(t.get("pnl", 0) for t in trades)
    expectancy = total_pnl / total_trades if total_trades > 0 else 0.0
    avg_trades_per_day = total_trades / days if days > 0 else 0.0

    # Drawdown
    balance = 10000.0
    peak = balance
    max_dd = 0.0
    for t in trades:
        balance += t.get("pnl", 0)
        if balance > peak:
            peak = balance
        dd = (peak - balance) / peak * 100 if peak > 0 else 0
        max_dd = max(max_dd, dd)

    # Longest losing streak
    longest_losing = 0
    current_streak = 0
    for t in trades:
        if t.get("pnl", 0) < 0:
            current_streak += 1
            longest_losing = max(longest_losing, current_streak)
        else:
            current_streak = 0

    # Session breakdown
    session_breakdown = {}
    for sess, st in session_trades.items():
        s_wins = [t for t in st if t.get("pnl", 0) > 0]
        session_breakdown[sess] = {
            "trades": len(st),
            "wins": len(s_wins),
            "win_rate": round(len(s_wins) / max(len(st), 1) * 100, 1),
            "total_pnl": round(sum(t.get("pnl", 0) for t in st), 2),
        }

    # News impact
    news_impact = {}
    for n_mode, nt in news_trades.items():
        n_wins = [t for t in nt if t.get("pnl", 0) > 0]
        news_impact[n_mode] = {
            "trades": len(nt),
            "wins": len(n_wins),
            "win_rate": round(len(n_wins) / max(len(nt), 1) * 100, 1),
            "total_pnl": round(sum(t.get("pnl", 0) for t in nt), 2),
        }

    # Volatility filter effectiveness (for full system only)
    vol_filter_effect = {}
    if use_vol_filter:
        vol_filter_effect = {
            "total_blocks": len([b for b in blocked_signals if b["reason"] == "volatility"]),
            "total_signals_blocked": len(blocked_signals),
            "average_blocked_score": round(
                np.mean([b["score"] for b in blocked_signals]) if blocked_signals else 0, 1
            ),
        }

    # Regime breakdown
    regime_breakdown = {}
    for reg, rt in regime_trades.items():
        r_wins = [t for t in rt if t.get("pnl", 0) > 0]
        regime_breakdown[reg] = {
            "trades": len(rt),
            "wins": len(r_wins),
            "win_rate": round(len(r_wins) / max(len(rt), 1) * 100, 1),
            "total_pnl": round(sum(t.get("pnl", 0) for t in rt), 2),
        }

    # Worst trades
    sorted_trades = sorted(trades, key=lambda t: t.get("pnl", 0))
    worst_5 = sorted_trades[:5]

    report = {
        "system_label": label,
        "backtest_days": days,
        "win_rate": round(win_rate, 1),
        "profit_factor": round(profit_factor, 2),
        "expectancy_per_trade": round(expectancy, 2),
        "max_drawdown_pct": round(max_dd, 2),
        "total_trades": total_trades,
        "winning_trades": win_count,
        "losing_trades": loss_count,
        "avg_trades_per_day": round(avg_trades_per_day, 2),
        "longest_losing_streak": longest_losing,
        "total_pnl": round(total_pnl, 2),
        "final_balance": round(10000 + total_pnl, 2),
        "session_breakdown": session_breakdown,
        "news_impact_analysis": news_impact,
        "volatility_filter_effectiveness": vol_filter_effect,
        "regime_breakdown": regime_breakdown,
        "worst_5_trades": [
            {
                "time": t["time"],
                "direction": t["direction"],
                "pnl": t.get("pnl", 0),
                "session": t["session"],
                "regime": t["regime"],
                "news_mode": t["news_mode"],
                "exit_reason": t.get("exit_reason", "unknown"),
            }
            for t in worst_5
        ],
        "blocked_signals_count": len(blocked_signals),
    }

    return report


# ---------------------------------------------------------------------------
# A/B Comparison Runner
# ---------------------------------------------------------------------------

def run_full_validation() -> dict:
    """Run system A (full) vs system B (reduced) and generate final verdict."""
    logger.info("=" * 70)
    logger.info("GOLD SCALPING BOT — COMPREHENSIVE BACKTEST VALIDATION")
    logger.info("=" * 70)

    # System A: FULL (all modules active)
    logger.info("\n>>> RUNNING SYSTEM A: FULL PIPELINE <<<")
    report_a = run_gold_backtest(days=180, use_vol_filter=True, use_news_filter=True,
                                  use_mtf=True, session_filter=True)

    # System B: REDUCED (filters disabled)
    logger.info("\n>>> RUNNING SYSTEM B: REDUCED PIPELINE <<<")
    report_b = run_gold_backtest(days=180, use_vol_filter=False, use_news_filter=False,
                                  use_mtf=False, session_filter=False)

    # --- Comparison ---
    comparison = {
        "metric": ["Win Rate (%)", "Profit Factor", "Expectancy ($)", "Max Drawdown (%)",
                    "Total Trades", "Avg Trades/Day", "Longest Losing Streak",
                    "Total P&L ($)"],
        "system_a_full": [
            report_a["win_rate"],
            report_a["profit_factor"],
            report_a["expectancy_per_trade"],
            report_a["max_drawdown_pct"],
            report_a["total_trades"],
            report_a["avg_trades_per_day"],
            report_a["longest_losing_streak"],
            report_a["total_pnl"],
        ],
        "system_b_reduced": [
            report_b["win_rate"],
            report_b["profit_factor"],
            report_b["expectancy_per_trade"],
            report_b["max_drawdown_pct"],
            report_b["total_trades"],
            report_b["avg_trades_per_day"],
            report_b["longest_losing_streak"],
            report_b["total_pnl"],
        ],
        "winner": [],
    }

    for i, metric_name in enumerate(comparison["metric"]):
        val_a = comparison["system_a_full"][i]
        val_b = comparison["system_b_reduced"][i]

        if metric_name in ("Max Drawdown (%)", "Longest Losing Streak"):
            # Lower is better
            if val_a < val_b:
                comparison["winner"].append("A")
            elif val_b < val_a:
                comparison["winner"].append("B")
            else:
                comparison["winner"].append("TIE")
        else:
            # Higher is better
            if val_a > val_b:
                comparison["winner"].append("A")
            elif val_b > val_a:
                comparison["winner"].append("B")
            else:
                comparison["winner"].append("TIE")

    # --- Pass/Fail assessment ---
    wr_ok = report_a["win_rate"] >= 60
    pf_ok = report_a["profit_factor"] >= 1.5
    dd_ok = report_a["max_drawdown_pct"] <= 20
    stability_ok = True  # Will check below

    # Check session consistency
    session_wr = report_a["session_breakdown"]
    if session_wr:
        rates = [s["win_rate"] for s in session_wr.values() if s["trades"] > 5]
        if rates:
            wr_spread = max(rates) - min(rates)
            stability_ok = wr_spread < 30  # Not more than 30% spread between sessions

    passes = sum([wr_ok, pf_ok, dd_ok, stability_ok])
    total_checks = 4

    if passes == total_checks:
        verdict = "APPROVED FOR DEMO"
        verdict_detail = "All pass criteria met. System is stable and safe for controlled demo trading."
    elif passes >= 3:
        verdict = "CONDITIONAL APPROVAL"
        verdict_detail = f"Passed {passes}/{total_checks} checks. Run in demo with reduced position sizes."
    else:
        verdict = "NOT READY FOR LIVE TRADING"
        verdict_detail = f"Failed {total_checks - passes}/{total_checks} critical checks. Review and fix before demo."

    # Build final report
    final_report = {
        "validation_timestamp": str(datetime.now()),
        "data_source": "Synthetic 6-month XAUUSD (realistic stochastic model)",
        "system_a_performance": report_a,
        "system_b_performance": report_b,
        "ab_comparison": comparison,
        "pass_fail_assessment": {
            "win_rate_ge_60": wr_ok,
            "profit_factor_ge_1_5": pf_ok,
            "max_drawdown_le_20": dd_ok,
            "session_stability": stability_ok,
            "checks_passed": f"{passes}/{total_checks}",
            "verdict": verdict,
            "verdict_detail": verdict_detail,
        },
        "session_breakdown_full": report_a["session_breakdown"],
        "news_impact_full": report_a["news_impact_analysis"],
        "worst_trades_analysis": report_a["worst_5_trades"],
    }

    # Print summary
    logger.info("\n" + "=" * 70)
    logger.info("FINAL VALIDATION REPORT")
    logger.info("=" * 70)
    logger.info(f"System A (Full):    WR={report_a['win_rate']}% PF={report_a['profit_factor']} "
                 f"DD={report_a['max_drawdown_pct']}% Trades={report_a['total_trades']} "
                 f"P&L=${report_a['total_pnl']:.2f}")
    logger.info(f"System B (Reduced): WR={report_b['win_rate']}% PF={report_b['profit_factor']} "
                 f"DD={report_b['max_drawdown_pct']}% Trades={report_b['total_trades']} "
                 f"P&L=${report_b['total_pnl']:.2f}")
    logger.info(f"A/B Winners: {comparison['winner']}")
    logger.info(f"\nPass/Fail: {passes}/{total_checks}")
    logger.info(f"WR≥60%: {wr_ok} | PF≥1.5: {pf_ok} | DD≤20%: {dd_ok} | Stable: {stability_ok}")
    logger.info(f"\nVERDICT: {verdict}")
    logger.info(f"{verdict_detail}")
    logger.info("=" * 70)

    return final_report


# ---------------------------------------------------------------------------
# Standalone execution
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    report = run_full_validation()

    # Save to file
    output_path = "trading_bot/backtest/validation_report.json"
    with open(output_path, "w") as f:
        json.dump(report, f, indent=2, default=str)

    logger.info(f"\nReport saved to: {output_path}")
    print(json.dumps(report, indent=2, default=str))