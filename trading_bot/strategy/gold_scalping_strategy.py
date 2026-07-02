"""
Gold Scalping Strategy — XAUUSD-Only (v2 — Aggressive Low-Risk Scalping).

Uses M1/M5/M15 multi-timeframe analysis. Designed for 5-15 trades/day
with strict 2% risk per trade.

Changes from v1:
  - Entry trigger: RSI in zone + pullback = valid (candle pattern is bonus, not required)
  - Scoring: RSI (25 pts) + Pullback (35 pts) + Bias (20 pts) = 80 baseline
  - Candle pattern bonus: +15 pts (optional)
  - RSI mean reversion entries for neutral bias (RSI < 25 buy, RSI > 75 sell)
  - Cooldowns: 2-5 min active sessions, 10 min asian
  - Max positions: 3 (was 2)
  - Min score threshold: 30 (was 40)
"""

import math
from datetime import datetime, timezone
from typing import Optional

import pandas as pd
import numpy as np

from trading_bot.utils.logger import logger


# ---------------------------------------------------------------------------
# Session times in UTC
# ---------------------------------------------------------------------------
LONDON_START = 8
LONDON_END = 17
NY_START = 13
NY_END = 22
ASIAN_START = 23
ASIAN_END = 8


class GoldScalpingStrategy:
    """
    XAUUSD scalping strategy using M1/M5/M15 timeframes.

    Entry types:
      1. Trend continuation: bias + pullback + RSI ok = ENTRY (candle pattern = bonus)
      2. RSI mean reversion: neutral bias + extreme RSI = ENTRY (counter-trend scalp)
    """

    def __init__(self):
        self._last_trade_time: Optional[datetime] = None
        self._cooldown_minutes = 0
        self._max_open_positions = 10
        self._trades_today = 0
        self._max_trades_per_day = 50
        self._min_trades_per_day = 3
        self._daily_reset_hour = 0  # UTC midnight
        self._current_session = "unknown"
        logger.info("GoldScalpingStrategy v4 initialized (XAUUSD only, 20 trades/day target).")

    # ------------------------------------------------------------------
    # Main analysis entry point
    # ------------------------------------------------------------------

    def analyze(
        self,
        m1_indicators: dict,
        m5_indicators: dict,
        m15_indicators: dict,
        m1_ohlcv: pd.DataFrame,
        m5_ohlcv: pd.DataFrame,
        m15_ohlcv: pd.DataFrame,
        news_context: Optional[dict] = None,
    ) -> dict:
        # 1. Session
        session = self._detect_session()

        # 2. M15 bias
        bias = self._determine_m15_bias(m15_indicators, m15_ohlcv)

        # 3. M5 pullback
        pullback = self._detect_m5_pullback(bias, m5_indicators, m5_ohlcv)

        # 4. M1 entry trigger (RSI now standalone, pattern = bonus)
        entry = self._check_m1_entry(bias, m1_indicators, m1_ohlcv)
        rsi_value = entry.get("rsi_value")

        # 5. Check mean reversion (for neutral bias)
        mean_rev = self._check_mean_reversion(bias, m1_indicators, m1_ohlcv, m15_indicators)

        # 6. Spread & news
        spread_ok = self._check_spread(m1_ohlcv)
        news_ok = self._check_news(news_context)

        # 7. Assemble
        score, direction, confidence, reason = self._assemble_score_v2(
            bias, pullback, entry, rsi_value, mean_rev, spread_ok, news_ok, session
        )

        result = {
            "setup_score": score,
            "direction": direction,
            "confidence": confidence,
            "reason": reason,
            "session": session,
            "bias": bias,
            "pullback_detected": pullback.get("detected", False),
            "entry_trigger": entry.get("rsi_ok", False) or mean_rev.get("active", False),
            "spread_ok": spread_ok,
            "news_ok": news_ok,
            "is_mean_reversion": mean_rev.get("active", False),
        }

        logger.info(
            f"GoldScalping: session={session} bias={bias} "
            f"dir={direction} score={score} conf={confidence:.2f} "
            f"mr={mean_rev.get('active', False)} reason={reason[:80]}"
        )
        return result

    # ------------------------------------------------------------------
    # Session detection
    # ------------------------------------------------------------------

    @staticmethod
    def _detect_session() -> str:
        now = datetime.now(timezone.utc)
        hour = now.hour
        if LONDON_START <= hour < LONDON_END and NY_START <= hour < NY_END:
            return "overlap"
        elif LONDON_START <= hour < LONDON_END:
            return "london"
        elif NY_START <= hour < NY_END:
            return "new_york"
        elif hour >= ASIAN_START or hour < ASIAN_END:
            return "asian"
        return "transition"

    # ------------------------------------------------------------------
    # M15 bias filter
    # ------------------------------------------------------------------

    @staticmethod
    def _determine_m15_bias(indicators: dict, ohlcv: pd.DataFrame) -> str:
        emas = indicators.get("emas", pd.DataFrame())
        if emas.empty or len(emas) < 2:
            return "neutral"
        try:
            ema20 = float(emas["EMA_20"].iloc[-1]) if "EMA_20" in emas.columns else None
            ema50 = float(emas["EMA_50"].iloc[-1]) if "EMA_50" in emas.columns else None
            close = float(ohlcv["close"].iloc[-1])
        except (KeyError, IndexError, ValueError):
            return "neutral"
        if ema20 is None or ema50 is None or np.isnan(ema20) or np.isnan(ema50):
            return "neutral"
        if ema20 > ema50 and close > ema20:
            return "bullish"
        elif ema20 < ema50 and close < ema20:
            return "bearish"
        elif ema20 > ema50:
            return "bullish" if close > ema50 else "neutral"
        elif ema20 < ema50:
            return "bearish" if close < ema50 else "neutral"
        return "neutral"

    # ------------------------------------------------------------------
    # M5 pullback detection (unchanged)
    # ------------------------------------------------------------------

    @staticmethod
    def _detect_m5_pullback(bias: str, indicators: dict, ohlcv: pd.DataFrame) -> dict:
        emas = indicators.get("emas", pd.DataFrame())
        if emas.empty:
            return {"detected": False, "ema_touched": None, "distance_pct": 0.0}
        try:
            close = float(ohlcv["close"].iloc[-1])
            ema20 = float(emas["EMA_20"].iloc[-1]) if "EMA_20" in emas.columns else None
            ema50 = float(emas["EMA_50"].iloc[-1]) if "EMA_50" in emas.columns else None
            low = float(ohlcv["low"].iloc[-3:].min())
            high = float(ohlcv["high"].iloc[-3:].max())
        except (KeyError, IndexError, ValueError):
            return {"detected": False, "ema_touched": None, "distance_pct": 0.0}
        if ema20 is None or np.isnan(ema20):
            ema20 = ema50

        detected = False
        ema_touched = None
        distance_pct = 0.0

        if bias == "bullish":
            dist20 = abs(close - ema20) / ema20 if ema20 and ema20 > 0 else 999
            dist50 = abs(close - ema50) / ema50 if ema50 and ema50 > 0 else 999
            min_dist = min(dist20, dist50)
            distance_pct = round(min_dist * 100, 2)
            touched20 = ema20 and (low <= ema20 * 1.005)
            touched50 = ema50 and (low <= ema50 * 1.005)
            if touched20:
                ema_touched = "EMA_20"; detected = True
            elif touched50:
                ema_touched = "EMA_50"; detected = True
            elif min_dist < 0.008:  # 0.8% for more signals
                ema_touched = "EMA_20" if dist20 < dist50 else "EMA_50"
                detected = True

        elif bias == "bearish":
            dist20 = abs(close - ema20) / ema20 if ema20 and ema20 > 0 else 999
            dist50 = abs(close - ema50) / ema50 if ema50 and ema50 > 0 else 999
            min_dist = min(dist20, dist50)
            distance_pct = round(min_dist * 100, 2)
            touched20 = ema20 and (high >= ema20 * 0.995)
            touched50 = ema50 and (high >= ema50 * 0.995)
            if touched20:
                ema_touched = "EMA_20"; detected = True
            elif touched50:
                ema_touched = "EMA_50"; detected = True
            elif min_dist < 0.008:
                ema_touched = "EMA_20" if dist20 < dist50 else "EMA_50"
                detected = True

        return {"detected": detected, "ema_touched": ema_touched, "distance_pct": distance_pct}

    # ------------------------------------------------------------------
    # M1 entry trigger — RSI is PRIMARY, pattern is BONUS
    # ------------------------------------------------------------------

    @staticmethod
    def _check_m1_entry(bias: str, indicators: dict, ohlcv: pd.DataFrame) -> dict:
        """
        Check M1 for entry. RSI zone is the primary trigger.
        Candle pattern adds bonus but does NOT block the trade.

        BUY:  RSI 25-60 → rsi_ok (widened)
        SELL: RSI 40-75 → rsi_ok (widened)
        """
        rsi_series = indicators.get("rsi", pd.Series(dtype=float))
        if rsi_series.empty:
            return {"rsi_ok": False, "rsi_value": None, "candle_pattern": "none", "pattern_ok": False}

        try:
            rsi_value = float(rsi_series.iloc[-1])
        except (IndexError, ValueError):
            return {"rsi_ok": False, "rsi_value": None, "candle_pattern": "none", "pattern_ok": False}

        if np.isnan(rsi_value):
            return {"rsi_ok": False, "rsi_value": None, "candle_pattern": "none", "pattern_ok": False}

        # Candle pattern (bonus only)
        try:
            open_1 = float(ohlcv["open"].iloc[-2])
            high_1 = float(ohlcv["high"].iloc[-2])
            low_1 = float(ohlcv["low"].iloc[-2])
            close_1 = float(ohlcv["close"].iloc[-2])
            pattern = _classify_candle(open_1, high_1, low_1, close_1)
        except (IndexError, ValueError):
            pattern = "none"

        rsi_ok = False
        pattern_ok = False

        if bias == "bullish":
            rsi_ok = 25 <= rsi_value <= 60
            pattern_ok = pattern in ("hammer", "bullish_engulfing", "long_lower_wick")
        elif bias == "bearish":
            rsi_ok = 40 <= rsi_value <= 75
            pattern_ok = pattern in ("shooting_star", "bearish_engulfing", "long_upper_wick")

        return {
            "rsi_ok": rsi_ok,
            "rsi_value": round(rsi_value, 1),
            "candle_pattern": pattern,
            "pattern_ok": pattern_ok,
        }

    # ------------------------------------------------------------------
    # RSI Mean Reversion (NEW — for neutral bias / sideways markets)
    # ------------------------------------------------------------------

    @staticmethod
    def _check_mean_reversion(
        bias: str,
        m1_indicators: dict,
        m1_ohlcv: pd.DataFrame,
        m15_indicators: dict,
    ) -> dict:
        """
        When M15 bias is neutral OR trend exists, check for RSI extremes as mean reversion.

        RSI < 20 → BUY (oversold bounce, any bias)
        RSI > 80 → SELL (overbought fade, any bias)
        """
        # Mean reversion now works in ALL bias modes for more entries

        rsi_series = m1_indicators.get("rsi", pd.Series(dtype=float))
        if rsi_series.empty:
            return {"active": False, "direction": "NONE", "rsi_value": None}

        try:
            rsi_value = float(rsi_series.iloc[-1])
        except (IndexError, ValueError):
            return {"active": False, "direction": "NONE", "rsi_value": None}

        if np.isnan(rsi_value):
            return {"active": False, "direction": "NONE", "rsi_value": None}

        direction = "NONE"
        if rsi_value <= 25:
            direction = "BUY"
        elif rsi_value >= 75:
            direction = "SELL"

        return {
            "active": direction != "NONE",
            "direction": direction,
            "rsi_value": round(rsi_value, 1),
        }

    # ------------------------------------------------------------------
    # Spread check
    # ------------------------------------------------------------------

    @staticmethod
    def _check_spread(ohlcv: pd.DataFrame) -> bool:
        if "spread" not in ohlcv.columns:
            return True
        spreads = ohlcv["spread"].iloc[-20:]
        if len(spreads) < 5:
            return True
        avg_spread = float(spreads[:-1].mean()) if len(spreads) > 1 else float(spreads.iloc[0])
        current_spread = float(spreads.iloc[-1])
        if avg_spread <= 0:
            return True
        if current_spread > avg_spread * 2.5:
            logger.debug(f"Spread spike: {current_spread:.0f} > {avg_spread * 2.5:.0f}")
            return False
        return True

    # ------------------------------------------------------------------
    # News check
    # ------------------------------------------------------------------

    @staticmethod
    def _check_news(news_context: Optional[dict]) -> bool:
        if news_context is None:
            return True
        if news_context.get("global_risk_mode") == "news_blackout":
            logger.info("News blackout active — blocking trade")
            return False
        return True

    # ------------------------------------------------------------------
    # Score assembly V2 (relaxed thresholds, RSI standalone, mean reversion)
    # ------------------------------------------------------------------

    @staticmethod
    def _assemble_score_v2(
        bias: str,
        pullback: dict,
        entry: dict,
        rsi_value: Optional[float],
        mean_rev: dict,
        spread_ok: bool,
        news_ok: bool,
        session: str,
    ) -> tuple:
        """
        New scoring:
          Bias:             20 pts (bullish/bearish)
          Pullback:         35 pts
          RSI ok:           25 pts (standalone)
          Candle pattern:  +15 pts (bonus, optional)
          Mean reversion:   25 pts (replaces bias+pullback for neutral)
          Spread penalty:  -50
          News penalty:    -50
          Session bonus:   0-10

        Minimum valid: 30 (was 40)
        """
        score = 0
        reasons = []
        direction = "NONE"

        # --- Determine entry path ---
        if mean_rev.get("active"):
            # Mean reversion path (RSI extreme in ANY bias)
            direction = mean_rev["direction"]
            score += 25  # mean reversion base
            reasons.append(f"mean_rev_rsi_{rsi_value}")
            if entry.get("pattern_ok"):
                score += 10
                reasons.append(f"bonus_{entry.get('candle_pattern', 'pattern')}")
        elif bias in ("bullish", "bearish"):
            # Trend continuation path
            direction = "BUY" if bias == "bullish" else "SELL"
            score += 20  # bias
            if pullback.get("detected", False):
                score += 35
                reasons.append(f"pullback_{pullback.get('ema_touched', 'EMA')}")
            else:
                score += 10
                reasons.append("near_ema")
            if entry.get("rsi_ok", False):
                score += 25
                reasons.append(f"rsi_ok_{rsi_value}")
            else:
                score += 5
                reasons.append(f"rsi_wide_{rsi_value}")
            if entry.get("pattern_ok", False):
                score += 15
                reasons.append(f"pattern_{entry.get('candle_pattern', 'pattern')}")
        else:
            # Neutral bias — still allow RSI-based entries using M1 only
            # Use recent price momentum to determine direction
            try:
                rsi_val = float(rsi_value) if rsi_value is not None else 50
            except (ValueError, TypeError):
                rsi_val = 50
            if rsi_val <= 45:
                direction = "BUY"
            elif rsi_val >= 55:
                direction = "SELL"
            else:
                direction = "NONE"

            if direction != "NONE":
                score += 15  # neutral bias reduced
                if pullback.get("detected", False):
                    score += 25
                    reasons.append("neutral_pullback")
                else:
                    score += 8
                    reasons.append("neutral_near")
                if entry.get("rsi_ok", False):
                    score += 25
                    reasons.append(f"rsi_ok_{rsi_value}")
                else:
                    score += 5
                    reasons.append(f"rsi_wide_{rsi_value}")
                if entry.get("pattern_ok", False):
                    score += 15
                    reasons.append(f"pattern_{entry.get('candle_pattern', 'pattern')}")
            else:
                score += 5
                direction = "NONE"
                reasons.append("no_direction")

        # --- Penalties ---
        if not spread_ok:
            score -= 50
            reasons.append("spread_spike")
            direction = "NONE"

        if not news_ok:
            score -= 50
            reasons.append("news_blackout")
            direction = "NONE"

        # --- Session bonus ---
        session_bonus = 0
        if session == "overlap":
            session_bonus = 10
            reasons.append("overlap")
        elif session in ("london", "new_york"):
            session_bonus = 5
            reasons.append("active_sess")
        elif session == "asian":
            reasons.append("asian_slow")

        score += session_bonus

        # Clamp
        score = max(0, min(100, score))
        confidence = score / 100.0

        # Direction override if penalties killed it
        if direction == "NONE":
            score = 0
            confidence = 0.0

        # Minimum threshold: 20 (allows near_ema + rsi_wide entries)
        if score < 20:
            direction = "NONE"
            confidence = 0.0
            reasons.append("score_lt_20")

        reason_str = "; ".join(reasons) if reasons else "no_signal"
        return score, direction, round(confidence, 2), reason_str

    # ------------------------------------------------------------------
    # Position & frequency limits
    # ------------------------------------------------------------------

    def can_trade(self, open_position_count: int) -> tuple:
        now = datetime.now(timezone.utc)

        if open_position_count >= self._max_open_positions:
            return False, f"max_positions_{open_position_count}/{self._max_open_positions}"

        if self._trades_today >= self._max_trades_per_day:
            return False, f"max_daily_trades_{self._max_trades_per_day}"

        if self._last_trade_time is not None:
            elapsed = (now - self._last_trade_time).total_seconds() / 60.0
            cooldown = self._get_cooldown_minutes()
            if elapsed < cooldown:
                return False, f"cooldown_{elapsed:.1f}/{cooldown}min"

        return True, "ok"

    def record_trade(self):
        self._last_trade_time = datetime.now(timezone.utc)
        self._trades_today += 1
        logger.info(f"GoldScalping: trade recorded — {self._trades_today} today")

    def reset_daily(self):
        now = datetime.now(timezone.utc)
        if now.hour == self._daily_reset_hour and self._trades_today > 0:
            logger.info(f"GoldScalping: daily reset — was {self._trades_today} trades")
            self._trades_today = 0

    def _get_cooldown_minutes(self) -> int:
        return 0  # no cooldown — maximize trade frequency


# ---------------------------------------------------------------------------
# Candle pattern classification (unchanged)
# ---------------------------------------------------------------------------

def _classify_candle(open_: float, high: float, low: float, close_: float) -> str:
    body = abs(close_ - open_)
    total_range = high - low
    if total_range <= 0 or body <= 0:
        return "none"

    lower_wick = min(open_, close_) - low
    upper_wick = high - max(open_, close_)
    body_pct = body / total_range
    lower_wick_pct = lower_wick / total_range if total_range > 0 else 0
    upper_wick_pct = upper_wick / total_range if total_range > 0 else 0
    is_bullish = close_ > open_

    if lower_wick_pct > 0.55 and body_pct < 0.35 and lower_wick > body * 2:
        return "hammer"
    if upper_wick_pct > 0.55 and body_pct < 0.35 and upper_wick > body * 2:
        return "shooting_star"
    if lower_wick_pct > 0.6:
        return "long_lower_wick"
    if upper_wick_pct > 0.6:
        return "long_upper_wick"
    if is_bullish and body_pct > 0.6:
        return "bullish_engulfing"
    if not is_bullish and body_pct > 0.6:
        return "bearish_engulfing"

    return "none"