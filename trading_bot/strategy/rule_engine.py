"""
Strategy Rules Engine — Deterministic setup validation with scoring.

Evaluates market conditions using technical indicators and predefined rules.
Instead of hard binary setup_valid, outputs setup_strength (0-100) and
detailed breakdown for the risk scoring engine.

The rule engine identifies CONFIRMED setups (not rejects them).
Risk scoring is done by the RiskManager.
"""

from typing import Optional
import pandas as pd
import numpy as np
from trading_bot.config import Config
from trading_bot.utils.logger import logger


class RuleEngine:
    """
    Deterministic rule-based strategy engine.

    Evaluates market conditions and outputs:
        - setup_valid: True if no critical contradictions
        - setup_strength: 0-100 score of setup quality
        - breakdown: per-condition assessment
    """

    def __init__(self):
        self.logger = logger

    def analyze(self, ohlcv: pd.DataFrame, indicators: dict) -> dict:
        close = ohlcv["close"]
        high = ohlcv["high"]
        low = ohlcv["low"]

        latest = self._get_latest_values(indicators)
        trend = self._determine_trend(latest["emas"])
        volatility = self._assess_volatility(latest["atr_value"], close)
        rsi_condition = self._classify_rsi(latest["rsi_value"])
        macd_condition = self._classify_macd(latest["macd"])
        ema_condition = self._classify_ema_position(close.iloc[-1], latest["emas"])

        # --- Scoring breakdown (0 = bad, 100 = perfect) ---
        breakdown = {}
        penalties = []

        # Trend score: bullish/bearish = good, neutral = penalty
        if trend != "neutral":
            breakdown["trend"] = 100
        else:
            breakdown["trend"] = 30
            penalties.append("trend_neutral")

        # RSI: avoid extremes for trend trading
        if trend == "bullish" and rsi_condition == "overbought":
            breakdown["rsi"] = 40
            penalties.append("rsi_overbought_bullish")
        elif trend == "bearish" and rsi_condition == "oversold":
            breakdown["rsi"] = 40
            penalties.append("rsi_oversold_bearish")
        else:
            breakdown["rsi"] = 100

        # MACD alignment
        if (trend == "bullish" and macd_condition == "bullish") or \
           (trend == "bearish" and macd_condition == "bearish"):
            breakdown["macd"] = 100
        elif (trend == "bullish" and macd_condition == "bearish") or \
             (trend == "bearish" and macd_condition == "bullish"):
            breakdown["macd"] = 30
            penalties.append("macd_opposing_trend")
        else:
            breakdown["macd"] = 60
            penalties.append("macd_neutral")

        # EMA position
        if (trend == "bullish" and ema_condition != "price_below_all_emas") or \
           (trend == "bearish" and ema_condition != "price_above_all_emas"):
            breakdown["ema"] = 100
        else:
            breakdown["ema"] = 40
            penalties.append("ema_against_trend")

        # Volatility: very high = risky, very low = uncertain
        if volatility == "medium":
            breakdown["volatility"] = 100
        elif volatility == "high":
            breakdown["volatility"] = 40
            penalties.append("volatility_high")
        else:
            breakdown["volatility"] = 70
            penalties.append("volatility_low")

        # Calculate total strength score (average of all components)
        scores = list(breakdown.values())
        setup_strength = int(np.mean(scores)) if scores else 0

        # setup_valid = True if no critical contradictions and strength >= 50
        critical_penalties = [p for p in penalties if p in (
            "trend_neutral", "macd_opposing_trend", "ema_against_trend"
        )]
        setup_valid = len(critical_penalties) < 2 and setup_strength >= 40

        reason = f"Strength={setup_strength}/100, valid={setup_valid}"
        if penalties:
            reason += f", issues: {', '.join(penalties)}"

        decision = {
            "symbol": getattr(ohlcv, "attrs", {}).get("symbol", "UNKNOWN"),
            "timeframe": getattr(ohlcv, "attrs", {}).get("timeframe", "UNKNOWN"),
            "timestamp": str(ohlcv.index[-1]) if len(ohlcv) > 0 else "N/A",
            "trend": trend,
            "volatility": volatility,
            "rsi_value": round(latest["rsi_value"], 2) if not np.isnan(latest["rsi_value"]) else None,
            "rsi_condition": rsi_condition,
            "macd_condition": macd_condition,
            "ema_condition": ema_condition,
            "atr_value": round(latest["atr_value"], 5) if not np.isnan(latest["atr_value"]) else None,
            "setup_valid": setup_valid,
            "setup_strength": setup_strength,
            "breakdown": breakdown,
            "reason": reason,
        }

        self.logger.info(
            f"RuleEngine: {decision['trend']} trend, strength={setup_strength}, "
            f"valid={setup_valid}, RSI: {rsi_condition}"
        )
        return decision

    @staticmethod
    def _get_latest_values(indicators: dict) -> dict:
        rsi_series = indicators.get("rsi", pd.Series(dtype=float))
        macd_df = indicators.get("macd", pd.DataFrame())
        emas_df = indicators.get("emas", pd.DataFrame())
        atr_series = indicators.get("atr", pd.Series(dtype=float))
        return {
            "rsi_value": rsi_series.iloc[-1] if len(rsi_series) > 0 and not rsi_series.isna().all() else np.nan,
            "macd": {
                "macd": macd_df["macd"].iloc[-1] if len(macd_df) > 0 and "macd" in macd_df else np.nan,
                "signal": macd_df["signal"].iloc[-1] if len(macd_df) > 0 and "signal" in macd_df else np.nan,
                "histogram": macd_df["histogram"].iloc[-1] if len(macd_df) > 0 and "histogram" in macd_df else np.nan,
            },
            "emas": {col: emas_df[col].iloc[-1] for col in emas_df.columns if len(emas_df) > 0},
            "atr_value": atr_series.iloc[-1] if len(atr_series) > 0 and not atr_series.isna().all() else np.nan,
        }

    @staticmethod
    def _determine_trend(emas: dict) -> str:
        if len(emas) < 2:
            return "neutral"
        sorted_periods = sorted(emas.keys(), key=lambda k: int(k.split("_")[1]))
        sorted_values = [emas[k] for k in sorted_periods if not np.isnan(emas.get(k, np.nan))]
        if len(sorted_values) < 2:
            return "neutral"
        if all(sorted_values[i] > sorted_values[i + 1] for i in range(len(sorted_values) - 1)):
            return "bullish"
        elif all(sorted_values[i] < sorted_values[i + 1] for i in range(len(sorted_values) - 1)):
            return "bearish"
        else:
            return "neutral"

    @staticmethod
    def _assess_volatility(atr_value: float, close: pd.Series) -> str:
        if np.isnan(atr_value) or len(close) < 2:
            return "medium"
        avg_close = close.iloc[-20:].mean() if len(close) >= 20 else close.mean()
        atr_pct = (atr_value / avg_close) * 100 if avg_close != 0 else 0
        if atr_pct > 1.5:
            return "high"
        elif atr_pct < 0.5:
            return "low"
        else:
            return "medium"

    @staticmethod
    def _classify_rsi(rsi_value: float) -> str:
        if np.isnan(rsi_value):
            return "neutral"
        if rsi_value >= 70:
            return "overbought"
        elif rsi_value <= 30:
            return "oversold"
        else:
            return "neutral"

    @staticmethod
    def _classify_macd(macd_data: dict) -> str:
        macd_val = macd_data.get("macd", np.nan)
        signal_val = macd_data.get("signal", np.nan)
        if np.isnan(macd_val) or np.isnan(signal_val):
            return "neutral"
        if macd_val > signal_val:
            return "bullish"
        elif macd_val < signal_val:
            return "bearish"
        else:
            return "neutral"

    @staticmethod
    def _classify_ema_position(price: float, emas: dict) -> str:
        if np.isnan(price) or not emas:
            return "unknown"
        above = 0
        below = 0
        for ema_name, ema_value in emas.items():
            if not np.isnan(ema_value):
                if price > ema_value:
                    above += 1
                else:
                    below += 1
        if above > 0 and below == 0:
            return "price_above_all_emas"
        elif below > 0 and above == 0:
            return "price_below_all_emas"
        else:
            return "price_mixed_emas"