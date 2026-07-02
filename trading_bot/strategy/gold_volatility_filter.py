"""
Gold Volatility Filter — XAUUSD-specific volatility analysis.

Analyzes ATR levels, spread behavior, and market conditions specific to gold.
Provides lot size reduction recommendations and trade frequency modifiers
without overriding the RiskManager.

This module is advisory — the RiskManager is the final authority.
"""

import numpy as np
import pandas as pd
from typing import Optional

from trading_bot.utils.logger import logger


class GoldVolatilityFilter:
    """
    Analyzes XAUUSD volatility conditions.

    Tracks:
      - ATR ratio (current ATR / average ATR)
      - Spread stability
      - Market regime (trending, flat, volatile)

    Outputs advisory adjustments only.
    """

    def __init__(self):
        self._atr_history = []  # rolling ATR values for averaging
        self._max_history = 50
        logger.info("GoldVolatilityFilter initialized.")

    def analyze(
        self,
        m1_ohlcv: pd.DataFrame,
        m5_ohlcv: pd.DataFrame,
        m15_ohlcv: pd.DataFrame,
        m1_indicators: dict,
        m5_indicators: dict,
        m15_indicators: dict,
    ) -> dict:
        """
        Analyze volatility conditions across all timeframes.

        Returns:
            dict: {
                "trade_ok": bool,
                "lot_reduction_factor": float (1.0 = normal, 0.4-1.0 range),
                "frequency_modifier": str ("normal"|"reduced"|"minimal"),
                "atr_assessment": str,
                "spread_assessment": str,
                "market_regime": str,
                "reason": str,
            }
        """
        # 1. ATR analysis (use M5 as primary, cross-check M15)
        atr_info = self._analyze_atr(m5_indicators, m1_indicators)

        # 2. Spread analysis (use M1 for scalp precision)
        spread_info = self._analyze_spread(m1_ohlcv)

        # 3. Market regime detection
        regime = self._detect_regime(m5_indicators, m5_ohlcv, m15_indicators)

        # 4. Compile result
        lot_reduction = 1.0
        reasons = []

        # ATR-based lot reduction
        atr_ratio = atr_info.get("atr_ratio", 1.0)
        if atr_ratio > 2.0:
            lot_reduction *= 0.4  # Reduce lot by 60%
            reasons.append(f"atr_spike_{atr_ratio:.1f}x")
        elif atr_ratio > 1.5:
            lot_reduction *= 0.7  # Reduce lot by 30%
            reasons.append(f"atr_elevated_{atr_ratio:.1f}x")
        elif atr_ratio < 0.4:
            lot_reduction *= 0.8  # Reduce lot by 20% (flat market, poor scalp conditions)
            reasons.append(f"atr_flat_{atr_ratio:.1f}x")

        # Spread-based adjustments
        if spread_info.get("spread_spiking", False):
            lot_reduction = 0.0  # Block trade
            reasons.append("spread_spike_block")

        # Regime-based adjustments
        if regime == "volatile":
            reasons.append("volatile_regime")
        elif regime == "flat":
            reasons.append("flat_regime")
            if lot_reduction > 0.5:
                lot_reduction *= 0.6  # Further reduce in flat markets
        elif regime == "trending":
            reasons.append("trending_regime")  # Ideal for scalping

        # Frequency modifier
        if atr_ratio > 2.0 or regime == "volatile":
            frequency = "reduced"
        elif atr_ratio < 0.4 or regime == "flat":
            frequency = "reduced"
        else:
            frequency = "normal"

        trade_ok = lot_reduction > 0 and not spread_info.get("spread_spiking", False)
        lot_reduction = round(max(0.0, min(1.0, lot_reduction)), 2)

        result = {
            "trade_ok": trade_ok,
            "lot_reduction_factor": lot_reduction,
            "frequency_modifier": frequency,
            "atr_assessment": atr_info.get("assessment", "normal"),
            "spread_assessment": spread_info.get("assessment", "normal"),
            "market_regime": regime,
            "reason": "; ".join(reasons) if reasons else "volatility_normal",
            "atr_ratio": round(atr_ratio, 2),
        }

        logger.info(
            f"GoldVolFilter: regime={regime} atr_ratio={atr_ratio:.2f} "
            f"lot_factor={lot_reduction:.2f} trade_ok={trade_ok}"
        )
        return result

    # ------------------------------------------------------------------
    # ATR analysis
    # ------------------------------------------------------------------

    def _analyze_atr(self, m5_indicators: dict, m1_indicators: dict) -> dict:
        """
        Compare current ATR to recent average across both M1 and M5.

        Returns:
            dict: {
                "atr_ratio": float,
                "assessment": "normal"|"elevated"|"spike"|"low",
                "current_atr": float,
                "avg_atr": float,
            }
        """
        m5_atr_series = m5_indicators.get("atr", pd.Series(dtype=float))
        m1_atr_series = m1_indicators.get("atr", pd.Series(dtype=float))

        current_atr = None
        avg_atr = None

        # Prefer M5 ATR for stability
        if not m5_atr_series.empty and len(m5_atr_series) >= 20:
            try:
                current_atr = float(m5_atr_series.iloc[-1])
                avg_atr = float(m5_atr_series.iloc[-21:-1].mean())  # 20-period avg (exclude current)
            except (IndexError, ValueError):
                pass

        # Fallback to M1
        if current_atr is None and not m1_atr_series.empty and len(m1_atr_series) >= 20:
            try:
                current_atr = float(m1_atr_series.iloc[-1])
                avg_atr = float(m1_atr_series.iloc[-21:-1].mean())
            except (IndexError, ValueError):
                pass

        if current_atr is None or avg_atr is None or avg_atr <= 0:
            return {"atr_ratio": 1.0, "assessment": "normal", "current_atr": 0, "avg_atr": 0}

        ratio = current_atr / avg_atr

        if ratio > 2.0:
            assessment = "spike"
        elif ratio > 1.5:
            assessment = "elevated"
        elif ratio < 0.4:
            assessment = "low"
        else:
            assessment = "normal"

        return {
            "atr_ratio": round(ratio, 2),
            "assessment": assessment,
            "current_atr": round(current_atr, 5),
            "avg_atr": round(avg_atr, 5),
        }

    # ------------------------------------------------------------------
    # Spread analysis (gold-specific)
    # ------------------------------------------------------------------

    @staticmethod
    def _analyze_spread(ohlcv: pd.DataFrame) -> dict:
        """
        Analyze spread conditions for XAUUSD scalp trading.
        Gold spreads are naturally wider — check for spikes specifically.

        Returns:
            dict: {
                "spread_spiking": bool,
                "assessment": str,
                "current_spread": float,
                "avg_spread": float,
            }
        """
        if "spread" not in ohlcv.columns:
            return {"spread_spiking": False, "assessment": "normal", "current_spread": 0, "avg_spread": 0}

        spreads = ohlcv["spread"].iloc[-30:]  # Last 30 candles
        if len(spreads) < 10:
            return {"spread_spiking": False, "assessment": "normal", "current_spread": 0, "avg_spread": 0}

        current_spread = float(spreads.iloc[-1])
        avg_spread = float(spreads.iloc[:-1].mean()) if len(spreads) > 1 else current_spread

        if avg_spread <= 0:
            return {"spread_spiking": False, "assessment": "normal", "current_spread": current_spread, "avg_spread": avg_spread}

        ratio = current_spread / avg_spread
        spiking = ratio > 2.5  # Stricter threshold for gold scalp

        if spiking:
            assessment = "spike"
        elif ratio > 1.8:
            assessment = "elevated"
        else:
            assessment = "normal"

        return {
            "spread_spiking": spiking,
            "assessment": assessment,
            "current_spread": round(current_spread, 0),
            "avg_spread": round(avg_spread, 0),
        }

    # ------------------------------------------------------------------
    # Market regime detection
    # ------------------------------------------------------------------

    @staticmethod
    def _detect_regime(
        m5_indicators: dict,
        m5_ohlcv: pd.DataFrame,
        m15_indicators: dict,
    ) -> str:
        """
        Detect current market regime: trending, volatile, or flat.

        Uses:
          - ATR ratio (volatility expansion/contraction)
          - EMA spread (trend strength)
          - Price range relative to ATR
        """
        # Check ATR for volatility
        m5_atr = m5_indicators.get("atr", pd.Series(dtype=float))
        if m5_atr.empty or len(m5_atr) < 20:
            return "unknown"

        try:
            current_atr = float(m5_atr.iloc[-1])
            avg_atr = float(m5_atr.iloc[-21:-1].mean())
        except (IndexError, ValueError):
            return "unknown"

        if avg_atr <= 0:
            return "unknown"

        atr_ratio = current_atr / avg_atr

        # Check EMA spread on M5 for trend strength
        emas = m5_indicators.get("emas", pd.DataFrame())
        ema_spread_pct = 0.0
        if not emas.empty and "EMA_20" in emas.columns and "EMA_50" in emas.columns:
            try:
                ema20 = float(emas["EMA_20"].iloc[-1])
                ema50 = float(emas["EMA_50"].iloc[-1])
                if ema50 > 0:
                    ema_spread_pct = abs(ema20 - ema50) / ema50
            except (IndexError, ValueError):
                pass

        # Regime classification
        if atr_ratio > 1.8:
            return "volatile"  # High volatility — risk of slippage
        elif atr_ratio < 0.4:
            return "flat"  # Very low volatility — poor scalp conditions
        elif ema_spread_pct > 0.003 and 0.4 <= atr_ratio <= 1.5:
            return "trending"  # Clear trend + normal ATR — ideal for scalping
        elif 0.4 <= atr_ratio <= 1.5:
            return "normal"  # Normal conditions
        else:
            return "unknown"