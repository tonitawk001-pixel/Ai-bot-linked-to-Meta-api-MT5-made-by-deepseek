"""
Volume Filter — Lightweight volume analysis for secondary signal validation.

Rules:
- Volume spike during breakout (>1.5x avg) → small confidence boost (+5%)
- Low volume trend (<0.5x avg) → reduce confidence (-5%)
- Otherwise neutral

Volume is ONLY a secondary filter, NOT a decision maker.
"""

import numpy as np
import pandas as pd
from trading_bot.utils.logger import logger


def analyze_volume(ohlcv: pd.DataFrame, period: int = 20) -> dict:
    """
    Analyze volume relative to recent average.

    Args:
        ohlcv: DataFrame with 'tick_volume' column.
        period: Lookback period for average (default: 20).

    Returns:
        dict: {
            "current_volume": float,
            "avg_volume": float,
            "volume_ratio": float,
            "is_spike": bool,
            "is_low": bool,
            "confidence_modifier": float (-0.05 to +0.05),
        }
    """
    result = {
        "current_volume": 0,
        "avg_volume": 0,
        "volume_ratio": 1.0,
        "is_spike": False,
        "is_low": False,
        "confidence_modifier": 0.0,
    }

    if "tick_volume" not in ohlcv.columns:
        return result

    volumes = ohlcv["tick_volume"].values
    if len(volumes) < period + 1:
        return result

    current_vol = float(volumes[-1])
    avg_vol = float(np.mean(volumes[-period-1:-1]))  # exclude current

    if avg_vol == 0:
        return result

    ratio = current_vol / avg_vol
    result["current_volume"] = round(current_vol, 0)
    result["avg_volume"] = round(avg_vol, 0)
    result["volume_ratio"] = round(ratio, 2)

    if ratio > 1.5:
        result["is_spike"] = True
        result["confidence_modifier"] = 0.05  # +5% confidence
        logger.debug(f"Volume spike: {ratio:.1f}x avg (+5% confidence)")
    elif ratio < 0.5:
        result["is_low"] = True
        result["confidence_modifier"] = -0.05  # -5% confidence
        logger.debug(f"Low volume: {ratio:.1f}x avg (-5% confidence)")
    else:
        result["confidence_modifier"] = 0.0

    return result