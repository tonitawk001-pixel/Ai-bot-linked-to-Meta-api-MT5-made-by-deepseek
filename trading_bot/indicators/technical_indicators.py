"""
Technical indicator calculation module.

Provides clean, reusable functions for computing common technical
indicators from OHLCV data. All functions accept and return pandas
DataFrames/Series for seamless integration with the data feed.

Indicators implemented:
    - RSI (Relative Strength Index)
    - MACD (Moving Average Convergence Divergence)
    - EMA (Exponential Moving Average) at configurable periods
    - ATR (Average True Range) for volatility measurement
"""

import pandas as pd
import numpy as np

from trading_bot.config import Config
from trading_bot.utils.logger import logger


# ------------------------------------------------------------------
# RSI
# ------------------------------------------------------------------

def compute_rsi(
    close: pd.Series,
    period: int = 14,
) -> pd.Series:
    """
    Compute the Relative Strength Index (RSI).

    RSI measures the speed and magnitude of recent price changes to
    evaluate overbought (typically > 70) or oversold (< 30) conditions.

    Args:
        close: Series of closing prices.
        period: Lookback period (default: 14).

    Returns:
        pd.Series: RSI values, with the first `period` entries as NaN.
    """
    if len(close) < period + 1:
        logger.warning(f"RSI: Not enough data points ({len(close)} < {period + 1})")
        return pd.Series([np.nan] * len(close), index=close.index)

    delta = close.diff()

    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.ewm(span=period, adjust=False).mean()
    avg_loss = loss.ewm(span=period, adjust=False).mean()

    # Avoid division by zero
    avg_loss = avg_loss.replace(0, 1e-10)

    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))

    rsi.name = f"RSI_{period}"
    return rsi


# ------------------------------------------------------------------
# MACD
# ------------------------------------------------------------------

def compute_macd(
    close: pd.Series,
    fast: int = 12,
    slow: int = 26,
    signal: int = 9,
) -> pd.DataFrame:
    """
    Compute the Moving Average Convergence Divergence (MACD).

    MACD shows the relationship between two exponential moving averages
    and is used to identify changes in trend momentum.

    Args:
        close: Series of closing prices.
        fast: Fast EMA period (default: 12).
        slow: Slow EMA period (default: 26).
        signal: Signal line EMA period (default: 9).

    Returns:
        pd.DataFrame with columns:
            - macd: MACD line (fast EMA - slow EMA)
            - signal: Signal line (EMA of MACD line)
            - histogram: MACD line - Signal line
    """
    if len(close) < slow + signal:
        logger.warning(f"MACD: Not enough data points ({len(close)} < {slow + signal})")
        return pd.DataFrame({
            "macd": pd.Series([np.nan] * len(close), index=close.index),
            "signal": pd.Series([np.nan] * len(close), index=close.index),
            "histogram": pd.Series([np.nan] * len(close), index=close.index),
        })

    ema_fast = close.ewm(span=fast, adjust=False).mean()
    ema_slow = close.ewm(span=slow, adjust=False).mean()

    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    histogram = macd_line - signal_line

    result = pd.DataFrame({
        "macd": macd_line,
        "signal": signal_line,
        "histogram": histogram,
    })

    return result


# ------------------------------------------------------------------
# EMA
# ------------------------------------------------------------------

def compute_ema(
    close: pd.Series,
    period: int = 20,
) -> pd.Series:
    """
    Compute an Exponential Moving Average (EMA).

    EMAs give more weight to recent prices, making them more responsive
    to new information than simple moving averages.

    Args:
        close: Series of closing prices.
        period: EMA period (default: 20).

    Returns:
        pd.Series: EMA values, with the first `period - 1` entries as NaN.
    """
    if len(close) < period:
        logger.warning(f"EMA({period}): Not enough data ({len(close)} < {period})")
        return pd.Series([np.nan] * len(close), index=close.index)

    ema = close.ewm(span=period, adjust=False).mean()
    ema.name = f"EMA_{period}"
    return ema


def compute_multiple_emas(
    close: pd.Series,
    periods: list = None,
) -> pd.DataFrame:
    """
    Compute EMAs for multiple periods in one call.

    Args:
        close: Series of closing prices.
        periods: List of EMA periods (default: [20, 50, 200]).

    Returns:
        pd.DataFrame where each column is an EMA at the specified period.
    """
    if periods is None:
        periods = Config.EMA_PERIODS

    result = pd.DataFrame(index=close.index)
    for period in periods:
        result[f"EMA_{period}"] = compute_ema(close, period)

    return result


# ------------------------------------------------------------------
# ATR
# ------------------------------------------------------------------

def compute_atr(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    period: int = 14,
) -> pd.Series:
    """
    Compute the Average True Range (ATR) for volatility measurement.

    ATR measures market volatility by decomposing the entire range of
    an asset price for a given period. Higher ATR indicates higher volatility.

    Args:
        high: Series of high prices.
        low: Series of low prices.
        close: Series of closing prices.
        period: ATR period (default: 14).

    Returns:
        pd.Series: ATR values, with the first `period` entries as NaN.
    """
    if len(close) < period + 1:
        logger.warning(f"ATR: Not enough data ({len(close)} < {period + 1})")
        return pd.Series([np.nan] * len(close), index=close.index)

    # True Range components
    prev_close = close.shift(1)

    tr1 = high - low
    tr2 = (high - prev_close).abs()
    tr3 = (low - prev_close).abs()

    true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

    # ATR as an EMA of True Range (standard approach)
    atr = true_range.ewm(span=period, adjust=False).mean()
    atr.name = f"ATR_{period}"

    return atr


# ------------------------------------------------------------------
# Batch computation
# ------------------------------------------------------------------

def compute_all_indicators(
    ohlcv: pd.DataFrame,
) -> dict:
    """
    Compute all standard technical indicators from a single OHLCV DataFrame.

    This is the primary entry point for the data analysis pipeline.

    Args:
        ohlcv: DataFrame with columns: open, high, low, close, tick_volume.

    Returns:
        dict: {
            "rsi": pd.Series,
            "macd": pd.DataFrame (macd, signal, histogram),
            "emas": pd.DataFrame (EMA_20, EMA_50, EMA_200),
            "atr": pd.Series,
        }
    """
    close = ohlcv["close"]
    high = ohlcv["high"]
    low = ohlcv["low"]

    logger.debug("Computing all technical indicators...")

    rsi = compute_rsi(close, period=Config.RSI_PERIOD)
    macd = compute_macd(close, fast=Config.MACD_FAST, slow=Config.MACD_SLOW, signal=Config.MACD_SIGNAL)
    emas = compute_multiple_emas(close, periods=Config.EMA_PERIODS)
    atr = compute_atr(high, low, close, period=Config.ATR_PERIOD)

    return {
        "rsi": rsi,
        "macd": macd,
        "emas": emas,
        "atr": atr,
    }