"""
Data feed module for fetching market data from MetaTrader 5.

Provides functions to retrieve OHLCV candle data across multiple
timeframes and symbols. All data is returned as pandas DataFrames
for downstream analysis.
"""

from typing import List, Optional

import MetaTrader5 as mt5
import pandas as pd

from trading_bot.config import Config
from trading_bot.utils.logger import logger


# Mapping from string timeframe labels to MT5 constants
TIMEFRAME_MAP = {
    "M1": mt5.TIMEFRAME_M1,
    "M5": mt5.TIMEFRAME_M5,
    "M15": mt5.TIMEFRAME_M15,
    "M30": mt5.TIMEFRAME_M30,
    "H1": mt5.TIMEFRAME_H1,
    "H4": mt5.TIMEFRAME_H4,
    "D1": mt5.TIMEFRAME_D1,
    "W1": mt5.TIMEFRAME_W1,
    "MN1": mt5.TIMEFRAME_MN1,
}


def get_candles(
    symbol: str,
    timeframe: str = "H1",
    count: int = 100,
) -> Optional[pd.DataFrame]:
    """
    Fetch the latest OHLCV candles for a given symbol and timeframe.

    Args:
        symbol: MT5 symbol name (e.g., "EURUSD", "XAUUSD", "GBPUSD").
        timeframe: String timeframe label ("M1", "M5", "H1", "H4", "D1", etc.).
        count: Number of candles to retrieve (default: 100).

    Returns:
        Optional[pd.DataFrame]: DataFrame with columns:
            - time (datetime, index)
            - open, high, low, close, tick_volume, spread, real_volume
            Returns None if data retrieval fails.
    """
    # Resolve timeframe constant
    tf = TIMEFRAME_MAP.get(timeframe.upper())
    if tf is None:
        logger.error(f"Invalid timeframe '{timeframe}'. Valid options: {list(TIMEFRAME_MAP.keys())}")
        return None

    # Validate symbol exists in MT5
    symbol_info = mt5.symbol_info(symbol)
    if symbol_info is None:
        logger.error(f"Symbol '{symbol}' not found in MT5. Check symbol name.")
        return None

    # Ensure symbol is enabled for trading/quoting
    if not symbol_info.visible:
        logger.info(f"Symbol '{symbol}' is not visible. Attempting to select...")
        if not mt5.symbol_select(symbol, True):
            logger.error(f"Failed to select symbol '{symbol}' in Market Watch.")
            return None

    # Fetch rate data
    rates = mt5.copy_rates_from_pos(symbol, tf, 0, count)

    if rates is None or len(rates) == 0:
        error_code = mt5.last_error()
        logger.error(f"Failed to fetch candles for {symbol} ({timeframe}). Error: {error_code}")
        return None

    # Convert to DataFrame
    df = pd.DataFrame(rates)

    # Convert timestamp to datetime
    df["time"] = pd.to_datetime(df["time"], unit="s")

    # Set time as index
    df.set_index("time", inplace=True)

    logger.debug(f"Fetched {len(df)} candles for {symbol} ({timeframe})")
    return df


def get_candles_multiple_timeframes(
    symbol: str,
    timeframes: Optional[List[str]] = None,
    count: int = 100,
) -> dict:
    """
    Fetch candles for a symbol across multiple timeframes.

    Args:
        symbol: MT5 symbol name.
        timeframes: List of timeframe strings (default: from Config).
        count: Number of candles per timeframe.

    Returns:
        dict: {timeframe_label: pd.DataFrame} for each successful fetch.
    """
    if timeframes is None:
        timeframes = Config.DEFAULT_TIMEFRAMES

    result = {}
    for tf in timeframes:
        df = get_candles(symbol=symbol, timeframe=tf, count=count)
        if df is not None:
            result[tf] = df

    logger.info(f"Retrieved data for {symbol}: {list(result.keys())}")
    return result


def get_available_symbols() -> List[str]:
    """
    List all symbols available in the MT5 Market Watch.

    Returns:
        List[str]: Sorted list of symbol names.
    """
    symbols = mt5.symbols_get()
    if symbols is None:
        logger.error("Failed to retrieve symbols list from MT5.")
        return []

    names = [s.name for s in symbols]
    names.sort()
    logger.debug(f"Retrieved {len(names)} symbols from MT5.")
    return names