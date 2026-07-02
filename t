"""
MT5 Health Monitor — Continuously checks MT5 connection status.

Features:
- Check MT5 connection every cycle
- Verify account login is active
- Detect silent disconnection (no data received)
- Trigger immediate reconnect on failure
- Log disconnection events as CRITICAL
- Pause trading until reconnection succeeds
"""

import time
from datetime import datetime, timedelta
from typing import Optional

import MetaTrader5 as mt5

from trading_bot.config import Config
from trading_bot.utils.logger import logger


class MT5HealthMonitor:
    """
    Monitors MT5 connection health and handles reconnection.

    Tracks:
    - Connection status (connected/disconnected)
    - Last data received timestamp
    - Reconnection attempts
    - Silent disconnection detection
    """

    def __init__(self, max_reconnect_retries: int = 10, reconnect_delay: int = 5):
        self._connected = False
        self._last_data_time: Optional[datetime] = None
        self._reconnect_count = 0
        self._max_retries = max_reconnect_retries
        self._reconnect_delay = reconnect_delay
        self._paused = False
        self._last_check_time: Optional[datetime] = None
        logger.info("MT5HealthMonitor initialized")

    def check_connection(self) -> bool:
        """
        Check if MT5 connection is healthy.

        Returns:
            bool: True if connected and healthy, False otherwise.
        """
        self._last_check_time = datetime.now()

        # Check 1: Is terminal reachable?
        terminal_info = None
        try:
            terminal_info = mt5.terminal_info()
        except Exception as e:
            logger.error(f"MT5 terminal_info() raised exception: {e}")

        if terminal_info is None:
            self._connected = False
            logger.critical("MT5 HEALTH: Terminal unreachable — NOT CONNECTED")
            self._trigger_reconnect()
            return False

        # Check 2: Is account logged in?
        account_info = None
        try:
            account_info = mt5.account_info()
        except Exception as e:
            logger.error(f"MT5 account_info() raised exception: {e}")

        if account_info is None:
            self._connected = False
            logger.critical("MT5 HEALTH: Account login lost — NOT CONNECTED")
            self._trigger_reconnect()
            return False

        # Check 3: Are we receiving data? (silent disconnection)
        if self._last_data_time:
            time_since_data = (datetime.now() - self._last_data_time).total_seconds()
            if time_since_data > 300:  # 5 minutes without data
                logger.warning(f"MT5 HEALTH: No data for {time_since_data:.0f}s — possible silent disconnection")
                self._trigger_reconnect()
                return False

        # All checks passed
        if not self._connected:
            logger.info("MT5 HEALTH: Connection RESTORED")
            self._reconnect_count = 0

        self._connected = True
        self._paused = False
        logger.debug("MT5 HEALTH: Connected and healthy")
        return True

    def report_data_received(self):
        """Call this when data is successfully fetched from MT5."""
        self._last_data_time = datetime.now()

    def _trigger_reconnect(self):
        """Attempt to reconnect MT5."""
        self._reconnect_count += 1
        logger.critical(f"MT5 HEALTH: Reconnecting (attempt #{self._reconnect_count})...")

        if self._reconnect_count > self._max_retries:
            logger.critical(f"MT5 HEALTH: Max reconnection attempts ({self._max_retries}) exceeded. Pausing trading.")
            self._paused = True
            return

        try:
            mt5.shutdown()
        except:
            pass

        time.sleep(self._reconnect_delay)

        try:
            initialized = mt5.initialize(
                path=Config.MT5_PATH,
                login=Config.MT5_LOGIN,
                password=Config.MT5_PASSWORD,
                server=Config.MT5_SERVER,
            )
            if initialized:
                self._connected = True
                self._paused = False
                self._reconnect_count = 0
                logger.info("MT5 HEALTH: Reconnection SUCCESSFUL")
            else:
                logger.error(f"MT5 HEALTH: Reconnection FAILED (attempt #{self._reconnect_count})")
        except Exception as e:
            logger.error(f"MT5 HEALTH: Reconnection exception: {e}")

    @property
    def is_healthy(self) -> bool:
        return self._connected and not self._paused

    @property
    def is_paused(self) -> bool:
        return self._paused

    @property
    def status(self) -> dict:
        """Get detailed health status."""
        return {
            "connected": self._connected,
            "paused": self._paused,
            "reconnect_attempts": self._reconnect_count,
            "last_check": str(self._last_check_time) if self._last_check_time else None,
            "last_data_time": str(self._last_data_time) if self._last_data_time else None,
        }