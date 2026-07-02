"""
MetaTrader 5 connection management module.

Handles initialization, login, status checking, and reconnection logic.
This module does NOT execute trades - it only manages the connection
and provides account information for analysis purposes.
"""

import time
from typing import Optional, Tuple

import MetaTrader5 as mt5

from trading_bot.config import Config
from trading_bot.utils.logger import logger


class MT5Connection:
    """
    Manages the lifecycle of an MT5 terminal connection.

    Provides safe initialization, login verification, connection health
    checks, and automated reconnection with exponential backoff.
    """

    def __init__(self):
        self._connected = False
        self._account_info = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def initialize(self) -> bool:
        """
        Initialize the MT5 terminal and attempt to log in.

        Returns:
            bool: True if initialization and login were successful, else False.

        Safe handling:
            - Checks if MT5 is already initialized to prevent double init.
            - Uses configured path, login, password, and server.
            - Logs detailed error messages on failure.
        """
        if self._connected:
            logger.info("MT5 is already initialized and connected.")
            return True

        logger.info("Initializing MT5 terminal...")

        # Step 1: Initialize the MT5 terminal
        initialized = mt5.initialize(
            path=Config.MT5_PATH,
            login=Config.MT5_LOGIN,
            password=Config.MT5_PASSWORD,
            server=Config.MT5_SERVER,
        )

        if not initialized:
            error_code = mt5.last_error()
            logger.error(f"MT5 initialization FAILED. Error code: {error_code}")
            logger.error(
                "Check: MT5 path, login credentials, server name, and "
                "that MT5 terminal is not already running."
            )
            self._shutdown_safe()
            return False

        # Step 2: Verify login status
        if not mt5.login(Config.MT5_LOGIN, Config.MT5_PASSWORD, Config.MT5_SERVER):
            error_code = mt5.last_error()
            logger.error(f"MT5 login FAILED. Error code: {error_code}")
            self._shutdown_safe()
            return False

        # Step 3: Retrieve and log account info
        self._account_info = mt5.account_info()
        if self._account_info:
            logger.info(
                f"Connected to MT5 | Account: {self._account_info.login} | "
                f"Server: {self._account_info.server} | "
                f"Balance: {self._account_info.balance:.2f} | "
                f"Equity: {self._account_info.equity:.2f}"
            )
        else:
            logger.warning("Connected to MT5 but could not retrieve account info.")

        self._connected = True
        logger.info("MT5 connection established successfully.")
        return True

    def is_connected(self) -> bool:
        """
        Check if the MT5 terminal connection is still alive.

        Returns:
            bool: True if connected and terminal is reachable, else False.
        """
        if not self._connected:
            return False

        if not mt5.terminal_info():
            logger.warning("MT5 terminal is no longer reachable.")
            self._connected = False
            return False

        return True

    def ensure_connected(self, max_retries: int = 3, retry_delay: int = 5) -> bool:
        """
        Ensure the MT5 connection is active, with automatic reconnection.

        Args:
            max_retries: Maximum number of reconnection attempts.
            retry_delay: Seconds to wait between retry attempts.

        Returns:
            bool: True if connection is active, False after all retries fail.
        """
        if self.is_connected():
            return True

        logger.warning("MT5 connection lost. Attempting reconnection...")

        for attempt in range(1, max_retries + 1):
            logger.info(f"Reconnection attempt {attempt}/{max_retries}")
            self._shutdown_safe()
            time.sleep(retry_delay)

            if self.initialize():
                logger.info("Reconnection successful.")
                return True

        logger.critical(
            f"Failed to reconnect to MT5 after {max_retries} attempts. "
            "Manual intervention required."
        )
        return False

    def get_account_info(self) -> Optional[dict]:
        """
        Retrieve the current account information as a dictionary.

        Returns:
            Optional[dict]: Account info dict, or None if not connected.
        """
        if not self._connected:
            logger.warning("Cannot get account info: not connected to MT5.")
            return None

        info = mt5.account_info()
        if info is None:
            logger.error("Failed to retrieve account info from MT5.")
            return None

        # Convert to dictionary for safe serialization
        return {
            "login": info.login,
            "server": info.server,
            "balance": info.balance,
            "equity": info.equity,
            "margin": info.margin,
            "margin_free": info.margin_free,
            "margin_level": info.margin_level,
            "currency": info.currency,
            "leverage": info.leverage,
            "name": info.name,
        }

    def shutdown(self):
        """Gracefully shut down the MT5 connection."""
        logger.info("Shutting down MT5 connection...")
        self._shutdown_safe()
        self._connected = False
        self._account_info = None
        logger.info("MT5 connection closed.")

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _shutdown_safe():
        """Safely shut down MT5 without raising exceptions."""
        try:
            mt5.shutdown()
        except Exception as exc:
            logger.debug(f"MT5 shutdown encountered an issue (non-critical): {exc}")