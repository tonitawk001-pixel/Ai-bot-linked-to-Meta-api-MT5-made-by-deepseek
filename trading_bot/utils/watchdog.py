"""
Global Watchdog — Monitors main loop heartbeat and restarts if frozen.

Features:
- Monitor heartbeat every 60 seconds
- If no heartbeat for 120 seconds → restart main trading process
- Prevent infinite restart loops (max 5 restarts per hour)
- Log all restart events with timestamp
"""

import time
import os
import sys
import signal
from datetime import datetime, timedelta
from threading import Thread, Event
from typing import Optional

from trading_bot.utils.logger import logger


class Watchdog:
    """
    Monitors system heartbeat and triggers recovery if frozen.

    Runs in a separate thread and checks for heartbeat updates.
    If heartbeat stops updating, it logs the event and can restart
    the main process.
    """

    def __init__(self, timeout_seconds: int = 120, check_interval: int = 30):
        self.timeout = timeout_seconds
        self.check_interval = check_interval
        self._last_heartbeat: Optional[datetime] = None
        self._stop_event = Event()
        self._thread: Optional[Thread] = None
        self._restart_count = 0
        self._restart_window: list = []  # timestamps of restarts in current hour
        self._max_restarts_per_hour = 5
        logger.info(f"Watchdog initialized (timeout={timeout_seconds}s, check={check_interval}s)")

    def start(self):
        """Start the watchdog monitoring thread."""
        if self._thread and self._thread.is_alive():
            logger.debug("Watchdog already running")
            return
        self._last_heartbeat = datetime.now()
        self._stop_event.clear()
        self._thread = Thread(target=self._run, daemon=True, name="watchdog")
        self._thread.start()
        logger.info("Watchdog started")

    def stop(self):
        """Stop the watchdog thread."""
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=5)
        logger.info("Watchdog stopped")

    def heartbeat(self):
        """Signal that the main loop is still alive. Call this every cycle."""
        self._last_heartbeat = datetime.now()

    def _run(self):
        """Watchdog monitoring loop."""
        while not self._stop_event.is_set():
            try:
                self._check()
            except Exception as e:
                logger.error(f"Watchdog check error: {e}")
            self._stop_event.wait(timeout=self.check_interval)

    def _check(self):
        """Check if heartbeat is still alive."""
        if self._last_heartbeat is None:
            return

        elapsed = (datetime.now() - self._last_heartbeat).total_seconds()

        if elapsed > self.timeout:
            logger.critical(
                f"WATCHDOG: No heartbeat for {elapsed:.0f}s (timeout={self.timeout}s). "
                f"System may be frozen."
            )
            self._attempt_recovery()
        else:
            logger.debug(f"Watchdog: heartbeat OK (last={elapsed:.0f}s ago)")

    def _attempt_recovery(self):
        """Attempt to recover the system."""
        now = datetime.now()

        # Clean restart window (keep only restarts from last hour)
        self._restart_window = [t for t in self._restart_window if t > now - timedelta(hours=1)]

        if len(self._restart_window) >= self._max_restarts_per_hour:
            logger.critical(
                f"WATCHDOG: {self._max_restarts_per_hour} restarts in last hour. "
                f"Max limit reached. System will NOT auto-restart."
            )
            return

        self._restart_count += 1
        self._restart_window.append(now)

        logger.critical(f"WATCHDOG: Attempting recovery #{self._restart_count}...")

        # Option 1: Try to restart via systemd (preferred on VPS)
        try:
            import subprocess
            subprocess.run(["systemctl", "--user", "restart", "trading-bot"],
                          capture_output=True, timeout=10)
            logger.info("Watchdog: Sent restart via systemctl")
            return
        except Exception as e:
            logger.warning(f"Watchdog: systemctl restart failed: {e}")

        # Option 2: Exit with error code (systemd will restart automatically)
        logger.critical("Watchdog: Exiting process. systemd should restart automatically.")
        os._exit(1)

    @property
    def status(self) -> dict:
        """Get watchdog status."""
        elapsed = (datetime.now() - self._last_heartbeat).total_seconds() if self._last_heartbeat else -1
        return {
            "heartbeat_seconds_ago": round(elapsed, 1),
            "timeout_seconds": self.timeout,
            "restart_count": self._restart_count,
            "restarts_last_hour": len([t for t in self._restart_window
                                       if t > datetime.now() - timedelta(hours=1)]),
            "running": self._thread is not None and self._thread.is_alive(),
        }


# Singleton watchdog for easy import
_watchdog: Optional[Watchdog] = None


def get_watchdog() -> Watchdog:
    """Get or create the global watchdog singleton."""
    global _watchdog
    if _watchdog is None:
        _watchdog = Watchdog()
    return _watchdog