"""
Logging utility for the AI Trading Bot.

Provides structured logging with rotation support and multiple log levels.
Ensures logs directory exists at startup.
Rotates logs daily to prevent VPS disk overflow.
"""

import logging
import sys
from pathlib import Path
from logging.handlers import RotatingFileHandler

from trading_bot.config import Config


LOG_DIR = Path("logs")


def setup_logger(name: str = "TradingBot") -> logging.Logger:
    """
    Set up and return a configured logger instance.

    Args:
        name: The logger name (default: "TradingBot")

    Returns:
        logging.Logger: Configured logger instance
    """
    logger_instance = logging.getLogger(name)
    logger_instance.setLevel(getattr(logging, Config.LOG_LEVEL.upper(), logging.INFO))

    # Prevent duplicate handlers if logger is re-initialized
    if logger_instance.handlers:
        return logger_instance

    # --- Console handler ---
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.DEBUG)
    console_fmt = logging.Formatter(
        "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    console_handler.setFormatter(console_fmt)
    logger_instance.addHandler(console_handler)

    # --- Ensure logs directory exists ---
    log_dir = Path("logs")
    log_dir.mkdir(parents=True, exist_ok=True)

    # --- File handler with rotation (10 MB per file, 5 backup files) ---
    file_handler = RotatingFileHandler(
        filename=log_dir / "trading_bot.log",
        maxBytes=10 * 1024 * 1024,  # 10 MB
        backupCount=5,
        encoding="utf-8",
    )
    file_handler.setLevel(logging.DEBUG)
    file_fmt = logging.Formatter(
        "%(asctime)s | %(levelname)-8s | %(name)s | %(module)s:%(lineno)d | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    file_handler.setFormatter(file_fmt)
    logger_instance.addHandler(file_handler)

    # --- Heartbeat log (always-critical events) ---
    heartbeat_log = log_dir / "heartbeat.log"
    heartbeat_handler = logging.FileHandler(
        filename=heartbeat_log,
        encoding="utf-8",
    )
    heartbeat_handler.setLevel(logging.CRITICAL)
    heartbeat_handler.setFormatter(logging.Formatter(
        "%(asctime)s | CRITICAL | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    ))
    logger_instance.addHandler(heartbeat_handler)

    logger_instance.info(f"Logger initialized. Log directory: {log_dir.resolve()}")
    return logger_instance


# Pre-initialized default logger for quick import
logger = setup_logger()