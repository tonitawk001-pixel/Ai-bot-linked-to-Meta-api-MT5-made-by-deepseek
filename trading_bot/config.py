import os
import json
from dotenv import load_dotenv

load_dotenv()


class Config:
    # MT5 Connection (primary account)
    MT5_PATH = os.getenv("MT5_PATH", r"C:\Program Files\MetaTrader 5\terminal64.exe")
    MT5_LOGIN = int(os.getenv("MT5_LOGIN", "0"))
    MT5_PASSWORD = os.getenv("MT5_PASSWORD", "")
    MT5_SERVER = os.getenv("MT5_SERVER", "ICMarkets-Demo")

    # Multi-Account
    ACCOUNTS_JSON = os.getenv("ACCOUNTS_JSON", "[]")
    ACCOUNTS = json.loads(ACCOUNTS_JSON)

    # DeepSeek AI
    DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY", "")

    # Analysis symbols — LOCKED to XAUUSD only
    SYMBOLS = ["XAUUSD"]
    # Ignore environment variable override — bot is XAUUSD-only
    _env_symbols = os.getenv("SYMBOLS", "")
    if _env_symbols and "XAUUSD" in _env_symbols.upper():
        SYMBOLS = ["XAUUSD"]

    # --- Gold Scalping Parameters ---
    GOLD_MIN_TRADES_PER_DAY = int(os.getenv("GOLD_MIN_TRADES_PER_DAY", "3"))
    GOLD_MAX_TRADES_PER_DAY = int(os.getenv("GOLD_MAX_TRADES_PER_DAY", "20"))
    GOLD_MIN_COOLDOWN_MINUTES = int(os.getenv("GOLD_MIN_COOLDOWN_MINUTES", "2"))
    GOLD_MAX_OPEN_POSITIONS = int(os.getenv("GOLD_MAX_OPEN_POSITIONS", "3"))
    GOLD_ATR_HIGH_MULTIPLIER = float(os.getenv("GOLD_ATR_HIGH_MULTIPLIER", "2.0"))
    GOLD_SPREAD_MAX_MULTIPLIER = float(os.getenv("GOLD_SPREAD_MAX_MULTIPLIER", "2.5"))
    GOLD_LOSS_STREAK_RISK_PERCENT = float(os.getenv("GOLD_LOSS_STREAK_RISK_PERCENT", "1.0"))

    # --- Gold Scalping Session Times (UTC) ---
    LONDON_OPEN = int(os.getenv("LONDON_OPEN", "8"))
    LONDON_CLOSE = int(os.getenv("LONDON_CLOSE", "17"))
    NY_OPEN = int(os.getenv("NY_OPEN", "13"))
    NY_CLOSE = int(os.getenv("NY_CLOSE", "22"))

    # --- Gold Timeframes for Scalping ---
    GOLD_TIMEFRAMES = ["M1", "M5", "M15"]

    # Indicator defaults
    RSI_PERIOD = int(os.getenv("RSI_PERIOD", "14"))
    MACD_FAST = int(os.getenv("MACD_FAST", "12"))
    MACD_SLOW = int(os.getenv("MACD_SLOW", "26"))
    MACD_SIGNAL = int(os.getenv("MACD_SIGNAL", "9"))
    EMA_PERIODS = [int(x) for x in os.getenv("EMA_PERIODS", "20,50,200").split(",")]
    ATR_PERIOD = int(os.getenv("ATR_PERIOD", "14"))

    # Data feed
    DEFAULT_TIMEFRAMES = os.getenv("DEFAULT_TIMEFRAMES", "M1,M5,H1").split(",")
    CANDLE_COUNT = int(os.getenv("CANDLE_COUNT", "500"))

    # Logging
    LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")

    # --- Risk Management ---
    MAX_RISK_PERCENT = float(os.getenv("MAX_RISK_PERCENT", "2.0"))
    MAX_DAILY_LOSS_PERCENT = float(os.getenv("MAX_DAILY_LOSS_PERCENT", "5.0"))
    MAX_CONSECUTIVE_LOSSES = int(os.getenv("MAX_CONSECUTIVE_LOSSES", "3"))
    MAX_SPREAD_MULTIPLIER = float(os.getenv("MAX_SPREAD_MULTIPLIER", "2.0"))
    MAX_ATR_PERCENT = float(os.getenv("MAX_ATR_PERCENT", "2.0"))
    AI_MIN_CONFIDENCE = int(os.getenv("AI_MIN_CONFIDENCE", "70"))
    BLOCK_AI_HIGH_RISK = os.getenv("BLOCK_AI_HIGH_RISK", "true").lower() == "true"
    BLOCK_SETUP_INVALID = os.getenv("BLOCK_SETUP_INVALID", "true").lower() == "true"

    # --- Execution ---
    DEFAULT_LOT_SIZE = float(os.getenv("DEFAULT_LOT_SIZE", "0.01"))
    MIN_LOT_SIZE = float(os.getenv("MIN_LOT_SIZE", "0.01"))
    MAX_LOT_SIZE = float(os.getenv("MAX_LOT_SIZE", "10.0"))
    SL_PIPS = int(os.getenv("SL_PIPS", "20"))
    TP_PIPS = int(os.getenv("TP_PIPS", "40"))
    EXECUTION_ENABLED = os.getenv("EXECUTION_ENABLED", "false").lower() == "true"
    DEFAULT_TIMEFRAME_EXEC = os.getenv("DEFAULT_TIMEFRAME_EXEC", "H1")

    # Emergency stop
    EMERGENCY_STOP = False

    # --- Simulation / Backtest ---
    SIMULATION_MODE = os.getenv("SIMULATION_MODE", "true").lower() == "true"
    SIMULATION_INITIAL_BALANCE = float(os.getenv("SIMULATION_INITIAL_BALANCE", "10000.0"))
    SIMULATION_SPREAD_COST = float(os.getenv("SIMULATION_SPREAD_COST", "0.0001"))
    SIMULATION_SLIPPAGE = float(os.getenv("SIMULATION_SLIPPAGE", "0.00005"))
    SIMULATION_FAIL_RATE = float(os.getenv("SIMULATION_FAIL_RATE", "0.05"))

    # --- Continuous Run (VPS 24/7) ---
    CONTINUOUS_MODE = os.getenv("CONTINUOUS_MODE", "false").lower() == "true"
    ANALYSIS_INTERVAL_MINUTES = int(os.getenv("ANALYSIS_INTERVAL_MINUTES", "60"))

    # --- Safety Audit Limits ---
    MAX_DRAWDOWN_PERCENT = float(os.getenv("MAX_DRAWDOWN_PERCENT", "20.0"))
    MIN_WIN_RATE = float(os.getenv("MIN_WIN_RATE", "40.0"))
    MAX_RISK_BLOCK_RATE = float(os.getenv("MAX_RISK_BLOCK_RATE", "80.0"))