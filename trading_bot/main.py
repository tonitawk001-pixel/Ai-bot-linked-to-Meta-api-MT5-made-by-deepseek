"""
AI Trading Bot — Main Entry Point (XAUUSD Gold Scalping)

Full pipeline: DATA → INDICATORS → GOLD SCALPING STRATEGY → VOLATILITY FILTER
               → NEWS FILTER → AI (optional) → RISK MANAGER → EXECUTION

Supports:
- Single-pass analysis mode
- Continuous 24/7 loop mode (VPS)
- Auto-reconnection on MT5/API failure

LOCKED to XAUUSD only — all other symbols are ignored.
"""

import sys
import json
import time
import os
import math
from datetime import datetime, timedelta
import pandas as pd
import numpy as np

from trading_bot.config import Config
from trading_bot.utils.logger import logger
from trading_bot.mt5.connection import MT5Connection
from trading_bot.mt5.data_feed import get_candles, TIMEFRAME_MAP
from trading_bot.indicators.technical_indicators import compute_all_indicators
from trading_bot.strategy.gold_scalping_strategy import GoldScalpingStrategy
from trading_bot.strategy.gold_volatility_filter import GoldVolatilityFilter
from trading_bot.ai.deepseek_client import DeepSeekClient
from trading_bot.risk.risk_manager import RiskManager
from trading_bot.execution.mt5_executor import execute_trade
from trading_bot.news.news_aggregator import NewsAggregator

import MetaTrader5 as mt5


# ---------------------------------------------------------------------------
# Symbol restriction
# ---------------------------------------------------------------------------

def is_valid_symbol(symbol: str) -> bool:
    """Only XAUUSD is allowed."""
    return symbol.strip().upper() == "XAUUSD"


# ---------------------------------------------------------------------------
# AI payload builder (adapted for gold scalping)
# ---------------------------------------------------------------------------

def build_gold_ai_payload(
    symbol: str,
    strategy_result: dict,
    m5_ohlcv: pd.DataFrame,
    m5_indicators: dict,
    news_context: dict = None,
    account_info: dict = None,
) -> dict:
    """
    Build DeepSeek payload from gold scalping strategy output.
    DeepSeek is advisory only — reads context, returns sentiment/confidence/risk.
    """
    close = m5_ohlcv["close"].tolist()
    latest_close = round(float(close[-1]), 2) if close else 0

    rsi_val = None
    rsi_series = m5_indicators.get("rsi", pd.Series(dtype=float))
    if not rsi_series.empty:
        try:
            rsi_val = round(float(rsi_series.iloc[-1]), 1)
        except (IndexError, ValueError):
            pass

    atr_val = None
    atr_series = m5_indicators.get("atr", pd.Series(dtype=float))
    if not atr_series.empty:
        try:
            atr_val = round(float(atr_series.iloc[-1]), 5)
        except (IndexError, ValueError):
            pass

    payload = {
        "symbol": symbol,
        "strategy": "gold_scalping",
        "price_data": {
            "latest_close": latest_close,
            "close_last_5": close[-5:] if len(close) >= 5 else close,
        },
        "indicators": {
            "rsi_m5": rsi_val,
            "atr_m5": atr_val,
            "ema_trend": strategy_result.get("bias", "neutral"),
        },
        "strategy_result": {
            "setup_score": strategy_result.get("setup_score", 0),
            "direction": strategy_result.get("direction", "NONE"),
            "confidence": strategy_result.get("confidence", 0.0),
            "session": strategy_result.get("session", "unknown"),
            "reason": strategy_result.get("reason", ""),
        },
        "market_context": {
            "candles_analyzed": len(m5_ohlcv),
            "pullback_detected": strategy_result.get("pullback_detected", False),
            "entry_trigger": strategy_result.get("entry_trigger", False),
        },
    }

    if news_context:
        payload["news_context"] = {
            "global_risk_mode": news_context.get("global_risk_mode", "low"),
            "news_items_count": news_context.get("news_items_count", 0),
        }

    if account_info:
        payload["account"] = {
            "balance": account_info.get("balance", 0),
            "equity": account_info.get("equity", 0),
            "current_drawdown_pct": account_info.get("drawdown_pct", 0),
            "risk_per_trade": 0.02,
        }

    return payload


# ---------------------------------------------------------------------------
# Lot size computation with gold-specific scaling
# ---------------------------------------------------------------------------

def compute_gold_lot_size(
    account_balance: float,
    atr_value: float,
    m5_ohlcv: pd.DataFrame,
    gold_vol_filter_result: dict = None,
) -> float:
    """
    Compute XAUUSD lot size based on 2% risk.
    Gold lot sizing: standard lot = 100 oz, so pip value differs.
    """
    if atr_value is None or atr_value <= 0:
        return Config.DEFAULT_LOT_SIZE

    risk_amount = account_balance * (Config.MAX_RISK_PERCENT / 100.0)
    sl_distance = atr_value * 1.5  # 1.5 ATR stop loss

    # XAUUSD: 1 lot = 100 oz, 1 point = $1 per 0.01 lot
    # For simplicity: lot * sl_distance * 100 = risk_amount
    lot = risk_amount / (sl_distance * 100) if sl_distance > 0 else Config.DEFAULT_LOT_SIZE

    # Apply gold volatility lot reduction if available
    if gold_vol_filter_result:
        vol_lot_factor = gold_vol_filter_result.get("lot_reduction_factor", 1.0)
        lot *= vol_lot_factor

    return max(Config.MIN_LOT_SIZE, min(lot, Config.MAX_LOT_SIZE))


# ---------------------------------------------------------------------------
# Compute SL/TP for gold scalping
# ---------------------------------------------------------------------------

def compute_gold_sl_tp(action: str, ohlcv: pd.DataFrame, atr_value: float) -> tuple:
    """
    Compute stop loss and take profit for XAUUSD scalping.
    SL: 1.5x ATR, TP: 3.0x ATR (scalp-friendly ratio).
    """
    try:
        close = float(ohlcv["close"].iloc[-1])
    except (IndexError, ValueError):
        return 0.0, 0.0

    if atr_value is None or atr_value <= 0:
        return 0.0, 0.0

    sl_distance = atr_value * 1.5
    tp_distance = atr_value * 3.0

    if action == "BUY":
        return round(close - sl_distance, 2), round(close + tp_distance, 2)
    elif action == "SELL":
        return round(close + sl_distance, 2), round(close - tp_distance, 2)
    return 0.0, 0.0


# ---------------------------------------------------------------------------
# Count open XAUUSD positions via MT5
# ---------------------------------------------------------------------------

def count_open_xauusd_positions() -> int:
    """Count currently open XAUUSD positions across all accounts."""
    try:
        positions = mt5.positions_get(symbol="XAUUSD")
        if positions is None:
            return 0
        return len(positions)
    except Exception:
        return 0


# ---------------------------------------------------------------------------
# Main analysis + execution for XAUUSD gold scalping
# ---------------------------------------------------------------------------

def analyze_and_execute_gold(
    symbol: str,
    strategy: GoldScalpingStrategy,
    vol_filter: GoldVolatilityFilter,
    deepseek: DeepSeekClient,
    risk_manager: RiskManager,
    news_agg: NewsAggregator,
    account_id: str = "default",
    account_balance: float = 10000.0,
    candle_count: int = 300,
) -> dict:
    """
    Full gold scalping pipeline for XAUUSD.

    1. Fetch M1, M5, M15 data
    2. Compute indicators on all timeframes
    3. Run GoldScalpingStrategy
    4. Apply GoldVolatilityFilter
    5. Check news blackout
    6. Run DeepSeek (optional, advisory only)
    7. RiskManager.validate() + gold_specific_adjustments()
    8. Position limit checks
    9. Execute if approved
    """

    if not is_valid_symbol(symbol):
        logger.warning(f"Symbol {symbol} is not XAUUSD — skipping.")
        return {"error": "Invalid symbol", "symbol": symbol}

    logger.info(f"\n{'='*60}\nXAUUSD Gold Scalping | Account: {account_id} | Balance: ${account_balance:.2f}\n{'='*60}")

    # Reset daily counter at midnight
    strategy.reset_daily()

    # --- Step 1: Fetch all timeframes ---
    timeframes = Config.GOLD_TIMEFRAMES
    data = {}
    for tf in timeframes:
        try:
            df = get_candles(symbol="XAUUSD", timeframe=tf, count=candle_count)
            if df is None or df.empty:
                logger.warning(f"No data for XAUUSD ({tf})")
                return {"error": f"No data for {tf}"}
            data[tf] = df
        except Exception as e:
            logger.error(f"Data fetch failed for XAUUSD {tf}: {e}")
            return {"error": f"Fetch failed: {e}"}

    # --- Step 2: Compute indicators on all timeframes ---
    indicators = {}
    for tf in timeframes:
        try:
            indicators[tf] = compute_all_indicators(data[tf])
        except Exception as e:
            logger.error(f"Indicator computation failed for {tf}: {e}")
            return {"error": f"Indicators error {tf}: {e}"}

    m1_ohlcv = data["M1"]
    m5_ohlcv = data["M5"]
    m15_ohlcv = data["M15"]
    m1_ind = indicators["M1"]
    m5_ind = indicators["M5"]
    m15_ind = indicators["M15"]

    # --- Step 3: News context ---
    news_context = news_agg.get_news_context() if news_agg else None
    news_overlay = news_agg.get_risk_overlay() if news_agg else None

    # If news overlay blocks all trades, exit early
    if news_overlay and news_overlay.get("news_block_all_trades", False):
        logger.info("News blackout active — no trades allowed this cycle.")
        return {
            "symbol": symbol,
            "decision": "BLOCKED_NEWS",
            "reason": news_overlay.get("reason", "News blackout"),
            "session": strategy._detect_session(),
        }

    # --- Step 4: Run gold scalping strategy ---
    strategy_result = strategy.analyze(
        m1_indicators=m1_ind,
        m5_indicators=m5_ind,
        m15_indicators=m15_ind,
        m1_ohlcv=m1_ohlcv,
        m5_ohlcv=m5_ohlcv,
        m15_ohlcv=m15_ohlcv,
        news_context=news_context,
    )

    direction = strategy_result.get("direction", "NONE")
    setup_score = strategy_result.get("setup_score", 0)
    confidence = strategy_result.get("confidence", 0.0)

    logger.info(
        f"Gold Scalping Result: dir={direction} score={setup_score} "
        f"conf={confidence:.2f} session={strategy_result.get('session')} "
        f"reason={strategy_result.get('reason', '')[:100]}"
    )

    # If no valid direction from strategy, exit
    if direction == "NONE":
        logger.info("Gold scalping: no valid setup detected.")
        return {
            "symbol": symbol,
            "decision": "NO_SETUP",
            "strategy_result": strategy_result,
            "session": strategy_result.get("session"),
        }

    # --- Step 5: Gold volatility filter ---
    vol_filter_result = vol_filter.analyze(
        m1_ohlcv=m1_ohlcv,
        m5_ohlcv=m5_ohlcv,
        m15_ohlcv=m15_ohlcv,
        m1_indicators=m1_ind,
        m5_indicators=m5_ind,
        m15_indicators=m15_ind,
    )

    if not vol_filter_result.get("trade_ok", False):
        logger.info(f"Gold volatility filter BLOCKED: {vol_filter_result.get('reason')}")
        return {
            "symbol": symbol,
            "decision": "BLOCKED_VOLATILITY",
            "strategy_result": strategy_result,
            "vol_filter": vol_filter_result,
            "session": strategy_result.get("session"),
            "reason": vol_filter_result.get("reason", "Volatility block"),
        }

    # --- Step 6: Position limits ---
    open_count = count_open_xauusd_positions()
    can_trade, limit_reason = strategy.can_trade(open_count)
    if not can_trade:
        logger.info(f"Position limit reached: {limit_reason}")
        return {
            "symbol": symbol,
            "decision": "BLOCKED_LIMITS",
            "strategy_result": strategy_result,
            "vol_filter": vol_filter_result,
            "session": strategy_result.get("session"),
            "reason": limit_reason,
            "open_positions": open_count,
        }

    # --- Step 7: DeepSeek AI (optional, advisory only) ---
    account_summary = risk_manager.get_account_summary(account_id)
    ai_payload = build_gold_ai_payload(
        symbol=symbol,
        strategy_result=strategy_result,
        m5_ohlcv=m5_ohlcv,
        m5_indicators=m5_ind,
        news_context=news_context,
        account_info=account_summary,
    )
    ai_analysis = deepseek.analyze_market(ai_payload)

    # --- Step 8: Risk Manager validation ---
    # Build rule_decision compatible dict for RiskManager
    rule_decision = {
        "symbol": symbol,
        "timeframe": "M5",
        "trend": strategy_result.get("bias", "neutral"),
        "setup_valid": setup_score >= 40,
        "setup_strength": setup_score,
        "atr_value": float(m5_ind["atr"].iloc[-1]) if not m5_ind["atr"].empty else None,
        "volatility": vol_filter_result.get("market_regime", "medium"),
        "rsi_condition": "neutral",
        "rsi_value": None,
    }

    base_risk_eval = risk_manager.validate(
        rule_decision=rule_decision,
        ai_analysis=ai_analysis,
        ohlcv=m5_ohlcv,
        news_overlay=news_overlay,
    )

    # Apply gold-specific adjustments on top
    risk_eval = risk_manager.gold_specific_adjustments(
        base_result=base_risk_eval,
        account_balance=account_balance,
        volatility_info=vol_filter_result,
    )

    if not risk_eval.get("approved", False):
        logger.info(f"Risk Manager BLOCKED: {risk_eval.get('reason', 'Unknown')}")
        return {
            "symbol": symbol,
            "decision": "BLOCKED_RISK",
            "strategy_result": strategy_result,
            "vol_filter": vol_filter_result,
            "risk_eval": risk_eval,
            "session": strategy_result.get("session"),
            "reason": risk_eval.get("reason", "Risk block"),
        }

    # --- Step 9: Compute lot size, SL, TP ---
    atr_value = m5_ind["atr"].iloc[-1] if not m5_ind["atr"].empty else None
    atr_value = float(atr_value) if atr_value is not None else None

    lot = compute_gold_lot_size(
        account_balance=account_balance,
        atr_value=atr_value,
        m5_ohlcv=m5_ohlcv,
        gold_vol_filter_result=vol_filter_result,
    )

    sl, tp = compute_gold_sl_tp(direction, m5_ohlcv, atr_value)

    # Apply RiskManager lot scaling
    max_lot = risk_eval.get("max_lot_size", lot)
    adjusted_lot_scale = risk_eval.get("adjusted_lot_scale", 1.0)
    final_lot = round(lot * adjusted_lot_scale, 2)
    final_lot = max(Config.MIN_LOT_SIZE, min(final_lot, Config.MAX_LOT_SIZE))

    logger.info(
        f"Gold TL: lot={final_lot} sl={sl} tp={tp} "
        f"scale={adjusted_lot_scale:.2f} risk_score={risk_eval.get('risk_score', 0)}"
    )

    # --- Step 10: Execution ---
    exec_results = []
    final_decision = "WAIT"

    if Config.EXECUTION_ENABLED:
        logger.info(f"EXECUTING {direction} {final_lot} XAUUSD")
        try:
            exec_results = execute_trade(
                action=direction,
                symbol="XAUUSD",
                lot_size=final_lot,
                sl=sl,
                tp=tp,
                ohlcv=m5_ohlcv,
                risk_evaluation=risk_eval,
            )
            # Record trade in strategy tracker
            strategy.record_trade()
            final_decision = "EXECUTED"
        except Exception as e:
            logger.error(f"Execution failed: {e}")
            exec_results = [{"account": account_id, "success": False, "reason": str(e)}]
            final_decision = "EXECUTION_FAILED"
    else:
        logger.info(f"[PAPER] Gold scalping signal: {direction} {final_lot} XAUUSD SL={sl} TP={tp}")
        final_decision = "PAPER_TRADE"
        # Record paper trade
        strategy.record_trade()

    # --- Step 11: Build result ---
    result = {
        "symbol": symbol,
        "strategy_result": strategy_result,
        "vol_filter": vol_filter_result,
        "ai_analysis": ai_analysis,
        "risk_eval": risk_eval,
        "trade": {
            "action": direction,
            "lot_size": final_lot,
            "sl": sl,
            "tp": tp,
            "final_decision": final_decision,
            "execution_results": exec_results,
        },
        "session": strategy_result.get("session"),
        "open_positions": open_count,
    }

    logger.info(
        f"XAUUSD {strategy_result.get('session')}: "
        f"{direction} score={setup_score} lot={final_lot} "
        f"risk={'OK' if risk_eval.get('approved') else 'BLOCK'} final={final_decision}"
    )

    return result


# ---------------------------------------------------------------------------
# Live analysis mode (single pass)
# ---------------------------------------------------------------------------

def run_live_analysis():
    logger.info("=" * 60 + "\nXAUUSD GOLD SCALPING BOT — LIVE ANALYSIS MODE\n" + "=" * 60)

    mt5_conn = MT5Connection()
    if not mt5_conn.initialize():
        logger.critical("MT5 connection failed. Check MT5 terminal and credentials.")
        return False

    balance = mt5_conn.get_account_info().get("balance", 10000.0) if mt5_conn.get_account_info() else 10000.0
    logger.info(f"Account balance: ${balance:.2f}")

    # Initialize DeepSeek
    deepseek = DeepSeekClient()
    ai_avail = deepseek.initialize()
    if not ai_avail:
        logger.warning("DeepSeek unavailable — continuing with strategy signals only.")

    # Initialize News Aggregator
    news_agg = NewsAggregator(update_interval_minutes=15)
    try:
        news_agg.update(force=True)
        logger.info(f"News: risk_mode={news_agg.get_news_context().get('global_risk_mode', 'unknown')}")
    except Exception as e:
        logger.warning(f"News update failed: {e}")

    # Initialize core components
    risk_manager = RiskManager(default_balance=balance)
    strategy = GoldScalpingStrategy()
    vol_filter = GoldVolatilityFilter()

    # Analyze XAUUSD only
    result = analyze_and_execute_gold(
        symbol="XAUUSD",
        strategy=strategy,
        vol_filter=vol_filter,
        deepseek=deepseek,
        risk_manager=risk_manager,
        news_agg=news_agg,
        account_id="XAUUSD",
        account_balance=balance,
    )

    # Log summary
    logger.info(f"\n{'='*60}")
    logger.info(f"Analysis complete for XAUUSD")
    logger.info(f"Account summary: {json.dumps(risk_manager.all_accounts(), indent=2)}")
    logger.info(f"{'='*60}")

    deepseek.shutdown()
    mt5_conn.shutdown()
    return True


# ---------------------------------------------------------------------------
# Continuous 24/7 loop mode (VPS)
# ---------------------------------------------------------------------------

def run_continuous_loop():
    """
    Continuous 24/7 loop for VPS operation — XAUUSD gold scalping.

    Runs analysis at configurable intervals.
    Auto-reconnects MT5 on failure.
    Survives SSH disconnection.
    """
    logger.info("=" * 60)
    logger.info("XAUUSD GOLD SCALPING BOT — CONTINUOUS 24/7 MODE")
    logger.info(f"Interval: {getattr(Config, 'ANALYSIS_INTERVAL_MINUTES', 60)} minutes")
    logger.info(f"Execution enabled: {Config.EXECUTION_ENABLED}")
    logger.info("=" * 60)

    interval = getattr(Config, 'ANALYSIS_INTERVAL_MINUTES', 60)
    cycle_count = 0

    # Initialize persistent components once
    mt5_conn = MT5Connection()
    deepseek = None
    news_agg = None
    risk_manager = None
    strategy = GoldScalpingStrategy()
    vol_filter = GoldVolatilityFilter()

    while True:
        cycle_count += 1
        logger.info(f"\n{'='*60}\nCYCLE #{cycle_count} — {datetime.now().isoformat()}\n{'='*60}")

        try:
            # Reconnect MT5 if needed
            if not mt5_conn.initialize():
                logger.warning("MT5 reconnect attempt...")
                time.sleep(30)
                continue

            balance = mt5_conn.get_account_info().get("balance", 10000.0) if mt5_conn.get_account_info() else 10000.0

            # Initialize or update DeepSeek
            if deepseek is None:
                deepseek = DeepSeekClient()
                deepseek.initialize()

            # Initialize or update News
            if news_agg is None:
                news_agg = NewsAggregator(update_interval_minutes=15)
            try:
                news_agg.update(force=(cycle_count == 1))
            except Exception:
                pass

            # Initialize RiskManager
            if risk_manager is None:
                risk_manager = RiskManager(default_balance=balance)
            risk_manager.set_balance(balance)

            # Run gold scalping cycle
            result = analyze_and_execute_gold(
                symbol="XAUUSD",
                strategy=strategy,
                vol_filter=vol_filter,
                deepseek=deepseek,
                risk_manager=risk_manager,
                news_agg=news_agg,
                account_id="XAUUSD",
                account_balance=balance,
            )

            # Log next run time
            next_run = datetime.now() + timedelta(minutes=interval)
            logger.info(f"Cycle #{cycle_count} complete. Next run at {next_run.isoformat()}")

        except KeyboardInterrupt:
            logger.info("Stopped by user (Ctrl+C).")
            break

        except Exception as e:
            logger.critical(f"Unhandled error in cycle #{cycle_count}: {e}", exc_info=True)
            logger.info("Waiting 60s before retry...")
            time.sleep(60)

        # Wait for next interval
        logger.info(f"Sleeping for {interval} minute(s)...")
        for remaining in range(interval * 60, 0, -60):
            if remaining % 300 == 0:
                logger.debug(f"  Next cycle in {remaining // 60}m")
            time.sleep(min(60, remaining))

    # Cleanup
    if deepseek:
        deepseek.shutdown()
    mt5_conn.shutdown()


# ---------------------------------------------------------------------------
# Simulation / Backtest mode (unchanged, uses old pipeline for now)
# ---------------------------------------------------------------------------

def run_simulation_mode():
    from trading_bot.backtest.simulation_engine import run_simulation
    logger.info("=" * 60 + "\nSIMULATION / BACKTEST MODE\n" + "=" * 60)
    mt5_conn = MT5Connection()
    if not mt5_conn.initialize():
        logger.critical("Failed to connect to MT5.")
        return
    deepseek = DeepSeekClient()
    deepseek.initialize()
    news_agg = NewsAggregator(update_interval_minutes=15)
    try:
        news_agg.update(force=True)
    except:
        pass
    risk_manager = RiskManager(default_balance=Config.SIMULATION_INITIAL_BALANCE)
    from trading_bot.strategy.rule_engine import RuleEngine
    rule_engine = RuleEngine()
    for symbol in Config.SYMBOLS:
        symbol = symbol.strip()
        if not symbol or not is_valid_symbol(symbol):
            continue
        for tf in [Config.DEFAULT_TIMEFRAME_EXEC]:
            result = run_simulation(
                symbol=symbol, timeframe=tf, candle_count=Config.CANDLE_COUNT,
                rule_engine=rule_engine, deepseek=deepseek, risk_manager=risk_manager,
            )
            if "error" not in result:
                audit = result.get("safety_audit", {})
                print(
                    f"\n{symbol}_{tf}: Trades={audit.get('total_trades')} "
                    f"WR={audit.get('win_rate')}% "
                    f"PF={audit.get('profit_factor')} "
                    f"DD={audit.get('max_drawdown')}% "
                    f"Stability={audit.get('system_stability', 'N/A').upper()}"
                )
    deepseek.shutdown()
    mt5_conn.shutdown()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    if Config.SIMULATION_MODE:
        logger.info("SIMULATION_MODE = True → running backtest")
        run_simulation_mode()
    elif Config.CONTINUOUS_MODE:
        logger.info("CONTINUOUS_MODE = True → running 24/7 gold scalping loop")
        run_continuous_loop()
    else:
        logger.info("Single-pass gold scalping analysis mode")
        run_live_analysis()