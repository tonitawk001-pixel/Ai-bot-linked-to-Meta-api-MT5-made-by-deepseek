"""
Simulation Engine — Backtest + Safety Audit Framework.

Replays market conditions candle-by-candle, runs the full trading pipeline
without executing real trades. Tracks virtual portfolio, generates safety
audit reports, and validates system stability before live deployment.

In simulation mode:
    - NO REAL TRADING EVER
    - All execution routes to simulated orders
    - Risk engine remains fully active
"""

import random
import math
from datetime import datetime
from typing import Optional
import pandas as pd
import numpy as np

from trading_bot.config import Config
from trading_bot.utils.logger import logger
from trading_bot.mt5.data_feed import get_candles
from trading_bot.indicators.technical_indicators import compute_all_indicators


# ---------------------------------------------------------------------------
# Simulated portfolio tracker
# ---------------------------------------------------------------------------

class SimulatedPortfolio:
    """
    Tracks virtual balance, open positions, and trade history during simulation.
    """

    def __init__(self, initial_balance: float = 10000.0):
        self.initial_balance = initial_balance
        self.balance = initial_balance
        self.equity = initial_balance
        self.equity_peak = initial_balance
        self.equity_low = initial_balance
        self.open_positions = []   # list of simulated position dicts
        self.trade_history = []    # list of closed trade dicts
        self.daily_pnl = 0.0
        self.consecutive_losses = 0

    def open_position(self, action: str, symbol: str, lot: float,
                      entry_price: float, sl: float, tp: float,
                      timestamp: str, spread_cost: float = 0.0001,
                      slippage: float = 0.00005, fail_rate: float = 0.05) -> Optional[dict]:
        """Simulate opening a position. Returns position dict or None if failed."""
        if random.random() < fail_rate:
            logger.debug(f"Simulated order failure for {symbol} {action}")
            return None

        fill_price = entry_price + (slippage if action == "BUY" else -slippage)
        fill_price = round(fill_price, 5)

        position = {
            "ticket": len(self.trade_history) + len(self.open_positions) + 1,
            "symbol": symbol,
            "action": action,
            "lot": lot,
            "entry_price": fill_price,
            "sl": sl,
            "tp": tp,
            "volume": lot * 100000,
            "open_time": timestamp,
            "spread_cost": spread_cost,
        }
        self.open_positions.append(position)
        logger.info(f"[SIM] Opened {action} {lot} {symbol} @ {fill_price} "
                    f"SL={sl} TP={tp}")
        return position

    def update_positions(self, current_price: float, timestamp: str,
                         spread: float = 0.0001):
        """Check SL/TP hits and update equity for all open positions."""
        closed_positions = []
        remaining = []

        for pos in self.open_positions:
            action = pos["action"]
            entry = pos["entry_price"]
            sl = pos["sl"]
            tp = pos["tp"]
            lot = pos["lot"]

            # Check SL / TP
            hit_sl = False
            hit_tp = False
            if action == "BUY":
                if sl and current_price <= sl:
                    hit_sl = True
                    exit_price = sl
                elif tp and current_price >= tp:
                    hit_tp = True
                    exit_price = tp
            else:  # SELL
                if sl and current_price >= sl:
                    hit_sl = True
                    exit_price = sl
                elif tp and current_price <= tp:
                    hit_tp = True
                    exit_price = tp

            if hit_sl or hit_tp:
                pnl = (exit_price - entry) * lot * 100000 if action == "BUY" else (entry - exit_price) * lot * 100000
                pnl -= spread * lot * 100000
                self.balance += pnl
                self.daily_pnl += pnl
                if pnl < 0:
                    self.consecutive_losses += 1
                else:
                    self.consecutive_losses = 0

                closed = {
                    "ticket": pos["ticket"],
                    "symbol": pos["symbol"],
                    "action": action,
                    "lot": lot,
                    "entry": entry,
                    "exit": exit_price,
                    "pnl": round(pnl, 2),
                    "reason": "SL" if hit_sl else "TP",
                    "open_time": pos["open_time"],
                    "close_time": timestamp,
                }
                closed_positions.append(closed)
                self.trade_history.append(closed)
                logger.info(f"[SIM] Closed {action} {lot} {pos['symbol']} @ {exit_price} "
                            f"P/L={pnl:.2f} ({closed['reason']})")
            else:
                # Unrealized P/L
                if action == "BUY":
                    unrealized = (current_price - entry) * lot * 100000
                else:
                    unrealized = (entry - current_price) * lot * 100000
                pos["unrealized_pnl"] = unrealized
                remaining.append(pos)

        self.open_positions = remaining

        # Update equity
        unrealized_total = sum(p.get("unrealized_pnl", 0) for p in self.open_positions)
        self.equity = self.balance + unrealized_total
        if self.equity > self.equity_peak:
            self.equity_peak = self.equity
        self.equity_low = min(self.equity_low, self.equity)

        return closed_positions

    def get_drawdown(self) -> float:
        """Current drawdown percentage from peak equity."""
        if self.equity_peak == 0:
            return 0.0
        return max(0, (self.equity_peak - self.equity) / self.equity_peak * 100)

    def get_max_drawdown(self) -> float:
        """Maximum historical drawdown percentage."""
        if self.initial_balance == 0:
            return 0.0
        return max(0, (self.equity_peak - self.equity_low) / self.equity_peak * 100)

    def reset(self):
        """Reset portfolio to initial state."""
        self.balance = self.initial_balance
        self.equity = self.initial_balance
        self.equity_peak = self.initial_balance
        self.equity_low = self.initial_balance
        self.open_positions = []
        self.trade_history = []
        self.daily_pnl = 0.0
        self.consecutive_losses = 0


# ---------------------------------------------------------------------------
# Safety Audit Report
# ---------------------------------------------------------------------------

class SafetyAudit:
    """
    Evaluates system behavior during simulation and generates a structured
    safety report with warnings and stability classification.
    """

    def __init__(self):
        self.all_decisions = []      # all pipeline decisions from simulation
        self.all_risk_evaluations = []
        self.trade_results = []      # closed trade dicts

    def record_decision(self, decision: dict, risk_eval: dict):
        """Record a pipeline decision for audit analysis."""
        self.all_decisions.append(decision)
        self.all_risk_evaluations.append(risk_eval)

    def record_trade(self, trade: dict):
        """Record a closed trade result."""
        self.trade_results.append(trade)

    def generate_report(self) -> dict:
        """
        Generate a structured safety audit report.

        Returns:
            dict: {
                "total_trades": int,
                "win_rate": float,
                "max_drawdown": float,
                "profit_factor": float,
                "ai_dependency_score": 0-100,
                "risk_block_rate": 0-100,
                "rule_engine_false_positives": int,
                "consecutive_loss_streaks": int,
                "exposure_per_symbol": dict,
                "warnings": [str],
                "system_stability": "stable" | "unstable" | "dangerous"
            }
        """
        trades = self.trade_results
        decisions = self.all_decisions
        risk_evals = self.all_risk_evaluations

        total_trades = len(trades)
        wins = sum(1 for t in trades if t.get("pnl", 0) > 0)
        losses = sum(1 for t in trades if t.get("pnl", 0) < 0)
        win_rate = (wins / total_trades * 100) if total_trades > 0 else 0.0

        gross_profit = sum(t["pnl"] for t in trades if t.get("pnl", 0) > 0)
        gross_loss = abs(sum(t["pnl"] for t in trades if t.get("pnl", 0) < 0))
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else 0.0

        # Max drawdown from trade history
        peak = Config.SIMULATION_INITIAL_BALANCE
        low = Config.SIMULATION_INITIAL_BALANCE
        balance = Config.SIMULATION_INITIAL_BALANCE
        for t in trades:
            balance += t.get("pnl", 0)
            if balance > peak:
                peak = balance
            if balance < low:
                low = balance
        max_drawdown = (peak - low) / peak * 100 if peak > 0 else 0.0

        # AI dependency score — how often AI agreed with or overrode rule engine
        ai_decisions = [d for d in decisions if d.get("ai_analysis", {}).get("ai_unavailable") == False]
        ai_dependency = len(ai_decisions) / max(len(decisions), 1) * 100

        # Risk block rate
        total_checks = len(risk_evals)
        blocked = sum(1 for r in risk_evals if not r.get("approved", True))
        risk_block_rate = (blocked / max(total_checks, 1)) * 100

        # Consecutive loss streaks
        max_consecutive_losses = 0
        current_streak = 0
        for t in trades:
            if t.get("pnl", 0) < 0:
                current_streak += 1
                max_consecutive_losses = max(max_consecutive_losses, current_streak)
            else:
                current_streak = 0

        # Exposure per symbol
        exposure = {}
        for t in trades:
            sym = t.get("symbol", "UNKNOWN")
            if sym not in exposure:
                exposure[sym] = {"trades": 0, "pnl": 0.0}
            exposure[sym]["trades"] += 1
            exposure[sym]["pnl"] += t.get("pnl", 0)

        # Rule engine false positives (setup_valid=True but trade lost money)
        false_positives = 0
        for t in trades:
            if t.get("pnl", 0) < 0:
                false_positives += 1

        # Warnings
        warnings = []
        if max_drawdown > Config.MAX_DRAWDOWN_PERCENT:
            warnings.append(f"DRAWDOWN: {max_drawdown:.1f}% exceeds limit {Config.MAX_DRAWDOWN_PERCENT}%")
        if win_rate < Config.MIN_WIN_RATE:
            warnings.append(f"WIN RATE: {win_rate:.1f}% below minimum {Config.MIN_WIN_RATE}%")
        if risk_block_rate > Config.MAX_RISK_BLOCK_RATE:
            warnings.append(f"RISK BLOCK RATE: {risk_block_rate:.1f}% exceeds limit {Config.MAX_RISK_BLOCK_RATE}%")
        if max_consecutive_losses >= Config.MAX_CONSECUTIVE_LOSSES:
            warnings.append(f"CONSECUTIVE LOSSES: {max_consecutive_losses} reaches limit {Config.MAX_CONSECUTIVE_LOSSES}")
        if ai_dependency > 80:
            warnings.append(f"AI DEPENDENCY: {ai_dependency:.0f}% — system overly reliant on AI")

        # System stability classification
        if max_drawdown > 20 or win_rate < 30:
            system_stability = "dangerous"
        elif max_drawdown > 10 or win_rate < 40 or risk_block_rate > 70:
            system_stability = "unstable"
        else:
            system_stability = "stable"

        report = {
            "total_trades": total_trades,
            "win_rate": round(win_rate, 2),
            "max_drawdown": round(max_drawdown, 2),
            "profit_factor": round(profit_factor, 2),
            "ai_dependency_score": round(ai_dependency, 1),
            "risk_block_rate": round(risk_block_rate, 1),
            "rule_engine_false_positives": false_positives,
            "consecutive_loss_streaks": max_consecutive_losses,
            "exposure_per_symbol": exposure,
            "warnings": warnings,
            "system_stability": system_stability,
        }

        logger.info("=" * 60)
        logger.info("SAFETY AUDIT REPORT")
        logger.info("=" * 60)
        for key, val in report.items():
            logger.info(f"  {key}: {val}")
        if warnings:
            logger.warning("WARNINGS:")
            for w in warnings:
                logger.warning(f"  - {w}")
        logger.info(f"  System classification: {system_stability.upper()}")
        logger.info("=" * 60)

        return report


# ---------------------------------------------------------------------------
# Simulation Runner
# ---------------------------------------------------------------------------

def run_simulation(symbol: str, timeframe: str, candle_count: int = 500,
                   rule_engine=None, deepseek=None, risk_manager=None) -> dict:
    """
    Run a full pipeline simulation on historical data.

    Process:
        1. Fetch historical data
        2. Walk forward candle-by-candle (or step by step)
        3. At each step: compute indicators → rule engine → AI → risk → simulated execution
        4. Track virtual portfolio
        5. Generate safety audit report

    Args:
        symbol: MT5 symbol.
        timeframe: Timeframe string.
        candle_count: Number of candles to simulate.
        rule_engine: Initialized RuleEngine.
        deepseek: Initialized DeepSeekClient.
        risk_manager: Initialized RiskManager.

    Returns:
        dict with simulation results + audit report.
    """
    from trading_bot.indicators.technical_indicators import compute_all_indicators
    from trading_bot.strategy.rule_engine import RuleEngine
    from trading_bot.risk.risk_manager import RiskManager
    from trading_bot.ai.deepseek_client import DeepSeekClient
    from trading_bot.main import (
        build_gold_ai_payload as build_ai_payload,
        compute_gold_lot_size,
        compute_gold_sl_tp,
    )

    logger.info(f"\n{'='*60}")
    logger.info(f"SIMULATION: {symbol} ({timeframe})")
    logger.info(f"Periods to simulate: {candle_count}")
    logger.info(f"Initial balance: ${Config.SIMULATION_INITIAL_BALANCE:.2f}")
    logger.info(f"{'='*60}")

    # Fetch data
    ohlcv = get_candles(symbol=symbol, timeframe=timeframe, count=candle_count + 200)
    if ohlcv is None or len(ohlcv) < 100:
        logger.error(f"Insufficient data for simulation: {symbol}")
        return {"error": "Insufficient data"}

    # Portfolio + audit
    portfolio = SimulatedPortfolio(initial_balance=Config.SIMULATION_INITIAL_BALANCE)
    audit = SafetyAudit()

    # Warm-up: first 100 candles for indicators
    warmup = 100
    total_candles = len(ohlcv)

    for i in range(warmup, total_candles):
        window = ohlcv.iloc[:i+1]
        current_candle = ohlcv.iloc[i]
        current_price = float(current_candle["close"])
        current_time = str(current_candle.name) if hasattr(current_candle, 'name') else str(i)
        spread = float(current_candle.get("spread", 1)) * 1e-5 if "spread" in ohlcv.columns else Config.SIMULATION_SPREAD_COST

        # Compute indicators
        indicators = compute_all_indicators(window)

        # Run rule engine
        decision = rule_engine.analyze(ohlcv=window, indicators=indicators)

        # Run AI (build payload from decision)
        try:
            atr_series = indicators.get("atr", pd.Series(dtype=float))
            atr_val = float(atr_series.iloc[-1]) if not atr_series.empty else None
        except Exception:
            atr_val = None

        # Build gold AI payload using available context
        strategy_result = {
            "bias": decision.get("trend", "neutral"),
            "setup_score": decision.get("setup_strength", 0),
            "direction": decision.get("trend", "NONE"),
            "confidence": 0.5,
            "session": "unknown",
            "reason": "",
            "pullback_detected": False,
            "entry_trigger": decision.get("setup_valid", False),
        }
        m5_indicators = {"rsi": indicators.get("rsi", pd.Series()), "atr": indicators.get("atr", pd.Series())}
        ai_payload = build_ai_payload(
            symbol="XAUUSD",
            strategy_result=strategy_result,
            m5_ohlcv=window,
            m5_indicators=m5_indicators,
        )
        ai_analysis = deepseek.analyze_market(ai_payload)

        # Risk check
        risk_eval = risk_manager.validate(rule_decision=decision, ai_analysis=ai_analysis, ohlcv=window)
        audit.record_decision(decision, risk_eval)

        # Determine action from rule_engine decision
        trend = decision.get("trend", "neutral")
        if trend == "bullish" and decision.get("setup_valid", False):
            action = "BUY"
        elif trend == "bearish" and decision.get("setup_valid", False):
            action = "SELL"
        else:
            action = "NONE"

        # Compute lot, SL, TP using gold functions
        if action != "NONE" and atr_val is not None:
            lot = compute_gold_lot_size(
                account_balance=portfolio.balance,
                atr_value=atr_val,
                m5_ohlcv=window,
            )
            sl, tp = compute_gold_sl_tp(action, window, atr_val)
        else:
            lot = Config.DEFAULT_LOT_SIZE
            sl, tp = 0.0, 0.0

        # Simulated execution
        if risk_eval["approved"] and decision.get("setup_valid") and action != "NONE":
            # Don't enter if already in a position for this symbol
            in_position = any(p["symbol"] == symbol for p in portfolio.open_positions)
            if not in_position:
                pos = portfolio.open_position(
                    action=action,
                    symbol=symbol,
                    lot=lot,
                    entry_price=current_price,
                    sl=sl,
                    tp=tp,
                    timestamp=current_time,
                    spread_cost=Config.SIMULATION_SPREAD_COST,
                    slippage=Config.SIMULATION_SLIPPAGE,
                    fail_rate=Config.SIMULATION_FAIL_RATE,
                )
                if pos is None:
                    logger.debug(f"[SIM] Order placement failed (simulated) for {symbol}")
                else:
                    # Update risk manager with simulated P/L tracking
                    pass

        # Update positions with current price
        closed = portfolio.update_positions(current_price, current_time, spread=Config.SIMULATION_SPREAD_COST)
        for c in closed:
            audit.record_trade(c)
            risk_manager.record_result(c["pnl"])

        # Progress logging every 100 candles
        if (i - warmup) % 100 == 0:
            logger.info(f"[{i-warmup}/{total_candles-warmup}] "
                        f"Balance={portfolio.balance:.2f} Equity={portfolio.equity:.2f} "
                        f"Drawdown={portfolio.get_drawdown():.2f}% "
                        f"Open={len(portfolio.open_positions)}")

    # Close any remaining positions at final price
    final_price = float(ohlcv["close"].iloc[-1])
    final_time = str(ohlcv.index[-1])
    closed_final = portfolio.update_positions(final_price, final_time, spread=Config.SIMULATION_SPREAD_COST)
    for c in closed_final:
        audit.record_trade(c)

    # Generate audit report
    audit_report = audit.generate_report()

    # Portfolio summary
    total_pnl = portfolio.balance - Config.SIMULATION_INITIAL_BALANCE
    logger.info(f"\nSIMULATION COMPLETE: {symbol}")
    logger.info(f"  Final balance: ${portfolio.balance:.2f}")
    logger.info(f"  Total P/L: ${total_pnl:.2f}")
    logger.info(f"  Total trades: {len(portfolio.trade_history)}")
    logger.info(f"  Max drawdown: {portfolio.get_max_drawdown():.2f}%")
    logger.info(f"  System stability: {audit_report['system_stability'].upper()}")

    return {
        "symbol": symbol,
        "timeframe": timeframe,
        "candles_simulated": total_candles - warmup,
        "initial_balance": Config.SIMULATION_INITIAL_BALANCE,
        "final_balance": round(portfolio.balance, 2),
        "total_pnl": round(total_pnl, 2),
        "total_trades": len(portfolio.trade_history),
        "trades": portfolio.trade_history,
        "equity_curve": [],  # Would store full history for plotting
        "safety_audit": audit_report,
    }