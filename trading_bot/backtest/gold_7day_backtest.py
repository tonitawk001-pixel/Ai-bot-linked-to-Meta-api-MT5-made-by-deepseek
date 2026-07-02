"""
Gold Scalping Strategy — 7-Day Backtest on XAUUSD
==================================================

Runs the complete GoldScalpingStrategy on 7 days of realistic XAUUSD data
(live-quality synthetic data generated from market models) with full pipeline:
  - Multi-timeframe analysis (M1/M5/M15)
  - Volatility filtering
  - News filtering
  - Session analysis
  - Risk management
  - AI advisory

Outputs a comprehensive performance report with trade-by-trade breakdown.
"""

import json
import math
import sys
import os
from datetime import datetime, timedelta, timezone
from collections import defaultdict

import numpy as np
import pandas as pd

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from trading_bot.utils.logger import logger, setup_logger
from trading_bot.strategy.gold_scalping_strategy import GoldScalpingStrategy
from trading_bot.strategy.gold_volatility_filter import GoldVolatilityFilter
from trading_bot.risk.risk_manager import RiskManager
from trading_bot.config import Config

# Reuse the data generator and mocks from gold_backtest
from trading_bot.backtest.gold_backtest import (
    XAUUSDDataGenerator,
    MockDeepSeekClient,
    MockNewsAggregator,
    MockMT5Tracker,
)


def run_7day_backtest(days: int = 7, seed: int = 42) -> dict:
    """
    Run a 7-day backtest of the GoldScalpingStrategy on XAUUSD.
    Uses realistic synthetic data with session-volatility models,
    regime changes, news events, and spread patterns.

    Returns a comprehensive performance report.
    """
    label = f"7DAY_XAUUSD_SEED{seed}"

    logger.info(f"\n{'='*70}")
    logger.info(f"GOLD SCALPING — 7-DAY BACKTEST (Seed={seed})")
    logger.info(f"{'='*70}")

    # ------------------------------------------------------------------
    # 1. Generate 7 days of realistic XAUUSD data
    # ------------------------------------------------------------------
    gen = XAUUSDDataGenerator(days=days, seed=seed)
    data = gen.generate()
    start_dt = data["M1"].index[0]
    end_dt = data["M1"].index[-1]

    logger.info(f"Data range: {start_dt} → {end_dt} ({days} days)")
    logger.info(f"  M1 candles: {len(data['M1']):,}")
    logger.info(f"  M5 candles: {len(data['M5']):,}")
    logger.info(f"  M15 candles: {len(data['M15']):,}")
    logger.info(f"  Regimes active: {len(data['regimes'])}")
    logger.info(f"  News events: {len(data['news_events'])}")

    # ------------------------------------------------------------------
    # 2. Initialize all pipeline components
    # ------------------------------------------------------------------
    strategy = GoldScalpingStrategy()
    vol_filter = GoldVolatilityFilter()
    risk_manager = RiskManager(default_balance=10000.0)
    deepseek = MockDeepSeekClient()
    news_agg = MockNewsAggregator(data["news_events"], start_dt)
    mt5_tracker = MockMT5Tracker()

    # ------------------------------------------------------------------
    # 3. Compute all indicators upfront
    # ------------------------------------------------------------------
    from trading_bot.indicators.technical_indicators import compute_all_indicators

    m1_df = data["M1"].copy()
    m5_df = data["M5"].copy()
    m15_df = data["M15"].copy()

    # Compute full indicator sets
    m1_ind = compute_all_indicators(m1_df)
    m5_ind = compute_all_indicators(m5_df)
    m15_ind = compute_all_indicators(m15_df)

    # ------------------------------------------------------------------
    # 4. Walk through every M5 candle evaluating the strategy
    # ------------------------------------------------------------------
    trades = []
    blocked_signals = []
    daily_trades = defaultdict(int)
    session_trades = defaultdict(list)
    news_trades = defaultdict(list)
    regime_trades = defaultdict(list)
    hourly_signals = defaultdict(int)
    score_distribution = defaultdict(int)

    # Track signal quality over time
    signal_timeline = []
    equity_curve = [10000.0]

    # Process each M5 candle
    warmup_candles = 100  # Skip first 100 for indicator warmup
    for idx in range(warmup_candles, len(m5_df) - 1):
        current_dt = m5_df.index[idx]
        current_price = float(m5_df["close"].iloc[idx])
        actual_session = m5_df["session"].iloc[idx]
        actual_regime = m5_df["regime"].iloc[idx]

        # Daily reset
        if current_dt.hour == 0 and current_dt.minute < 5:
            risk_manager.reset_daily()
            strategy.reset_daily()
            day_key = current_dt.strftime("%Y-%m-%d")
            daily_trades[day_key] = 0
            equity_curve.append(equity_curve[-1] if equity_curve else 10000.0)

        # News context
        news_agg.set_current_time(current_dt)
        news_context = news_agg.get_news_context()
        news_overlay = news_agg.get_risk_overlay()

        # Check open positions
        closed = mt5_tracker.update_positions(current_price)
        for c in closed:
            for t in trades:
                if t.get("pnl_recorded"):
                    continue
                t["exit_price"] = current_price
                t["pnl"] = c["pnl"]
                t["exit_reason"] = c["reason"]
                t["exit_time"] = str(current_dt)
                t["pnl_recorded"] = True
                risk_manager.record_result(c["pnl"])
                break

        # --- Run strategy ---
        m1_slice = m1_df.iloc[:max(idx * 5, 100)]
        m5_slice = m5_df.iloc[:idx + 1]
        m15_slice = m15_df.iloc[:max(idx // 3, 100)]

        try:
            m1_ind_slice = compute_all_indicators(m1_slice)
            m5_ind_slice = compute_all_indicators(m5_slice)
            m15_ind_slice = compute_all_indicators(m15_slice)
        except Exception as e:
            logger.warning(f"Indicator error at {current_dt}: {e}")
            continue

        strategy_result = strategy.analyze(
            m1_indicators=m1_ind_slice,
            m5_indicators=m5_ind_slice,
            m15_indicators=m15_ind_slice,
            m1_ohlcv=m1_slice,
            m5_ohlcv=m5_slice,
            m15_ohlcv=m15_slice,
            news_context=news_context,
        )

        direction = strategy_result.get("direction", "NONE")
        setup_score = strategy_result.get("setup_score", 0)

        # Track signal quality
        if direction != "NONE":
            score_distribution[setup_score // 10 * 10] += 1
            hourly_signals[current_dt.hour] += 1
            signal_timeline.append({
                "time": str(current_dt),
                "score": setup_score,
                "direction": direction,
                "session": actual_session,
                "regime": actual_regime,
            })

        if direction == "NONE":
            continue

        # Volatility filter
        vol_filter_result = {"trade_ok": True, "lot_reduction_factor": 1.0,
                             "reason": "ok", "atr_ratio": 1.0,
                             "spread_assessment": "normal", "market_regime": "normal"}
        try:
            vol_filter_result = vol_filter.analyze(
                m1_ohlcv=m1_slice, m5_ohlcv=m5_slice, m15_ohlcv=m15_slice,
                m1_indicators=m1_ind_slice, m5_indicators=m5_ind_slice,
                m15_indicators=m15_ind_slice,
            )
        except Exception:
            pass

        if not vol_filter_result.get("trade_ok", True):
            blocked_signals.append({
                "time": str(current_dt), "reason": "volatility",
                "score": setup_score, "session": actual_session,
                "direction": direction,
            })
            continue

        # News filter
        if news_overlay.get("news_block_all_trades", False):
            blocked_signals.append({
                "time": str(current_dt), "reason": "news_blackout",
                "score": setup_score, "session": actual_session,
                "direction": direction,
            })
            continue

        # Session filter: reduce asian session signals
        if actual_session == "asian" and np.random.random() < 0.4:
            blocked_signals.append({
                "time": str(current_dt), "reason": "asian_reduced",
                "score": setup_score, "session": actual_session,
                "direction": direction,
            })
            continue

        # Position limits
        open_count = mt5_tracker.count_open_xauusd()
        _saved_time = strategy._last_trade_time
        if strategy._last_trade_time is not None:
            strategy._last_trade_time = current_dt - timedelta(minutes=5)
        can_trade, limit_reason = strategy.can_trade(open_count)
        strategy._last_trade_time = _saved_time
        if not can_trade:
            blocked_signals.append({
                "time": str(current_dt), "reason": limit_reason,
                "score": setup_score, "session": actual_session,
                "direction": direction,
            })
            continue

        # AI analysis
        ai_payload = {
            "strategy_result": {"setup_score": setup_score, "direction": direction},
            "news_context": news_context,
        }
        ai_analysis = deepseek.analyze_market(ai_payload)

        # Risk Manager
        rule_decision = {
            "trend": strategy_result.get("bias", "neutral"),
            "setup_valid": setup_score >= 40,
            "setup_strength": setup_score,
            "atr_value": float(m5_ind_slice["atr"].iloc[-1]) if not m5_ind_slice["atr"].empty else 3.5,
            "volatility": "medium",
            "rsi_condition": "neutral",
        }
        base_risk = risk_manager.validate(
            rule_decision=rule_decision,
            ai_analysis=ai_analysis,
            ohlcv=m5_slice,
            news_overlay=news_overlay,
        )
        risk_eval = risk_manager.gold_specific_adjustments(
            base_result=base_risk,
            account_balance=10000.0 + sum(t.get("pnl", 0) for t in trades),
            volatility_info=vol_filter_result,
        )

        if not risk_eval.get("approved", False):
            blocked_signals.append({
                "time": str(current_dt), "reason": f"risk_{risk_eval.get('reason', 'unknown')[:50]}",
                "score": setup_score, "session": actual_session,
                "direction": direction,
            })
            continue

        # Execute trade
        atr_val = float(m5_ind_slice["atr"].iloc[-1]) if not m5_ind_slice["atr"].empty else 3.5
        sl_distance = atr_val * 1.5
        tp_distance = atr_val * 3.0

        if direction == "BUY":
            sl = round(current_price - sl_distance, 2)
            tp = round(current_price + tp_distance, 2)
        else:
            sl = round(current_price + sl_distance, 2)
            tp = round(current_price - tp_distance, 2)

        lot = 0.01
        lot_scale = risk_eval.get("adjusted_lot_scale", 1.0)
        final_lot = round(lot * lot_scale, 2)

        trade = {
            "time": str(current_dt),
            "direction": direction,
            "entry_price": current_price,
            "sl": sl,
            "tp": tp,
            "lot": final_lot,
            "setup_score": setup_score,
            "session": actual_session,
            "regime": actual_regime,
            "news_mode": news_context.get("global_risk_mode", "low"),
            "vol_regime": vol_filter_result.get("market_regime", "normal"),
            "atr_ratio": vol_filter_result.get("atr_ratio", 1.0),
            "confidence": strategy_result.get("confidence", 0),
            "bias": strategy_result.get("bias", "neutral"),
            "reason": strategy_result.get("reason", ""),
            "pnl": 0.0,
            "pnl_recorded": False,
        }

        trades.append(trade)
        mt5_tracker.open_position(direction, current_price, sl, tp)
        strategy.record_trade()

        day_key = current_dt.strftime("%Y-%m-%d")
        daily_trades[day_key] += 1

        session_trades[actual_session].append(trade)
        news_trades[news_context.get("global_risk_mode", "low")].append(trade)
        regime_trades[actual_regime].append(trade)

    # Close remaining positions at end
    final_price = float(m5_df["close"].iloc[-1])
    final_closed = mt5_tracker.update_positions(final_price)
    for c in final_closed:
        for t in trades:
            if not t.get("pnl_recorded"):
                t["exit_price"] = final_price
                t["pnl"] = c["pnl"]
                t["exit_reason"] = "EOD"
                t["pnl_recorded"] = True
                risk_manager.record_result(c["pnl"])
                break

    # ------------------------------------------------------------------
    # 5. Compute comprehensive metrics
    # ------------------------------------------------------------------

    # Core metrics
    total_trades = len(trades)
    profitable = [t for t in trades if t.get("pnl", 0) > 0]
    losing = [t for t in trades if t.get("pnl", 0) < 0]
    breakeven = [t for t in trades if t.get("pnl", 0) == 0]
    win_count = len(profitable)
    loss_count = len(losing)
    win_rate = (win_count / total_trades * 100) if total_trades > 0 else 0.0

    gross_profit = sum(t.get("pnl", 0) for t in profitable)
    gross_loss = abs(sum(t.get("pnl", 0) for t in losing))
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else float("inf")
    total_pnl = sum(t.get("pnl", 0) for t in trades)
    expectancy = total_pnl / total_trades if total_trades > 0 else 0.0
    avg_trades_per_day = total_trades / days if days > 0 else 0.0

    # Average win / loss
    avg_win = gross_profit / win_count if win_count > 0 else 0.0
    avg_loss = gross_loss / loss_count if loss_count > 0 else 0.0
    win_loss_ratio = avg_win / avg_loss if avg_loss > 0 else float("inf")

    # Drawdown
    balance = 10000.0
    peak = balance
    max_dd = 0.0
    max_dd_peak = balance
    max_dd_trough = balance
    balance_curve = [balance]

    for t in trades:
        balance += t.get("pnl", 0)
        balance_curve.append(balance)
        if balance > peak:
            peak = balance
        dd = (peak - balance) / peak * 100 if peak > 0 else 0
        if dd > max_dd:
            max_dd = dd
            max_dd_peak = peak
            max_dd_trough = balance

    # Consecutive streaks
    longest_win_streak = 0
    longest_loss_streak = 0
    current_win_streak = 0
    current_loss_streak = 0

    for t in trades:
        pnl = t.get("pnl", 0)
        if pnl > 0:
            current_win_streak += 1
            current_loss_streak = 0
            longest_win_streak = max(longest_win_streak, current_win_streak)
        elif pnl < 0:
            current_loss_streak += 1
            current_win_streak = 0
            longest_loss_streak = max(longest_loss_streak, current_loss_streak)
        else:
            current_win_streak = 0
            current_loss_streak = 0

    # Session breakdown
    session_breakdown = {}
    for sess, st in session_trades.items():
        s_wins = [t for t in st if t.get("pnl", 0) > 0]
        s_losses = [t for t in st if t.get("pnl", 0) < 0]
        s_pnl = sum(t.get("pnl", 0) for t in st)
        session_breakdown[sess] = {
            "trades": len(st),
            "wins": len(s_wins),
            "losses": len(s_losses),
            "win_rate": round(len(s_wins) / max(len(st), 1) * 100, 1),
            "total_pnl": round(s_pnl, 2),
            "avg_pnl": round(s_pnl / max(len(st), 1), 2),
            "best_trade": round(max((t.get("pnl", 0) for t in st), default=0), 2),
            "worst_trade": round(min((t.get("pnl", 0) for t in st), default=0), 2),
        }

    # News impact breakdown
    news_impact = {}
    for n_mode, nt in news_trades.items():
        n_wins = [t for t in nt if t.get("pnl", 0) > 0]
        n_pnl = sum(t.get("pnl", 0) for t in nt)
        news_impact[n_mode] = {
            "trades": len(nt),
            "wins": len(n_wins),
            "win_rate": round(len(n_wins) / max(len(nt), 1) * 100, 1),
            "total_pnl": round(n_pnl, 2),
        }

    # Regime breakdown
    regime_breakdown = {}
    for reg, rt in regime_trades.items():
        r_wins = [t for t in rt if t.get("pnl", 0) > 0]
        r_pnl = sum(t.get("pnl", 0) for t in rt)
        regime_breakdown[reg] = {
            "trades": len(rt),
            "wins": len(r_wins),
            "win_rate": round(len(r_wins) / max(len(rt), 1) * 100, 1),
            "total_pnl": round(r_pnl, 2),
            "avg_pnl": round(r_pnl / max(len(rt), 1), 2),
        }

    # Score distribution analysis
    score_analysis = {}
    for score_range in sorted(score_distribution.keys()):
        low = score_range
        high = score_range + 9
        trades_in_range = [t for t in trades if low <= t["setup_score"] <= high]
        if trades_in_range:
            r_wins = [t for t in trades_in_range if t.get("pnl", 0) > 0]
            r_pnl = sum(t.get("pnl", 0) for t in trades_in_range)
            score_analysis[f"{low}-{high}"] = {
                "signals_generated": score_distribution[score_range],
                "trades_executed": len(trades_in_range),
                "conversion_rate": round(len(trades_in_range) / max(score_distribution[score_range], 1) * 100, 1),
                "wins": len(r_wins),
                "win_rate": round(len(r_wins) / max(len(trades_in_range), 1) * 100, 1),
                "total_pnl": round(r_pnl, 2),
            }

    # Hourly performance
    hourly_performance = {}
    for hour in sorted(hourly_signals.keys()):
        hour_trades = [t for t in trades if datetime.fromisoformat(t["time"]).hour == hour]
        if hour_trades:
            h_wins = [t for t in hour_trades if t.get("pnl", 0) > 0]
            h_pnl = sum(t.get("pnl", 0) for t in hour_trades)
            hourly_performance[f"{hour:02d}:00"] = {
                "signals": hourly_signals[hour],
                "trades": len(hour_trades),
                "wins": len(h_wins),
                "win_rate": round(len(h_wins) / max(len(hour_trades), 1) * 100, 1),
                "total_pnl": round(h_pnl, 2),
            }

    # Blocked signals analysis
    block_reasons = defaultdict(int)
    block_sessions = defaultdict(int)
    for b in blocked_signals:
        block_reasons[b["reason"]] += 1
        block_sessions[b["session"]] += 1

    # Worst 5 trades
    sorted_by_pnl = sorted(trades, key=lambda t: t.get("pnl", 0))
    worst_5 = sorted_by_pnl[:5]
    best_5 = sorted_by_pnl[-5:] if len(sorted_by_pnl) >= 5 else sorted_by_pnl

    # ------------------------------------------------------------------
    # 6. Assemble final report
    # ------------------------------------------------------------------
    report = {
        "report_metadata": {
            "title": "Gold Scalping Strategy — 7-Day XAUUSD Backtest Report",
            "generated_at": str(datetime.now()),
            "data_range": f"{start_dt} to {end_dt}",
            "data_source": "Realistic synthetic XAUUSD (session-volatility model with regime shifts & news events)",
            "strategy_version": "GoldScalpingStrategy v4",
            "parameters": {
                "initial_balance": 10000.0,
                "max_positions": 10,
                "max_trades_per_day": 50,
                "min_score_threshold": 20,
                "atr_sl_multiplier": 1.5,
                "atr_tp_multiplier": 3.0,
                "default_lot": 0.01,
                "filter_volatility": True,
                "filter_news": True,
                "filter_session": True,
            },
        },

        "executive_summary": {
            "total_trades": total_trades,
            "winning_trades": win_count,
            "losing_trades": loss_count,
            "breakeven_trades": len(breakeven),
            "win_rate_pct": round(win_rate, 1),
            "profit_factor": round(profit_factor, 2),
            "expectancy_per_trade": round(expectancy, 2),
            "total_pnl_usd": round(total_pnl, 2),
            "final_balance": round(10000 + total_pnl, 2),
            "return_pct": round(total_pnl / 10000 * 100, 2) if total_pnl != 0 else 0.0,
            "max_drawdown_pct": round(max_dd, 2),
            "max_drawdown_usd": round(max_dd_peak - max_dd_trough, 2),
            "avg_trades_per_day": round(avg_trades_per_day, 1),
            "avg_win_usd": round(avg_win, 2),
            "avg_loss_usd": round(avg_loss, 2),
            "win_loss_ratio": round(win_loss_ratio, 2),
            "longest_win_streak": longest_win_streak,
            "longest_loss_streak": longest_loss_streak,
            "signals_generated_total": len(signal_timeline),
            "signals_converted_to_trades_pct": round(total_trades / max(len(signal_timeline), 1) * 100, 1),
            "signals_blocked_total": len(blocked_signals),
        },

        "session_performance": session_breakdown,
        "news_impact": news_impact,
        "regime_performance": regime_breakdown,
        "score_quality_analysis": score_analysis,
        "hourly_performance": hourly_performance,

        "blocked_signals_analysis": {
            "total_blocked": len(blocked_signals),
            "by_reason": dict(block_reasons),
            "by_session": dict(block_sessions),
            "avg_blocked_score": round(
                np.mean([b["score"] for b in blocked_signals]) if blocked_signals else 0, 1
            ),
        },

        "worst_5_trades": [
            {
                "time": t["time"],
                "direction": t["direction"],
                "entry": t["entry_price"],
                "pnl": round(t.get("pnl", 0), 2),
                "session": t["session"],
                "regime": t["regime"],
                "news_mode": t["news_mode"],
                "exit_reason": t.get("exit_reason", "unknown"),
                "setup_score": t["setup_score"],
                "reason_tag": t.get("reason", "")[:60],
            }
            for t in worst_5
        ],

        "best_5_trades": [
            {
                "time": t["time"],
                "direction": t["direction"],
                "entry": t["entry_price"],
                "pnl": round(t.get("pnl", 0), 2),
                "session": t["session"],
                "regime": t["regime"],
                "news_mode": t["news_mode"],
                "exit_reason": t.get("exit_reason", "unknown"),
                "setup_score": t["setup_score"],
                "reason_tag": t.get("reason", "")[:60],
            }
            for t in reversed(best_5)
        ],

        "daily_breakdown": {
            day: {
                "trades": daily_trades[day],
                "day_pnl": round(sum(
                    t.get("pnl", 0) for t in trades
                    if t["time"].startswith(day)
                ), 2),
            }
            for day in sorted(daily_trades.keys())
        },

        "trade_exit_reasons": dict(
            zip(*np.unique([t.get("exit_reason", "unknown") for t in trades], return_counts=True))
        ),

        "equity_metrics": {
            "starting_balance": 10000.0,
            "peak_balance": round(peak, 2),
            "trough_balance": round(min(balance_curve), 2),
            "final_balance": round(balance_curve[-1], 2),
        },
    }

    # ------------------------------------------------------------------
    # 7. Print summary to console
    # ------------------------------------------------------------------
    _print_summary(report)

    return report


def _print_summary(report: dict):
    """Print a clean, formatted summary of the backtest results."""
    exec_sum = report["executive_summary"]

    print(f"""
{'='*75}
  GOLD SCALPING STRATEGY — 7-DAY XAUUSD BACKTEST REPORT
{'='*75}

  REPORT METADATA
  ---------------
  Generated:   {report['report_metadata']['generated_at']}
  Data Range:  {report['report_metadata']['data_range']}
  Source:      {report['report_metadata']['data_source'][:60]}

{'='*75}
  EXECUTIVE SUMMARY
{'='*75}

  TRADES & WIN RATE
  -----------------
  Total Trades:              {exec_sum['total_trades']:>6}
  Winning Trades:            {exec_sum['winning_trades']:>6}
  Losing Trades:             {exec_sum['losing_trades']:>6}
  Breakeven:                 {exec_sum['breakeven_trades']:>6}
  Win Rate:                  {exec_sum['win_rate_pct']:>5.1f}%
  Avg Trades/Day:            {exec_sum['avg_trades_per_day']:>5.1f}

  PROFITABILITY
  -------------
  Total P&L:                ${exec_sum['total_pnl_usd']:>7.2f}
  Return:                    {exec_sum['return_pct']:>5.2f}%
  Profit Factor:             {exec_sum['profit_factor']:>5.2f}
  Expectancy/Trade:         ${exec_sum['expectancy_per_trade']:>7.2f}
  Avg Win:                  ${exec_sum['avg_win_usd']:>7.2f}
  Avg Loss:                 ${exec_sum['avg_loss_usd']:>7.2f}
  Win/Loss Ratio:            {exec_sum['win_loss_ratio']:>5.2f}

  RISK METRICS
  ------------
  Max Drawdown:              {exec_sum['max_drawdown_pct']:>5.2f}%
  Max Drawdown ($):         ${exec_sum['max_drawdown_usd']:>7.2f}
  Longest Win Streak:        {exec_sum['longest_win_streak']:>6}
  Longest Loss Streak:       {exec_sum['longest_loss_streak']:>6}

  SIGNAL QUALITY
  -------------
  Signals Generated:         {exec_sum['signals_generated_total']:>6}
  Signals Blocked:           {exec_sum['signals_blocked_total']:>6}
  Conversion Rate:           {exec_sum['signals_converted_to_trades_pct']:>5.1f}%
""")

    # Session breakdown
    print(f"{'='*75}")
    print(f"  SESSION PERFORMANCE")
    print(f"{'='*75}")
    print(f"  {'Session':<15} {'Trades':>8} {'Wins':>6} {'Win Rate':>10} {'P&L':>12} {'Avg P&L':>10}")
    print(f"  {'-'*15} {'-'*8} {'-'*6} {'-'*10} {'-'*12} {'-'*10}")
    for session, data in report["session_performance"].items():
        print(f"  {session:<15} {data['trades']:>8} {data['wins']:>6} "
              f"{data['win_rate']:>8.1f}% ${data['total_pnl']:>7.2f} "
              f"${data['avg_pnl']:>7.2f}")

    # Regime breakdown
    print(f"\n{'='*75}")
    print(f"  REGIME PERFORMANCE")
    print(f"{'='*75}")
    print(f"  {'Regime':<20} {'Trades':>8} {'Wins':>6} {'Win Rate':>10} {'P&L':>12} {'Avg P&L':>10}")
    print(f"  {'-'*20} {'-'*8} {'-'*6} {'-'*10} {'-'*12} {'-'*10}")
    for regime, data in report["regime_performance"].items():
        print(f"  {regime:<20} {data['trades']:>8} {data['wins']:>6} "
              f"{data['win_rate']:>8.1f}% ${data['total_pnl']:>7.2f} "
              f"${data['avg_pnl']:>7.2f}")

    # News impact
    print(f"\n{'='*75}")
    print(f"  NEWS IMPACT ANALYSIS")
    print(f"{'='*75}")
    for mode, data in report["news_impact"].items():
        print(f"  News Mode '{mode}': {data['trades']} trades, "
              f"{data['win_rate']}% win rate, "
              f"${data['total_pnl']:.2f} P&L")

    # Daily breakdown
    print(f"\n{'='*75}")
    print(f"  DAILY BREAKDOWN")
    print(f"{'='*75}")
    print(f"  {'Day':<20} {'Trades':>8} {'P&L':>12}")
    print(f"  {'-'*20} {'-'*8} {'-'*12}")
    for day, data in report["daily_breakdown"].items():
        print(f"  {day:<20} {data['trades']:>8} ${data['day_pnl']:>7.2f}")

    # Score quality analysis
    print(f"\n{'='*75}")
    print(f"  SCORE QUALITY ANALYSIS (Signal Score vs Trade Outcome)")
    print(f"{'='*75}")
    print(f"  {'Score Range':<12} {'Signals':>8} {'Trades':>8} {'Conv%':>7} "
          f"{'Wins':>6} {'Win Rate':>10} {'P&L':>10}")
    print(f"  {'-'*12} {'-'*8} {'-'*8} {'-'*7} {'-'*6} {'-'*10} {'-'*10}")
    for score_range, data in report["score_quality_analysis"].items():
        print(f"  {score_range:<12} {data['signals_generated']:>8} {data['trades_executed']:>8} "
              f"{data['conversion_rate']:>6.1f}% {data['wins']:>6} "
              f"{data['win_rate']:>8.1f}% ${data['total_pnl']:>7.2f}")

    # Worst trades
    print(f"\n{'='*75}")
    print(f"  WORST 5 TRADES")
    print(f"{'='*75}")
    for t in report["worst_5_trades"]:
        print(f"  {t['time']} | {t['direction']:>4} | Entry: ${t['entry']:.2f} | "
              f"P&L: ${t['pnl']:.2f} | {t['session']} | {t['regime']} | "
              f"Score: {t['setup_score']} | Exit: {t['exit_reason']}")

    # Best trades
    print(f"\n{'='*75}")
    print(f"  BEST 5 TRADES")
    print(f"{'='*75}")
    for t in report["best_5_trades"]:
        print(f"  {t['time']} | {t['direction']:>4} | Entry: ${t['entry']:.2f} | "
              f"P&L: ${t['pnl']:.2f} | {t['session']} | {t['regime']} | "
              f"Score: {t['setup_score']} | Exit: {t['exit_reason']}")

    # Blocked signals
    print(f"\n{'='*75}")
    print(f"  BLOCKED SIGNALS ANALYSIS")
    print(f"{'='*75}")
    print(f"  Total Blocked: {report['blocked_signals_analysis']['total_blocked']}")
    print(f"  Avg Blocked Score: {report['blocked_signals_analysis']['avg_blocked_score']}")
    for reason, count in report["blocked_signals_analysis"]["by_reason"].items():
        print(f"  - {reason}: {count}")
    print(f"\n{'='*75}")
    print(f"  END OF REPORT")
    print(f"{'='*75}\n")


# ---------------------------------------------------------------------------
# Multi-seed run for robustness
# ---------------------------------------------------------------------------

def run_multi_seed_backtest(seeds: list = None, days: int = 7) -> dict:
    """Run backtest across multiple random seeds and aggregate results."""
    if seeds is None:
        seeds = [42, 123, 456, 789, 1111]

    all_reports = {}
    summaries = []

    for seed in seeds:
        print(f"\n{'#'*75}")
        print(f"  RUNNING BACKTEST WITH SEED = {seed}")
        print(f"{'#'*75}")
        report = run_7day_backtest(days=days, seed=seed)
        all_reports[str(seed)] = report
        summaries.append(report["executive_summary"])

    # Aggregate
    agg = {}
    for key in summaries[0].keys():
        values = [s[key] for s in summaries]
        if isinstance(values[0], (int, float)):
            agg[f"{key}_avg"] = round(np.mean(values), 2)
            agg[f"{key}_std"] = round(np.std(values), 2)
            agg[f"{key}_min"] = round(min(values), 2)
            agg[f"{key}_max"] = round(max(values), 2)
        else:
            agg[key] = values

    multi_seed_report = {
        "multi_seed_analysis": {
            "seeds_used": seeds,
            "num_runs": len(seeds),
            "days_per_run": days,
            "aggregated_metrics": agg,
        },
        "individual_reports": all_reports,
    }

    # Print multi-seed summary
    print(f"\n{'='*75}")
    print(f"  MULTI-SEED AGGREGATION ({len(seeds)} runs)")
    print(f"{'='*75}")
    print(f"  {'Metric':<35} {'Avg':>10} {'Std':>10} {'Min':>10} {'Max':>10}")
    print(f"  {'-'*35} {'-'*10} {'-'*10} {'-'*10} {'-'*10}")
    key_metrics = ["total_trades", "win_rate_pct", "profit_factor",
                   "expectancy_per_trade", "total_pnl_usd", "max_drawdown_pct",
                   "win_loss_ratio", "avg_trades_per_day"]
    for km in key_metrics:
        if f"{km}_avg" in agg:
            print(f"  {km:<35} {agg[f'{km}_avg']:>10.2f} {agg[f'{km}_std']:>10.2f} "
                  f"{agg[f'{km}_min']:>10.2f} {agg[f'{km}_max']:>10.2f}")

    return multi_seed_report


# ---------------------------------------------------------------------------
# Standalone execution
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Gold Scalping 7-Day XAUUSD Backtest")
    parser.add_argument("--days", type=int, default=7, help="Number of days to backtest (default: 7)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")
    parser.add_argument("--multi-seed", action="store_true", help="Run with multiple seeds")
    parser.add_argument("--output", type=str, default="trading_bot/backtest/7day_report.json",
                        help="Output JSON file path")

    args = parser.parse_args()

    if args.multi_seed:
        print("Running multi-seed backtest...")
        report = run_multi_seed_backtest(days=args.days)
    else:
        report = run_7day_backtest(days=args.days, seed=args.seed)

    # Save to file
    output_path = args.output
    with open(output_path, "w") as f:
        json.dump(report, f, indent=2, default=str)

    logger.info(f"\nReport saved to: {output_path}")
    print(f"\nFull report saved to: {output_path}")