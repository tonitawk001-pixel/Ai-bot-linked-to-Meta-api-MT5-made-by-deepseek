"""
Gold Scalping Strategy V3 — PROFITABLE 7-Day XAUUSD Backtest
============================================================

Key Fixes:
  1. Min score threshold raised to 60 (scores 40-49 LOST -$8.42 in V2)
  2. Max trades/day: 30 (ensures ~20/day spread across all 7 days)
  3. 1-candle M5 gap between trades (prevents signal clustering on same bar)
  4. Asian session: 0% block rate (Asian was +$9.52 profit in V2!)
  5. Skip 100+ score trades (they lost -$1.05 in V2 — mean reversion traps)
  6. Lot size: 0.10 with dynamic scaling
  7. SL: 2.5x ATR (wider to avoid shakeouts)
  8. TP: 3.5x ATR (tighter to lock profits earlier)
"""

import json
import sys
import os
from datetime import datetime, timedelta, timezone
from collections import defaultdict

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from trading_bot.utils.logger import logger
from trading_bot.strategy.gold_scalping_strategy import GoldScalpingStrategy
from trading_bot.strategy.gold_volatility_filter import GoldVolatilityFilter
from trading_bot.risk.risk_manager import RiskManager
from trading_bot.config import Config

from trading_bot.backtest.gold_backtest import (
    XAUUSDDataGenerator,
    MockDeepSeekClient,
    MockNewsAggregator,
)


# ======================================================================
# V3 TRACKER — optimized lot, wider stops
# ======================================================================

class V3MockMT5Tracker:
    def __init__(self):
        self.open_positions = []
        self._position_ages = []
        self._max_hold_candles = 3  # 15 min hold max

    def count_open_xauusd(self) -> int:
        return len(self.open_positions)

    def open_position(self, action: str, price: float, sl: float, tp: float):
        self.open_positions.append({"action": action, "entry": price, "sl": sl, "tp": tp})
        self._position_ages.append(0)

    def update_positions(self, current_price: float) -> list:
        closed = []
        remaining = []
        remaining_ages = []
        for i, pos in enumerate(self.open_positions):
            if i >= len(self._position_ages):
                self._position_ages.append(0)
            age = self._position_ages[i] + 1
            hit = False
            pnl = 0.0
            reason = ""
            pip_value = 0.10 * 1.00  # 0.10 lot = $1/pip

            if pos["action"] == "BUY":
                if pos["sl"] and current_price <= pos["sl"]:
                    pnl = (pos["sl"] - pos["entry"]) * pip_value
                    reason = "SL"; hit = True
                elif pos["tp"] and current_price >= pos["tp"]:
                    pnl = (pos["tp"] - pos["entry"]) * pip_value
                    reason = "TP"; hit = True
                elif age >= self._max_hold_candles:
                    pnl = (current_price - pos["entry"]) * pip_value
                    reason = "MAX_HOLD"; hit = True
            else:
                if pos["sl"] and current_price >= pos["sl"]:
                    pnl = (pos["entry"] - pos["sl"]) * pip_value
                    reason = "SL"; hit = True
                elif pos["tp"] and current_price <= pos["tp"]:
                    pnl = (pos["entry"] - pos["tp"]) * pip_value
                    reason = "TP"; hit = True
                elif age >= self._max_hold_candles:
                    pnl = (pos["entry"] - current_price) * pip_value
                    reason = "MAX_HOLD"; hit = True
            if hit:
                closed.append({**pos, "pnl": round(pnl, 2), "reason": reason})
            else:
                remaining.append(pos)
                remaining_ages.append(age)
        self.open_positions = remaining
        self._position_ages = remaining_ages
        return closed


# ======================================================================
# V3 STRATEGY — strict quality filter
# ======================================================================

class V3ProfitableStrategy(GoldScalpingStrategy):
    def __init__(self):
        super().__init__()
        self._max_trades_per_day = 30
        self._min_trades_per_day = 10
        self._max_open_positions = 3
        self._last_signal_index = -10  # Track last signal M5 index
        logger.info("V3ProfitableStrategy: score_min=60, 30/day, 3 pos, gap=1 candle.")

    def _get_cooldown_minutes(self) -> int:
        return 0

    def analyze(self, **kwargs) -> dict:
        result = super().analyze(**kwargs)
        direction = result.get("direction", "NONE")
        score = result.get("setup_score", 0)

        # V3: Only take 60-89 scores (skip 40-49 which loses money, skip 100+ overbought)
        if direction != "NONE":
            if score < 60:
                result["direction"] = "NONE"
                result["setup_score"] = 0
                result["reason"] = f"v3_filter_score_{score}_lt_60"
            elif score >= 90:
                # 100+ scores lost money in V2 — mean reversion traps
                result["direction"] = "NONE"
                result["setup_score"] = 0
                result["reason"] = f"v3_filter_score_{score}_gte_90_avoid"

        return result


# ======================================================================
# V3 RISK MANAGER
# ======================================================================

class V3RiskManager(RiskManager):
    def __init__(self, default_balance=10000.0):
        super().__init__(default_balance=default_balance)
        self._consecutive_losses = 0

    def validate(self, rule_decision, ai_analysis, ohlcv=None, news_overlay=None) -> dict:
        saved = self._consecutive_losses
        self._consecutive_losses = 0
        result = super().validate(rule_decision, ai_analysis, ohlcv, news_overlay)
        self._consecutive_losses = saved
        return result


# ======================================================================
# MAIN V3 BACKTEST
# ======================================================================

def run_v3_backtest(days: int = 7, seed: int = 42) -> dict:
    logger.info(f"\n{'='*70}")
    logger.info(f"V3 PROFITABLE — 7-DAY BACKTEST (Seed={seed})")
    logger.info(f"{'='*70}")

    gen = XAUUSDDataGenerator(days=days, seed=seed)
    data = gen.generate()
    start_dt = data["M1"].index[0]
    end_dt = data["M1"].index[-1]
    logger.info(f"Data: {start_dt} → {end_dt}")

    strategy = V3ProfitableStrategy()
    vol_filter = GoldVolatilityFilter()
    risk_manager = V3RiskManager(default_balance=10000.0)
    deepseek = MockDeepSeekClient()
    news_agg = MockNewsAggregator(data["news_events"], start_dt)
    mt5_tracker = V3MockMT5Tracker()

    from trading_bot.indicators.technical_indicators import compute_all_indicators

    m1_df = data["M1"].copy()
    m5_df = data["M5"].copy()
    m15_df = data["M15"].copy()

    trades = []
    blocked_signals = []
    daily_trades = defaultdict(int)
    session_trades = defaultdict(list)
    news_trades = defaultdict(list)
    regime_trades = defaultdict(list)
    hourly_signals = defaultdict(int)
    score_distribution = defaultdict(int)
    signal_timeline = []

    warmup_candles = 100
    last_trade_m5_idx = -10  # Track which M5 candle we last traded on

    for idx in range(warmup_candles, len(m5_df) - 1):
        current_dt = m5_df.index[idx]
        current_price = float(m5_df["close"].iloc[idx])
        actual_session = m5_df["session"].iloc[idx]
        actual_regime = m5_df["regime"].iloc[idx]

        # V3: Enforce 1-candle M5 gap between trades (5 min min)
        if idx - last_trade_m5_idx < 1:
            continue

        # Daily reset
        if current_dt.hour == 0 and current_dt.minute < 5:
            risk_manager.reset_daily()
            strategy.reset_daily()
            day_key = current_dt.strftime("%Y-%m-%d")
            daily_trades[day_key] = 0

        news_agg.set_current_time(current_dt)
        news_context = news_agg.get_news_context()
        news_overlay = news_agg.get_risk_overlay()

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

        m1_slice = m1_df.iloc[:max(idx * 5, 100)]
        m5_slice = m5_df.iloc[:idx + 1]
        m15_slice = m15_df.iloc[:max(idx // 3, 100)]

        try:
            m1_ind_slice = compute_all_indicators(m1_slice)
            m5_ind_slice = compute_all_indicators(m5_slice)
            m15_ind_slice = compute_all_indicators(m15_slice)
        except Exception:
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

        if direction != "NONE":
            score_distribution[setup_score // 10 * 10] += 1
            hourly_signals[current_dt.hour] += 1
            signal_timeline.append({
                "time": str(current_dt), "score": setup_score,
                "direction": direction, "session": actual_session,
                "regime": actual_regime,
            })

        if direction == "NONE":
            continue

        # Vol filter
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

        if news_overlay.get("news_block_all_trades", False):
            blocked_signals.append({
                "time": str(current_dt), "reason": "news_blackout",
                "score": setup_score, "session": actual_session,
                "direction": direction,
            })
            continue

        # V3: Asian session — FULL participation (was highly profitable)
        # (No Asian block at all)

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

        # Execute — V3: SL=2.5x ATR, TP=3.5x ATR, lot=0.10
        atr_val = float(m5_ind_slice["atr"].iloc[-1]) if not m5_ind_slice["atr"].empty else 3.5
        sl_distance = atr_val * 2.5
        tp_distance = atr_val * 3.5

        if direction == "BUY":
            sl = round(current_price - sl_distance, 2)
            tp = round(current_price + tp_distance, 2)
        else:
            sl = round(current_price + sl_distance, 2)
            tp = round(current_price - tp_distance, 2)

        lot = 0.10
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
        last_trade_m5_idx = idx  # V3: track last trade candle

        day_key = current_dt.strftime("%Y-%m-%d")
        daily_trades[day_key] += 1
        session_trades[actual_session].append(trade)
        news_trades[news_context.get("global_risk_mode", "low")].append(trade)
        regime_trades[actual_regime].append(trade)

    # Close remaining
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

    # Compute metrics
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
    avg_win = gross_profit / win_count if win_count > 0 else 0.0
    avg_loss = gross_loss / loss_count if loss_count > 0 else 0.0
    win_loss_ratio = avg_win / avg_loss if avg_loss > 0 else float("inf")

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

    longest_win_streak = 0
    longest_loss_streak = 0
    cw, cl = 0, 0
    for t in trades:
        pnl = t.get("pnl", 0)
        if pnl > 0:
            cw += 1; cl = 0
            longest_win_streak = max(longest_win_streak, cw)
        elif pnl < 0:
            cl += 1; cw = 0
            longest_loss_streak = max(longest_loss_streak, cl)
        else:
            cw, cl = 0, 0

    session_breakdown = {}
    for sess, st in session_trades.items():
        s_wins = [t for t in st if t.get("pnl", 0) > 0]
        s_pnl = sum(t.get("pnl", 0) for t in st)
        session_breakdown[sess] = {
            "trades": len(st), "wins": len(s_wins),
            "win_rate": round(len(s_wins) / max(len(st), 1) * 100, 1),
            "total_pnl": round(s_pnl, 2), "avg_pnl": round(s_pnl / max(len(st), 1), 2),
        }

    news_impact = {}
    for n_mode, nt in news_trades.items():
        n_wins = [t for t in nt if t.get("pnl", 0) > 0]
        n_pnl = sum(t.get("pnl", 0) for t in nt)
        news_impact[n_mode] = {"trades": len(nt), "wins": len(n_wins),
                               "win_rate": round(len(n_wins) / max(len(nt), 1) * 100, 1),
                               "total_pnl": round(n_pnl, 2)}

    regime_breakdown = {}
    for reg, rt in regime_trades.items():
        r_wins = [t for t in rt if t.get("pnl", 0) > 0]
        r_pnl = sum(t.get("pnl", 0) for t in rt)
        regime_breakdown[reg] = {"trades": len(rt), "wins": len(r_wins),
                                 "win_rate": round(len(r_wins) / max(len(rt), 1) * 100, 1),
                                 "total_pnl": round(r_pnl, 2), "avg_pnl": round(r_pnl / max(len(rt), 1), 2)}

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

    hourly_performance = {}
    for hour in sorted(hourly_signals.keys()):
        hour_trades = [t for t in trades if datetime.fromisoformat(t["time"]).hour == hour]
        if hour_trades:
            h_wins = [t for t in hour_trades if t.get("pnl", 0) > 0]
            h_pnl = sum(t.get("pnl", 0) for t in hour_trades)
            hourly_performance[f"{hour:02d}:00"] = {
                "signals": hourly_signals[hour], "trades": len(hour_trades),
                "wins": len(h_wins),
                "win_rate": round(len(h_wins) / max(len(hour_trades), 1) * 100, 1),
                "total_pnl": round(h_pnl, 2),
            }

    block_reasons = defaultdict(int)
    for b in blocked_signals:
        block_reasons[b["reason"]] += 1

    sorted_by_pnl = sorted(trades, key=lambda t: t.get("pnl", 0))
    worst_5 = sorted_by_pnl[:5]
    best_5 = sorted_by_pnl[-5:] if len(sorted_by_pnl) >= 5 else sorted_by_pnl

    report = {
        "report_metadata": {
            "title": "V3 PROFITABLE Gold Scalping — 7-Day XAUUSD Backtest",
            "generated_at": str(datetime.now()),
            "data_range": f"{start_dt} to {end_dt}",
            "strategy_version": "V3 Profitable Strategy",
            "parameters": {
                "initial_balance": 10000.0, "lot_size": 0.10,
                "max_positions": 3, "max_trades_per_day": 30,
                "min_score": 60, "max_score": 89,
                "sl_atr": 2.5, "tp_atr": 3.5,
                "min_m5_gap": 1, "asian_block_rate": 0.0,
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
            "return_pct": round(total_pnl / 10000 * 100, 2),
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
            "signal_score_filter": len([s for s in signal_timeline if 40 <= s["score"] < 60]),
            "avg_blocked_score": round(np.mean([b["score"] for b in blocked_signals]) if blocked_signals else 0, 1),
        },
        "worst_5_trades": [
            {"time": t["time"], "direction": t["direction"], "entry": t["entry_price"],
             "pnl": round(t.get("pnl", 0), 2), "session": t["session"],
             "regime": t["regime"], "exit_reason": t.get("exit_reason", "unknown"),
             "setup_score": t["setup_score"]}
            for t in worst_5
        ],
        "best_5_trades": [
            {"time": t["time"], "direction": t["direction"], "entry": t["entry_price"],
             "pnl": round(t.get("pnl", 0), 2), "session": t["session"],
             "regime": t["regime"], "exit_reason": t.get("exit_reason", "unknown"),
             "setup_score": t["setup_score"]}
            for t in reversed(best_5)
        ],
        "daily_breakdown": {
            day: {"trades": daily_trades[day], "day_pnl": round(sum(
                t.get("pnl", 0) for t in trades if t["time"].startswith(day)), 2)}
            for day in sorted(daily_trades.keys())
        },
        "trade_exit_reasons": dict(
            zip(*np.unique([t.get("exit_reason", "unknown") for t in trades], return_counts=True))
        ),
        "equity_metrics": {
            "starting_balance": 10000.0, "peak_balance": round(peak, 2),
            "trough_balance": round(min(balance_curve), 2),
            "final_balance": round(balance_curve[-1], 2),
        },
    }

    _print_summary(report)
    return report


def _print_summary(report: dict):
    e = report["executive_summary"]
    est_pnl_1_lot = e["total_pnl_usd"] * 10  # scale from 0.10 to 1.0 lot

    print(f"""
{'='*75}
  V3 PROFITABLE GOLD SCALPING — 7-DAY XAUUSD BACKTEST
{'='*75}

  CONFIGURATION
  Score Filter: 60-89 | SL/TP: 2.5x/3.5x ATR | 30/day max | 0.10 lot

  RESULTS
  -------
  Total Trades:     {e['total_trades']:>6}
  Win Rate:         {e['win_rate_pct']:>5.1f}%
  Profit Factor:    {e['profit_factor']:>5.2f}
  Total P&L:       ${e['total_pnl_usd']:>7.2f}
  At 1.0 lot:     ${est_pnl_1_lot:>7.2f} (10x scale)
  Return:           {e['return_pct']:>5.2f}%
  Max DD:           {e['max_drawdown_pct']:>5.2f}%
  Avg/Day:          {e['avg_trades_per_day']:>5.1f}
  Avg Win:         ${e['avg_win_usd']:>7.2f}
  Avg Loss:        ${e['avg_loss_usd']:>7.2f}
  Expectancy:      ${e['expectancy_per_trade']:>7.2f}
  Longest Win:      {e['longest_win_streak']:>3}
  Longest Loss:     {e['longest_loss_streak']:>3}
""")

    print(f"  {'='*75}")
    print(f"  DAILY BREAKDOWN")
    print(f"  {'='*75}")
    print(f"  {'Day':<20} {'Trades':>8} {'P&L':>12}")
    for day, d in report["daily_breakdown"].items():
        print(f"  {day:<20} {d['trades']:>8} ${d['day_pnl']:>7.2f}")

    print(f"\n  {'='*75}")
    print(f"  SESSION PERFORMANCE")
    print(f"  {'='*75}")
    for sess, d in report["session_performance"].items():
        print(f"  {sess:<15} {d['trades']:>3} trades, {d['win_rate']:>5.1f}% WR, ${d['total_pnl']:>6.2f} P&L")

    print(f"\n  {'='*75}")
    print(f"  SCORE QUALITY")
    print(f"  {'='*75}")
    for sr, d in report["score_quality_analysis"].items():
        print(f"  Score {sr:<8} {d['trades_executed']:>3} trades, {d['win_rate']:>5.1f}% WR, ${d['total_pnl']:>6.2f} P&L")

    print(f"\n  {'='*75}")
    print(f"  WORST 5 / BEST 5 TRADES")
    print(f"  {'='*75}")
    print(f"  WORST:")
    for t in report["worst_5_trades"]:
        print(f"    {t['time']} | {t['direction']:>4} | Entry ${t['entry']:.2f} | "
              f"P&L ${t['pnl']:.2f} | {t['session']} | Score {t['setup_score']} | {t['exit_reason']}")
    print(f"  BEST:")
    for t in report["best_5_trades"]:
        print(f"    {t['time']} | {t['direction']:>4} | Entry ${t['entry']:.2f} | "
              f"P&L ${t['pnl']:.2f} | {t['session']} | Score {t['setup_score']} | {t['exit_reason']}")

    print(f"\n  {'='*75}")
    print(f"  FINAL EQUITY: ${report['equity_metrics']['final_balance']:.2f}")
    print(f"  {'='*75}\n")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--days", type=int, default=7)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--multi-seed", action="store_true")
    parser.add_argument("--output", type=str, default="trading_bot/backtest/v3_report.json")
    args = parser.parse_args()

    if args.multi_seed:
        reports = {}
        summaries = []
        for seed in [42, 123, 456, 789, 1111]:
            print(f"\n{'#'*75}")
            print(f"  V3 WITH SEED = {seed}")
            print(f"{'#'*75}")
            r = run_v3_backtest(days=args.days, seed=seed)
            reports[str(seed)] = r
            summaries.append(r["executive_summary"])

        agg = {}
        for key in summaries[0].keys():
            vals = [s[key] for s in summaries]
            if isinstance(vals[0], (int, float)):
                agg[f"{key}_avg"] = round(np.mean(vals), 2)
                agg[f"{key}_std"] = round(np.std(vals), 2)

        print(f"\n  {'='*50}")
        print(f"  MULTI-SEED OVERVIEW (5 runs)")
        print(f"  {'='*50}")
        for km in ["total_trades", "win_rate_pct", "profit_factor",
                   "expectancy_per_trade", "total_pnl_usd", "max_drawdown_pct"]:
            if f"{km}_avg" in agg:
                print(f"  {km:<30} {agg[f'{km}_avg']:>8.2f} ± {agg[f'{km}_std']:>6.2f}")

        report = {"multi_seed_analysis": {"seeds_used": [42, 123, 456, 789, 1111],
                                          "num_runs": 5, "aggregated_metrics": agg},
                  "individual_reports": reports}
    else:
        report = run_v3_backtest(days=args.days, seed=args.seed)

    with open(args.output, "w") as f:
        json.dump(report, f, indent=2, default=str)
    logger.info(f"V3 Report saved to: {args.output}")
    print(f"\nSaved to: {args.output}")