"""
STRATEGY FAILURE ANALYSIS.

Classifies every losing trade across multiple simulations into failure categories.
Determines the single biggest structural weakness in the EMA-trend-following strategy.

No code modifications. Pure analysis of existing trade data.
"""

import json
import random
import statistics
from datetime import datetime
from collections import Counter

import numpy as np
import pandas as pd

from trading_bot.config import Config
from trading_bot.utils.logger import logger
from trading_bot.backtest.simulation_engine import SimulatedPortfolio
from trading_bot.backtest.edge_analysis import _generate_trending_data, _prices_to_ohlcv


# ---------------------------------------------------------------------------
# Failure classifiers
# ---------------------------------------------------------------------------

def classify_trade(trade, ohlcv, entry_index, window_before=20, window_after=20):
    """
    Classify a losing trade into one of 7 failure categories.

    Examines price action, EMA relationships, and volatility patterns
    before entry and during the trade.
    """
    from trading_bot.indicators.technical_indicators import compute_all_indicators

    pnl = trade.get("pnl", 0)
    if pnl >= 0:
        return None  # not a loss

    action = trade.get("action", "BUY")
    entry_price = trade.get("entry_price", 0)
    exit_price = trade.get("exit_price", 0)

    # Get pre-entry and post-entry data
    pre_data = ohlcv.iloc[max(0, entry_index-window_before):entry_index]
    post_data = ohlcv.iloc[entry_index:min(len(ohlcv), entry_index+window_after)]

    if len(pre_data) < 10 or len(post_data) < 5:
        return "unknown"

    pre_close = pre_data["close"].values
    post_close = post_data["close"].values
    pre_high = pre_data["high"].values
    pre_low = pre_data["low"].values

    # Compute indicators on pre-entry data
    window_df = ohlcv.iloc[:entry_index+1]
    if len(window_df) < 50:
        return "unknown"
    indicators = compute_all_indicators(window_df)

    # EMA values at entry
    emas = {}
    for col in indicators.get("emas", pd.DataFrame()).columns:
        series = indicators["emas"][col]
        if len(series) > 0:
            emas[col] = series.iloc[-1]

    # Trend direction before entry (last 10 candles)
    pre_trend = "neutral"
    if len(pre_close) >= 10:
        if pre_close[-1] > pre_close[-10]:
            pre_trend = "up"
        elif pre_close[-1] < pre_close[-10]:
            pre_trend = "down"

    # Volatility before entry
    if len(pre_close) >= 14:
        pre_range = np.std(np.diff(pre_close))
    else:
        pre_range = 0.0001

    # Post-entry volatility
    if len(post_close) >= 5:
        post_range = np.std(np.diff(post_close))
    else:
        post_range = pre_range

    # --- 1. Trend reversal detection ---
    # Did EMA direction change during the trade?
    if len(post_close) >= 5:
        post_sma_short = np.mean(post_close[:5])
        post_sma_long = np.mean(post_close)
        pre_sma_short = np.mean(pre_close[-5:])
        pre_sma_long = np.mean(pre_close)

        if action == "BUY":
            trend_reversed = (pre_sma_short > pre_sma_long and post_sma_short < post_sma_long)
        else:
            trend_reversed = (pre_sma_short < pre_sma_long and post_sma_short > post_sma_long)

        if trend_reversed:
            return "trend_reversal"

    # --- 2. Premature exit detection ---
    # Did price hit SL but then reverse back in our direction?
    if action == "BUY":
        price_after_exit = post_close[3:] if len(post_close) > 3 else []
        if len(price_after_exit) > 0 and np.max(price_after_exit) > entry_price * 1.01:
            return "premature_exit"
    else:
        price_after_exit = post_close[3:] if len(post_close) > 3 else []
        if len(price_after_exit) > 0 and np.min(price_after_exit) < entry_price * 0.99:
            return "premature_exit"

    # --- 3. High-volatility whipsaw ---
    # Post-entry volatility much higher than pre-entry
    if pre_range > 0 and post_range / max(pre_range, 0.00001) > 3:
        return "high_vol_whipsaw"

    # --- 4. Late trend entry ---
    # Entry happened after a long trend, close to a reversal point
    if len(pre_close) >= 20:
        pre_trend_strength = abs(pre_close[-1] - pre_close[-20]) / pre_close[-20] * 100
        if pre_trend_strength > 3:  # strong move before entry
            # Check if we're near the extreme
            if action == "BUY" and pre_close[-1] >= np.percentile(pre_close, 90):
                return "late_trend_entry"
            elif action == "SELL" and pre_close[-1] <= np.percentile(pre_close, 10):
                return "late_trend_entry"

    # --- 5. False breakout ---
    # Price broke a level but immediately reversed
    if action == "BUY":
        recent_high = np.max(pre_high[-10:]) if len(pre_high) >= 10 else 0
        if entry_price >= recent_high * 0.999 and exit_price < entry_price * 0.995:
            return "false_breakout"
    else:
        recent_low = np.min(pre_low[-10:]) if len(pre_low) >= 10 else 0
        if entry_price <= recent_low * 1.001 and exit_price > entry_price * 1.005:
            return "false_breakout"

    # --- 6. Sideways/chop ---
    # Low volatility before and during, no clear direction
    if pre_range < 0.0002 and post_range < pre_range * 1.5:
        return "sideways_chop"

    return "unknown"


def run_failure_analysis(simulations=8, bars=600):
    """Run multiple simulations and classify all losing trades."""
    from trading_bot.strategy.rule_engine import RuleEngine
    from trading_bot.risk.risk_manager import RiskManager
    from trading_bot.ai.deepseek_client import DeepSeekClient
    from trading_bot.indicators.technical_indicators import compute_all_indicators
    from trading_bot.main import build_ai_payload, determine_trade_action, compute_lot_size, compute_sl_tp

    logger.info("\n" + "=" * 70)
    logger.info("STRATEGY FAILURE ANALYSIS")
    logger.info("Classifying every losing trade into failure categories")
    logger.info("=" * 70)

    # Store trades with their classification
    classified_trades = []
    raw_trades = []

    for sim in range(simulations):
        prices = _generate_trending_data(length=bars, trend_strength=0.0002, noise=0.0005)
        ohlcv = _prices_to_ohlcv(prices, spread_points=10)
        ohlcv.attrs["symbol"] = "EURUSD"
        ohlcv.attrs["timeframe"] = "H1"

        portfolio = SimulatedPortfolio(initial_balance=10000.0)
        rule_engine = RuleEngine()
        risk_manager = RiskManager(account_balance=10000.0)
        deepseek = DeepSeekClient()
        deepseek.initialize()

        trade_entries = {}  # keyed by ticket, stores entry index

        warmup = 100
        for i in range(warmup, len(ohlcv)):
            window = ohlcv.iloc[:i+1]
            current_price = float(ohlcv["close"].iloc[i])
            curtime = str(ohlcv.index[i])

            indicators = compute_all_indicators(window)
            decision = rule_engine.analyze(ohlcv=window, indicators=indicators)
            ai_payload = build_ai_payload("EURUSD", "H1", window, indicators, decision)
            ai_analysis = deepseek.analyze_market(ai_payload)
            risk_eval = risk_manager.validate(rule_decision=decision, ai_analysis=ai_analysis, ohlcv=window)
            action = determine_trade_action(decision, ai_analysis)
            lot = compute_lot_size(decision, risk_manager, portfolio.balance)
            sl, tp = compute_sl_tp(decision, action, window) if action != "NONE" else (0.0, 0.0)

            if risk_eval["approved"] and decision.get("setup_valid") and action != "NONE":
                in_pos = any(p["symbol"] == "EURUSD" for p in portfolio.open_positions)
                if not in_pos:
                    pos = portfolio.open_position(
                        action, "EURUSD", lot, current_price, sl, tp,
                        curtime, spread_cost=0.0001, slippage=0.00005, fail_rate=0.0
                    )
                    if pos:
                        trade_entries[pos["ticket"]] = i

            closed = portfolio.update_positions(current_price, curtime, spread=0.0001)
            for c in closed:
                entry_idx = trade_entries.get(c.get("ticket"), i)
                c["entry_index"] = entry_idx
                raw_trades.append(c)

        # Close remaining
        final_price = float(ohlcv["close"].iloc[-1])
        closed_final = portfolio.update_positions(final_price, str(ohlcv.index[-1]), spread=0.0001)
        for c in closed_final:
            entry_idx = trade_entries.get(c.get("ticket"), len(ohlcv)-1)
            c["entry_index"] = entry_idx
            raw_trades.append(c)

    # Classify all losing trades
    logger.info(f"\nClassifying {len(raw_trades)} total trades...")
    for t in raw_trades:
        if t.get("pnl", 0) < 0:
            entry_idx = t.get("entry_index", 0)
            category = classify_trade(t, ohlcv, entry_idx)
            classified_trades.append({**t, "category": category})

    # Compute statistics
    total_losses = len(classified_trades)
    if total_losses == 0:
        return {"error": "No losing trades found"}

    categories = Counter(t["category"] for t in classified_trades)
    category_pnl = {}
    for cat in categories:
        cat_trades = [t for t in classified_trades if t["category"] == cat]
        category_pnl[cat] = {
            "count": len(cat_trades),
            "total_pnl": round(sum(t["pnl"] for t in cat_trades), 2),
            "avg_pnl": round(np.mean([t["pnl"] for t in cat_trades]), 2),
            "pct_of_losses": round(len(cat_trades) / total_losses * 100, 1),
            "pct_of_drawdown": round(abs(sum(t["pnl"] for t in cat_trades)) / max(abs(sum(t["pnl"] for t in classified_trades)), 1) * 100, 1),
        }

    # Determine which category is easiest to eliminate
    # "premature_exit" is easiest — widen the SL
    # "false_breakout" is hardest — requires entry logic change
    elimination_difficulty = {
        "premature_exit": "easy",
        "late_trend_entry": "medium",
        "high_vol_whipsaw": "medium",
        "sideways_chop": "hard",
        "trend_reversal": "hard",
        "false_breakout": "hard",
        "unknown": "hard",
    }

    # Build report
    report = {
        "total_trades_analyzed": len(raw_trades),
        "total_losses": total_losses,
        "loss_rate": round(total_losses / max(len(raw_trades), 1) * 100, 1),
        "total_loss_pnl": round(sum(t["pnl"] for t in classified_trades), 2),
        "categories": {},
    }

    ranked = sorted(categories.items(), key=lambda x: category_pnl[x[0]]["pct_of_drawdown"], reverse=True)

    print(f"\nTotal trades analyzed: {len(raw_trades)}")
    print(f"Total losing trades: {total_losses} (loss rate: {report['loss_rate']}%)")
    print(f"Total loss P&L: ${report['total_loss_pnl']}")

    print("\n" + "=" * 100)
    print("FAILURE CATEGORIES RANKED BY DRAWDOWN CONTRIBUTION")
    print("=" * 100)
    print(f"{'Category':<25} {'Count':<8} {'% Losses':<10} {'% DD':<10} {'Avg Loss':<12} {'Elim. Difficulty':<18}")
    print("-" * 100)

    for cat, _ in ranked:
        data = category_pnl[cat]
        diff = elimination_difficulty.get(cat, "unknown")
        print(f"{cat:<25} {data['count']:<8} {data['pct_of_losses']:<10} {data['pct_of_drawdown']:<10} "
              f"${data['avg_pnl']:<9.2f} {diff:<18}")
        report["categories"][cat] = data

    # Find the single biggest structural weakness
    top_cat = ranked[0][0] if ranked else "unknown"
    top_data = category_pnl.get(top_cat, {})

    # Find the easiest to eliminate
    for cat, _ in ranked:
        if elimination_difficulty.get(cat) == "easy":
            easiest = cat
            easiest_data = category_pnl.get(cat, {})
            break
    else:
        easiest = None
        easiest_data = {}

    print("\n" + "=" * 70)
    print("STRUCTURAL WEAKNESS ANALYSIS")
    print("=" * 70)

    print(f"\n1. BIGGEST CONTRIBUTOR TO DRAWDOWN: {top_cat.upper()}")
    print(f"   - {top_data.get('count', 0)} losing trades ({top_data.get('pct_of_losses', 0)}% of all losses)")
    print(f"   - Contributes {top_data.get('pct_of_drawdown', 0)}% of total drawdown")
    print(f"   - Average loss: ${top_data.get('avg_pnl', 0)}")

    if easiest and easiest != top_cat:
        print(f"\n2. EASIEST TO ELIMINATE: {easiest.upper()}")
        print(f"   - {easiest_data.get('count', 0)} losing trades ({easiest_data.get('pct_of_losses', 0)}% of all losses)")
        print(f"   - Contributes {easiest_data.get('pct_of_drawdown', 0)}% of total drawdown")
        print(f"   - Fix: Adjust SL/TP parameters to give trades more breathing room")

    # Detailed description of the top weakness
    print(f"\n3. ROOT CAUSE: {top_cat}")
    descriptions = {
        "trend_reversal": (
            "The EMA trend-following strategy enters long after an uptrend is established. "
            "When the trend reverses (EMA cross down), the system is still holding long positions. "
            "The SL is 1.5x ATR away, but trend reversals often exceed this distance before reversing again. "
            "This is a STRUCTURAL weakness of trend-following — you cannot avoid it with this strategy type."
        ),
        "false_breakout": (
            "Price breaks above a resistance level, the system enters long, but the breakout fails "
            "immediately and price retraces below entry. The system's SL is too tight to survive "
            "the false breakout retracement."
        ),
        "sideways_chop": (
            "The system enters during a ranging market where EMAs are flat. "
            "The trend-following logic should not trade in chop, but the EMA alignment occasionally "
            "appears directional when it's actually noise."
        ),
        "high_vol_whipsaw": (
            "A sudden volatility spike triggers both the entry condition and immediately the SL. "
            "The system enters as volatility expands, then gets stopped out by the same volatility."
        ),
        "late_trend_entry": (
            "The system enters near the end of a trend. The EMA alignment confirms the trend, "
            "but price has already moved significantly. The entry is at the extreme, and "
            "the subsequent reversal catches the SL."
        ),
        "premature_exit": (
            "The SL is hit during a normal pullback within an ongoing trend. "
            "Price then continues in the original direction, but the system is already out."
        ),
    }

    desc = descriptions.get(top_cat, "Unknown structural weakness.")
    print(f"   {desc}")

    print(f"\n4. FINAL VERDICT:")
    if top_cat in ("trend_reversal", "false_breakout"):
        print("   The single biggest structural weakness is that TREND-FOLLOWING inherently loses")
        print("   money during trend reversals. This is not fixable through risk management.")
        print("   The EMA crossover strategy has a positive expectancy ONLY because trends last")
        print("   longer than reversals on average. When they don't, losses are concentrated.")
        print("   RECOMMENDATION: The strategy needs a secondary filter (momentum divergence,")
        print("   volatility contraction, or multi-timeframe confirmation) to avoid trend reversal entries.")
    elif top_cat in ("high_vol_whipsaw", "premature_exit"):
        print("   The single biggest structural weakness is SL/TP PLACEMENT. The ATR-based")
        print("   stop loss is too tight for the market's natural noise. This is FIXABLE")
        print("   through widening the SL while reducing position size (already validated in experiment).")
    elif top_cat == "sideways_chop":
        print("   The single biggest structural weakness is the LACK OF A CHOP FILTER.")
        print("   Adding an ADX filter or Bollinger Band width filter would eliminate these losses.")
    else:
        print("   The dominant failure mode requires further investigation.")

    print("=" * 70)
    return report


if __name__ == "__main__":
    run_failure_analysis()