"""
PROFITABILITY & EDGE ANALYSIS — Honest Evaluation of Trading System Performance.

This module runs structured analysis on simulation results to determine:
- Does the system have a real trading edge?
- What is the expectancy per trade?
- Does AI add value or just noise?
- Is the system viable for demo trading?

This is STRICTLY ANALYSIS — no architecture changes, no optimizations.
"""

import json
import math
import random
from datetime import datetime
from typing import Optional

import numpy as np

from trading_bot.config import Config
from trading_bot.utils.logger import logger
from trading_bot.backtest.simulation_engine import SimulatedPortfolio, SafetyAudit


# ---------------------------------------------------------------------------
# Mock data for edge analysis (no MT5 dependency)
# ---------------------------------------------------------------------------

def _generate_trending_data(length=800, trend_strength=0.0002,
                            noise=0.0005, start_price=1.1000):
    """Generate data that has a detectable trend for edge testing."""
    prices = [start_price]
    direction = 1
    for i in range(1, length):
        trend = trend_strength * direction
        noise_val = random.gauss(0, noise)
        price = prices[-1] + trend + noise_val
        if price < start_price * 0.9:
            direction = 1
        elif price > start_price * 1.1:
            direction = -1
        prices.append(max(0.0001, price))
    return prices


def _generate_random_walk(length=800, volatility=0.0008, start_price=1.1000):
    """Pure random walk — no edge should exist."""
    prices = [start_price]
    for i in range(1, length):
        prices.append(prices[-1] + random.gauss(0, volatility))
    return [max(0.0001, p) for p in prices]


def _prices_to_ohlcv(prices, spread_points=10):
    """Convert price array to OHLCV DataFrame-like structure."""
    import pandas as pd
    closes = np.array(prices)
    highs = closes + abs(np.random.normal(0, 0.0003, len(closes)))
    lows = closes - abs(np.random.normal(0, 0.0003, len(closes)))
    opens = closes - np.random.normal(0, 0.0002, len(closes))
    spreads = np.full(len(closes), spread_points)
    volumes = np.random.randint(100, 1000, len(closes))
    times = pd.date_range(end=datetime.now(), periods=len(closes), freq="h")
    df = pd.DataFrame({
        "time": times, "open": opens, "high": highs, "low": lows,
        "close": closes, "tick_volume": volumes, "spread": spreads, "real_volume": volumes
    })
    df.set_index("time", inplace=True)
    return df


# ---------------------------------------------------------------------------
# Edge Analysis — the honest evaluation
# ---------------------------------------------------------------------------

class EdgeAnalyzer:
    """
    Analyzes trading system performance to determine if a real edge exists.

    Methodology:
        1. Run simulations on trending data (should detect edge)
        2. Run simulations on random walk data (should NOT detect edge — if it does, it's overfitting)
        3. Compare AI vs no-AI scenarios
        4. Compute expectancy, profit factor, risk-reward metrics
    """

    def __init__(self):
        self.results = {}

    def analyze_all(self):
        """Run full edge analysis and return structured report."""
        logger.info("\n" + "=" * 70)
        logger.info("PROFITABILITY & EDGE ANALYSIS")
        logger.info("=" * 70)

        report = {
            "timestamp": datetime.now().isoformat(),
            "tests": {},
            "expectancy": None,
            "win_rate": None,
            "profit_factor": None,
            "max_drawdown": None,
            "ai_value_added": "unknown",
            "rule_engine_quality": "unknown",
            "system_edge": "unknown",
            "recommendation": "unknown",
            "critical_findings": [],
        }

        # --- Test 1: Trending data with full pipeline ---
        logger.info("\n--- Test 1: Trending Data (should show edge) ---")
        trending_result = self._test_market("trending", _generate_trending_data)
        report["tests"]["trending_market"] = trending_result

        # --- Test 2: Random walk data (should show no edge) ---
        logger.info("\n--- Test 2: Random Walk Data (should show NO edge) ---")
        random_result = self._test_market("random_walk", _generate_random_walk)
        report["tests"]["random_walk"] = random_result

        # --- Test 3: AI vs No-AI comparison ---
        logger.info("\n--- Test 3: AI vs No-AI Comparison ---")
        ai_comp = self._compare_ai_vs_noai()
        report["tests"]["ai_comparison"] = ai_comp

        # --- Test 4: Rule engine effectiveness ---
        logger.info("\n--- Test 4: Rule Engine Effectiveness ---")
        re_eff = self._rule_engine_analysis()
        report["tests"]["rule_engine"] = re_eff

        # --- Synthesize final conclusions ---
        self._synthesize(report)

        # Print report
        self._print_report(report)
        return report

    def _test_market(self, name: str, price_generator, iterations=3) -> dict:
        """Run multiple simulations on a market type and aggregate results."""
        from trading_bot.strategy.rule_engine import RuleEngine
        from trading_bot.risk.risk_manager import RiskManager
        from trading_bot.ai.deepseek_client import DeepSeekClient

        all_trades = []
        total_pnl = 0.0
        peak_balance = 10000.0
        low_balance = 10000.0
        balance = 10000.0

        for run in range(iterations):
            prices = price_generator(length=600)
            ohlcv = _prices_to_ohlcv(prices)
            ohlcv.attrs["symbol"] = name.upper()
            ohlcv.attrs["timeframe"] = "H1"

            portfolio = SimulatedPortfolio(initial_balance=10000.0)
            rule_engine = RuleEngine()
            risk_manager = RiskManager(account_balance=10000.0)
            deepseek = DeepSeekClient()
            deepseek.initialize()

            from trading_bot.indicators.technical_indicators import compute_all_indicators
            from trading_bot.main import build_ai_payload, determine_trade_action, compute_lot_size, compute_sl_tp

            warmup = 100
            for i in range(warmup, len(ohlcv)):
                window = ohlcv.iloc[:i+1]
                current_price = float(ohlcv["close"].iloc[i])
                current_time = str(ohlcv.index[i])

                indicators = compute_all_indicators(window)
                decision = rule_engine.analyze(ohlcv=window, indicators=indicators)
                ai_payload = build_ai_payload(name.upper(), "H1", window, indicators, decision)
                ai_analysis = deepseek.analyze_market(ai_payload)
                risk_eval = risk_manager.validate(rule_decision=decision, ai_analysis=ai_analysis, ohlcv=window)
                action = determine_trade_action(decision, ai_analysis)
                lot = compute_lot_size(decision, risk_manager, portfolio.balance)
                sl, tp = compute_sl_tp(decision, action, window) if action != "NONE" else (0.0, 0.0)

                if risk_eval["approved"] and decision.get("setup_valid") and action != "NONE":
                    in_pos = any(p["symbol"] == name.upper() for p in portfolio.open_positions)
                    if not in_pos:
                        portfolio.open_position(action, name.upper(), lot, current_price, sl, tp,
                                                current_time, spread_cost=0.0001, slippage=0.00005, fail_rate=0.0)

                closed = portfolio.update_positions(current_price, current_time, spread=0.0001)
                for c in closed:
                    all_trades.append(c)
                    risk_manager.record_result(c["pnl"])

            # Close remaining
            final_price = float(ohlcv["close"].iloc[-1])
            closed_final = portfolio.update_positions(final_price, str(ohlcv.index[-1]), spread=0.0001)
            for c in closed_final:
                all_trades.append(c)

            total_pnl += portfolio.balance - 10000.0
            balance = portfolio.balance
            peak_balance = max(peak_balance, portfolio.equity_peak)
            low_balance = min(low_balance, portfolio.equity_low)

        return self._compute_metrics(all_trades, total_pnl / iterations, peak_balance, low_balance, name)

    def _compute_metrics(self, trades, avg_pnl, peak, low, label) -> dict:
        """Compute detailed performance metrics from a list of trades."""
        total = len(trades)
        if total == 0:
            return {"label": label, "trades": 0, "message": "No trades generated"}

        wins = [t for t in trades if t.get("pnl", 0) > 0]
        losses = [t for t in trades if t.get("pnl", 0) < 0]
        win_count = len(wins)
        loss_count = len(losses)
        win_rate = (win_count / total * 100) if total > 0 else 0

        avg_win = np.mean([t["pnl"] for t in wins]) if wins else 0
        avg_loss = abs(np.mean([t["pnl"] for t in losses])) if losses else 0

        # Expectancy = (Win% * AvgWin) - (Loss% * AvgLoss)
        win_pct = win_rate / 100
        loss_pct = 1 - win_pct
        expectancy = (win_pct * avg_win) - (loss_pct * avg_loss)

        # Profit factor
        gross_profit = sum(t["pnl"] for t in wins)
        gross_loss = abs(sum(t["pnl"] for t in losses))
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else 0

        # Risk-reward ratio
        risk_reward = avg_win / avg_loss if avg_loss > 0 else 0

        # Drawdown
        max_dd = round((peak - low) / max(peak, 1) * 100, 2) if peak > 0 else 0

        return {
            "label": label,
            "total_trades": total,
            "win_count": win_count,
            "loss_count": loss_count,
            "win_rate": round(win_rate, 2),
            "avg_win": round(avg_win, 2),
            "avg_loss": round(avg_loss, 2),
            "expectancy_per_trade": round(expectancy, 2),
            "profit_factor": round(profit_factor, 2),
            "risk_reward_ratio": round(risk_reward, 2),
            "max_drawdown": max_dd,
            "total_pnl": round(avg_pnl, 2),
        }

    def _compare_ai_vs_noai(self) -> dict:
        """Compare system performance WITH and WITHOUT AI."""
        from trading_bot.strategy.rule_engine import RuleEngine
        from trading_bot.risk.risk_manager import RiskManager
        from trading_bot.ai.deepseek_client import DeepSeekClient
        from trading_bot.indicators.technical_indicators import compute_all_indicators
        from trading_bot.main import build_ai_payload, determine_trade_action, compute_lot_size, compute_sl_tp

        results = {"with_ai": {}, "without_ai": {}}

        for variant, use_ai in [("with_ai", True), ("without_ai", False)]:
            all_trades = []
            balance = 10000.0
            peak = 10000.0
            low = 10000.0

            for run in range(2):
                prices = _generate_trending_data(length=500)
                ohlcv = _prices_to_ohlcv(prices)
                ohlcv.attrs["symbol"] = "TEST"
                ohlcv.attrs["timeframe"] = "H1"

                portfolio = SimulatedPortfolio(initial_balance=10000.0)
                rule_engine = RuleEngine()
                risk_manager = RiskManager(account_balance=10000.0)
                deepseek = DeepSeekClient()
                deepseek.initialize()

                warmup = 100
                for i in range(warmup, len(ohlcv)):
                    window = ohlcv.iloc[:i+1]
                    current_price = float(ohlcv["close"].iloc[i])
                    current_time = str(ohlcv.index[i])

                    indicators = compute_all_indicators(window)
                    decision = rule_engine.analyze(ohlcv=window, indicators=indicators)

                    if use_ai:
                        ai_payload = build_ai_payload("TEST", "H1", window, indicators, decision)
                        ai_analysis = deepseek.analyze_market(ai_payload)
                    else:
                        ai_analysis = {"sentiment": "neutral", "confidence": 50,
                                       "risk_flag": "medium", "conflicts_detected": False,
                                       "ai_unavailable": False}

                    risk_eval = risk_manager.validate(rule_decision=decision, ai_analysis=ai_analysis, ohlcv=window)
                    action = determine_trade_action(decision, ai_analysis)
                    lot = compute_lot_size(decision, risk_manager, portfolio.balance)
                    sl, tp = compute_sl_tp(decision, action, window) if action != "NONE" else (0.0, 0.0)

                    if risk_eval["approved"] and decision.get("setup_valid") and action != "NONE":
                        if not any(p["symbol"] == "TEST" for p in portfolio.open_positions):
                            portfolio.open_position(action, "TEST", lot, current_price, sl, tp,
                                                    current_time, spread_cost=0.0001, slippage=0.00005, fail_rate=0.0)

                    closed = portfolio.update_positions(current_price, current_time, spread=0.0001)
                    for c in closed:
                        all_trades.append(c)
                        risk_manager.record_result(c["pnl"])

                final_price = float(ohlcv["close"].iloc[-1])
                closed_final = portfolio.update_positions(final_price, str(ohlcv.index[-1]), spread=0.0001)
                for c in closed_final:
                    all_trades.append(c)

            metrics = self._compute_metrics(all_trades, 0, peak, low, variant)
            results[variant] = metrics

        # Compute AI value
        if results.get("with_ai", {}).get("expectancy_per_trade", 0) > results.get("without_ai", {}).get("expectancy_per_trade", 0):
            ai_value = "positive"
        elif abs(results.get("with_ai", {}).get("expectancy_per_trade", 0) - results.get("without_ai", {}).get("expectancy_per_trade", 0)) < 0.5:
            ai_value = "neutral"
        else:
            ai_value = "negative"

        results["ai_value_added"] = ai_value
        return results

    def _rule_engine_analysis(self) -> dict:
        """Evaluate rule engine setup quality vs outcome."""
        from trading_bot.strategy.rule_engine import RuleEngine
        from trading_bot.indicators.technical_indicators import compute_all_indicators

        rule_engine = RuleEngine()
        strong_setups = []
        weak_setups = []

        for run in range(5):
            prices = _generate_trending_data(length=300)
            ohlcv = _prices_to_ohlcv(prices)
            ohlcv.attrs["symbol"] = "TEST"
            ohlcv.attrs["timeframe"] = "H1"

            warmup = 100
            for i in range(warmup, len(ohlcv)):
                window = ohlcv.iloc[:i+1]
                indicators = compute_all_indicators(window)
                decision = rule_engine.analyze(ohlcv=window, indicators=indicators)
                strength = decision.get("setup_strength", 0)
                if strength >= 70:
                    strong_setups.append(decision)
                elif strength < 40:
                    weak_setups.append(decision)

        return {
            "label": "rule_engine_analysis",
            "total_analyses": len(strong_setups) + len(weak_setups),
            "strong_setups_count": len(strong_setups),
            "weak_setups_count": len(weak_setups),
            "strong_to_weak_ratio": round(len(strong_setups) / max(len(weak_setups), 1), 2),
            "note": "Higher ratio = rule engine effectively filters low-quality setups",
        }

    def _synthesize(self, report):
        """Synthesize all test results into honest conclusions."""
        tests = report["tests"]

        # --- Expectancy ---
        trending = tests.get("trending_market", {})
        random_walk = tests.get("random_walk", {})

        trending_exp = trending.get("expectancy_per_trade", 0)
        random_exp = random_walk.get("expectancy_per_trade", 0)

        if trending_exp > 0 and random_exp <= 0:
            report["system_edge"] = "profitable"
        elif trending_exp > 0 and random_exp > 0:
            report["system_edge"] = "needs_verification"
            report["critical_findings"].append("System shows positive expectancy on RANDOM data — possible overfitting")
        elif trending_exp <= 0:
            report["system_edge"] = "unprofitable"
            report["critical_findings"].append(f"Negative expectancy on trending data: ${trending_exp}/trade")

        report["expectancy"] = trending_exp

        # --- Win rate ---
        wr = trending.get("win_rate", 0)
        report["win_rate"] = wr
        if wr < 35:
            report["critical_findings"].append(f"Low win rate ({wr}%) — may require high R:R to be profitable")

        # --- Profit factor ---
        pf = trending.get("profit_factor", 0)
        report["profit_factor"] = pf
        if pf < 1.0:
            report["critical_findings"].append(f"Profit factor {pf} < 1.0 — system loses money over time")
        elif pf < 1.5:
            report["critical_findings"].append(f"Profit factor {pf} — marginal edge, high risk of drawdown")

        # --- Drawdown ---
        dd = trending.get("max_drawdown", 0)
        report["max_drawdown"] = dd
        if dd > 20:
            report["critical_findings"].append(f"Max drawdown {dd}% exceeds safe threshold")

        # --- AI value ---
        ai_comp = tests.get("ai_comparison", {})
        ai_val = ai_comp.get("ai_value_added", "unknown")
        report["ai_value_added"] = ai_val
        if ai_val == "neutral":
            report["critical_findings"].append("AI does not significantly improve edge — may be adding noise")
        elif ai_val == "negative":
            report["critical_findings"].append("AI REDUCES profitability — consider disabling AI sentiment")

        # --- Rule engine ---
        re = tests.get("rule_engine", {})
        ratio = re.get("strong_to_weak_ratio", 0)
        if ratio >= 2.0:
            report["rule_engine_quality"] = "strong"
        elif ratio >= 1.0:
            report["rule_engine_quality"] = "neutral"
        else:
            report["rule_engine_quality"] = "weak"

        # --- Recommendation ---
        if report["system_edge"] == "profitable" and pf >= 1.3 and dd < 20 and len(report["critical_findings"]) <= 2:
            report["recommendation"] = "ready_for_demo"
        elif report["system_edge"] == "profitable" and pf >= 1.0:
            report["recommendation"] = "needs_adjustment"
        else:
            report["recommendation"] = "not_viable"

    def _print_report(self, report):
        """Print a clean, honest report to stdout."""
        print("\n" + "=" * 70)
        print("PROFITABILITY & EDGE ANALYSIS — FINAL REPORT")
        print("=" * 70)

        tests = report["tests"]

        # Trending market
        t = tests.get("trending_market", {})
        print(f"\n[TRENDING MARKET]")
        print(f"  Trades: {t.get('total_trades', 'N/A')}")
        print(f"  Win Rate: {t.get('win_rate', 'N/A')}%")
        print(f"  Avg Win: ${t.get('avg_win', 'N/A')}")
        print(f"  Avg Loss: ${t.get('avg_loss', 'N/A')}")
        print(f"  Expectancy: ${t.get('expectancy_per_trade', 'N/A')}/trade")
        print(f"  Profit Factor: {t.get('profit_factor', 'N/A')}")
        print(f"  R:R Ratio: {t.get('risk_reward_ratio', 'N/A')}")
        print(f"  Max Drawdown: {t.get('max_drawdown', 'N/A')}%")
        print(f"  Net P&L: ${t.get('total_pnl', 'N/A')}")

        # Random walk
        r = tests.get("random_walk", {})
        print(f"\n[RANDOM WALK — no edge baseline]")
        print(f"  Trades: {r.get('total_trades', 'N/A')}")
        print(f"  Win Rate: {r.get('win_rate', 'N/A')}%")
        print(f"  Expectancy: ${r.get('expectancy_per_trade', 'N/A')}/trade")
        print(f"  Profit Factor: {r.get('profit_factor', 'N/A')}")
        print(f"  Net P&L: ${r.get('total_pnl', 'N/A')}")

        # AI comparison
        ai = tests.get("ai_comparison", {})
        print(f"\n[AI vs NO-AI]")
        print(f"  With AI expectancy: ${ai.get('with_ai', {}).get('expectancy_per_trade', 'N/A')}")
        print(f"  Without AI expectancy: ${ai.get('without_ai', {}).get('expectancy_per_trade', 'N/A')}")
        print(f"  AI value: {ai.get('ai_value_added', 'unknown').upper()}")

        # Conclusions
        print(f"\n{'='*70}")
        print("HONEST ASSESSMENT")
        print(f"{'='*70}")
        print(f"  System Edge: {report.get('system_edge', 'unknown').upper()}")
        print(f"  Expectancy: ${report.get('expectancy', 'N/A')}/trade")
        print(f"  Win Rate: {report.get('win_rate', 'N/A')}%")
        print(f"  Profit Factor: {report.get('profit_factor', 'N/A')}")
        print(f"  Max Drawdown: {report.get('max_drawdown', 'N/A')}%")
        print(f"  AI Value: {report.get('ai_value_added', 'unknown').upper()}")
        print(f"  Rule Engine: {report.get('rule_engine_quality', 'unknown').upper()}")
        print(f"  Recommendation: {report.get('recommendation', 'unknown')}")

        if report.get("critical_findings"):
            print(f"\n  CRITICAL FINDINGS ({len(report['critical_findings'])}):")
            for f in report["critical_findings"]:
                print(f"    - {f}")

        print(f"\n{'='*70}")
        print("BRUTALLY HONEST SUMMARY:")
        if report["system_edge"] == "profitable" and report["recommendation"] == "ready_for_demo":
            print("  The system shows a measurable edge in trending conditions.")
            print("  It correctly produces no edge on random data (no overfitting).")
            print("  Current profitability is marginal — expect variance in live trading.")
            print("  RECOMMENDATION: Proceed to demo testing with conservative sizing.")
        elif report["system_edge"] == "profitable":
            print("  The system has a weak positive edge but needs refinement.")
            print("  RECOMMENDATION: Adjust parameters before demo.")
        else:
            print("  The system does NOT demonstrate a reliable trading edge.")
            print("  Current strategy logic is insufficient for consistent profitability.")
            print("  RECOMMENDATION: Not viable for live or demo trading in current state.")
        print("=" * 70)


def run_edge_analysis():
    """Run the complete edge analysis."""
    analyzer = EdgeAnalyzer()
    return analyzer.analyze_all()


if __name__ == "__main__":
    run_edge_analysis()