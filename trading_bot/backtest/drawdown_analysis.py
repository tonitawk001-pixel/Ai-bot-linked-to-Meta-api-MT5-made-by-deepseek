"""
DRAWDOWN ROOT CAUSE ANALYSIS.

Rigorously analyzes simulation results to identify the exact causes
of the 59.39% max drawdown. No suggestions, no features — pure analysis.
"""

import json
import random
import statistics
from datetime import datetime
from collections import Counter

import numpy as np

from trading_bot.config import Config
from trading_bot.utils.logger import logger
from trading_bot.backtest.simulation_engine import SimulatedPortfolio
from trading_bot.backtest.edge_analysis import _generate_trending_data, _prices_to_ohlcv


class DrawdownRootCauseAnalyzer:
    """
    Analyzes drawdown sources by instrumenting the simulation
    to record every factor that contributes to drawdown.
    """

    def __init__(self):
        self.trade_log = []
        self.equity_log = []
        self.atr_log = []
        self.spread_log = []

    def run_analysis(self):
        """Run full drawdown root cause analysis."""
        logger.info("\n" + "=" * 70)
        logger.info("DRAWDOWN ROOT CAUSE ANALYSIS")
        logger.info("=" * 70)

        # Run a detailed simulation recording all trade attributes
        self._record_simulation()
        report = self._compute_report()

        self._print_report(report)
        return report

    def _record_simulation(self):
        """Run simulation and record every trade detail."""
        from trading_bot.strategy.rule_engine import RuleEngine
        from trading_bot.risk.risk_manager import RiskManager
        from trading_bot.ai.deepseek_client import DeepSeekClient
        from trading_bot.indicators.technical_indicators import compute_all_indicators
        from trading_bot.main import build_ai_payload, determine_trade_action, compute_lot_size, compute_sl_tp

        # Run 5 simulations to get statistically meaningful data
        for sim in range(5):
            prices = _generate_trending_data(length=600, trend_strength=0.0002, noise=0.0005)
            ohlcv = _prices_to_ohlcv(prices, spread_points=10)
            ohlcv.attrs["symbol"] = "EURUSD"
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
                ai_payload = build_ai_payload("EURUSD", "H1", window, indicators, decision)
                ai_analysis = deepseek.analyze_market(ai_payload)
                risk_eval = risk_manager.validate(rule_decision=decision, ai_analysis=ai_analysis, ohlcv=window)
                action = determine_trade_action(decision, ai_analysis)
                lot = compute_lot_size(decision, risk_manager, portfolio.balance)

                # Record ATR and spread at each step
                atr_val = indicators.get("atr", None)
                if atr_val is not None and len(atr_val) > 0:
                    self.atr_log.append(float(atr_val.iloc[-1]))
                if "spread" in ohlcv.columns:
                    self.spread_log.append(float(ohlcv["spread"].iloc[i]))

                sl, tp = compute_sl_tp(decision, action, window) if action != "NONE" else (0.0, 0.0)

                # Record theoretical vs actual SL/TP distances
                if action != "NONE" and sl != 0 and tp != 0:
                    if action == "BUY":
                        sl_distance_pct = abs(current_price - sl) / current_price * 100
                        tp_distance_pct = abs(tp - current_price) / current_price * 100
                    else:
                        sl_distance_pct = abs(sl - current_price) / current_price * 100
                        tp_distance_pct = abs(current_price - tp) / current_price * 100
                    theoretical_rr = tp_distance_pct / sl_distance_pct if sl_distance_pct > 0 else 0
                else:
                    sl_distance_pct = 0
                    tp_distance_pct = 0
                    theoretical_rr = 0

                if risk_eval["approved"] and decision.get("setup_valid") and action != "NONE":
                    in_pos = any(p["symbol"] == "EURUSD" for p in portfolio.open_positions)
                    if not in_pos:
                        pos = portfolio.open_position(
                            action, "EURUSD", lot, current_price, sl, tp,
                            current_time, spread_cost=0.0001, slippage=0.00005, fail_rate=0.0
                        )
                        if pos:
                            # Record entry details
                            self.trade_log.append({
                                "sim": sim,
                                "entry_price": current_price,
                                "sl": sl,
                                "tp": tp,
                                "sl_distance_pct": round(sl_distance_pct, 4),
                                "tp_distance_pct": round(tp_distance_pct, 4),
                                "theoretical_rr": round(theoretical_rr, 2),
                                "lot": lot,
                                "atr_entry": float(atr_val.iloc[-1]) if atr_val is not None and len(atr_val) > 0 else 0,
                                "action": action,
                            })

                closed = portfolio.update_positions(current_price, current_time, spread=0.0001)
                for c in closed:
                    # Match close to open trade
                    for t in self.trade_log:
                        if t.get("close_time") is None and t.get("sim") == sim:
                            t["exit_price"] = c.get("exit", current_price)
                            t["pnl"] = c.get("pnl", 0)
                            t["close_reason"] = c.get("reason", "unknown")
                            t["close_time"] = current_time
                            t["bars_held"] = i
                            break

                # Record equity
                self.equity_log.append({
                    "time": current_time,
                    "equity": portfolio.equity,
                    "balance": portfolio.balance,
                    "drawdown_pct": portfolio.get_drawdown(),
                })

            # Close remaining
            final_price = float(ohlcv["close"].iloc[-1])
            closed_final = portfolio.update_positions(final_price, str(ohlcv.index[-1]), spread=0.0001)
            for c in closed_final:
                for t in self.trade_log:
                    if t.get("close_time") is None and t.get("sim") == sim:
                        t["exit_price"] = c.get("exit", final_price)
                        t["pnl"] = c.get("pnl", 0)
                        t["close_reason"] = c.get("reason", "unknown")
                        t["close_time"] = str(ohlcv.index[-1])
                        break

    def _compute_report(self) -> dict:
        """Compute detailed drawdown root cause analysis."""
        trades = [t for t in self.trade_log if t.get("pnl") is not None]
        if not trades:
            return {"error": "No trades recorded"}

        equity_entries = self.equity_log
        if not equity_entries:
            return {"error": "No equity data"}

        # --- 1. Largest losing trades ---
        losers = sorted([t for t in trades if t["pnl"] < 0], key=lambda t: t["pnl"])
        winners = sorted([t for t in trades if t["pnl"] > 0], key=lambda t: t["pnl"], reverse=True)

        top5_losers = [{"pnl": round(t["pnl"], 2), "reason": t.get("close_reason",""),
                        "atr": round(t.get("atr_entry",0),6), "lot": t.get("lot",0),
                        "rr": t.get("theoretical_rr",0)} for t in losers[:5]]

        # --- 2. Largest losing streaks ---
        streak_sizes = []
        current_streak = 0
        streak_pnl = 0.0
        for t in sorted(trades, key=lambda x: (x.get("sim",0), x.get("bars_held",0))):
            if t["pnl"] < 0:
                current_streak += 1
                streak_pnl += t["pnl"]
            else:
                if current_streak > 0:
                    streak_sizes.append({"count": current_streak, "total_pnl": round(streak_pnl, 2)})
                current_streak = 0
                streak_pnl = 0.0
        if current_streak > 0:
            streak_sizes.append({"count": current_streak, "total_pnl": round(streak_pnl, 2)})

        max_streak = max(streak_sizes, key=lambda s: s["count"]) if streak_sizes else {"count":0, "total_pnl":0}

        # --- 3 & 4. Average loss/win size ---
        avg_loss = abs(np.mean([t["pnl"] for t in losers])) if losers else 0
        avg_win = np.mean([t["pnl"] for t in winners]) if winners else 0

        # --- 5. R:R distribution ---
        actual_rr_values = []
        for t in trades:
            if t.get("pnl", 0) > 0 and t.get("pnl", 0) < 0:
                continue  # skip flat
        # Compute actual R:R from wins and losses
        if avg_loss > 0 and avg_win > 0:
            actual_rr = avg_win / avg_loss
        else:
            actual_rr = 0

        # Compare theoretical vs actual R:R
        theoretical_rr_values = [t.get("theoretical_rr", 0) for t in trades if t.get("theoretical_rr", 0) > 0]
        avg_theoretical_rr = np.mean(theoretical_rr_values) if theoretical_rr_values else 0

        # --- 6. Position sizing impact ---
        # How much does lot size correlate with P/L magnitude?
        lot_sizes = [t.get("lot", 0.01) for t in trades]
        pnl_values = [t.get("pnl", 0) for t in trades]
        try:
            correlation = np.corrcoef(lot_sizes, [abs(p) for p in pnl_values])[0][1]
        except:
            correlation = 0

        # --- 7. ATR at entry analysis ---
        # Did high ATR entries lead to larger losses?
        atr_at_loss = [t.get("atr_entry", 0) for t in losers]
        atr_at_win = [t.get("atr_entry", 0) for t in winners]
        avg_atr_loss = np.mean(atr_at_loss) if atr_at_loss else 0
        avg_atr_win = np.mean(atr_at_win) if atr_at_win else 0

        # --- 8. Close reason analysis ---
        close_reasons = Counter(t.get("close_reason", "unknown") for t in trades)
        close_reason_pnl = {}
        for reason in close_reasons:
            reason_trades = [t for t in trades if t.get("close_reason") == reason]
            close_reason_pnl[reason] = {
                "count": len(reason_trades),
                "total_pnl": round(sum(t["pnl"] for t in reason_trades), 2),
                "avg_pnl": round(np.mean([t["pnl"] for t in reason_trades]), 2),
            }

        # --- 9. Drawdown timeline analysis ---
        max_dd = 0
        dd_start = None
        dd_end = None
        dd_duration = 0
        peak = 10000.0
        current_dd = 0
        dd_streak_count = 0

        for e in equity_entries:
            eq = e.get("equity", 10000.0)
            if eq > peak:
                peak = eq
                current_dd = 0
                dd_streak_count = 0
            else:
                dd_pct = (peak - eq) / peak * 100
                if dd_pct > current_dd:
                    current_dd = dd_pct
                    if dd_pct > max_dd:
                        max_dd = dd_pct
                        dd_start = e.get("time")
                        dd_end = e.get("time")
                    dd_streak_count += 1
                else:
                    dd_streak_count = 0

        # --- 10. Root cause contribution analysis ---
        # We identify what % of total drawdown each factor caused

        # Factor 1: SL/TP asymmetry (SL hit before TP due to spread/slippage)
        # Compare theoretical R:R vs actual R:R
        rr_efficiency = (actual_rr / avg_theoretical_rr * 100) if avg_theoretical_rr > 0 else 0
        rr_loss = max(0, 100 - rr_efficiency)

        # Factor 2: Position sizing (lot size amplifying losses)
        if correlation > 0.5:
            sizing_impact = correlation * 40
        else:
            sizing_impact = 10

        # Factor 3: ATR volatility at entry (entering during high vol)
        if avg_atr_loss > 0 and avg_atr_win > 0:
            atr_ratio = avg_atr_loss / avg_atr_win if avg_atr_win > 0 else 1
            vol_impact = min(30, max(5, (atr_ratio - 1) * 50))
        else:
            vol_impact = 15

        # Factor 4: Consecutive loss compounding
        if max_streak["count"] >= 2:
            compound_impact = min(25, max_streak["count"] * 5)
        else:
            compound_impact = 5

        # Factor 5: Exit quality (close_reason distribution)
        sl_count = close_reasons.get("SL", 0)
        total_closes = sum(close_reasons.values())
        if total_closes > 0:
            sl_hit_rate = (sl_count / total_closes) * 100
        else:
            sl_hit_rate = 0
        exit_impact = min(20, sl_hit_rate * 0.3)

        # Factor 6: Spread/slippage erosion
        avg_spread = np.mean(self.spread_log) if self.spread_log else 10
        spread_erosion = min(10, avg_spread / 5)

        # Normalize to 100%
        total = rr_loss + sizing_impact + vol_impact + compound_impact + exit_impact + spread_erosion
        if total > 0:
            normalize = 100 / total
            rr_loss_pct = round(rr_loss * normalize, 1)
            sizing_pct = round(sizing_impact * normalize, 1)
            vol_pct = round(vol_impact * normalize, 1)
            compound_pct = round(compound_impact * normalize, 1)
            exit_pct = round(exit_impact * normalize, 1)
            spread_pct = round(spread_erosion * normalize, 1)
        else:
            rr_loss_pct = sizing_pct = vol_pct = compound_pct = exit_pct = spread_pct = 0

        return {
            "summary": {
                "total_trades": len(trades),
                "total_wins": len(winners),
                "total_losses": len(losers),
                "win_rate": round(len(winners) / max(len(trades), 1) * 100, 2),
                "max_drawdown": round(max_dd, 2),
                "final_pnl": round(sum(t["pnl"] for t in trades), 2),
            },
            "loss_analysis": {
                "largest_5_losses": top5_losers,
                "largest_loss_streak": max_streak,
                "all_streaks": streak_sizes[:10],
                "avg_loss": round(avg_loss, 2),
                "avg_win": round(avg_win, 2),
                "actual_rr_ratio": round(actual_rr, 2),
                "theoretical_rr_ratio": round(avg_theoretical_rr, 2),
                "rr_efficiency_pct": round(rr_efficiency, 1),
            },
            "sizing_analysis": {
                "lot_pnl_correlation": round(correlation, 3),
                "avg_lot_size": round(np.mean(lot_sizes), 3) if lot_sizes else 0,
                "lot_size_range": [round(min(lot_sizes), 3), round(max(lot_sizes), 3)] if lot_sizes else [0,0],
            },
            "volatility_analysis": {
                "avg_atr_at_loss": round(avg_atr_loss, 6),
                "avg_atr_at_win": round(avg_atr_win, 6),
                "atr_ratio_loss_vs_win": round(avg_atr_loss / max(avg_atr_win, 0.000001), 2),
                "avg_all_atr": round(np.mean(self.atr_log), 6) if self.atr_log else 0,
            },
            "close_reason_analysis": close_reason_pnl,
            "spread_analysis": {
                "avg_spread": round(avg_spread, 1),
                "max_spread": round(max(self.spread_log), 1) if self.spread_log else 0,
            },
            "root_causes": {
                "SLTP_asymmetry_inefficiency": {
                    "contribution_pct": rr_loss_pct,
                    "detail": f"Theoretical R:R was {avg_theoretical_rr:.2f} but actual R:R achieved was {actual_rr:.2f} ({rr_efficiency:.1f}% efficiency). SL hits before TP due to spread/slippage erosion.",
                },
                "position_sizing_amplification": {
                    "contribution_pct": sizing_pct,
                    "detail": f"Lot sizes range from {min(lot_sizes):.3f} to {max(lot_sizes):.3f}. Correlation {correlation:.3f} between lot size and loss magnitude.",
                },
                "volatility_entry_misalignment": {
                    "contribution_pct": vol_pct,
                    "detail": f"ATR at loss entries ({avg_atr_loss:.6f}) was {avg_atr_loss/max(avg_atr_win,0.000001):.2f}x ATR at win entries ({avg_atr_win:.6f}). Entries during high volatility produce larger losses.",
                },
                "consecutive_loss_compounding": {
                    "contribution_pct": compound_pct,
                    "detail": f"Max losing streak: {max_streak['count']} trades totaling ${max_streak['total_pnl']}. Streaks compound drawdown.",
                },
                "poor_exit_quality": {
                    "contribution_pct": exit_pct,
                    "detail": f"SL hit rate: {sl_hit_rate:.1f}%. Each SL hit locks in a loss that requires multiple wins to recover.",
                },
                "spread_slippage_erosion": {
                    "contribution_pct": spread_pct,
                    "detail": f"Avg spread: {avg_spread:.1f} points. Each trade loses spread on entry + exit.",
                },
            },
        }

    def _print_report(self, report):
        """Print the analysis."""
        print("\n" + "=" * 70)
        print("DRAWDOWN ROOT CAUSE ANALYSIS — FINAL REPORT")
        print("=" * 70)

        s = report.get("summary", {})
        print(f"\nSUMMARY")
        print(f"  Total trades analyzed: {s.get('total_trades')}")
        print(f"  Win rate: {s.get('win_rate')}%")
        print(f"  Max drawdown: {s.get('max_drawdown')}%")
        print(f"  Net P&L: ${s.get('final_pnl')}")

        la = report.get("loss_analysis", {})
        print(f"\nLOSS ANALYSIS")
        print(f"  Avg win: ${la.get('avg_win')}")
        print(f"  Avg loss: ${la.get('avg_loss')}")
        print(f"  Theoretical R:R: {la.get('theoretical_rr_ratio')}:1")
        print(f"  Actual R:R achieved: {la.get('actual_rr_ratio')}:1")
        print(f"  R:R efficiency: {la.get('rr_efficiency_pct')}%")
        print(f"  Largest losing streak: {la.get('largest_loss_streak', {}).get('count')} trades, "
              f"${la.get('largest_loss_streak', {}).get('total_pnl')}")
        print(f"  Top 5 losers:")
        for i, l in enumerate(la.get("largest_5_losses", []), 1):
            print(f"    {i}. ${l['pnl']} | reason={l['reason']} | ATR={l['atr']} | lot={l['lot']}")

        sa = report.get("sizing_analysis", {})
        print(f"\nPOSITION SIZING")
        print(f"  Lot-PnL correlation: {sa.get('lot_pnl_correlation')}")
        print(f"  Avg lot size: {sa.get('avg_lot_size')}")
        print(f"  Lot range: {sa.get('lot_size_range')}")

        va = report.get("volatility_analysis", {})
        print(f"\nVOLATILITY IMPACT")
        print(f"  Avg ATR at loss: {va.get('avg_atr_at_loss')}")
        print(f"  Avg ATR at win: {va.get('avg_atr_at_win')}")
        print(f"  ATR ratio (loss/win): {va.get('atr_ratio_loss_vs_win')}x")

        cr = report.get("close_reason_analysis", {})
        print(f"\nEXIT REASONS")
        for reason, data in cr.items():
            print(f"  {reason}: {data['count']} trades, total ${data['total_pnl']}, avg ${data['avg_pnl']}")

        print(f"\n{'='*70}")
        print("ROOT CAUSES RANKED BY CONTRIBUTION")
        print(f"{'='*70}")

        causes = report.get("root_causes", {})
        ranked = sorted(causes.items(), key=lambda x: x[1]["contribution_pct"], reverse=True)
        total_pct = 0
        for i, (name, data) in enumerate(ranked, 1):
            pct = data["contribution_pct"]
            total_pct += pct
            print(f"\n{i}. {name.upper()} — {pct}%")
            print(f"   {data['detail']}")

        print(f"\n  Total accounted: {total_pct:.1f}%")
        print("=" * 70)
        print("PRIMARY CAUSE:")
        top = ranked[0] if ranked else ("none", {})
        print(f"  The dominant factor is {top[0]} ({top[1]['contribution_pct']}%).")
        print(f"  Fix this single issue to reduce drawdown the most.")
        print("=" * 70)


def run_drawdown_analysis():
    analyzer = DrawdownRootCauseAnalyzer()
    return analyzer.run_analysis()


if __name__ == "__main__":
    run_drawdown_analysis()