"""
Evaluation Engine — Final dashboard for forward test validation.

Aggregates all live metrics, performance reports, news analysis,
and risk control status into a single structured output.

After 2-4 weeks of demo trading, this answers:
1. Is strategy still profitable in live conditions?
2. Is drawdown acceptable (<20%)?
3. Is there overfitting compared to backtest?
4. Does news affect performance?
"""

from datetime import datetime, timedelta

from trading_bot.utils.logger import logger
from trading_bot.forward_test.live_metrics import LiveMetrics
from trading_bot.forward_test.performance_tracker import PerformanceTracker
from trading_bot.forward_test.trade_logger import TradeLogger


class EvaluationEngine:
    """
    Aggregates all forward test data into a single structured dashboard.
    Designed to run after every trade or on demand.
    """

    def __init__(self, metrics: LiveMetrics, tracker: PerformanceTracker,
                 trade_logger: TradeLogger, news_agg=None):
        self.metrics = metrics
        self.tracker = tracker
        self.trade_logger = trade_logger
        self.news_agg = news_agg
        self._start_time = datetime.now()
        logger.info("EvaluationEngine initialized")

    def generate_dashboard(self, force_report: bool = False) -> dict:
        """Generate the complete forward test dashboard."""
        # Run periodic checks
        self.tracker.check(force=force_report)
        self.tracker.check_stability(force=force_report)

        snap = self.metrics.snapshot()
        latest_report = self.tracker.latest_report() or {}
        trades_from_log = self.trade_logger.get_recent(50)

        # Compare backtest vs live
        bt_vs_live = {}
        if self.tracker.backtest:
            bt = self.tracker.backtest
            bt_vs_live = {
                "backtest_expectancy": bt.get("expectancy", "N/A"),
                "live_expectancy": snap["rolling_expectancy"],
                "backtest_win_rate": bt.get("win_rate", "N/A"),
                "live_win_rate": snap["rolling_win_rate"],
                "backtest_profit_factor": bt.get("profit_factor", "N/A"),
                "live_profit_factor": snap["rolling_profit_factor"],
                "backtest_max_drawdown": bt.get("max_drawdown", "N/A"),
                "live_max_drawdown": snap["max_drawdown_pct"],
            }

        # Slippage analysis
        slippage_analysis = {}
        if trades_from_log:
            bt_slippage = self.tracker.backtest.get("slippage_estimate", 0.00005)
            live_slippages = [abs(t.get("slippage", 0)) for t in trades_from_log if t.get("slippage")]
            slippage_analysis = {
                "backtest_slippage_estimate": bt_slippage,
                "live_avg_slippage": round(sum(live_slippages) / max(len(live_slippages), 1), 6) if live_slippages else 0,
                "slippage_impact": "higher" if (live_slippages and np.mean(live_slippages) > bt_slippage * 2) else "within_expected",
            }

        # News analysis
        news_analysis = {}
        if self.news_agg:
            news_context = self.news_agg.get_news_context()
            news_analysis = {
                "global_risk_mode": news_context.get("global_risk_mode", "unknown"),
                "news_period_trades": snap.get("news_vs_normal", {}).get("news", {}).get("trades", 0),
                "normal_period_trades": snap.get("news_vs_normal", {}).get("normal", {}).get("trades", 0),
                "news_impact": self._evaluate_news_impact(),
            }

        # Duration
        duration = datetime.now() - self._start_time
        duration_hours = duration.total_seconds() / 3600

        # Final state
        strategy_state = latest_report.get("strategy_state", "unknown")
        recommendation = latest_report.get("recommendation", "unknown")
        data_drift = latest_report.get("data_drift", False)

        # Check risk control triggers
        risk_triggers = self._check_risk_triggers(snap)

        dashboard = {
            "dashboard_timestamp": datetime.now().isoformat(),
            "forward_test_duration_hours": round(duration_hours, 1),
            "backtest_vs_live": bt_vs_live,
            "live_metrics": {
                "total_trades": snap["total_trades"],
                "rolling_expectancy": snap["rolling_expectancy"],
                "rolling_win_rate": snap["rolling_win_rate"],
                "rolling_profit_factor": snap["rolling_profit_factor"],
                "rolling_avg_win": snap["rolling_avg_win"],
                "rolling_avg_loss": snap["rolling_avg_loss"],
                "balance": snap["balance"],
                "equity": snap.get("equity", 0),
            },
            "risk_status": {
                "max_drawdown_pct": snap["max_drawdown_pct"],
                "current_drawdown_pct": snap["current_drawdown_pct"],
                "consecutive_losses": snap["consecutive_losses"],
                "risk_triggers": risk_triggers,
            },
            "slippage_analysis": slippage_analysis,
            "news_analysis": news_analysis,
            "strategy_state": strategy_state,
            "data_drift_detected": data_drift,
            "recommendation": recommendation,
        }

        return dashboard

    def _evaluate_news_impact(self) -> str:
        """Evaluate whether news affects performance."""
        nv = self.metrics.news_vs_normal
        news = nv.get("news", {})
        normal = nv.get("normal", {})
        if news.get("trades", 0) < 3:
            return "insufficient_data"
        n_avg = news.get("avg_trade", 0)
        no_avg = normal.get("avg_trade", 0)
        if no_avg > 0 and n_avg < no_avg * 0.5:
            return "negative_impact"
        elif n_avg > no_avg * 1.2:
            return "positive_impact"
        return "neutral"

    def _check_risk_triggers(self, snap: dict) -> list:
        """Check if any risk control conditions are active."""
        triggers = []
        dd = snap["max_drawdown_pct"]
        cl = snap["consecutive_losses"]
        if dd > 15:
            triggers.append("drawdown_exceeded_15pct_lot_scale_50pct")
        if dd > 25:
            triggers.append("drawdown_exceeded_25pct_trading_paused_24h")
        if cl >= 5:
            triggers.append("consecutive_losses_5_reduce_lot_75pct")
        return triggers

    def summary_text(self, dashboard: dict) -> str:
        """Generate a human-readable summary of the dashboard."""
        lines = []
        lines.append(f"Forward Test: {dashboard['forward_test_duration_hours']:.0f}h")
        lines.append(f"Trades: {dashboard['live_metrics']['total_trades']}")
        lines.append(f"Expectancy: ${dashboard['live_metrics']['rolling_expectancy']}/trade")
        lines.append(f"Win Rate: {dashboard['live_metrics']['rolling_win_rate']}%")
        lines.append(f"Profit Factor: {dashboard['live_metrics']['rolling_profit_factor']}")
        lines.append(f"Max Drawdown: {dashboard['risk_status']['max_drawdown_pct']}%")
        lines.append(f"State: {dashboard['strategy_state'].upper()}")
        lines.append(f"Drift: {'YES' if dashboard['data_drift_detected'] else 'NO'}")
        lines.append(f"Recommendation: {dashboard['recommendation']}")
        if dashboard['risk_status']['risk_triggers']:
            lines.append(f"Risk triggers active: {len(dashboard['risk_status']['risk_triggers'])}")
        return " | ".join(lines)


def create_evaluation_engine(metrics: LiveMetrics, backtest_benchmark: dict = None,
                              news_agg=None) -> EvaluationEngine:
    """Factory function to create a fully wired evaluation engine."""
    logger = TradeLogger()
    tracker = PerformanceTracker(metrics, backtest_benchmark=backtest_benchmark)
    engine = EvaluationEngine(metrics, tracker, logger, news_agg=news_agg)
    return engine