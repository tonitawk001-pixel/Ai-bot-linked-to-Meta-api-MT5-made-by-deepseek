"""
Performance Tracker — Rolling evaluation reports during forward testing.
Generates reports every 50 trades and 100 trades. Detects drift.
"""

from datetime import datetime


class PerformanceTracker:
    def __init__(self, metrics, backtest_benchmark=None):
        self.metrics = metrics
        self.backtest = backtest_benchmark or {}
        self._last_report_at = 0
        self._reports = []

    def check(self, force=False):
        t = self.metrics.total_trades
        if not force and t - self._last_report_at < 50:
            return None
        if t == self._last_report_at:
            return None
        self._last_report_at = t
        return self._generate("periodic")

    def check_stability(self, force=False):
        t = self.metrics.total_trades
        if t < 100 and not force:
            return None
        if t % 100 != 0 and not force:
            return None
        return self._generate("stability")

    def _generate(self, rtype):
        s = self.metrics.snapshot()
        r = {"timestamp": datetime.now().isoformat(), "type": rtype,
             "total_trades": s["total_trades"],
             "rolling_expectancy": s["rolling_expectancy"],
             "rolling_win_rate": s["rolling_win_rate"],
             "rolling_profit_factor": s["rolling_profit_factor"],
             "max_drawdown": s["max_drawdown_pct"],
             "consecutive_losses": s["consecutive_losses"],
             "data_drift": self._detect_drift(s)}
        if rtype == "stability":
            r["strategy_state"] = self._classify_state(s)
            r["recommendation"] = self._recommend(r)
        self._reports.append(r)
        return r

    def _detect_drift(self, s):
        if not self.backtest:
            return False
        bt_e = self.backtest.get("expectancy", 0)
        lv_e = s.get("rolling_expectancy", 0)
        if bt_e > 0 and lv_e < bt_e * 0.5:
            return True
        bt_d = self.backtest.get("max_drawdown", 20)
        lv_d = s.get("max_drawdown_pct", 0)
        if lv_d > bt_d * 1.5:
            return True
        return False

    def _classify_state(self, s):
        dd = s["max_drawdown_pct"]
        wr = s["rolling_win_rate"]
        pf = s["rolling_profit_factor"]
        if dd > 20 or wr < 25 or pf < 0.8:
            return "failing"
        elif dd > 10 or wr < 35 or pf < 1.0:
            return "unstable"
        return "stable"

    def _recommend(self, r):
        if r.get("strategy_state") == "failing" or r.get("data_drift"):
            return "stop_trading"
        elif r.get("strategy_state") == "unstable":
            return "adjust_strategy"
        return "continue_demo"

    def latest_report(self):
        return self._reports[-1] if self._reports else None