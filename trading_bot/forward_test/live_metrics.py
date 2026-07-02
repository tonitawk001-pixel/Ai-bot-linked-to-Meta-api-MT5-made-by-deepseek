"""Live Metrics - Tracks real-time performance during forward testing."""

import numpy as np
from collections import deque


class LiveMetrics:
    def __init__(self, initial_balance=10000.0, rolling_window=50):
        self.initial_balance = initial_balance
        self.balance = initial_balance
        self.equity = initial_balance
        self.equity_peak = initial_balance
        self.equity_low = initial_balance
        self.window_size = rolling_window
        self._pnl_window = deque(maxlen=rolling_window)
        self.total_trades = 0
        self.consecutive_losses = 0
        self.max_consecutive_losses = 0
        self.total_pnl = 0.0
        self._news_trades = []
        self._normal_trades = []

    def record_trade(self, trade):
        pnl = trade.get("pnl", 0.0)
        self.total_trades += 1
        self.total_pnl += pnl
        self.balance += pnl
        self._pnl_window.append(pnl)
        if pnl < 0:
            self.consecutive_losses += 1
            self.max_consecutive_losses = max(self.max_consecutive_losses, self.consecutive_losses)
        else:
            self.consecutive_losses = 0
        nm = trade.get("news_risk_mode", "low")
        if nm in ("high", "news_blackout"):
            self._news_trades.append(trade)
        else:
            self._normal_trades.append(trade)

    def update_equity(self, equity):
        self.equity = equity
        self.equity_peak = max(self.equity_peak, equity)
        self.equity_low = min(self.equity_low, equity)

    @property
    def rolling_expectancy(self):
        return round(float(np.mean(self._pnl_window)), 2) if self._pnl_window else 0.0

    @property
    def rolling_win_rate(self):
        if not self._pnl_window: return 0.0
        return round(sum(1 for p in self._pnl_window if p > 0) / len(self._pnl_window) * 100, 1)

    @property
    def rolling_profit_factor(self):
        if not self._pnl_window: return 0.0
        gp = sum(p for p in self._pnl_window if p > 0)
        gl = abs(sum(p for p in self._pnl_window if p < 0))
        return round(gp / max(gl, 0.01), 2)

    @property
    def max_drawdown_pct(self):
        return round((self.equity_peak - self.equity_low) / max(self.equity_peak, 1) * 100, 2)

    def snapshot(self):
        return {"total_trades": self.total_trades, "rolling_expectancy": self.rolling_expectancy,
                "rolling_win_rate": self.rolling_win_rate, "rolling_profit_factor": self.rolling_profit_factor,
                "max_drawdown_pct": self.max_drawdown_pct, "consecutive_losses": self.consecutive_losses,
                "total_pnl": round(self.total_pnl, 2), "balance": round(self.balance, 2)}