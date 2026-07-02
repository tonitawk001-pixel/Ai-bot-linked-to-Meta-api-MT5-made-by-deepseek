"""
Trade Logger — Structured logging for all trades during forward testing.

Each trade is logged with: timestamp, symbol, direction, entry/exit price,
SL/TP, P&L, spread, slippage, news context, and risk state.

Stored as JSON lines for easy analysis.
"""

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Optional

from trading_bot.utils.logger import logger


class TradeLogger:
    """
    Records every trade during forward testing to structured JSON files.

    Stores trades in logs/trades/ directory, rotated by month.
    """

    def __init__(self, log_dir: str = "logs/trades"):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self._current_file = None
        self._buffer = []
        logger.info(f"TradeLogger initialized: {self.log_dir}")

    def _get_file(self) -> Path:
        """Get the current log file based on month."""
        now = datetime.now()
        name = f"trades_{now.year}_{now.month:02d}.jsonl"
        return self.log_dir / name

    def log_trade(self, trade_data: dict):
        """Log a completed trade. Accepts any dict with trade info."""
        trade_data["logged_at"] = datetime.now().isoformat()
        self._buffer.append(trade_data)
        filepath = self._get_file()
        try:
            with open(filepath, "a", encoding="utf-8") as f:
                f.write(json.dumps(trade_data) + "\n")
            logger.debug(f"Trade logged: {trade_data.get('symbol','?')} "
                         f"{trade_data.get('action','?')} P/L={trade_data.get('pnl',0):.2f}")
        except Exception as e:
            logger.error(f"Failed to write trade log: {e}")

    def flush(self):
        """Flush buffer to disk."""
        if not self._buffer:
            return
        filepath = self._get_file()
        try:
            with open(filepath, "a", encoding="utf-8") as f:
                for entry in self._buffer:
                    f.write(json.dumps(entry) + "\n")
            self._buffer = []
        except Exception as e:
            logger.error(f"Flush failed: {e}")

    def load_trades(self, months_back: int = 1) -> list:
        """Load recent trades from log files for analysis."""
        trades = []
        now = datetime.now()
        for m in range(months_back):
            dt = datetime(now.year, now.month - m, 1) if now.month > m else datetime(now.year - 1, 12 + now.month - m, 1)
            fp = self.log_dir / f"trades_{dt.year}_{dt.month:02d}.jsonl"
            if fp.exists():
                try:
                    with open(fp, "r") as f:
                        for line in f:
                            line = line.strip()
                            if line:
                                trades.append(json.loads(line))
                except Exception as e:
                    logger.warning(f"Failed to load {fp}: {e}")
        return trades

    def get_stats(self, trades: Optional[list] = None) -> dict:
        """Compute basic stats from a list of trades."""
        if trades is None:
            trades = self.load_trades()
        if not trades:
            return {"total": 0}
        wins = [t for t in trades if t.get("pnl", 0) > 0]
        losses = [t for t in trades if t.get("pnl", 0) < 0]
        total = len(trades)
        return {
            "total": total,
            "wins": len(wins),
            "losses": len(losses),
            "avg_win": round(sum(t["pnl"] for t in wins) / max(len(wins), 1), 2),
            "avg_loss": round(abs(sum(t["pnl"] for t in losses)) / max(len(losses), 1), 2),
            "win_rate": round(len(wins) / max(total, 1) * 100, 1),
            "total_pnl": round(sum(t["pnl"] for t in trades), 2),
        }

    def get_recent(self, n: int = 50) -> list:
        """Get the N most recent trades."""
        trades = self.load_trades()
        return trades[-n:]