"""
Gold Scalping V12 — ADAPTIVE (BUY in bullish, SELL in bearish)
==============================================================

Seed 999 revealed: when market turns bearish, BUY-only strategy loses money.
Fix: detect regime from M15 EMA50 slope, trade WITH the trend.

Rules:
  - EMA50 rising = bullish → BUY only (score 80-89)
  - EMA50 falling = bearish → SELL only (score 80-89)
  - No trend → skip
  - SL cooldown: 45 min after any loss
  - SL: 10x ATR | TP: 2.5x ATR
  - Lot: 0.03 ($300 acct)
"""

import json, sys, os
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
from trading_bot.backtest.gold_backtest import XAUUSDDataGenerator, MockDeepSeekClient, MockNewsAggregator


def get_regime(m15_indicators):
    """Detect bullish/bearish from M15 EMA50 slope."""
    emas = m15_indicators.get("emas", pd.DataFrame())
    if emas.empty or "EMA_50" not in emas.columns or len(emas) < 20:
        return "neutral"
    try:
        vals = emas["EMA_50"].dropna().values
        if len(vals) < 10: return "neutral"
        diff = float(vals[-1]) - float(vals[-10])
        if diff > 0: return "bullish"
        elif diff < 0: return "bearish"
        return "neutral"
    except: return "neutral"


def run_v12(days=60, seed=42):
    gen = XAUUSDDataGenerator(days=days, seed=seed)
    data = gen.generate()

    strategy = GoldScalpingStrategy()
    strategy._max_trades_per_day = 100
    strategy._max_open_positions = 10

    vol_filter = GoldVolatilityFilter()
    risk_manager = RiskManager(default_balance=300.0)
    deepseek = MockDeepSeekClient()
    news_agg = MockNewsAggregator(data["news_events"], data["M1"].index[0])

    class Tracker:
        def __init__(self):
            self.open_positions = []
            self.sl_cooldown_until = None

        def count_open(self): return len(self.open_positions)

        def open_pos(self, entry, tp, sl):
            self.open_positions.append({"entry": entry, "tp": tp, "sl": sl})

        def update_all(self, price, dt):
            closed, remaining = [], []
            pv = 0.03 * 1.00
            for pos in self.open_positions:
                hit = False; pnl = 0.0; reason = ""
                if pos["tp"]:
                    if (pos["tp"] >= pos["entry"] and price >= pos["tp"]) or \
                       (pos["tp"] < pos["entry"] and price <= pos["tp"]):
                        pnl = (pos["tp"] - pos["entry"]) * pv; reason = "TP"; hit = True
                if pos["sl"] and not hit:
                    if (pos["sl"] < pos["entry"] and price <= pos["sl"]) or \
                       (pos["sl"] > pos["entry"] and price >= pos["sl"]):
                        pnl = (pos["sl"] - pos["entry"]) * pv; reason = "SL"; hit = True
                if hit:
                    closed.append({**pos, "pnl": round(pnl, 2), "reason": reason})
                    if reason == "SL":
                        self.sl_cooldown_until = dt + timedelta(minutes=45)
                else:
                    remaining.append(pos)
            self.open_positions = remaining
            return closed

        def is_on_cooldown(self, dt):
            if self.sl_cooldown_until is None: return False
            if dt >= self.sl_cooldown_until: self.sl_cooldown_until = None; return False
            return True

        def force_close_all(self, price):
            closed = []; pv = 0.03 * 1.00
            for pos in self.open_positions:
                pnl = (price - pos["entry"]) * pv
                closed.append({**pos, "pnl": round(pnl, 2), "reason": "EOD"})
            self.open_positions = []
            return closed

    tracker = Tracker()
    from trading_bot.indicators.technical_indicators import compute_all_indicators
    m1_df, m5_df, m15_df = data["M1"].copy(), data["M5"].copy(), data["M15"].copy()

    trades = []; daily_t = defaultdict(int)

    for idx in range(30, len(m15_df)):
        dt = m15_df.index[idx]
        price = float(m15_df["close"].iloc[idx])

        if dt.hour == 0 and dt.minute < 15:
            risk_manager.reset_daily(); strategy.reset_daily()
            daily_t[dt.strftime("%Y-%m-%d")] = 0

        if tracker.is_on_cooldown(dt):
            for c in tracker.update_all(price, dt):
                for t in trades:
                    if t.get("pnl_recorded"): continue
                    t["exit"] = price; t["pnl"] = c["pnl"]; t["exit_reason"] = c["reason"]
                    t["pnl_recorded"] = True; risk_manager.record_result(c["pnl"])
                    break
            continue

        for c in tracker.update_all(price, dt):
            for t in trades:
                if t.get("pnl_recorded"): continue
                t["exit"] = price; t["pnl"] = c["pnl"]; t["exit_reason"] = c["reason"]
                t["pnl_recorded"] = True; risk_manager.record_result(c["pnl"])
                break

        news_agg.set_current_time(dt); ctx = news_agg.get_news_context()

        m1_s = m1_df.iloc[:max(idx * 15, 100)]
        m5_s = m5_df.iloc[:max(idx * 3, 100)]
        m15_s = m15_df.iloc[:idx + 1]

        try:
            m1i = compute_all_indicators(m1_s)
            m5i = compute_all_indicators(m5_s)
            m15i = compute_all_indicators(m15_s)
        except: continue

        regime = get_regime(m15i)
        if regime == "neutral": continue

        sr = strategy.analyze(m1_indicators=m1i, m5_indicators=m5i, m15_indicators=m15i,
                              m1_ohlcv=m1_s, m5_ohlcv=m5_s, m15_ohlcv=m15_s, news_context=ctx)

        direction = sr.get("direction", "NONE")
        score = sr.get("setup_score", 0)
        bias = sr.get("bias", "neutral")

        # V12: trade WITH regime
        if regime == "bullish":
            if direction != "BUY": continue
        elif regime == "bearish":
            if direction != "SELL": continue

        if score < 80 or score >= 90: continue
        if regime == "bullish" and bias != "bullish": continue
        if regime == "bearish" and bias != "bearish": continue

        try:
            vfr = vol_filter.analyze(m1_ohlcv=m1_s, m5_ohlcv=m5_s, m15_ohlcv=m15_s,
                                     m1_indicators=m1i, m5_indicators=m5i, m15_indicators=m15i)
            if not vfr.get("trade_ok", True): continue
        except: pass

        if tracker.count_open() >= 10: continue
        oc = tracker.count_open()
        saved = strategy._last_trade_time
        if strategy._last_trade_time is not None:
            strategy._last_trade_time = dt - timedelta(minutes=15)
        ok, reason = strategy.can_trade(oc)
        strategy._last_trade_time = saved
        if not ok: continue

        atr = float(m15i["atr"].iloc[-1]) if not m15i["atr"].empty else 7.0
        if direction == "BUY":
            sl = round(price - atr * 10, 2)
            tp = round(price + atr * 2.5, 2)
        else:
            sl = round(price + atr * 10, 2)
            tp = round(price - atr * 2.5, 2)

        lot = 0.03
        trade = {"time": str(dt), "entry": price, "sl": sl, "tp": tp, "lot": lot,
                 "score": score, "session": m15_df["session"].iloc[idx],
                 "regime": regime, "direction": direction,
                 "pnl": 0.0, "pnl_recorded": False, "exit_reason": "?", "exit": None}
        trades.append(trade)
        tracker.open_pos(price, tp, sl)
        strategy.record_trade()
        daily_t[dt.strftime("%Y-%m-%d")] += 1

    fp = float(m15_df["close"].iloc[-1])
    for c in tracker.force_close_all(fp):
        for t in trades:
            if not t.get("pnl_recorded"):
                t["exit"] = fp; t["pnl"] = c["pnl"]; t["exit_reason"] = c["reason"]
                t["pnl_recorded"] = True; risk_manager.record_result(c["pnl"])
                break

    total = len(trades)
    wins = [t for t in trades if t.get("pnl", 0) > 0]
    losses = [t for t in trades if t.get("pnl", 0) < 0]
    even = [t for t in trades if t.get("pnl", 0) == 0]
    wr = (len(wins) / total * 100) if total > 0 else 0
    gp = sum(t.get("pnl", 0) for t in wins)
    gl = abs(sum(t.get("pnl", 0) for t in losses))
    pf = gp / gl if gl > 0 else float("inf")
    tp_pnl = sum(t.get("pnl", 0) for t in trades)

    bal = 300.0; peak = bal; mdd = 0.0; mdd_t = bal
    for t in trades:
        bal += t.get("pnl", 0)
        if bal > peak: peak = bal
        dd = (peak - bal) / peak * 100
        if dd > mdd: mdd = dd; mdd_t = bal

    lws = lls = cw = cl = 0
    for t in trades:
        p = t.get("pnl", 0)
        if p > 0: cw += 1; cl = 0; lws = max(lws, cw)
        elif p < 0: cl += 1; cw = 0; lls = max(lls, cl)
        else: cw = cl = 0

    tp_c = len([t for t in trades if t.get("exit_reason") == "TP"])
    sl_c = len([t for t in trades if t.get("exit_reason") == "SL"])

    print(f"""
{'='*75}
  V12 — ADAPTIVE REGIME (Seed {seed}) — 60 DAYS $300 ACCOUNT
{'='*75}

  {regime.upper()} market (EMA50 slope)
  LOT: 0.03 | TP: 2.5x ATR | SL: 10x ATR | SL COOLDOWN: 45min

  Total: {total:>4} | TP={tp_c} | SL={sl_c} | EOD={total-tp_c-sl_c}
  Win Rate: {wr:.1f}%
  P&L: ${tp_pnl:.2f}
  Return: {tp_pnl/300*100:.2f}%
  Max DD: {mdd:.2f}%
  Final: ${300+tp_pnl:.2f}
""")
    for d in sorted(daily_t.keys()):
        cnt = daily_t[d]
        dpnl = sum(t.get("pnl", 0) for t in trades if t["time"].startswith(d))
        if cnt > 0:
            print(f"  {d:<20} {cnt:>3} t | ${dpnl:>8.2f}")

    print(f"\n{'='*75}\n")
    return {"summary": {"total_trades": total, "wins": len(wins), "losses": len(losses),
                        "win_rate": round(wr, 1), "profit_factor": round(pf, 2),
                        "total_pnl": round(tp_pnl, 2), "return_pct": round(tp_pnl/300*100, 2),
                        "max_dd_pct": round(mdd, 2), "final_balance": round(300+tp_pnl, 2),
                        "longest_win": lws, "longest_loss": lls}}


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--days", type=int, default=60)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output", default="trading_bot/backtest/v12_report.json")
    a = p.parse_args()

    print(f"\n--- SEED {a.seed} ---")
    report = run_v12(days=a.days, seed=a.seed)

    with open(a.output, "w") as f:
        json.dump(report, f, indent=2, default=str)
    print(f"Saved: {a.output}")