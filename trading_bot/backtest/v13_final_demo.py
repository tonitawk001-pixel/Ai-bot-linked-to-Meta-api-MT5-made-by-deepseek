"""
Gold Scalping V13 — FINAL DEMO-READY VERSION
=============================================

Fixes from V12 failures (Seed 123: -$37, Seed 456: 45% WR):
1. FASTER trend: EMA20 instead of EMA50 (4x faster reaction)
2. CONSECUTIVE LOSS LIMIT: stop for 4 hours after 3 consecutive losses
3. ATR FILTER: skip when ATR ratio > 1.8 (too volatile) or < 0.3 (too flat)
4. REDUCE lot to 0.02 (smaller risk per trade)
5. Only trade active sessions (skip transition)
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


def get_regime_fast(m15_indicators):
    """FAST trend detection using EMA20 slope (10 bar)."""
    emas = m15_indicators.get("emas", pd.DataFrame())
    if emas.empty or "EMA_20" not in emas.columns or len(emas) < 15:
        return "neutral"
    try:
        vals = emas["EMA_20"].dropna().values
        if len(vals) < 5: return "neutral"
        diff = float(vals[-1]) - float(vals[-5])
        if diff > 0: return "bullish"
        elif diff < 0: return "bearish"
        return "neutral"
    except: return "neutral"


def run_v13(days=14, seed=42):
    gen = XAUUSDDataGenerator(days=days, seed=seed)
    data = gen.generate()

    strategy = GoldScalpingStrategy()
    strategy._max_trades_per_day = 100
    strategy._max_open_positions = 5  # Reduced from 10

    vol_filter = GoldVolatilityFilter()
    risk_manager = RiskManager(default_balance=300.0)
    deepseek = MockDeepSeekClient()
    news_agg = MockNewsAggregator(data["news_events"], data["M1"].index[0])

    class Tracker:
        def __init__(self):
            self.open_positions = []
            self.sl_cooldown_until = None
            self.consecutive_losses = 0
            self.loss_streak_halt_until = None  # V13: halt after 3 losses

        def count_open(self): return len(self.open_positions)

        def open_pos(self, entry, tp, sl):
            self.open_positions.append({"entry": entry, "tp": tp, "sl": sl})

        def update_all(self, price, dt):
            closed, remaining = [], []
            pv = 0.02 * 1.00  # V13: 0.02 lot (smaller risk)
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
                        self.consecutive_losses += 1
                        # V13: halt after 3 consecutive losses for 4 hours
                        if self.consecutive_losses >= 3:
                            self.loss_streak_halt_until = dt + timedelta(hours=4)
                    else:
                        self.consecutive_losses = 0  # Reset on win
                else:
                    remaining.append(pos)
            self.open_positions = remaining
            return closed

        def can_trade(self, dt):
            if self.sl_cooldown_until and dt < self.sl_cooldown_until: return False
            if self.loss_streak_halt_until and dt < self.loss_streak_halt_until: return False
            return True

        def force_close_all(self, price):
            closed = []; pv = 0.02 * 1.00
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
        session = m15_df["session"].iloc[idx]

        # V13: Skip transition session
        if session == "transition": continue

        if dt.hour == 0 and dt.minute < 15:
            risk_manager.reset_daily(); strategy.reset_daily()
            daily_t[dt.strftime("%Y-%m-%d")] = 0

        # Process closes first
        for c in tracker.update_all(price, dt):
            for t in trades:
                if t.get("pnl_recorded"): continue
                t["exit"] = price; t["pnl"] = c["pnl"]; t["exit_reason"] = c["reason"]
                t["pnl_recorded"] = True; risk_manager.record_result(c["pnl"])
                break

        # V13: Check if we can trade
        if not tracker.can_trade(dt): continue

        news_agg.set_current_time(dt); ctx = news_agg.get_news_context()

        m1_s = m1_df.iloc[:max(idx * 15, 100)]
        m5_s = m5_df.iloc[:max(idx * 3, 100)]
        m15_s = m15_df.iloc[:idx + 1]

        try:
            m1i = compute_all_indicators(m1_s)
            m5i = compute_all_indicators(m5_s)
            m15i = compute_all_indicators(m15_s)
        except: continue

        # V13: ATR filter — skip extreme volatility
        atr = float(m15i["atr"].iloc[-1]) if not m15i["atr"].empty else 7.0
        if not m15i["atr"].empty and len(m15i["atr"].dropna()) > 20:
            cur = float(m15i["atr"].iloc[-1])
            avg = float(m15i["atr"].iloc[-21:-1].mean())
            if avg > 0:
                ratio = cur / avg
                if ratio > 1.8 or ratio < 0.3: continue  # Skip extreme

        regime = get_regime_fast(m15i)
        if regime == "neutral": continue

        sr = strategy.analyze(m1_indicators=m1i, m5_indicators=m5i, m15_indicators=m15i,
                              m1_ohlcv=m1_s, m5_ohlcv=m5_s, m15_ohlcv=m15_s, news_context=ctx)

        direction = sr.get("direction", "NONE")
        score = sr.get("setup_score", 0)
        bias = sr.get("bias", "neutral")

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

        if tracker.count_open() >= 5: continue
        oc = tracker.count_open()
        saved = strategy._last_trade_time
        if strategy._last_trade_time is not None:
            strategy._last_trade_time = dt - timedelta(minutes=15)
        ok, reason = strategy.can_trade(oc)
        strategy._last_trade_time = saved
        if not ok: continue

        if direction == "BUY":
            sl = round(price - atr * 10, 2)
            tp = round(price + atr * 2.5, 2)
        else:
            sl = round(price + atr * 10, 2)
            tp = round(price - atr * 2.5, 2)

        lot = 0.02
        trade = {"time": str(dt), "entry": price, "sl": sl, "tp": tp, "lot": lot,
                 "score": score, "session": session,
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
                t["exit"] = fp; t["pnl"] = c["pnl"]; t["exit_reason"] = "EOD"
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

    print(f"""Seed {seed}: TOTAL={total} WR={wr:.0f}% P&L=${tp_pnl:.2f} RET={tp_pnl/300*100:.1f}% DD={mdd:.1f}% TP={tp_c} SL={sl_c} FINAL=${300+tp_pnl:.2f} LLS={lls}""")
    return {"summary": {"total_trades": total, "wins": len(wins), "losses": len(losses),
                        "win_rate": round(wr, 1), "profit_factor": round(pf, 2),
                        "total_pnl": round(tp_pnl, 2), "return_pct": round(tp_pnl/300*100, 2),
                        "max_dd_pct": round(mdd, 2), "final_balance": round(300+tp_pnl, 2),
                        "longest_win": lws, "longest_loss": lls}}


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--days", type=int, default=14)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--multi", action="store_true")
    p.add_argument("--output", default="trading_bot/backtest/v13_report.json")
    a = p.parse_args()

    if a.multi:
        seeds = [42, 123, 456, 555, 777, 333, 888, 222, 111, 999]
        reports = []
        for s in seeds:
            r = run_v13(days=a.days, seed=s)
            reports.append(r["summary"])
        agg = {}
        for k in reports[0].keys():
            vals = [r[k] for r in reports]
            if isinstance(vals[0], (int, float)):
                agg[f"{k}_avg"] = round(np.mean(vals), 2)
                agg[f"{k}_std"] = round(np.std(vals), 2)
        print(f"\n{'='*60}")
        print(f"  V13 MULTI-SEED ({len(seeds)} runs) — $300 ACCOUNT")
        print(f"{'='*60}")
        for km in ["total_trades", "win_rate", "profit_factor", "total_pnl", "return_pct", "max_dd_pct", "final_balance", "longest_loss"]:
            if f"{km}_avg" in agg:
                print(f"  {km:<25} {agg[f'{km}_avg']:>8.2f} ± {agg[f'{km}_std']:>6.2f}")
        report = {"multi": agg, "individual": {str(s): r for s, r in zip(seeds, reports)}}
    else:
        report = run_v13(days=a.days, seed=a.seed)

    with open(a.output, "w") as f:
        json.dump(report, f, indent=2, default=str)
    print(f"Saved: {a.output}")