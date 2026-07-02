"""
Gold Scalping V11 — SL Cooldown + 2-Week Test on $300 Account
=============================================================

V10 had a flaw: multiple positions hit SL in a row during market dips.
V11 adds:
  - SL cooldown: after SL hit, skip 3 M15 candles (45 min)
  - 14 day backtest (2 weeks)
  - $300 account scaling: lot = 0.03 (1/33 of $10k size)
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


def run_v11(days=14, seed=42):
    gen = XAUUSDDataGenerator(days=days, seed=seed)
    data = gen.generate()

    strategy = GoldScalpingStrategy()
    strategy._max_trades_per_day = 100
    strategy._max_open_positions = 10

    vol_filter = GoldVolatilityFilter()
    risk_manager = RiskManager(default_balance=300.0)  # $300 account
    deepseek = MockDeepSeekClient()
    news_agg = MockNewsAggregator(data["news_events"], data["M1"].index[0])

    class M15Tracker:
        def __init__(self):
            self.open_positions = []
            self.sl_cooldown_until = None  # V11: cooldown after SL

        def count_open(self):
            return len(self.open_positions)

        def open_pos(self, entry, tp, sl):
            self.open_positions.append({"entry": entry, "tp": tp, "sl": sl})

        def update_all(self, current_price, current_dt):
            closed, remaining = [], []
            pv = 0.03 * 1.00  # V11: 0.03 lot for $300 account
            for pos in self.open_positions:
                hit = False; pnl = 0.0; reason = ""
                if pos["tp"] and current_price >= pos["tp"]:
                    pnl = (pos["tp"] - pos["entry"]) * pv; reason = "TP"; hit = True
                elif pos["sl"] and current_price <= pos["sl"]:
                    pnl = (pos["sl"] - pos["entry"]) * pv; reason = "SL"; hit = True
                if hit:
                    closed.append({**pos, "pnl": round(pnl, 2), "reason": reason})
                    # V11: set cooldown after SL
                    if reason == "SL":
                        self.sl_cooldown_until = current_dt + timedelta(minutes=45)
                else:
                    remaining.append(pos)
            self.open_positions = remaining
            return closed

        def is_on_cooldown(self, current_dt):
            """V11: check if we're in SL cooldown period."""
            if self.sl_cooldown_until is None:
                return False
            if current_dt >= self.sl_cooldown_until:
                self.sl_cooldown_until = None
                return False
            return True

        def force_close_all(self, current_price):
            closed = []; pv = 0.03 * 1.00
            for pos in self.open_positions:
                pnl = (current_price - pos["entry"]) * pv
                closed.append({**pos, "pnl": round(pnl, 2), "reason": "EOD"})
            self.open_positions = []
            return closed

    tracker = M15Tracker()

    from trading_bot.indicators.technical_indicators import compute_all_indicators
    m1_df, m5_df, m15_df = data["M1"].copy(), data["M5"].copy(), data["M15"].copy()

    trades = []
    daily_t = defaultdict(int)

    warmup = 30
    for idx in range(warmup, len(m15_df)):
        dt = m15_df.index[idx]
        price = float(m15_df["close"].iloc[idx])

        if dt.hour == 0 and dt.minute < 15:
            risk_manager.reset_daily(); strategy.reset_daily()
            daily_t[dt.strftime("%Y-%m-%d")] = 0

        # V11: Check SL cooldown
        if tracker.is_on_cooldown(dt):
            # Still close positions but don't enter new ones
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

        news_agg.set_current_time(dt)
        ctx = news_agg.get_news_context()

        m1_slice = m1_df.iloc[:max(idx * 15, 100)]
        m5_slice = m5_df.iloc[:max(idx * 3, 100)]
        m15_slice = m15_df.iloc[:idx + 1]

        try:
            m1i = compute_all_indicators(m1_slice)
            m5i = compute_all_indicators(m5_slice)
            m15i = compute_all_indicators(m15_slice)
        except Exception: continue

        sr = strategy.analyze(m1_indicators=m1i, m5_indicators=m5i, m15_indicators=m15i,
                              m1_ohlcv=m1_slice, m5_ohlcv=m5_slice, m15_ohlcv=m15_slice, news_context=ctx)

        direction = sr.get("direction", "NONE")
        score = sr.get("setup_score", 0)
        bias = sr.get("bias", "neutral")

        if direction != "BUY": continue
        if score < 80 or score >= 90: continue
        if bias != "bullish": continue

        try:
            vfr = vol_filter.analyze(m1_ohlcv=m1_slice, m5_ohlcv=m5_slice, m15_ohlcv=m15_slice,
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
        sl = round(price - atr * 10, 2)
        tp = round(price + atr * 2.5, 2)
        lot = 0.03  # V11: scaled for $300 account

        trade = {"time": str(dt), "entry": price, "sl": sl, "tp": tp, "lot": lot,
                 "score": score, "session": m15_df["session"].iloc[idx],
                 "bias": bias,
                 "reason": sr.get("reason", "")[:40],
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
    atpd = total / days

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

    tp_count = len([t for t in trades if t.get("exit_reason") == "TP"])
    sl_count = len([t for t in trades if t.get("exit_reason") == "SL"])
    eod_count = len([t for t in trades if t.get("exit_reason") == "EOD"])

    print(f"""
{'='*75}
  V11 — $300 ACCOUNT · 2-WEEK TEST (Seed {seed})
{'='*75}

  ACCT: $300 | LOT: 0.03 | M15 | TP: 2.5x ATR | SL: 10x ATR | SL COOLDOWN: 45min
  SCORE: 80-89 | BUY ONLY | BULLISH BIAS

  RESULTS
  -------
  Total Trades: {total:>4}
  WIN RATE: *** {wr:.1f}% ***
  Profit Factor: {pf:.2f}
  Total P&L: ${tp_pnl:.2f}
  Return: {tp_pnl/300*100:.2f}%
  Max DD: {mdd:.2f}%
  Avg/Day: {atpd:.1f}
  Longest Win: {lws} | Longest Loss: {lls}
  TP={tp_count} | SL={sl_count} | EOD={eod_count}
""")
    for d in sorted(daily_t.keys()):
        cnt = daily_t[d]
        dpnl = sum(t.get("pnl", 0) for t in trades if t["time"].startswith(d))
        print(f"  {d:<20} {cnt:>3} t | ${dpnl:>8.2f}")

    # Count cooldown activations
    print(f"\n  FINAL BALANCE: ${300+tp_pnl:.2f}")
    print(f"{'='*75}\n")

    return {"summary": {"total_trades": total, "wins": len(wins), "losses": len(losses),
                        "win_rate": round(wr, 1), "profit_factor": round(pf, 2),
                        "total_pnl": round(tp_pnl, 2), "return_pct": round(tp_pnl/300*100, 2),
                        "max_dd_pct": round(mdd, 2), "avg_trades_day": round(atpd, 1),
                        "final_balance": round(300+tp_pnl, 2),
                        "longest_win": lws, "longest_loss": lls}}


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--days", type=int, default=14)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--multi-seed", action="store_true")
    p.add_argument("--output", default="trading_bot/backtest/v11_report.json")
    a = p.parse_args()

    if a.multi_seed:
        seeds = [42, 123, 456, 789, 1111]
        reports = []
        for s in seeds:
            print(f"\n--- SEED {s} ---")
            r = run_v11(days=a.days, seed=s)
            reports.append(r["summary"])
        agg = {}
        for k in reports[0].keys():
            vals = [r[k] for r in reports]
            if isinstance(vals[0], (int, float)):
                agg[f"{k}_avg"] = round(np.mean(vals), 2)
                agg[f"{k}_std"] = round(np.std(vals), 2)
        print(f"\n{'='*50}")
        print(f"  MULTI-SEED ({len(seeds)} runs) — 2 WEEKS $300 ACCOUNT")
        print(f"{'='*50}")
        for km in ["total_trades", "win_rate", "profit_factor", "total_pnl", "return_pct", "max_dd_pct", "final_balance", "avg_trades_day"]:
            if f"{km}_avg" in agg:
                print(f"  {km:<25} {agg[f'{km}_avg']:>8.2f} ± {agg[f'{km}_std']:>6.2f}")
        print(f"  {'='*50}")
        print(f"  AVG FINAL BALANCE: ${agg.get('final_balance_avg', 0):.2f} (start: $300)")
        report = {"multi": agg}
    else:
        report = run_v11(days=a.days, seed=a.seed)

    with open(a.output, "w") as f:
        json.dump(report, f, indent=2, default=str)
    print(f"Saved: {a.output}")