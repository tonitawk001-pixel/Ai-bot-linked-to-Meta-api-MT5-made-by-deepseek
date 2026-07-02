"""
V21 - SAFE COMPOUNDING Strategy
===============================
Key change from V20: Graduated risk to prevent deep drawdowns on low balance.

Risk schedule:
  Balance < $200  -> 1.0% risk (protect survival)
  Balance $200-500 -> 2.0% risk (moderate growth)
  Balance $500+    -> 3.0% risk (aggressive compound)

Safety nets:
  - Daily loss limit: -5% of CURRENT balance (dynamic)
  - 3-consecutive-loss -> 2-hour halt
  - Max 5 positions
  - Trailing SL to breakeven + trail
  - Min score 50
  - BUY requires rising EMA200 slope

Tested on TWO windows for robustness.
"""

import sys, os, time
from datetime import datetime, timedelta, timezone
from collections import defaultdict
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from trading_bot.utils.logger import logger
from trading_bot.strategy.gold_scalping_strategy import GoldScalpingStrategy
from trading_bot.strategy.gold_volatility_filter import GoldVolatilityFilter
from trading_bot.indicators.technical_indicators import compute_all_indicators
from trading_bot.config import Config

import MetaTrader5 as mt5

DAYS = 180
INITIAL_BALANCE = 300.0
SYMBOL = "XAUUSD"
MT5_LOGIN = Config.MT5_LOGIN
MT5_PASSWORD = Config.MT5_PASSWORD
MT5_SERVER = Config.MT5_SERVER

MIN_SCORE = 50
MAX_POS = 5
TRAIL_SL = True
HALT_LOSSES = 3
HALT_HOURS = 2


def get_risk_percent(balance):
    """Graduated risk: 4-tier. Super-safe at low balance, aggressive when big."""
    if balance < 250:
        return 0.5
    elif balance < 500:
        return 1.5
    elif balance < 1000:
        return 2.5
    else:
        return 3.0


def get_daily_loss_limit(balance):
    """-5% of current balance."""
    return -balance * 0.05


class Tracker:
    def __init__(self, bal=300):
        self.bal = bal
        self.opens = []
        self.losses = 0
        self.halt_until = None
        self.daily = 0.0
        self.trades = []
        self.days = defaultdict(int)
        self.peak = bal
        self.trough = bal

    def open_pos(self, e, tp, sl, dt, d, lot):
        self.opens.append({"e": e, "tp": tp, "sl": sl, "d": d, "lot": lot, "t": dt, "be": False})

    def update(self, px, dt, atr):
        atr = atr or 3.5
        cl, rm = [], []
        for p in self.opens:
            e, d, lot, sl = p["e"], p["d"], p["lot"], p["sl"]
            pv = lot * 100

            if TRAIL_SL and not p["be"]:
                pm = px - e if d == "BUY" else e - px
                if pm >= atr:
                    p["be"] = True
                    p["sl"] = e
            elif TRAIL_SL and p["be"]:
                if d == "BUY":
                    ns = px - atr * 0.7
                    if ns > sl + 0.5:
                        p["sl"] = round(ns, 2)
                else:
                    ns = px + atr * 0.7
                    if ns < sl - 0.5:
                        p["sl"] = round(ns, 2)

            sl = p["sl"]
            tp = p["tp"]
            hit = False
            pnl = 0.0
            r = ""

            if d == "BUY":
                if tp and px >= tp:
                    pnl = (tp - e) * pv; r = "TP"; hit = True
                elif sl and px <= sl:
                    pnl = (sl - e) * pv; r = "SL" if sl <= e else "TRAIL"; hit = True
            else:
                if tp and px <= tp:
                    pnl = (e - tp) * pv; r = "TP"; hit = True
                elif sl and px >= sl:
                    pnl = (e - sl) * pv; r = "SL" if sl >= e else "TRAIL"; hit = True

            if hit:
                self.bal += pnl
                self.daily += pnl
                self.peak = max(self.peak, self.bal)
                self.trough = min(self.trough, self.bal)
                cl.append({**p, "pnl": round(pnl, 2), "reason": r, "close": dt})
                self.trades.append({**p, "pnl": round(pnl, 2), "reason": r, "close": dt})
                if r == "SL":
                    self.losses += 1
                    if self.losses >= HALT_LOSSES:
                        self.halt_until = dt + timedelta(hours=HALT_HOURS)
                else:
                    self.losses = 0
            else:
                rm.append(p)
        self.opens = rm
        return cl

    def can(self, dt):
        if self.halt_until and dt < self.halt_until:
            return False
        if self.daily <= get_daily_loss_limit(self.bal):
            return False
        return True

    def reset(self):
        self.daily = 0.0

    def close_all(self, px, dt):
        for p in self.opens:
            pnl = ((px - p["e"]) if p["d"] == "BUY" else (p["e"] - px)) * p["lot"] * 100
            self.trades.append({**p, "pnl": round(pnl, 2), "reason": "EOD", "close": dt})
            self.bal += pnl
        self.opens = []


def fetch_data(sym, days, offset):
    mt5.initialize(login=MT5_LOGIN, password=MT5_PASSWORD, server=MT5_SERVER)
    mt5.symbol_select(sym, True)
    end = datetime.now(timezone.utc) - timedelta(days=offset)
    start = end - timedelta(days=days)
    data = {}
    for tf_name, tf_val in [("M5", mt5.TIMEFRAME_M5), ("M15", mt5.TIMEFRAME_M15)]:
        rates = mt5.copy_rates_range(sym, tf_val, start, end)
        if rates is None or len(rates) == 0:
            mt5.shutdown()
            return None
        df = pd.DataFrame(rates)
        df["time"] = pd.to_datetime(df["time"], unit="s", utc=True)
        df.set_index("time", inplace=True)
        df["session"] = df.index.map(
            lambda d: "weekend" if d.weekday() >= 5 else (
                "london" if 8 <= d.hour < 13 else (
                    "overlap" if 13 <= d.hour < 17 else (
                        "new_york" if 17 <= d.hour < 22 else "asian"))))
        data[tf_name] = df
    mt5.shutdown()
    return data


def run_test(offset_days, label):
    logger.info("=" * 70)
    logger.info(f"V21 SAFE COMPOUNDING - {label}")
    logger.info(f"Offset: {offset_days}d | Graduated risk: 1%/2%/3%")
    logger.info("=" * 70)

    data = fetch_data(SYMBOL, DAYS, offset_days)
    if data is None:
        logger.critical("No data.")
        return None
    m5, m15 = data["M5"], data["M15"]
    if len(m15) < 500:
        return None

    strat = GoldScalpingStrategy()
    strat._max_trades_per_day = 200; strat._max_open_positions = 10
    vf = GoldVolatilityFilter()
    trk = Tracker(INITIAL_BALANCE)

    warmup, total = 200, len(m15)
    t0 = time.time()

    for i in range(warmup, total):
        dt = m15.index[i]
        px = float(m15["close"].iloc[i])
        sess = m15["session"].iloc[i]

        if i % 1000 == 0:
            logger.info(f"  {int((i-warmup)/(total-warmup)*100)}% | Bal:${trk.bal:.0f} | T:{len(trk.trades)} | Op:{len(trk.opens)}")

        if sess == "weekend":
            trk.close_all(px, dt)
            continue
        if dt.hour == 0 and dt.minute < 15:
            trk.reset()

        m5w = m5[m5.index <= dt].iloc[-500:]
        m15w = m15[m15.index <= dt].iloc[max(0, i+1-500):i+1]
        if len(m5w) < 50 or len(m15w) < 50:
            continue

        try:
            m5i = compute_all_indicators(m5w)
            m15i = compute_all_indicators(m15w)
        except Exception:
            continue

        try:
            atr_s = m5i.get("atr", pd.Series())
            atr = float(atr_s.iloc[-1]) if not atr_s.empty else 3.5
        except Exception:
            atr = 3.5

        trk.update(px, dt, atr)
        if not trk.can(dt) or len(trk.opens) >= MAX_POS:
            continue

        try:
            res = strat.analyze(
                m1_indicators=m5i, m5_indicators=m5i, m15_indicators=m15i,
                m1_ohlcv=m5w, m5_ohlcv=m5w, m15_ohlcv=m15w, news_context=None)
        except Exception:
            continue

        d = res.get("direction", "NONE")
        sc = res.get("setup_score", 0)
        if d == "NONE" or sc < MIN_SCORE:
            continue

        if d == "BUY":
            close = m15w["close"].values
            if len(close) >= 200:
                e200 = pd.Series(close).ewm(200, adjust=False).mean().values
                if len(e200) >= 10 and float(e200[-1]) <= float(e200[-10]):
                    continue

        try:
            vo = vf.analyze(
                m1_ohlcv=m5w, m5_ohlcv=m5w, m15_ohlcv=m15w,
                m1_indicators=m5i, m5_indicators=m5i, m15_indicators=m15i)
            if not vo.get("trade_ok", False):
                continue
        except Exception:
            continue

        sl_d = atr * 1.5
        tp_d = atr * 3.0
        if d == "BUY":
            sl = round(px - sl_d, 2); tp = round(px + tp_d, 2)
        else:
            sl = round(px + sl_d, 2); tp = round(px - tp_d, 2)

        risk_pct = get_risk_percent(trk.bal)
        risk = trk.bal * (risk_pct / 100)
        lot = max(0.01, min(risk / (sl_d * 100) if sl_d else 0.01, 50))
        lot = round(lot, 2)

        trk.open_pos(px, tp, sl, dt, d, lot)
        trk.days[dt.date()] += 1

    trk.close_all(float(m15["close"].iloc[-1]), m15.index[-1])
    elapsed = time.time() - t0

    t = trk.trades; n = len(t)
    w = [x for x in t if x["pnl"] > 0]; l = [x for x in t if x["pnl"] < 0]
    wr = len(w) / max(n, 1) * 100
    gp = sum(x["pnl"] for x in w); gl = abs(sum(x["pnl"] for x in l))
    pf = gp / max(gl, 0.01)
    tp = sum(x["pnl"] for x in t); fb = INITIAL_BALANCE + tp
    bp = sum(x["pnl"] for x in t if x["d"] == "BUY")
    sp = sum(x["pnl"] for x in t if x["d"] == "SELL")
    bc = sum(1 for x in t if x["d"] == "BUY"); sc = sum(1 for x in t if x["d"] == "SELL")
    ad = len(trk.days); atd = n / max(ad, 1)
    mc = 0; cc = 0
    for x in t:
        if x["pnl"] < 0: cc += 1; mc = max(mc, cc)
        else: cc = 0
    tpc = len([x for x in t if x["reason"] == "TP"])
    slc = len([x for x in t if x["reason"] == "SL"])
    trc = len([x for x in t if x["reason"] == "TRAIL"])

    logger.info("\n" + "=" * 70)
    logger.info(f"V21 RESULTS - {label}")
    logger.info("=" * 70)
    logger.info(f"  Window: {m15.index[0].date()} to {m15.index[-1].date()}")
    logger.info(f"  Initial: ${INITIAL_BALANCE:.0f} -> Final: ${fb:.0f} | P/L: ${tp:+.0f} (+{tp/INITIAL_BALANCE*100:.0f}%)")
    logger.info(f"  LOWEST BALANCE: ${trk.trough:.0f} | Peak: ${trk.peak:.0f} | DD: {(trk.peak-trk.trough)/max(trk.peak,1)*100:.1f}%")
    logger.info(f"  Trades: {n} (BUY:{bc} SELL:{sc}) | WR: {wr:.1f}% | PF: {pf:.2f}")
    logger.info(f"  Avg Win: ${gp/max(len(w),1):.0f} | Avg Loss: ${gl/max(len(l),1):.0f}")
    logger.info(f"  TP: {tpc} | SL: {slc} | TRAIL: {trc} | MaxConsLoss: {mc}")
    logger.info(f"  BUY: ${bp:+.0f} | SELL: ${sp:+.0f}")
    logger.info(f"  Days: {ad} | Avg/day: {atd:.1f} | Runtime: {elapsed:.1f}s")
    logger.info("=" * 70)

    return {"label": label, "final": fb, "pnl": tp, "wr": wr, "pf": pf, "trades": n,
            "low": trk.trough, "dd": (trk.peak-trk.trough)/max(trk.peak,1)*100, "days": ad, "atd": atd}


def run():
    results = []
    offsets_labels = [
        (0, "Window 1 (Jan-Jul 2026)"),
        (180, "Window 2 (Jul-Dec 2025)"),
        (360, "Window 3 (Jan-Jun 2025)"),
    ]

    for offset, label in offsets_labels:
        r = run_test(offset, label)
        if r:
            results.append(r)

    # 3-Window Comparison Table
    if len(results) > 0:
        windows = [r["label"] for r in results]
        cols = len(results)
        logger.info("\n" + "=" * (25 + cols * 18))
        logger.info(f"V22 3-WINDOW CROSS-VALIDATION (18 months, {cols} windows)")
        logger.info("=" * (25 + cols * 18))
        header = f"{'Metric':<22s}"
        for i, w in enumerate(windows):
            header += f" {'W'+str(i+1):>17s}"
        logger.info(header)
        logger.info("-" * (22 + cols * 18))

        for key, label in [("final", "Final Balance"), ("pnl", "Total P/L"),
                            ("wr", "Win Rate %"), ("pf", "Profit Factor"),
                            ("trades", "Total Trades"), ("low", "Lowest Bal"),
                            ("dd", "Max Drawdown %"), ("atd", "Avg Trades/Day")]:
            row = f"{label:<22s}"
            for r in results:
                v = r[key]
                if isinstance(v, float):
                    if key == "wr" or key == "dd" or key == "atd":
                        row += f" {v:>17.1f}"
                    else:
                        row += f" ${v:>16,.0f}"
                else:
                    row += f" {v:>17}"
            logger.info(row)
        logger.info("=" * (22 + cols * 18))


if __name__ == "__main__":
    run()