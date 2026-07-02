"""
V25 — 3-Month M15 Backtest (Live Bot Logic)
=============================================
Uses M15 candles with entry at each bar (matches 15-min cooldown).
Same V22 strategy: graduated risk, trailing SL, 3-loss halt,
BUY EMA200 filter, 5 max positions, 90 days MT5 data.
"""

import sys, os, time
from datetime import datetime, timedelta, timezone
import pandas as pd
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from trading_bot.utils.logger import logger
from trading_bot.strategy.gold_scalping_strategy import GoldScalpingStrategy
from trading_bot.strategy.gold_volatility_filter import GoldVolatilityFilter
from trading_bot.indicators.technical_indicators import compute_all_indicators
from trading_bot.config import Config
import MetaTrader5 as mt5

DAYS = 90
INITIAL_BALANCE = 300.0
MIN_SCORE = 50
MAX_POS = 5


def get_risk_pct(bal):
    if bal < 250: return 0.5
    elif bal < 500: return 1.5
    elif bal < 1000: return 2.5
    return 3.0


class Tracker:
    def __init__(self):
        self.bal = 300; self.opens = []; self.losses = 0
        self.halt = None; self.daily = 0.0; self.trades = []
        self.peak = 300; self.low = 300

    def update(self, px, atr):
        atr = atr or 3.5; cl, rm = [], []
        for p in self.opens:
            e, d, sl, tp, lot = p["e"], p["d"], p["sl"], p["tp"], p["lot"]
            pv = lot * 100
            if not p.get("be"):
                if (px - e if d == "BUY" else e - px) >= atr:
                    p["be"] = True; p["sl"] = e
            elif p.get("be"):
                if d == "BUY":
                    ns = px - atr * 0.7
                    if ns > sl + 0.5: p["sl"] = round(ns, 2)
                else:
                    ns = px + atr * 0.7
                    if ns < sl - 0.5: p["sl"] = round(ns, 2)
            sl, tp = p["sl"], p["tp"]; hit = False; pnl = 0; r = ""
            if d == "BUY":
                if tp and px >= tp: pnl = (tp - e) * pv; r = "TP"; hit = True
                elif sl and px <= sl: pnl = (sl - e) * pv; r = "SL" if sl <= e else "TRAIL"; hit = True
            else:
                if tp and px <= tp: pnl = (e - tp) * pv; r = "TP"; hit = True
                elif sl and px >= sl: pnl = (e - sl) * pv; r = "SL" if sl >= e else "TRAIL"; hit = True
            if hit:
                self.bal += pnl; self.daily += pnl
                self.peak = max(self.peak, self.bal)
                self.low = min(self.low, self.bal)
                cl.append({**p, "pnl": round(pnl, 2), "reason": r})
                self.trades.append({**p, "pnl": round(pnl, 2), "reason": r})
                if r == "SL":
                    self.losses += 1
                    if self.losses >= 3: self.halt = time.time() + 7200
                else: self.losses = 0
            else: rm.append(p)
        self.opens = rm; return cl

    def can(self):
        if self.halt and time.time() < self.halt: return False
        if self.daily <= -self.bal * 0.05: return False
        return True

    def reset(self): self.daily = 0.0


def run():
    logger.info(f"V25 — 3-MONTH M15 BACKTEST ({DAYS} days)")
    mt5.initialize(login=Config.MT5_LOGIN, password=Config.MT5_PASSWORD, server=Config.MT5_SERVER)
    mt5.symbol_select("XAUUSD", True)
    end = datetime.now(timezone.utc)
    start = end - timedelta(days=DAYS)

    data = {}
    for tf_name, tf_val in [("M1", mt5.TIMEFRAME_M1), ("M5", mt5.TIMEFRAME_M5), ("M15", mt5.TIMEFRAME_M15)]:
        rates = mt5.copy_rates_range("XAUUSD", tf_val, start, end)
        if rates is None or len(rates) == 0:
            if tf_name == "M1":
                logger.warning("M1 unavailable — using M5 for M1")
                rates = mt5.copy_rates_range("XAUUSD", mt5.TIMEFRAME_M5, start, end)
            if rates is None or len(rates) == 0:
                mt5.shutdown(); logger.critical(f"No {tf_name}"); return
        df = pd.DataFrame(rates)
        df["time"] = pd.to_datetime(df["time"], unit="s", utc=True)
        df.set_index("time", inplace=True)
        data[tf_name] = df
        logger.info(f"  {tf_name}: {len(df)} candles")
    mt5.shutdown()

    m1, m5, m15 = data["M1"], data["M5"], data["M15"]
    trk = Tracker()
    strat = GoldScalpingStrategy(); strat._max_trades_per_day = 200; strat._max_open_positions = 10
    vf = GoldVolatilityFilter()
    t0 = time.time(); total = len(m15)
    logger.info(f"M15 candles: {total} | Entry at each M15 bar (=15-min cooldown)")

    for i in range(200, total):
        dt = m15.index[i]; px = float(m15["close"].iloc[i])
        if i % 800 == 0:
            logger.info(f"  {int(i/total*100)}% | Bal:${trk.bal:.0f} | T:{len(trk.trades)} | Op:{len(trk.opens)}")

        if dt.hour == 0 and dt.minute == 0: trk.reset()

        m1w = m1[m1.index <= dt].iloc[-500:]
        m5w = m5[m5.index <= dt].iloc[-500:]
        m15w = m15[max(0, i+1-500):i+1]
        if len(m1w) < 50 or len(m5w) < 50 or len(m15w) < 50: continue

        try:
            m1i = compute_all_indicators(m1w)
            m5i = compute_all_indicators(m5w)
            m15i = compute_all_indicators(m15w)
        except: continue
        try: atr = float(m5i["atr"].iloc[-1]) if not m5i.get("atr", pd.Series()).empty else 3.5
        except: atr = 3.5

        trk.update(px, atr)
        if not trk.can() or len(trk.opens) >= MAX_POS: continue

        try:
            res = strat.analyze(m1_indicators=m1i, m5_indicators=m5i, m15_indicators=m15i,
                                m1_ohlcv=m1w, m5_ohlcv=m5w, m15_ohlcv=m15w, news_context=None)
        except: continue
        d = res.get("direction", "NONE"); sc = res.get("setup_score", 0)
        if d == "NONE" or sc < MIN_SCORE: continue
        if d == "BUY":
            c = m15w["close"].values
            if len(c) >= 200:
                e200 = pd.Series(c).ewm(200, adjust=False).mean().values
                if len(e200) >= 10 and float(e200[-1]) <= float(e200[-10]): continue
        try:
            vo = vf.analyze(m1_ohlcv=m1w, m5_ohlcv=m5w, m15_ohlcv=m15w,
                            m1_indicators=m1i, m5_indicators=m5i, m15_indicators=m15i)
            if not vo.get("trade_ok", False): continue
        except: continue

        sd = atr * 1.5; td = 3.0 * atr
        sl = round(px - sd, 2) if d == "BUY" else round(px + sd, 2)
        tp = round(px + td, 2) if d == "BUY" else round(px - td, 2)
        rp = get_risk_pct(trk.bal)
        lot = max(0.01, min(trk.bal * (rp/100) / (sd * 100) if sd else 0.01, 50))
        lot = round(lot, 2)
        trk.opens.append({"e": px, "tp": tp, "sl": sl, "d": d, "lot": lot, "be": False})

    final_px = float(m15["close"].iloc[-1])
    trk.update(final_px, atr)
    trk.bal += sum(((final_px - p["e"]) if p["d"]=="BUY" else (p["e"]-final_px))*p["lot"]*100 for p in trk.opens)
    trk.opens = []

    t = trk.trades; n = len(t)
    w = [x for x in t if x["pnl"] > 0]; l = [x for x in t if x["pnl"] < 0]
    wr = len(w) / max(n, 1) * 100
    gp = sum(x["pnl"] for x in w); gl = abs(sum(x["pnl"] for x in l))
    pf = gp / max(gl, 0.01)
    tp = sum(x["pnl"] for x in t); fb = INITIAL_BALANCE + tp
    bp = sum(x["pnl"] for x in t if x["d"] == "BUY")
    sp = sum(x["pnl"] for x in t if x["d"] == "SELL")
    bc = sum(1 for x in t if x["d"] == "BUY"); sc = sum(1 for x in t if x["d"] == "SELL")
    mc = 0; cc = 0
    for x in t:
        if x["pnl"] < 0: cc += 1; mc = max(mc, cc)
        else: cc = 0
    tpc = len([x for x in t if x["reason"] == "TP"])
    slc = len([x for x in t if x["reason"] == "SL"])
    trc = len([x for x in t if x["reason"] == "TRAIL"])

    logger.info("\n" + "=" * 70)
    logger.info(f"V25 3-MONTH M15 BACKTEST RESULTS")
    logger.info("=" * 70)
    logger.info(f"  Window: {m15.index[0].date()} to {m15.index[-1].date()} ({DAYS} days)")
    logger.info(f"  M15 candles: {total}")
    logger.info(f"  Initial: ${INITIAL_BALANCE:.0f} -> Final: ${fb:.0f} | P/L: ${tp:+.0f} (+{tp/INITIAL_BALANCE*100:.0f}%)")
    logger.info(f"  LOWEST: ${trk.low:.0f} | Peak: ${trk.peak:.0f}")
    logger.info(f"  Trades: {n} (BUY:{bc} SELL:{sc}) | WR: {wr:.1f}% | PF: {pf:.2f}")
    logger.info(f"  TP: {tpc} | SL: {slc} | TRAIL: {trc} | MaxConsLoss: {mc}")
    logger.info(f"  BUY P/L: ${bp:+.0f} | SELL P/L: ${sp:+.0f}")
    logger.info(f"  Avg Win: ${gp/max(len(w),1):.0f} | Avg Loss: ${gl/max(len(l),1):.0f}")
    logger.info(f"  Runtime: {time.time()-t0:.1f}s")
    logger.info("=" * 70)


if __name__ == "__main__":
    run()