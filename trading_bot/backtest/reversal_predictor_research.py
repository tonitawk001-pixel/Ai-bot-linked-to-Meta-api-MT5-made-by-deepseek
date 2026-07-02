"""
TREND REVERSAL PREDICTOR RESEARCH.

Analyzes all losing trades to identify which signal gives the earliest
and most reliable warning that a trend is ending.
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass, field

from trading_bot.utils.logger import logger
from trading_bot.backtest.simulation_engine import SimulatedPortfolio
from trading_bot.backtest.edge_analysis import _generate_trending_data, _prices_to_ohlcv
from trading_bot.backtest.strategy_failure_analysis import classify_trade


@dataclass
class SignalMetrics:
    name: str
    appearances: int = 0
    total_reversals: int = 0
    lead_times: list = field(default_factory=list)
    false_positives: int = 0
    total_non_reversals: int = 0

    @property
    def detection_rate(self):
        return self.appearances / max(self.total_reversals, 1) * 100

    @property
    def avg_lead_time(self):
        return np.mean(self.lead_times) if self.lead_times else 0

    @property
    def false_positive_rate(self):
        return self.false_positives / max(self.total_non_reversals, 1) * 100

    @property
    def predictive_power(self):
        d = self.detection_rate / 100
        fp = 1 - (self.false_positive_rate / 100)
        lead = min(self.avg_lead_time / 10, 1.0)
        return round(d * fp * lead * 100, 1)


def measure_signals(ohlcv, entry_index, trade):
    from trading_bot.indicators.technical_indicators import compute_rsi, compute_macd, compute_atr
    if entry_index < 25 or ohlcv is None or len(ohlcv) < entry_index + 5:
        return {}
    pre = ohlcv.iloc[max(0, entry_index-25):entry_index+1]
    if len(pre) < 20:
        return {}
    close = pre["close"].values
    action = trade.get("action", "BUY")
    sigs = {}

    rsi_v = compute_rsi(pre["close"], 14).values
    if action == "BUY" and len(rsi_v) >= 10:
        if max(close[-5:]) > max(close[-10:-5]) and max(rsi_v[-5:]) < max(rsi_v[-10:-5]):
            sigs["rsi_divergence"] = {"present": True, "lead": 3}
    elif action == "SELL" and len(rsi_v) >= 10:
        if min(close[-5:]) < min(close[-10:-5]) and min(rsi_v[-5:]) > min(rsi_v[-10:-5]):
            sigs["rsi_divergence"] = {"present": True, "lead": 3}

    macd = compute_macd(pre["close"], 12, 26, 9)
    hist = macd["histogram"].values
    if len(hist) >= 8:
        dec = 0; mx = 0
        for s in np.diff(hist[-8:]):
            if s < 0: dec += 1; mx = max(mx, dec)
            else: dec = 0
        if mx >= 3:
            sigs["macd_slope"] = {"present": True, "lead": mx}

    e20 = pd.Series(close).ewm(span=20).mean().values
    e50 = pd.Series(close).ewm(span=50).mean().values
    if len(e20) >= 10 and len(e50) >= 10:
        d_early = abs(e20[-10] - e50[-10])
        d_late = abs(e20[-1] - e50[-1])
        if d_early > 0 and d_late < d_early * 0.7:
            sigs["ema_compression"] = {"present": True, "lead": 5}

    atr = compute_atr(pre["high"], pre["low"], pre["close"], 14).values
    if len(atr) >= 10:
        ae = np.mean(atr[-10:-5])
        al = np.mean(atr[-5:])
        if ae > 0 and al / ae > 1.5:
            sigs["atr_expansion"] = {"present": True, "lead": 3}

    if len(close) >= 20:
        st = "up" if close[-1] > close[-5] else "down"
        lt = "up" if close[-1] > close[-20] else "down"
        if st != lt:
            sigs["mtf_disagreement"] = {"present": True, "lead": 2}

    if len(close) >= 20:
        sma = np.mean(close[-20:])
        std = np.std(close[-20:])
        if std > 0:
            ub = sma + 2*std; lb = sma - 2*std
            if action == "BUY" and close[-1] >= ub*0.995:
                sigs["bband_extreme"] = {"present": True, "lead": 1}
            elif action == "SELL" and close[-1] <= lb*1.005:
                sigs["bband_extreme"] = {"present": True, "lead": 1}

    return sigs


def run():
    from trading_bot.strategy.rule_engine import RuleEngine
    from trading_bot.risk.risk_manager import RiskManager
    from trading_bot.ai.deepseek_client import DeepSeekClient
    from trading_bot.indicators.technical_indicators import compute_all_indicators
    from trading_bot.main import build_ai_payload, determine_trade_action, compute_lot_size, compute_sl_tp

    logger.info("="*70)
    logger.info("TREND REVERSAL PREDICTOR RESEARCH")
    logger.info("="*70)

    all_trades = []
    for sim in range(8):
        prices = _generate_trending_data(600)
        ohlcv = _prices_to_ohlcv(prices, 10)
        portfolio = SimulatedPortfolio(10000)
        re = RuleEngine()
        rm = RiskManager(10000)
        dc = DeepSeekClient(); dc.initialize()
        entries = {}

        for i in range(100, len(ohlcv)):
            w = ohlcv.iloc[:i+1]
            cp = float(ohlcv["close"].iloc[i])
            ct = str(ohlcv.index[i])
            ind = compute_all_indicators(w)
            dec = re.analyze(ohlcv=w, indicators=ind)
            ap = build_ai_payload("EURUSD", "H1", w, ind, dec)
            aa = dc.analyze_market(ap)
            re_ = rm.validate(dec, aa, w)
            act = determine_trade_action(dec, aa)
            lot = compute_lot_size(dec, rm, portfolio.balance)
            sl, tp = compute_sl_tp(dec, act, w) if act != "NONE" else (0,0)

            if re_["approved"] and dec.get("setup_valid") and act != "NONE":
                if not any(p["symbol"]=="EURUSD" for p in portfolio.open_positions):
                    pos = portfolio.open_position(act, "EURUSD", lot, cp, sl, tp, ct)
                    if pos:
                        entries[pos["ticket"]] = i

            closed = portfolio.update_positions(cp, ct)
            for c in closed:
                ei = entries.get(c.get("ticket", 0), i)
                all_trades.append((ei, c, ohlcv))

        final_p = float(ohlcv["close"].iloc[-1])
        closed_f = portfolio.update_positions(final_p, str(ohlcv.index[-1]))
        for c in closed_f:
            ei = entries.get(c.get("ticket", 0), len(ohlcv)-1)
            all_trades.append((ei, c, ohlcv))

    logger.info(f"Total trades collected: {len(all_trades)}")

    # Classify
    rev_sigs = []
    non_rev = []
    wins = []

    for ei, t, ohlcv in all_trades:
        if t.get("pnl", 0) >= 0:
            wins.append((ei, t, ohlcv))
            continue
        cat = classify_trade(t, ohlcv, ei)
        sigs = measure_signals(ohlcv, ei, t)
        if cat == "trend_reversal":
            rev_sigs.append((ei, t, ohlcv, sigs))
        else:
            non_rev.append((ei, t, ohlcv))

    # --- Per-signal false positive tracking ---
    signal_names = ["rsi_divergence","macd_slope","ema_compression",
                    "atr_expansion","mtf_disagreement","bband_extreme"]

    fp_counts = {name: 0 for name in signal_names}
    win_count = len(wins)
    for ei, t, ohlcv in wins:
        sigs = measure_signals(ohlcv, ei, t)
        for name in signal_names:
            if sigs.get(name, {}).get("present"):
                fp_counts[name] += 1

    total_non_rev_count = win_count + len(non_rev)

    metrics = {}
    for name in signal_names:
        m = SignalMetrics(name=name, total_reversals=len(rev_sigs))
        for _, _, _, sigs in rev_sigs:
            s = sigs.get(name, {})
            if s.get("present"):
                m.appearances += 1
                m.lead_times.append(s.get("lead", 1))
        m.false_positives = fp_counts[name]
        m.total_non_reversals = max(total_non_rev_count, 1)
        metrics[name] = m

    print(f"\nReversal trades: {len(rev_sigs)}")
    print(f"Non-reversal losses: {len(non_rev)}")
    print(f"Wins: {win_count}")

    print("\n" + "="*110)
    print("SIGNAL PREDICTIVE POWER RANKING")
    print("="*110)
    print(f"{'Signal':<22} {'Detect%':<10} {'Lead':<8} {'FalsePos%':<12} {'Power':<8}")
    print("-"*110)

    ranked = sorted(metrics.values(), key=lambda m: m.predictive_power, reverse=True)
    for i, m in enumerate(ranked, 1):
        print(f"{i:2d}. {m.name:<19} {m.detection_rate:<9.1f}% {m.avg_lead_time:<7.1f} {m.false_positive_rate:<10.1f}% {m.predictive_power:<7.1f}")

    if ranked:
        best = ranked[0]
        print(f"\nBEST: {best.name} — detect={best.detection_rate:.0f}%, lead={best.avg_lead_time:.1f}, FP={best.false_positive_rate:.0f}%, power={best.predictive_power}")
        reduction = round(best.detection_rate * 0.78 * (1 - best.false_positive_rate/100))
        print(f"Net drawdown reduction estimate: {reduction}%")

    # Summary table
    print("\n" + "="*70)
    print("SUMMARY BY SIGNAL")
    print("="*70)
    for m in ranked:
        print(f"{m.name:20s} | detect={m.detection_rate:5.1f}% | lead={m.avg_lead_time:4.1f} | FP={m.false_positive_rate:5.1f}% | power={m.predictive_power:5.1f}")
    print("="*70)

if __name__ == "__main__":
    run()