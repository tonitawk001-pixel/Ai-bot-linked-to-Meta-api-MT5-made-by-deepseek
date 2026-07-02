"""Analyze V15 backtest trades to profile loss patterns."""
import json
from collections import defaultdict
from datetime import datetime

import sys

version = sys.argv[1] if len(sys.argv) > 1 else "v15"
trades_file = f"logs/{version}_trades.json"

with open(trades_file) as f:
    trades = json.load(f)

total = len(trades)
wins = [t for t in trades if t["pnl"] > 0]
losses = [t for t in trades if t["pnl"] < 0]

print("=" * 60)
print("V15 LOSS PATTERN ANALYSIS")
print("=" * 60)

# 1. By direction
for d in ["BUY", "SELL"]:
    d_trades = [t for t in trades if t["direction"] == d]
    d_wins = [t for t in d_trades if t["pnl"] > 0]
    d_losses = [t for t in d_trades if t["pnl"] < 0]
    wr = len(d_wins) / max(len(d_trades), 1) * 100
    pnl = sum(t["pnl"] for t in d_trades)
    avg_w = sum(t["pnl"] for t in d_wins) / max(len(d_wins), 1)
    avg_l = sum(t["pnl"] for t in d_losses) / max(len(d_losses), 1)
    print(f"\n{d}: {len(d_trades)} trades, WR={wr:.1f}%, PnL=${pnl:+.2f}")
    print(f"  Wins: {len(d_wins)} (avg +${avg_w:.2f})")
    print(f"  Losses: {len(d_losses)} (avg ${avg_l:.2f})")

# 2. By session
sessions = defaultdict(lambda: {"trades": 0, "wins": 0, "losses": 0, "pnl": 0})
for t in trades:
    dt = datetime.fromisoformat(t["open_time"].replace("Z", "+00:00"))
    h = dt.hour
    if 8 <= h < 13:
        sess = "London"
    elif 13 <= h < 17:
        sess = "Overlap"
    elif 17 <= h < 22:
        sess = "NY"
    else:
        sess = "Asian"
    sessions[sess]["trades"] += 1
    if t["pnl"] > 0:
        sessions[sess]["wins"] += 1
    else:
        sessions[sess]["losses"] += 1
    sessions[sess]["pnl"] += t["pnl"]

print("\n--- BY SESSION ---")
for s in ["Asian", "London", "Overlap", "NY"]:
    d = sessions[s]
    wr = d["wins"] / max(d["trades"], 1) * 100
    print(f'{s:10s}: {d["trades"]:4d} tr | WR={wr:.1f}% | PnL=${d["pnl"]:+.2f}')

# 3. Loss rate by hour (UTC)
loss_by_hour = defaultdict(lambda: {"trades": 0, "losses": 0})
for t in trades:
    dt = datetime.fromisoformat(t["open_time"].replace("Z", "+00:00"))
    h = dt.hour
    loss_by_hour[h]["trades"] += 1
    if t["pnl"] < 0:
        loss_by_hour[h]["losses"] += 1

print("\n--- LOSS RATE BY HOUR (UTC) ---")
for h in sorted(loss_by_hour.keys()):
    d = loss_by_hour[h]
    lr = d["losses"] / max(d["trades"], 1) * 100
    bar = "#" * int(lr / 2)
    print(f'  {h:02d}:00 | {d["trades"]:3d} tr | Loss: {lr:.0f}% {bar}')

# 4. Loss streak analysis
loss_streaks = []
curr = 0
for t in trades:
    if t["pnl"] < 0:
        curr += 1
    else:
        if curr > 0:
            loss_streaks.append(curr)
        curr = 0
if curr > 0:
    loss_streaks.append(curr)
print("\n--- LOSS STREAKS ---")
print(f"Total streaks: {len(loss_streaks)}")
print(f"Max streak: {max(loss_streaks)}")
print(f"Avg streak: {sum(loss_streaks)/len(loss_streaks):.1f}")
hist = {}
for s in loss_streaks:
    hist[s] = hist.get(s, 0) + 1
for k in sorted(hist):
    print(f"  Streak of {k}: {hist[k]}x")

# 5. R:R analysis
print("\n--- R:R ANALYSIS ---")
for d in ["BUY", "SELL"]:
    d_trades = [t for t in trades if t["direction"] == d]
    rr_sum = 0
    count = 0
    for t in d_trades:
        sl_dist = abs(t["entry"] - t["sl"])
        tp_dist = abs(t["tp"] - t["entry"])
        if sl_dist > 0:
            rr_sum += tp_dist / sl_dist
            count += 1
    print(f"{d}: avg R:R = {(rr_sum / max(count, 1)):.2f}")

# 6. PnL by day of week
dow_pnl = defaultdict(float)
dow_trades = defaultdict(int)
for t in trades:
    dt = datetime.fromisoformat(t["open_time"].replace("Z", "+00:00"))
    d = dt.strftime("%a")
    dow_pnl[d] += t["pnl"]
    dow_trades[d] += 1
print("\n--- PnL BY DAY OF WEEK ---")
for d in ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]:
    if d in dow_trades:
        print(f"{d}: {dow_trades[d]} tr | PnL=${dow_pnl[d]:+.2f}")

# 7. Top 10 worst & best days
daily = {}
for t in trades:
    dt = datetime.fromisoformat(t["open_time"].replace("Z", "+00:00"))
    d = dt.strftime("%Y-%m-%d")
    if d not in daily:
        daily[d] = {"pnl": 0, "trades": 0, "losses": 0}
    daily[d]["pnl"] += t["pnl"]
    daily[d]["trades"] += 1
    if t["pnl"] < 0:
        daily[d]["losses"] += 1

sorted_days = sorted(daily.items(), key=lambda x: x[1]["pnl"])
print("\n--- WORST 10 DAYS ---")
for d, v in sorted_days[:10]:
    print(f'{d}: PnL=${v["pnl"]:+.2f} | {v["trades"]} trades | {v["losses"]} losses')

print("\n--- BEST 10 DAYS ---")
sorted_best = sorted(daily.items(), key=lambda x: x[1]["pnl"], reverse=True)
for d, v in sorted_best[:10]:
    print(f'{d}: PnL=${v["pnl"]:+.2f} | {v["trades"]} trades | {v["losses"]} losses')

print("\n" + "=" * 60)
print("SUMMARY")
print("=" * 60)
print(f"Total Trades: {total}")
print(f"Wins: {len(wins)} ({len(wins)/total*100:.1f}%)")
print(f"Losses: {len(losses)} ({len(losses)/total*100:.1f}%)")
print(f"Total PnL: ${sum(t['pnl'] for t in trades):+.2f}")