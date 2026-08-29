"""Expectancy report — the number that decides whether a strategy is worth running.

Win rate alone is meaningless. A strategy winning 70% of the time still bleeds
if the losses are four times the size of the wins. What matters is expectancy:

    expectancy = (win_rate x avg_win) - (loss_rate x avg_loss)

Positive means the edge survives its own losses. Negative means it does not,
however good the win rate looks.

    python edge_report.py --days 7
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime, timedelta, timezone

import config
from alpaca.trading.client import TradingClient
from alpaca.trading.enums import QueryOrderStatus
from alpaca.trading.requests import GetOrdersRequest


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--days", type=int, default=7)
    args = p.parse_args()

    client = TradingClient(config.ALPACA_API_KEY, config.ALPACA_SECRET_KEY,
                           paper=config.ALPACA_PAPER)
    orders = client.get_orders(GetOrdersRequest(
        status=QueryOrderStatus.ALL,
        after=datetime.now(timezone.utc) - timedelta(days=args.days),
        limit=500,
    ))

    # Pair fills FIFO per symbol into round trips.
    legs: dict[str, list] = defaultdict(list)
    for o in sorted(orders, key=lambda x: x.submitted_at):
        if not o.filled_qty or float(o.filled_qty) <= 0:
            continue
        legs[o.symbol].append((
            o.side.value, float(o.filled_qty), float(o.filled_avg_price),
            o.submitted_at.astimezone(),
        ))

    trips = []
    for symbol, ls in legs.items():
        open_lots = []
        for side, qty, price, ts in ls:
            if side == "buy":
                open_lots.append((qty, price, ts))
            elif open_lots:
                bq, bp, bts = open_lots.pop(0)
                q = min(bq, qty)
                trips.append({
                    "symbol": symbol, "entry": bp, "exit": price, "qty": q,
                    "pnl": (price - bp) * q, "pct": 100 * (price - bp) / bp,
                    "held_min": (ts - bts).total_seconds() / 60,
                    "day": bts.strftime("%Y-%m-%d"),
                })

    if not trips:
        print("no completed round trips")
        return

    wins = [t for t in trips if t["pnl"] > 0]
    losses = [t for t in trips if t["pnl"] <= 0]
    avg_win = sum(t["pnl"] for t in wins) / len(wins) if wins else 0.0
    avg_loss = abs(sum(t["pnl"] for t in losses) / len(losses)) if losses else 0.0
    wr = len(wins) / len(trips)
    expectancy = wr * avg_win - (1 - wr) * avg_loss
    total = sum(t["pnl"] for t in trips)

    print("=" * 68)
    print(f"EDGE REPORT — {len(trips)} round trips over {args.days} days")
    print("=" * 68)
    print(f"  win rate        : {100*wr:5.1f}%   ({len(wins)}W / {len(losses)}L)")
    print(f"  average win     : ${avg_win:+7.2f}")
    print(f"  average loss    : ${-avg_loss:+7.2f}")
    print(f"  win/loss ratio  : {(avg_win/avg_loss if avg_loss else 0):5.2f}  "
          f"(need > {(1-wr)/wr if wr else 0:.2f} to break even at this win rate)")
    print(f"  EXPECTANCY      : ${expectancy:+7.2f} per trade")
    print(f"  total realized  : ${total:+7.2f}")

    print(f"\n  worst 5 trades:")
    for t in sorted(trips, key=lambda x: x["pnl"])[:5]:
        print(f"    {t['day']} {t['symbol']:5s} ${t['pnl']:+7.2f} ({t['pct']:+6.2f}%)  "
              f"held {t['held_min']:5.0f} min")
    print(f"\n  best 5 trades:")
    for t in sorted(trips, key=lambda x: -x["pnl"])[:5]:
        print(f"    {t['day']} {t['symbol']:5s} ${t['pnl']:+7.2f} ({t['pct']:+6.2f}%)  "
              f"held {t['held_min']:5.0f} min")

    print("\n  by symbol:")
    by_sym: dict[str, list] = defaultdict(list)
    for t in trips:
        by_sym[t["symbol"]].append(t)
    for sym, ts in sorted(by_sym.items(), key=lambda kv: sum(t["pnl"] for t in kv[1])):
        w = sum(1 for t in ts if t["pnl"] > 0)
        print(f"    {sym:5s} ${sum(t['pnl'] for t in ts):+8.2f}  "
              f"{len(ts):3d} trips  {w}W/{len(ts)-w}L")

    print("\n  by day:")
    by_day: dict[str, list] = defaultdict(list)
    for t in trips:
        by_day[t["day"]].append(t)
    for day, ts in sorted(by_day.items()):
        w = sum(1 for t in ts if t["pnl"] > 0)
        print(f"    {day}  ${sum(t['pnl'] for t in ts):+8.2f}  "
              f"{len(ts):3d} trips  {w}W/{len(ts)-w}L")


if __name__ == "__main__":
    main()
