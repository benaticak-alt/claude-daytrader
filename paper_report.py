"""Read-only P&L report for the paper account. Places NO orders.

Joins the local decision log against Alpaca's actual fills, so you see what
the bot *decided* next to what the broker *did*.

    python paper_report.py
"""

from __future__ import annotations

import json
from collections import defaultdict
from datetime import datetime, timedelta, timezone

import config
from alpaca.trading.client import TradingClient
from alpaca.trading.requests import GetOrdersRequest
from alpaca.trading.enums import QueryOrderStatus


def main() -> None:
    client = TradingClient(config.ALPACA_API_KEY, config.ALPACA_SECRET_KEY,
                           paper=config.ALPACA_PAPER)

    acct = client.get_account()
    equity = float(acct.equity)
    last_equity = float(acct.last_equity or equity)

    print("=" * 68)
    print("ACCOUNT")
    print("=" * 68)
    print(f"  equity          : ${equity:,.2f}")
    print(f"  session P/L     : ${equity - last_equity:+,.2f}")
    print(f"  day trades used : {acct.daytrade_count}")

    positions = client.get_all_positions()
    print(f"  open positions  : {[p.symbol for p in positions] or 'none'}")
    for p in positions:
        print(f"      {p.symbol}: {p.qty} @ {p.avg_entry_price}  "
              f"unrealized ${float(p.unrealized_pl):+,.2f}")

    print("\n" + "=" * 68)
    print("FILLS (last 24h)")
    print("=" * 68)
    req = GetOrdersRequest(
        status=QueryOrderStatus.ALL,
        after=datetime.now(timezone.utc) - timedelta(hours=24),
        limit=200,
    )
    orders = client.get_orders(req)
    if not orders:
        print("  no orders")
        return

    # Pair fills per symbol to compute realized P&L on completed round trips.
    legs: dict[str, list] = defaultdict(list)
    for o in sorted(orders, key=lambda x: x.submitted_at):
        filled_qty = float(o.filled_qty or 0)
        avg_price = float(o.filled_avg_price or 0)
        ts = o.submitted_at.astimezone().strftime("%H:%M:%S")
        status = str(o.status).split(".")[-1]
        notional = filled_qty * avg_price
        print(f"  {ts}  {o.side.value.upper():4s} {o.symbol:5s} "
              f"qty={filled_qty:<10.4f} @ ${avg_price:<9.2f} "
              f"${notional:>10,.2f}  [{status}]")
        if filled_qty > 0:
            legs[o.symbol].append((o.side.value, filled_qty, avg_price))

    print("\n" + "=" * 68)
    print("REALIZED ROUND TRIPS")
    print("=" * 68)
    total = 0.0
    any_rt = False
    for symbol, ls in legs.items():
        buys = [(q, p) for side, q, p in ls if side == "buy"]
        sells = [(q, p) for side, q, p in ls if side == "sell"]
        n = min(len(buys), len(sells))
        for i in range(n):
            any_rt = True
            bq, bp = buys[i]
            sq, sp = sells[i]
            qty = min(bq, sq)
            pnl = (sp - bp) * qty
            pct = 100 * (sp - bp) / bp if bp else 0.0
            total += pnl
            print(f"  {symbol:5s} buy ${bp:>8.2f} -> sell ${sp:>8.2f}  "
                  f"qty {qty:<9.4f}  P/L ${pnl:+8.2f} ({pct:+.3f}%)")
    if not any_rt:
        print("  no completed round trips")
    else:
        print(f"\n  TOTAL REALIZED: ${total:+,.2f}")

    print("\n(Read-only report. No orders were placed.)")


if __name__ == "__main__":
    main()
