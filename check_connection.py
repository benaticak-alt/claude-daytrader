"""Read-only connectivity check. Places NO orders.

Verifies: .env loaded, Alpaca credentials valid, account is paper, market data
reachable, and the data-quality gate's verdict on each watchlist symbol.

    python check_connection.py
"""

from __future__ import annotations

import sys

import config


def main() -> None:
    print("=" * 60)
    print("CONFIG")
    print("=" * 60)
    print(f"  decider backend : {config.DECIDER_BACKEND}")
    print(f"  paper mode      : {config.ALPACA_PAPER}")
    print(f"  watchlist       : {', '.join(config.WATCHLIST)}")
    print(f"  alpaca key      : {'set (' + str(len(config.ALPACA_API_KEY)) + ' chars)' if config.ALPACA_API_KEY else 'MISSING'}")
    print(f"  alpaca secret   : {'set (' + str(len(config.ALPACA_SECRET_KEY)) + ' chars)' if config.ALPACA_SECRET_KEY else 'MISSING'}")

    if not config.ALPACA_API_KEY or not config.ALPACA_SECRET_KEY:
        sys.exit("\nFAIL: Alpaca keys not loaded from .env")

    if config.DECIDER_BACKEND == "claude":
        import os
        if not os.getenv("ANTHROPIC_API_KEY"):
            print("\n  WARNING: backend is 'claude' but ANTHROPIC_API_KEY is not set.")

    print("\n" + "=" * 60)
    print("BROKER (read-only)")
    print("=" * 60)
    from executor import Executor

    ex = Executor()
    acct = ex.account_state()
    if not acct:
        sys.exit("FAIL: could not fetch account — check that the keys are PAPER keys")

    print(f"  equity          : ${acct['equity']:,.2f}")
    print(f"  cash            : ${acct['cash']:,.2f}")
    print(f"  daily P/L       : ${acct['daily_pl']:,.2f}")
    print(f"  day trades used : {acct['daytrade_count']}")
    print(f"  PDT flagged     : {acct['pattern_day_trader']}")
    print(f"  market open     : {ex.market_open()}")

    positions = ex.open_positions()
    print(f"  open positions  : {[p['symbol'] for p in positions] or 'none'}")

    if acct["equity"] < config.PDT_MIN_EQUITY:
        print(f"\n  NOTE: equity below ${config.PDT_MIN_EQUITY:,.0f} — PDT rules apply "
              f"(max {config.PDT_MAX_DAY_TRADES} day trades / 5 business days)")

    print("\n" + "=" * 60)
    print("MARKET DATA + QUALITY GATE (read-only)")
    print("=" * 60)
    from data_feed import DataFeed

    feed = DataFeed()
    ok_count = 0
    for symbol in config.WATCHLIST:
        try:
            snap = feed.snapshot(symbol)
        except Exception as exc:
            print(f"  {symbol:5s} ERROR  {type(exc).__name__}: {str(exc)[:60]}")
            continue
        if snap.data_ok:
            ok_count += 1
            print(f"  {symbol:5s} OK     bid={snap.bid:.2f} ask={snap.ask:.2f} "
                  f"spread={snap.spread_pct:.3f}% bars={len(snap.bars)}")
        else:
            print(f"  {symbol:5s} GATED  {'; '.join(snap.data_issues)}")

    print(f"\n  {ok_count}/{len(config.WATCHLIST)} symbols passed the data gate")
    print("\nAll checks complete. No orders were placed.")


if __name__ == "__main__":
    main()
