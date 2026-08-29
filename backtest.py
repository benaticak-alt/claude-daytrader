"""Replay historical bars through the real strategies.

Live observation produces ~2 trades per strategy per day. Reaching a sample
worth acting on would take months. This replays years of history through the
*same* decider classes, the *same* context builder, and the *same* risk gate,
so results are comparable to live rather than a parallel invention.

    python backtest.py --years 2
    python backtest.py --years 1 --strategies orb,meanrev --symbols SPY,QQQ,NVDA

WHAT MAKES THIS FAITHFUL (and where it still is not):

  same  strategy code, context shape, risk gate, slippage model, EOD flatten
  same  session isolation (VWAP resets daily, pre/post-market excluded)
  NOT   fills assume the close of the signal bar plus slippage; a real order
        arrives a moment later at whatever the book offers
  NOT   IEX historical bars are a partial view of national volume
  NOT   no borrow costs, halts, or gaps mid-bar

Treat the output as a ranking device, not a P&L forecast. A strategy that
cannot clear zero here will not clear it live.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from typing import Dict, List

import config
from context_builder import build_context
from data_feed import EASTERN, Bar, SymbolSnapshot
from models import TradeDecision
from risk_gate import RiskGate
from shadow import ShadowBook

logging.basicConfig(level=logging.WARNING, format="%(levelname)s: %(message)s")
log = logging.getLogger("backtest")

from alpaca.data.enums import DataFeed as AlpacaFeed  # noqa: E402
from alpaca.data.historical import StockHistoricalDataClient  # noqa: E402
from alpaca.data.requests import StockBarsRequest  # noqa: E402
from alpaca.data.timeframe import TimeFrame, TimeFrameUnit  # noqa: E402


class ReplaySnapshot(SymbolSnapshot):
    """SymbolSnapshot whose session bars are supplied rather than derived.

    The live property converts every bar's timezone on each access, which is
    fine once per cycle but crippling across ~100k replayed cycles. The day's
    bars are already known here, so hand them over directly.
    """

    def __init__(self, symbol, bars, session, bid, ask, ts):
        super().__init__(symbol, bars, bid, ask, ts)
        self._session = session

    @property
    def session_bars(self):
        return self._session


def fetch(client, symbol: str, start, end, feed) -> List[Bar]:
    """Page through history; the API caps each response."""
    out: List[Bar] = []
    cursor = start
    while cursor < end:
        req = StockBarsRequest(
            symbol_or_symbols=symbol,
            timeframe=TimeFrame(config.BAR_TIMEFRAME_MINUTES, TimeFrameUnit.Minute),
            start=cursor, end=end, feed=feed, limit=10000,
        )
        bars = client.get_stock_bars(req).data.get(symbol, [])
        if not bars:
            break
        out.extend(Bar(b.timestamp, b.open, b.high, b.low, b.close, b.volume)
                   for b in bars)
        newest = bars[-1].timestamp
        if newest <= cursor:
            break
        cursor = newest + timedelta(minutes=config.BAR_TIMEFRAME_MINUTES)
    # De-duplicate and keep regular hours only.
    seen, rth = set(), []
    for b in sorted(out, key=lambda x: x.ts):
        if b.ts in seen:
            continue
        seen.add(b.ts)
        et = b.ts.astimezone(EASTERN)
        mins = et.hour * 60 + et.minute
        if 9 * 60 + 30 <= mins < 16 * 60:
            rth.append(b)
    return rth


def make_decider(name: str):
    if name == "rule":
        from decider_rule import RuleDecider
        return RuleDecider()
    if name == "orb":
        from decider_orb import ORBDecider
        return ORBDecider()
    if name == "meanrev":
        from decider_meanrev import MeanReversionDecider
        return MeanReversionDecider()
    if name == "scheduled":
        from decider_scheduled import ScheduledDecider, parse_schedule
        sched = parse_schedule(config.STRATEGY_SCHEDULE)
        subs = {n: make_decider(n) for n in sorted({s for _, _, s in sched})}
        return ScheduledDecider(subs, sched)
    raise ValueError(f"cannot backtest backend {name!r}")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--years", type=float, default=2.0)
    p.add_argument("--symbols", default=",".join(config.WATCHLIST))
    p.add_argument("--strategies", default="rule,orb,meanrev,scheduled")
    p.add_argument("--notional", type=float, default=config.FLAT_POSITION_NOTIONAL)
    p.add_argument("--out", default="logs/backtest_trades.json")
    args = p.parse_args()

    symbols = [s.strip().upper() for s in args.symbols.split(",") if s.strip()]
    names = [s.strip().lower() for s in args.strategies.split(",") if s.strip()]

    client = StockHistoricalDataClient(config.ALPACA_API_KEY, config.ALPACA_SECRET_KEY)
    feed = AlpacaFeed.SIP if config.ALPACA_DATA_FEED == "sip" else AlpacaFeed.IEX
    end = datetime.now(timezone.utc) - timedelta(minutes=20)
    start = end - timedelta(days=int(365 * args.years))

    print(f"fetching {args.years}y of bars for {', '.join(symbols)} ...")
    bars_by_symbol: Dict[str, List[Bar]] = {}
    for sym in symbols:
        bars_by_symbol[sym] = fetch(client, sym, start, end, feed)
        print(f"  {sym}: {len(bars_by_symbol[sym]):,} bars")

    # Group by ET session date.
    by_day: Dict[str, Dict[str, List[Bar]]] = defaultdict(dict)
    for sym, bars in bars_by_symbol.items():
        day_map: Dict[str, List[Bar]] = defaultdict(list)
        for b in bars:
            day_map[b.ts.astimezone(EASTERN).strftime("%Y-%m-%d")].append(b)
        for day, lst in day_map.items():
            by_day[day][sym] = lst

    days = sorted(by_day)
    print(f"  {len(days)} trading sessions\n")

    deciders = {n: make_decider(n) for n in names}
    books = {n: ShadowBook(n, args.notional) for n in names}
    gate = RiskGate()
    trades: Dict[str, list] = defaultdict(list)

    # Rolling history per symbol so RSI/ATR work from the opening bell.
    history: Dict[str, List[Bar]] = {s: [] for s in symbols}

    for di, day in enumerate(days):
        if di % 25 == 0:
            print(f"  replaying {day} ({di}/{len(days)}) ...")
        day_bars = by_day[day]
        # Longest session among the symbols present this day.
        length = max((len(v) for v in day_bars.values()), default=0)

        for i in range(length):
            snaps = []
            prices: Dict[str, float] = {}
            for sym in symbols:
                todays = day_bars.get(sym)
                if not todays or i >= len(todays):
                    continue
                session = todays[:i + 1]
                window = (history[sym] + session)[-config.BAR_LOOKBACK:]
                if len(window) < config.MIN_BARS_FOR_INDICATORS:
                    continue
                last = session[-1]
                # Model the quote as the close plus a token spread; the real
                # gate's spread test cannot be reproduced from bars alone.
                half = last.close * 0.00005
                snaps.append(ReplaySnapshot(
                    sym, window, session, last.close - half, last.close + half, last.ts
                ))
                prices[sym] = last.close

            if not snaps:
                continue

            et = snaps[0].bars[-1].ts.astimezone(EASTERN)
            mins_to_close = (16 * 60) - (et.hour * 60 + et.minute)

            for name, decider in deciders.items():
                book = books[name]
                # End of day: flatten, matching the live rule.
                if mins_to_close <= config.FLATTEN_BEFORE_CLOSE_MIN:
                    for ev in book.flatten(prices):
                        trades[name].append({**ev, "day": day})
                    continue

                account = {
                    "equity": 100_000.0, "cash": 100_000.0, "daily_pl": 0.0,
                    "daytrade_count": 0, "minutes_to_close": float(mins_to_close),
                }
                ctx = build_context(snaps, account, book.as_context_positions())
                try:
                    result = decider.decide(ctx)
                except Exception:
                    log.exception("%s failed on %s", name, day)
                    continue

                # Apply the SAME risk gate as live, so backtested trades are
                # only those the real system would actually have taken.
                approved: List[TradeDecision] = []
                n_new = 0
                for d in result.decisions:
                    verdict = gate.evaluate(
                        d, account, book.as_context_positions(), n_new
                    )
                    if verdict.approved:
                        approved.append(d)
                        if d.action == "buy":
                            n_new += 1
                for ev in book.apply(approved, prices):
                    trades[name].append({**ev, "day": day})

        # Roll today into history for tomorrow's indicators.
        for sym in symbols:
            if sym in day_bars:
                history[sym] = (history[sym] + day_bars[sym])[-config.BAR_LOOKBACK:]

    # ---- report ----------------------------------------------------------
    print("\n" + "=" * 78)
    print(f"BACKTEST — {len(days)} sessions, {', '.join(symbols)}")
    print("=" * 78)
    rows = []
    for name in names:
        ts = [t for t in trades[name] if "pnl" in t]
        wins = [t["pnl"] for t in ts if t["pnl"] > 0]
        losses = [t["pnl"] for t in ts if t["pnl"] <= 0]
        n = len(ts)
        if not n:
            continue
        wr = len(wins) / n
        aw = sum(wins) / len(wins) if wins else 0.0
        al = abs(sum(losses) / len(losses)) if losses else 0.0
        exp = wr * aw - (1 - wr) * al
        biggest = max((abs(t["pnl"]) for t in ts), default=0.0)
        share = biggest / max(sum(abs(t["pnl"]) for t in ts), 1e-9)
        rows.append((exp, name, n, wr, aw, al, sum(t["pnl"] for t in ts), share))

    rows.sort(reverse=True)
    print(f"  {'strategy':10s} {'expect':>8s} {'total':>10s} {'trades':>7s} "
          f"{'win%':>6s} {'avg W':>8s} {'avg L':>8s} {'top%':>6s}")
    print("  " + "-" * 74)
    for exp, name, n, wr, aw, al, total, share in rows:
        print(f"  {name:10s} {exp:+8.2f} {total:+10.2f} {n:7d} {100*wr:5.1f}% "
              f"{aw:+8.2f} {-al:+8.2f} {100*share:5.0f}%")

    print("\n  top% = share of all P&L moved by the single largest trade.")
    print("  High values mean the result rests on one or two lucky fills.")
    print("  Expectancy must clear zero AFTER slippage to be worth trading.")

    out = {n: trades[n] for n in names}
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(out, f)
    print(f"\n  trade-level detail -> {args.out}")


if __name__ == "__main__":
    main()
