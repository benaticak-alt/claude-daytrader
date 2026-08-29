"""Tests for time-of-day strategy routing and mean reversion.

The interesting case is the handoff: a position opened by one strategy while a
different strategy is active. Naive routing strands it under a strategy whose
exit rules do not apply to it.

    python test_scheduled.py
"""

from __future__ import annotations

import json
import sys
from datetime import datetime

import config
from data_feed import EASTERN
from decider_meanrev import MeanReversionDecider
from decider_orb import ORBDecider
from decider_rule import RuleDecider
from decider_scheduled import ScheduledDecider, parse_schedule

PASS, FAIL = "PASS", "FAIL"
results: list[tuple[str, str, str]] = []


def check(name: str, cond: bool, detail: str = "") -> None:
    results.append((PASS if cond else FAIL, name, detail))


def at(hhmm: str) -> datetime:
    h, m = (int(x) for x in hhmm.split(":"))
    return datetime.now(EASTERN).replace(hour=h, minute=m, second=0, microsecond=0)


# --- schedule parsing ------------------------------------------------------

sched = parse_schedule("09:30-11:00=orb,11:00-14:30=meanrev,14:30-15:40=rule")
check("schedule parses", len(sched) == 3, str(sched))

bad = parse_schedule("09:30-11:00=orb,GARBAGE,11:00-14:30=meanrev")
check("malformed entry skipped, rest kept", len(bad) == 2, str(bad))

router = ScheduledDecider(
    {"orb": ORBDecider(), "meanrev": MeanReversionDecider(), "rule": RuleDecider()},
    sched,
)

check("09:45 -> orb", router.active_strategy(at("09:45")) == "orb")
check("12:00 -> meanrev", router.active_strategy(at("12:00")) == "meanrev")
check("15:00 -> rule", router.active_strategy(at("15:00")) == "rule")
check("08:00 -> none (pre-market)", router.active_strategy(at("08:00")) is None)
check("15:50 -> none (EOD window)", router.active_strategy(at("15:50")) is None)


# --- mean reversion entry band --------------------------------------------

def row(**kw) -> dict:
    base = {
        "symbol": "NVDA", "last": 100.0, "pct_vs_vwap": -1.2, "rsi_14": 32.0,
        "atr_14": 1.0, "spread_pct": 0.05, "volume_last_bar": 1e6,
        "avg_volume_20": 1e6,
        "opening_range": {"high": 105.0, "low": 99.0, "bars": 3, "complete": True},
    }
    base.update(kw)
    return base


def ctx(rows, positions=None, at_time: str | None = None) -> str:
    """`at_time` stamps the snapshot so routing is deterministic.

    Without it the router falls back to the wall clock and these tests pass or
    fail depending on the hour they are run — which is exactly the bug that let
    a two-year backtest silently use one strategy for every bar.
    """
    payload = {
        "account": {"equity": 100_000.0, "cash": 100_000.0},
        "open_positions": positions or [],
        "symbols": rows, "excluded_symbols": [],
    }
    if at_time:
        payload["as_of_et"] = at(at_time).isoformat()
    return json.dumps(payload)


mr = MeanReversionDecider()
buys = lambda r: [d for d in r.decisions if d.action == "buy"]
closes = lambda r: [d for d in r.decisions if d.action == "close"]

check("stretched + oversold enters", len(buys(mr.decide(ctx([row()])))) == 1)
check("too shallow refused", not buys(mr.decide(ctx([row(pct_vs_vwap=-0.3)]))),
      "0.3% below VWAP — no edge")
check("too deep refused", not buys(mr.decide(ctx([row(pct_vs_vwap=-4.0)]))),
      "4% below VWAP — trending, not stretched")
check("RSI not exhausted refused", not buys(mr.decide(ctx([row(rsi_14=55.0)]))))
check("wide spread refused", not buys(mr.decide(ctx([row(spread_pct=1.2)]))),
      "widening spread = real stress")
check("above VWAP refused", not buys(mr.decide(ctx([row(pct_vs_vwap=+1.0)]))))

deep = mr.decide(ctx([row(pct_vs_vwap=-2.2, rsi_14=28.0)]))
shallow = mr.decide(ctx([row(pct_vs_vwap=-0.7, rsi_14=37.0)]))
check("deeper stretch scores higher",
      buys(deep)[0].confidence > buys(shallow)[0].confidence,
      f"{buys(deep)[0].confidence} vs {buys(shallow)[0].confidence}")

pos = [{"symbol": "NVDA", "avg_entry": 100.0, "qty": 10}]
check("reverting to VWAP exits",
      len(closes(mr.decide(ctx([row(pct_vs_vwap=-0.05)], pos)))) == 1)
check("ATR stop exits",
      len(closes(mr.decide(ctx([row(last=98.0, pct_vs_vwap=-2.0)], pos)))) == 1,
      "1.5 ATR below 100.0 entry")


# --- THE HANDOFF: ownership across a strategy switch -----------------------

r = ScheduledDecider(
    {"orb": ORBDecider(), "meanrev": MeanReversionDecider()},
    parse_schedule("09:30-11:00=orb,11:00-14:30=meanrev"),
)
# Pretend ORB opened NVDA during its window.
r._owner["NVDA"] = "orb"

# Now it is midday: meanrev is active, but NVDA is still ORB's position.
# Price has fallen back inside ORB's opening range -> ORB should close it,
# even though meanrev is the active strategy.
held = [{"symbol": "NVDA", "avg_entry": 100.0, "qty": 10}]
out = r.decide(ctx([row(symbol="NVDA", last=101.0, pct_vs_vwap=-1.2,
                        opening_range={"high": 105.0, "low": 99.0,
                                       "bars": 3, "complete": True})],
                   held, at_time="12:00"))   # midday: meanrev is active
orb_exit = [d for d in out.decisions if d.action == "close" and d.symbol == "NVDA"]
check("owning strategy exits its own position across a switch",
      len(orb_exit) == 1,
      orb_exit[0].thesis[:60] if orb_exit else "position stranded!")

check("ownership shown in market read", "orb" in out.market_read,
      out.market_read[:80])

# Ownership is forgotten once the position is gone. Stamped pre-market so no
# strategy is active and nothing re-enters the symbol on this same call.
r.decide(ctx([row()], [], at_time="08:00"))
check("ownership cleared after close", "NVDA" not in r._owner)

# Entries only ever come from the active strategy.
r2 = ScheduledDecider(
    {"orb": ORBDecider(), "meanrev": MeanReversionDecider()},
    parse_schedule("00:00-23:59=meanrev"),
)
out2 = r2.decide(ctx([row()], at_time="12:00"))
entries = [d for d in out2.decisions if d.action == "buy"]
check("entries come from the active strategy only",
      len(entries) == 1 and r2._owner.get(entries[0].symbol) == "meanrev",
      f"owner={r2._owner}")

# Outside every window: exits still work, no new entries.
r3 = ScheduledDecider({"meanrev": MeanReversionDecider()},
                      parse_schedule("09:30-10:00=meanrev"))
out3 = r3.decide(ctx([row()], at_time="13:00"))   # deterministically outside
check("no entries outside scheduled windows",
      not [d for d in out3.decisions if d.action == "buy"])

# The regression that made a two-year backtest meaningless: routing must follow
# the snapshot, not the wall clock.
r4 = ScheduledDecider(
    {"orb": ORBDecider(), "meanrev": MeanReversionDecider(), "rule": RuleDecider()},
    sched,
)
reads = {t: r4.decide(ctx([], at_time=t)).market_read for t in ("09:45", "12:00", "15:00")}
check("routes by snapshot time, not wall clock",
      "[orb]" in reads["09:45"] and "[meanrev]" in reads["12:00"]
      and "[rule]" in reads["15:00"],
      " / ".join(f"{k}->{v[:12]}" for k, v in reads.items()))


width = max(len(n) for _, n, _ in results)
failures = sum(1 for s, _, _ in results if s == FAIL)
for status, name, detail in results:
    print(f"  [{status}] {name:<{width}}  {detail}")
print(f"\n{len(results) - failures}/{len(results)} passed")
sys.exit(1 if failures else 0)
