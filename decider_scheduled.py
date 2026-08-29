"""Time-of-day strategy router — run different strategies at different times.

Intraday character is not constant. Volume and volatility trace a U-curve:
directional at the open, choppy and thin through midday, directional again into
the close. One strategy across all of it is the wrong strategy for hours a day.

    09:30-11:00   orb        breakouts work while the tape is directional
    11:00-14:30   meanrev    fade stretches while it is range-bound
    14:30-15:40   rule       momentum as volume and direction return

THE HARD PART IS NOT ROUTING, IT IS OWNERSHIP.

If ORB opens NVDA at 10:00 and the clock rolls to 11:00, the mean-reversion
strategy now holds a position it never took and has no idea how to exit — its
exit rule ("has it reverted to VWAP?") is meaningless for a breakout trade.
Naive routing strands positions under a strategy that cannot manage them.

So entries and exits route differently:
  * ENTRIES  only from the strategy active for the current time.
  * EXITS    from whichever strategy OPENED the position, whatever time it is.

Ownership is tracked in memory. It is lost on restart, so an unknown position
falls back to the active strategy — degraded but never stranded, and the
end-of-day flatten is a backstop regardless.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime
from typing import Dict, List, Optional

import config
from data_feed import EASTERN
from models import CycleDecisions, TradeDecision

log = logging.getLogger(__name__)


def parse_schedule(spec: str) -> List[tuple[int, int, str]]:
    """'09:30-11:00=orb,11:00-14:30=meanrev' -> [(570, 660, 'orb'), ...]

    Times are ET minutes-since-midnight. Malformed entries are skipped loudly
    rather than silently dropping a chunk of the trading day.
    """
    out: List[tuple[int, int, str]] = []
    for chunk in spec.split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        try:
            window, name = chunk.split("=")
            start_s, end_s = window.split("-")
            sh, sm = (int(x) for x in start_s.split(":"))
            eh, em = (int(x) for x in end_s.split(":"))
            out.append((sh * 60 + sm, eh * 60 + em, name.strip().lower()))
        except (ValueError, AttributeError):
            log.error("bad schedule entry %r — skipping", chunk)
    return sorted(out)


class ScheduledDecider:
    """Routes to sub-deciders by time of day. Same .decide() interface."""

    def __init__(self, deciders: Dict[str, object], schedule: List[tuple[int, int, str]]):
        self.deciders = deciders
        self.schedule = schedule
        # symbol -> name of the strategy that opened it
        self._owner: Dict[str, str] = {}

    def active_strategy(self, now_et: Optional[datetime] = None) -> Optional[str]:
        now_et = now_et or datetime.now(EASTERN)
        minutes = now_et.hour * 60 + now_et.minute
        for start, end, name in self.schedule:
            if start <= minutes < end:
                return name if name in self.deciders else None
        return None

    def decide(self, context_json: str) -> CycleDecisions:
        ctx = json.loads(context_json)
        positions = ctx.get("open_positions", [])
        held = [p["symbol"] for p in positions]

        # Route on the snapshot's own timestamp, never the wall clock. They
        # coincide live, but in a backtest the wall clock is whenever the replay
        # runs — which once produced two years of "scheduled" results that were
        # really a single strategy applied to every bar.
        as_of = None
        raw = ctx.get("as_of_et")
        if raw:
            try:
                as_of = datetime.fromisoformat(raw)
            except ValueError:
                log.warning("unparseable as_of_et %r — falling back to wall clock", raw)
        active = self.active_strategy(as_of)

        # Forget positions we no longer hold (closed, or flattened at EOD).
        for symbol in list(self._owner):
            if symbol not in held:
                self._owner.pop(symbol, None)

        decisions: List[TradeDecision] = []
        reads: List[str] = []

        # --- EXITS: ask each owning strategy about its own positions ---------
        owners_needed = {
            self._owner.get(sym) or active
            for sym in held
        }
        owners_needed.discard(None)

        for name in owners_needed:
            decider = self.deciders.get(name)
            if decider is None:
                continue
            # Show this strategy only the positions it owns, so it does not try
            # to exit another strategy's trade using rules that do not apply.
            mine = [
                p for p in positions
                if (self._owner.get(p["symbol"]) or active) == name
            ]
            scoped = dict(ctx)
            scoped["open_positions"] = mine
            try:
                result = decider.decide(json.dumps(scoped))
            except Exception:
                log.exception("scheduled sub-decider %s failed on exits", name)
                continue
            for d in result.decisions:
                if d.action == "close":
                    decisions.append(d)

        # --- ENTRIES: only the strategy active right now ---------------------
        if active is None:
            reads.append("no strategy scheduled for this time — exits only")
        else:
            decider = self.deciders[active]
            try:
                result = decider.decide(context_json)
                reads.append(f"[{active}] {result.market_read}")
                for d in result.decisions:
                    if d.action in ("buy", "sell"):
                        decisions.append(d)
                        self._owner[d.symbol] = active
            except Exception:
                log.exception("scheduled sub-decider %s failed on entries", active)

        held_note = (
            "  holdings: "
            + ", ".join(f"{s}({self._owner.get(s, '?')})" for s in held)
        ) if held else ""

        return CycleDecisions(
            market_read=" | ".join(reads) + held_note if reads else f"exits only{held_note}",
            decisions=decisions,
        )
