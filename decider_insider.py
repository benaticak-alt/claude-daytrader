"""Insider-buy decider — trades fresh SEC Form 4 open-market purchases.

The only signal in this project that survived a symbol- and period-matched
control (+$13.51/trade excess, t=2.21 on non-overlapping events). It is a
different animal from the intraday strategies:

    universe   ~80 symbols, not 5 — events are rare per name
    cadence    daily; a filing is news for days, not seconds
    trigger    an event, not a price pattern
    hold       21 trading days, or ±1.5 ATR

NO MODEL. The trained classifier scored AUC 0.473 — below random — and every
probability threshold performed WORSE than simply taking every event. There is
no skill in selecting among insider buys, so this takes them all and says so
rather than dressing a coin flip in a confidence score.

WHY THIS RUNS IN SHADOW: the backtest could only see currently-listed symbols,
so insiders who bought companies that later delisted are invisible, and the
measured edge is biased upward by an unknown amount. Forward testing is the
only cure — it trades what exists now and finds out. Do not promote this to
live capital on the strength of the backtest alone.

FILING DATE IS THE ONLY TRADEABLE DATE. Form 4 allows two business days after
the transaction, so `filed_today` / recency is keyed on the filing, never the
transaction date.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime
from typing import Dict, List, Optional

import config
from models import CycleDecisions, TradeDecision

log = logging.getLogger(__name__)

TP_ATR = 1.5
SL_ATR = 1.5
HOLD_TRADING_DAYS = 21
# A filing is actionable for a few days — the disclosure is the event, and the
# documented drift plays out over weeks, so there is no need to catch the tick.
MAX_FILING_AGE_DAYS = 3


class InsiderDecider:
    """Same interface as the other deciders: .decide(context_json)."""

    # The end-of-day flatten exists because the intraday strategies size for
    # ~0.1% moves and must never eat an overnight gap. This strategy's edge is
    # measured over 21 TRADING DAYS — flattening it nightly would destroy the
    # very thing being tested. Overnight gap risk is accepted here by design,
    # and it is priced into the ±1.5 ATR barriers the effect was measured with.
    holds_overnight = True

    def __init__(self) -> None:
        # symbol -> {"atr": entry ATR, "ts": entry time iso}; lost on restart,
        # after which positions fall back to barrier management only.
        self._entries: Dict[str, dict] = {}

    # ------------------------------------------------------------------
    def decide(self, context_json: str) -> CycleDecisions:
        ctx = json.loads(context_json)
        rows = {r["symbol"]: r for r in ctx.get("symbols", [])}
        positions = ctx.get("open_positions", [])
        held = {p["symbol"] for p in positions}
        as_of = ctx.get("as_of_et")

        for sym in list(self._entries):
            if sym not in held:
                self._entries.pop(sym, None)

        decisions: List[TradeDecision] = []
        for pos in positions:
            d = self._maybe_exit(pos, rows.get(pos["symbol"]), as_of)
            if d:
                decisions.append(d)

        candidates = []
        for sym, row in rows.items():
            if sym in held:
                continue
            scored = self._score(row)
            if scored:
                candidates.append(scored)

        # Rank by conviction so the position cap spends itself on the strongest
        # events, but take every one that qualifies — there is no selection model.
        candidates.sort(key=lambda c: c[0], reverse=True)
        for confidence, sym, note in candidates[:2]:
            row = rows[sym]
            decisions.append(TradeDecision(
                symbol=sym,
                action="buy",
                confidence=confidence,
                thesis=f"Insider open-market buy: {note}",
                invalidation=(
                    f"-{SL_ATR:g} ATR from entry, +{TP_ATR:g} ATR target, or "
                    f"{HOLD_TRADING_DAYS} trading days elapsed."
                ),
                suggested_notional=config.FLAT_POSITION_NOTIONAL,
                time_horizon_minutes=HOLD_TRADING_DAYS * 390,
            ))
            self._entries[sym] = {"atr": row.get("atr_14"), "ts": as_of}

        with_events = sum(
            1 for r in rows.values()
            if (r.get("insider_form4") or {}).get("filings_today")
        )
        return CycleDecisions(
            market_read=(
                f"Insider: {len(rows)} symbols, {with_events} with a filing today, "
                f"{len(candidates)} actionable buy events."
            ),
            decisions=decisions,
        )

    # ------------------------------------------------------------------
    def _score(self, row: dict) -> Optional[tuple[float, str, str]]:
        ins = row.get("insider_form4")
        if not ins or ins.get("insider_data") == "unavailable":
            return None

        buys = ins.get("open_market_buys") or {}
        n = buys.get("transactions") or 0
        if n <= 0:
            return None

        # Only act on a RECENT filing. The summary window is 90 days by default,
        # so without this every symbol with any historical buy would look live
        # forever and the bot would re-enter the same stale event daily.
        recent = ins.get("filings_today") or 0
        most_recent = buys.get("most_recent")
        if not recent and not self._is_fresh(most_recent):
            return None

        insiders = buys.get("distinct_insiders") or 0
        usd = buys.get("total_usd") or 0
        cluster = bool(ins.get("cluster_buy"))
        csuite = bool(ins.get("c_suite_buy"))

        # Conviction ordering follows the measured effect sizes:
        # C-suite (+$36/trade) > director (+$31) > 10% owner (+$11); clusters
        # strongest of all but rare (n=19), so weighted but not decisive.
        confidence = 0.68
        if cluster:
            confidence += 0.10
        if csuite:
            confidence += 0.08
        if insiders >= 2:
            confidence += 0.03
        if usd >= 1_000_000:
            confidence += 0.02
        confidence = round(min(confidence, 0.90), 2)

        note = (
            f"{n} buy(s) by {insiders} insider(s), ${usd:,.0f} total"
            + (", CLUSTER" if cluster else "")
            + (", C-SUITE" if csuite else "")
            + (f", {recent} filed today" if recent else "")
        )
        return confidence, row["symbol"], note

    @staticmethod
    def _is_fresh(most_recent: Optional[str]) -> bool:
        """True if the newest buy FILING is within the actionable window."""
        if not most_recent:
            return False
        try:
            d = datetime.fromisoformat(str(most_recent)).date()
        except ValueError:
            return False
        return (datetime.now().date() - d).days <= MAX_FILING_AGE_DAYS

    # ------------------------------------------------------------------
    def _maybe_exit(self, pos: dict, row: Optional[dict],
                    as_of: Optional[str]) -> Optional[TradeDecision]:
        if row is None:
            return None
        last, entry = row.get("last"), pos.get("avg_entry")
        if not last or not entry:
            return None

        rec = self._entries.get(pos["symbol"], {})
        atr = rec.get("atr") or row.get("atr_14")
        if not atr:
            return None

        reason = None
        if last >= entry + TP_ATR * atr:
            reason = f"target: {last:.2f} >= {entry:.2f} + {TP_ATR:g} ATR"
        elif last <= entry - SL_ATR * atr:
            reason = f"stop: {last:.2f} <= {entry:.2f} - {SL_ATR:g} ATR"
        elif rec.get("ts") and as_of:
            try:
                days = (datetime.fromisoformat(as_of)
                        - datetime.fromisoformat(rec["ts"])).days
                if days >= HOLD_TRADING_DAYS * 1.45:      # trading -> calendar
                    reason = f"hold expired: {days} calendar days"
            except ValueError:
                pass

        if reason is None:
            return None
        return TradeDecision(
            symbol=pos["symbol"], action="close", confidence=0.70,
            thesis=f"Insider exit — {reason}.",
            invalidation="n/a — this is an exit",
            suggested_notional=0.0, time_horizon_minutes=5,
        )
