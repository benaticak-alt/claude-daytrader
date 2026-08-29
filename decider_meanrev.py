"""VWAP mean reversion — the midday counterpart to breakout/momentum.

Intraday volume and volatility follow a U-curve: directional at the open,
choppy and low-volume through the middle of the session, directional again into
the close. Momentum strategies bleed in that middle stretch — price crosses
VWAP repeatedly and every crossing looks like a signal. Both Aug 6 losses were
exactly that: unconfirmed momentum entries into a flat tape, reverting within
minutes.

This takes the opposite side. When price is stretched below session VWAP and
momentum is exhausted, it fades the move and targets a return to VWAP.

    ENTRY   price MEANREV_MIN..MAX % BELOW session VWAP (stretched, not collapsing)
            AND RSI below MEANREV_RSI_MAX (exhausted)
            AND spread still tight (a widening spread means real trouble)

    EXIT    price returns to within MEANREV_TARGET_PCT of VWAP (target hit)
            OR price falls MEANREV_ATR_STOP ATRs below entry (thesis wrong)

The floor on the entry band matters as much as the ceiling: a stock only
slightly below VWAP has no edge to capture, and one collapsing 4% below it is
not mean-reverting, it is trending down. This trades the middle.
"""

from __future__ import annotations

import json
import logging
from typing import List, Optional

import config
from models import CycleDecisions, TradeDecision

log = logging.getLogger(__name__)


class MeanReversionDecider:
    """Same interface as the other deciders: .decide(context_json)."""

    def decide(self, context_json: str) -> CycleDecisions:
        ctx = json.loads(context_json)
        rows = {r["symbol"]: r for r in ctx.get("symbols", [])}
        positions = ctx.get("open_positions", [])
        held = {p["symbol"] for p in positions}

        decisions: List[TradeDecision] = []

        for pos in positions:
            exit_d = self._maybe_exit(pos, rows.get(pos["symbol"]))
            if exit_d:
                decisions.append(exit_d)

        candidates = []
        for symbol, row in rows.items():
            if symbol in held:
                continue
            scored = self._score_entry(row)
            if scored:
                candidates.append(scored)

        candidates.sort(key=lambda c: c[0], reverse=True)
        for confidence, symbol, note in candidates[:2]:
            decisions.append(
                TradeDecision(
                    symbol=symbol,
                    action="buy",
                    confidence=confidence,
                    thesis=f"VWAP mean reversion: {note}",
                    invalidation=(
                        f"Price falls {config.MEANREV_ATR_STOP:g} ATR below entry, "
                        "meaning this is a trend rather than a stretch."
                    ),
                    suggested_notional=config.FLAT_POSITION_NOTIONAL,
                    time_horizon_minutes=60,
                )
            )

        return CycleDecisions(
            market_read=(
                f"MeanRev: {len(rows)} symbols scanned, "
                f"{len(candidates)} stretched below VWAP."
            ),
            decisions=decisions,
        )

    # ------------------------------------------------------------------
    def _maybe_exit(self, pos: dict, row: Optional[dict]) -> Optional[TradeDecision]:
        if row is None:
            return None
        last = row.get("last")
        vwap_pct = row.get("pct_vs_vwap")
        if last is None or vwap_pct is None:
            return None

        reason = None
        # Target: price has come back to (or through) VWAP.
        if vwap_pct >= -config.MEANREV_TARGET_PCT:
            reason = f"reverted to {vwap_pct:+.2f}% of VWAP — target reached"
        else:
            atr = row.get("atr_14")
            entry = pos.get("avg_entry")
            if atr and entry:
                stop = entry - config.MEANREV_ATR_STOP * atr
                if last < stop:
                    reason = (
                        f"price {last:.2f} below the {config.MEANREV_ATR_STOP:g} ATR "
                        f"stop at {stop:.2f} — trending, not stretched"
                    )

        if reason is None:
            return None

        return TradeDecision(
            symbol=pos["symbol"],
            action="close",
            confidence=0.70,
            thesis=f"MeanRev exit: {reason}.",
            invalidation="n/a — this is an exit",
            suggested_notional=0.0,
            time_horizon_minutes=5,
        )

    # ------------------------------------------------------------------
    def _score_entry(self, row: dict) -> Optional[tuple[float, str, str]]:
        vwap_pct = row.get("pct_vs_vwap")
        rsi = row.get("rsi_14")
        spread = row.get("spread_pct")
        if vwap_pct is None or rsi is None:
            return None

        stretch = -vwap_pct  # positive when below VWAP
        if not (config.MEANREV_MIN_STRETCH_PCT <= stretch <= config.MEANREV_MAX_STRETCH_PCT):
            return None
        if rsi > config.MEANREV_RSI_MAX:
            return None
        # A widening spread during a drop signals genuine stress, not a stretch.
        if spread is not None and spread > config.MEANREV_MAX_SPREAD_PCT:
            return None

        # Deeper stretch and more exhausted RSI both raise conviction, but the
        # band caps how deep we are willing to go — beyond it, it is a downtrend.
        span = config.MEANREV_MAX_STRETCH_PCT - config.MEANREV_MIN_STRETCH_PCT
        depth = (stretch - config.MEANREV_MIN_STRETCH_PCT) / span if span > 0 else 0.5
        confidence = 0.64 + 0.10 * depth
        if rsi < 30:
            confidence += 0.06
        elif rsi < 34:
            confidence += 0.03
        confidence = round(min(confidence, 0.85), 2)

        note = (
            f"{stretch:.2f}% below session VWAP with RSI {rsi:.1f} "
            f"(band {config.MEANREV_MIN_STRETCH_PCT:g}-{config.MEANREV_MAX_STRETCH_PCT:g}%)"
        )
        return confidence, row["symbol"], note
