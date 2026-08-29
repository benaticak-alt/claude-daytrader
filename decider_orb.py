"""Opening Range Breakout — free, deterministic, no API key.

The opening range is the high/low set in the first N minutes after the bell.
That window absorbs overnight order flow, so a decisive move beyond it on real
volume signals directional commitment rather than noise.

    ENTRY   opening range window has CLOSED
            AND price breaks above the range high
            AND volume on the breakout bar > ORB_VOLUME_MULT x the 20-bar average
            AND price above session VWAP
            AND not already extended more than ORB_MAX_EXTENSION_PCT past the high

    EXIT    price falls back inside the range (breakout failed)
            OR price drops ORB_ATR_STOP_MULT ATRs below entry (stop)
            (end-of-day flatten is handled by the risk gate and main loop)

Why this shape: the plain VWAP/RSI rule committed on a single bar with no
confirmation, and both Aug 6 losers reverted within minutes. Two guards here
target that directly — the volume test demands the move be real, and the
"back inside the range" exit cuts a failed breakout immediately rather than
waiting for RSI to roll over.

This decider is stateless, like the others: entry price comes from the
account's open positions, not from remembered state.
"""

from __future__ import annotations

import json
import logging
from typing import List, Optional

import config
from models import CycleDecisions, TradeDecision

log = logging.getLogger(__name__)


class ORBDecider:
    """Same interface as the other deciders: .decide(context_json)."""

    def decide(self, context_json: str) -> CycleDecisions:
        ctx = json.loads(context_json)
        rows = {r["symbol"]: r for r in ctx.get("symbols", [])}
        positions = ctx.get("open_positions", [])
        held = {p["symbol"] for p in positions}

        decisions: List[TradeDecision] = []

        # --- exits first, so a closed position frees a slot this same cycle ---
        for pos in positions:
            exit_decision = self._maybe_exit(pos, rows.get(pos["symbol"]))
            if exit_decision:
                decisions.append(exit_decision)

        # --- entries ---
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
                    thesis=f"Opening range breakout: {note}",
                    invalidation=(
                        "Price closes back inside the opening range, or falls "
                        f"{config.ORB_ATR_STOP_MULT:g} ATR below entry."
                    ),
                    suggested_notional=config.FLAT_POSITION_NOTIONAL,
                    time_horizon_minutes=120,
                )
            )

        scanned = len(rows)
        ready = sum(
            1 for r in rows.values()
            if (r.get("opening_range") or {}).get("complete")
        )
        return CycleDecisions(
            market_read=(
                f"ORB: {scanned} symbols scanned, {ready} with a completed opening "
                f"range, {len(candidates)} breaking out."
            ),
            decisions=decisions,
        )

    # ------------------------------------------------------------------
    def _maybe_exit(self, pos: dict, row: Optional[dict]) -> Optional[TradeDecision]:
        if row is None:
            return None  # no fresh data this cycle — leave the position alone
        last = row.get("last")
        rng = row.get("opening_range")
        if last is None or not rng:
            return None

        reason = None
        if last < rng["high"]:
            reason = (
                f"price {last:.2f} fell back inside the opening range "
                f"(high {rng['high']:.2f}) — breakout failed"
            )
        else:
            atr = row.get("atr_14")
            entry = pos.get("avg_entry")
            if atr and entry:
                stop = entry - config.ORB_ATR_STOP_MULT * atr
                if last < stop:
                    reason = (
                        f"price {last:.2f} below the {config.ORB_ATR_STOP_MULT:g} ATR "
                        f"stop at {stop:.2f} (entry {entry:.2f})"
                    )

        if reason is None:
            return None

        return TradeDecision(
            symbol=pos["symbol"],
            action="close",
            confidence=0.70,
            thesis=f"ORB exit: {reason}.",
            invalidation="n/a — this is an exit",
            suggested_notional=0.0,
            time_horizon_minutes=5,
        )

    # ------------------------------------------------------------------
    def _score_entry(self, row: dict) -> Optional[tuple[float, str, str]]:
        rng = row.get("opening_range")
        if not rng or not rng.get("complete"):
            return None  # range still forming — nothing to break out of yet

        last = row.get("last")
        vwap_pct = row.get("pct_vs_vwap")
        vol = row.get("volume_last_bar")
        avg_vol = row.get("avg_volume_20")
        if last is None or vwap_pct is None or not vol or not avg_vol:
            return None

        high = rng["high"]
        if last <= high:
            return None  # no breakout

        extension = 100.0 * (last - high) / high
        if extension > config.ORB_MAX_EXTENSION_PCT:
            return None  # move already made; entering here is chasing

        if vwap_pct <= 0:
            return None  # breaking out but below VWAP is a weak tape

        vol_ratio = vol / avg_vol
        if vol_ratio < config.ORB_VOLUME_MULT:
            return None  # unconfirmed — this is the guard the old rule lacked

        # Confidence: strong volume is the main signal; a *fresh* breakout beats
        # an extended one, so the extension term rewards being early.
        confidence = 0.64
        if vol_ratio >= 2.5:
            confidence += 0.10
        elif vol_ratio >= 2.0:
            confidence += 0.07
        elif vol_ratio >= 1.75:
            confidence += 0.04
        if extension <= 0.5:
            confidence += 0.05
        elif extension <= 1.0:
            confidence += 0.02
        confidence = round(min(confidence, 0.85), 2)

        note = (
            f"broke {extension:.2f}% above the {rng['bars']}-bar range high "
            f"{high:.2f} on {vol_ratio:.1f}x average volume, {vwap_pct:+.2f}% vs VWAP"
        )
        return confidence, row["symbol"], note
