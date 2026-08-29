"""Rule-based decider — zero cost, no API key, fully deterministic.

Two jobs:

1. **Plumbing validation.** Exercises every stage of the pipeline (data gate,
   risk gate, executor, logging) without spending a cent or depending on any
   external model. If the bot crashes, misroutes an order, or the gate
   misbehaves, you find out here for free.

2. **The control group.** This is a deliberately dumb VWAP + RSI rule. If the
   Claude decider cannot beat it on the same watchlist over the same period,
   then the LLM is adding nothing and you have your answer. Every claim of
   "the model has edge" is meaningless without a baseline to beat, and this
   is that baseline.

Deliberately simple and transparent — the point is a floor to measure against,
not a good strategy.
"""

from __future__ import annotations

import json
import logging
from typing import List

from models import CycleDecisions, TradeDecision

log = logging.getLogger(__name__)


class RuleDecider:
    """Same interface as Decider: .decide(context_json) -> CycleDecisions"""

    # Entry: trending above VWAP with momentum but not yet overbought.
    RSI_ENTRY_MIN = 52.0
    RSI_ENTRY_MAX = 68.0
    MIN_PCT_ABOVE_VWAP = 0.05
    MAX_PCT_ABOVE_VWAP = 1.50   # too extended = chasing

    # Exit: momentum gone or trend broken.
    RSI_EXIT = 45.0

    def decide(self, context_json: str) -> CycleDecisions:
        ctx = json.loads(context_json)
        decisions: List[TradeDecision] = []

        held = {p["symbol"] for p in ctx.get("open_positions", [])}

        # --- exits first: free up slots before considering entries ---
        for pos in ctx.get("open_positions", []):
            sym = pos["symbol"]
            row = next((s for s in ctx.get("symbols", []) if s["symbol"] == sym), None)
            if row is None:
                continue  # no fresh data this cycle; leave the position alone
            rsi = row.get("rsi_14")
            pct_vwap = row.get("pct_vs_vwap")
            if rsi is None or pct_vwap is None:
                continue
            if rsi < self.RSI_EXIT or pct_vwap < 0:
                decisions.append(
                    TradeDecision(
                        symbol=sym,
                        action="close",
                        confidence=0.70,
                        thesis=(
                            f"Baseline exit rule: RSI {rsi:.1f} "
                            f"({'below ' + str(self.RSI_EXIT) if rsi < self.RSI_EXIT else 'ok'}), "
                            f"price {pct_vwap:+.2f}% vs VWAP. Trend/momentum condition broken."
                        ),
                        invalidation="n/a — this is an exit",
                        suggested_notional=0.0,
                        time_horizon_minutes=5,
                    )
                )

        # --- entries ---
        candidates = []
        for row in ctx.get("symbols", []):
            sym = row["symbol"]
            if sym in held:
                continue
            rsi = row.get("rsi_14")
            pct_vwap = row.get("pct_vs_vwap")
            if rsi is None or pct_vwap is None:
                continue
            if not (self.RSI_ENTRY_MIN <= rsi <= self.RSI_ENTRY_MAX):
                continue
            if not (self.MIN_PCT_ABOVE_VWAP <= pct_vwap <= self.MAX_PCT_ABOVE_VWAP):
                continue

            # Mechanical confidence: strongest mid-band, tapering at the edges.
            band_center = (self.RSI_ENTRY_MIN + self.RSI_ENTRY_MAX) / 2
            band_half = (self.RSI_ENTRY_MAX - self.RSI_ENTRY_MIN) / 2
            closeness = 1.0 - abs(rsi - band_center) / band_half
            confidence = round(0.62 + 0.12 * closeness, 2)

            candidates.append((confidence, sym, rsi, pct_vwap))

        candidates.sort(reverse=True)
        for confidence, sym, rsi, pct_vwap in candidates[:2]:
            decisions.append(
                TradeDecision(
                    symbol=sym,
                    action="buy",
                    confidence=confidence,
                    thesis=(
                        f"Baseline entry rule: price {pct_vwap:+.2f}% above session VWAP "
                        f"with RSI {rsi:.1f} in the {self.RSI_ENTRY_MIN:.0f}-"
                        f"{self.RSI_ENTRY_MAX:.0f} momentum band. No discretionary input."
                    ),
                    invalidation=(
                        f"Close below session VWAP, or RSI falling under {self.RSI_EXIT:.0f}."
                    ),
                    suggested_notional=1000.0,
                    time_horizon_minutes=60,
                )
            )

        return CycleDecisions(
            market_read=(
                f"Rule baseline: {len(ctx.get('symbols', []))} symbols scanned, "
                f"{len(candidates)} passed the entry filter."
            ),
            decisions=decisions,
        )
