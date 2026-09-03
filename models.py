"""Typed contracts between the pipeline stages."""

from __future__ import annotations

from typing import List, Literal, Optional

from pydantic import BaseModel, Field


class TradeDecision(BaseModel):
    """Claude's structured thesis for one symbol. This is a *proposal* —
    the deterministic risk gate decides whether it becomes an order."""

    symbol: str
    action: Literal["buy", "sell", "close", "hold"]
    confidence: float = Field(ge=0.0, le=1.0, description="0-1 conviction in the thesis")
    thesis: str = Field(description="2-4 sentence reasoning for the decision")
    invalidation: str = Field(description="Concrete condition that would prove the thesis wrong")
    suggested_notional: float = Field(ge=0.0, description="Suggested position size in dollars")
    # 390 minutes is one trading day. The cap was originally that, from an
    # intraday-only assumption; the insider strategy holds 21 trading days, so
    # the ceiling is now 30 trading days. Anything beyond that is a bug, not a
    # thesis. Strategies that hold overnight must also set
    # `holds_overnight = True` on the decider, or the end-of-day flatten will
    # close the position the evening it opens.
    time_horizon_minutes: int = Field(
        ge=5, le=390 * 30, description="Expected holding period, minutes"
    )


class CycleDecisions(BaseModel):
    """Claude's full output for one decision cycle."""

    market_read: str = Field(description="1-3 sentence read of the current tape")
    decisions: List[TradeDecision]


class GateResult(BaseModel):
    """Outcome of the deterministic risk gate for one decision."""

    approved: bool
    reasons: List[str] = []
    capped_notional: Optional[float] = None  # gate may shrink but never grow size
