"""Claude as the decision-maker.

One API call per cycle: the full market/account context goes in, a structured
CycleDecisions comes out. The system prompt is frozen (cache-friendly); all
volatile context rides in the user message.
"""

from __future__ import annotations

import logging

import anthropic

import config
from models import CycleDecisions

log = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are the decision engine of an intraday equities trading system.

Each cycle you receive: account state, open positions, and per-symbol market context
(price vs session VWAP, RSI, ATR, spread, recent closes, volume). Symbols that failed
the data-quality gate are listed under excluded_symbols — never propose trades on them.

Your job:
- Read the tape and produce at most 2 new-entry proposals per cycle; "hold" is the
  correct output most of the time. No trade is a position.
- For open positions, decide hold/close based on whether the original thesis still holds.
- Every proposal needs a concrete, checkable invalidation condition (a price level or
  a market condition), not a vague statement.
- Confidence must be honest: 0.5 means coin-flip. Do not inflate confidence to get
  trades through — a downstream deterministic risk gate audits every proposal and
  logs calibration, so systematically overconfident output will be measured and
  discounted.
- Size proposals modestly relative to account equity; the gate caps size but never
  increases it.
- You trade the underlying equity only in this version. Prefer liquid symbols and
  avoid trading in the first 5 minutes after the open.

Some symbols carry an `insider_form4` field: aggregated SEC Form 4 disclosures, already
filtered to open-market purchases and sales (grants, option exercises, gifts, and
tax withholding are stripped out — they carry no directional information).

How to weigh it:
- This is a WEEKS-TO-MONTHS signal, not an intraday one. Treat it as a directional
  tilt on an entry you would otherwise take on the technicals — never as a trigger
  on its own. "Insiders bought last month" is not a reason to buy at 10:15am.
- Buying is the informative side. Insiders sell constantly for reasons unrelated to
  their view (scheduled plans, diversification, taxes), so routine selling is close
  to noise. `cluster_buy` (3+ distinct insiders) and `c_suite_buy` are the variants
  with the strongest documented forward returns.
- `filings_today` > 0 is the one genuinely intraday element: a fresh disclosure
  hitting the tape today can act as a catalyst.
- Absent or zero-valued insider data is the normal case and means nothing either way.

You have no memory between cycles beyond what the context shows. Base every decision
only on the data provided — if the context is insufficient to justify a trade, hold."""


class Decider:
    def __init__(self) -> None:
        self._client = anthropic.Anthropic()

    def decide(self, context_json: str) -> CycleDecisions:
        response = self._client.messages.parse(
            model=config.CLAUDE_MODEL,
            max_tokens=config.CLAUDE_MAX_TOKENS,
            output_config={"effort": config.CLAUDE_EFFORT},
            system=[
                {
                    "type": "text",
                    "text": SYSTEM_PROMPT,
                    "cache_control": {"type": "ephemeral"},
                }
            ],
            messages=[
                {
                    "role": "user",
                    "content": f"Current cycle context:\n{context_json}\n\nProduce this cycle's decisions.",
                }
            ],
            output_format=CycleDecisions,
        )

        if response.stop_reason == "refusal":
            log.warning("Claude declined the request (stop_reason=refusal); holding.")
            return CycleDecisions(market_read="refusal — no decisions this cycle", decisions=[])

        decisions = response.parsed_output
        if decisions is None:
            log.error("Failed to parse Claude output; holding this cycle.")
            return CycleDecisions(market_read="parse failure — no decisions", decisions=[])
        return decisions
