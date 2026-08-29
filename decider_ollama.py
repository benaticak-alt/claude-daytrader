"""Local-LLM decider via Ollama — free, no API key, no network egress.

Install Ollama from https://ollama.com, then pull a model:

    ollama pull qwen2.5:7b        # good instruction-following at 7B
    ollama pull llama3.1:8b       # alternative

Ollama supports JSON-schema-constrained output, so the same pydantic contract
(`CycleDecisions`) is enforced here as on the Claude path — which is the point:
it proves the *plumbing* handles structured decisions correctly.

IMPORTANT: a local model is not a substitute for Opus 5 as a *strategy*. Its
market judgment is materially worse, and validating it tells you nothing about
how the real decider would trade. Use this to shake out bugs, then switch back
to Claude for anything you intend to draw conclusions from.

Observed failure mode worth knowing: schema constraint guarantees the response
*parses*, not that it means anything. In testing, gemma4 produced a coherent
market read and a sensible symbol choice, but filled `invalidation` with raw
control tokens (`></decisions><channel|>...`). Nothing crashed — the pipeline
accepted it. Structurally valid, semantically junk. Do not mistake "it ran
without errors" for "the decisions were good."
"""

from __future__ import annotations

import json
import logging

import requests

import config
from models import CycleDecisions

log = logging.getLogger(__name__)

OLLAMA_URL = "http://localhost:11434/api/chat"

SYSTEM_PROMPT = """You are the decision engine of an intraday equities trading system.

You receive account state, open positions, and per-symbol market context (price vs
session VWAP, RSI, ATR, spread, recent closes, volume).

Rules:
- Propose at most 2 new entries per cycle. "hold" is usually correct — no trade is a position.
- Only buy. Never propose selling short.
- Never propose a symbol listed under excluded_symbols.
- Every proposal needs a concrete invalidation condition (a price level).
- Confidence must be honest: 0.5 means coin-flip. Do not inflate it.
- For open positions, decide whether to hold or close.

Respond with JSON matching the required schema."""


class OllamaDecider:
    def __init__(self, model: str | None = None) -> None:
        self.model = model or config.OLLAMA_MODEL

    def decide(self, context_json: str) -> CycleDecisions:
        payload = {
            "model": self.model,
            "stream": False,
            "format": CycleDecisions.model_json_schema(),
            "options": {"temperature": 0.3},
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": f"Current cycle context:\n{context_json}\n\nProduce this cycle's decisions.",
                },
            ],
        }
        try:
            resp = requests.post(OLLAMA_URL, json=payload, timeout=180)
            resp.raise_for_status()
            content = resp.json()["message"]["content"]
            return CycleDecisions.model_validate_json(content)
        except requests.RequestException as exc:
            log.error("Ollama request failed (is `ollama serve` running?): %s", exc)
        except (KeyError, ValueError) as exc:
            log.error("Ollama returned unusable output: %s", exc)

        return CycleDecisions(market_read="ollama unavailable — no decisions", decisions=[])
