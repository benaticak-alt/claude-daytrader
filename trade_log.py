"""Append-only JSONL decision log.

Every cycle logs: the context hash, Claude's full output (thesis + confidence),
the gate verdicts, and any order IDs. This is the dataset for later calibration
analysis (win rate by confidence bucket, thesis quality, gate rejection reasons)
— the same discipline as logging signals on the Polymarket bot.
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone

import config
from models import CycleDecisions, GateResult, TradeDecision


def _ensure_dir() -> None:
    config.LOG_DIR.mkdir(parents=True, exist_ok=True)


def log_cycle(
    context_json: str,
    cycle: CycleDecisions,
    gate_results: list[tuple[TradeDecision, GateResult, str | None]],
    shadow: dict | None = None,
) -> None:
    """gate_results: (decision, gate_result, order_id_or_none) per decision.

    `shadow` holds what the non-executing parallel strategies would have done,
    keyed by strategy name — the head-to-head comparison data.
    """
    _ensure_dir()
    record = {
        "ts": datetime.now(timezone.utc).isoformat(),
        # Which engine produced these decisions — essential for comparing the
        # Claude runs against the rule baseline in calibration.py.
        "backend": config.DECIDER_BACKEND,
        "context_sha1": hashlib.sha1(context_json.encode()).hexdigest()[:12],
        "market_read": cycle.market_read,
        "decisions": [
            {
                **d.model_dump(),
                "gate_approved": g.approved,
                "gate_reasons": g.reasons,
                "capped_notional": g.capped_notional,
                "order_id": order_id,
            }
            for d, g, order_id in gate_results
        ],
    }
    if shadow:
        record["shadow"] = shadow
    with open(config.DECISION_LOG, "a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")

    # Keep the full context separately (bigger, but needed to replay decisions)
    ctx_file = config.LOG_DIR / f"context_{record['context_sha1']}.json"
    if not ctx_file.exists():
        ctx_file.write_text(context_json, encoding="utf-8")
