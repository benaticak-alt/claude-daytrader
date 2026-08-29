"""Deterministic risk gate — the boundary between Claude and the broker.

Every check here is plain code with values from config.py. Claude's output is
treated as an untrusted proposal: the gate can reject or shrink it, never the
reverse. If any check cannot be evaluated (e.g. account fetch failed), the
gate rejects — fail closed.
"""

from __future__ import annotations

import logging
from typing import List

import config
from models import GateResult, TradeDecision

log = logging.getLogger(__name__)


class RiskGate:
    def evaluate(
        self,
        decision: TradeDecision,
        account: dict,
        open_positions: List[dict],
        approved_this_cycle: int,
    ) -> GateResult:
        reasons: List[str] = []

        if decision.action == "hold":
            return GateResult(approved=False, reasons=["hold — nothing to execute"])

        # 0. Kill switch — a human touched the file, stop everything.
        if config.KILL_SWITCH_FILE.exists():
            return GateResult(approved=False, reasons=["KILL_SWITCH file present"])

        # 1. Fail closed on missing account data.
        equity = account.get("equity")
        daily_pl = account.get("daily_pl")
        if equity is None or daily_pl is None:
            return GateResult(approved=False, reasons=["account state unavailable — fail closed"])

        # 2. Daily loss limit.
        if daily_pl <= -config.MAX_DAILY_LOSS:
            reasons.append(
                f"daily loss limit breached ({daily_pl:.2f} <= -{config.MAX_DAILY_LOSS})"
            )

        # Closing an existing position is allowed even when entry checks fail —
        # reducing risk should never be blocked by risk limits.
        if decision.action == "close":
            held = any(p["symbol"] == decision.symbol for p in open_positions)
            if not held:
                return GateResult(approved=False, reasons=[f"no open position in {decision.symbol}"])
            return GateResult(approved=True, reasons=["close of existing position"])

        # --- Entry checks (buy/sell) ---

        # 3. Confidence floor.
        if decision.confidence < config.MIN_CONFIDENCE:
            reasons.append(
                f"confidence {decision.confidence:.2f} below floor {config.MIN_CONFIDENCE}"
            )

        # 4. Concurrent position cap (existing + already approved this cycle).
        if len(open_positions) + approved_this_cycle >= config.MAX_CONCURRENT_POSITIONS:
            reasons.append(
                f"position cap reached ({len(open_positions)} open + {approved_this_cycle} pending)"
            )

        # 5. No doubling into an existing position.
        if any(p["symbol"] == decision.symbol for p in open_positions):
            reasons.append(f"already holding {decision.symbol}")

        # 6. PDT compliance: under $25k equity, cap round-trip day trades.
        if equity < config.PDT_MIN_EQUITY:
            day_trades = account.get("daytrade_count")
            if day_trades is None:
                reasons.append("daytrade count unavailable — fail closed under PDT threshold")
            elif day_trades >= config.PDT_MAX_DAY_TRADES:
                reasons.append(
                    f"PDT limit: {day_trades} day trades used, equity below "
                    f"${config.PDT_MIN_EQUITY:,.0f}"
                )

        # 7. Short selling disabled in v1 — long-only keeps assignment/borrow
        #    complexity out until the model's long-side calibration is proven.
        if decision.action == "sell":
            reasons.append("short entries disabled (long-only v1)")

        # 8. No new entries near the close. An overnight gap is a risk this
        #    strategy's sizing never priced in. Unknown time-to-close fails
        #    closed rather than assuming there's plenty of session left.
        mins = account.get("minutes_to_close")
        if mins is None:
            reasons.append("time-to-close unknown — fail closed")
        elif mins <= config.NO_ENTRY_BEFORE_CLOSE_MIN:
            reasons.append(
                f"too close to the bell ({mins:.0f}min left, "
                f"cutoff {config.NO_ENTRY_BEFORE_CLOSE_MIN}min)"
            )

        if reasons:
            return GateResult(approved=False, reasons=reasons)

        # 9. Sizing. The deterministic layer owns this outright.
        sized = self._target_notional(decision, account)
        if sized < 1.0:
            return GateResult(approved=False, reasons=["computed size below $1"])

        if decision.suggested_notional > sized:
            log.info("%s requested $%.0f, sized to $%.0f",
                     decision.symbol, decision.suggested_notional, sized)

        return GateResult(approved=True, reasons=["all checks passed"], capped_notional=sized)

    # ------------------------------------------------------------------
    def _target_notional(self, decision: TradeDecision, account: dict) -> float:
        """Compute position size deterministically.

        `decision.suggested_notional` is ADVISORY ONLY — it is logged so we can
        analyse what the decider wanted, but it never drives execution. A model
        (or a buggy rule) must not be able to influence its own position size;
        that is the whole point of having a gate.

        Two modes, per config.CONFIDENCE_SIZING:
          off — flat notional, so every trade is comparable for calibration.
          on  — equity * BASE_POSITION_PCT * a confidence multiplier.

        Both are then floored by the hard per-position ceiling and by half of
        available cash. Missing equity or cash yields 0, so the caller rejects.
        """
        if not config.CONFIDENCE_SIZING:
            target = config.FLAT_POSITION_NOTIONAL
        else:
            equity = account.get("equity")
            if equity is None:
                return 0.0
            # Map confidence from the gate's floor..1.0 onto the multiplier range.
            span = 1.0 - config.MIN_CONFIDENCE
            conf_norm = (decision.confidence - config.MIN_CONFIDENCE) / span if span > 0 else 1.0
            conf_norm = max(0.0, min(1.0, conf_norm))
            mult = config.SIZE_MIN_MULT + conf_norm * (config.SIZE_MAX_MULT - config.SIZE_MIN_MULT)
            target = equity * config.BASE_POSITION_PCT * mult

        # Never more than half the cash on one position. Unknown cash fails closed.
        cash = account.get("cash")
        if cash is None:
            log.warning("cash unavailable — cannot size safely, failing closed")
            return 0.0

        return max(0.0, min(target, config.MAX_POSITION_NOTIONAL, cash * config.MAX_CASH_FRACTION))
