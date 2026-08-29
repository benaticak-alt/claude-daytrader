"""Adversarial tests for the risk gate — the component that must never fail open.

The gate is the only thing standing between a bad decision (from Claude, a
local model, or a buggy rule) and a real order. These tests feed it decisions
that SHOULD be refused and assert that they are.

    python test_gate.py

No API keys, no network. Exit code 0 = all passed.
"""

from __future__ import annotations

import sys

import config
from models import TradeDecision
from risk_gate import RiskGate

PASS, FAIL = "PASS", "FAIL"
results: list[tuple[str, str, str]] = []


def check(name: str, condition: bool, detail: str = "") -> None:
    results.append((PASS if condition else FAIL, name, detail))


def decision(**kw) -> TradeDecision:
    base = dict(
        symbol="NVDA",
        action="buy",
        confidence=0.90,
        thesis="test",
        invalidation="test",
        suggested_notional=1000.0,
        time_horizon_minutes=60,
    )
    base.update(kw)
    return TradeDecision(**base)


HEALTHY = {"equity": 100_000.0, "cash": 100_000.0, "daily_pl": 0.0,
           "daytrade_count": 0, "minutes_to_close": 180.0}
gate = RiskGate()


# 1. Sanity: a good decision under healthy conditions is approved.
r = gate.evaluate(decision(), HEALTHY, [], 0)
check("baseline good decision approved", r.approved, "; ".join(r.reasons))

# 2. Confidence below the floor is refused.
r = gate.evaluate(decision(confidence=config.MIN_CONFIDENCE - 0.01), HEALTHY, [], 0)
check("sub-floor confidence refused", not r.approved, "; ".join(r.reasons))

# 3. A decider cannot inflate its own size — the gate sizes, not the decider.
r = gate.evaluate(decision(suggested_notional=999_999.0), HEALTHY, [], 0)
check(
    "decider cannot inflate its own size",
    r.approved and r.capped_notional <= config.MAX_POSITION_NOTIONAL,
    f"requested 999,999 -> {r.capped_notional}",
)

# 4. Position cap enforced (existing positions).
full = [{"symbol": s} for s in ("AAPL", "TSLA", "SPY")][: config.MAX_CONCURRENT_POSITIONS]
r = gate.evaluate(decision(), HEALTHY, full, 0)
check("position cap enforced", not r.approved, "; ".join(r.reasons))

# 5. Position cap counts entries already approved THIS cycle (off-by-one guard).
r = gate.evaluate(decision(), HEALTHY, [{"symbol": "AAPL"}], config.MAX_CONCURRENT_POSITIONS - 1)
check("in-cycle approvals count toward cap", not r.approved, "; ".join(r.reasons))

# 6. Daily loss limit halts new entries.
blown = {**HEALTHY, "daily_pl": -config.MAX_DAILY_LOSS - 0.01}
r = gate.evaluate(decision(), blown, [], 0)
check("daily loss limit halts entries", not r.approved, "; ".join(r.reasons))

# 7. Closing a position is still allowed when the daily loss limit is breached.
#    Risk-reducing actions must never be blocked by risk limits.
r = gate.evaluate(decision(action="close"), blown, [{"symbol": "NVDA"}], 0)
check("close allowed despite loss limit", r.approved, "; ".join(r.reasons))

# 8. Short entries refused (long-only v1).
r = gate.evaluate(decision(action="sell"), HEALTHY, [], 0)
check("short entry refused", not r.approved, "; ".join(r.reasons))

# 9. No doubling into an existing position.
r = gate.evaluate(decision(symbol="NVDA"), HEALTHY, [{"symbol": "NVDA"}], 0)
check("no doubling existing position", not r.approved, "; ".join(r.reasons))

# 10. Closing a position we do not hold is refused.
r = gate.evaluate(decision(action="close"), HEALTHY, [], 0)
check("close of unheld position refused", not r.approved, "; ".join(r.reasons))

# 11. PDT: under the equity threshold with day trades exhausted -> refused.
#     Fixtures derive from HEALTHY so only the field under test differs.
pdt = {**HEALTHY, "equity": config.PDT_MIN_EQUITY - 1,
       "daytrade_count": config.PDT_MAX_DAY_TRADES}
r = gate.evaluate(decision(), pdt, [], 0)
check("PDT limit enforced under $25k", not r.approved, "; ".join(r.reasons))

# 12. PDT does not apply above the equity threshold.
rich = {**HEALTHY, "equity": config.PDT_MIN_EQUITY + 1,
        "daytrade_count": config.PDT_MAX_DAY_TRADES + 5}
r = gate.evaluate(decision(), rich, [], 0)
check("PDT ignored above $25k", r.approved, "; ".join(r.reasons))

# 13. FAIL CLOSED: missing account data must refuse, not assume defaults.
r = gate.evaluate(decision(), {}, [], 0)
check("missing account data fails closed", not r.approved, "; ".join(r.reasons))

# 14. FAIL CLOSED: missing daytrade count under PDT threshold.
r = gate.evaluate(decision(), {**HEALTHY, "equity": 10_000.0, "daytrade_count": None},
                  [], 0)
check("missing daytrade count fails closed", not r.approved, "; ".join(r.reasons))

# 15. "hold" never produces an order.
r = gate.evaluate(decision(action="hold"), HEALTHY, [], 0)
check("hold never executes", not r.approved, "; ".join(r.reasons))

# 16. Kill switch overrides everything, including a perfect decision.
config.KILL_SWITCH_FILE.touch()
try:
    r = gate.evaluate(decision(), HEALTHY, [], 0)
    check("kill switch blocks all orders", not r.approved, "; ".join(r.reasons))
finally:
    config.KILL_SWITCH_FILE.unlink(missing_ok=True)

# 17. Gate recovers once the kill switch is removed.
r = gate.evaluate(decision(), HEALTHY, [], 0)
check("gate resumes after kill switch cleared", r.approved, "; ".join(r.reasons))

# 18. No new entries inside the end-of-day cutoff (overnight gap protection).
near_close = {**HEALTHY, "minutes_to_close": config.NO_ENTRY_BEFORE_CLOSE_MIN - 1}
r = gate.evaluate(decision(), near_close, [], 0)
check("no entries near the close", not r.approved, "; ".join(r.reasons))

# 19. Entries still allowed comfortably before the cutoff.
r = gate.evaluate(decision(), {**HEALTHY, "minutes_to_close": config.NO_ENTRY_BEFORE_CLOSE_MIN + 5},
                  [], 0)
check("entries allowed before cutoff", r.approved, "; ".join(r.reasons))

# 20. Closing is STILL allowed inside the cutoff — the EOD flatten depends on it.
r = gate.evaluate(decision(action="close"), near_close, [{"symbol": "NVDA"}], 0)
check("close allowed inside EOD window", r.approved, "; ".join(r.reasons))

# 21. FAIL CLOSED: unknown time-to-close must refuse entries.
no_clock = {k: v for k, v in HEALTHY.items() if k != "minutes_to_close"}
r = gate.evaluate(decision(), no_clock, [], 0)
check("unknown time-to-close fails closed", not r.approved, "; ".join(r.reasons))


# --- position sizing -------------------------------------------------------

# 22. Flat mode (default): size ignores confidence entirely.
assert not config.CONFIDENCE_SIZING, "these tests assume CONFIDENCE_SIZING starts off"
lo = gate.evaluate(decision(confidence=0.66), HEALTHY, [], 0).capped_notional
hi = gate.evaluate(decision(confidence=0.99), HEALTHY, [], 0).capped_notional
check("flat mode ignores confidence", lo == hi == config.FLAT_POSITION_NOTIONAL,
      f"0.66 -> ${lo:,.0f}, 0.99 -> ${hi:,.0f}")

# 23. Half-cash ceiling binds even in flat mode.
broke = {**HEALTHY, "cash": 400.0}   # half of 400 = 200 < flat 1000
r = gate.evaluate(decision(), broke, [], 0)
check("half-cash ceiling enforced", r.approved and r.capped_notional == 200.0,
      f"cash $400 -> ${r.capped_notional:,.0f}")

# 24. FAIL CLOSED: unknown cash cannot be sized safely.
no_cash = {k: v for k, v in HEALTHY.items() if k != "cash"}
r = gate.evaluate(decision(), no_cash, [], 0)
check("unknown cash fails closed", not r.approved, "; ".join(r.reasons))

# --- confidence-linked sizing (feature flag flipped on for these) ----------
config.CONFIDENCE_SIZING = True
try:
    low = gate.evaluate(decision(confidence=0.66), HEALTHY, [], 0).capped_notional
    high = gate.evaluate(decision(confidence=0.99), HEALTHY, [], 0).capped_notional

    # 25. Higher confidence must produce a strictly bigger position.
    check("more confidence -> bigger bet", high > low,
          f"0.66 -> ${low:,.0f}   0.99 -> ${high:,.0f}")

    # 26. Size scales with the portfolio, not a fixed dollar amount.
    small_acct = {**HEALTHY, "equity": 10_000.0, "cash": 10_000.0}
    small = gate.evaluate(decision(confidence=0.80), small_acct, [], 0).capped_notional
    big = gate.evaluate(decision(confidence=0.80), HEALTHY, [], 0).capped_notional
    check("size scales with portfolio", small < big,
          f"$10k acct -> ${small:,.0f}   $100k acct -> ${big:,.0f}")

    # 27. The hard per-position ceiling still wins over confidence scaling.
    huge = {**HEALTHY, "equity": 10_000_000.0, "cash": 10_000_000.0}
    r = gate.evaluate(decision(confidence=0.99), huge, [], 0)
    check("hard ceiling beats confidence scaling",
          r.capped_notional == config.MAX_POSITION_NOTIONAL,
          f"$10M acct -> ${r.capped_notional:,.0f}")

    # 28. Never more than half the cash, even at max confidence.
    thin = {**HEALTHY, "equity": 100_000.0, "cash": 600.0}
    r = gate.evaluate(decision(confidence=0.99), thin, [], 0)
    check("never more than half the cash", r.capped_notional <= 300.0,
          f"cash $600 -> ${r.capped_notional:,.0f}")
finally:
    config.CONFIDENCE_SIZING = False


# ---- report ---------------------------------------------------------------
width = max(len(name) for _, name, _ in results)
failures = 0
for status, name, detail in results:
    if status == FAIL:
        failures += 1
    print(f"  [{status}] {name:<{width}}  {detail}")

print(f"\n{len(results) - failures}/{len(results)} passed")
sys.exit(1 if failures else 0)
