"""Tests for shadow-mode fidelity — chiefly that shadows obey the risk gate.

A shadow that trades under rules the live system forbids measures a strategy
that cannot actually be run, and quietly becomes incomparable to the backtest
(which has always gated).

    python test_shadow.py
"""

from __future__ import annotations

import json
import sys
from datetime import datetime

import config
from data_feed import EASTERN
from models import CycleDecisions, TradeDecision
from shadow import ShadowRunner

results: list[tuple[bool, str, str]] = []


def check(ok: bool, label: str, detail: str = "") -> None:
    results.append((ok, label, detail))


class Scripted:
    """A decider that emits exactly what the test tells it to."""

    def __init__(self, decisions):
        self._decisions = decisions

    def decide(self, ctx_json):
        return CycleDecisions(market_read="scripted", decisions=list(self._decisions))


def buy(symbol="NVDA", confidence=0.80, notional=1000.0) -> TradeDecision:
    return TradeDecision(
        symbol=symbol, action="buy", confidence=confidence, thesis="t",
        invalidation="t", suggested_notional=notional, time_horizon_minutes=60,
    )


def ctx(symbols, minutes_to_close=180.0) -> str:
    now = datetime.now(EASTERN).replace(hour=12, minute=0, second=0, microsecond=0)
    return json.dumps({
        "as_of_et": now.isoformat(),
        "account": {"equity": 100_000.0, "cash": 100_000.0, "daily_pl": 0.0,
                    "daytrade_count": 0, "minutes_to_close": minutes_to_close},
        "open_positions": [],
        "symbols": [{"symbol": s, "last": 100.0} for s in symbols],
        "excluded_symbols": [],
    })


# --- 1. Sub-floor confidence must be refused, exactly as live ---------------
r = ShadowRunner({"x": Scripted([buy(confidence=config.MIN_CONFIDENCE - 0.05)])})
out = r.run(ctx(["NVDA"]))
check(not out["x"]["events"] and out["x"]["gate_rejected"],
      "sub-floor confidence rejected in shadow",
      out["x"]["gate_rejected"][0]["why"] if out["x"]["gate_rejected"] else "TRADED")

# --- 2. Position cap must bind across one cycle ----------------------------
many = [buy(s) for s in ("NVDA", "AAPL", "TSLA", "SPY", "QQQ")]
r = ShadowRunner({"x": Scripted(many)})
out = r.run(ctx(["NVDA", "AAPL", "TSLA", "SPY", "QQQ"]))
opened = len(r.books["x"].positions)
check(opened <= config.MAX_CONCURRENT_POSITIONS,
      "position cap enforced in shadow",
      f"{opened} opened, cap {config.MAX_CONCURRENT_POSITIONS}")

# --- 3. No entries inside the end-of-day cutoff ----------------------------
r = ShadowRunner({"x": Scripted([buy()])})
out = r.run(ctx(["NVDA"], minutes_to_close=config.NO_ENTRY_BEFORE_CLOSE_MIN - 1))
check(not out["x"]["events"], "EOD entry cutoff enforced in shadow",
      out["x"]["gate_rejected"][0]["why"] if out["x"]["gate_rejected"] else "TRADED")

# --- 4. A clean decision still trades -------------------------------------
r = ShadowRunner({"x": Scripted([buy()])})
out = r.run(ctx(["NVDA"]))
check(len(out["x"]["events"]) == 1, "valid decision still fills",
      str(out["x"]["events"]))

# --- 5. Sizing comes from the gate, not the decider ------------------------
r = ShadowRunner({"x": Scripted([buy(notional=999_999.0)])})
r.run(ctx(["NVDA"]))
pos = r.books["x"].positions.get("NVDA")
value = pos.qty * pos.entry_price if pos else 0.0
check(0 < value <= config.MAX_POSITION_NOTIONAL * 1.001,
      "decider cannot inflate shadow position size",
      f"requested 999,999 -> ${value:,.0f}")

# --- 6. Books stay isolated between strategies -----------------------------
r = ShadowRunner({"a": Scripted([buy("NVDA")]), "b": Scripted([buy("AAPL")])})
r.run(ctx(["NVDA", "AAPL"]))
check(set(r.books["a"].positions) == {"NVDA"} and set(r.books["b"].positions) == {"AAPL"},
      "virtual books stay isolated",
      f"a={sorted(r.books['a'].positions)} b={sorted(r.books['b'].positions)}")

# --- 7. A broken strategy cannot take down the others ----------------------
class Exploding:
    def decide(self, ctx_json):
        raise RuntimeError("boom")


r = ShadowRunner({"bad": Exploding(), "good": Scripted([buy()])})
out = r.run(ctx(["NVDA"]))
check(out["bad"].get("error") and len(out["good"]["events"]) == 1,
      "one strategy's exception is contained",
      f"bad={out['bad']}, good traded={len(out['good']['events'])}")

# --- 8. Slippage is charged on both sides ---------------------------------
r = ShadowRunner({"x": Scripted([buy()])})
r.run(ctx(["NVDA"]))
entry = r.books["x"].positions["NVDA"].entry_price
check(entry > 100.0, "entry pays the offer, not the midpoint",
      f"mid 100.00 -> filled {entry:.4f}")

width = max(len(l) for _, l, _ in results)
failures = sum(1 for ok, _, _ in results if not ok)
for ok, label, detail in results:
    print(f"  [{'PASS' if ok else 'FAIL'}] {label:<{width}}  {detail}")
print(f"\n{len(results) - failures}/{len(results)} passed")
sys.exit(1 if failures else 0)
