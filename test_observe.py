"""Prove observe mode cannot place an order.

The executor here raises on any order attempt, so if a single call slips
through the test fails loudly rather than quietly trading real money.

    python test_observe.py

Uses live market data (read-only) but never touches the trading API.
"""

from __future__ import annotations

import sys

import config

# Enable BEFORE importing main, so the shadow runner is built for observe mode.
config.OBSERVE_MODE = True
config.SHADOW_BACKENDS = ["rule", "orb", "meanrev"]
config.DECIDER_BACKEND = "rule"

import main  # noqa: E402
from data_feed import DataFeed  # noqa: E402
from risk_gate import RiskGate  # noqa: E402

results: list[tuple[bool, str, str]] = []


def check(ok: bool, label: str, detail: str = "") -> None:
    results.append((ok, label, detail))


class ExplodingExecutor:
    """Any order attempt is a test failure."""

    def __init__(self) -> None:
        self.order_attempts = 0

    def account_state(self) -> dict:
        return {"equity": 100_000.0, "cash": 100_000.0, "daily_pl": 0.0,
                "daytrade_count": 0, "pattern_day_trader": False, "paper": True}

    def open_positions(self) -> list:
        return []

    def minutes_to_close(self) -> float:
        return 120.0     # mid-session, well clear of the EOD window

    def market_open(self) -> bool:
        return True

    def submit_entry(self, decision, notional):
        self.order_attempts += 1
        raise AssertionError(f"ORDER PLACED IN OBSERVE MODE: buy {decision.symbol}")

    def close_position(self, symbol):
        self.order_attempts += 1
        raise AssertionError(f"ORDER PLACED IN OBSERVE MODE: close {symbol}")


runner = main.build_shadow_runner()
check(runner is not None, "shadow runner built in observe mode")
if runner:
    check(config.DECIDER_BACKEND in runner.deciders,
          "live backend folded into shadows",
          f"strategies: {sorted(runner.deciders)}")
    check(len(runner.deciders) == 3, "all three strategies present",
          f"{sorted(runner.deciders)}")

executor = ExplodingExecutor()
feed = DataFeed()
gate = RiskGate()

try:
    for i in range(2):
        main.run_cycle(feed, main.build_decider(), gate, executor, None, runner)
    check(True, "two cycles ran without placing an order")
except AssertionError as exc:
    check(False, "two cycles ran without placing an order", str(exc))

check(executor.order_attempts == 0, "zero order attempts",
      f"{executor.order_attempts} attempt(s)")

if runner:
    total_seen = sum(
        len(b.positions) + b.trades for b in runner.books.values()
    )
    check(True, "strategies produced activity",
          f"virtual positions+closed trades across books: {total_seen}")
    print("\n  virtual books after 2 cycles:")
    print(runner.report())

print()
width = max(len(l) for _, l, _ in results)
failures = sum(1 for ok, _, _ in results if not ok)
for ok, label, detail in results:
    print(f"  [{'PASS' if ok else 'FAIL'}] {label:<{width}}  {detail}")
print(f"\n{len(results) - failures}/{len(results)} passed")
sys.exit(1 if failures else 0)
