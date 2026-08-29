"""Unit tests for the opening range breakout strategy.

Handcrafted scenarios, not random data — each one isolates a single condition
so a failure names the exact rule that broke.

    python test_orb.py

No API keys, no network. Exit code 0 = all passed.
"""

from __future__ import annotations

import json
import sys

import config
from decider_orb import ORBDecider

PASS, FAIL = "PASS", "FAIL"
results: list[tuple[str, str, str]] = []


def check(name: str, condition: bool, detail: str = "") -> None:
    results.append((PASS if condition else FAIL, name, detail))


def symbol(**kw) -> dict:
    """A symbol row that breaks out cleanly unless overridden."""
    base = {
        "symbol": "NVDA",
        "last": 101.0,                    # above the 100.0 range high
        "pct_vs_vwap": 0.40,
        "atr_14": 1.0,
        "volume_last_bar": 2_000_000.0,   # 2.0x the average
        "avg_volume_20": 1_000_000.0,
        "opening_range": {"high": 100.0, "low": 99.0, "bars": 3, "complete": True},
    }
    base.update(kw)
    return base


def ctx(symbols: list[dict], positions: list[dict] | None = None) -> str:
    return json.dumps({
        "account": {"equity": 100_000.0, "cash": 100_000.0},
        "open_positions": positions or [],
        "symbols": symbols,
        "excluded_symbols": [],
    })


d = ORBDecider()


def buys(result) -> list:
    return [x for x in result.decisions if x.action == "buy"]


def closes(result) -> list:
    return [x for x in result.decisions if x.action == "close"]


# --- entry conditions ------------------------------------------------------

r = d.decide(ctx([symbol()]))
check("clean breakout enters", len(buys(r)) == 1,
      f"conf={buys(r)[0].confidence}" if buys(r) else "no entry")

# The guard the plain momentum rule lacked.
r = d.decide(ctx([symbol(volume_last_bar=1_100_000.0)]))   # only 1.1x
check("breakout WITHOUT volume refused", not buys(r),
      f"1.1x avg vs {config.ORB_VOLUME_MULT}x required")

r = d.decide(ctx([symbol(opening_range={"high": 100.0, "low": 99.0,
                                        "bars": 2, "complete": False})]))
check("range still forming refused", not buys(r), "opening range not yet closed")

r = d.decide(ctx([symbol(last=99.5)]))
check("no breakout (below range high) refused", not buys(r), "99.5 < 100.0")

r = d.decide(ctx([symbol(last=103.5)]))   # 3.5% past the high
check("over-extended breakout refused", not buys(r),
      f"3.5% past high vs {config.ORB_MAX_EXTENSION_PCT}% max — no chasing")

r = d.decide(ctx([symbol(pct_vs_vwap=-0.20)]))
check("breakout below VWAP refused", not buys(r), "weak tape")

r = d.decide(ctx([symbol(opening_range=None)]))
check("missing opening range refused", not buys(r), "no range data")

r = d.decide(ctx([symbol(avg_volume_20=None)]))
check("missing volume data refused", not buys(r), "cannot confirm without volume")

# --- confidence shape ------------------------------------------------------

fresh = d.decide(ctx([symbol(last=100.3, volume_last_bar=3_000_000.0)]))
stale = d.decide(ctx([symbol(last=101.8, volume_last_bar=1_600_000.0)]))
check("fresh + high volume scores higher",
      buys(fresh)[0].confidence > buys(stale)[0].confidence,
      f"{buys(fresh)[0].confidence} vs {buys(stale)[0].confidence}")

r = d.decide(ctx([symbol(volume_last_bar=10_000_000.0, last=100.1)]))
check("confidence capped at 0.85", buys(r)[0].confidence <= 0.85,
      f"10x volume -> {buys(r)[0].confidence}")

# --- exits -----------------------------------------------------------------

pos = [{"symbol": "NVDA", "avg_entry": 101.0, "qty": 10}]

r = d.decide(ctx([symbol(last=99.5)]), )
r = d.decide(ctx([symbol(last=99.5)], pos))
check("falls back inside range -> close", len(closes(r)) == 1,
      closes(r)[0].thesis[:60] if closes(r) else "no exit")

# 101.0 entry, ATR 1.0, 1x stop -> exit below 100.0. Use 100.5: still above the
# range high so the range rule does not fire, but must not trip the stop either.
r = d.decide(ctx([symbol(last=100.5)], pos))
check("above range and above stop -> hold", not closes(r), "100.5 > stop 100.0")

# Raise the range high so the range rule cannot fire, isolating the ATR stop.
r = d.decide(ctx([symbol(last=99.0, opening_range={"high": 98.0, "low": 97.0,
                                                   "bars": 3, "complete": True})], pos))
check("ATR stop triggers", len(closes(r)) == 1,
      closes(r)[0].thesis[:70] if closes(r) else "no exit")

r = d.decide(ctx([symbol()], pos))
check("no double-entry while held", not buys(r), "already holding NVDA")

# --- multi-symbol ----------------------------------------------------------

many = [symbol(symbol=s, volume_last_bar=v) for s, v in
        [("AAPL", 3_000_000.0), ("NVDA", 2_500_000.0), ("TSLA", 2_000_000.0),
         ("SPY", 1_800_000.0)]]
r = d.decide(ctx(many))
check("at most 2 entries per cycle", len(buys(r)) == 2,
      f"{len(buys(r))} of 4 candidates")
check("highest conviction picked first",
      buys(r)[0].confidence >= buys(r)[1].confidence,
      " > ".join(f"{b.symbol}:{b.confidence}" for b in buys(r)))


# --- report ----------------------------------------------------------------
width = max(len(n) for _, n, _ in results)
failures = sum(1 for s, _, _ in results if s == FAIL)
for status, name, detail in results:
    print(f"  [{status}] {name:<{width}}  {detail}")
print(f"\n{len(results) - failures}/{len(results)} passed")
sys.exit(1 if failures else 0)
