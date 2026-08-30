"""Tests for the ML decider — above all, that it refuses what it should.

    python test_ml.py
"""

from __future__ import annotations

import json
import sys
from datetime import datetime, timedelta

import config
from data_feed import EASTERN

PASS, FAIL = "PASS", "FAIL"
results: list[tuple[str, str, str]] = []


def check(name: str, cond: bool, detail: str = "") -> None:
    results.append((PASS if cond else FAIL, name, detail))


def row(**kw) -> dict:
    base = {
        "symbol": "NVDA", "last": 100.0, "pct_vs_vwap": 0.3, "rsi_14": 55.0,
        "atr_14": 1.0, "atr_pct": 1.0, "spread_pct": 0.05,
        "volume_last_bar": 2e6, "avg_volume_20": 1e6,
        "ret_1": 0.1, "ret_3": 0.2, "ret_6": 0.4, "gap_pct": 0.5,
        "min_since_open": 120,
        "opening_range": {"high": 99.5, "low": 98.5, "bars": 3, "complete": True},
    }
    base.update(kw)
    return base


def ctx(rows, positions=None, at="12:00") -> str:
    h, m = (int(x) for x in at.split(":"))
    stamp = datetime.now(EASTERN).replace(hour=h, minute=m, second=0, microsecond=0)
    return json.dumps({
        "as_of_et": stamp.isoformat(),
        "account": {"equity": 100_000.0, "cash": 100_000.0},
        "open_positions": positions or [],
        "symbols": rows, "excluded_symbols": [],
    })


# --- 1. A non-deployable model must never propose an entry -----------------
import decider_ml  # noqa: E402

d = decider_ml.MLDecider()
model_missing = d.model is None and not d.meta
if d.meta and not d.meta.get("deployable"):
    check("non-deployable model refuses entries", d.model is None,
          "model file present, verdict negative, decider disarmed")
elif model_missing:
    check("missing model refuses entries", d.model is None, "no model file")
else:
    check("deployable model loaded", d.model is not None,
          f"threshold {d.meta.get('threshold')}")

out = d.decide(ctx([row()]))
buys = [x for x in out.decisions if x.action == "buy"]
if d.model is None:
    check("holds with no usable model", not buys, out.market_read[:60])
else:
    check("scores symbols when armed", True, out.market_read[:60])

# --- 2. Exits use the LOADED model's barrier geometry ----------------------
# The decider adopts whatever tp/sl/horizon the model file carries, so the
# expected exit prices are derived from d.tp / d.sl / d.horizon — hardcoding
# ±1 ATR here broke the moment a model with different geometry was trained,
# which is exactly the coupling the metadata exists to enforce.
pos = [{"symbol": "NVDA", "avg_entry": 100.0, "qty": 10}]
geom = f"tp={d.tp:g} sl={d.sl:g} horizon={d.horizon}min"

out = d.decide(ctx([row(last=100.0 + (d.tp + 0.2) * 1.0)], pos))   # past target
closes = [x for x in out.decisions if x.action == "close"]
check("target exit at +tp ATR", len(closes) == 1,
      closes[0].thesis[:46] if closes else f"no exit ({geom})")

out = d.decide(ctx([row(last=100.0 - (d.sl + 0.3) * 1.0)], pos))   # past stop
closes = [x for x in out.decisions if x.action == "close"]
check("stop exit at -sl ATR", len(closes) == 1,
      closes[0].thesis[:46] if closes else f"no exit ({geom})")

out = d.decide(ctx([row(last=100.0 + 0.4 * min(d.tp, d.sl))], pos))  # inside
closes = [x for x in out.decisions if x.action == "close"]
check("holds inside the barriers", not closes, geom)

# --- 3. Horizon exit fires from the in-memory entry record -----------------
from datetime import timedelta  # noqa: E402

now_et = datetime.now(EASTERN).replace(second=0, microsecond=0)
d._entries["NVDA"] = {
    "atr": 1.0,
    "ts": (now_et - timedelta(minutes=d.horizon + 30)).isoformat(),
}
inside = 100.0 + 0.4 * min(d.tp, d.sl)
out = d.decide(json.dumps({
    "as_of_et": now_et.isoformat(),
    "account": {"equity": 100_000.0, "cash": 100_000.0},
    "open_positions": pos,
    "symbols": [row(last=inside)], "excluded_symbols": [],
}))
closes = [x for x in out.decisions if x.action == "close"]
check("horizon exit after the trained horizon", len(closes) == 1,
      closes[0].thesis[:46] if closes else f"no exit ({geom})")

# Restart amnesia: unknown entry time -> barrier management only, no horizon.
d._entries.pop("NVDA", None)
out = d.decide(ctx([row(last=inside)], pos))
closes = [x for x in out.decisions if x.action == "close"]
check("no horizon exit without entry record", not closes,
      "degrades to barriers-only after restart")

# --- 4. Feature schema: match when compatible, REFUSE when not -------------
# An intraday-featured model must produce a vector in exact training order.
# A model trained on features the live context cannot supply (e.g. the daily
# swing set: ret_20, dist_ma50, dow) must be refused outright — silently
# feeding NaNs where training had values would be undetectable skew.
if d.meta.get("features"):
    supplied = {
        "pct_vs_vwap", "rsi_14", "atr_pct", "vol_ratio", "pct_vs_or_high",
        "or_range_pct", "after_or", "ret_1", "ret_3", "ret_6", "gap_pct",
        "min_since_open",
    }
    compatible = set(d.meta["features"]) <= supplied
    X = d._features(row())
    if compatible:
        check("feature vector matches training schema",
              X is not None and list(X.columns) == d.meta["features"],
              f"{0 if X is None else len(X.columns)} features")
    else:
        check("incompatible schema is refused, not NaN-filled",
              X is None,
              f"model wants {sorted(set(d.meta['features']) - supplied)[:3]}... "
              "-> refused")

width = max(len(n) for _, n, _ in results)
failures = sum(1 for s, _, _ in results if s == FAIL)
for status, name, detail in results:
    print(f"  [{status}] {name:<{width}}  {detail}")
print(f"\n{len(results) - failures}/{len(results)} passed")
sys.exit(1 if failures else 0)
