"""Tests for the insider decider — chiefly that it acts only on FRESH filings.

The dangerous failure is staleness: the insider summary looks back days, so a
symbol with any historical buy would look actionable forever and the bot would
re-enter the same event every cycle until the position cap filled with ghosts.

    python test_insider_decider.py
"""

from __future__ import annotations

import json
import sys
from datetime import datetime, timedelta

import config
from data_feed import EASTERN
from decider_insider import HOLD_TRADING_DAYS, SL_ATR, TP_ATR, InsiderDecider

results: list[tuple[bool, str, str]] = []


def check(ok: bool, label: str, detail: str = "") -> None:
    results.append((ok, label, detail))


def ins(buys=1, insiders=1, usd=500_000, cluster=False, csuite=False,
        filed_today=0, days_ago=0):
    recent = (datetime.now().date() - timedelta(days=days_ago)).isoformat()
    return {
        "lookback_days": 10,
        "open_market_buys": {"transactions": buys, "distinct_insiders": insiders,
                             "total_usd": usd, "most_recent": recent},
        "open_market_sells": {"transactions": 0, "distinct_insiders": 0,
                              "total_usd": 0, "most_recent": None},
        "cluster_buy": cluster, "c_suite_buy": csuite,
        "filings_today": filed_today,
    }


def row(symbol="BAC", last=100.0, atr=1.0, insider=None):
    r = {"symbol": symbol, "last": last, "atr_14": atr, "rsi_14": 50.0,
         "pct_vs_vwap": 0.0, "spread_pct": 0.03}
    if insider is not None:
        r["insider_form4"] = insider
    return r


def ctx(rows, positions=None, at="12:00"):
    h, m = (int(x) for x in at.split(":"))
    stamp = datetime.now(EASTERN).replace(hour=h, minute=m, second=0, microsecond=0)
    return json.dumps({
        "as_of_et": stamp.isoformat(),
        "account": {"equity": 100_000.0, "cash": 100_000.0, "daily_pl": 0.0,
                    "daytrade_count": 0, "minutes_to_close": 180.0},
        "open_positions": positions or [],
        "symbols": rows, "excluded_symbols": [],
    })


d = InsiderDecider()
buys = lambda r: [x for x in r.decisions if x.action == "buy"]
closes = lambda r: [x for x in r.decisions if x.action == "close"]

# --- freshness: the failure that would matter most -------------------------
check(len(buys(d.decide(ctx([row(insider=ins(days_ago=0))])))) == 1,
      "fresh filing triggers a buy")

check(not buys(d.decide(ctx([row(insider=ins(days_ago=45))]))),
      "stale filing does NOT trigger", "45 days old, inside the lookback window")

check(len(buys(d.decide(ctx([row(insider=ins(days_ago=99, filed_today=2))])))) == 1,
      "filings_today overrides a stale most_recent")

# --- no event, no trade ----------------------------------------------------
check(not buys(d.decide(ctx([row(insider=ins(buys=0))]))),
      "zero buys -> no trade", "selling-only symbols ignored")
check(not buys(d.decide(ctx([row()]))), "missing insider block -> no trade")
check(not buys(d.decide(ctx([row(insider={"insider_data": "unavailable"})]))),
      "unavailable insider data -> no trade")

# --- conviction ordering must match the measured effect sizes --------------
plain = buys(d.decide(ctx([row(insider=ins())])))[0].confidence
csuite = buys(d.decide(ctx([row(insider=ins(csuite=True))])))[0].confidence
cluster = buys(d.decide(ctx([row(insider=ins(cluster=True, csuite=True,
                                             insiders=4))])))[0].confidence
check(cluster > csuite > plain, "conviction ranks cluster > C-suite > plain",
      f"{plain} < {csuite} < {cluster}")
check(cluster <= 0.90, "confidence capped", str(cluster))

# --- every proposal must clear the live risk gate's floor ------------------
check(plain >= config.MIN_CONFIDENCE,
      "even the weakest event clears the gate floor",
      f"{plain} vs floor {config.MIN_CONFIDENCE}")

# --- exits use the barriers the edge was measured with ---------------------
pos = [{"symbol": "BAC", "avg_entry": 100.0, "qty": 10}]
check(len(closes(d.decide(ctx([row(last=100.0 + TP_ATR + 0.2,
                                   insider=ins())], pos)))) == 1,
      f"target exit at +{TP_ATR:g} ATR")
check(len(closes(d.decide(ctx([row(last=100.0 - SL_ATR - 0.2,
                                   insider=ins())], pos)))) == 1,
      f"stop exit at -{SL_ATR:g} ATR")
check(not closes(d.decide(ctx([row(last=100.5, insider=ins())], pos))),
      "holds inside the barriers")

# --- hold expiry, and graceful degradation after a restart -----------------
now = datetime.now(EASTERN).replace(second=0, microsecond=0)
d._entries["BAC"] = {"atr": 1.0,
                     "ts": (now - timedelta(days=HOLD_TRADING_DAYS * 2)).isoformat()}
out = d.decide(json.dumps({
    "as_of_et": now.isoformat(),
    "account": {"equity": 100_000.0, "cash": 100_000.0, "daily_pl": 0.0,
                "daytrade_count": 0, "minutes_to_close": 180.0},
    "open_positions": pos,
    "symbols": [row(last=100.5, insider=ins())], "excluded_symbols": [],
}))
check(len(closes(out)) == 1, "hold expiry closes the position",
      closes(out)[0].thesis[:44] if closes(out) else "no exit")

d._entries.pop("BAC", None)
check(not closes(d.decide(ctx([row(last=100.5, insider=ins())], pos))),
      "no time exit without an entry record", "barriers-only after restart")

# --- already-held symbols are not re-entered -------------------------------
check(not buys(d.decide(ctx([row(insider=ins())], pos))),
      "held symbol is not re-entered")

width = max(len(l) for _, l, _ in results)
failures = sum(1 for ok, _, _ in results if not ok)
for ok, label, detail in results:
    print(f"  [{'PASS' if ok else 'FAIL'}] {label:<{width}}  {detail}")
print(f"\n{len(results) - failures}/{len(results)} passed")
sys.exit(1 if failures else 0)
