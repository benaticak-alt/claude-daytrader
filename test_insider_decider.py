"""Tests for the insider decider — chiefly that it acts only on FRESH filings,
and only ONCE per filing.

The dangerous failures are staleness and churn: the insider summary looks back
days, so a symbol with any historical buy would look actionable forever; and
once a position closes, the same disclosure would be re-entered every cycle
(observed live: GME six times in two days). Neither is the strategy whose edge
was measured — one trade per event, held 21 days or to ±1.5 DAILY ATR.

    python test_insider_decider.py
"""

from __future__ import annotations

import json
import sys
import tempfile
from datetime import datetime, timedelta
from pathlib import Path

import config
import decider_insider
from data_feed import EASTERN
from decider_insider import HOLD_TRADING_DAYS, SL_ATR, TP_ATR, InsiderDecider

# Never touch the live state files from a test run.
decider_insider.STATE_DIR = Path(tempfile.mkdtemp())

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


def row(symbol="BAC", last=100.0, atr=1.0, insider=None, atr_intraday=0.1,
        dollar_vol=50e6, dist_ma50=None):
    # atr_intraday is deliberately a tenth of the daily unit, as it is live —
    # any test that passes with the wrong one is a test that would have let
    # the first forward test's bug through.
    r = {"symbol": symbol, "last": last, "atr_14": atr_intraday, "atr_daily": atr,
         "rsi_14": 50.0, "pct_vs_vwap": 0.0, "spread_pct": 0.03,
         "dollar_vol_20": dollar_vol, "dist_ma50_pct": dist_ma50}
    if insider is not None:
        r["insider_form4"] = insider
    return r


def ctx(rows, positions=None, at="12:00", events=None):
    h, m = (int(x) for x in at.split(":"))
    stamp = datetime.now(EASTERN).replace(hour=h, minute=m, second=0, microsecond=0)
    return json.dumps({
        "as_of_et": stamp.isoformat(),
        "account": {"equity": 100_000.0, "cash": 100_000.0, "daily_pl": 0.0,
                    "daytrade_count": 0, "minutes_to_close": 180.0},
        "open_positions": positions or [],
        "symbols": rows, "excluded_symbols": [],
        "insider_events": events or [],
    })


def event(symbol="BAC", usd=250_000, officer=1, director=0, csuite=0,
          trader_type="unclassified", days_ago=0, paid=100.0, acc="0001-26-000001"):
    return {"accession": acc, "symbol": symbol, "issuer_cik": "1", "owner_cik": "9",
            "owner_name": "Doe Jane",
            "filing_date": (datetime.now().date() - timedelta(days=days_ago)).isoformat(),
            "buy_usd": usd, "buy_shares": usd / paid, "price_paid": paid,
            "is_officer": officer, "is_director": director, "is_10pct": 0,
            "is_csuite": csuite, "title": "", "trader_type": trader_type}


def fresh(regime=False) -> InsiderDecider:
    """A decider with no memory — each check gets its own so the
    one-entry-per-filing rule does not bleed between unrelated checks."""
    for f in decider_insider.STATE_DIR.glob("*_decider_state.json"):
        f.unlink()
    return InsiderDecider(regime=regime, name="insider_regime" if regime else "insider")


buys = lambda r: [x for x in r.decisions if x.action == "buy"]
closes = lambda r: [x for x in r.decisions if x.action == "close"]

# --- freshness: the failure that would matter most -------------------------
check(len(buys(fresh().decide(ctx([row(insider=ins(days_ago=0))])))) == 1,
      "fresh filing triggers a buy")

# --- universe-feed events and the pre-registered filters -------------------
ok = fresh().decide(ctx([row()], events=[event()]))
check(len(buys(ok)) == 1, "universe event: officer, $250k, liquid -> buy")
check(not buys(fresh().decide(ctx([row()], events=[event(usd=50_000)]))),
      "purchase below $100k rejected")
check(not buys(fresh().decide(ctx([row()], events=[event(officer=0, director=0)]))),
      "10%-owner-only filer rejected", "t=+0.9 in the study")
check(not buys(fresh().decide(ctx([row()], events=[event(trader_type="routine")]))),
      "routine trader rejected", "same month every year = schedule, not information")
check(not buys(fresh().decide(ctx([row(dollar_vol=400_000)], events=[event()]))),
      "illiquid name rejected", "<$1M/day")
check(not buys(fresh().decide(ctx([row(last=100.0)], events=[event(paid=80.0)]))),
      "deep-discount purchase rejected", "20% below market = placement coded P")
check(not buys(fresh().decide(ctx([row()], events=[event(days_ago=6)]))),
      "stale universe event rejected")
check(not buys(fresh().decide(ctx([row(symbol="XYZ")], events=[event(symbol="ABC")]))),
      "event with no market-data row cannot be traded")

d = fresh()
first = buys(d.decide(ctx([row()], events=[event()])))
again = buys(d.decide(ctx([row()], events=[event()])))
check(len(first) == 1 and not again, "same accession is not re-entered after close")
check(len(buys(d.decide(ctx([row()], events=[event(acc="0001-26-000002")])))) == 1,
      "a new filing (new accession) on the same symbol triggers")

# --- position cap is the strategy's own, not the intraday 3 ----------------
many_rows = [row(symbol=f"S{i}") for i in range(15)]
many_ev = [event(symbol=f"S{i}", acc=f"acc{i}") for i in range(15)]
out = fresh().decide(ctx(many_rows, events=many_ev))
check(len(buys(out)) == config.INSIDER_MAX_POSITIONS,
      f"enters up to INSIDER_MAX_POSITIONS ({config.INSIDER_MAX_POSITIONS})", str(len(buys(out))))
gp = fresh().gate_params
check(gp["max_positions"] == config.INSIDER_MAX_POSITIONS
      and abs(gp["size_pct_equity"] - config.INSIDER_EXPOSURE / config.INSIDER_MAX_POSITIONS) < 1e-12,
      "gate_params declare the book's cap and per-slot size", str(gp))

# --- regime variant --------------------------------------------------------
spy_up = row(symbol="SPY", dist_ma50=+2.0)
spy_dn = row(symbol="SPY", dist_ma50=-2.0)
check(len(buys(fresh(True).decide(ctx([row(), spy_up], events=[event()])))) == 1,
      "regime: SPY above 50d MA permits entry")
check(not buys(fresh(True).decide(ctx([row(), spy_dn], events=[event()]))),
      "regime: SPY below 50d MA blocks entry")
check(not buys(fresh(True).decide(ctx([row()], events=[event()]))),
      "regime: unknown SPY state fails closed")
check(len(buys(fresh().decide(ctx([row(), spy_dn], events=[event()])))) == 1,
      "non-regime variant ignores SPY")
pos_r = [{"symbol": "BAC", "avg_entry": 100.0, "qty": 10}]
check(len(closes(fresh(True).decide(ctx([row(last=98.0), spy_dn], pos_r)))) == 1,
      "regime blocks entries only — exits still fire")

check(not buys(fresh().decide(ctx([row(insider=ins(days_ago=45))]))),
      "stale filing does NOT trigger", "45 days old, inside the lookback window")

check(len(buys(fresh().decide(ctx([row(insider=ins(days_ago=99, filed_today=2))])))) == 1,
      "filings_today overrides a stale most_recent")

# --- no event, no trade ----------------------------------------------------
check(not buys(fresh().decide(ctx([row(insider=ins(buys=0))]))),
      "zero buys -> no trade", "selling-only symbols ignored")
check(not buys(fresh().decide(ctx([row()]))), "missing insider block -> no trade")
check(not buys(fresh().decide(ctx([row(insider={"insider_data": "unavailable"})]))),
      "unavailable insider data -> no trade")

# --- conviction ordering must match the measured effect sizes --------------
plain = buys(fresh().decide(ctx([row(insider=ins())])))[0].confidence
csuite = buys(fresh().decide(ctx([row(insider=ins(csuite=True))])))[0].confidence
cluster = buys(fresh().decide(ctx([row(insider=ins(cluster=True, csuite=True,
                                                   insiders=4))])))[0].confidence
check(cluster > csuite > plain, "conviction ranks cluster > C-suite > plain",
      f"{plain} < {csuite} < {cluster}")
check(cluster <= 0.90, "confidence capped", str(cluster))

# --- every proposal must clear the live risk gate's floor ------------------
check(plain >= config.MIN_CONFIDENCE,
      "even the weakest event clears the gate floor",
      f"{plain} vs floor {config.MIN_CONFIDENCE}")

# --- the barrier UNIT: daily ATR, never the 5-minute one -------------------
check(not buys(fresh().decide(ctx([row(insider=ins(), atr=None)]))),
      "no daily ATR -> no entry", "refuses to size barriers off the intraday unit")

d = fresh()
d.decide(ctx([row(insider=ins())]))                     # enter: records atr_daily=1.0
pos = [{"symbol": "BAC", "avg_entry": 100.0, "qty": 10}]
check(d._entries.get("BAC", {}).get("atr") == 1.0,
      "entry records the DAILY atr", str(d._entries.get("BAC")))
check(not closes(d.decide(ctx([row(last=100.5, insider=ins())], pos))),
      "+0.5 move holds (5x the intraday ATR, a third of the daily)",
      "this exact move closed GME within minutes before the fix")

# --- exits use the barriers the edge was measured with ---------------------
check(len(closes(d.decide(ctx([row(last=100.0 + TP_ATR + 0.2,
                                   insider=ins())], pos)))) == 1,
      f"target exit at +{TP_ATR:g} daily ATR")
check(len(closes(d.decide(ctx([row(last=100.0 - SL_ATR - 0.2,
                                   insider=ins())], pos)))) == 1,
      f"stop exit at -{SL_ATR:g} daily ATR")

# Restart with no entry record: fall back to the row's DAILY atr, not atr_14.
d2 = fresh()
check(not closes(d2.decide(ctx([row(last=100.5, insider=ins())], pos))),
      "restart fallback uses atr_daily, not atr_14",
      "0.5 would be a +5 ATR target on the intraday unit")
check(len(closes(d2.decide(ctx([row(last=100.0 + TP_ATR + 0.2,
                                    insider=ins())], pos)))) == 1,
      "restart fallback still exits at the daily barrier")

# --- hold expiry, and graceful degradation after a restart -----------------
now = datetime.now(EASTERN).replace(second=0, microsecond=0)
d = fresh()
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
check(not buys(fresh().decide(ctx([row(insider=ins())], pos))),
      "held symbol is not re-entered")

# --- ONE ENTRY PER FILING: the churn bug -----------------------------------
d = fresh()
first = buys(d.decide(ctx([row(insider=ins())])))
# Position closes (book no longer holds it) and the same filing is still fresh.
second = buys(d.decide(ctx([row(insider=ins())])))
check(len(first) == 1 and not second,
      "same filing is not re-entered after the position closes",
      "GME was bought six times in two days on one disclosure")

newer = ins()
newer["open_market_buys"]["most_recent"] = (
    datetime.now().date() + timedelta(days=1)).isoformat()   # a NEW filing date
check(len(buys(d.decide(ctx([row(insider=newer)])))) == 1,
      "a NEW filing on the same symbol does trigger")

# --- state survives a restart ----------------------------------------------
d = fresh()
d.decide(ctx([row(insider=ins())]))
d_after_restart = InsiderDecider()                        # loads STATE_PATH
check(d_after_restart._acted.get("BAC") == d._acted.get("BAC"),
      "acted-on filings persist across restart", str(d_after_restart._acted))
check(d_after_restart._entries.get("BAC", {}).get("atr") == 1.0,
      "entry barriers persist across restart")
check(not buys(d_after_restart.decide(ctx([row(insider=ins())]))),
      "restarted process does not re-enter the spent filing")

width = max(len(l) for _, l, _ in results)
failures = sum(1 for ok, _, _ in results if not ok)
for ok, label, detail in results:
    print(f"  [{'PASS' if ok else 'FAIL'}] {label:<{width}}  {detail}")
print(f"\n{len(results) - failures}/{len(results)} passed")
sys.exit(1 if failures else 0)
