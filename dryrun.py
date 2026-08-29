"""Offline smoke test — runs the full decision chain with no API keys.

Exercises: context builder -> decider -> risk gate -> (mock) executor -> logging.
Uses synthetic bars and a fake broker, so it proves the *machinery* works
without touching Alpaca, Anthropic, or the network.

What this DOES prove: the pipeline wires together, the rule decider produces
valid decisions, the risk gate enforces its limits, sizing is capped, logging
writes correctly.

What it does NOT prove: anything about real market data quality, real order
fills, or whether the strategy makes money.

    python dryrun.py                    # rule backend (default)
    python dryrun.py --cycles 5
"""

from __future__ import annotations

import argparse
import logging
import random
from datetime import datetime, timedelta, timezone
from typing import List

import config

# Keep dry-run output out of the real decision log.
config.DECISION_LOG = config.LOG_DIR / "dryrun_decisions.jsonl"

from context_builder import build_context          # noqa: E402
from data_feed import EASTERN, Bar, SymbolSnapshot  # noqa: E402
from decider_orb import ORBDecider                 # noqa: E402
from decider_rule import RuleDecider               # noqa: E402
from models import TradeDecision                   # noqa: E402
from risk_gate import RiskGate                      # noqa: E402
from shadow import ShadowRunner                    # noqa: E402
from trade_log import log_cycle                    # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
log = logging.getLogger("dryrun")


class MockExecutor:
    """Fake broker. Records orders instead of sending them."""

    def __init__(self, equity: float = 100_000.0, minutes_to_close: float = 180.0) -> None:
        self.equity = equity
        self.daily_pl = 0.0
        self.daytrade_count = 0
        self.minutes_to_close = minutes_to_close
        self.positions: List[dict] = []
        self.orders: List[tuple[str, str, float]] = []

    def account_state(self) -> dict:
        return {
            "equity": self.equity,
            "cash": self.equity,
            "daily_pl": self.daily_pl,
            "daytrade_count": self.daytrade_count,
            "pattern_day_trader": False,
            "paper": True,
            # Mid-session: comfortably outside the end-of-day entry cutoff.
            "minutes_to_close": self.minutes_to_close,
        }

    def open_positions(self) -> List[dict]:
        return self.positions

    def submit_entry(self, decision: TradeDecision, notional: float) -> str:
        self.orders.append(("BUY", decision.symbol, notional))
        self.positions.append({
            "symbol": decision.symbol,
            "qty": 1.0,
            "avg_entry": 100.0,
            "market_value": notional,
            "unrealized_pl": 0.0,
            "unrealized_pl_pct": 0.0,
        })
        return f"mock-{len(self.orders)}"

    def close_position(self, symbol: str) -> bool:
        self.orders.append(("CLOSE", symbol, 0.0))
        self.positions = [p for p in self.positions if p["symbol"] != symbol]
        return True


def synth_snapshot(symbol: str, rng: random.Random, regime: str) -> SymbolSnapshot:
    """Generate a bar series in a named intraday regime.

    A constant drift is unrealistic — it produces either a runaway trend
    (RSI ~100) or a flat line, and never the moderate-momentum state the
    entry rule targets. Real intraday action trends, then consolidates.
    Each regime below is shaped to land in a specific indicator zone so the
    filter is genuinely exercised in both directions.
    """
    price = rng.uniform(50, 400)
    # Anchor synthetic bars to TODAY'S 09:30 ET open. Generating them relative
    # to "now" puts them outside regular hours whenever this is run after the
    # close, and session_bars would then be empty — no VWAP, no opening range.
    session_open = datetime.now(EASTERN).replace(
        hour=9, minute=30, second=0, microsecond=0
    )
    bars: List[Bar] = []

    # (n_bars, per-bar drift) legs. Values tuned empirically so each regime
    # lands in its intended indicator zone; sigma must exceed drift or RSI
    # saturates at 0/100, which real 5-minute bars never do.
    sigma = 0.0025
    if regime == "pullback_uptrend":       # -> RSI ~60, ~+1.0% vs VWAP: in band ~38% of draws
        legs = [(24, 0.0008), (5, -0.0015), (11, 0.0009)]
    elif regime == "overbought":           # -> RSI ~87: rejected
        legs = [(40, 0.0020)]
    elif regime == "downtrend":            # -> RSI ~15, below VWAP: rejected
        legs = [(40, -0.0016)]
    elif regime == "extended":             # -> ~+6% vs VWAP: rejected
        legs = [(20, 0.0004), (20, 0.0038)]
    else:                                  # chop -> RSI ~50, flat
        legs = [(40, 0.0)]

    for n_bars, drift in legs:
        for _ in range(n_bars):
            price = max(1.0, price * (1 + drift + rng.gauss(0, sigma)))
            high = price * (1 + abs(rng.gauss(0, 0.0008)))
            low = price * (1 - abs(rng.gauss(0, 0.0008)))
            bars.append(
                Bar(session_open + timedelta(minutes=5 * len(bars)), price, high, low,
                    price, rng.uniform(5e5, 2e6))
            )

    spread = price * 0.0002
    return SymbolSnapshot(
        symbol, bars, price - spread, price + spread, datetime.now(timezone.utc)
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cycles", type=int, default=3)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    rng = random.Random(args.seed)
    decider = RuleDecider()
    gate = RiskGate()
    executor = MockExecutor()
    # Run ORB in shadow alongside the live rule, exactly as the real loop does.
    shadow = ShadowRunner({"orb": ORBDecider()}, config.FLAT_POSITION_NOTIONAL)

    regimes = {
        "NVDA": "pullback_uptrend",   # should pass the filter
        "AAPL": "pullback_uptrend",   # should pass the filter
        "SPY":  "overbought",         # should be rejected (RSI too high)
        "QQQ":  "extended",           # should be rejected (too far above VWAP)
        "TSLA": "downtrend",          # should be rejected (below VWAP)
    }

    for cycle in range(1, args.cycles + 1):
        print(f"\n{'=' * 62}\nCYCLE {cycle}\n{'=' * 62}")

        snapshots = [synth_snapshot(s, rng, r) for s, r in regimes.items()]

        # Force a data-gate failure on one symbol to prove exclusion works.
        if cycle == 2:
            snapshots[0].data_ok = False
            snapshots[0].data_issues = ["stale quote (47s old)"]
            print(f"  [injected] {snapshots[0].symbol} fails the data gate this cycle")

        account = executor.account_state()
        positions = executor.open_positions()
        context_json = build_context(snapshots, account, positions)

        # Show the indicators the strategies key on, so filters are legible.
        import json as _json
        for row in _json.loads(context_json)["symbols"]:
            rsi = row.get("rsi_14")
            pv = row.get("pct_vs_vwap")
            orange = row.get("opening_range") or {}   # NB: not `rng` — that is the RNG
            rsi_s = f"RSI={rsi:5.1f}" if rsi is not None else "RSI=  n/a"
            pv_s = f"vs VWAP={pv:+6.2f}%" if pv is not None else "vs VWAP=   n/a"
            or_s = (f"ORhigh={orange['high']:.2f} complete={orange.get('complete')}"
                    if orange.get("high") is not None else "OR=none")
            print(f"    {row['symbol']:5s} {rsi_s}  {pv_s}  {or_s}   "
                  f"[{regimes[row['symbol']]}]")

        cycle_out = decider.decide(context_json)
        print(f"  read: {cycle_out.market_read}")

        shadow_out = shadow.run(context_json)
        for name, res in shadow_out.items():
            if res.get("events"):
                for e in res["events"]:
                    pnl = f"  pnl=${e['pnl']:+.2f}" if "pnl" in e else ""
                    print(f"  [shadow:{name}] {e['action'].upper()} {e['symbol']} "
                          f"@ {e['price']:.2f}{pnl}")

        results = []
        approved = 0
        for d in cycle_out.decisions:
            verdict = gate.evaluate(d, account, positions, approved)
            order_id = None
            if verdict.approved:
                if d.action == "close":
                    executor.close_position(d.symbol)
                    order_id = "closed"
                else:
                    order_id = executor.submit_entry(d, verdict.capped_notional)
                    approved += 1
            flag = "APPROVED" if verdict.approved else "REJECTED"
            size = f" ${verdict.capped_notional:,.0f}" if verdict.capped_notional else ""
            print(f"  {d.action.upper():6s} {d.symbol:5s} conf={d.confidence:.2f} "
                  f"-> {flag}{size}  ({'; '.join(verdict.reasons)})")
            results.append((d, verdict, order_id))

        log_cycle(context_json, cycle_out, results, shadow_out)

    print(f"\n{'=' * 62}\nSUMMARY\n{'=' * 62}")
    print(f"  orders placed : {len(executor.orders)}")
    for side, sym, notional in executor.orders:
        print(f"    {side:6s} {sym:5s} ${notional:,.0f}" if notional else f"    {side:6s} {sym}")
    print(f"  open positions: {[p['symbol'] for p in executor.positions]}")
    print(f"  decision log  : {config.DECISION_LOG}")
    print("\n  SHADOW STRATEGIES (logged, never executed):")
    print(shadow.report())


if __name__ == "__main__":
    main()
