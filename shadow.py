"""Shadow strategies — run several deciders in parallel, trade only one.

Each shadow keeps its OWN virtual position book. That matters: if every
strategy saw the live account's positions, a shadow's exit logic would fire on
trades it never made, and its record would be meaningless. Here each one gets a
context with its own holdings substituted, so its decisions are self-consistent
and its P&L is genuinely its own.

Nothing here can place an order. Shadows are logged, never executed, and any
exception inside one is swallowed so a broken experiment cannot disturb live
trading.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import config
from models import CycleDecisions

log = logging.getLogger(__name__)


@dataclass
class VirtualPosition:
    symbol: str
    qty: float
    entry_price: float


@dataclass
class ShadowBook:
    """A strategy's imaginary portfolio."""

    name: str
    notional: float = 1000.0
    positions: Dict[str, VirtualPosition] = field(default_factory=dict)
    realized_pnl: float = 0.0
    wins: int = 0
    losses: int = 0
    trades: int = 0

    def as_context_positions(self) -> List[dict]:
        """Shaped like the live account's open_positions, so deciders can't tell."""
        return [
            {
                "symbol": p.symbol,
                "qty": p.qty,
                "avg_entry": p.entry_price,
                "market_value": p.qty * p.entry_price,
                "unrealized_pl": 0.0,
                "unrealized_pl_pct": 0.0,
            }
            for p in self.positions.values()
        ]

    def apply(self, decisions, prices: Dict[str, float]) -> List[dict]:
        """Execute decisions against the virtual book. Returns a log record.

        Fills are charged SHADOW_SLIPPAGE_BPS per side. Filling at the midpoint
        would flatter every strategy — real orders cross the spread, and at the
        ~0.17% average loss seen live, a few bps is a real share of the outcome.
        """
        slip = config.SHADOW_SLIPPAGE_BPS / 10_000.0
        events = []
        for d in decisions:
            mid = prices.get(d.symbol)
            if mid is None or mid <= 0:
                continue

            if d.action == "close" and d.symbol in self.positions:
                fill = mid * (1 - slip)          # sell into the bid
                pos = self.positions.pop(d.symbol)
                pnl = (fill - pos.entry_price) * pos.qty
                self.realized_pnl += pnl
                self.trades += 1
                if pnl >= 0:
                    self.wins += 1
                else:
                    self.losses += 1
                events.append({
                    "action": "close", "symbol": d.symbol, "price": round(fill, 4),
                    "entry": round(pos.entry_price, 4), "pnl": round(pnl, 2),
                })

            elif d.action == "buy" and d.symbol not in self.positions:
                fill = mid * (1 + slip)          # buy at the offer
                qty = self.notional / fill
                self.positions[d.symbol] = VirtualPosition(d.symbol, qty, fill)
                events.append({
                    "action": "buy", "symbol": d.symbol, "price": round(fill, 4),
                    "confidence": d.confidence,
                })
        return events

    def flatten(self, prices: Dict[str, float]) -> List[dict]:
        """Force-close everything, mirroring the live end-of-day flatten.

        Without this, shadow strategies hold overnight — something the real
        system forbids — and their P&L would include gap moves the live bot
        could never have experienced.
        """
        slip = config.SHADOW_SLIPPAGE_BPS / 10_000.0
        events = []
        for symbol in list(self.positions):
            mid = prices.get(symbol)
            if mid is None or mid <= 0:
                continue
            fill = mid * (1 - slip)
            pos = self.positions.pop(symbol)
            pnl = (fill - pos.entry_price) * pos.qty
            self.realized_pnl += pnl
            self.trades += 1
            if pnl >= 0:
                self.wins += 1
            else:
                self.losses += 1
            events.append({
                "action": "eod_flatten", "symbol": symbol, "price": round(fill, 4),
                "entry": round(pos.entry_price, 4), "pnl": round(pnl, 2),
            })
        return events

    def unrealized(self, prices: Dict[str, float]) -> float:
        total = 0.0
        for p in self.positions.values():
            price = prices.get(p.symbol)
            if price:
                total += (price - p.entry_price) * p.qty
        return total

    def summary(self, prices: Dict[str, float]) -> dict:
        return {
            "realized_pnl": round(self.realized_pnl, 2),
            "unrealized_pnl": round(self.unrealized(prices), 2),
            "closed_trades": self.trades,
            "wins": self.wins,
            "losses": self.losses,
            "open": sorted(self.positions),
        }


class ShadowRunner:
    """Runs non-executing deciders alongside the live one.

    Shadow decisions pass through the SAME risk gate as live orders. Without
    that, a shadow strategy trades under rules the real system would never
    allow — sub-floor confidence, more than the position cap, entries inside
    the end-of-day cutoff — and its record measures a strategy that could not
    actually be run. The backtest has always gated; this path did not, which
    made the two silently incomparable.
    """

    def __init__(self, deciders: Dict[str, object], notional: float = 1000.0) -> None:
        self.deciders = deciders
        self.books = {name: ShadowBook(name, notional) for name in deciders}
        from risk_gate import RiskGate
        self.gate = RiskGate()

    def run(self, context_json: str) -> dict:
        """Returns {strategy: {decisions, events, summary}} for logging."""
        base = json.loads(context_json)
        prices = {
            r["symbol"]: r["last"]
            for r in base.get("symbols", [])
            if r.get("last")
        }

        out: dict = {}
        for name, decider in self.deciders.items():
            book = self.books[name]
            try:
                # Substitute this strategy's own holdings before it decides.
                scoped = dict(base)
                scoped["open_positions"] = book.as_context_positions()
                result: CycleDecisions = decider.decide(json.dumps(scoped))

                # Gate every decision exactly as the live path would, against
                # THIS strategy's own book. daily_pl is passed as 0.0: shadow
                # books do not track per-day realized P&L, and at these sizes
                # the daily-loss limit would need ~100 losing trades in one
                # session to bind, so it never gates in practice.
                account = dict(base.get("account") or {})
                account.setdefault("equity", 100_000.0)
                account.setdefault("cash", 100_000.0)
                account.setdefault("daytrade_count", 0)
                account["daily_pl"] = 0.0

                approved, rejected, n_new = [], [], 0
                for d in result.decisions:
                    verdict = self.gate.evaluate(
                        d, account, book.as_context_positions(), n_new
                    )
                    if verdict.approved:
                        approved.append(d)
                        if d.action == "buy":
                            n_new += 1
                    else:
                        rejected.append({
                            "action": d.action, "symbol": d.symbol,
                            "confidence": d.confidence,
                            "why": "; ".join(verdict.reasons),
                        })

                events = book.apply(approved, prices)
                out[name] = {
                    "market_read": result.market_read,
                    "proposed": [
                        {"action": d.action, "symbol": d.symbol,
                         "confidence": d.confidence}
                        for d in result.decisions
                    ],
                    "gate_rejected": rejected,
                    "events": events,
                    "summary": book.summary(prices),
                }
                if events:
                    log.info("shadow[%s]: %s", name,
                             "; ".join(f"{e['action']} {e['symbol']}"
                                       + (f" pnl={e['pnl']:+.2f}" if "pnl" in e else "")
                                       for e in events))
            except Exception:
                # A broken experiment must never disturb live trading.
                log.exception("shadow decider %s failed — ignoring", name)
                out[name] = {"error": True}
        return out

    def flatten_all(self, context_json: str) -> dict:
        """End-of-day flatten across every shadow book, mirroring the live rule."""
        base = json.loads(context_json)
        prices = {
            r["symbol"]: r["last"]
            for r in base.get("symbols", []) if r.get("last")
        }
        out = {}
        for name, book in self.books.items():
            # Multi-day strategies opt out: flattening a 21-day hold every
            # evening would destroy the effect being measured.
            if getattr(self.deciders.get(name), "holds_overnight", False):
                out[name] = {"events": [], "summary": book.summary(prices),
                             "held_overnight": True}
                continue
            events = book.flatten(prices)
            if events:
                log.info("shadow[%s] EOD flatten: %s", name,
                         "; ".join(f"{e['symbol']} pnl={e['pnl']:+.2f}" for e in events))
            out[name] = {"events": events, "summary": book.summary(prices)}
        return out

    def report(self, prices: Optional[Dict[str, float]] = None) -> str:
        prices = prices or {}
        lines = []
        for name, book in self.books.items():
            s = book.summary(prices)
            total = s["realized_pnl"] + s["unrealized_pnl"]
            wr = (100 * s["wins"] / s["closed_trades"]) if s["closed_trades"] else 0.0
            lines.append(
                f"  {name:8s} total ${total:+8.2f}  realized ${s['realized_pnl']:+8.2f}  "
                f"trades {s['closed_trades']:3d}  win rate {wr:5.1f}%  open {s['open']}"
            )
        return "\n".join(lines)
