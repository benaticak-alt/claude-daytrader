"""Broker adapter (Alpaca). Also the only module that talks to the trading API.

Refuses to construct a live client unless ALPACA_PAPER=false was set explicitly
in the environment — the default is always paper.
"""

from __future__ import annotations

import logging
from typing import List, Optional

from alpaca.trading.client import TradingClient
from alpaca.trading.enums import OrderSide, TimeInForce
from alpaca.trading.requests import MarketOrderRequest

import config
from models import TradeDecision

log = logging.getLogger(__name__)


def _as_float(value) -> Optional[float]:
    """Alpaca returns numerics as strings, and as None on fresh accounts."""
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _as_int(value) -> Optional[int]:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


class Executor:
    def __init__(self) -> None:
        if not config.ALPACA_PAPER:
            log.warning("LIVE TRADING MODE — ALPACA_PAPER=false was set explicitly")
        self._client = TradingClient(
            config.ALPACA_API_KEY, config.ALPACA_SECRET_KEY, paper=config.ALPACA_PAPER
        )

    # ---- account / positions ------------------------------------------------

    def account_state(self) -> dict:
        """Returns {} on failure so the risk gate fails closed.

        Alpaca leaves daytrade_count / pattern_day_trader as None on fresh
        paper accounts with no trading history, so every field is parsed
        defensively. daytrade_count is passed through as None rather than
        coerced to 0 — the gate only consults it below the PDT equity
        threshold, and there it must fail closed rather than assume zero.
        """
        try:
            acct = self._client.get_account()

            equity = _as_float(acct.equity)
            last_equity = _as_float(acct.last_equity)
            if equity is None:
                log.error("account returned no equity value")
                return {}

            return {
                "equity": equity,
                "cash": _as_float(acct.cash) or 0.0,
                # A fresh account can report no prior equity; treat P/L as flat.
                "daily_pl": equity - last_equity if last_equity is not None else 0.0,
                "daytrade_count": _as_int(acct.daytrade_count),   # may be None
                "pattern_day_trader": bool(acct.pattern_day_trader),
                "paper": config.ALPACA_PAPER,
            }
        except Exception:
            log.exception("account fetch failed")
            return {}  # risk gate fails closed on missing keys

    def open_positions(self) -> List[dict]:
        """Parsed per-position so one malformed field can't wipe the whole list.

        A position silently missing from this list would let the gate approve
        new entries as though the account were flat, so failures are loud.
        """
        try:
            raw = self._client.get_all_positions()
        except Exception:
            log.exception("positions fetch failed — treating as flat may be unsafe")
            return []

        positions: List[dict] = []
        for p in raw:
            try:
                pl_pct = _as_float(p.unrealized_plpc)
                positions.append({
                    "symbol": p.symbol,
                    "qty": _as_float(p.qty) or 0.0,
                    "avg_entry": _as_float(p.avg_entry_price) or 0.0,
                    "market_value": _as_float(p.market_value) or 0.0,
                    "unrealized_pl": _as_float(p.unrealized_pl) or 0.0,
                    "unrealized_pl_pct": round(100 * pl_pct, 2) if pl_pct is not None else 0.0,
                })
            except Exception:
                # Keep the symbol even if the numbers are unparseable — the gate
                # must still see that we hold it.
                log.exception("could not parse position %s; including symbol only",
                              getattr(p, "symbol", "?"))
                positions.append({"symbol": getattr(p, "symbol", "UNKNOWN")})
        return positions

    def market_open(self) -> bool:
        try:
            return bool(self._client.get_clock().is_open)
        except Exception:
            log.exception("clock fetch failed")
            return False

    def minutes_to_close(self) -> Optional[float]:
        """Minutes until the bell. None if unknown — callers must fail closed."""
        try:
            clock = self._client.get_clock()
            if not clock.is_open:
                return 0.0
            delta = clock.next_close - clock.timestamp
            return delta.total_seconds() / 60.0
        except Exception:
            log.exception("clock fetch failed")
            return None

    # ---- orders -------------------------------------------------------------

    def submit_entry(self, decision: TradeDecision, notional: float) -> Optional[str]:
        order = MarketOrderRequest(
            symbol=decision.symbol,
            notional=round(notional, 2),
            side=OrderSide.BUY,
            time_in_force=TimeInForce.DAY,
        )
        try:
            result = self._client.submit_order(order)
            log.info("ENTRY %s $%.0f order_id=%s", decision.symbol, notional, result.id)
            return str(result.id)
        except Exception:
            log.exception("entry order failed for %s", decision.symbol)
            return None

    def close_position(self, symbol: str) -> bool:
        try:
            self._client.close_position(symbol)
            log.info("CLOSE %s", symbol)
            return True
        except Exception:
            log.exception("close failed for %s", symbol)
            return False
