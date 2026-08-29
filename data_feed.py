"""Market data layer with a hard data-quality gate.

The gate exists because a model reasoning over stale or synthetic quotes
produces fictional edge (see: the Polymarket bot's Gamma-fallback incident,
where half the P&L turned out to be a data artifact). A symbol that fails
the freshness/spread checks is dropped from the cycle entirely — Claude
never sees it.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional
from zoneinfo import ZoneInfo

EASTERN = ZoneInfo("America/New_York")

from alpaca.data.enums import DataFeed as AlpacaFeed  # aliased: our class is DataFeed too
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest, StockLatestQuoteRequest
from alpaca.data.timeframe import TimeFrame, TimeFrameUnit

import config

log = logging.getLogger(__name__)


@dataclass
class Bar:
    ts: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float
    vwap: Optional[float] = None


@dataclass
class SymbolSnapshot:
    symbol: str
    bars: List[Bar]
    bid: float
    ask: float
    quote_ts: datetime
    data_ok: bool = True
    data_issues: List[str] = field(default_factory=list)

    @property
    def mid(self) -> float:
        return (self.bid + self.ask) / 2

    @property
    def spread_pct(self) -> float:
        return 100.0 * (self.ask - self.bid) / self.mid if self.mid > 0 else float("inf")

    @property
    def session_bars(self) -> List[Bar]:
        """Today's REGULAR-hours bars only — 09:30 up to (not including) 16:00 ET.

        `bars` deliberately spans several days so RSI/ATR work at the open, but
        VWAP is a session measure: anchored at the opening bell and reset daily.
        Computing it over multi-day history produces a number that means nothing
        and drifts further from reality as the week goes on.

        Both ends must be trimmed. Extended-hours bars are thin and erratic —
        one QQQ sample carried bars out to 16:50, which would drag session VWAP
        and the daily range toward prices no RTH order ever filled at.

        Caveat: 16:00 is hardcoded, so early-close half-days (Thanksgiving Friday,
        Christmas Eve) will include an hour or two of post-close bars. Use the
        broker clock here if that ever matters.
        """
        if not self.bars:
            return []
        last_et = self.bars[-1].ts.astimezone(EASTERN)
        session_date = last_et.date()
        out = []
        for bar in self.bars:
            et = bar.ts.astimezone(EASTERN)
            if et.date() != session_date:
                continue
            minutes = et.hour * 60 + et.minute
            if minutes < 9 * 60 + 30:
                continue   # pre-market
            if minutes >= 16 * 60:
                continue   # post-market
            out.append(bar)
        return out


class DataFeed:
    def __init__(self) -> None:
        self._client = StockHistoricalDataClient(
            config.ALPACA_API_KEY, config.ALPACA_SECRET_KEY
        )
        self._feed = (
            AlpacaFeed.SIP if config.ALPACA_DATA_FEED == "sip" else AlpacaFeed.IEX
        )
        if self._feed is AlpacaFeed.IEX:
            log.info("using IEX feed (free tier) — partial view of national volume")

    def snapshot(self, symbol: str) -> SymbolSnapshot:
        quote_req = StockLatestQuoteRequest(symbol_or_symbols=symbol, feed=self._feed)
        quote = self._client.get_stock_latest_quote(quote_req)[symbol]

        end = datetime.now(timezone.utc)
        # Calendar days, not bar count — must span weekends/holidays so a Monday
        # morning still has prior-session history for the rolling indicators.
        start = end - timedelta(days=config.BAR_HISTORY_DAYS)
        bars_req = StockBarsRequest(
            symbol_or_symbols=symbol,
            timeframe=TimeFrame(config.BAR_TIMEFRAME_MINUTES, TimeFrameUnit.Minute),
            start=start,
            end=end,
            feed=self._feed,
        )
        raw_bars = self._client.get_stock_bars(bars_req).data.get(symbol, [])
        bars = [
            Bar(b.timestamp, b.open, b.high, b.low, b.close, b.volume, getattr(b, "vwap", None))
            for b in raw_bars
        ]
        # Keep enough history for the rolling windows, but never trim away
        # today's session — the tail must always cover the current day.
        if len(bars) > config.BAR_LOOKBACK:
            bars = bars[-config.BAR_LOOKBACK:]

        snap = SymbolSnapshot(
            symbol=symbol,
            bars=bars,
            bid=float(quote.bid_price or 0),
            ask=float(quote.ask_price or 0),
            quote_ts=quote.timestamp,
        )
        self._validate(snap)
        return snap

    def _validate(self, snap: SymbolSnapshot) -> None:
        now = datetime.now(timezone.utc)

        if snap.bid <= 0 or snap.ask <= 0 or snap.ask < snap.bid:
            snap.data_issues.append(f"degenerate quote bid={snap.bid} ask={snap.ask}")

        age = (now - snap.quote_ts).total_seconds() if snap.quote_ts else float("inf")
        if age > config.MAX_QUOTE_AGE_SECONDS:
            snap.data_issues.append(f"stale quote ({age:.0f}s old)")

        max_spread = (
            config.MAX_SPREAD_PCT_IEX
            if config.ALPACA_DATA_FEED == "iex"
            else config.MAX_SPREAD_PCT
        )
        if snap.mid > 0 and snap.spread_pct > max_spread:
            snap.data_issues.append(f"wide spread ({snap.spread_pct:.2f}% > {max_spread}%)")

        if len(snap.bars) < config.MIN_BARS_FOR_INDICATORS:
            snap.data_issues.append(
                f"insufficient bars ({len(snap.bars)} < {config.MIN_BARS_FOR_INDICATORS})"
            )
        elif len(snap.session_bars) < config.MIN_SESSION_BARS:
            # Right at the bell there is history for RSI/ATR but not enough of
            # today for a meaningful VWAP.
            snap.data_issues.append(
                f"session just opened ({len(snap.session_bars)} bars today)"
            )

        if snap.data_issues and config.REQUIRE_LIVE_QUOTE:
            snap.data_ok = False
            log.warning("%s failed data gate: %s", snap.symbol, "; ".join(snap.data_issues))
