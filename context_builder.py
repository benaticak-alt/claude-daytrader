"""Turns raw snapshots + account state into the compact context Claude reads.

Indicators are computed here in plain code — Claude gets the numbers, not the
raw bar arrays, so the prompt stays small and the arithmetic stays exact.
"""

from __future__ import annotations

import json
from typing import Dict, List, Optional

import config
from data_feed import EASTERN, SymbolSnapshot


def _rsi(closes: List[float], period: int = 14) -> Optional[float]:
    if len(closes) < period + 1:
        return None
    gains, losses = [], []
    for prev, cur in zip(closes[-period - 1:-1], closes[-period:]):
        change = cur - prev
        gains.append(max(change, 0.0))
        losses.append(max(-change, 0.0))
    avg_gain = sum(gains) / period
    avg_loss = sum(losses) / period
    if avg_loss == 0:
        return 100.0
    rs = avg_gain / avg_loss
    return 100.0 - 100.0 / (1.0 + rs)


def _atr(snap: SymbolSnapshot, period: int = 14) -> Optional[float]:
    bars = snap.bars
    if len(bars) < period + 1:
        return None
    trs = []
    for prev, cur in zip(bars[-period - 1:-1], bars[-period:]):
        tr = max(cur.high - cur.low, abs(cur.high - prev.close), abs(cur.low - prev.close))
        trs.append(tr)
    return sum(trs) / period


def _session_vwap(snap: SymbolSnapshot) -> Optional[float]:
    """VWAP over TODAY'S session only — it resets at the opening bell.

    `snap.bars` spans several days so the rolling indicators work at the open;
    using that full range here would produce a multi-day average that drifts
    further from the real session VWAP with every passing day of the week.
    """
    total_pv, total_v = 0.0, 0.0
    for b in snap.session_bars:
        typical = (b.high + b.low + b.close) / 3
        total_pv += typical * b.volume
        total_v += b.volume
    return total_pv / total_v if total_v > 0 else None


def _opening_range(snap: SymbolSnapshot) -> Optional[dict]:
    """High/low established in the first N minutes after the bell.

    Returns None before any session bars exist. `complete` is False while the
    range is still forming — a breakout is only meaningful once the window has
    actually closed, otherwise you are just trading the first tick.
    """
    session = snap.session_bars
    if not session:
        return None

    open_min = 9 * 60 + 30
    cutoff = open_min + config.OPENING_RANGE_MINUTES

    def et_minutes(bar) -> int:
        et = bar.ts.astimezone(EASTERN)
        return et.hour * 60 + et.minute

    or_bars = [b for b in session if et_minutes(b) < cutoff]
    if not or_bars:
        return None

    return {
        "high": round(max(b.high for b in or_bars), 4),
        "low": round(min(b.low for b in or_bars), 4),
        "bars": len(or_bars),
        # The window has closed once the newest bar sits at or past the cutoff.
        "complete": et_minutes(session[-1]) >= cutoff,
    }


def _ml_features(snap: SymbolSnapshot, closes: List[float], last: Optional[float],
                 atr: Optional[float]) -> dict:
    """Features the trained model consumes, computed with the SAME semantics as
    build_dataset.py. Any drift between here and training silently degrades the
    model — training-serving skew never announces itself. None where unknown;
    the model handles missing values natively.
    """
    out: dict = {}
    # Short-horizon momentum over the multi-day bar sequence, like pct_change(n).
    for n in (1, 3, 6):
        key = f"ret_{n}"
        out[key] = (
            round(100 * (closes[-1] - closes[-1 - n]) / closes[-1 - n], 4)
            if len(closes) > n and closes[-1 - n] else None
        )
    out["atr_pct"] = round(100 * atr / last, 4) if (atr and last) else None

    # Overnight gap: today's opening print vs the previous session's last close.
    session = snap.session_bars
    if session and last:
        first_today = session[0]
        prev = [b for b in snap.bars if b.ts < first_today.ts]
        # Previous session close = last bar before today's first bar.
        out["gap_pct"] = (
            round(100 * (first_today.open - prev[-1].close) / prev[-1].close, 4)
            if prev else None
        )
        et = session[-1].ts.astimezone(EASTERN)
        out["min_since_open"] = (et.hour * 60 + et.minute) - (9 * 60 + 30)
    else:
        out["gap_pct"] = None
        out["min_since_open"] = None
    return out


def symbol_context(snap: SymbolSnapshot) -> dict:
    closes = [b.close for b in snap.bars]           # multi-day: rolling indicators
    session = snap.session_bars                      # today only: VWAP, daily range
    session_closes = [b.close for b in session]
    last = closes[-1] if closes else None
    vwap = _session_vwap(snap)
    atr = _atr(snap)
    rsi = _rsi(closes)
    return {
        "symbol": snap.symbol,
        "last": last,
        "bid": snap.bid,
        "ask": snap.ask,
        "spread_pct": round(snap.spread_pct, 3),
        "session_vwap": round(vwap, 4) if vwap else None,
        "pct_vs_vwap": round(100 * (last - vwap) / vwap, 3) if (vwap and last) else None,
        "rsi_14": round(rsi, 1) if rsi is not None else None,
        "atr_14": round(atr, 4) if atr else None,
        # Today's range — computed from session bars so it isn't a multi-day span.
        "range_pct_today": round(100 * (max(session_closes) - min(session_closes)) / last, 2)
        if (session_closes and last) else None,
        "bars_today": len(session),
        "opening_range": _opening_range(snap),
        **_ml_features(snap, closes, last, atr),
        "last_5_closes": [round(c, 2) for c in closes[-5:]],
        "volume_last_bar": snap.bars[-1].volume if snap.bars else None,
        "avg_volume_20": round(sum(b.volume for b in snap.bars[-20:]) / min(20, len(snap.bars)), 0)
        if snap.bars else None,
    }


def build_context(
    snapshots: List[SymbolSnapshot],
    account: dict,
    positions: List[dict],
    insider: Optional[dict] = None,
) -> str:
    """insider: {symbol -> summary dict} from InsiderFeed, or None if disabled."""
    symbols = []
    for s in snapshots:
        if not s.data_ok:
            continue
        ctx = symbol_context(s)
        if insider and s.symbol in insider:
            ctx["insider_form4"] = insider[s.symbol]
        symbols.append(ctx)

    # The moment this snapshot represents, in ET. Anything time-dependent must
    # read THIS rather than the wall clock: during a backtest the wall clock is
    # whenever the replay happens to run, which silently produced two years of
    # results using a single strategy instead of the scheduled rotation.
    as_of = None
    for s in snapshots:
        if s.data_ok and s.bars:
            as_of = s.bars[-1].ts.astimezone(EASTERN).isoformat()
            break

    payload = {
        "as_of_et": as_of,
        "account": account,
        "open_positions": positions,
        "symbols": symbols,
        "excluded_symbols": [
            {"symbol": s.symbol, "issues": s.data_issues} for s in snapshots if not s.data_ok
        ],
    }
    return json.dumps(payload, indent=2, default=str)
