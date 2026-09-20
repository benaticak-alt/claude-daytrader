"""Insider-buy decider — trades fresh SEC Form 4 open-market purchases.

The only signal in this project that survived a symbol- and period-matched
control. Re-measured 2026-09-17 on the FULL universe (65,844 events, 5,764
symbols incl. 1,800 later-delisted, entry at the next open after filing):
excess over control +$12/trade per $1,000, t=+10 on non-overlapping events.
Small per event; real. It is a different animal from the intraday strategies:

    universe   every issuer on EDGAR — events are rare per name
    cadence    daily; a filing is news for days, not seconds
    trigger    an event, not a price pattern
    hold       21 trading days, or ±1.5 DAILY ATR
    book       up to INSIDER_MAX_POSITIONS, INSIDER_EXPOSURE of equity in total

FILTERS (all pre-registered from the literature, then checked in the study —
none tuned on the outcome):
    liquidity   20d avg $ volume >= INSIDER_MIN_DOLLAR_VOL (thin names: fills
                are fiction and the control is noisy)
    size        purchase >= INSIDER_MIN_BUY_USD (t rose monotonically with size)
    role        officer or director; 10%-owner-only filers excluded (t=+0.9)
    routine     excluded — an insider who trades the same month every year is
                on a schedule, not on information (routine excess +$1, t=-0.8;
                everyone else +$8..12, t=+6..9)
    10b5-1      excluded — plan trades are scheduled months ahead (~3% of buys)
    discount    price paid within INSIDER_MAX_DISCOUNT_PCT of market — a deep
                discount is a placement coded P, not a vote of confidence

NO MODEL. The trained classifier scored AUC 0.473 — below random. There is
no skill in selecting among qualifying events, so this takes them all,
ranked only so the position cap spends itself on the strongest roles.

WHY THIS RUNS IN SHADOW: the account-level simulation (insider_portfolio_sim)
at half exposure did ~16% CAGR with a 37% drawdown — 94 straight dip-buys into
March 2020 at a 10% win rate. SPY did 13.5% / -34% over the same window. The
per-event edge is real; whether a capital-constrained book harvests enough of
it to beat the index is exactly what the forward test is for. The `regime`
variant (no new entries while SPY is below its 50-day average) cut the
drawdown to 23% in simulation; it is run as a SECOND shadow, not assumed.

FILING DATE IS THE ONLY TRADEABLE DATE. Form 4 allows two business days after
the transaction, so freshness is keyed on the filing, never the transaction.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import config
from models import CycleDecisions, TradeDecision

log = logging.getLogger(__name__)

STATE_DIR = Path(__file__).parent / "cache"

TP_ATR = 1.5
SL_ATR = 1.5
HOLD_TRADING_DAYS = 21
# A filing is actionable for a few days — the disclosure is the event, and the
# documented drift plays out over weeks, so there is no need to catch the tick.
MAX_FILING_AGE_DAYS = config.INSIDER_MAX_FILING_AGE_DAYS


class InsiderDecider:
    """Same interface as the other deciders: .decide(context_json).

    regime=True adds the SPY-above-50d-MA entry permission. Both variants share
    the universe feed; each keeps its own state file so their books stay
    independent in shadow.
    """

    # The end-of-day flatten exists because the intraday strategies size for
    # ~0.1% moves and must never eat an overnight gap. This strategy's edge is
    # measured over 21 TRADING DAYS — flattening it nightly would destroy the
    # very thing being tested. Overnight gap risk is accepted here by design,
    # and it is priced into the ±1.5 ATR barriers the effect was measured with.
    holds_overnight = True

    def __init__(self, regime: bool = False, name: str = "insider") -> None:
        self.regime = regime
        self.name = name
        self.state_path = STATE_DIR / f"{name}_decider_state.json"
        # Per-strategy risk limits, read by main and the shadow runner. The
        # intraday defaults (3 positions, $2k) would make the book untestable.
        self.gate_params = {
            "max_positions": config.INSIDER_MAX_POSITIONS,
            "size_pct_equity": config.INSIDER_EXPOSURE / config.INSIDER_MAX_POSITIONS,
            "max_position_notional": 1e9,     # the half-of-cash rule still binds
        }
        # symbol -> {"atr": entry DAILY ATR, "ts": entry time iso, "key": event}.
        # Persisted: a 21-day hold outlives any single process.
        self._entries: Dict[str, dict] = {}
        # symbol -> event key already acted on. ONE ENTRY PER FILING. Without
        # this the strategy re-enters the same disclosure every time the
        # previous position closes — observed live: GME six times in two days.
        self._acted: Dict[str, str] = {}
        self._load_state()

    # ------------------------------------------------------------------
    def _load_state(self) -> None:
        try:
            if self.state_path.exists():
                st = json.loads(self.state_path.read_text(encoding="utf-8"))
                self._entries = st.get("entries", {})
                self._acted = st.get("acted", {})
                log.info("%s state loaded: %d open entries, %d filings acted on",
                         self.name, len(self._entries), len(self._acted))
        except Exception:
            log.exception("could not load %s state — starting fresh", self.name)

    def _save_state(self) -> None:
        try:
            self.state_path.parent.mkdir(parents=True, exist_ok=True)
            self.state_path.write_text(
                json.dumps({"entries": self._entries, "acted": self._acted}),
                encoding="utf-8",
            )
        except Exception:
            log.exception("could not save %s state", self.name)

    # ------------------------------------------------------------------
    def decide(self, context_json: str) -> CycleDecisions:
        ctx = json.loads(context_json)
        rows = {r["symbol"]: r for r in ctx.get("symbols", [])}
        positions = ctx.get("open_positions", [])
        held = {p["symbol"] for p in positions}
        as_of = ctx.get("as_of_et")

        changed = False
        for sym in list(self._entries):
            if sym not in held:
                self._entries.pop(sym, None)
                changed = True

        decisions: List[TradeDecision] = []
        for pos in positions:
            d = self._maybe_exit(pos, rows.get(pos["symbol"]), as_of)
            if d:
                decisions.append(d)

        regime_ok, regime_note = self._regime(rows)

        # Candidates from the universe feed (preferred) and, failing that, the
        # per-symbol summaries the older feed attaches to watchlist rows.
        candidates: List[tuple] = []
        seen_syms: set = set()
        for ev in ctx.get("insider_events") or []:
            sym = ev.get("symbol")
            if not sym or sym in held or sym in seen_syms or sym not in rows:
                continue
            scored = self._score_event(ev, rows[sym])
            if scored:
                candidates.append(scored)
                seen_syms.add(sym)
        for sym, row in rows.items():
            if sym in held or sym in seen_syms:
                continue
            scored = self._score_legacy(row)
            if scored:
                candidates.append(scored)

        # Rank by conviction so the position cap spends itself on the strongest
        # events, but take every one that qualifies — there is no selection model.
        candidates.sort(key=lambda c: c[0], reverse=True)
        free = max(0, config.INSIDER_MAX_POSITIONS - len(held))
        entered = 0
        skipped_regime = 0
        for confidence, sym, note, key in candidates:
            if entered >= free:
                break
            if not regime_ok:
                skipped_regime += 1
                continue
            row = rows[sym]
            atr_d = row.get("atr_daily")
            if not atr_d:
                # No daily ATR means no correctly-sized barriers. Skip rather
                # than fall back to the intraday ATR, which is the exact unit
                # error that made the first forward test meaningless.
                log.warning("%s: insider event but no daily ATR — skipping", sym)
                continue
            decisions.append(TradeDecision(
                symbol=sym,
                action="buy",
                confidence=confidence,
                thesis=f"Insider open-market buy: {note}",
                invalidation=(
                    f"-{SL_ATR:g} DAILY ATR from entry, +{TP_ATR:g} daily ATR target, "
                    f"or {HOLD_TRADING_DAYS} trading days elapsed."
                ),
                suggested_notional=config.FLAT_POSITION_NOTIONAL,   # advisory; gate sizes
                time_horizon_minutes=HOLD_TRADING_DAYS * 390,
            ))
            self._entries[sym] = {"atr": atr_d, "ts": as_of, "key": key}
            self._acted[sym] = key
            entered += 1
            changed = True

        if changed:
            self._save_state()

        n_events = len(ctx.get("insider_events") or [])
        read = (f"Insider[{self.name}]: {n_events} fresh filings, {len(candidates)} pass "
                f"filters, {entered} entered, {len(held)} held")
        if skipped_regime:
            read += f", {skipped_regime} blocked by regime ({regime_note})"
        return CycleDecisions(market_read=read, decisions=decisions)

    # ------------------------------------------------------------------
    def _regime(self, rows: dict) -> tuple[bool, str]:
        """Entry permission. Non-regime variant: always. Regime variant: SPY's
        last close above its 50-day average; unknown fails CLOSED — a strategy
        whose worst month was a crash must not add risk blind."""
        if not self.regime:
            return True, ""
        spy = rows.get("SPY") or {}
        dist = spy.get("dist_ma50_pct")
        if dist is None:
            return False, "SPY 50d MA unknown"
        return dist > 0, f"SPY {dist:+.1f}% vs 50d MA"

    # ------------------------------------------------------------------
    def _score_event(self, ev: dict, row: dict) -> Optional[tuple]:
        """Apply the pre-registered filters to a universe-feed event."""
        sym = ev["symbol"]
        key = ev.get("accession") or f"{sym}:{ev.get('filing_date')}"
        if self._acted.get(sym) == key:
            return None
        if not self._is_fresh(ev.get("filing_date")):
            return None
        if (ev.get("buy_usd") or 0) < config.INSIDER_MIN_BUY_USD:
            return None
        if not (ev.get("is_officer") or ev.get("is_director")):
            return None
        if ev.get("trader_type") == "routine":
            return None
        if ev.get("is_10b5_1"):
            return None          # pre-scheduled plan trade: no information by construction
        dv = row.get("dollar_vol_20")
        if dv is None or dv < config.INSIDER_MIN_DOLLAR_VOL:
            return None
        paid, last = ev.get("price_paid"), row.get("last")
        if paid and last:
            # Compared against the current price rather than the transaction-day
            # close (not in the context). A stock that has moved >10% since the
            # insider bought is also not the setup the study measured, so the
            # stricter reading is acceptable.
            if abs(paid / last - 1) * 100 > config.INSIDER_MAX_DISCOUNT_PCT:
                return None

        # Conviction ordering follows the measured effect sizes — C-suite and
        # other officers above directors, size as a tiebreak. Every value clears
        # the gate floor; the ordering only decides who gets a slot on a busy day.
        confidence = 0.68
        if ev.get("is_csuite"):
            confidence += 0.08
        elif ev.get("is_officer"):
            confidence += 0.06
        if ev.get("trader_type") == "opportunistic":
            confidence += 0.03
        if (ev.get("buy_usd") or 0) >= 1_000_000:
            confidence += 0.03
        confidence = round(min(confidence, 0.90), 2)

        role = ("C-SUITE" if ev.get("is_csuite") else
                "officer" if ev.get("is_officer") else "director")
        note = (f"{role} {ev.get('owner_name', '')[:24]} bought ${ev.get('buy_usd', 0):,.0f} "
                f"({ev.get('trader_type', 'unclassified')}), filed {ev.get('filing_date')}")
        return confidence, sym, note, key

    def _score_legacy(self, row: dict) -> Optional[tuple]:
        """Per-symbol summary path (insider_feed.py). Kept so the watchlist
        still produces events when the universe feed is disabled; the summary
        lacks owner identity, so the role/routine filters cannot apply here."""
        ins = row.get("insider_form4")
        if not ins or ins.get("insider_data") == "unavailable":
            return None
        buys = ins.get("open_market_buys") or {}
        n = buys.get("transactions") or 0
        if n <= 0:
            return None
        recent = ins.get("filings_today") or 0
        most_recent = buys.get("most_recent")
        if not recent and not self._is_fresh(most_recent):
            return None
        key = str(most_recent or datetime.now().date().isoformat())
        if self._acted.get(row["symbol"]) == key:
            return None
        usd = buys.get("total_usd") or 0
        if usd < config.INSIDER_MIN_BUY_USD:
            return None
        dv = row.get("dollar_vol_20")
        if dv is not None and dv < config.INSIDER_MIN_DOLLAR_VOL:
            return None

        insiders = buys.get("distinct_insiders") or 0
        cluster = bool(ins.get("cluster_buy"))
        csuite = bool(ins.get("c_suite_buy"))
        confidence = 0.68
        if cluster:
            confidence += 0.10
        if csuite:
            confidence += 0.08
        if insiders >= 2:
            confidence += 0.03
        if usd >= 1_000_000:
            confidence += 0.02
        confidence = round(min(confidence, 0.90), 2)
        note = (f"{n} buy(s) by {insiders} insider(s), ${usd:,.0f} total"
                + (", CLUSTER" if cluster else "") + (", C-SUITE" if csuite else "")
                + (f", {recent} filed today" if recent else ""))
        return confidence, row["symbol"], note, key

    @staticmethod
    def _is_fresh(filing_date: Optional[str]) -> bool:
        """True if the FILING is within the actionable window."""
        if not filing_date:
            return False
        try:
            d = datetime.fromisoformat(str(filing_date)).date()
        except ValueError:
            return False
        return (datetime.now().date() - d).days <= MAX_FILING_AGE_DAYS

    # ------------------------------------------------------------------
    def _maybe_exit(self, pos: dict, row: Optional[dict],
                    as_of: Optional[str]) -> Optional[TradeDecision]:
        if row is None:
            return None
        last, entry = row.get("last"), pos.get("avg_entry")
        if not last or not entry:
            return None

        rec = self._entries.get(pos["symbol"], {})
        atr = rec.get("atr") or row.get("atr_daily")
        if not atr:
            return None

        reason = None
        if last >= entry + TP_ATR * atr:
            reason = f"target: {last:.2f} >= {entry:.2f} + {TP_ATR:g} ATR"
        elif last <= entry - SL_ATR * atr:
            reason = f"stop: {last:.2f} <= {entry:.2f} - {SL_ATR:g} ATR"
        elif rec.get("ts") and as_of:
            try:
                days = (datetime.fromisoformat(as_of)
                        - datetime.fromisoformat(rec["ts"])).days
                if days >= HOLD_TRADING_DAYS * 1.45:      # trading -> calendar
                    reason = f"hold expired: {days} calendar days"
            except ValueError:
                pass

        if reason is None:
            return None
        return TradeDecision(
            symbol=pos["symbol"], action="close", confidence=0.70,
            thesis=f"Insider exit — {reason}.",
            invalidation="n/a — this is an exit",
            suggested_notional=0.0, time_horizon_minutes=5,
        )
