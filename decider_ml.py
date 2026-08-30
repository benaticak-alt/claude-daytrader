"""ML signal decider — trades the trained model's probability, or refuses to.

Loads models/signal_model.joblib (produced by train_model.py). Two hard rules:

  * THE MODEL'S OWN VERDICT IS BINDING. train_model.py embeds
    `deployable: False` when no threshold cleared $0/trade out-of-sample, and
    this decider then refuses to propose entries no matter what. A model that
    failed its evaluation cannot be traded by accident.

  * EXITS MIRROR THE TRAINING BARRIERS. The model estimates P(hit +1 ATR
    before -1 ATR within 12 bars). Managing the position any other way makes
    the probability meaningless — so exits are exactly: target +1 ATR, stop
    -1 ATR, horizon 12 bars, with the EOD flatten as backstop.

Entry ATR and entry time are tracked in memory; after a restart they are
unknown, so the position falls back to barrier management with the current
ATR and no horizon exit (documented degradation, EOD flatten still applies).

Confidence: the risk gate's floor is MIN_CONFIDENCE; raw probabilities live
near 0.5, so the model's probability is mapped linearly from
[threshold, 1.0] onto [floor+0.01, 0.90]. The raw probability is logged in
the thesis for calibration analysis.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
from joblib import load

import config
from models import CycleDecisions, TradeDecision

log = logging.getLogger(__name__)

MODEL_PATH = Path(__file__).parent / "models" / "signal_model.joblib"
# Fallback geometry only. A loaded model's OWN metadata overrides these —
# exits must use the same barriers the probability was trained on, whatever
# they were; managing a ±2 ATR probability with ±1 ATR exits invalidates it.
HORIZON_MINUTES = 60
TP_ATR = 1.0
SL_ATR = 1.0


class MLDecider:
    """Same interface as the other deciders: .decide(context_json)."""

    def __init__(self) -> None:
        self.model = None
        self.meta: dict = {}
        self.tp, self.sl, self.horizon = TP_ATR, SL_ATR, HORIZON_MINUTES
        # symbol -> {"atr": entry ATR, "ts": entry time iso} (lost on restart)
        self._entries: Dict[str, dict] = {}

        if not MODEL_PATH.exists():
            log.error("no trained model at %s — ML decider will only hold", MODEL_PATH)
            return
        bundle = load(MODEL_PATH)
        self.meta = bundle["meta"]
        self.tp = float(self.meta.get("tp_atr", TP_ATR))
        self.sl = float(self.meta.get("sl_atr", SL_ATR))
        self.horizon = int(self.meta.get("horizon_minutes", HORIZON_MINUTES))
        if not self.meta.get("deployable"):
            log.error(
                "model at %s is marked NOT DEPLOYABLE by its own walk-forward "
                "evaluation — refusing to trade it. Retrain until it clears $0.",
                MODEL_PATH,
            )
            return
        self.model = bundle["model"]
        log.info(
            "ML model loaded: threshold %.2f, expected %+.3f $/trade OOS, AUC %.3f",
            self.meta["threshold"], self.meta["expected_dollars_per_trade"],
            self.meta["oos_auc"],
        )

    # ------------------------------------------------------------------
    def _features(self, row: dict) -> Optional[pd.DataFrame]:
        rng = row.get("opening_range") or {}
        complete = bool(rng.get("complete"))
        high, low = rng.get("high"), rng.get("low")
        vol, avg = row.get("volume_last_bar"), row.get("avg_volume_20")
        last = row.get("last")
        vals = {
            "pct_vs_vwap": row.get("pct_vs_vwap"),
            "rsi_14": row.get("rsi_14"),
            "atr_pct": row.get("atr_pct"),
            "vol_ratio": (vol / avg) if (vol and avg) else None,
            "pct_vs_or_high": (100 * (last - high) / high)
            if (complete and last and high) else None,
            "or_range_pct": (100 * (high - low) / low)
            if (complete and high and low) else None,
            "after_or": 1 if complete else 0,
            "ret_1": row.get("ret_1"),
            "ret_3": row.get("ret_3"),
            "ret_6": row.get("ret_6"),
            "gap_pct": row.get("gap_pct"),
            "min_since_open": row.get("min_since_open"),
        }
        feats = self.meta.get("features") or list(vals)
        if any(f not in vals for f in feats):
            log.error("model expects features the context lacks: %s",
                      [f for f in feats if f not in vals])
            return None
        return pd.DataFrame([[vals[f] for f in feats]], columns=feats).astype(float)

    def _confidence(self, proba: float) -> float:
        t = self.meta["threshold"]
        span = max(1.0 - t, 1e-9)
        frac = min(max((proba - t) / span, 0.0), 1.0)
        lo = config.MIN_CONFIDENCE + 0.01
        return round(lo + frac * (0.90 - lo), 2)

    # ------------------------------------------------------------------
    def decide(self, context_json: str) -> CycleDecisions:
        ctx = json.loads(context_json)
        rows = {r["symbol"]: r for r in ctx.get("symbols", [])}
        positions = ctx.get("open_positions", [])
        held = {p["symbol"] for p in positions}
        as_of = ctx.get("as_of_et")

        # Forget entry records for positions we no longer hold.
        for sym in list(self._entries):
            if sym not in held:
                self._entries.pop(sym, None)

        decisions: List[TradeDecision] = []

        # --- exits: the training barriers, nothing else -------------------
        for pos in positions:
            d = self._maybe_exit(pos, rows.get(pos["symbol"]), as_of)
            if d:
                decisions.append(d)

        if self.model is None:
            return CycleDecisions(
                market_read="ML: no deployable model — holding (exits only)",
                decisions=decisions,
            )

        # --- entries: probability above the evaluated threshold -----------
        candidates = []
        for sym, row in rows.items():
            if sym in held:
                continue
            X = self._features(row)
            if X is None:
                continue
            proba = float(self.model.predict_proba(X)[0, 1])
            if proba >= self.meta["threshold"]:
                candidates.append((proba, sym, row))

        candidates.sort(reverse=True)
        for proba, sym, row in candidates[:2]:
            atr = row.get("atr_14")
            decisions.append(TradeDecision(
                symbol=sym,
                action="buy",
                confidence=self._confidence(proba),
                thesis=(
                    f"ML signal: P(+{self.tp:g} ATR before -{self.sl:g} ATR within "
                    f"{self.horizon}min) = {proba:.3f} vs threshold "
                    f"{self.meta['threshold']:.2f}."
                ),
                invalidation=(
                    f"-{self.sl:g} ATR from entry, +{self.tp:g} ATR target, or "
                    f"{self.horizon} minutes elapsed — the exact barriers the "
                    "probability was trained on."
                ),
                suggested_notional=config.FLAT_POSITION_NOTIONAL,
                time_horizon_minutes=self.horizon,
            ))
            self._entries[sym] = {"atr": atr, "ts": as_of}

        return CycleDecisions(
            market_read=(
                f"ML: {len(rows)} symbols scored, {len(candidates)} above "
                f"threshold {self.meta['threshold']:.2f}."
            ),
            decisions=decisions,
        )

    # ------------------------------------------------------------------
    def _maybe_exit(self, pos: dict, row: Optional[dict], as_of: Optional[str]
                    ) -> Optional[TradeDecision]:
        if row is None:
            return None
        last = row.get("last")
        entry = pos.get("avg_entry")
        if not last or not entry:
            return None

        rec = self._entries.get(pos["symbol"], {})
        atr = rec.get("atr") or row.get("atr_14")
        if not atr:
            return None

        reason = None
        if last >= entry + self.tp * atr:
            reason = f"target: {last:.2f} >= entry {entry:.2f} + {self.tp:g} ATR"
        elif last <= entry - self.sl * atr:
            reason = f"stop: {last:.2f} <= entry {entry:.2f} - {self.sl:g} ATR"
        elif rec.get("ts") and as_of:
            try:
                held_min = (datetime.fromisoformat(as_of)
                            - datetime.fromisoformat(rec["ts"])).total_seconds() / 60
                if held_min >= self.horizon:
                    reason = f"horizon: held {held_min:.0f}min >= {self.horizon}min"
            except ValueError:
                pass

        if reason is None:
            return None
        return TradeDecision(
            symbol=pos["symbol"], action="close", confidence=0.70,
            thesis=f"ML exit — {reason}.",
            invalidation="n/a — this is an exit",
            suggested_notional=0.0, time_horizon_minutes=5,
        )
