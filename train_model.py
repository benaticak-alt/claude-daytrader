"""Train the signal model and evaluate it walk-forward.

    python train_model.py                    # data/dataset_full.parquet|csv
    python train_model.py --file data/x.csv

METHOD — the parts that keep this honest:

  * WALK-FORWARD ONLY. The model is always tested on data strictly AFTER
    everything it trained on, with a 1-day embargo (labels look forward at
    most one hour and never cross sessions, so a day is generous). Random
    cross-validation on time series leaks the future and produces confident
    nonsense — it is never used here.

  * Evaluation is POOLED out-of-sample expectancy in dollars, not accuracy.
    Every test prediction from every fold goes into one bucket; thresholds are
    then priced with the realized ATR outcome of each simulated trade
    (fwd_ret_atr), the row's own volatility (atr_pct), a $1,000 notional, and
    round-trip slippage. Accuracy can rise while money is lost; this metric
    cannot be fooled that way.

  * The deployment gate is written into the model file. The saved metadata
    carries `deployable: True/False` based on pooled results, and the live
    decider refuses to trade a model whose own evaluation said no.

The final artifact is trained on ALL data (correct for deployment — you want
the freshest model), but the QUOTED numbers are exclusively walk-forward.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import dump
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score

FEATURES = [
    "pct_vs_vwap", "rsi_14", "atr_pct", "vol_ratio",
    "pct_vs_or_high", "or_range_pct", "after_or",
    "ret_1", "ret_3", "ret_6", "gap_pct", "min_since_open",
]
NOTIONAL = 1000.0
SLIPPAGE_ROUNDTRIP = 2 * 2.0 / 10_000 * NOTIONAL   # 2bp per side = $0.40
THRESHOLDS = [0.50, 0.53, 0.56, 0.60, 0.65]
MODEL_DIR = Path(__file__).parent / "models"


def make_model() -> HistGradientBoostingClassifier:
    # Conservative settings; HGB handles NaN features natively.
    return HistGradientBoostingClassifier(
        max_iter=400, learning_rate=0.06, max_leaf_nodes=31,
        min_samples_leaf=200, l2_regularization=1.0,
        early_stopping=False, random_state=7,
    )


def dollars(row_ret_atr: pd.Series, atr_pct: pd.Series) -> pd.Series:
    """Realized $ for a $1k trade: ATR-units x (1 ATR in $) - slippage."""
    return row_ret_atr * (atr_pct / 100.0 * NOTIONAL) - SLIPPAGE_ROUNDTRIP


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--file", default="data/dataset_full.parquet")
    p.add_argument("--test-windows", type=int, default=6)
    p.add_argument("--test-days", type=int, default=42, help="~2 months per window")
    # Label geometry, embedded in the metadata so the live decider manages
    # positions with the SAME barriers the probability was trained on. Trading
    # a ±2 ATR probability with ±1 ATR exits silently invalidates the model.
    p.add_argument("--tp-atr", type=float, default=1.0)
    p.add_argument("--sl-atr", type=float, default=1.0)
    p.add_argument("--horizon-bars", type=int, default=12)
    p.add_argument("--features", default=None,
                   help="comma-separated feature columns (default: intraday set)")
    # The embargo must be AT LEAST the label's lookahead. Intraday labels look
    # <=1h ahead and never cross sessions, so 1 day is generous; a 10-trading-day
    # daily label needs ~14 calendar days or the folds leak.
    p.add_argument("--embargo-days", type=int, default=1)
    # Sized for the 654k-row intraday datasets. Sparse EVENT datasets (e.g.
    # ~1.9k insider filings) legitimately train on far less — but lowering this
    # buys statistical fragility, so it is an explicit choice, never a default.
    p.add_argument("--min-train-rows", type=int, default=5000)
    args = p.parse_args()

    features = ([f.strip() for f in args.features.split(",") if f.strip()]
                if args.features else FEATURES)

    path = Path(args.file)
    if not path.exists() and path.with_suffix(".csv").exists():
        path = path.with_suffix(".csv")
    if not path.exists():
        sys.exit(f"dataset not found: {args.file} — run build_dataset.py first")

    df = pd.read_parquet(path) if path.suffix == ".parquet" else pd.read_csv(path)
    df["ts"] = pd.to_datetime(df["ts"], utc=True, format="mixed")
    df = df.dropna(subset=["label", "fwd_ret_atr", "atr_pct"]).sort_values("ts")
    df = df.reset_index(drop=True)

    base = df["label"].mean()
    print(f"{len(df):,} rows, {df['symbol'].nunique()} symbols, "
          f"{df['ts'].min():%Y-%m-%d} .. {df['ts'].max():%Y-%m-%d}")
    print(f"base rate: {100*base:.1f}% positive — the number to beat\n")

    # ---- walk-forward splits by calendar day -----------------------------
    days = np.array(sorted(df["ts"].dt.normalize().unique()))
    test_span = args.test_windows * args.test_days
    if test_span >= len(days) - 60:
        sys.exit("not enough history for the requested test windows")
    test_start_idx = len(days) - test_span

    day_of_row = df["ts"].dt.normalize().to_numpy()
    pooled = []

    for w in range(args.test_windows):
        lo = test_start_idx + w * args.test_days
        hi = min(lo + args.test_days, len(days))
        test_days_w = days[lo:hi]
        # Embargo between the end of training and the start of testing.
        train_cutoff = test_days_w[0] - np.timedelta64(args.embargo_days, "D")

        train_mask = day_of_row < train_cutoff
        test_mask = np.isin(day_of_row, test_days_w)
        if train_mask.sum() < args.min_train_rows or test_mask.sum() == 0:
            continue

        model = make_model()
        model.fit(df.loc[train_mask, features], df.loc[train_mask, "label"])
        proba = model.predict_proba(df.loc[test_mask, features])[:, 1]

        fold = df.loc[test_mask, ["label", "fwd_ret_atr", "atr_pct", "symbol", "ts"]].copy()
        fold["proba"] = proba
        pooled.append(fold)

        auc = roc_auc_score(fold["label"], proba) if fold["label"].nunique() > 1 else float("nan")
        print(f"  fold {w+1}: train {train_mask.sum():>7,} rows -> "
              f"test {test_mask.sum():>6,} rows "
              f"({pd.Timestamp(test_days_w[0]):%Y-%m-%d}..{pd.Timestamp(test_days_w[-1]):%m-%d})  "
              f"AUC {auc:.3f}")

    if not pooled:
        sys.exit("no folds ran")
    oos = pd.concat(pooled, ignore_index=True)
    oos["pnl"] = dollars(oos["fwd_ret_atr"], oos["atr_pct"])

    print("\n" + "=" * 74)
    print(f"POOLED OUT-OF-SAMPLE — {len(oos):,} predictions, "
          f"AUC {roc_auc_score(oos['label'], oos['proba']):.3f}")
    print("=" * 74)
    print(f"  {'thresh':>6s} {'trades':>8s} {'cover':>7s} {'win%':>6s} "
          f"{'avg $/trade':>12s} {'total $':>10s}")
    print("  " + "-" * 60)

    # The null hypothesis is NOT "zero" — it is "buy every bar". In a rising
    # market, indiscriminate long-only buying is profitable (that is beta, and
    # it is free via an index fund). A model only earns deployment by SELECTING
    # better than indiscriminate buying: excess over the every-bar baseline,
    # at t >= 2, on 300+ trades, while also clearing $0. Gating on $0 alone
    # once marked a skill-free model (AUC 0.504) deployable off pure drift.
    baseline = oos["pnl"].mean()

    best = None
    table = {}
    for t in THRESHOLDS:
        sel = oos[oos["proba"] >= t]
        n = len(sel)
        if n == 0:
            print(f"  {t:6.2f} {0:8d}       —      —            —          —")
            continue
        wr = sel["label"].mean()
        avg = sel["pnl"].mean()
        excess = avg - baseline
        se = sel["pnl"].std() / max(np.sqrt(n), 1e-9)
        tstat = excess / se if se > 0 else 0.0
        row = {"trades": int(n), "coverage": n / len(oos), "win_rate": float(wr),
               "avg_pnl": float(avg), "total_pnl": float(sel["pnl"].sum()),
               "excess_vs_all": float(excess), "excess_tstat": float(tstat)}
        table[str(t)] = row
        print(f"  {t:6.2f} {n:8,d} {100*n/len(oos):6.1f}% {100*wr:5.1f}% "
              f"{avg:+12.3f} {sel['pnl'].sum():+10.2f}   "
              f"excess {excess:+7.3f} (t={tstat:+.1f})")
        if n >= 300 and avg > 0 and tstat >= 2.0 and (best is None or excess > best[1]):
            best = (t, excess, n, avg)

    # Every-bar baseline for contrast.
    print(f"  {'ALL':>6s} {len(oos):8,d}  100.0% {100*oos['label'].mean():5.1f}% "
          f"{oos['pnl'].mean():+12.3f} {oos['pnl'].sum():+10.2f}")

    print("\n  avg $/trade is AFTER slippage, on a $1,000 notional, using each")
    print("  trade's realized barrier outcome — not accuracy, money.")

    deployable = best is not None
    if deployable:
        print(f"\n  VERDICT: deployable at threshold {best[0]} — "
              f"{best[3]:+.3f} $/trade, {best[1]:+.3f} above the every-bar "
              f"baseline (n={best[2]:,})")
    else:
        print("\n  VERDICT: NOT deployable — no threshold beats the every-bar "
              "baseline at t>=2 on 300+ OOS trades (selection skill, not drift).")
        print("  The live decider will refuse to trade this model.")

    # ---- final artifact: trained on everything, verdict embedded ----------
    MODEL_DIR.mkdir(exist_ok=True)
    final = make_model()
    final.fit(df[features], df["label"])
    meta = {
        "trained_at": datetime.now().isoformat(timespec="seconds"),
        "dataset": str(path), "rows": int(len(df)),
        "features": features, "base_rate": float(base),
        "oos_auc": float(roc_auc_score(oos["label"], oos["proba"])),
        "thresholds": table,
        "deployable": bool(deployable),
        "threshold": float(best[0]) if deployable else None,
        "expected_dollars_per_trade": float(best[3]) if deployable else None,
        "excess_vs_every_bar": float(best[1]) if deployable else None,
        "every_bar_baseline": float(baseline),
        "tp_atr": args.tp_atr,
        "sl_atr": args.sl_atr,
        "horizon_minutes": args.horizon_bars * 5,
    }
    dump({"model": final, "meta": meta}, MODEL_DIR / "signal_model.joblib")
    (MODEL_DIR / "signal_model.meta.json").write_text(json.dumps(meta, indent=2))
    print(f"\n  saved -> models/signal_model.joblib (deployable={deployable})")


if __name__ == "__main__":
    main()
