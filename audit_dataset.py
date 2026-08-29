"""Audit a built dataset for lookahead leakage and basic sanity.

The failure mode this catches is the one that kills quant projects: a feature
that quietly encodes the future. It produces a spectacular backtest and a model
that is worthless live, and it is invisible unless you go looking.

    python audit_dataset.py --file data/dataset.parquet
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd
import numpy as np

FEATURES = [
    "pct_vs_vwap", "rsi_14", "atr_pct", "vol_ratio", "pct_vs_or_high",
    "or_range_pct", "after_or", "ret_1", "ret_3", "ret_6", "gap_pct",
    "min_since_open",
]

issues: list[str] = []


def report(ok: bool, label: str, detail: str = "") -> None:
    print(f"  [{'OK  ' if ok else 'BAD '}] {label:<44} {detail}")
    if not ok:
        issues.append(label)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--file", default="data/dataset.parquet")
    args = p.parse_args()

    path = Path(args.file)
    if not path.exists():
        alt = path.with_suffix(".csv")
        if alt.exists():
            path = alt
        else:
            sys.exit(f"not found: {args.file}")

    df = pd.read_parquet(path) if path.suffix == ".parquet" else pd.read_csv(path)
    print(f"\n{len(df):,} rows from {path}\n")
    print("=" * 68)
    print("LEAKAGE AUDIT")
    print("=" * 68)

    # 1. Opening-range features must be absent before the window closes.
    pre = df[df["after_or"] == 0]
    if len(pre):
        leaked = pre["or_high"].notna().sum()
        report(leaked == 0, "opening range absent before its window closes",
               f"{leaked} pre-cutoff rows carry or_high")
    else:
        report(True, "opening range absent before its window closes", "no pre-OR rows")

    # 2. No single feature should predict the label too well. A feature that
    #    correlates strongly with a forward-looking label is the signature of
    #    leakage — real intraday signals are weak, in the 0.0-0.1 range.
    worst = None
    for f in FEATURES:
        if f not in df.columns:
            continue
        s = df[[f, "label"]].dropna()
        if len(s) < 100 or s[f].nunique() < 3:
            continue
        c = abs(np.corrcoef(s[f], s["label"])[0, 1])
        if worst is None or c > worst[1]:
            worst = (f, c)
    if worst:
        report(worst[1] < 0.25, "no feature correlates suspiciously with label",
               f"strongest: {worst[0]} r={worst[1]:.3f}")

    # 3. Label balance. Horizon expiry counts as a loss, so a base rate somewhat
    #    under 50% is expected and correct.
    rate = df["label"].mean()
    report(0.30 < rate < 0.70, "label base rate is plausible",
           f"{100*rate:.1f}% positive — this is the number a model must beat")

    # 4. Chronological ordering per symbol, so walk-forward splits are valid.
    ok_order = all(
        g["ts"].is_monotonic_increasing for _, g in df.groupby("symbol", sort=False)
    )
    report(ok_order, "rows chronological within each symbol")

    # 5. Regular trading hours only.
    bad_hours = ((df["min_since_open"] < 0) | (df["min_since_open"] > 390)).sum()
    report(bad_hours == 0, "regular trading hours only",
           f"{bad_hours} rows outside 09:30-16:00")

    # 6. Features must not be constant or near-empty.
    for f in FEATURES:
        if f in df.columns:
            nn = df[f].notna().mean()
            if nn < 0.5:
                report(False, f"feature {f} mostly null", f"{100*nn:.0f}% populated")

    print("\n" + "=" * 68)
    print("BASELINE (what a model must beat)")
    print("=" * 68)
    print(f"  always-predict-majority accuracy : {max(rate, 1-rate)*100:.1f}%")
    print("  A model at 52% when the base rate is 50% has a real, small edge.")
    print("  A model at 95% has a leak. Find it before you believe it.")

    print("\n" + "=" * 68)
    if issues:
        print("VERDICT: problems found -> " + ", ".join(issues))
        sys.exit(1)
    print("VERDICT: no leakage detected")


if __name__ == "__main__":
    main()
