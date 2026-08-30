"""Relabel an existing dataset with different barriers — no re-download.

Features are label-independent, so changing the barrier geometry only requires
recomputing labels from the OHLC columns already in the file.

    python relabel_dataset.py --file data/dataset_full.csv \
        --horizon 48 --tp-atr 2 --sl-atr 2 --out dataset_2atr.csv

Why this variant is pre-registered rather than a fishing expedition: the ±1 ATR
model failed for a MEASURED reason — gross edge ~$0.07/trade vs $0.40 fixed
round-trip cost. Larger barriers scale the gross dollars per trade while the
cost stays fixed. This tests that one hypothesis; it is not a grid search.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

from build_dataset import add_labels


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--file", default="data/dataset_full.csv")
    p.add_argument("--horizon", type=int, required=True)
    p.add_argument("--tp-atr", type=float, required=True)
    p.add_argument("--sl-atr", type=float, required=True)
    p.add_argument("--out", required=True)
    args = p.parse_args()

    path = Path(args.file)
    if not path.exists():
        sys.exit(f"not found: {path}")
    df = pd.read_csv(path)
    df["ts"] = pd.to_datetime(df["ts"], utc=True, format="mixed")

    out_frames = []
    # Relabel per symbol — labels must never look across symbols.
    for sym, g in df.groupby("symbol", sort=False):
        g = g.sort_values("ts").reset_index(drop=True)
        g = g.drop(columns=["label", "fwd_ret_atr"], errors="ignore")
        g = add_labels(g, args.horizon, args.tp_atr, args.sl_atr)
        out_frames.append(g)
        print(f"  {sym}: {len(g):,} rows relabeled")

    out = pd.concat(out_frames, ignore_index=True).dropna(subset=["label", "atr_14"])
    out_path = Path("data") / args.out
    out.to_csv(out_path, index=False)
    print(f"\nwrote {len(out):,} rows -> {out_path}")
    print(f"  positive: {100*out['label'].mean():.1f}%  "
          f"(barriers ±{args.tp_atr:g}/{args.sl_atr:g} ATR, horizon {args.horizon} bars)")


if __name__ == "__main__":
    main()
