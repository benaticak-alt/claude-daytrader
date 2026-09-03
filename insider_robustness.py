"""Stress the insider result against the two things most likely to be faking it.

1. OVERLAPPING WINDOWS. A 21-day holding period means consecutive events in one
   symbol share most of their outcome. Treating them as independent inflates
   every t-statistic, often by 2-3x. Fix: keep at most one event per symbol per
   holding period, and re-test on that non-overlapping subset.

2. REGIME. A single strong quarter can carry a whole result. Fix: break the
   excess down by calendar period; a real effect should appear in most of them,
   not one.

Also reports a block bootstrap (resampling whole months, preserving the
within-month correlation that the naive standard error ignores).

    python insider_robustness.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from insider_control import dollars, label_all_days

RNG = np.random.default_rng(7)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--events", default="data/dataset_insider.csv")
    p.add_argument("--prices", default="data/dataset_daily.csv")
    p.add_argument("--horizon", type=int, default=21)
    p.add_argument("--tp-atr", type=float, default=1.5)
    p.add_argument("--sl-atr", type=float, default=1.5)
    p.add_argument("--oos-days", type=int, default=550)
    args = p.parse_args()

    ev = pd.read_csv(args.events)
    ev["ts"] = pd.to_datetime(ev["ts"], utc=True, format="mixed")
    cutoff = ev["ts"].max() - pd.Timedelta(days=args.oos_days)
    oos = ev[ev["ts"] >= cutoff].copy()
    oos["pnl"] = dollars(oos["fwd_ret_atr"], oos["atr_pct"])
    oos = oos.sort_values(["symbol", "ts"]).reset_index(drop=True)

    px = pd.read_csv(args.prices)
    px["ts"] = pd.to_datetime(px["ts"], utc=True, format="mixed")

    frames = []
    for s in sorted(oos["symbol"].unique()):
        g = px[px["symbol"] == s].sort_values("ts").reset_index(drop=True)
        if len(g) < args.horizon + 20:
            continue
        g = label_all_days(g, args.horizon, args.tp_atr, args.sl_atr)
        frames.append(g[g["ts"] >= cutoff])
    ctrl = pd.concat(frames, ignore_index=True).dropna(subset=["ctrl_ret_atr", "atr_pct"])
    ctrl["pnl"] = dollars(ctrl["ctrl_ret_atr"], ctrl["atr_pct"])
    ctrl_mean = ctrl["pnl"].mean()

    print("=" * 78)
    print("ROBUSTNESS — insider excess over symbol/period-matched control")
    print("=" * 78)
    print(f"  control mean: {ctrl_mean:+.2f} $/trade over {len(ctrl):,} symbol-days\n")

    # --- 1. non-overlapping events ---------------------------------------
    keep = []
    last_by_sym: dict[str, pd.Timestamp] = {}
    gap = pd.Timedelta(days=int(args.horizon * 1.45))   # 21 trading ~ 30 calendar
    for i, r in oos.iterrows():
        prev = last_by_sym.get(r["symbol"])
        if prev is None or (r["ts"] - prev) >= gap:
            keep.append(i)
            last_by_sym[r["symbol"]] = r["ts"]
    nov = oos.loc[keep]

    def report(tag, s):
        if len(s) < 5:
            print(f"  {tag:32s}   n={len(s):<4d} too few")
            return
        d = s.mean() - ctrl_mean
        se = np.sqrt(s.var() / len(s) + ctrl["pnl"].var() / len(ctrl))
        print(f"  {tag:32s}   n={len(s):<4d} {s.mean():+8.2f} $/trade   "
              f"excess {d:+7.2f}   t={d/se if se > 0 else 0:+5.2f}")

    report("all OOS events", oos["pnl"])
    report("NON-OVERLAPPING only", nov["pnl"])
    print(f"    (dropped {len(oos) - len(nov)} overlapping events — their outcomes")
    print("     largely duplicate an earlier event in the same symbol)\n")

    # --- 2. by period ------------------------------------------------------
    print("  by half-year (a real effect should not live in one window):")
    oos["period"] = oos["ts"].dt.to_period("Q").astype(str)
    for per, g in oos.groupby("period"):
        if len(g) < 10:
            continue
        c = ctrl[ctrl["ts"].dt.to_period("Q").astype(str) == per]["pnl"]
        cm = c.mean() if len(c) else ctrl_mean
        print(f"    {per}   n={len(g):<4d} events {g['pnl'].mean():+8.2f}   "
              f"control {cm:+7.2f}   excess {g['pnl'].mean() - cm:+7.2f}")

    # --- 3. month-block bootstrap -----------------------------------------
    oos["month"] = oos["ts"].dt.to_period("M").astype(str)
    months = oos["month"].unique()
    if len(months) >= 6:
        means = []
        for _ in range(4000):
            pick = RNG.choice(months, size=len(months), replace=True)
            vals = np.concatenate([oos.loc[oos["month"] == m, "pnl"].to_numpy()
                                   for m in pick])
            if len(vals):
                means.append(vals.mean())
        means = np.array(means)
        lo, hi = np.percentile(means - ctrl_mean, [2.5, 97.5])
        print(f"\n  month-block bootstrap 95% CI for excess: "
              f"[{lo:+.2f}, {hi:+.2f}] $/trade")
        print("  (resamples whole months, so within-month correlation is kept)")
        if lo > 0:
            print("  -> CI excludes zero even with correlated months")
        else:
            print("  -> CI includes zero: the naive t-stat was overstating it")

    print("\n  NOT corrected for: survivorship bias. Alpaca serves only")
    print("  currently-listed symbols, so insiders who bought companies that")
    print("  later delisted are absent — and insiders are known for buying")
    print("  falling knives. This biases the result UPWARD by an unknown amount.")


if __name__ == "__main__":
    main()
