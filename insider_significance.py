"""How significant is the insider edge REALLY? Correcting for time clustering.

    python insider_significance.py --labelled insider_labelled_21_open1.csv

The headline t=+10 from insider_universe_study.py is overstated, and this
script exists to say by how much. That t treats every event as an independent
observation after removing overlap WITHIN a symbol — but events cluster in
CALENDAR time across symbols: insiders buy their own companies during the same
market-wide selloffs, so hundreds of "independent" trades are really one bet on
one month. March 2020 alone supplies a large block of them.

Three progressively honest estimators:

  1. naive        every event independent (what the study prints)
  2. by-month     mean excess per calendar month, t across ~80 months — the
                  standard cluster correction when the clustering is temporal
  3. block boot   stationary bootstrap over calendar months (expected block
                  length 3) — makes no normality assumption and keeps
                  neighbouring months together

Also reported: the fraction of months positive, the contribution of the single
best month, and the result with the largest month removed. If one month carries
the effect, the edge is a story about that month, not a strategy.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def tstat(x: np.ndarray) -> float:
    x = x[np.isfinite(x)]
    if len(x) < 3:
        return float("nan")
    return float(x.mean() / (x.std(ddof=1) / np.sqrt(len(x))))


def stationary_bootstrap(monthly: pd.Series, n_boot: int = 10_000,
                         mean_block: float = 3.0, seed: int = 7) -> np.ndarray:
    """Politis-Romano stationary bootstrap on the monthly mean series."""
    rng = np.random.default_rng(seed)
    v = monthly.to_numpy(float)
    n = len(v)
    p = 1.0 / mean_block
    out = np.empty(n_boot)
    for b in range(n_boot):
        idx = np.empty(n, dtype=int)
        i = rng.integers(n)
        for k in range(n):
            idx[k] = i
            if rng.random() < p:
                i = rng.integers(n)
            else:
                i = (i + 1) % n
        out[b] = v[idx].mean()
    return out


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--labelled", default="insider_labelled_21_open1.csv")
    p.add_argument("--filter", default="core", choices=["core", "liquid", "all"])
    args = p.parse_args()

    d = pd.read_csv(Path("data") / args.labelled, parse_dates=["entry_date"])
    d = d[d["discount_pct"].abs() < 10]
    if "is_10b5_1" in d.columns:
        d = d[d["is_10b5_1"] == 0]
    if args.filter == "core":
        d = d[(d["dollar_vol_20"] >= 1e6) & (d["trader_type"] != "routine")
              & (d["buy_usd"] >= 1e5)
              & ((d["is_csuite"] == 1) | (d["is_director"] == 1))]
    elif args.filter == "liquid":
        d = d[d["dollar_vol_20"] >= 1e6]

    notional = 1000.0
    slip = np.where(d["dollar_vol_20"] < 1e6, 2 * 20.0, 2 * 2.0) / 10_000 * notional
    scale = d["atr_pct"] / 100 * notional
    d = d.assign(usd=d["fwd_atr"] * scale - slip,
                 ctrl_usd=d["ctrl_atr"] * scale - slip)
    d["excess"] = d["usd"] - d["ctrl_usd"]
    d["month"] = d["entry_date"].dt.to_period("M")

    print(f"filter={args.filter}  n={len(d):,} events  "
          f"{d['entry_date'].min():%Y-%m} .. {d['entry_date'].max():%Y-%m}\n")

    # --- 1. naive -----------------------------------------------------------
    e = d["excess"].to_numpy()
    print(f"  1. naive (every event independent)   mean ${e.mean():+6.2f}  "
          f"t={tstat(e):+6.2f}   n={len(e):,}")

    # --- 1b. non-overlapping within symbol (what the study reports) ---------
    if "nonoverlap" in d.columns:
        no = d[d["nonoverlap"]]["excess"].to_numpy()
        print(f"  1b. non-overlapping within symbol    mean ${no.mean():+6.2f}  "
              f"t={tstat(no):+6.2f}   n={len(no):,}")

    # --- 2. clustered by calendar month -------------------------------------
    m = d.groupby("month")["excess"].mean()
    cnt = d.groupby("month").size()
    print(f"\n  2. CLUSTERED BY MONTH                mean ${m.mean():+6.2f}  "
          f"t={tstat(m.to_numpy()):+6.2f}   n={len(m)} months")
    print(f"     months positive: {100*(m > 0).mean():.0f}%   "
          f"median month ${m.median():+.2f}   worst ${m.min():+.2f}  best ${m.max():+.2f}")

    # Equal-weight by month is the right estimator, but weight by event count
    # too — if they disagree the effect lives in the busy (crisis) months.
    wm = (m * cnt).sum() / cnt.sum()
    print(f"     event-weighted mean ${wm:+.2f} vs equal-weighted ${m.mean():+.2f}"
          f"  ({'agree' if abs(wm - m.mean()) < 0.35 * abs(m.mean()) else 'DISAGREE — crisis months carry it'})")

    # --- 3. stationary bootstrap over months --------------------------------
    boot = stationary_bootstrap(m)
    lo, hi = np.percentile(boot, [2.5, 97.5])
    print(f"\n  3. stationary bootstrap (block~3mo)   95% CI [${lo:+.2f}, ${hi:+.2f}]   "
          f"P(mean<=0) = {100*(boot <= 0).mean():.1f}%")

    # --- robustness: drop the best month, drop 2020 -------------------------
    best = m.idxmax()
    m_drop = m.drop(best)
    print(f"\n  drop best month ({best})            mean ${m_drop.mean():+6.2f}  "
          f"t={tstat(m_drop.to_numpy()):+6.2f}")
    m_ex20 = m[m.index.year != 2020]
    print(f"  excluding all of 2020                mean ${m_ex20.mean():+6.2f}  "
          f"t={tstat(m_ex20.to_numpy()):+6.2f}   n={len(m_ex20)} months")
    m_hold = m[m.index.year >= 2024]
    print(f"  holdout only (2024+)                 mean ${m_hold.mean():+6.2f}  "
          f"t={tstat(m_hold.to_numpy()):+6.2f}   n={len(m_hold)} months")

    # --- yearly, for the record ---------------------------------------------
    print("\n  by year (month-clustered t within each year):")
    for y, s in d.groupby(d["entry_date"].dt.year):
        my = s.groupby("month")["excess"].mean()
        print(f"    {y}  n={len(s):6,}  mean ${my.mean():+6.2f}  t={tstat(my.to_numpy()):+5.2f}  "
              f"months+ {100*(my > 0).mean():3.0f}%")

    # --- cost sensitivity ----------------------------------------------------
    print("\n  cost sensitivity (round-trip bps off the month-clustered mean):")
    base_scale = scale.to_numpy()
    raw = (d["fwd_atr"].to_numpy() - d["ctrl_atr"].to_numpy()) * base_scale
    for bps in (0, 5, 10, 20, 40, 60):
        # Costs hit the event leg only: the control is a counterfactual that
        # pays the same costs, so they cancel in `excess`. What does NOT cancel
        # is the cost of ACTUALLY trading the event, so charge it once here.
        net = raw - bps / 10_000 * notional
        mm = pd.Series(net, index=d["month"].to_numpy()).groupby(level=0).mean()
        flag = "" if mm.mean() > 0 else "   <-- edge gone"
        print(f"    {bps:3d}bp round trip   mean ${mm.mean():+6.2f}  "
              f"t={tstat(mm.to_numpy()):+5.2f}{flag}")


if __name__ == "__main__":
    main()
