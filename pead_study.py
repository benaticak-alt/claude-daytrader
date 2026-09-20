"""Post-earnings-announcement drift (PEAD) on free data — second event signal.

    python pead_study.py --horizon 63

Inputs: data/earnings_events.csv    (fetch_earnings_events.py — 8-K Item 2.02)
        data/daily_universe.parquet (fetch_daily_universe.py --symbols-from earnings)

The surprise measure is the market's OWN reaction — the "earnings announcement
return" (Brandt, Kishore, Santa-Clara & Venkatachalam 2008): close of the day
before the 8-K to close of the day after, standardised by the stock's 20-day
return volatility. It needs no analyst estimates, and the paper found it
predicts drift at least as well as the classic SUE.

Timing, all causal:
    t      8-K Item 2.02 filing date (press release the same day, timing unknown)
    EAR    close[t-1] -> close[t+1], covers a pre-open or post-close release
    entry  OPEN of t+2 — the reaction is fully observable by then
    exit   `horizon` trading days, gap-aware barriers at ±`atr` ATR (wide by
           default — PEAD is a slow drift and tight barriers just clip it)

Control: every non-event day of the same symbol-year, labelled identically.
Report: excess by EAR decile/quintile, then the long top-quintile leg through
the same slices as the insider study (liquidity, size, year).

RESULT 2026-09-20 — NEGATIVE. 80,084 announcements, 62,761 liquid. At 21 days
the top surprise quintile earns +$1.72/trade excess (0.17%, t=+2.0) with no
monotonic decile pattern and a sign that flips year to year (-8, +3, +15, -1,
+11, -0.4, -14). At 63 days the top decile is NEGATIVE (t=-4.9) — reversal,
not drift. Consistent with the literature on PEAD's post-2000s decay. The
insider signal, by contrast, is +$12/trade, t=+10, positive every year. PEAD is
not a usable second leg on this data; the code stays as the record of the test.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from insider_universe_study import barrier_labels, tstat

EVENTS = Path(__file__).parent / "data" / "earnings_events.csv"
PRICES = Path(__file__).parent / "data" / "daily_universe.parquet"
NOTIONAL = 1000.0


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--horizon", type=int, default=63)
    p.add_argument("--atr", type=float, default=3.0, help="symmetric barrier width")
    p.add_argument("--out", default="")
    args = p.parse_args()
    H = args.horizon

    ev = pd.read_csv(EVENTS)
    ev["filing_date"] = pd.to_datetime(ev["filing_date"], errors="coerce")
    ev = ev[ev["filing_date"].notna()].sort_values(["symbol", "filing_date"])
    # One announcement per symbol per 30 days (an 8-K/A or a second 2.02 for
    # the same quarter is not a new event).
    ev = ev[ev.groupby("symbol")["filing_date"].diff().fillna(pd.Timedelta(days=999))
            > pd.Timedelta(days=30)]
    px = pd.read_parquet(PRICES)
    px["date"] = px["ts"].dt.tz_convert("America/New_York").dt.normalize().dt.tz_localize(None)
    sample_end = px["date"].max()
    print(f"{len(ev):,} announcements, {ev['symbol'].nunique():,} symbols; "
          f"{px['symbol'].nunique():,} priced symbols; horizon {H}d ±{args.atr:g} ATR\n")

    rows, no_px = [], 0
    for sym, evs in ev.groupby("symbol", sort=False):
        g = px[px["symbol"] == sym].sort_values("date").reset_index(drop=True)
        if len(g) < 80:
            no_px += len(evs)
            continue
        ended = g["date"].iloc[-1] < sample_end - pd.Timedelta(days=30)
        # Labels from the OPEN two days after each bar (entry="open1" shifted
        # one more day): compute open1 labels and index them at t+1.
        ret, held = barrier_labels(g, H, args.atr, args.atr, ended, "open1")
        c = g["close"].to_numpy(float)
        dates = g["date"].to_numpy()
        dret = pd.Series(c).pct_change()
        vol20 = dret.rolling(20).std().shift(2).to_numpy()   # known at t-1 close
        dv20 = g["dollar_vol_20"].to_numpy(float)
        atrp = g["atr_pct"].to_numpy(float)

        # Control: every other day of the symbol-year, excluding only the
        # announcement reaction days themselves. Quarterly events with a
        # 63-day horizon cover the whole calendar, so excluding entire forward
        # windows (as the insider study does for its rare events) would leave
        # no control at all — the baseline here is "any day's 63-day return".
        # ...but a control day whose FORWARD window contains the announcement
        # jump is contaminated by the very event being tested (positive-
        # surprise names get an inflated control, negative ones a deflated
        # one). So exclude every day within H bars BEFORE an announcement too.
        # That only leaves a control at horizons well under a quarter.
        pos_all = np.searchsorted(dates, evs["filing_date"].to_numpy(), side="left")
        mask = np.ones(len(g), bool)
        for pz in pos_all:
            mask[max(pz - H - 2, 0): min(pz + 3, len(g))] = False
        ctrl = pd.Series(ret)[mask & np.isfinite(ret)].groupby(g["date"].dt.year[mask & np.isfinite(ret)])
        ctrl_mean, ctrl_n = ctrl.mean().to_dict(), ctrl.size().to_dict()

        for (_, e), t in zip(evs.iterrows(), pos_all):
            if t < 21 or t + 2 >= len(g) or (dates[t] - e["filing_date"]).days > 5:
                continue
            ear = c[t + 1] / c[t - 1] - 1
            v = vol20[t + 1] if t + 1 < len(vol20) else np.nan
            if not (np.isfinite(ear) and np.isfinite(v) and v > 0):
                continue
            fwd = ret[t + 1]                    # open1 label at t+1 == entry at open of t+2
            if not np.isfinite(fwd):
                continue
            yr = int(g["date"].iat[t].year)
            if ctrl_n.get(yr, 0) < 60:
                continue
            rows.append({
                "symbol": sym, "ann_date": dates[t], "entry_date": dates[min(t + 2, len(g) - 1)],
                "year": yr, "ear": ear, "ear_z": ear / v,
                "fwd_atr": fwd, "ctrl_atr": ctrl_mean[yr], "hold_days": held[t + 1],
                "atr_pct": atrp[t + 1], "dollar_vol_20": dv20[t + 1],
                "bar_idx": int(t + 1), "series_ended": bool(ended),
            })

    d = pd.DataFrame(rows)
    print(f"{len(d):,} labelled announcements ({no_px:,} lacked price history)")
    slip = np.where(d["dollar_vol_20"] < 1e6, 2 * 20.0, 2 * 2.0) / 10_000 * NOTIONAL
    scale = d["atr_pct"] / 100 * NOTIONAL
    d["usd"] = d["fwd_atr"] * scale - slip
    d["ctrl_usd"] = d["ctrl_atr"] * scale - slip
    d["excess"] = d["usd"] - d["ctrl_usd"]
    d["win"] = (d["fwd_atr"] > 0).astype(int)
    d = d.sort_values(["symbol", "bar_idx"]).reset_index(drop=True)
    # Quarterly events never overlap at H<=63 within a symbol, so every event
    # counts for t; across symbols the same calendar quarter is shared, so the
    # by-year table is the honest cross-check.
    liq = d[d["dollar_vol_20"] >= 1e6]

    def line(tag, s):
        if len(s) < 30:
            print(f"  {tag:30s} n={len(s):6,}   (too few)"); return
        print(f"  {tag:30s} n={len(s):6,}  win {100*s['win'].mean():4.1f}%  ${s['usd'].mean():+7.2f}  "
              f"ctrl ${s['ctrl_usd'].mean():+6.2f}  excess ${s['excess'].mean():+7.2f}  "
              f"t={tstat(s['excess'].to_numpy()):+5.2f}   EAR med {100*s['ear'].median():+.1f}%")

    print("\n== ALL (liquid >= $1M/day) ==")
    line("all announcements", liq)
    print("\n== BY EAR_Z DECILE (surprise = standardised 2-day reaction) ==")
    liq = liq.assign(dec=pd.qcut(liq["ear_z"], 10, labels=False, duplicates="drop"))
    for k, s in liq.groupby("dec"):
        line(f"decile {int(k)+1}", s)
    top = liq[liq["dec"] >= 8]
    bot = liq[liq["dec"] <= 1]
    print("\n== LONG LEG: top quintile of EAR_Z ==")
    line("top quintile", top)
    line("bottom quintile (for contrast)", bot)
    print("\n== TOP QUINTILE BY LIQUIDITY ==")
    for lo, hi, tag in ((1e6, 1e7, "$1M-10M"), (1e7, 1e8, "$10M-100M"), (1e8, np.inf, ">= $100M")):
        line(tag, top[(top["dollar_vol_20"] >= lo) & (top["dollar_vol_20"] < hi)])
    print("\n== TOP QUINTILE BY YEAR ==")
    for y, s in top.groupby("year"):
        line(str(y), s)
    print("\n== TOP QUINTILE, big positive reaction only (EAR > +5%) ==")
    line("EAR > +5%", top[top["ear"] > 0.05])
    line("EAR > +10%", top[top["ear"] > 0.10])

    if args.out:
        d.to_csv(Path("data") / args.out, index=False)
        print(f"\nwrote {len(d):,} -> data/{args.out}")


if __name__ == "__main__":
    main()
