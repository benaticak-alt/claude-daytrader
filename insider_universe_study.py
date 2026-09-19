"""Universe-wide insider-buy study: which slices of the signal are real?

    python insider_universe_study.py --horizon 21
    python insider_universe_study.py --horizon 63 --min-dollar-vol 1e6

Inputs: data/insider_events_universe.csv  (build_insider_universe.py)
        data/daily_universe.parquet        (fetch_daily_universe.py)

Method (same as the 80-symbol pass, so numbers are comparable):
  entry    close of the first trading day ON OR AFTER the filing date
  exit     gap-aware daily triple barrier, ±TP/SL ATR(14), `horizon` days;
           a series that ENDS (delisting) inside the window exits at its last
           close — the falling-knife outcomes survivorship used to hide
  control  every non-event day of the same symbol in the same calendar year,
           labelled identically. Excess = event − control mean. This nets out
           both market drift and the fact that insiders buy particular kinds
           of companies.
  dollars  $1,000 notional, 2bp slippage per side (20bp for names under
           $1M/day — IEX quotes on thin names are not 2bp wide)

Every slice prints n, mean $, excess $ over control, and a t-stat computed on
the NON-OVERLAPPING subset (one event per symbol per horizon window), since
overlapping 21-day holds in one name share an outcome and inflate t.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view

EVENTS = Path(__file__).parent / "data" / "insider_events_universe.csv"
PRICES = Path(__file__).parent / "data" / "daily_universe.parquet"
RAW_CLOSES = Path(__file__).parent / "data" / "daily_universe_rawclose.parquet"

NOTIONAL = 1000.0


def barrier_labels(g: pd.DataFrame, horizon: int, tp_atr: float, sl_atr: float,
                   series_ended: bool, entry: str = "close0") -> tuple[np.ndarray, np.ndarray]:
    """Vectorised gap-aware triple barrier from EVERY day.

    entry  close0  enter at day i's close (the original assumption)
           open1   enter at day i+1's OPEN — what a bot that learns of the
                   filing after the bell can actually get. Form 4s cluster
                   after 4pm; a same-day-close entry on those is lookahead.
    Barriers are set from the entry price using ATR known at day i's close.

    Returns (fwd_ret in ATR units, bars held) per day (NaN where unresolvable).
    Priority within a day: gap-through-stop, gap-through-target, stop touch,
    target touch — pessimistic, exactly like label_from_entry().
    """
    o, h = g["open"].to_numpy(float), g["high"].to_numpy(float)
    l, c = g["low"].to_numpy(float), g["close"].to_numpy(float)
    a = g["atr_14"].to_numpy(float)
    n = len(g)
    H = horizon
    if entry == "open1":
        # Entry price is tomorrow's open; on that first day the "gap" check
        # compares the open to itself and can never fire, so the same window
        # logic applies unchanged.
        c = np.concatenate([o[1:], [np.nan]])
    elif entry != "close0":
        raise ValueError(entry)
    pad = np.full(H, np.nan)
    O = sliding_window_view(np.concatenate([o[1:], pad]), H)   # O[i, k] = o[i+1+k]
    Hh = sliding_window_view(np.concatenate([h[1:], pad]), H)
    L = sliding_window_view(np.concatenate([l[1:], pad]), H)
    C = sliding_window_view(np.concatenate([c[1:], pad]), H)
    tp = (c + tp_atr * a)[:, None]
    sl = (c - sl_atr * a)[:, None]

    with np.errstate(invalid="ignore"):
        hit = np.zeros((n, H), dtype=np.int8)
        hit[Hh >= tp] = 4
        hit[L <= sl] = 3
        hit[O >= tp] = 2
        hit[O <= sl] = 1       # highest priority written last
    any_hit = hit > 0
    first = np.where(any_hit.any(1), any_hit.argmax(1), -1)

    ret = np.full(n, np.nan)
    held = np.full(n, np.nan)
    idx = np.arange(n)
    ok = (first >= 0) & np.isfinite(a) & (a > 0)
    kind = hit[idx[ok], first[ok]]
    oj = O[idx[ok], first[ok]]
    r = np.where(kind == 1, (oj - c[ok]) / a[ok],
        np.where(kind == 2, (oj - c[ok]) / a[ok],
        np.where(kind == 3, -sl_atr, tp_atr)))
    ret[ok] = r
    held[ok] = first[ok] + 1

    # No barrier hit: expire at the horizon close if the window is complete.
    valid_fwd = np.isfinite(C).sum(1)                 # bars available ahead
    exp = (~ok) & np.isfinite(a) & (a > 0) & (valid_fwd >= H)
    ret[exp] = (C[exp, H - 1] - c[exp]) / a[exp]
    held[exp] = H
    # Window incomplete because the SERIES ENDED: exit at the last close.
    if series_ended:
        tail = (~ok) & (~exp) & np.isfinite(a) & (a > 0) & (valid_fwd > 0)
        last_c = np.array([C[i, valid_fwd[i] - 1] for i in np.where(tail)[0]])
        ret[tail] = (last_c - c[tail]) / a[tail]
        held[tail] = valid_fwd[tail]

    # Data-quality guard: a >3x or <-75% one-day close change that survives
    # split adjustment is a bad print (PME closing at $0.0001), not a return.
    # Any window containing one is unlabelled rather than a 200-ATR win.
    with np.errstate(invalid="ignore", divide="ignore"):
        jump = np.zeros(n, bool)
        jump[1:] = (c[1:] / c[:-1] > 3.0) | (c[1:] / c[:-1] < 0.25) | (c[1:] <= 0.01)
    J = sliding_window_view(np.concatenate([jump[1:], np.zeros(H, bool)]), H)
    ret[J.any(1)] = np.nan
    held[J.any(1)] = np.nan
    return ret, held


def tstat(x: np.ndarray) -> float:
    x = x[np.isfinite(x)]
    return float(x.mean() / (x.std(ddof=1) / np.sqrt(len(x)))) if len(x) > 2 else float("nan")


def nonoverlap(ev: pd.DataFrame, horizon: int) -> pd.Series:
    """True for events at least `horizon` trading days after the previous kept
    event in the same symbol."""
    keep = np.zeros(len(ev), bool)
    for _, g in ev.groupby("symbol", sort=False):
        last = -10**9
        for i, pos in zip(g.index, g["bar_idx"]):
            if pos - last >= horizon:
                keep[ev.index.get_loc(i)] = True
                last = pos
    return pd.Series(keep, index=ev.index)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--horizon", type=int, default=21)
    p.add_argument("--tp-atr", type=float, default=1.5)
    p.add_argument("--sl-atr", type=float, default=1.5)
    p.add_argument("--min-dollar-vol", type=float, default=0.0,
                   help="drop events in names below this 20d avg $ volume")
    p.add_argument("--entry", default="close0", choices=["close0", "open1", "close1"],
                   help="close0: filing-day close (lookahead for post-4pm filings); "
                        "open1: next open; close1: next close")
    p.add_argument("--out", default="")
    args = p.parse_args()
    H = args.horizon

    ev = pd.read_csv(EVENTS)
    ev["filing_date"] = pd.to_datetime(ev["filing_date"], errors="coerce")
    ev["trans_date"] = pd.to_datetime(ev["trans_date"], errors="coerce")
    ev = ev[ev["filing_date"].notna()]
    px = pd.read_parquet(PRICES)
    px["date"] = px["ts"].dt.tz_convert("America/New_York").dt.normalize().dt.tz_localize(None)
    # Unadjusted closes for the discount check: Form 4 prices are as-traded,
    # and comparing them with split-adjusted closes flags every later splitter
    # as a "placement" — which would quietly drop the distressed names.
    raw = None
    if RAW_CLOSES.exists():
        raw = pd.read_parquet(RAW_CLOSES)
        raw["date"] = raw["ts"].dt.tz_convert("America/New_York").dt.normalize().dt.tz_localize(None)
        raw = raw.set_index(["symbol", "date"])["close"].sort_index()
    else:
        print("WARNING: no raw closes — run fetch_daily_universe.py --raw-close; "
              "discount check will misfire on split symbols")
    sample_end = px["date"].max()
    print(f"{len(ev):,} events, {px['symbol'].nunique():,} priced symbols, "
          f"bars to {sample_end:%Y-%m-%d}, horizon {H}d ±{args.tp_atr:g}/{args.sl_atr:g} ATR\n")

    rows = []
    no_price = 0
    for sym, evs in ev.groupby("symbol", sort=False):
        g = px[px["symbol"] == sym].sort_values("date").reset_index(drop=True)
        if g.empty:
            no_price += len(evs)
            continue
        ended = g["date"].iloc[-1] < sample_end - pd.Timedelta(days=30)
        if args.entry == "close1":
            # Same as close0 evaluated one bar later: shift the event forward.
            ret, held = barrier_labels(g, H, args.tp_atr, args.sl_atr, ended, "close0")
        else:
            ret, held = barrier_labels(g, H, args.tp_atr, args.sl_atr, ended, args.entry)
        g["fwd"] = ret
        g["held"] = held
        dates = g["date"].to_numpy()
        close = g["close"].to_numpy()

        # Control: every non-event day in the same symbol-year.
        ev_pos = np.searchsorted(dates, evs["filing_date"].to_numpy(), side="left")
        if args.entry == "close1":
            ev_pos = ev_pos + 1
        mask = np.ones(len(g), bool)
        for pz in ev_pos:
            mask[max(pz, 0): min(pz + H + 1, len(g))] = False
        ctrl = g.loc[mask & np.isfinite(g["fwd"])].groupby(g["date"].dt.year)["fwd"]
        ctrl_mean = ctrl.mean().to_dict()
        ctrl_n = ctrl.size().to_dict()

        for (_, e), pos in zip(evs.iterrows(), ev_pos):
            if pos >= len(g) or (dates[pos] - e["filing_date"]).days > 10:
                continue                 # not trading around the filing
            fwd = g["fwd"].iat[pos]
            if not np.isfinite(fwd):
                continue
            bar = g.iloc[pos]
            yr = int(bar["date"].year)
            if ctrl_n.get(yr, 0) < 60:
                continue
            # Price paid vs the market that day: a deep discount means a
            # placement or offering coded P, not an open-market vote of confidence.
            tpos = (np.searchsorted(dates, np.datetime64(e["trans_date"]), side="left")
                    if pd.notna(e["trans_date"]) else pos)
            tpos = min(tpos, len(g) - 1)
            tclose = close[tpos]
            if raw is not None:
                try:
                    tclose = float(raw.loc[(sym, dates[tpos])])
                except KeyError:
                    pass
            paid = e["buy_usd"] / e["buy_shares"] if e["buy_shares"] else np.nan
            rows.append({
                **e.to_dict(),
                "bar_idx": int(pos), "year": yr,
                "entry_date": dates[min(pos + (1 if args.entry == "open1" else 0), len(g) - 1)],
                "fwd_atr": fwd, "ctrl_atr": ctrl_mean[yr],
                "hold_days": int(g["held"].iat[pos]),
                "exit_date": dates[min(pos + int(g["held"].iat[pos]), len(g) - 1)],
                "atr_pct": bar["atr_pct"], "ret_20": bar["ret_20"],
                "dist_ma50": bar["dist_ma50"], "dollar_vol_20": bar["dollar_vol_20"],
                "series_ended": bool(ended),
                "discount_pct": 100 * (paid / tclose - 1) if tclose else np.nan,
            })

    d = pd.DataFrame(rows)
    print(f"{len(d):,} labelled events ({no_price:,} had no price history)")
    if args.min_dollar_vol:
        d = d[d["dollar_vol_20"] >= args.min_dollar_vol]
        print(f"{len(d):,} after dollar-volume floor ${args.min_dollar_vol:,.0f}")

    # Dollars: ATR units -> $ at $1,000 notional, slippage by liquidity.
    slip = np.where(d["dollar_vol_20"] < 1e6, 2 * 20.0, 2 * 2.0) / 10_000 * NOTIONAL
    scale = d["atr_pct"] / 100.0 * NOTIONAL
    d["usd"] = d["fwd_atr"] * scale - slip
    d["ctrl_usd"] = d["ctrl_atr"] * scale - slip
    d["excess"] = d["usd"] - d["ctrl_usd"]
    d = d.sort_values(["symbol", "bar_idx"]).reset_index(drop=True)
    d["nonoverlap"] = nonoverlap(d, H).to_numpy()
    d["win"] = (d["fwd_atr"] > 0).astype(int)

    def line(tag: str, s: pd.DataFrame) -> None:
        if len(s) < 30:
            print(f"  {tag:34s} n={len(s):6,}   (too few)")
            return
        no = s[s["nonoverlap"]]
        print(f"  {tag:34s} n={len(s):6,}  win {100*s['win'].mean():4.1f}%  "
              f"${s['usd'].mean():+7.2f}  ctrl ${s['ctrl_usd'].mean():+6.2f}  "
              f"excess ${s['excess'].mean():+7.2f}   "
              f"non-overlap n={len(no):5,} t={tstat(no['excess'].to_numpy()):+5.2f}")

    print("\n== ALL ==")
    line("all events", d)
    line("series still trading", d[~d["series_ended"]])
    line("series ENDED (delisted etc.)", d[d["series_ended"]])
    line("near-market price (|disc|<5%)", d[d["discount_pct"].abs() < 5])
    line("bought at >10% discount", d[d["discount_pct"] < -10])

    base = d[(d["discount_pct"].abs() < 10)]        # exclude placements from here on
    print("\n== BY TRADER TYPE (Cohen-Malloy-Pomorski) — placements excluded ==")
    for t in ("opportunistic", "routine", "unclassified"):
        line(t, base[base["trader_type"] == t])

    print("\n== BY ROLE ==")
    line("C-suite", base[base["is_csuite"] == 1])
    line("other officer", base[(base["is_officer"] == 1) & (base["is_csuite"] == 0)])
    line("director (non-officer)", base[(base["is_director"] == 1) & (base["is_officer"] == 0)])
    line("10% owner only", base[(base["is_10pct"] == 1) & (base["is_officer"] == 0)
                                 & (base["is_director"] == 0)])

    print("\n== CLUSTER (distinct insiders buying, 30d) ==")
    line("1 insider", base[base["n_insiders_30d"] == 1])
    line("2 insiders", base[base["n_insiders_30d"] == 2])
    line("3+ insiders", base[base["n_insiders_30d"] >= 3])

    print("\n== SIZE OF PURCHASE ==")
    for lo, hi, tag in ((0, 25e3, "< $25k"), (25e3, 100e3, "$25k-100k"),
                        (100e3, 1e6, "$100k-1M"), (1e6, np.inf, ">= $1M")):
        line(tag, base[(base["buy_usd"] >= lo) & (base["buy_usd"] < hi)])

    print("\n== LIQUIDITY (20d avg $ volume) ==")
    for lo, hi, tag in ((0, 1e6, "< $1M/day"), (1e6, 1e7, "$1M-10M"),
                        (1e7, 1e8, "$10M-100M"), (1e8, np.inf, ">= $100M")):
        line(tag, base[(base["dollar_vol_20"] >= lo) & (base["dollar_vol_20"] < hi)])

    print("\n== PRICE CONTEXT AT ENTRY ==")
    line("after >15% 20d drop", base[base["ret_20"] < -15])
    line("20d ret -15..0%", base[(base["ret_20"] >= -15) & (base["ret_20"] < 0)])
    line("20d ret > 0", base[base["ret_20"] >= 0])
    line("below 50d MA", base[base["dist_ma50"] < 0])
    line("above 50d MA", base[base["dist_ma50"] >= 0])

    print("\n== BY YEAR ==")
    for y, s in base.groupby("year"):
        line(str(y), s)

    print("\n== COMBINED CANDIDATE FILTERS ==")
    liq = base[base["dollar_vol_20"] >= 1e6]
    line("liquid (>=$1M/day)", liq)
    line("liquid + opportunistic", liq[liq["trader_type"] == "opportunistic"])
    line("liquid + C-suite", liq[liq["is_csuite"] == 1])
    line("liquid + cluster 3+", liq[liq["n_insiders_30d"] >= 3])
    line("liquid + >= $100k", liq[liq["buy_usd"] >= 1e5])
    line("liquid + not routine + >=$100k",
         liq[(liq["trader_type"] != "routine") & (liq["buy_usd"] >= 1e5)])
    line("liquid + C-suite/director + >=$100k",
         liq[((liq["is_csuite"] == 1) | (liq["is_director"] == 1)) & (liq["buy_usd"] >= 1e5)])
    line("liquid + opp/unclass + csuite + >=$100k",
         liq[(liq["trader_type"] != "routine") & (liq["is_csuite"] == 1) & (liq["buy_usd"] >= 1e5)])

    if args.out:
        d.to_csv(Path("data") / args.out, index=False)
        print(f"\nwrote {len(d):,} labelled events -> data/{args.out}")


if __name__ == "__main__":
    main()
