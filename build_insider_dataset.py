"""Join insider buy events to daily price history and label them.

The last structurally different signal: everything else tested here was derived
from price and volume. This is event data — who bought, what role, how much,
how many of them.

    python build_insider_dataset.py --horizon 21 --tp-atr 1.5 --sl-atr 1.5

THE FILING DATE IS THE ONLY TRADEABLE DATE. Form 4 allows two business days
between a transaction and its disclosure. Keying on transaction_date would
trade on information that was not yet public — a lookahead leak that inflates
results and evaporates live. Entry is therefore assumed at the CLOSE OF THE
FIRST TRADING DAY ON OR AFTER the filing date.

Labels reuse the daily builder's gap-aware triple barrier: a day that OPENS
through a barrier fills at the open, not the barrier, and a day touching both
resolves pessimistically (stop first).

Every feature is computed from data available at entry: price/volatility state
from the daily bars, plus the event's own attributes.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

CSUITE = ("chief executive", "chief financial", "ceo", "cfo", "president",
          "chief operating", "coo")


def load_prices(path: Path) -> pd.DataFrame:
    px = pd.read_csv(path)
    px["ts"] = pd.to_datetime(px["ts"], utc=True, format="mixed")
    px["date"] = px["ts"].dt.tz_convert("America/New_York").dt.date
    return px.sort_values(["symbol", "ts"]).reset_index(drop=True)


def label_from_entry(g: pd.DataFrame, idx: int, horizon: int,
                     tp_atr: float, sl_atr: float) -> tuple[float, float]:
    """Gap-aware triple barrier from the bar at `idx` (entry = its close)."""
    o = g["open"].to_numpy(); h = g["high"].to_numpy()
    l = g["low"].to_numpy();  c = g["close"].to_numpy()
    atr = g["atr_14"].to_numpy()
    a = atr[idx]
    if not np.isfinite(a) or a <= 0:
        return np.nan, np.nan
    tp, sl = c[idx] + tp_atr * a, c[idx] - sl_atr * a
    end = min(idx + horizon + 1, len(g))
    for j in range(idx + 1, end):
        if o[j] <= sl:
            return 0.0, (o[j] - c[idx]) / a          # gapped through the stop
        if o[j] >= tp:
            return 1.0, (o[j] - c[idx]) / a          # gapped through the target
        if l[j] <= sl:
            return 0.0, -sl_atr                       # pessimistic: stop first
        if h[j] >= tp:
            return 1.0, tp_atr
    return 0.0, (c[min(end - 1, len(g) - 1)] - c[idx]) / a   # horizon expiry


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--events", default="data/insider_history.csv")
    p.add_argument("--prices", default="data/dataset_daily.csv")
    p.add_argument("--horizon", type=int, default=21, help="trading days")
    p.add_argument("--tp-atr", type=float, default=1.5)
    p.add_argument("--sl-atr", type=float, default=1.5)
    p.add_argument("--cluster-days", type=int, default=30)
    p.add_argument("--out", default="dataset_insider.csv")
    args = p.parse_args()

    ev_path, px_path = Path(args.events), Path(args.prices)
    for f in (ev_path, px_path):
        if not f.exists():
            sys.exit(f"missing {f} — run fetch_insider_history.py / build_daily_dataset.py")

    ev = pd.read_csv(ev_path)
    ev = ev[(ev["code"] == "P") & (ev["acquired"] == 1)].copy()
    if ev.empty:
        sys.exit("no open-market buys in the event file")
    ev["filing_date"] = pd.to_datetime(ev["filing_date"]).dt.date
    ev = ev.sort_values(["symbol", "filing_date"]).reset_index(drop=True)

    px = load_prices(px_path)
    rows = []
    skipped_no_price = 0

    for sym, evs in ev.groupby("symbol", sort=False):
        g = px[px["symbol"] == sym].reset_index(drop=True)
        if g.empty:
            skipped_no_price += len(evs)
            continue
        dates = g["date"].to_numpy()

        for _, e in evs.iterrows():
            # Entry: first trading day ON OR AFTER the filing date.
            pos = np.searchsorted(dates, e["filing_date"], side="left")
            if pos >= len(g) - args.horizon - 1:
                continue          # not enough forward bars to resolve a label
            label, ret = label_from_entry(g, int(pos), args.horizon,
                                          args.tp_atr, args.sl_atr)
            if not np.isfinite(label):
                continue

            bar = g.iloc[pos]
            # Cluster context: distinct insiders buying this name in the
            # preceding window, counted only from filings already public.
            lo = e["filing_date"] - pd.Timedelta(days=args.cluster_days)
            prior = evs[(evs["filing_date"] <= e["filing_date"])
                        & (evs["filing_date"] > lo)]
            title = str(e.get("title") or "").lower()

            rows.append({
                "symbol": sym,
                "ts": bar["ts"],
                "label": label,
                "fwd_ret_atr": ret,
                # --- event features ---
                "buy_usd_log": float(np.log10(max(e["usd"], 1))),
                "n_insiders_30d": int(prior["owner"].nunique()),
                "n_buys_30d": int(len(prior)),
                "is_csuite": int(any(k in title for k in CSUITE)),
                "is_director": int("director" in title),
                "is_10pct": int("10%" in title),
                # --- price state at entry (all causal) ---
                "atr_pct": float(bar["atr_pct"]),
                "rsi_14": float(bar["rsi_14"]),
                "ret_5": float(bar["ret_5"]),
                "ret_20": float(bar["ret_20"]),
                "dist_ma20": float(bar["dist_ma20"]),
                "dist_ma50": float(bar["dist_ma50"]),
                "pos_in_20d_range": float(bar["pos_in_20d_range"]),
                "vol_ratio": float(bar["vol_ratio"]),
            })

    if not rows:
        sys.exit("no events could be joined to price history")
    out = pd.DataFrame(rows).dropna(subset=["label", "atr_pct"])
    path = Path("data") / args.out
    out.to_csv(path, index=False)

    print(f"\nwrote {len(out):,} insider-buy events -> {path}")
    print(f"  symbols     : {out['symbol'].nunique()}")
    print(f"  date range  : {out['ts'].min():%Y-%m-%d} .. {out['ts'].max():%Y-%m-%d}")
    print(f"  positive    : {100*out['label'].mean():.1f}%  "
          f"(±{args.tp_atr:g} ATR, {args.horizon}d horizon)")
    print(f"  clusters(3+): {int((out['n_insiders_30d'] >= 3).sum()):,}")
    print(f"  C-suite     : {int(out['is_csuite'].sum()):,}")
    if skipped_no_price:
        print(f"  skipped (no price history): {skipped_no_price:,}")
    top = out["symbol"].value_counts().head(5)
    print("\n  concentration — top 5 symbols:")
    for s, n in top.items():
        print(f"    {s:6s} {n:4d}  ({100*n/len(out):.0f}% of all events)")
    print("\n  If a few names dominate, any 'signal' may be their idiosyncratic")
    print("  history rather than insider behaviour. Check this before believing.")


if __name__ == "__main__":
    main()
