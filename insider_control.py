"""Is the insider-buy result a signal, or just the stocks it happened to pick?

An apparent +$23/trade on insider events means nothing until compared with the
right control. The wrong control is "all bars of all symbols over all time" —
it differs in symbol mix AND period, so a concentrated, well-timed subset beats
it on composition alone.

The right control is SYMBOL- AND PERIOD-MATCHED: for every insider event, what
would entering the SAME stock on a RANDOM day in the SAME window have returned,
with the SAME barriers? That differences out both the market's drift and the
particular stocks involved, leaving only the event's timing.

    python insider_control.py

Also reports the result with the dominant symbol removed, because a dataset
where one name is 30% of rows can be one company's history wearing a costume.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

NOTIONAL = 1000.0
SLIPPAGE = 2 * 2.0 / 10_000 * NOTIONAL     # 2bp per side


def label_all_days(g: pd.DataFrame, horizon: int, tp_atr: float,
                   sl_atr: float) -> pd.DataFrame:
    """Gap-aware triple barrier from EVERY day's close — the control universe."""
    o, h = g["open"].to_numpy(), g["high"].to_numpy()
    l, c = g["low"].to_numpy(), g["close"].to_numpy()
    atr = g["atr_14"].to_numpy()
    n = len(g)
    ret = np.full(n, np.nan)
    for i in range(n):
        a = atr[i]
        if not np.isfinite(a) or a <= 0:
            continue
        tp, sl = c[i] + tp_atr * a, c[i] - sl_atr * a
        end = min(i + horizon + 1, n)
        val = None
        for j in range(i + 1, end):
            if o[j] <= sl:
                val = (o[j] - c[i]) / a; break
            if o[j] >= tp:
                val = (o[j] - c[i]) / a; break
            if l[j] <= sl:
                val = -sl_atr; break
            if h[j] >= tp:
                val = tp_atr; break
        if val is None:
            val = (c[min(end - 1, n - 1)] - c[i]) / a
        ret[i] = val
    g = g.copy()
    g["ctrl_ret_atr"] = ret
    return g


def dollars(ret_atr, atr_pct):
    return ret_atr * (atr_pct / 100.0 * NOTIONAL) - SLIPPAGE


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--events", default="data/dataset_insider.csv")
    p.add_argument("--prices", default="data/dataset_daily.csv")
    p.add_argument("--horizon", type=int, default=21)
    p.add_argument("--tp-atr", type=float, default=1.5)
    p.add_argument("--sl-atr", type=float, default=1.5)
    p.add_argument("--oos-days", type=int, default=550,
                   help="trailing window treated as out-of-sample")
    args = p.parse_args()

    for f in (Path(args.events), Path(args.prices)):
        if not f.exists():
            sys.exit(f"missing {f}")

    ev = pd.read_csv(args.events)
    ev["ts"] = pd.to_datetime(ev["ts"], utc=True, format="mixed")
    cutoff = ev["ts"].max() - pd.Timedelta(days=args.oos_days)
    oos = ev[ev["ts"] >= cutoff].copy()
    oos["pnl"] = dollars(oos["fwd_ret_atr"], oos["atr_pct"])

    px = pd.read_csv(args.prices)
    px["ts"] = pd.to_datetime(px["ts"], utc=True, format="mixed")

    # Control: every day of every symbol that produced an OOS event, over the
    # same window, labelled with identical barriers.
    syms = sorted(oos["symbol"].unique())
    frames = []
    for s in syms:
        g = px[px["symbol"] == s].sort_values("ts").reset_index(drop=True)
        if len(g) < args.horizon + 20:
            continue
        g = label_all_days(g, args.horizon, args.tp_atr, args.sl_atr)
        frames.append(g[g["ts"] >= cutoff])
    if not frames:
        sys.exit("no control data")
    ctrl = pd.concat(frames, ignore_index=True).dropna(subset=["ctrl_ret_atr", "atr_pct"])
    ctrl["pnl"] = dollars(ctrl["ctrl_ret_atr"], ctrl["atr_pct"])

    def line(tag, s):
        if len(s) == 0:
            print(f"  {tag:34s}      —")
            return
        print(f"  {tag:34s} {len(s):6,d} trades   "
              f"{s.mean():+8.2f} $/trade   win {100*(s > 0).mean():5.1f}%")

    print("=" * 78)
    print(f"INSIDER SIGNAL vs SYMBOL/PERIOD-MATCHED CONTROL "
          f"({cutoff:%Y-%m-%d} onward)")
    print("=" * 78)
    line("insider buy events", oos["pnl"])
    line("same symbols, EVERY day (control)", ctrl["pnl"])

    diff = oos["pnl"].mean() - ctrl["pnl"].mean()
    se = np.sqrt(oos["pnl"].var() / len(oos) + ctrl["pnl"].var() / len(ctrl))
    t = diff / se if se > 0 else 0.0
    print(f"\n  EXCESS over matched control: {diff:+.2f} $/trade   (t = {t:+.2f})")
    print("  t < 2 means the events are indistinguishable from buying these")
    print("  same stocks on any random day in the same period.")

    # Concentration: does one symbol carry it?
    top = oos["symbol"].value_counts()
    print(f"\n  concentration: {top.index[0]} is "
          f"{100*top.iloc[0]/len(oos):.0f}% of OOS events")
    for drop in list(top.index[:2]):
        sub_e = oos[oos["symbol"] != drop]
        sub_c = ctrl[ctrl["symbol"] != drop]
        if len(sub_e) < 20 or len(sub_c) < 20:
            continue
        d = sub_e["pnl"].mean() - sub_c["pnl"].mean()
        s = np.sqrt(sub_e["pnl"].var() / len(sub_e) + sub_c["pnl"].var() / len(sub_c))
        print(f"    excluding {drop:5s}: events {sub_e['pnl'].mean():+7.2f} vs "
              f"control {sub_c['pnl'].mean():+7.2f}   excess {d:+6.2f} "
              f"(t={d/s if s > 0 else 0:+.2f}, n={len(sub_e)})")

    print("\n  by insider role:")
    for col in ("is_csuite", "is_director", "is_10pct"):
        sel = oos[oos[col] == 1]
        if len(sel) >= 20:
            line(f"    {col}", sel["pnl"])

    cl = oos[oos["n_insiders_30d"] >= 3]
    if len(cl) >= 10:
        line("    cluster (3+ insiders/30d)", cl["pnl"])


if __name__ == "__main__":
    main()
