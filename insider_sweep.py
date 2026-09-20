"""Sweep the account-level levers with a fit / holdout split.

    python insider_sweep.py

Every configuration is scored on 2020-01..2023-12 (FIT) and separately on
2024-01..2026-06 (HOLDOUT). A lever that helps on the fit period and not on
the holdout is curve-fitting, and the table makes that visible instead of
letting the best fit-period number win by default. The holdout is 2.5 years
and ~40% of the events — small, so read differences of a few points as noise.

Also reports a 50/50 SPY blend (daily-rebalanced) for each config, since the
strategy's returns are largely uncorrelated with the index and the realistic
use is a tilt on top of an index position, not a replacement for it.
"""

from __future__ import annotations

import argparse
import itertools
from pathlib import Path

import numpy as np
import pandas as pd

from insider_portfolio_sim import FILTERS, simulate, spy_closes

FIT_END = pd.Timestamp("2023-12-31")


def metrics(curve: pd.Series) -> dict:
    if len(curve) < 30 or curve.iloc[0] <= 0:
        return {"cagr": np.nan, "dd": np.nan, "sharpe": np.nan}
    years = (curve.index[-1] - curve.index[0]).days / 365.25
    if years <= 0:
        return {"cagr": np.nan, "dd": np.nan, "sharpe": np.nan}
    m = curve.resample("ME").last().pct_change().dropna()
    return {
        "cagr": (curve.iloc[-1] / curve.iloc[0]) ** (1 / years) - 1,
        "dd": (curve / curve.cummax() - 1).min(),
        "sharpe": m.mean() / m.std() * np.sqrt(12) if m.std() > 0 else np.nan,
    }


def daily(curve: pd.Series, index: pd.DatetimeIndex) -> pd.Series:
    return curve.reindex(index, method="ffill").bfill()


def blend(strat: pd.Series, spy: pd.Series, w: float = 0.5) -> pd.Series:
    r = w * strat.pct_change().fillna(0) + (1 - w) * spy.pct_change().fillna(0)
    return (1 + r).cumprod()


def split(curve: pd.Series) -> tuple[pd.Series, pd.Series]:
    fit = curve[curve.index <= FIT_END]
    hold = curve[curve.index > FIT_END]
    return fit, hold


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--labelled", nargs="+",
                   default=["insider_labelled_21_open1.csv"])
    p.add_argument("--slots", default="10,20")
    p.add_argument("--exposure", default="0.5,1.0")
    p.add_argument("--sizing", default="equal,risk")
    p.add_argument("--regime", default="none,spy50")
    p.add_argument("--filters", default="core,liquid")
    p.add_argument("--slip-bps", type=float, default=5.0)
    args = p.parse_args()

    rows = []
    spy_cache = None
    for lab in args.labelled:
        d = pd.read_csv(Path("data") / lab, parse_dates=["entry_date", "exit_date"])
        d = d[d["discount_pct"].abs() < 10]
        if "is_10b5_1" in d.columns:
            d = d[d["is_10b5_1"] == 0]
        d["rank"] = -(d["is_csuite"] * 2 + d["is_director"]) - np.log10(d["buy_usd"].clip(1)) / 10
        if spy_cache is None:
            closes = spy_closes(d["entry_date"].min().tz_localize("UTC"),
                                d["exit_date"].max().tz_localize("UTC"))
            prev = closes.shift(1)
            regimes = {"none": None,
                       "spy50": prev > prev.rolling(50).mean(),
                       "spy200": prev > prev.rolling(200).mean()}
            spy_cache = (closes, regimes)
        closes, regimes = spy_cache
        idx = closes.index[(closes.index >= d["entry_date"].min())
                           & (closes.index <= d["exit_date"].max())]
        spy_d = closes.reindex(idx)
        spy_m = metrics(spy_d)
        spy_fit, spy_hold = split(spy_d)

        grid = itertools.product(
            args.filters.split(","), [int(s) for s in args.slots.split(",")],
            [float(x) for x in args.exposure.split(",")],
            args.sizing.split(","), args.regime.split(","),
        )
        for filt, slots, expo, sizing, reg in grid:
            sub = d[FILTERS[filt](d)]
            curve, trades = simulate(sub, slots, 100_000.0, args.slip_bps, expo, sizing, 1.0,
                                     regimes[reg])
            if trades.empty:
                continue
            cd = daily(curve, idx)
            fit, hold = split(cd)
            mf, mh, ma = metrics(fit), metrics(hold), metrics(cd)
            bl = blend(cd, spy_d)
            bf, bh = split(bl)
            mbf, mbh = metrics(bf), metrics(bh)
            rows.append({
                "labels": lab.replace("insider_labelled_", "").replace(".csv", ""),
                "filter": filt, "slots": slots, "expo": expo, "sizing": sizing, "regime": reg,
                "trades": len(trades),
                "fit_cagr": mf["cagr"], "fit_dd": mf["dd"], "fit_sh": mf["sharpe"],
                "hold_cagr": mh["cagr"], "hold_dd": mh["dd"], "hold_sh": mh["sharpe"],
                "all_cagr": ma["cagr"], "all_dd": ma["dd"], "all_sh": ma["sharpe"],
                "blend_fit_sh": mbf["sharpe"], "blend_hold_sh": mbh["sharpe"],
                "blend_hold_cagr": mbh["cagr"], "blend_hold_dd": mbh["dd"],
            })
        print(f"[{lab}] SPY  fit CAGR {100*metrics(spy_fit)['cagr']:+.1f}% DD {100*metrics(spy_fit)['dd']:.0f}% "
              f"Sh {metrics(spy_fit)['sharpe']:.2f} | hold CAGR {100*metrics(spy_hold)['cagr']:+.1f}% "
              f"DD {100*metrics(spy_hold)['dd']:.0f}% Sh {metrics(spy_hold)['sharpe']:.2f}")

    t = pd.DataFrame(rows)
    pd.set_option("display.width", 250)
    fmt = {c: (lambda v: f"{100*v:+.0f}%") for c in t.columns if "cagr" in c or "_dd" in c}
    fmt.update({c: (lambda v: f"{v:.2f}") for c in t.columns if "_sh" in c})
    print("\n" + t.sort_values("hold_sh", ascending=False).to_string(index=False, formatters=fmt))
    out = Path("data") / "insider_sweep.csv"
    t.to_csv(out, index=False)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
