"""What would an ACCOUNT have earned trading the insider signal?

    python insider_portfolio_sim.py --labelled insider_labelled_21.csv --slots 10

Per-event means (insider_universe_study.py) answer "is there a signal?". This
answers "what does a capital-constrained portfolio do with it?" — which is a
different question, because events arrive in bursts (March 2020, mid 2022) and
a fixed number of slots cannot take them all, and because the drawdown of a
long-only book of beaten-down stocks is the thing that actually decides
whether anyone can hold the strategy.

Mechanics, deliberately simple and fully causal:
  * `slots` equal-weight positions; size = equity / slots at entry (compounds)
  * each day: close positions whose exit date has come, then fill free slots
    with that day's qualifying events, ranked by conviction (C-suite, then
    director, then purchase size); one position per symbol
  * P&L per trade is the labelled barrier outcome × ATR% × size − slippage,
    exactly as the study measured it
  * monthly-cluster t-stat on the equity curve, since March-2020-style bursts
    make events correlated across symbols and per-trade t overstates it

Benchmark: buy-and-hold SPY over the same window, fetched from Alpaca.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd

import config
from alpaca.data.enums import DataFeed as AlpacaFeed
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame

FILTERS = {
    # name: callable(df) -> mask. Literature-backed only; nothing fitted here.
    "all":        lambda d: np.ones(len(d), bool),
    "liquid":     lambda d: d["dollar_vol_20"] >= 1e6,
    "core":       lambda d: (d["dollar_vol_20"] >= 1e6) & (d["trader_type"] != "routine")
                            & (d["buy_usd"] >= 1e5) & ((d["is_csuite"] == 1) | (d["is_director"] == 1)),
    "core_dip":   lambda d: (d["dollar_vol_20"] >= 1e6) & (d["trader_type"] != "routine")
                            & (d["buy_usd"] >= 1e5) & ((d["is_csuite"] == 1) | (d["is_director"] == 1))
                            & (d["dist_ma50"] < 0),
    "core_10m":   lambda d: (d["dollar_vol_20"] >= 1e7) & (d["trader_type"] != "routine")
                            & (d["buy_usd"] >= 1e5) & ((d["is_csuite"] == 1) | (d["is_director"] == 1)),
}


def spy_closes(start, end) -> pd.Series:
    c = StockHistoricalDataClient(config.ALPACA_API_KEY, config.ALPACA_SECRET_KEY)
    bars = c.get_stock_bars(StockBarsRequest(
        symbol_or_symbols="SPY", timeframe=TimeFrame.Day,
        start=start - timedelta(days=400), end=min(end, datetime.now(timezone.utc) - timedelta(days=1)),
        feed=AlpacaFeed.SIP)).data.get("SPY", [])
    return pd.Series([b.close for b in bars],
                     index=[pd.Timestamp(b.timestamp).tz_convert("America/New_York").normalize().tz_localize(None)
                            for b in bars])


def simulate(ev: pd.DataFrame, slots: int, equity0: float, slip_bps: float,
             exposure: float = 1.0, sizing: str = "equal",
             risk_pct: float = 1.0, regime_ok: pd.Series | None = None,
             closes: pd.DataFrame | None = None) -> tuple[pd.Series, pd.DataFrame]:
    """sizing  equal  every slot = equity * exposure / slots
              risk   size so that the 1.5-ATR stop costs risk_pct of equity,
                     capped at the equal-weight slot - volatile names get less
    regime_ok  optional date -> bool; no NEW entries on days where False
               (exits still run). The March-2020 drawdown was 94 straight
               dip-buys into a crash at a 10% win rate — a long-only event
               strategy needs permission from the tape to add risk.
    closes     optional date x symbol close matrix. With it the equity curve is
               MARKED TO MARKET every trading day; without it only realized
               P&L moves the curve, which understates drawdown."""
    ev = ev.sort_values(["entry_date", "rank"]).reset_index(drop=True)
    by_day = {d: g for d, g in ev.groupby("entry_date", sort=True)}
    days = pd.DatetimeIndex(sorted(set(ev["entry_date"]) | set(ev["exit_date"])))

    if closes is not None:
        days = closes.index[(closes.index >= days[0]) & (closes.index <= days[-1])]
    equity = equity0
    open_pos: list[dict] = []
    curve, trades = [], []
    for day in days:
        # 1. exits
        still = []
        for p in open_pos:
            if p["exit_date"] <= day:
                pnl = p["fwd_atr"] * p["atr_pct"] / 100 * p["size"] - 2 * slip_bps / 1e4 * p["size"]
                equity += pnl
                trades.append({**p, "pnl": pnl, "closed": day})
            else:
                still.append(p)
        open_pos = still
        # 2. entries
        g = by_day.get(day)
        if regime_ok is not None and not bool(regime_ok.get(day, True)):
            g = None
        if g is not None:
            held = {p["symbol"] for p in open_pos}
            for _, e in g.iterrows():
                if len(open_pos) >= slots:
                    break
                if e["symbol"] in held:
                    continue
                size = equity * exposure / slots
                if sizing == "risk" and e["atr_pct"] > 0:
                    size = min(size, equity * risk_pct / 100 / (1.5 * e["atr_pct"] / 100))
                open_pos.append({"symbol": e["symbol"], "entry": day, "exit_date": e["exit_date"],
                                 "fwd_atr": e["fwd_atr"], "atr_pct": e["atr_pct"], "size": size,
                                 "entry_px": (closes.at[day, e["symbol"]]
                                              if closes is not None and e["symbol"] in closes.columns
                                              else np.nan)})
                held.add(e["symbol"])
        mtm = 0.0
        if closes is not None:
            for p in open_pos:
                px = closes.at[day, p["symbol"]] if p["symbol"] in closes.columns else np.nan
                if np.isfinite(px) and np.isfinite(p["entry_px"]) and p["entry_px"] > 0:
                    mtm += p["size"] * (px / p["entry_px"] - 1)
        curve.append((day, equity + mtm))
    return pd.Series(dict(curve)), pd.DataFrame(trades)


def stats(curve: pd.Series, label: str, spy: pd.Series | None = None) -> None:
    ret = curve.pct_change().dropna()
    years = (curve.index[-1] - curve.index[0]).days / 365.25
    cagr = (curve.iloc[-1] / curve.iloc[0]) ** (1 / years) - 1
    dd = (curve / curve.cummax() - 1).min()
    monthly = curve.resample("ME").last().pct_change().dropna()
    sharpe = monthly.mean() / monthly.std() * np.sqrt(12) if monthly.std() > 0 else float("nan")
    t_month = monthly.mean() / (monthly.std() / np.sqrt(len(monthly)))
    line = (f"  {label:12s} total {100*(curve.iloc[-1]/curve.iloc[0]-1):+7.1f}%  "
            f"CAGR {100*cagr:+5.1f}%  maxDD {100*dd:6.1f}%  Sharpe {sharpe:4.2f}  "
            f"months>0 {100*(monthly>0).mean():4.0f}%  t(monthly) {t_month:+4.2f}")
    if spy is not None:
        s = spy.reindex(curve.index, method="ffill")
        s_cagr = (s.iloc[-1] / s.iloc[0]) ** (1 / years) - 1
        s_dd = (s / s.cummax() - 1).min()
        line += f"   | SPY CAGR {100*s_cagr:+5.1f}% maxDD {100*s_dd:6.1f}%"
    print(line)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--labelled", default="insider_labelled_21.csv")
    p.add_argument("--slots", type=int, default=10)
    p.add_argument("--equity", type=float, default=100_000.0)
    p.add_argument("--slip-bps", type=float, default=5.0, help="per side")
    p.add_argument("--filters", default="all,liquid,core,core_dip,core_10m")
    p.add_argument("--exposure", type=float, default=1.0,
                   help="max fraction of equity deployed across all slots")
    p.add_argument("--sizing", default="equal", choices=["equal", "risk"])
    p.add_argument("--risk-pct", type=float, default=1.0,
                   help="risk sizing: percent of equity lost if the stop is hit")
    p.add_argument("--regime", default="none", choices=["none", "spy200", "spy50", "spy_dd10"],
                   help="entry permission: SPY above its 200d/50d MA, or SPY within "
                        "10 percent of its 1y high")
    p.add_argument("--mtm", action="store_true",
                   help="mark open positions to market daily (loads daily_universe.parquet)")
    args = p.parse_args()

    d = pd.read_csv(Path("data") / args.labelled, parse_dates=["entry_date", "exit_date"])
    d = d[d["discount_pct"].abs() < 10]            # placements are not the signal
    if "is_10b5_1" in d.columns:
        d = d[d["is_10b5_1"] == 0]                 # scheduled plan trades: no information
    d["rank"] = -(d["is_csuite"] * 2 + d["is_director"]) - np.log10(d["buy_usd"].clip(1)) / 10
    print(f"{len(d):,} events {d['entry_date'].min():%Y-%m-%d} .. {d['entry_date'].max():%Y-%m-%d}, "
          f"{args.slots} slots, ${args.equity:,.0f}, exposure {args.exposure:g}, "
          f"sizing {args.sizing}, {args.slip_bps:g}bp/side\n")

    closes = spy_closes(d["entry_date"].min().tz_localize("UTC"), d["exit_date"].max().tz_localize("UTC"))
    spy = closes[closes.index >= d["entry_date"].min()]
    spy = spy / spy.iloc[0]
    # Regime is evaluated on the PREVIOUS close — the bot decides during the
    # session and cannot see today's close.
    prev = closes.shift(1)
    regime = {
        "none": None,
        "spy200": prev > prev.rolling(200).mean(),
        "spy50": prev > prev.rolling(50).mean(),
        "spy_dd10": prev > 0.9 * prev.rolling(252).max(),
    }[args.regime]
    if regime is not None:
        print(f"regime {args.regime}: entries permitted on "
              f"{100*regime[regime.index >= spy.index[0]].mean():.0f}% of days\n")

    for name in args.filters.split(","):
        sub = d[FILTERS[name](d)]
        closes_mx = None
        if args.mtm:
            px = pd.read_parquet(Path("data") / "daily_universe.parquet",
                                 columns=["symbol", "ts", "close"])
            px = px[px["symbol"].isin(sub["symbol"].unique())]
            px["date"] = px["ts"].dt.tz_convert("America/New_York").dt.normalize().dt.tz_localize(None)
            closes_mx = px.pivot_table(index="date", columns="symbol", values="close").sort_index()
        curve, trades = simulate(sub, args.slots, args.equity, args.slip_bps,
                                 args.exposure, args.sizing, args.risk_pct, regime, closes_mx)
        if trades.empty:
            print(f"  {name}: no trades"); continue
        util = trades["size"].sum() / (args.equity * len(curve)) if len(curve) else 0
        print(f"[{name}] {len(sub):,} qualifying events -> {len(trades):,} trades taken, "
              f"win {100*(trades['pnl']>0).mean():.1f}%, mean ${trades['pnl'].mean():+.2f}, "
              f"avg hold {trades.apply(lambda r: (r['closed']-r['entry']).days, axis=1).mean():.0f} cal days")
        stats(curve, name, spy)
        yearly = curve.resample("YE").last().pct_change()
        yearly.iloc[0] = curve.resample("YE").last().iloc[0] / args.equity - 1
        n_by_year = trades.groupby(trades["closed"].dt.year).size()
        print("     by year: " + "  ".join(f"{ts.year}: {100*r:+.1f}% ({n_by_year.get(ts.year, 0)})"
                                          for ts, r in yearly.items()))
        print()


if __name__ == "__main__":
    main()
