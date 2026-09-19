"""Daily bars for every symbol that has an insider buy event.

    python fetch_daily_universe.py            # all symbols in the event file
    python fetch_daily_universe.py --limit 50 # smoke test

Two things make this materially better than the 80-symbol research pass:

  * SIP, not IEX. The free tier forbids only the LAST 15 MINUTES of SIP data;
    historical daily bars are full consolidated volume. IEX-only daily bars
    for a thin name are built from a sliver of the tape and skip days with no
    IEX print (MLVF: 396 IEX days vs 890 SIP days over the same span).

  * Delisted symbols ARE served. Alpaca returns history for inactive assets
    (CNBKA: bars through its 2021 delisting), so insiders who bought companies
    that later disappeared are IN the sample. That was the single largest
    unquantified bias in the earlier result; here it can be measured.

Output: data/daily_universe.parquet with per-symbol causal features
(atr_14, atr_pct, ret_5/20, dist_ma20/50, pos_in_20d_range, vol_ratio,
dollar_vol_20) plus each symbol's last bar date, so a series that simply ends
can be told apart from one that is still trading.
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd

import config
from alpaca.data.enums import Adjustment
from alpaca.data.enums import DataFeed as AlpacaFeed
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger("daily_universe")

EVENTS = Path(__file__).parent / "data" / "insider_events_universe.csv"
OUT = Path(__file__).parent / "data" / "daily_universe.parquet"
BATCH = 200


def features(df: pd.DataFrame) -> pd.DataFrame:
    """Causal daily features — same definitions as build_daily_dataset.py."""
    df = df.sort_values("ts").reset_index(drop=True)
    c, h, l = df["close"], df["high"], df["low"]
    prev_c = c.shift(1)
    tr = pd.concat([h - l, (h - prev_c).abs(), (l - prev_c).abs()], axis=1).max(axis=1)
    df["atr_14"] = tr.rolling(14).mean()
    df["atr_pct"] = 100 * df["atr_14"] / c
    df["ret_5"] = 100 * c.pct_change(5)
    df["ret_20"] = 100 * c.pct_change(20)
    ma20, ma50 = c.rolling(20).mean(), c.rolling(50).mean()
    df["dist_ma20"] = 100 * (c - ma20) / ma20
    df["dist_ma50"] = 100 * (c - ma50) / ma50
    hi20, lo20 = h.rolling(20).max(), l.rolling(20).min()
    df["pos_in_20d_range"] = (c - lo20) / (hi20 - lo20).replace(0, np.nan)
    df["vol_ratio"] = df["volume"] / df["volume"].rolling(20).mean()
    df["dollar_vol_20"] = (c * df["volume"]).rolling(20).mean()
    return df


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--start", default="2019-10-01")
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--raw-close", action="store_true",
                   help="fetch UNADJUSTED closes only (data/daily_universe_rawclose.parquet) "
                        "— Form 4 prices are as-traded, so the paid-vs-market discount "
                        "check must compare against unadjusted closes")
    args = p.parse_args()

    ev = pd.read_csv(EVENTS)
    symbols = sorted(ev["symbol"].unique())
    if args.limit:
        symbols = symbols[: args.limit]
    log.info("%d symbols to fetch", len(symbols))

    client = StockHistoricalDataClient(config.ALPACA_API_KEY, config.ALPACA_SECRET_KEY)
    start = datetime.fromisoformat(args.start).replace(tzinfo=timezone.utc)
    # SIP is allowed once the data is older than 15 minutes; yesterday is safe.
    end = datetime.now(timezone.utc) - timedelta(days=1)

    frames, missing = [], 0
    t0 = time.time()
    for i in range(0, len(symbols), BATCH):
        chunk = symbols[i:i + BATCH]
        try:
            resp = client.get_stock_bars(StockBarsRequest(
                symbol_or_symbols=chunk, timeframe=TimeFrame.Day,
                start=start, end=end, feed=AlpacaFeed.SIP,
                adjustment=Adjustment.RAW if args.raw_close else Adjustment.ALL,
            ))
        except Exception:
            log.exception("batch %d failed — retrying once", i // BATCH)
            time.sleep(5)
            try:
                resp = client.get_stock_bars(StockBarsRequest(
                    symbol_or_symbols=chunk, timeframe=TimeFrame.Day,
                    start=start, end=end, feed=AlpacaFeed.SIP,
                    adjustment=Adjustment.RAW if args.raw_close else Adjustment.ALL,
                ))
            except Exception:
                log.exception("batch %d failed twice — skipping", i // BATCH)
                missing += len(chunk)
                continue
        for sym in chunk:
            bars = resp.data.get(sym, [])
            if len(bars) < 60:
                missing += 1
                continue
            df = pd.DataFrame({
                "ts": [b.timestamp for b in bars],
                "open": [b.open for b in bars], "high": [b.high for b in bars],
                "low": [b.low for b in bars], "close": [b.close for b in bars],
                "volume": [b.volume for b in bars],
            })
            df["symbol"] = sym
            if args.raw_close:
                frames.append(df[["symbol", "ts", "close"]])
                continue
            frames.append(features(df))
        done = min(i + BATCH, len(symbols))
        log.info("%d/%d symbols  (%.0fs)", done, len(symbols), time.time() - t0)

    if not frames:
        sys.exit("nothing fetched")
    out = pd.concat(frames, ignore_index=True)
    out["ts"] = pd.to_datetime(out["ts"], utc=True)
    OUT.parent.mkdir(exist_ok=True)
    if args.raw_close:
        raw_out = OUT.with_name("daily_universe_rawclose.parquet")
        out.to_parquet(raw_out, index=False)
        print(f"\nwrote {len(out):,} raw closes -> {raw_out}")
        return
    out.to_parquet(OUT, index=False)

    last = out.groupby("symbol")["ts"].max()
    ended = (last < out["ts"].max() - pd.Timedelta(days=30)).sum()
    print(f"\nwrote {len(out):,} daily bars, {out['symbol'].nunique():,} symbols -> {OUT}")
    print(f"  {missing:,} symbols had <60 bars or no data")
    print(f"  {ended:,} symbols' series END before the sample does (delisted / "
          f"acquired / renamed) — these are the survivorship cases now IN sample")


if __name__ == "__main__":
    main()
