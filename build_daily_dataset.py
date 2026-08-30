"""Daily-bar dataset for the swing-horizon hypothesis.

WHY THIS EXISTS: both intraday models found a real signal (AUC positive in
all 12 walk-forward folds) that could not pay the fixed ~$0.40 round-trip
cost. The signal STRENGTHENED with horizon (0.527 at 1h -> 0.564 at 4h) while
the cost stays fixed. At a daily horizon a 1-ATR move on $1,000 is ~$20, so
the cost hurdle falls from ~500%% of gross edge to ~2%%. This is the third and
final pre-registered test of that one hypothesis — not a parameter search.

    python build_daily_dataset.py --years 6

Labeling is MORE honest than the intraday version, because overnight gaps are
the dominant risk at this horizon:

  * gap-through fills: if a day OPENS beyond a barrier, the trade realizes the
    open price, not the barrier — stops routinely fill worse than placed.
  * both-barriers-in-one-day resolves PESSIMISTICALLY (stop first).

Entry is assumed at the signal day's close. Horizon 10 trading days,
barriers ±1.5 daily ATR.
"""

from __future__ import annotations

import argparse
import logging
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd

import config
from alpaca.data.enums import DataFeed as AlpacaFeed
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger("daily")

# Liquid US names across sectors. Alpaca only serves currently-listed symbols,
# so delisted losers are absent — survivorship bias flatters longs slightly.
# Acknowledged; it biases FOR deployability, so a negative verdict still stands.
UNIVERSE = """
SPY QQQ IWM DIA AAPL MSFT NVDA GOOGL AMZN META TSLA AVGO AMD INTC MU QCOM TXN
CRM ORCL ADBE NFLX PLTR SNOW UBER ABNB COIN SHOP PYPL SOFI HOOD JPM BAC WFC GS
MS C SCHW V MA AXP XOM CVX COP SLB OXY UNH JNJ PFE MRK ABBV LLY BMY WMT COST
TGT HD LOW NKE SBUX MCD DIS CMCSA T VZ BA CAT DE GE F GM RIVN LCID DAL UAL AAL
CCL MARA RIOT ROKU SNAP DKNG AFRM UPST CVNA GME AMC
""".split()

TP_ATR = 1.5
SL_ATR = 1.5
HORIZON_DAYS = 10


def features(df: pd.DataFrame) -> pd.DataFrame:
    """Causal daily features, all volatility- or percent-normalized."""
    c, h, l, v = df["close"], df["high"], df["low"], df["volume"]

    for n in (1, 5, 20):
        df[f"ret_{n}"] = 100 * c.pct_change(n)

    delta = c.diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()
    df["rsi_14"] = (100 - 100 / (1 + gain / loss.replace(0, np.nan))).fillna(50.0)

    prev_c = c.shift(1)
    tr = pd.concat([h - l, (h - prev_c).abs(), (l - prev_c).abs()], axis=1).max(axis=1)
    df["atr_14"] = tr.rolling(14).mean()
    df["atr_pct"] = 100 * df["atr_14"] / c

    df["vol_ratio"] = v / v.rolling(20).mean()
    df["dist_ma20"] = 100 * (c - c.rolling(20).mean()) / c.rolling(20).mean()
    df["dist_ma50"] = 100 * (c - c.rolling(50).mean()) / c.rolling(50).mean()
    rng_hi, rng_lo = h.rolling(20).max(), l.rolling(20).min()
    df["pos_in_20d_range"] = (c - rng_lo) / (rng_hi - rng_lo).replace(0, np.nan)
    df["gap_pct"] = 100 * (df["open"] - prev_c) / prev_c
    df["dow"] = df["ts"].dt.dayofweek
    return df


def label(df: pd.DataFrame) -> pd.DataFrame:
    """Triple barrier with gap-through fills, entry at signal-day close."""
    o = df["open"].to_numpy()
    h = df["high"].to_numpy()
    l = df["low"].to_numpy()
    c = df["close"].to_numpy()
    atr = df["atr_14"].to_numpy()
    n = len(df)
    lab = np.full(n, np.nan)
    ret = np.full(n, np.nan)

    for i in range(n):
        a = atr[i]
        if not np.isfinite(a) or a <= 0:
            continue
        tp = c[i] + TP_ATR * a
        sl = c[i] - SL_ATR * a
        outcome, realized = 0, None
        end = min(i + HORIZON_DAYS + 1, n)
        for j in range(i + 1, end):
            if o[j] <= sl:                      # gapped through the stop
                realized = (o[j] - c[i]) / a
                break
            if o[j] >= tp:                      # gapped through the target
                outcome, realized = 1, (o[j] - c[i]) / a
                break
            hit_sl = l[j] <= sl
            hit_tp = h[j] >= tp
            if hit_sl:                          # pessimistic: stop before target
                realized = -SL_ATR
                break
            if hit_tp:
                outcome, realized = 1, TP_ATR
                break
        if realized is None:                    # horizon expiry
            realized = (c[min(end - 1, n - 1)] - c[i]) / a
        lab[i], ret[i] = outcome, realized

    df["label"] = lab
    df["fwd_ret_atr"] = ret
    return df


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--years", type=float, default=6.0)
    p.add_argument("--out", default="dataset_daily.csv")
    args = p.parse_args()

    client = StockHistoricalDataClient(config.ALPACA_API_KEY, config.ALPACA_SECRET_KEY)
    end = datetime.now(timezone.utc) - timedelta(days=1)
    start = end - timedelta(days=int(365 * args.years))

    frames = []
    for sym in UNIVERSE:
        try:
            bars = client.get_stock_bars(StockBarsRequest(
                symbol_or_symbols=sym, timeframe=TimeFrame.Day,
                start=start, end=end, feed=AlpacaFeed.IEX, limit=10000,
            )).data.get(sym, [])
        except Exception as exc:
            log.warning("%s: fetch failed (%s) — skipping", sym, type(exc).__name__)
            continue
        if len(bars) < 120:
            log.warning("%s: only %d bars — skipping", sym, len(bars))
            continue
        df = pd.DataFrame([{
            "ts": b.timestamp, "open": b.open, "high": b.high, "low": b.low,
            "close": b.close, "volume": b.volume,
        } for b in bars]).drop_duplicates("ts").sort_values("ts").reset_index(drop=True)
        df["ts"] = pd.to_datetime(df["ts"], utc=True)
        df = label(features(df))
        df["symbol"] = sym
        frames.append(df)
        log.info("%s: %d rows", sym, len(df))

    if not frames:
        sys.exit("nothing fetched")
    out = pd.concat(frames, ignore_index=True).dropna(subset=["label", "atr_14"])
    path = Path("data") / args.out
    path.parent.mkdir(exist_ok=True)
    out.to_csv(path, index=False)
    print(f"\nwrote {len(out):,} rows ({out['symbol'].nunique()} symbols) -> {path}")
    print(f"  {out['ts'].min():%Y-%m-%d} .. {out['ts'].max():%Y-%m-%d}")
    print(f"  positive: {100*out['label'].mean():.1f}%  "
          f"(±{TP_ATR:g}/{SL_ATR:g} daily ATR, {HORIZON_DAYS}d horizon, gap-through fills)")


if __name__ == "__main__":
    main()
