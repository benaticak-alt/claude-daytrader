"""Build a labeled training dataset from historical bars.

Nothing about an ML strategy can start before this exists. Live logs will never
accumulate enough — at ~78 cycles a day you would need decades. This pulls
years of 5-minute bars and labels them offline.

    python build_dataset.py --symbols SPY,QQQ,NVDA,TSLA,AAPL --years 2
    python build_dataset.py --help

LABELING: triple barrier (López de Prado). For each bar, look forward up to
HORIZON bars and ask which happens first:
    price rises  +TP_ATR x ATR   -> label 1  (a trade taken here would have won)
    price falls  -SL_ATR x ATR   -> label 0  (it would have lost)
    neither, horizon expires     -> label 0  (capital tied up for nothing)

Why barriers rather than "return after N bars": a fixed-horizon return ignores
the path. A bar that dips 2% before recovering is not a win — you would have
been stopped out. Barriers encode the stop, so labels match what the risk gate
would actually experience. (Your Polymarket model's top feature is `barrier_z`
— same idea, different market.)

Scaling by ATR rather than fixed percentages matters just as much: a 0.5% move
is noise in TSLA and a large move in SPY. Without it the model learns which
ticker is volatile, not what predicts direction.
"""

from __future__ import annotations

import argparse
import logging
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import config

try:
    import pandas as pd
    import numpy as np
except ImportError:
    sys.exit("This script needs pandas and numpy:  pip install pandas numpy")

from alpaca.data.enums import DataFeed as AlpacaFeed
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame, TimeFrameUnit

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger("dataset")

ET = ZoneInfo("America/New_York")
OUT_DIR = Path(__file__).parent / "data"


def fetch_bars(client, symbol: str, start: datetime, end: datetime,
               feed: AlpacaFeed) -> pd.DataFrame:
    """Page through history — the API caps each response, so loop until done."""
    frames = []
    cursor = start
    while cursor < end:
        req = StockBarsRequest(
            symbol_or_symbols=symbol,
            timeframe=TimeFrame(config.BAR_TIMEFRAME_MINUTES, TimeFrameUnit.Minute),
            start=cursor, end=end, feed=feed, limit=10000,
        )
        bars = client.get_stock_bars(req).data.get(symbol, [])
        if not bars:
            break
        frames.append(pd.DataFrame([{
            "ts": b.timestamp, "open": b.open, "high": b.high, "low": b.low,
            "close": b.close, "volume": b.volume,
        } for b in bars]))
        newest = bars[-1].timestamp
        if newest <= cursor:
            break  # no forward progress; stop rather than loop forever
        cursor = newest + timedelta(minutes=config.BAR_TIMEFRAME_MINUTES)
        log.info("  %s: %d bars, through %s", symbol,
                 sum(len(f) for f in frames), newest.date())

    if not frames:
        return pd.DataFrame()
    df = pd.concat(frames, ignore_index=True).drop_duplicates("ts").sort_values("ts")
    return df.reset_index(drop=True)


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """Same signals the live context computes, vectorized over history.

    Everything here must be causal — computed only from data available at that
    bar. A single forward-looking column silently turns a mediocre model into a
    spectacular backtest that fails instantly in production.
    """
    et = df["ts"].dt.tz_convert(ET)
    df["date"] = et.dt.date
    df["minutes"] = et.dt.hour * 60 + et.dt.minute
    # Regular trading hours only — extended-hours bars are thin and erratic.
    df = df[(df["minutes"] >= 9 * 60 + 30) & (df["minutes"] < 16 * 60)].copy()
    df["min_since_open"] = df["minutes"] - (9 * 60 + 30)

    g = df.groupby("date", sort=False)

    # Session VWAP — resets daily, like the live one. cumsum is causal:
    # each row sees only itself and earlier rows of the same session.
    typical = (df["high"] + df["low"] + df["close"]) / 3
    df["_pv"] = typical * df["volume"]
    df["vwap"] = (g["_pv"].cumsum() / g["volume"].cumsum())
    df["pct_vs_vwap"] = 100 * (df["close"] - df["vwap"]) / df["vwap"]

    # RSI(14) — rolling, spans sessions like the live version.
    delta = df["close"].diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()
    df["rsi_14"] = 100 - 100 / (1 + gain / loss.replace(0, np.nan))
    df["rsi_14"] = df["rsi_14"].fillna(50.0)

    # ATR(14)
    prev_close = df["close"].shift(1)
    tr = pd.concat([
        df["high"] - df["low"],
        (df["high"] - prev_close).abs(),
        (df["low"] - prev_close).abs(),
    ], axis=1).max(axis=1)
    df["atr_14"] = tr.rolling(14).mean()
    df["atr_pct"] = 100 * df["atr_14"] / df["close"]

    # Volume relative to its own recent norm.
    df["vol_ratio"] = df["volume"] / df["volume"].rolling(20).mean()

    # Opening range position — the ORB strategy's core signal.
    #
    # LOOKAHEAD TRAP: the completed range is not known until the window closes.
    # Mapping it onto every bar of the day hands 09:30 bars a high computed from
    # 09:40 — a small leak that lets a model "predict" the first 15 minutes for
    # free and inflates every backtest metric. Null it out before the cutoff.
    or_cut = 9 * 60 + 30 + config.OPENING_RANGE_MINUTES
    in_or = df["minutes"] < or_cut
    or_high = df.loc[in_or].groupby("date")["high"].max()
    or_low = df.loc[in_or].groupby("date")["low"].min()
    df["after_or"] = (df["minutes"] >= or_cut).astype(int)
    df["or_high"] = df["date"].map(or_high).where(df["after_or"] == 1)
    df["or_low"] = df["date"].map(or_low).where(df["after_or"] == 1)
    df["pct_vs_or_high"] = 100 * (df["close"] - df["or_high"]) / df["or_high"]
    df["or_range_pct"] = 100 * (df["or_high"] - df["or_low"]) / df["or_low"]

    # Short-horizon momentum.
    for n in (1, 3, 6):
        df[f"ret_{n}"] = 100 * df["close"].pct_change(n)

    # Overnight gap: today's opening print against the PREVIOUS session's close.
    # (Comparing the first bar's open to its own close is not a gap, it is just
    # that bar's body.) Known at 09:30, so causal for the whole session.
    daily_open = g["open"].first()
    daily_close = g["close"].last()
    prev_close = daily_close.shift(1)
    gap = 100 * (daily_open - prev_close) / prev_close
    df["gap_pct"] = df["date"].map(gap)

    return df.drop(columns=["_pv"], errors="ignore")


def add_labels(df: pd.DataFrame, horizon: int, tp_atr: float, sl_atr: float) -> pd.DataFrame:
    """Triple-barrier labels. Path-aware, ATR-scaled, strictly forward-looking.

    Uses future bars BY DESIGN — that is what a label is. The features must
    never do this; only this function may.
    """
    highs = df["high"].to_numpy()
    lows = df["low"].to_numpy()
    close = df["close"].to_numpy()
    atr = df["atr_14"].to_numpy()
    dates = df["date"].to_numpy()

    n = len(df)
    labels = np.full(n, np.nan)
    for i in range(n):
        a = atr[i]
        if not np.isfinite(a) or a <= 0:
            continue
        tp = close[i] + tp_atr * a
        sl = close[i] - sl_atr * a
        end = min(i + horizon + 1, n)
        outcome = 0  # horizon expiry counts as a loss: capital tied up, no gain
        for j in range(i + 1, end):
            if dates[j] != dates[i]:
                break  # never label across an overnight gap
            if lows[j] <= sl:
                outcome = 0
                break
            if highs[j] >= tp:
                outcome = 1
                break
        labels[i] = outcome

    df["label"] = labels
    return df


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--symbols", default=",".join(config.WATCHLIST))
    p.add_argument("--years", type=float, default=2.0)
    p.add_argument("--horizon", type=int, default=12, help="bars to look forward (12 = 1h)")
    p.add_argument("--tp-atr", type=float, default=1.0, help="take-profit in ATRs")
    p.add_argument("--sl-atr", type=float, default=1.0, help="stop-loss in ATRs")
    p.add_argument("--out", default="dataset.parquet")
    args = p.parse_args()

    symbols = [s.strip().upper() for s in args.symbols.split(",") if s.strip()]
    client = StockHistoricalDataClient(config.ALPACA_API_KEY, config.ALPACA_SECRET_KEY)
    feed = AlpacaFeed.SIP if config.ALPACA_DATA_FEED == "sip" else AlpacaFeed.IEX

    end = datetime.now(timezone.utc) - timedelta(minutes=20)
    start = end - timedelta(days=int(365 * args.years))
    log.info("fetching %s from %s (%s feed)", symbols, start.date(), feed.value)

    frames = []
    for sym in symbols:
        raw = fetch_bars(client, sym, start, end, feed)
        if raw.empty:
            log.warning("%s: no data", sym)
            continue
        feat = add_features(raw)
        feat = add_labels(feat, args.horizon, args.tp_atr, args.sl_atr)
        feat["symbol"] = sym
        frames.append(feat)
        log.info("%s: %d rows", sym, len(feat))

    if not frames:
        sys.exit("no data fetched")

    df = pd.concat(frames, ignore_index=True).dropna(subset=["label", "atr_14"])
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUT_DIR / args.out
    try:
        df.to_parquet(path, index=False)
    except Exception:
        path = path.with_suffix(".csv")
        df.to_csv(path, index=False)

    print(f"\nwrote {len(df):,} labeled rows -> {path}")
    print(f"  date range : {df['ts'].min():%Y-%m-%d} .. {df['ts'].max():%Y-%m-%d}")
    print(f"  positive   : {100*df['label'].mean():.1f}%  "
          f"(a 50/50 split means the barriers are balanced)")
    print(f"  per symbol : {df.groupby('symbol').size().to_dict()}")
    print("\nNOTE: base rate is what any model must beat. If positives are 50%,"
          "\na model scoring 52% is a real (if small) edge; 50% is nothing.")


if __name__ == "__main__":
    main()
