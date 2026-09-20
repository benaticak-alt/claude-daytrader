"""Earnings-announcement dates for every listed issuer, from EDGAR 8-K Item 2.02.

    python fetch_earnings_events.py --from 2020-01-01 --to 2026-09-19

Item 2.02 ("Results of Operations and Financial Condition") is the 8-K a
company must file when it releases quarterly results — the actual
announcement, days to weeks before the 10-Q. EDGAR full-text search exposes
each filing's item list and the issuer's ticker, so one query per calendar day
yields every announcement that day without opening a single document.

Output: data/earnings_events.csv  (symbol, cik, filing_date, accession)

This is the event table for a post-earnings-announcement-drift (PEAD) test.
The "surprise" will be the market's own reaction around the announcement
(Brandt et al. 2008, "earnings announcement return") — no analyst consensus
data needed, which is what keeps this free.

Rate: one request per page of 100 hits, ~8 req/s, resumable by day.
"""

from __future__ import annotations

import argparse
import json
import re
import time
from datetime import date, timedelta
from pathlib import Path

import pandas as pd
import requests

import config

EFTS_URL = "https://efts.sec.gov/LATEST/search-index"
CACHE = Path(__file__).parent / "cache" / "earnings_days"
OUT = Path(__file__).parent / "data" / "earnings_events.csv"
_HEADERS = {"User-Agent": config.SEC_USER_AGENT, "Accept-Encoding": "gzip, deflate"}
_TICKER = re.compile(r"\(([A-Z][A-Z.\-]{0,5})\)\s+\(CIK (\d+)\)")


def fetch_day(d: date, s: requests.Session) -> list[dict]:
    rows, frm, last = [], 0, 0.0
    while True:
        wait = 0.13 - (time.time() - last)
        if wait > 0:
            time.sleep(wait)
        r = s.get(EFTS_URL, params={"q": '""', "forms": "8-K", "dateRange": "custom",
                                    "startdt": d.isoformat(), "enddt": d.isoformat(),
                                    "from": frm}, timeout=30)
        last = time.time()
        if r.status_code == 429:
            time.sleep(10)
            continue
        r.raise_for_status()
        page = r.json().get("hits", {}).get("hits", [])
        if not page:
            break
        for h in page:
            src = h.get("_source", {})
            if src.get("form") != "8-K" or "2.02" not in (src.get("items") or []):
                continue
            for name in src.get("display_names", []):
                m = _TICKER.search(name)
                if m:
                    rows.append({"symbol": m.group(1), "cik": m.group(2).lstrip("0"),
                                 "filing_date": src.get("file_date"),
                                 "accession": h["_id"].split(":")[0]})
                    break
        frm += len(page)
        if len(page) < 100 or frm >= 10_000:
            break
    return rows


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--from", dest="start", default="2020-01-01")
    p.add_argument("--to", dest="end", default=date.today().isoformat())
    args = p.parse_args()
    CACHE.mkdir(parents=True, exist_ok=True)
    s = requests.Session()
    s.headers.update(_HEADERS)

    d, end = date.fromisoformat(args.start), date.fromisoformat(args.end)
    t0, n_days = time.time(), 0
    while d <= end:
        f = CACHE / f"{d.isoformat()}.json"
        if d.weekday() < 5 and not f.exists():
            try:
                rows = fetch_day(d, s)
            except Exception as e:                       # noqa: BLE001
                print(f"  {d}: FAILED {e} — will retry on next run")
                d += timedelta(days=1)
                continue
            f.write_text(json.dumps(rows), encoding="utf-8")
            n_days += 1
            if n_days % 20 == 0:
                print(f"  {d}: {len(rows):4d} announcements  ({time.time()-t0:.0f}s, {n_days} days fetched)")
        d += timedelta(days=1)

    frames = [json.loads(f.read_text(encoding="utf-8")) for f in sorted(CACHE.glob("*.json"))]
    out = pd.DataFrame([r for rows in frames for r in rows]).drop_duplicates("accession")
    OUT.parent.mkdir(exist_ok=True)
    out.to_csv(OUT, index=False)
    print(f"\nwrote {len(out):,} earnings announcements, {out['symbol'].nunique():,} symbols, "
          f"{out['filing_date'].min()} .. {out['filing_date'].max()} -> {OUT}")


if __name__ == "__main__":
    main()
