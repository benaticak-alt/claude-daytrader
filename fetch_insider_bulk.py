"""Download the SEC's bulk Insider Transactions Data Sets (Forms 3/4/5).

Universe-wide insider history without a single per-filing request: the SEC
publishes every structured Form 3/4/5 filing as one quarterly ZIP of TSV
tables (SUBMISSION, REPORTINGOWNER, NONDERIV_TRANS, ...). Twenty quarters is a
few hundred megabytes and covers every listed issuer — versus the old
per-symbol EDGAR walk, which took an hour for 80 names.

    python fetch_insider_bulk.py --from 2017q1 --to 2026q2

Free, official, no key. SEC asks for a descriptive User-Agent and <10 req/s;
this makes one request per quarter with a pause between.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import requests

import config

RAW_DIR = Path(__file__).parent / "cache" / "insider_bulk"
# The SEC moved the files between two paths in 2026; try both.
URLS = (
    "https://www.sec.gov/files/structureddata/data/insider-transactions-data-sets/{q}_form345.zip",
    "https://www.sec.gov/files/datastandardsinnovation/data/insider-transactions-data-sets/{q}_form345.zip",
)


def quarters(start: str, end: str) -> list[str]:
    y0, q0 = int(start[:4]), int(start[5])
    y1, q1 = int(end[:4]), int(end[5])
    out = []
    y, q = y0, q0
    while (y, q) <= (y1, q1):
        out.append(f"{y}q{q}")
        q += 1
        if q == 5:
            y, q = y + 1, 1
    return out


def fetch(q: str, session: requests.Session) -> Path:
    dest = RAW_DIR / f"{q}_form345.zip"
    if dest.exists() and dest.stat().st_size > 100_000:
        return dest
    for tpl in URLS:
        r = session.get(tpl.format(q=q), timeout=120)
        if r.status_code == 200 and r.content[:2] == b"PK":
            dest.write_bytes(r.content)
            return dest
    raise RuntimeError(f"{q}: not found at either SEC path")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--from", dest="start", default="2017q1")
    p.add_argument("--to", dest="end", default="2026q2")
    args = p.parse_args()

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    s = requests.Session()
    s.headers["User-Agent"] = config.SEC_USER_AGENT

    qs = quarters(args.start, args.end)
    print(f"fetching {len(qs)} quarters -> {RAW_DIR}")
    for q in qs:
        t0 = time.time()
        try:
            path = fetch(q, s)
        except Exception as e:                       # noqa: BLE001
            print(f"  {q}: FAILED {e}")
            continue
        print(f"  {q}: {path.stat().st_size/1e6:5.1f} MB  ({time.time()-t0:.1f}s)")
        time.sleep(0.5)


if __name__ == "__main__":
    sys.exit(main())
