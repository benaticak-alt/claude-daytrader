"""Pull the full historical Form 4 archive for the daily universe.

EDGAR is an ARCHIVE, not a live feed — every Form 4 ever filed is retrievable
today. An earlier plan to "accumulate insider events over months" was simply
wrong; years of history can be built in one pass.

    python fetch_insider_history.py --years 6

Output: data/insider_history.csv — one row per open-market transaction
(codes P and S only; grants, option exercises, tax withholding and gifts carry
no directional information and are discarded).

THE FILING DATE IS THE ONLY DATE YOU MAY TRADE ON. Form 4 allows two business
days between the transaction and its disclosure, so a strategy keyed to
`transaction_date` would be trading on information that was not public yet —
a lookahead leak that would flatter results and vanish in production. Both
dates are recorded; downstream code must join on `filing_date`.

Resumable: each symbol writes its own cache file, so an interrupted run picks
up where it stopped.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import config
from build_daily_dataset import UNIVERSE
from insider_feed import InsiderFeed

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger("insider-history")

RAW_DIR = Path(__file__).parent / "cache" / "insider_history"
OUT = Path(__file__).parent / "data" / "insider_history.csv"


class HistoricalInsiderFetcher(InsiderFeed):
    """Reuses the live feed's polite HTTP, CIK map, and Form 4 XML parser,
    but walks the ENTIRE filing list rather than a recent window."""

    def all_filings(self, symbol: str, years: float) -> list[dict]:
        cik = self._load_cik_map().get(symbol.upper())
        if cik is None:
            log.warning("%s: not in SEC CIK map", symbol)
            return []

        resp = self._get(
            "https://data.sec.gov/submissions/CIK{:010d}.json".format(cik)
        )
        if resp is None:
            return []
        data = resp.json()

        # `recent` holds ~1000 filings; older ones live in referenced shards.
        blocks = [data.get("filings", {}).get("recent", {})]
        for extra in data.get("filings", {}).get("files", []):
            r = self._get("https://data.sec.gov/submissions/" + extra["name"])
            if r is not None:
                blocks.append(r.json())

        cutoff = (datetime.now(timezone.utc) - timedelta(days=int(365 * years))).date()
        out = []
        for blk in blocks:
            forms = blk.get("form", [])
            for i, form in enumerate(forms):
                if form != "4":
                    continue
                try:
                    filed = datetime.fromisoformat(blk["filingDate"][i]).date()
                except (ValueError, IndexError, KeyError):
                    continue
                if filed < cutoff:
                    continue
                out.append({
                    "filing_date": blk["filingDate"][i],
                    "accession": blk["accessionNumber"][i],
                    "doc": blk["primaryDocument"][i],
                    "cik": cik,
                })
        return out

    def transactions(self, symbol: str, years: float) -> list[dict]:
        rows = []
        filings = self.all_filings(symbol, years)
        log.info("%s: %d Form 4 filings in window", symbol, len(filings))
        for f in filings:
            doc = f["doc"].split("/")[-1]  # strip the XSL-rendered HTML prefix
            url = "https://www.sec.gov/Archives/edgar/data/{}/{}/{}".format(
                f["cik"], f["accession"].replace("-", ""), doc
            )
            resp = self._get(url)
            if resp is None:
                continue
            for txn in self._parse_form4(resp.content):
                rows.append({
                    "symbol": symbol,
                    "filing_date": f["filing_date"],      # tradeable date
                    "transaction_date": txn.date,          # NOT tradeable
                    "code": txn.code,
                    "acquired": int(txn.acquired),
                    "shares": txn.shares,
                    "price": txn.price,
                    "usd": round(txn.shares * txn.price, 2),
                    "owner": txn.owner,
                    "title": txn.title,
                })
        return rows


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--years", type=float, default=6.0)
    p.add_argument("--symbols", default=None, help="default: the daily universe")
    args = p.parse_args()

    symbols = ([s.strip().upper() for s in args.symbols.split(",")]
               if args.symbols else UNIVERSE)
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    OUT.parent.mkdir(parents=True, exist_ok=True)

    fetcher = HistoricalInsiderFetcher()
    all_rows: list[dict] = []
    started = time.monotonic()

    for i, sym in enumerate(symbols, 1):
        cache = RAW_DIR / f"{sym}.json"
        if cache.exists():                      # resumable across runs
            rows = json.loads(cache.read_text(encoding="utf-8"))
            log.info("[%d/%d] %s: %d rows (cached)", i, len(symbols), sym, len(rows))
        else:
            try:
                rows = fetcher.transactions(sym, args.years)
            except Exception:
                log.exception("%s failed — skipping", sym)
                rows = []
            cache.write_text(json.dumps(rows), encoding="utf-8")
            log.info("[%d/%d] %s: %d transactions (%.0fs elapsed)",
                     i, len(symbols), sym, len(rows), time.monotonic() - started)
        all_rows.extend(rows)

    if not all_rows:
        sys.exit("no transactions fetched")

    cols = ["symbol", "filing_date", "transaction_date", "code", "acquired",
            "shares", "price", "usd", "owner", "title"]
    with open(OUT, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(all_rows)

    buys = [r for r in all_rows if r["acquired"] and r["code"] == "P"]
    sells = [r for r in all_rows if not r["acquired"] and r["code"] == "S"]
    print(f"\nwrote {len(all_rows):,} transactions -> {OUT}")
    print(f"  symbols with data : {len({r['symbol'] for r in all_rows})}")
    print(f"  open-market BUYS  : {len(buys):,}   (the informative side)")
    print(f"  open-market SELLS : {len(sells):,}")
    print(f"  date range        : {min(r['filing_date'] for r in all_rows)} .. "
          f"{max(r['filing_date'] for r in all_rows)}")
    print("\n  Buys are what matters: insiders sell for diversification, taxes and")
    print("  scheduled plans, but they buy for one reason. If the buy count is")
    print("  tiny, this signal cannot support a testable strategy on this universe.")


if __name__ == "__main__":
    main()
