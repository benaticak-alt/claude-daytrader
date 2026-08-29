"""SEC Form 4 insider-transaction feed.

Public disclosure data only: Form 4 is the filing that corporate officers,
directors, and 10%+ owners are legally required to submit within two business
days of trading their own company's stock. Free, official, no API key.

THE ONLY THING THAT MATTERS HERE IS THE TRANSACTION CODE.
Most Form 4 rows carry no directional information at all:

    P  open-market purchase   <-- the real signal
    S  open-market sale       <-- weak signal (insiders sell for many reasons)
    A  grant / award              noise
    M  option exercise           noise
    F  shares withheld for tax   noise
    G  bona fide gift            noise  (a 500k-share "disposal" that means nothing)
    C  conversion                noise
    X  derivative exercise       noise

A naive "insider sold 500,000 shares" headline is usually code G, M, or F.
We keep P and S and throw the rest away.

TIMESCALE CAVEAT: the documented edge in insider buying plays out over weeks
to months, not 5-minute bars. This feed is a directional *bias* the model can
weigh, not an intraday entry trigger. The one genuinely intraday element is
`filed_today` -- a filing hitting EDGAR during the session is a real catalyst.
"""

from __future__ import annotations

import json
import logging
import time
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, List, Optional

import requests

import config

log = logging.getLogger(__name__)

SEC_TICKERS_URL = "https://www.sec.gov/files/company_tickers.json"
SEC_SUBMISSIONS_URL = "https://data.sec.gov/submissions/CIK{cik:010d}.json"
SEC_ARCHIVE_URL = "https://www.sec.gov/Archives/edgar/data/{cik}/{acc}/{doc}"

# SEC requires a descriptive User-Agent with contact info, and rate-limits to
# 10 req/sec. We stay well under that.
_HEADERS = {"User-Agent": config.SEC_USER_AGENT}
_MIN_REQUEST_INTERVAL = 0.15  # ~6.6 req/sec ceiling

MEANINGFUL_CODES = {"P", "S"}

CACHE_DIR = Path(__file__).parent / "cache"


@dataclass
class InsiderTxn:
    date: str
    code: str
    shares: float
    price: float
    acquired: bool          # True = acquired (buy), False = disposed (sell)
    owner: str
    title: str

    @property
    def usd(self) -> float:
        return self.shares * self.price


@dataclass
class InsiderSummary:
    symbol: str
    lookback_days: int
    buys: List[InsiderTxn] = field(default_factory=list)
    sells: List[InsiderTxn] = field(default_factory=list)
    filed_today: int = 0
    error: Optional[str] = None

    def to_context(self) -> dict:
        """Compact form for the model. Aggregates, not raw rows."""
        if self.error:
            return {"symbol": self.symbol, "insider_data": "unavailable", "error": self.error}

        buy_insiders = {t.owner for t in self.buys}
        sell_insiders = {t.owner for t in self.sells}
        exec_buy = any(
            kw in t.title.lower()
            for t in self.buys
            for kw in ("chief executive", "chief financial", "ceo", "cfo", "president")
        )
        return {
            "lookback_days": self.lookback_days,
            "open_market_buys": {
                "transactions": len(self.buys),
                "distinct_insiders": len(buy_insiders),
                "total_usd": round(sum(t.usd for t in self.buys)),
                "most_recent": max((t.date for t in self.buys), default=None),
            },
            "open_market_sells": {
                "transactions": len(self.sells),
                "distinct_insiders": len(sell_insiders),
                "total_usd": round(sum(t.usd for t in self.sells)),
                "most_recent": max((t.date for t in self.sells), default=None),
            },
            # Cluster buying (3+ distinct insiders buying) and C-suite buying are
            # the two variants with the strongest documented forward returns.
            "cluster_buy": len(buy_insiders) >= 3,
            "c_suite_buy": exec_buy,
            "filings_today": self.filed_today,
        }


class InsiderFeed:
    def __init__(self, lookback_days: int = 90, cache_ttl_minutes: int = 60) -> None:
        self.lookback_days = lookback_days
        self.cache_ttl = timedelta(minutes=cache_ttl_minutes)
        self._session = requests.Session()
        self._session.headers.update(_HEADERS)
        self._last_request = 0.0
        self._cik_map: Optional[Dict[str, int]] = None
        CACHE_DIR.mkdir(parents=True, exist_ok=True)

    # ---- polite HTTP ------------------------------------------------------

    def _get(self, url: str) -> Optional[requests.Response]:
        elapsed = time.monotonic() - self._last_request
        if elapsed < _MIN_REQUEST_INTERVAL:
            time.sleep(_MIN_REQUEST_INTERVAL - elapsed)
        self._last_request = time.monotonic()
        try:
            resp = self._session.get(url, timeout=20)
            if resp.status_code != 200:
                log.warning("SEC %s -> HTTP %s", url, resp.status_code)
                return None
            return resp
        except requests.RequestException:
            log.exception("SEC request failed: %s", url)
            return None

    # ---- ticker -> CIK ----------------------------------------------------

    def _load_cik_map(self) -> Dict[str, int]:
        if self._cik_map is not None:
            return self._cik_map

        cache_file = CACHE_DIR / "cik_map.json"
        if cache_file.exists():
            age = datetime.now(timezone.utc) - datetime.fromtimestamp(
                cache_file.stat().st_mtime, timezone.utc
            )
            if age < timedelta(days=7):
                self._cik_map = json.loads(cache_file.read_text(encoding="utf-8"))
                return self._cik_map

        resp = self._get(SEC_TICKERS_URL)
        if resp is None:
            self._cik_map = {}
            return self._cik_map

        raw = resp.json()
        self._cik_map = {
            entry["ticker"].upper(): int(entry["cik_str"]) for entry in raw.values()
        }
        cache_file.write_text(json.dumps(self._cik_map), encoding="utf-8")
        log.info("loaded CIK map (%d tickers)", len(self._cik_map))
        return self._cik_map

    # ---- Form 4 parsing ---------------------------------------------------

    @staticmethod
    def _text(node: Optional[ET.Element], path: str) -> str:
        if node is None:
            return ""
        found = node.find(path)
        return (found.text or "").strip() if found is not None else ""

    def _parse_form4(self, xml_bytes: bytes) -> List[InsiderTxn]:
        try:
            root = ET.fromstring(xml_bytes)
        except ET.ParseError:
            return []

        owner_node = root.find("reportingOwner")
        owner = self._text(owner_node, "reportingOwnerId/rptOwnerName") or "unknown"
        rel = owner_node.find("reportingOwnerRelationship") if owner_node is not None else None
        title = self._text(rel, "officerTitle")
        if not title and rel is not None:
            if self._text(rel, "isDirector") in ("1", "true"):
                title = "Director"
            elif self._text(rel, "isTenPercentOwner") in ("1", "true"):
                title = "10% Owner"

        txns: List[InsiderTxn] = []
        table = root.find("nonDerivativeTable")
        if table is None:
            return txns

        for t in table.findall("nonDerivativeTransaction"):
            code = self._text(t, "transactionCoding/transactionCode")
            if code not in MEANINGFUL_CODES:
                continue  # gifts, grants, tax withholding, option exercises -> discard
            try:
                shares = float(self._text(t, "transactionAmounts/transactionShares/value") or 0)
                price = float(
                    self._text(t, "transactionAmounts/transactionPricePerShare/value") or 0
                )
            except ValueError:
                continue
            if shares <= 0 or price <= 0:
                continue  # a P/S row with no price is unusable
            acquired = (
                self._text(t, "transactionAmounts/transactionAcquiredDisposedCode/value") == "A"
            )
            txns.append(
                InsiderTxn(
                    date=self._text(t, "transactionDate/value"),
                    code=code,
                    shares=shares,
                    price=price,
                    acquired=acquired,
                    owner=owner,
                    title=title,
                )
            )
        return txns

    # ---- public API -------------------------------------------------------

    def summary(self, symbol: str) -> InsiderSummary:
        symbol = symbol.upper()
        cache_file = CACHE_DIR / f"insider_{symbol}.json"

        if cache_file.exists():
            age = datetime.now(timezone.utc) - datetime.fromtimestamp(
                cache_file.stat().st_mtime, timezone.utc
            )
            if age < self.cache_ttl:
                cached = json.loads(cache_file.read_text(encoding="utf-8"))
                return self._from_cache(symbol, cached)

        result = self._fetch(symbol)
        cache_file.write_text(
            json.dumps(
                {
                    "lookback_days": result.lookback_days,
                    "buys": [t.__dict__ for t in result.buys],
                    "sells": [t.__dict__ for t in result.sells],
                    "filed_today": result.filed_today,
                    "error": result.error,
                }
            ),
            encoding="utf-8",
        )
        return result

    def _from_cache(self, symbol: str, cached: dict) -> InsiderSummary:
        return InsiderSummary(
            symbol=symbol,
            lookback_days=cached["lookback_days"],
            buys=[InsiderTxn(**t) for t in cached["buys"]],
            sells=[InsiderTxn(**t) for t in cached["sells"]],
            filed_today=cached["filed_today"],
            error=cached.get("error"),
        )

    def _fetch(self, symbol: str) -> InsiderSummary:
        cik = self._load_cik_map().get(symbol)
        if cik is None:
            return InsiderSummary(symbol, self.lookback_days, error="ticker not in SEC CIK map")

        resp = self._get(SEC_SUBMISSIONS_URL.format(cik=cik))
        if resp is None:
            return InsiderSummary(symbol, self.lookback_days, error="submissions fetch failed")

        recent = resp.json().get("filings", {}).get("recent", {})
        forms = recent.get("form", [])
        dates = recent.get("filingDate", [])
        accs = recent.get("accessionNumber", [])
        docs = recent.get("primaryDocument", [])

        cutoff = (datetime.now(timezone.utc) - timedelta(days=self.lookback_days)).date()
        today = datetime.now(timezone.utc).date().isoformat()

        result = InsiderSummary(symbol, self.lookback_days)
        fetched = 0
        for i, form in enumerate(forms):
            if form != "4":
                continue
            try:
                filed = datetime.fromisoformat(dates[i]).date()
            except (ValueError, IndexError):
                continue
            if filed < cutoff:
                break  # submissions are newest-first; everything after is older
            if dates[i] == today:
                result.filed_today += 1

            # Cap work per symbol -- a mega-cap can file dozens of Form 4s a month.
            if fetched >= config.INSIDER_MAX_FILINGS_PER_SYMBOL:
                continue
            fetched += 1

            # primaryDocument points at the XSL-rendered HTML; strip that
            # prefix to get the raw XML sibling.
            doc = docs[i].split("/")[-1]
            url = SEC_ARCHIVE_URL.format(cik=cik, acc=accs[i].replace("-", ""), doc=doc)
            doc_resp = self._get(url)
            if doc_resp is None:
                continue
            for txn in self._parse_form4(doc_resp.content):
                (result.buys if txn.acquired else result.sells).append(txn)

        log.info(
            "%s insider: %d buys / %d sells over %dd (%d filed today)",
            symbol, len(result.buys), len(result.sells), self.lookback_days, result.filed_today,
        )
        return result


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    feed = InsiderFeed()
    for sym in config.WATCHLIST:
        print(json.dumps({sym: feed.summary(sym).to_context()}, indent=2))
