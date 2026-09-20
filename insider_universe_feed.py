"""Universe-wide live Form 4 feed — every open-market insider BUY on EDGAR.

The per-symbol feed (insider_feed.py) asks "what did insiders do at these 25
names?". This asks the market-wide question the research was done on: "which
of ~6,000 issuers had an insider buy disclosed in the last few days?"

Discovery uses EDGAR full-text search (efts.sec.gov), which indexes filings
within minutes and returns, per hit, the accession number AND the primary
document name — so each Form 4 costs exactly one further request for its XML.
Volume is ~600 Form 4s per trading day; at the SEC's 10 req/s ceiling that is
a couple of minutes on the first pass and seconds per cycle after.

Each filing becomes at most one event: the sum of its code-P acquisitions,
with the reporting owner's role and CIK. The owner CIK matters: joined to the
bulk history (cache/insider_owner_history.json, from build_insider_universe.py)
it classifies the trader as routine / opportunistic / unclassified per
Cohen-Malloy-Pomorski — the filter that separated +$52/trade from +$0.19.

Free, official, no key. Nothing here places orders.
"""

from __future__ import annotations

import json
import logging
import time
import xml.etree.ElementTree as ET
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional

import requests

import config

log = logging.getLogger(__name__)

EFTS_URL = "https://efts.sec.gov/LATEST/search-index"
ARCHIVE_URL = "https://www.sec.gov/Archives/edgar/data/{cik}/{acc}/{doc}"
CACHE_DIR = Path(__file__).parent / "cache"
STATE_PATH = CACHE_DIR / "insider_universe_state.json"
OWNER_HISTORY_PATH = CACHE_DIR / "insider_owner_history.json"

_HEADERS = {"User-Agent": config.SEC_USER_AGENT, "Accept-Encoding": "gzip, deflate"}
_MIN_INTERVAL = 0.12                      # ~8 req/s, under the SEC's 10
CSUITE = ("chief executive", "chief financial", "ceo", "cfo", "president",
          "chief operating", "coo")
KEEP_DAYS = 45                            # events older than this are pruned


def _txt(node: Optional[ET.Element], path: str) -> str:
    if node is None:
        return ""
    found = node.find(path)
    return (found.text or "").strip() if found is not None else ""


def _flag(node: Optional[ET.Element], path: str) -> bool:
    return _txt(node, path).lower() in ("1", "true")


class InsiderUniverseFeed:
    def __init__(self, max_age_days: int = 3, cache_ttl_minutes: int = 30) -> None:
        self.max_age_days = max_age_days
        self.cache_ttl = timedelta(minutes=cache_ttl_minutes)
        self._last_request = 0.0
        self._last_refresh: Optional[datetime] = None
        self._seen: Dict[str, str] = {}         # accession -> filing date, already parsed
        self._events: Dict[str, dict] = {}      # accession -> event
        self._owner_hist: Dict[str, List[str]] = {}
        self._load()

    # ------------------------------------------------------------------ io
    def _load(self) -> None:
        try:
            if STATE_PATH.exists():
                st = json.loads(STATE_PATH.read_text(encoding="utf-8"))
                self._seen = st.get("seen", {})
                if isinstance(self._seen, list):     # legacy shape
                    self._seen = {a: "1970-01-01" for a in self._seen}
                self._events = st.get("events", {})
                self._prune()
                log.info("insider universe: %d filings seen, %d buy events cached",
                         len(self._seen), len(self._events))
        except Exception:
            log.exception("could not load insider universe state — starting fresh")
        try:
            if OWNER_HISTORY_PATH.exists():
                self._owner_hist = json.loads(OWNER_HISTORY_PATH.read_text(encoding="utf-8"))
                log.info("insider universe: %d owner histories", len(self._owner_hist))
            else:
                log.warning("no owner history at %s — every trader will be 'unclassified' "
                            "(run build_insider_universe.py)", OWNER_HISTORY_PATH)
        except Exception:
            log.exception("could not load owner history")

    def _save(self) -> None:
        try:
            CACHE_DIR.mkdir(exist_ok=True)
            STATE_PATH.write_text(json.dumps({
                "seen": self._seen, "events": self._events,
            }), encoding="utf-8")
        except Exception:
            log.exception("could not save insider universe state")

    def _prune(self) -> None:
        cutoff = (date.today() - timedelta(days=KEEP_DAYS)).isoformat()
        self._events = {k: v for k, v in self._events.items() if v["filing_date"] >= cutoff}
        # Keep the seen-set bounded too: anything older than the search window
        # can never come back from EFTS.
        recent_cut = (date.today() - timedelta(days=self.max_age_days + 7)).isoformat()
        self._seen = {a: d for a, d in self._seen.items() if d >= recent_cut}

    # ------------------------------------------------------------------ http
    def _get(self, url: str, **kw) -> Optional[requests.Response]:
        wait = _MIN_INTERVAL - (time.time() - self._last_request)
        if wait > 0:
            time.sleep(wait)
        try:
            r = requests.get(url, headers=_HEADERS, timeout=30, **kw)
            self._last_request = time.time()
            if r.status_code == 429:
                log.warning("SEC rate limit hit — backing off 10s")
                time.sleep(10)
                return None
            r.raise_for_status()
            return r
        except requests.RequestException as e:
            log.warning("SEC request failed: %s", e)
            return None

    # ------------------------------------------------------------------ discovery
    def _search(self, start: date, end: date) -> List[dict]:
        """All Form 4 (not 4/A) hits filed in [start, end]."""
        hits, frm = [], 0
        while True:
            r = self._get(EFTS_URL, params={
                "q": '""', "forms": "4", "dateRange": "custom",
                "startdt": start.isoformat(), "enddt": end.isoformat(), "from": frm,
            })
            if r is None:
                break
            page = r.json().get("hits", {}).get("hits", [])
            if not page:
                break
            for h in page:
                src = h.get("_source", {})
                if src.get("form") != "4":
                    continue           # amendments restate, they are not new events
                acc, _, doc = h["_id"].partition(":")
                hits.append({"acc": acc, "doc": doc, "ciks": src.get("ciks", []),
                             "file_date": src.get("file_date")})
            frm += len(page)
            if frm >= 10_000 or len(page) < 100:
                break
        return hits

    # ------------------------------------------------------------------ parsing
    def _fetch_xml(self, hit: dict) -> Optional[bytes]:
        acc_nodash = hit["acc"].replace("-", "")
        for cik in hit["ciks"]:
            r = self._get(ARCHIVE_URL.format(cik=int(cik), acc=acc_nodash, doc=hit["doc"]))
            if r is not None and r.content[:5] in (b"<?xml", b"<owne"):
                return r.content
        return None

    def _parse(self, xml_bytes: bytes, hit: dict) -> Optional[dict]:
        try:
            root = ET.fromstring(xml_bytes)
        except ET.ParseError:
            return None
        symbol = _txt(root, "issuer/issuerTradingSymbol").upper().strip()
        issuer_cik = _txt(root, "issuer/issuerCik").lstrip("0")
        if not symbol or symbol in ("NONE", "N/A") or not symbol.isalpha() or len(symbol) > 5:
            return None
        owner = root.find("reportingOwner")
        owner_cik = _txt(owner, "reportingOwnerId/rptOwnerCik").lstrip("0")
        rel = owner.find("reportingOwnerRelationship") if owner is not None else None
        title = _txt(rel, "officerTitle")
        is_officer = _flag(rel, "isOfficer")

        buy_usd = buy_shares = 0.0
        sell_usd = 0.0
        trans_dates = []
        for t in root.findall("nonDerivativeTable/nonDerivativeTransaction"):
            code = _txt(t, "transactionCoding/transactionCode")
            if code not in ("P", "S"):
                continue
            try:
                shares = float(_txt(t, "transactionAmounts/transactionShares/value") or 0)
                price = float(_txt(t, "transactionAmounts/transactionPricePerShare/value") or 0)
            except ValueError:
                continue
            if shares <= 0 or price <= 0:
                continue
            ad = _txt(t, "transactionAmounts/transactionAcquiredDisposedCode/value")
            if code == "P" and ad == "A":
                buy_usd += shares * price
                buy_shares += shares
                trans_dates.append(_txt(t, "transactionDate/value"))
            elif code == "S" and ad == "D":
                sell_usd += shares * price

        # Record the owner's activity month either way — routine detection
        # needs sells as well as buys — but only buys become events.
        filing_date = hit["file_date"] or date.today().isoformat()
        if owner_cik and issuer_cik and (buy_usd > 0 or sell_usd > 0):
            k = f"{owner_cik}|{issuer_cik}"
            ym = filing_date[:7]
            months = self._owner_hist.setdefault(k, [])
            if ym not in months:
                months.append(ym)
        if buy_usd <= 0:
            return None

        return {
            "accession": hit["acc"], "symbol": symbol, "issuer_cik": issuer_cik,
            "owner_cik": owner_cik, "owner_name": _txt(owner, "reportingOwnerId/rptOwnerName"),
            "filing_date": filing_date, "trans_date": min(trans_dates) if trans_dates else None,
            "buy_usd": round(buy_usd, 2), "buy_shares": buy_shares,
            "price_paid": round(buy_usd / buy_shares, 4) if buy_shares else None,
            "is_officer": int(is_officer), "is_director": int(_flag(rel, "isDirector")),
            "is_10pct": int(_flag(rel, "isTenPercentOwner")),
            "is_csuite": int(is_officer and any(k in title.lower() for k in CSUITE)),
            "is_10b5_1": int(_flag(root, "aff10b5One")),
            "title": title,
            "trader_type": self.classify_trader(owner_cik, issuer_cik, filing_date),
        }

    # ------------------------------------------------------------------ classification
    def classify_trader(self, owner_cik: str, issuer_cik: str, filing_date: str) -> str:
        """Cohen-Malloy-Pomorski: routine if the owner filed a trade in this
        issuer in the same calendar month in each of the three preceding years;
        opportunistic if they have three years of history but no such pattern;
        unclassified otherwise."""
        months = set(self._owner_hist.get(f"{owner_cik}|{issuer_cik}", []))
        if not months:
            return "unclassified"
        y, m = int(filing_date[:4]), filing_date[5:7]
        prev_years = [str(y - i) for i in (1, 2, 3)]
        if not all(any(ym.startswith(py) for ym in months) for py in prev_years):
            return "unclassified"
        if all(f"{py}-{m}" in months for py in prev_years):
            return "routine"
        return "opportunistic"

    # ------------------------------------------------------------------ public
    def refresh(self, force: bool = False, max_parse: int = 200) -> int:
        """Pull Form 4s filed in the last `max_age_days` that have not been
        parsed yet, newest first, at most `max_parse` per call so a cycle never
        stalls on a backlog (the first pass after a weekend is ~1,700 filings,
        ~15 minutes at the SEC rate limit). If a backlog remains, the cache TTL
        is NOT started, so the next cycle continues immediately.
        Returns the number of NEW buy events."""
        now = datetime.now()
        if not force and self._last_refresh and now - self._last_refresh < self.cache_ttl:
            return 0
        start = date.today() - timedelta(days=self.max_age_days)
        hits = self._search(start, date.today())
        new = [h for h in hits if h["acc"] not in self._seen]
        new.sort(key=lambda h: h["file_date"] or "", reverse=True)
        backlog = max(0, len(new) - max_parse)
        new = new[:max_parse]
        log.info("insider universe: %d Form 4s in window, parsing %d new (%d deferred)",
                 len(hits), len(new), backlog)
        added = 0
        for h in new:
            xml = self._fetch_xml(h)
            self._seen[h["acc"]] = h["file_date"] or date.today().isoformat()
            if xml is None:
                continue
            ev = self._parse(xml, h)
            if ev:
                self._events[h["acc"]] = ev
                added += 1
        if not backlog:
            self._last_refresh = now
        if new:
            self._prune()
            self._save()
        if added:
            log.info("insider universe: %d new buy events", added)
        return added

    def fresh_events(self) -> List[dict]:
        """Buy events filed within max_age_days, newest first."""
        cutoff = (date.today() - timedelta(days=self.max_age_days)).isoformat()
        evs = [e for e in self._events.values() if e["filing_date"] >= cutoff]
        return sorted(evs, key=lambda e: (e["filing_date"], e["buy_usd"]), reverse=True)

    def candidate_symbols(self, min_usd: float, roles: str = "officer_director") -> List[str]:
        """Symbols worth a market-data snapshot this cycle: fresh buys that pass
        the static (pre-price) part of the filter. Price-dependent checks —
        liquidity, discount, trend — happen in the decider once the row exists."""
        out = []
        for e in self.fresh_events():
            if e["buy_usd"] < min_usd or e["trader_type"] == "routine":
                continue
            if roles == "officer_director" and not (e["is_officer"] or e["is_director"]):
                continue
            if e["symbol"] not in out:
                out.append(e["symbol"])
        return out
