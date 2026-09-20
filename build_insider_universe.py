"""Universe-wide insider BUY events from the SEC bulk Form 3/4/5 data sets.

Input : cache/insider_bulk/*_form345.zip   (fetch_insider_bulk.py)
Output: data/insider_events_universe.csv   one row per Form 4 filing that
        reports open-market purchases (code P, acquired) — every issuer the
        SEC has, not the 80 names the per-symbol walk could afford.

    python build_insider_universe.py --from-year 2020

Beyond the raw event, this attaches the two refinements the literature says
matter, both computable only from the reporting owner's OWN history — which is
exactly what the bulk data makes available and per-symbol scraping did not:

  routine / opportunistic  (Cohen, Malloy & Pomorski 2012). An insider who
      traded in the same calendar month in each of the three preceding years
      is ROUTINE — a scheduled liquidity or diversification pattern that
      carries no information. One with three years of history who breaks that
      pattern is OPPORTUNISTIC; those trades carry essentially all the alpha
      in the paper. Fewer than three years of history: UNCLASSIFIED.

  cluster  distinct insiders buying the same issuer within the preceding 30
      days of FILINGS (only filings already public count).

FILING DATE IS THE ONLY TRADEABLE DATE. Everything here is keyed on it.
"""

from __future__ import annotations

import argparse
import json
import sys
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

RAW_DIR = Path(__file__).parent / "cache" / "insider_bulk"
OUT = Path(__file__).parent / "data" / "insider_events_universe.csv"
OWNER_HISTORY = Path(__file__).parent / "cache" / "insider_owner_history.json"

CSUITE = ("chief executive", "chief financial", "ceo", "cfo", "president",
          "chief operating", "coo")


def load_quarter(zpath: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    z = zipfile.ZipFile(zpath)
    sub = pd.read_csv(z.open("SUBMISSION.tsv"), sep="\t", dtype=str,
                      usecols=["ACCESSION_NUMBER", "FILING_DATE", "DOCUMENT_TYPE",
                               "ISSUERCIK", "ISSUERTRADINGSYMBOL", "AFF10B5ONE"])
    own = pd.read_csv(z.open("REPORTINGOWNER.tsv"), sep="\t", dtype=str,
                      usecols=["ACCESSION_NUMBER", "RPTOWNERCIK", "RPTOWNERNAME",
                               "RPTOWNER_RELATIONSHIP", "RPTOWNER_TITLE"])
    trn = pd.read_csv(z.open("NONDERIV_TRANS.tsv"), sep="\t", dtype=str,
                      usecols=["ACCESSION_NUMBER", "TRANS_DATE", "TRANS_CODE",
                               "TRANS_SHARES", "TRANS_PRICEPERSHARE",
                               "TRANS_ACQUIRED_DISP_CD"])
    return sub, own, trn


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--from-year", type=int, default=2020,
                   help="first year of EVENTS to emit (history before it is "
                        "still read, for the routine classification)")
    args = p.parse_args()

    zips = sorted(RAW_DIR.glob("*_form345.zip"))
    if not zips:
        sys.exit(f"no bulk files in {RAW_DIR} — run fetch_insider_bulk.py")

    subs, owns, trns = [], [], []
    for zp in zips:
        s, o, t = load_quarter(zp)
        subs.append(s); owns.append(o); trns.append(t)
        print(f"  {zp.name}: {len(t):>7,} non-derivative rows")
    sub = pd.concat(subs, ignore_index=True)
    own = pd.concat(owns, ignore_index=True)
    trn = pd.concat(trns, ignore_index=True)

    # --- Form 4 only (amendments would double-count), with a real ticker -----
    sub = sub[sub["DOCUMENT_TYPE"] == "4"].copy()
    sub["filing_date"] = pd.to_datetime(sub["FILING_DATE"], format="%d-%b-%Y",
                                        errors="coerce")
    sub["symbol"] = sub["ISSUERTRADINGSYMBOL"].str.strip().str.upper()
    sub = sub[sub["filing_date"].notna() & sub["symbol"].notna()
              & (sub["symbol"] != "NONE") & (sub["symbol"] != "N/A")
              & sub["symbol"].str.fullmatch(r"[A-Z]{1,5}")]
    sub = sub.drop_duplicates("ACCESSION_NUMBER")
    sub["ISSUERCIK"] = sub["ISSUERCIK"].str.strip().str.lstrip("0")
    # Rule 10b5-1 plan trades are scheduled months in advance — by construction
    # they carry no information about what the insider knows today.
    sub["is_10b5_1"] = sub["AFF10B5ONE"].fillna("").str.lower().isin(["1", "true"]).astype(int)

    # --- open-market transactions, both directions (sells feed the routine
    #     classification; only buys become events) ----------------------------
    trn = trn[trn["TRANS_CODE"].isin(["P", "S"])].copy()
    trn["shares"] = pd.to_numeric(trn["TRANS_SHARES"], errors="coerce")
    trn["price"] = pd.to_numeric(trn["TRANS_PRICEPERSHARE"], errors="coerce")
    trn = trn[(trn["shares"] > 0) & (trn["price"] > 0)]
    trn["trans_date"] = pd.to_datetime(trn["TRANS_DATE"], format="%d-%b-%Y",
                                       errors="coerce")
    trn["usd"] = trn["shares"] * trn["price"]
    trn["is_buy"] = ((trn["TRANS_CODE"] == "P")
                     & (trn["TRANS_ACQUIRED_DISP_CD"] == "A")).astype(int)
    trn["is_sell"] = ((trn["TRANS_CODE"] == "S")
                      & (trn["TRANS_ACQUIRED_DISP_CD"] == "D")).astype(int)
    trn = trn[(trn["is_buy"] == 1) | (trn["is_sell"] == 1)]

    # One reporting owner per filing (joint filings are rare; keep the first).
    own = own.drop_duplicates("ACCESSION_NUMBER")
    # CIKs as plain integers-as-strings: the live feed reads them from XML
    # without the zero padding the bulk files carry, and the keys must agree.
    own["owner_cik"] = own["RPTOWNERCIK"].str.strip().str.lstrip("0")
    rel = own["RPTOWNER_RELATIONSHIP"].fillna("").str.lower()
    title = own["RPTOWNER_TITLE"].fillna("").str.lower()
    own["is_director"] = rel.str.contains("director").astype(int)
    own["is_officer"] = rel.str.contains("officer").astype(int)
    own["is_10pct"] = rel.str.contains("tenpercentowner").astype(int)
    own["is_csuite"] = (own["is_officer"].astype(bool)
                        & title.apply(lambda t: any(k in t for k in CSUITE))).astype(int)

    # --- per-filing aggregate -------------------------------------------------
    trn["buy_usd"] = trn["usd"] * trn["is_buy"]
    trn["sell_usd"] = trn["usd"] * trn["is_sell"]
    trn["buy_shares"] = trn["shares"] * trn["is_buy"]
    agg = (trn.groupby("ACCESSION_NUMBER")
              .agg(buy_usd=("buy_usd", "sum"), sell_usd=("sell_usd", "sum"),
                   buy_shares=("buy_shares", "sum"),
                   n_buy_rows=("is_buy", "sum"), n_sell_rows=("is_sell", "sum"),
                   trans_date=("trans_date", "min"))
              .reset_index())
    f = (agg.merge(sub[["ACCESSION_NUMBER", "filing_date", "symbol", "ISSUERCIK", "is_10b5_1"]],
                   on="ACCESSION_NUMBER")
            .merge(own[["ACCESSION_NUMBER", "owner_cik", "RPTOWNERNAME", "is_director",
                        "is_officer", "is_10pct", "is_csuite"]],
                   on="ACCESSION_NUMBER", how="left"))
    f = f[f["owner_cik"].notna()].sort_values(["symbol", "filing_date"]).reset_index(drop=True)
    f["is_buy_filing"] = (f["n_buy_rows"] > 0).astype(int)
    f["month"] = f["filing_date"].dt.month
    f["year"] = f["filing_date"].dt.year
    print(f"\n{len(f):,} open-market Form 4 filings, {f['symbol'].nunique():,} symbols, "
          f"{f['filing_date'].min():%Y-%m} .. {f['filing_date'].max():%Y-%m}")

    # --- routine / opportunistic per (owner, issuer) --------------------------
    # For each filing, look at this owner's filings in the same issuer during the
    # three preceding calendar years. Routine = a trade in the same calendar
    # month in EACH of those years. Needs a trade in each of the three years to
    # be classified at all.
    key = ["owner_cik", "ISSUERCIK"]
    hist = f[key + ["year", "month"]].drop_duplicates()
    hist_set = set(map(tuple, hist.to_numpy()))
    years_traded = hist.groupby(key)["year"].apply(set).to_dict()

    def classify(row) -> str:
        k = (row["owner_cik"], row["ISSUERCIK"])
        yrs = years_traded.get(k, set())
        prev = [row["year"] - i for i in (1, 2, 3)]
        if not all(y in yrs for y in prev):
            return "unclassified"
        if all((row["owner_cik"], row["ISSUERCIK"], y, row["month"]) in hist_set
               for y in prev):
            return "routine"
        return "opportunistic"

    f["trader_type"] = f.apply(classify, axis=1)

    # Export the owner histories so the LIVE feed can classify a filing the
    # moment it lands: {"ownerCIK|issuerCIK": ["YYYY-MM", ...]} of every month
    # in which that insider filed an open-market trade in that issuer.
    hist_out = (f.assign(ym=f["filing_date"].dt.strftime("%Y-%m"))
                  .groupby(key)["ym"].apply(lambda s: sorted(set(s))))
    OWNER_HISTORY.parent.mkdir(exist_ok=True)
    OWNER_HISTORY.write_text(json.dumps(
        {f"{o}|{i}": v for (o, i), v in hist_out.items()}), encoding="utf-8")
    print(f"  owner histories: {len(hist_out):,} (owner, issuer) pairs -> {OWNER_HISTORY}")

    # Owner's prior activity in this issuer (public before this filing).
    f["owner_prior_filings"] = f.groupby(key).cumcount()
    f["owner_prior_buys"] = (f.groupby(key)["is_buy_filing"].cumsum()
                             - f["is_buy_filing"])

    # --- buy events only, with cluster context --------------------------------
    ev = f[f["is_buy_filing"] == 1].copy()
    ev = ev[ev["filing_date"].dt.year >= args.from_year]
    ev = ev.sort_values(["symbol", "filing_date"]).reset_index(drop=True)

    n_ins, n_buys = np.zeros(len(ev), int), np.zeros(len(ev), int)
    for sym, g in ev.groupby("symbol", sort=False):
        d = g["filing_date"].to_numpy()
        owners = g["owner_cik"].to_numpy()
        idx = g.index.to_numpy()
        for i in range(len(g)):
            lo = d[i] - np.timedelta64(30, "D")
            m = (d <= d[i]) & (d > lo)
            n_ins[idx[i]] = len(set(owners[m]))
            n_buys[idx[i]] = int(m.sum())
    ev["n_insiders_30d"] = n_ins
    ev["n_buys_30d"] = n_buys

    out = ev[["symbol", "ISSUERCIK", "filing_date", "trans_date", "owner_cik",
              "RPTOWNERNAME", "buy_usd", "buy_shares", "is_director", "is_officer",
              "is_10pct", "is_csuite", "is_10b5_1", "trader_type", "owner_prior_filings",
              "owner_prior_buys", "n_insiders_30d", "n_buys_30d"]].rename(
        columns={"ISSUERCIK": "issuer_cik", "RPTOWNERNAME": "owner_name"})
    OUT.parent.mkdir(exist_ok=True)
    out.to_csv(OUT, index=False)

    print(f"\nwrote {len(out):,} buy events ({out['symbol'].nunique():,} symbols) -> {OUT}")
    print(f"  from {out['filing_date'].min():%Y-%m-%d} to {out['filing_date'].max():%Y-%m-%d}")
    print("\n  trader type:")
    for t, n in out["trader_type"].value_counts().items():
        print(f"    {t:14s} {n:7,}  ({100*n/len(out):.0f}%)")
    print(f"\n  clusters (3+ insiders / 30d): {int((out['n_insiders_30d'] >= 3).sum()):,}")
    print(f"  C-suite: {int(out['is_csuite'].sum()):,}   "
          f"director: {int(out['is_director'].sum()):,}   "
          f"10% owner: {int(out['is_10pct'].sum()):,}")
    print(f"  median buy: ${out['buy_usd'].median():,.0f}   "
          f"events >= $100k: {int((out['buy_usd'] >= 1e5).sum()):,}")


if __name__ == "__main__":
    main()
