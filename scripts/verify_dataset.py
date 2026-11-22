"""
Integrity checks on TM-RugPull before training. Prints one line per check with PASS, WARN or
FAIL and the rows involved; exits with code 1 if any check FAILs.

Usage
    python scripts/verify_dataset.py data/TM-RugPull_v2.xlsx [results/onchain_horizons.csv]
"""
import re
import sys

import numpy as np
import pandas as pd

CHAIN_LAUNCH = {"ETH": "2015-07-30", "BSC": "2020-09-01", "POLYGON": "2020-05-30", "ARBI": "2021-08-31",
                "FANTOM": "2019-12-27", "BASE": "2023-07-13", "CRONO": "2021-11-08", "SNOW": "2020-09-21"}
NUMERIC = ["MaxPrice (Quarter 1)", "MaxPrice (Quarter 2)", "MaxPrice (Quarter 3)", "MaxPrice (Quarter 4)",
           "the number of Transactions", "Token concentration ratio per holder", "Total Variance",
           "Variance of holders with more than 1% tokens", "Token balance", "first deposits",
           "Google results for project title (first day)", "Google results for project title (project duration/2)",
           "Google results for project website (first day)", "Google results for project website (duration/2)",
           "Google results for project x profile (first days)", "Google results for project x profile (duration/2)"]
COLLECTION = (pd.Timestamp("2024-11-01"), pd.Timestamp("2025-01-31"))

results = []


def report(name, level, rows=()):
    rows = list(rows)
    results.append(level)
    tail = f"  rows: {', '.join(map(str, rows[:15]))}{' ...' if len(rows) > 15 else ''}" if rows else ""
    print(f"[{level}] {name} ({len(rows)}){tail}")


def main(path, hz_path=None):
    x = pd.read_excel(path, sheet_name="data")
    if "Candidate suitability" in x.columns:            # new candidates pending manual review
        cand = x["Candidate suitability"].notna()
        print(f"[INFO] {int(cand.sum())} candidate rows pending manual review are checked separately and skipped here")
        x = x[~cand]
    xl = lambda idx: [i + 2 for i in idx]          # Excel row numbers
    s, e = pd.to_datetime(x["project starting date"], errors="coerce"), pd.to_datetime(x["project end date"], errors="coerce")
    scam, legit = x["class"].eq("scam"), x["class"].eq("normal")

    report("class is scam or normal", "PASS" if (scam | legit).all() else "FAIL", xl(x.index[~(scam | legit)]))
    report("start date present and parseable", "PASS" if s.notna().all() else "FAIL", xl(x.index[s.isna()]))
    report("end date present and parseable", "PASS" if e.notna().all() else "FAIL", xl(x.index[e.isna()]))
    bad = x.index[(e < s).fillna(False)]
    report("end date not before start date", "PASS" if len(bad) == 0 else "FAIL", xl(bad))
    bad = x.index[legit & ~e.between(*COLLECTION)]
    report("legitimate end date = data collection window (Nov 2024 - Jan 2025)", "PASS" if len(bad) == 0 else "FAIL", xl(bad))
    bad = x.index[scam & (e > COLLECTION[1])]
    report("rug-pull collapse not after data collection", "PASS" if len(bad) == 0 else "FAIL", xl(bad))
    lc = pd.to_datetime(x.get("Label confirmation date"), errors="coerce")
    bad = x.index[scam & ((lc - e).dt.days != 3)]
    report("rug-pull label confirmation = collapse + 3 days", "PASS" if len(bad) == 0 else "WARN", xl(bad))
    launch = x["Blockchain"].map(lambda c: pd.Timestamp(CHAIN_LAUNCH.get(str(c).upper(), "1900-01-01")))
    bad = x.index[s < launch]
    review = set(pd.read_excel(path, sheet_name="Review list")["Excel row(s)"].astype(str))
    unrec = [r for r in xl(bad) if str(r) not in review]
    report("start date before chain launch (recorded in Review list)", "PASS" if not unrec else "WARN", xl(bad))
    if unrec:
        report("  ... of which NOT in Review list", "FAIL", unrec)

    for c in NUMERIC:
        v = pd.to_numeric(x[c].astype(str).str.replace(r"[,\s\xa0]", "", regex=True), errors="coerce")
        miss = x.index[v.isna()]
        neg = x.index[v < 0]
        if len(neg):
            report(f"{c}: non-negative", "FAIL", xl(neg))
        if len(miss):
            report(f"{c}: missing or non-numeric", "WARN" if len(miss) < 0.05 * len(x) else "FAIL", xl(miss))

    k = x["Project Title"].astype(str) + "|" + x["Blockchain"].astype(str) + "|" + s.astype(str)
    dup = x.index[k.duplicated(keep=False)]
    report("same title, chain and start date (recorded in Review list)", "WARN" if len(dup) else "PASS", xl(dup))
    full = x.drop(columns=[c for c in x.columns if c in ("Quality flags",)]).duplicated(keep=False)
    report("fully identical rows", "PASS" if not full.any() else "FAIL", xl(x.index[full]))

    a = x["Contract address"].dropna().astype(str).str.strip()
    bad = a.index[~a.str.fullmatch(r"0x[0-9a-fA-F]{40}")]
    report("contract address format", "PASS" if len(bad) == 0 else "FAIL", xl(bad))
    dupa = a[a.str.lower().duplicated(keep=False)]
    report("same address on two projects", "PASS" if dupa.empty else "WARN", xl(dupa.index))

    if hz_path:
        hz = pd.read_csv(hz_path).drop_duplicates("key")
        print("\nhorizon status by class:")
        print(pd.crosstab(hz["status"].str.replace(r":.*", "", regex=True), hz["class"]).to_string())
        okk = hz[hz["status"].isin(["ok", "start mismatch"])]
        dupa = okk[okk["address"].str.lower().duplicated(keep=False)]
        report("a confirmed address used by two projects", "PASS" if dupa.empty else "WARN", dupa["key"])
        for c in [f"transfers_d{h}" for h in (7, 14, 30)]:
            mono = okk[c].values
        ok_mono = ((okk["transfers_d7"] <= okk["transfers_d14"]) & (okk["transfers_d14"] <= okk["transfers_d30"])) | okk["capped"].astype(bool)
        report("horizon transfers increase with h", "PASS" if ok_mono.all() else "FAIL", okk.loc[~ok_mono, "key"])
        sh = okk[["top10_d30"]].dropna()
        bad = sh[~sh["top10_d30"].between(0, 1 + 1e-9)]
        report("top-10 share in [0,1]", "PASS" if bad.empty else "FAIL", okk.loc[bad.index, "key"])

    print(f"\n{results.count('PASS')} pass, {results.count('WARN')} warn, {results.count('FAIL')} fail")
    sys.exit(1 if "FAIL" in results else 0)


if __name__ == "__main__":
    if len(sys.argv) not in (2, 3):
        sys.exit(__doc__)
    main(*sys.argv[1:])
