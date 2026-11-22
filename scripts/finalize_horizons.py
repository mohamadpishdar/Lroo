"""
Merge the per-chain outputs of onchain_horizons.py and apply the last identity check.

Count check: a token cannot have many more transfers in its first 30 days than the dataset
records for its whole life. If transfers_d30 > 3 x recorded + 100, the confirmed address
belongs to a different token with the same symbol (for example the original token behind a
copycat rug pull) and the match is rejected. The check is skipped when the recorded count is
missing or equals 25, a value that the explorer page shows by default and that is not a real
count (see the Review list). The rule is the same for both classes.

Usage
    python scripts/finalize_horizons.py data/TM-RugPull_v2.xlsx results/onchain_horizons.csv \
        results/hz2_eth.csv results/hz2_bsc.csv results/hz2_bsc_s0.csv ...
"""
import sys

import numpy as np
import pandas as pd

ARTIFACT_COUNT = 25
FACTOR, SLACK = 3, 100


def main(data_path, out, *parts):
    x = pd.read_excel(data_path, sheet_name="data")
    x["key"] = x["Project Title"].astype(str) + "|" + x["Blockchain"].astype(str)
    rec = x.drop_duplicates("key").set_index("key")["the number of Transactions"]
    hz = pd.concat([pd.read_csv(p) for p in parts], ignore_index=True).drop_duplicates("key", keep="last")
    r = pd.to_numeric(hz["key"].map(rec), errors="coerce")
    known = r.notna() & (r > 0) & (r != ARTIFACT_COUNT)
    conf = hz["status"].isin(["ok", "start mismatch"])
    bad = conf & known & (hz["transfers_d30"] > FACTOR * r + SLACK)
    hz.loc[bad, "status"] = "rejected: count"
    feat = [c for c in hz.columns if c.split("_d")[0] in ("transfers", "holders", "top10", "hhi", "gini")]
    hz.loc[bad, feat] = np.nan
    hz.to_csv(out, index=False)
    s = hz["status"].str.replace(r"^rejected: .*", "rejected", regex=True)
    print(pd.crosstab(s, hz["class"]).to_string())
    print(f"rejected by the count check: {int(bad.sum())} ({hz.loc[bad, 'class'].value_counts().to_dict()})")


if __name__ == "__main__":
    if len(sys.argv) < 4:
        sys.exit(__doc__)
    main(*sys.argv[1:])
