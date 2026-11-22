"""
On-chain check of the labels and the recorded end dates.

For every project whose contract was confirmed by scripts/onchain_horizons.py, the script
counts token transfers in two windows around the recorded end date t_end:
    before = [t_end - 30 d, t_end)        after = [t_end + 3 d, t_end + 33 d)
For a rug pull, activity should have dropped sharply after the collapse (the labeling rule
requires no meaningful activity for 72 hours). For a legitimate project, t_end is the data
collection date and the token should still be active in the month before it.

Flags (written for manual review; nothing in the data is changed automatically)
    rug pull, after/before > 0.5 and after >= 50   activity did not collapse at t_end
    rug pull, before == 0                            already inactive a month before t_end
    legitimate, before == 0                          no activity before the collection date

Usage
    export ETHERSCAN_API_KEY=... NODEREAL_URL=...
    python scripts/audit_collapse.py data/TM-RugPull_v2.xlsx results/onchain_horizons.csv results/collapse_audit.csv
"""
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
import onchain_horizons as H  # noqa: E402

CAP = 20_000


def escan_count(chain_id, token, t_a, t_b):
    b_a = int(H.escan(chain_id, module="block", action="getblocknobytime", timestamp=t_a, closest="after"))
    b_b = int(H.escan(chain_id, module="block", action="getblocknobytime", timestamp=t_b, closest="before"))
    n, start = 0, b_a
    while start <= b_b and n < CAP:
        res = H.escan(chain_id, module="account", action="tokentx", contractaddress=token,
                      startblock=start, endblock=b_b, page=1, offset=H.PAGE, sort="asc")
        if not res:
            break
        if len(res) < H.PAGE:
            n += len(res)
            break
        last = int(res[-1]["blockNumber"])
        n += sum(int(r["blockNumber"]) < last for r in res) if last > start else len(res)
        start = last if last > start else last + 1
    return n


def bsc_count(token, t_a, t_b):
    b_a, b_b = H.bsc_block_at(t_a), H.bsc_block_at(t_b)
    n, start = 0, b_a
    while start <= b_b and n < CAP:
        end, key = min(start + 1_999_999, b_b), None
        while n < CAP:
            q = {"category": ["20"], "contractAddresses": [token], "fromBlock": hex(start),
                 "toBlock": hex(end), "order": "asc", "maxCount": hex(H.PAGE)}
            if key:
                q["pageKey"] = key
            res = H.nr("nr_getAssetTransfers", [q])
            n += len(res.get("transfers") or [])
            key = res.get("pageKey")
            if not key:
                break
        start = end + 1
    return n


def main(data_path, hz_path, out):
    x = pd.read_excel(data_path, sheet_name="data")
    x["key"] = x["Project Title"].astype(str) + "|" + x["Blockchain"].astype(str)
    hz = pd.read_csv(hz_path)
    hz = hz[hz["status"].isin(["ok", "start mismatch"])].drop_duplicates("key").set_index("key")
    done = set(pd.read_csv(out)["key"]) if os.path.exists(out) else set()
    for _, p in x.drop_duplicates("key").iterrows():
        k = p["key"]
        if k not in hz.index or k in done:
            continue
        chain, addr = str(p["Blockchain"]).upper(), hz.loc[k, "address"]
        t_end = int(pd.Timestamp(p["project end date"]).timestamp())
        row = {"key": k, "class": p["class"], "address": addr, "end date": p["project end date"]}
        try:
            f = bsc_count if chain == "BSC" else (lambda a, s, e: escan_count(H.ESCAN_CHAINS[chain], a, s, e))
            row["before"] = f(addr, t_end - 30 * 86400, t_end)
            row["after"] = f(addr, t_end + 3 * 86400, t_end + 33 * 86400) if t_end + 33 * 86400 < pd.Timestamp.now().timestamp() else None
            flag = ""
            if p["class"] == "scam":
                if row["before"] == 0:
                    flag = "inactive before recorded collapse"
                elif row["after"] is not None and row["after"] >= 50 and row["after"] > 0.5 * row["before"]:
                    flag = "activity did not collapse at recorded end"
            elif row["before"] == 0:
                flag = "no activity before collection date"
            row["flag"] = flag
        except Exception as exc:
            row["flag"] = "error: " + str(exc)[:100]
        pd.DataFrame([row]).to_csv(out, mode="a", header=not os.path.exists(out), index=False)
        done.add(k)
        print(len(done), k[:40], row.get("before"), row.get("after"), row["flag"], flush=True)


if __name__ == "__main__":
    if len(sys.argv) != 4:
        sys.exit(__doc__)
    main(*sys.argv[1:])
