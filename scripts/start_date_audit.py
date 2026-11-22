"""
What does the recorded start date correspond to on chain?

For every project with a confirmed contract (results/onchain_source.csv) the script collects
three on-chain dates and compares them with `project starting date`:
    first transfer    first Transfer event of the token (already in onchain_source.csv)
    pool creation     first transfer of the token into its main V2 pool (liquidity added,
                      i.e. the moment the token becomes tradable on a DEX)
    contract creation timestamp of the deployment transaction (Etherscan chains only)

Output: results/start_date_audit.csv with the three dates and their distance in days from
the recorded start date.

Usage
    export ETHERSCAN_API_KEY=... NODEREAL_URL=...
    python scripts/start_date_audit.py data/TM-RugPull_v2.xlsx results/onchain_source.csv results/start_date_audit.csv
"""
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
import onchain_horizons as H  # noqa: E402

COLS = ["Excel row", "Project Title", "class", "chain", "recorded_start", "first_transfer", "pool_created",
        "contract_created", "d_first_transfer", "d_pool", "d_contract", "note"]


def eth_pool_created(chain_id, token, pool):
    res = H.escan(chain_id, module="account", action="tokentx", contractaddress=token, address=pool,
                  startblock=0, endblock=99999999, page=1, offset=1, sort="asc")
    return int(res[0]["timeStamp"]) if res else None


def bsc_pool_created(token, pool, t_first):
    b0 = H.bsc_block_at(t_first) - 5
    latest = int(H.nr("eth_blockNumber", []), 16)
    for k in range(25):
        a = b0 + k * 2_000_000
        if a > latest:
            break
        res = H.nr("nr_getAssetTransfers", [{"category": ["20"], "contractAddresses": [token], "toAddress": pool,
                   "fromBlock": hex(a), "toBlock": hex(min(a + 1_999_999, latest)), "order": "asc", "maxCount": hex(1)}])
        t = res.get("transfers") or []
        if t:
            return int(t[0]["blockTimeStamp"])
    return None


def eth_contract_created(chain_id, token):
    res = H.escan(chain_id, module="contract", action="getcontractcreation", contractaddresses=token)
    if res and isinstance(res, list):
        ts = res[0].get("timestamp")
        return int(ts) if ts else None
    return None


def main(data_path, src_path, out):
    d = pd.read_excel(data_path, sheet_name="data")
    d["Excel row"] = d.index + 2
    s = pd.read_csv(src_path)
    s = s[s["status"].astype(str).str.startswith("ok") & s["first_transfer"].notna()]
    m = s.merge(d[["Excel row", "project starting date", "Contract address"]], on="Excel row")
    m = m[m["Contract address"].notna()]
    done = set(pd.read_csv(out)["Excel row"]) if os.path.exists(out) else set()
    shard, nshard = map(int, os.environ.get("SHARD", "0/1").split("/"))
    for n, (_, r) in enumerate(m.iterrows()):
        if r["Excel row"] in done or n % nshard != shard:
            continue
        chain, token = r["chain_used"], r["address"].lower()
        start = pd.Timestamp(r["project starting date"])
        ft = pd.Timestamp(r["first_transfer"])
        row = {"Excel row": r["Excel row"], "Project Title": r["Project Title"], "class": r["class"], "chain": chain,
               "recorded_start": start, "first_transfer": ft, "d_first_transfer": (ft - start).days}
        try:
            pool = r["pool"].lower() if isinstance(r.get("pool"), str) else None
            if pool:
                ts = (bsc_pool_created(token, pool, int(ft.timestamp())) if chain == "BSC"
                      else eth_pool_created(H.ESCAN_CHAINS[chain], token, pool))
                if ts:
                    row["pool_created"] = pd.Timestamp(ts, unit="s")
                    row["d_pool"] = (row["pool_created"] - start).days
            if chain in H.ESCAN_CHAINS:
                ts = eth_contract_created(H.ESCAN_CHAINS[chain], token)
                if ts:
                    row["contract_created"] = pd.Timestamp(ts, unit="s")
                    row["d_contract"] = (row["contract_created"] - start).days
        except Exception as exc:
            row["note"] = "error: " + str(exc)[:80]
        pd.DataFrame([row]).reindex(columns=COLS).to_csv(out, mode="a", header=not os.path.exists(out), index=False)
        print(n, chain, str(r["Project Title"])[:30], row.get("d_first_transfer"), row.get("d_pool"), row.get("d_contract"), flush=True)


if __name__ == "__main__":
    if len(sys.argv) != 4:
        sys.exit(__doc__)
    main(*sys.argv[1:])
