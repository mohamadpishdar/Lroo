"""
All on-chain evidence for every project, computed on the contract recorded at collection.

For each row with a `Contract address` (recorded in the dataset) the script
  1. picks the chain: the chain of the recorded explorer link, or the Blockchain column if the
     token has no transfers there (31 rows disagree; see the Review list),
  2. downloads the first 30 days of transfers and computes transfer count, holders, top-10
     share, HHI and Gini at days 7, 14 and 30 after the first transfer,
  3. counts transfers in the 30 days before the recorded end date and 3-33 days after it,
  4. measures the quote-token balance of the token's main V2 pool before and after the end
     date (rug pulls) or today (legitimate projects), as in liquidity_check.py,
  5. for rug pulls whose pool kept trading, counts swaps, distinct wallets and the share of the
     three most active wallets after the end date.
Results already computed for the same contract on the same chain are reused.

Output: one row per project (results/onchain_source.csv). Covered chains: Ethereum, Polygon,
Arbitrum (Etherscan V2) and BNB Smart Chain (NodeReal archive). Steps 4-5 run on Ethereum and
BNB Smart Chain only.

Usage
    export ETHERSCAN_API_KEY=... NODEREAL_URL=...
    [SHARD=i/n] [ONLY_CHAINS=BSC] python scripts/onchain_from_source.py data/TM-RugPull_v2.xlsx results/onchain_source.csv
"""
import collections
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
import audit_collapse as A  # noqa: E402
import liquidity_check as Q  # noqa: E402
import onchain_horizons as H  # noqa: E402

ROOT = os.path.join(os.path.dirname(__file__), "..", "results")
A.CAP = 5_000                                   # transfers counted per window: enough to compare before/after
_block_cache = {}
_bsc_block_at = H.bsc_block_at


def _cached_block(ts):
    ts = int(ts) // 600 * 600                    # 10-minute resolution is enough for daily samples
    if ts not in _block_cache:
        _block_cache[ts] = _bsc_block_at(ts)
    return _block_cache[ts]


H.bsc_block_at = _cached_block
COVERED = set(H.ESCAN_CHAINS) | {"BSC"}
DAY = 86400
FEAT = [f"{k}_d{h}" for h in H.HORIZONS for k in ("transfers", "holders", "top10", "hhi", "gini")]
COLS = (["key", "Excel row", "Project Title", "class", "address", "chain_used", "status", "symbol_match",
         "token_symbol", "token_name", "first_transfer", "days_first_vs_start", "capped"] + FEAT +
        ["tx_before_end", "tx_after_end", "pool", "dex", "quote", "liq_before", "liq_after", "liq_now",
         "liq_verdict", "swaps_after", "wallets_after", "top3_share_after", "reused"])


def previous():
    """Earlier results keyed by (address, chain), reused when the contract is the same."""
    hz, au, lq = {}, {}, {}
    p = os.path.join(ROOT, "onchain_horizons.csv")
    if os.path.exists(p):
        for _, r in pd.read_csv(p).iterrows():
            if r["status"] in ("ok", "start mismatch") and isinstance(r.get("address"), str):
                hz[(r["address"].lower(), str(r["Blockchain"]).upper())] = r
    for f in ("collapse_audit_eth.csv", "collapse_audit_bsc.csv"):
        p = os.path.join(ROOT, f)
        if os.path.exists(p):
            for _, r in pd.read_csv(p).iterrows():
                au[(str(r["address"]).lower(), r["key"].split("|")[-1].upper(), str(r["end date"])[:10])] = r
    p = os.path.join(ROOT, "liquidity_check.csv")
    if os.path.exists(p):
        for _, r in pd.read_csv(p).iterrows():
            lq[(str(r["address"]).lower(), r["key"].split("|")[-1].upper(), str(r["end date"])[:10], r["class"])] = r
    return hz, au, lq


def pool_activity(chain, token, pool, t_a, t_b):
    rows = []
    if chain == "ETH":
        ba = int(H.escan(1, module="block", action="getblocknobytime", timestamp=t_a, closest="after"))
        bb = int(H.escan(1, module="block", action="getblocknobytime", timestamp=t_b, closest="before"))
        start = ba
        while len(rows) < 20000:
            res = H.escan(1, module="account", action="tokentx", contractaddress=token, address=pool,
                          startblock=start, endblock=bb, page=1, offset=1000, sort="asc")
            if not res:
                break
            rows += [(x["from"].lower(), x["to"].lower()) for x in res]
            if len(res) < 1000:
                break
            start = int(res[-1]["blockNumber"]) + 1
    else:
        ba, bb = H.bsc_block_at(t_a), H.bsc_block_at(t_b)
        for side in ("fromAddress", "toAddress"):
            key = None
            while len(rows) < 20000:
                q = {"category": ["20"], "contractAddresses": [token], side: pool, "fromBlock": hex(ba),
                     "toBlock": hex(bb), "order": "asc", "maxCount": hex(1000)}
                if key:
                    q["pageKey"] = key
                res = H.nr("nr_getAssetTransfers", [q])
                rows += [(x["from"].lower(), x["to"].lower()) for x in res.get("transfers") or []]
                key = res.get("pageKey")
                if not key:
                    break
    cp = collections.Counter(f if t == pool else t for f, t in rows if pool in (f, t))
    n = sum(cp.values())
    top3 = sum(v for _, v in cp.most_common(3)) / n if n else np.nan
    return n, len(cp), top3


def liquidity(chain, token, cls, t_end):
    facs, quotes = (Q.ETH_FACTORIES, Q.ETH_QUOTES) if chain == "ETH" else (Q.BSC_FACTORIES, Q.BSC_QUOTES)
    pools = [(fn, qn, q, d, Q.get_pair(chain, fa, token, q)) for fn, fa in facs.items() for qn, (q, d) in quotes.items()]
    pools = [t for t in pools if t[4]]
    if not pools:
        return {"liq_verdict": "no V2 pool"}
    if cls != "scam":
        best = max(((Q.balance_now(chain, t[2], t[4]) / 10 ** t[3], t) for t in pools), key=lambda z: z[0])
        return {"pool": best[1][4], "dex": best[1][0], "quote": best[1][1], "liq_now": best[0],
                "liq_verdict": "pool with liquidity today" if best[0] > 0 else "pool empty today"}
    res = []
    for fn, qn, q, d, pr in pools:
        if chain == "ETH":
            ser, _ = Q.eth_balance_series(q, pr, t_end + 4 * DAY)
            before = max([v for t, v in ser if t_end - 30 * DAY <= t < t_end] + [Q.at(ser, t_end - 30 * DAY)])
            after = Q.at(ser, t_end + 3 * DAY)
        else:
            before = max(Q.bsc_balance_at(q, pr, t_end - j * DAY) for j in (1, 3, 7, 14, 30))
            after = Q.bsc_balance_at(q, pr, t_end + 3 * DAY)
        res.append((before / 10 ** d, after / 10 ** d, fn, qn, pr))
    before, after, fn, qn, pr = max(res, key=lambda r: r[0])
    out = {"pool": pr, "dex": fn, "quote": qn, "liq_before": before, "liq_after": after}
    if before == 0:
        out["liq_verdict"] = "pool empty before recorded end"
    else:
        r = after / before
        out["liq_verdict"] = "drained" if r < 0.1 else "frozen (no trading)" if abs(r - 1) < 0.02 else "still trading"
    return out


def main(data_path, out):
    x = pd.read_excel(data_path, sheet_name="data")
    x["Excel row"] = x.index + 2
    x["key"] = x["Project Title"].astype(str) + "|" + x["Blockchain"].astype(str) + "|" + x["Excel row"].astype(str)
    import glob
    done = set()
    for f in glob.glob(os.path.join(os.path.dirname(out) or ".", "onchain_source*.csv")) + [out]:
        if os.path.exists(f) and os.path.getsize(f) > 0:
            done |= set(pd.read_csv(f)["key"])            # keys finished by any worker
    shard, nshard = map(int, os.environ.get("SHARD", "0/1").split("/"))
    only = set(os.environ["ONLY_CHAINS"].split(",")) if os.environ.get("ONLY_CHAINS") else None
    hz_prev, au_prev, lq_prev = previous()
    for i, p in x.iterrows():
        if p["key"] in done or i % nshard != shard or not isinstance(p.get("Contract address"), str):
            continue
        addr = p["Contract address"].lower()
        rec = str(p["Blockchain"]).upper().replace("FTM", "FANTOM")
        expl = str(p.get("Explorer chain") or rec).upper()
        chains = [c for c in dict.fromkeys([expl, rec]) if c in COVERED]
        if only and not (set(chains) & only):
            continue
        row = {"key": p["key"], "Excel row": p["Excel row"], "Project Title": p["Project Title"], "class": p["class"],
               "address": addr, "reused": ""}
        if not chains:
            row["status"] = "chain not covered"
        for chain in chains:
            prev = hz_prev.get((addr, chain))
            if prev is not None:
                r = {k: prev.get(k) for k in ["token_symbol", "token_name", "first_transfer", "days_first_vs_start",
                                              "capped"] + FEAT}
                r.update(status="ok", symbol_match=True)
                row["reused"] = "features"
            else:
                try:
                    r = H.check_full(p, chain, addr, source=True)
                except Exception as exc:
                    r = {"status": "error: " + str(exc)[:100]}
            if r.get("status") != "no transfers" or chain == chains[-1]:
                row.update(r)
                row["chain_used"] = chain
                break
        chain = row.get("chain_used")
        if str(row.get("status", "")).startswith("ok"):
            t_end = int(pd.Timestamp(p["project end date"]).timestamp())
            end_s = str(p["project end date"])[:10]
            try:
                a = au_prev.get((addr, chain, end_s))
                if a is not None:
                    row["tx_before_end"], row["tx_after_end"] = a["before"], a["after"]
                    row["reused"] += "+activity"
                else:
                    f = A.bsc_count if chain == "BSC" else (lambda ad, s_, e_: A.escan_count(H.ESCAN_CHAINS[chain], ad, s_, e_))
                    row["tx_before_end"] = f(addr, t_end - 30 * DAY, t_end)
                    if t_end + 33 * DAY < pd.Timestamp.now().timestamp():
                        row["tx_after_end"] = f(addr, t_end + 3 * DAY, t_end + 33 * DAY)
                if chain in ("ETH", "BSC"):
                    l_ = lq_prev.get((addr, chain, end_s, p["class"]))
                    if l_ is not None and str(l_.get("verdict")) != "nan":
                        row.update(pool=l_.get("pool"), dex=l_.get("dex"), quote=l_.get("quote"),
                                   liq_before=l_.get("before"), liq_after=l_.get("after"), liq_now=l_.get("now"),
                                   liq_verdict=l_.get("verdict"))
                        row["reused"] += "+liquidity"
                    else:
                        row.update(liquidity(chain, addr, p["class"], t_end))
                    if p["class"] == "scam" and row.get("liq_verdict") == "still trading":
                        n, w, t3 = pool_activity(chain, addr, row["pool"].lower(), t_end + 3 * DAY, t_end + 33 * DAY)
                        row.update(swaps_after=n, wallets_after=w, top3_share_after=t3)
            except Exception as exc:
                row["liq_verdict"] = "error: " + str(exc)[:100]
        pd.DataFrame([row]).reindex(columns=COLS).to_csv(out, mode="a", header=not os.path.exists(out), index=False)
        done.add(p["key"])
        print(len(done), chain, str(p["Project Title"])[:30], row.get("status"), row.get("liq_verdict"), flush=True)


if __name__ == "__main__":
    if len(sys.argv) != 3:
        sys.exit(__doc__)
    main(*sys.argv[1:])
