"""
Candidate small legitimate projects for TM-RugPull, to reduce the fame bias of the
legitimate class. Every candidate must be reviewed by hand before it is used.

Selection
  * CoinGecko tokens with a market capitalization between 1 and 50 million USD and a contract
    on Ethereum or BNB Smart Chain; stablecoins, wrapped, bridged and tokenized assets removed;
    tokens already in the dataset (by address or symbol) removed.
  * A seeded random sample, then on-chain checks on the token's own contract:
      - first transfer between 2017-01-01 and 2024-11-01 (the period of the dataset, and at
        least two years of history before today),
      - a Uniswap/PancakeSwap V2 pool that still holds liquidity today,
      - transfers in the last 30 days.
  * CoinGecko project data: website, X account, categories, genesis date.

Suitability (column `Candidate suitability`)
  fully suitable          passes every check, has a website and an X account, is older than
                          three years and is not a meme token
  needs manual review     passes the on-chain checks but misses one of the above
Neither verdict checks incident databases: the reviewer must confirm that no rug pull, exit
scam or exploit was ever reported for the project.

Usage
    export ETHERSCAN_API_KEY=... NODEREAL_URL=...
    python scripts/find_small_legit.py candidates.csv results/small_legit_candidates.csv
"""
import os
import sys
import time

import pandas as pd
import requests

sys.path.insert(0, os.path.dirname(__file__))
import audit_collapse as A  # noqa: E402
import liquidity_check as Q  # noqa: E402
import onchain_horizons as H  # noqa: E402

A.CAP = 1000
COLS = ["id", "name", "symbol", "chain", "address", "mcap", "vol", "ath_date", "check", "first_transfer",
        "token_symbol", "token_name", "pool", "dex", "quote", "liquidity_now", "transfers_last_30d",
        "website", "x profile", "categories", "genesis_date"]
SAMPLE = {"ETH": 90, "BSC": 70}
LO, HI = pd.Timestamp("2017-01-01"), pd.Timestamp("2024-11-01")


def gecko(cid):
    for attempt in range(5):
        r = requests.get(f"https://api.coingecko.com/api/v3/coins/{cid}", timeout=30, params=dict(
            localization="false", tickers="false", market_data="false", community_data="false",
            developer_data="false"))
        if r.status_code == 429:
            time.sleep(30 * (attempt + 1))
            continue
        time.sleep(6)
        return r.json() if r.ok else {}
    return {}


def main(cand_path, out):
    c = pd.read_csv(cand_path)
    c = pd.concat([c[c["chain"] == ch].sample(min(n, (c["chain"] == ch).sum()), random_state=42)
                   for ch, n in SAMPLE.items()])
    done = set(pd.read_csv(out)["id"]) if os.path.exists(out) else set()
    now = int(time.time())
    for _, r in c.iterrows():
        if r["id"] in done:
            continue
        row = r.to_dict()
        try:
            pr = H.probe(r["chain"], r["address"])
            if pr is None:
                row["check"] = "no transfers"
            else:
                first = pd.Timestamp(pr[0], unit="s")
                row.update(first_transfer=first, token_symbol=pr[1], token_name=pr[2])
                if not (LO <= first <= HI):
                    row["check"] = "first transfer outside 2017-01 .. 2024-11"
                else:
                    liq = None
                    facs, quotes = (Q.ETH_FACTORIES, Q.ETH_QUOTES) if r["chain"] == "ETH" else (Q.BSC_FACTORIES, Q.BSC_QUOTES)
                    for fn, fa in facs.items():
                        for qn, (q, d) in quotes.items():
                            pair = Q.get_pair(r["chain"], fa, r["address"], q)
                            if pair:
                                b = Q.balance_now(r["chain"], q, pair) / 10 ** d
                                if liq is None or b > liq[0]:
                                    liq = (b, qn, fn, pair)
                    if not liq or liq[0] <= 0:
                        row["check"] = "no V2 pool with liquidity"
                    else:
                        row.update(pool=liq[3], dex=liq[2], quote=liq[1], liquidity_now=liq[0])
                        f = A.bsc_count if r["chain"] == "BSC" else (lambda a, s, e: A.escan_count(1, a, s, e))
                        row["transfers_last_30d"] = f(r["address"], now - 30 * 86400, now - 600)
                        row["check"] = "ok" if row["transfers_last_30d"] > 0 else "no transfers in last 30 days"
            if row.get("check") == "ok":
                g = gecko(r["id"])
                links = g.get("links") or {}
                row["website"] = next((u for u in links.get("homepage") or [] if u), "")
                tw = links.get("twitter_screen_name") or ""
                row["x profile"] = f"https://x.com/{tw}" if tw else ""
                row["categories"] = "; ".join(g.get("categories") or [])
                row["genesis_date"] = g.get("genesis_date")
        except Exception as exc:
            row["check"] = "error: " + str(exc)[:80]
        pd.DataFrame([row]).reindex(columns=COLS).to_csv(out, mode="a", header=not os.path.exists(out), index=False)
        done.add(r["id"])
        print(len(done), r["chain"], r["name"][:30], row.get("check"), flush=True)


if __name__ == "__main__":
    if len(sys.argv) != 3:
        sys.exit(__doc__)
    main(*sys.argv[1:])
