"""
Liquidity check of the rug-pull labels from the token's main DEX pool.

The labeling rule says that a rug pull leaves its pools with near-zero liquidity. For every
rug pull whose contract was confirmed (results/onchain_horizons.csv), the script finds the
token's Uniswap-V2-style pool against the chain's main quote tokens and measures the quote-token
balance held by the pool (WETH, WBNB or a stablecoin):
    before  the highest balance in the 30 days before the recorded end date (daily samples)
    after   the balance 3 days after the recorded end date
Verdicts
    drained              after < 10% of before (liquidity removed)
    frozen (no trading)  the balance changed by less than 2%: no trades in or out of the pool,
                         consistent with a dead pool whose liquidity is locked or unsellable
    still trading        anything else: the pool was still in use after the recorded end
    no V2 pool           no pool on the factories checked (V3 pools and other DEXs are not covered)
For legitimate projects the script records whether a pool with liquidity still exists today.

Ethereum: Etherscan's eth_call does not serve historical state, so the pool's WETH (or stable)
balance is rebuilt from the quote token's transfers to and from the pool. BNB Smart Chain: the
NodeReal archive node is queried at the block of each sample date.

Usage
    export ETHERSCAN_API_KEY=... NODEREAL_URL=...
    python scripts/liquidity_check.py data/TM-RugPull_v2.xlsx results/onchain_horizons.csv results/liquidity_check.csv
"""
import os
import sys

import pandas as pd
import requests

sys.path.insert(0, os.path.dirname(__file__))
import onchain_horizons as H  # noqa: E402

ETH_FACTORIES = {"uniswap-v2": "0x5c69bee701ef814a2b6a3edd4b1652cb9cc5aa6f",
                 "sushiswap": "0xc0aee478e3658e2610c5f7a4a2e1777ce9e4f2ac"}
ETH_QUOTES = {"WETH": ("0xc02aaa39b223fe8d0a0e5c4f27ead9083c756cc2", 18),
              "USDT": ("0xdac17f958d2ee523a2206206994597c13d831ec7", 6),
              "USDC": ("0xa0b86991c6218b36c1d19d4a2e9eb0ce3606eb48", 6),
              "DAI": ("0x6b175474e89094c44da98b954eedeac495271d0f", 18)}
BSC_FACTORIES = {"pancake-v2": "0xca143ce32fe78f1f7019d7d551a6402fc5350c73",
                 "pancake-v1": "0xbcfccbde45ce874adcb698cc183debcf17952812"}
BSC_QUOTES = {"WBNB": ("0xbb4cdb9cbd36b01bd1cbaebf2de08d9173bc095c", 18),
              "BUSD": ("0xe9e7cea3dedca5984780bafc599bd69add087d56", 18),
              "USDT": ("0x55d398326f99059ff775485246999027b3197955", 18)}
ZERO = "0x" + "0" * 40
DAY = 86400
MAX_TX = 50_000
COLS = ["key", "class", "address", "end date", "pool", "dex", "quote", "before", "after", "now", "capped", "verdict"]


def pad(a):
    return a.lower().replace("0x", "").rjust(64, "0")


def ecall_latest(to, data):
    import time
    for attempt in range(6):
        time.sleep(0.4)                                   # free tier: 3 calls per second
        r = requests.get("https://api.etherscan.io/v2/api", timeout=30, params=dict(
            chainid=1, module="proxy", action="eth_call", to=to, data=data, tag="latest",
            apikey=os.environ["ETHERSCAN_API_KEY"])).json()
        res = r.get("result") or "0x"
        if isinstance(res, str) and res.startswith("0x"):
            return res
        time.sleep(2 * (attempt + 1))                     # rate-limit message: wait and retry
    raise RuntimeError(f"etherscan eth_call failed: {str(res)[:80]}")


def get_pair(chain, factory, token, quote):
    data = "0xe6a43905" + pad(token) + pad(quote)
    res = ecall_latest(factory, data) if chain == "ETH" else H.nr("eth_call", [{"to": factory, "data": data}, "latest"])
    a = "0x" + res[-40:] if res and len(res) >= 42 else ZERO
    return None if a == ZERO else a


def balance_now(chain, quote, holder):
    data = "0x70a08231" + pad(holder)
    res = ecall_latest(quote, data) if chain == "ETH" else H.nr("eth_call", [{"to": quote, "data": data}, "latest"])
    return int(res, 16) if res and res != "0x" else 0


def bsc_balance_at(quote, holder, ts):
    b = H.bsc_block_at(ts)
    res = H.nr("eth_call", [{"to": quote, "data": "0x70a08231" + pad(holder)}, hex(b)])
    return int(res, 16) if res and res != "0x" else 0


def eth_balance_series(quote, pair, t_until):
    """Running balance of `quote` held by `pair`, from its transfers, up to t_until."""
    rows, start = [], 0
    while len(rows) < MAX_TX:
        res = H.escan(1, module="account", action="tokentx", contractaddress=quote, address=pair,
                      startblock=start, endblock=99999999, page=1, offset=H.PAGE, sort="asc")
        if not res:
            break
        rows += res if len(res) < H.PAGE else [r for r in res if int(r["blockNumber"]) < int(res[-1]["blockNumber"])]
        if len(res) < H.PAGE or int(res[-1]["timeStamp"]) > t_until:
            break
        start = int(res[-1]["blockNumber"])
    bal, series = 0, []
    for r in rows:
        v = int(r["value"])
        bal += v if r["to"].lower() == pair else -v if r["from"].lower() == pair else 0
        series.append((int(r["timeStamp"]), bal))
    return series, len(rows) >= MAX_TX


def at(series, ts):
    b = 0
    for t, v in series:
        if t > ts:
            break
        b = v
    return b


def main(data_path, hz_path, out):
    x = pd.read_excel(data_path, sheet_name="data")
    x = x[x.get("Use in training", "yes").astype(str).str.lower() != "no"].copy()
    x["key"] = x["Project Title"].astype(str) + "|" + x["Blockchain"].astype(str)
    hz = pd.read_csv(hz_path)
    hz = hz[hz["status"].isin(["ok", "start mismatch"])].drop_duplicates("key").set_index("key")
    done = set(pd.read_csv(out)["key"]) if os.path.exists(out) else set()
    for _, p in x.drop_duplicates("key").iterrows():
        k, chain = p["key"], str(p["Blockchain"]).upper()
        if k not in hz.index or k in done or chain not in ("ETH", "BSC"):
            continue
        token = hz.loc[k, "address"].lower()
        facs, quotes = (ETH_FACTORIES, ETH_QUOTES) if chain == "ETH" else (BSC_FACTORIES, BSC_QUOTES)
        row = {"key": k, "class": p["class"], "address": token, "end date": p["project end date"]}
        try:
            pools = [(fn, qn, q, d, get_pair(chain, fa, token, q))
                     for fn, fa in facs.items() for qn, (q, d) in quotes.items()]
            pools = [(fn, qn, q, d, pr) for fn, qn, q, d, pr in pools if pr]
            if not pools:
                row["verdict"] = "no V2 pool"
            elif p["class"] != "scam":
                # legitimate: does a pool with liquidity still exist today?
                best = max(pools, key=lambda t: balance_now(chain, t[2], t[4]) / 10 ** t[3])
                row.update(pool=best[4], dex=best[0], quote=best[1],
                           now=balance_now(chain, best[2], best[4]) / 10 ** best[3])
                row["verdict"] = "pool with liquidity today" if row["now"] > 0 else "pool empty today"
            else:
                t_end = int(pd.Timestamp(p["project end date"]).timestamp())
                res = []
                for fn, qn, q, d, pr in pools:
                    if chain == "ETH":
                        ser, capped = eth_balance_series(q, pr, t_end + 4 * DAY)
                        before = max([v for t, v in ser if t_end - 30 * DAY <= t < t_end] + [at(ser, t_end - 30 * DAY)])
                        after = at(ser, t_end + 3 * DAY)
                    else:
                        before = max(bsc_balance_at(q, pr, t_end - j * DAY) for j in (1, 3, 7, 14, 30))
                        after, capped = bsc_balance_at(q, pr, t_end + 3 * DAY), False
                    res.append((before / 10 ** d, after / 10 ** d, fn, qn, pr, capped))
                before, after, fn, qn, pr, capped = max(res, key=lambda r: r[0])
                row.update(pool=pr, dex=fn, quote=qn, before=before, after=after, capped=capped)
                if before == 0:
                    row["verdict"] = "pool empty before recorded end"
                else:
                    r = after / before
                    row["verdict"] = ("drained" if r < 0.1 else
                                      "frozen (no trading)" if abs(r - 1) < 0.02 else "still trading")
        except Exception as exc:
            row["verdict"] = "error: " + str(exc)[:100]
        pd.DataFrame([row]).reindex(columns=COLS).to_csv(out, mode="a", header=not os.path.exists(out), index=False)
        done.add(k)
        print(len(done), k[:40], row.get("verdict"), row.get("before"), row.get("after"), flush=True)


if __name__ == "__main__":
    if len(sys.argv) != 4:
        sys.exit(__doc__)
    main(*sys.argv[1:])
