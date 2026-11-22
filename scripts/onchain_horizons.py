"""
On-chain features at fixed horizons after launch (day 7, 14 and 30), for Ethereum-family
chains through Etherscan API V2 and for BNB Smart Chain through NodeReal.

These features depend only on the start date, never on the end date, so they are
observable at t_start + h and are free of the end-date asymmetry described in the paper.

For each project with a candidate contract address (the `Contract address` column of the dataset) the script
  1. downloads the token's Transfer events from its first transfer (its on-chain launch,
     t_launch) up to t_launch + 30 days,
  2. checks identity: the token symbol matches the symbol in the project title (or, when the
     title has none, the token name matches the project name), and the contract did not
     appear more than 60 days AFTER the recorded start date (that pattern indicates a later
     migrated contract rather than the original token),
  3. rebuilds balances from the transfers and computes, at each horizon h, transfer count,
     unique holders, top-10 share, Herfindahl-Hirschman index and Gini of balances at
     t_launch + h.

Status values
  ok              identity confirmed and first transfer within 60 days of the recorded start
  start mismatch  symbol AND name confirmed, but the contract existed more than 60 days BEFORE the
                  recorded start; features are still anchored on t_launch, which is earlier
                  than the recorded start, so they remain observable before it
  rejected: ...   symbol/name mismatch, no transfers, or a later (migrated) contract
The same rule is applied to both classes.

Usage
  export ETHERSCAN_API_KEY=...      # Ethereum, Polygon, Arbitrum
  export NODEREAL_URL=...           # https://bsc-mainnet.nodereal.io/v1/<key>, for BSC
  python scripts/onchain_horizons.py data/TM-RugPull_v2.xlsx addresses.csv,results/address_candidates.csv results/onchain_horizons.csv

The run is resumable: rows already in the output file are skipped. Keys are read from the
environment only and never written to disk.
"""
import os
import re
import sys
import time

import numpy as np
import pandas as pd
import requests

ETHERSCAN = "https://api.etherscan.io/v2/api"
EKEY = os.environ.get("ETHERSCAN_API_KEY", "")
NR = os.environ.get("NODEREAL_URL", "")
ESCAN_CHAINS = {"ETH": 1, "POLYGON": 137, "ARBI": 42161}
HORIZONS = [7, 14, 30]
MAX_DAYS = max(HORIZONS)
WINDOW_BEFORE = 60            # days before the recorded start that a first transfer may lie
MAX_TRANSFERS = 400_000       # safety cap per token
MAX_CANDIDATES = 5
ZERO = "0x0000000000000000000000000000000000000000"
PAGE = 1000
S = requests.Session()


# ------------------------------------------------------------------ Etherscan backend
def escan(chain_id, **p):
    p.update(chainid=chain_id, apikey=EKEY)
    for attempt in range(6):
        try:
            r = S.get(ETHERSCAN, params=p, timeout=30)
            js = r.json()
        except Exception:
            time.sleep(2 ** attempt)
            continue
        time.sleep(0.21)
        if js.get("status") == "1" or js.get("message") == "No transactions found":
            return js.get("result", [])
        msg = str(js.get("result", ""))
        if "rate limit" in msg.lower() or "max calls" in msg.lower():
            time.sleep(1 + attempt)
            continue
        raise RuntimeError(f"etherscan: {js.get('message')} {msg}"[:160])
    raise RuntimeError("etherscan: retries exhausted")


def escan_transfers(chain_id, token, days, cap=MAX_TRANSFERS):
    """Transfers in ascending order from the first one until first + `days` days."""
    rows, start, t_stop = [], 0, None
    while True:
        res = escan(chain_id, module="account", action="tokentx", contractaddress=token,
                    startblock=start, endblock=99999999, page=1, offset=PAGE, sort="asc")
        if not res:
            break
        if len(res) < PAGE:
            rows += res
            break
        last = int(res[-1]["blockNumber"])
        if last == start:                         # > PAGE transfers in one block
            rows += res
            pg = 2
            while True:
                more = escan(chain_id, module="account", action="tokentx", contractaddress=token,
                             startblock=last, endblock=last, page=pg, offset=PAGE, sort="asc")
                rows += more
                if len(more) < PAGE:
                    break
                pg += 1
            start = last + 1
        else:
            rows += [r for r in res if int(r["blockNumber"]) < last]
            start = last
        if t_stop is None and rows:
            t_stop = int(rows[0]["timeStamp"]) + days * 86400
        if int(rows[-1]["timeStamp"]) > t_stop or len(rows) > cap:
            break
    out = [{"ts": int(r["timeStamp"]), "from": r["from"].lower(), "to": r["to"].lower(),
            "value": int(r["value"]) if r["value"] else 0, "dec": int(r.get("tokenDecimal") or 0),
            "symbol": r.get("tokenSymbol", ""), "name": r.get("tokenName", "")} for r in rows]
    if t_stop is None and out:                    # fewer than PAGE transfers in total: one page only
        t_stop = out[0]["ts"] + days * 86400
    if t_stop is None:
        return [], False
    return [r for r in out if r["ts"] <= t_stop], len(rows) > cap


# ------------------------------------------------------------------ NodeReal (BSC) backend
def nr(method, params):
    for attempt in range(6):
        try:
            js = S.post(NR, json={"jsonrpc": "2.0", "id": 1, "method": method, "params": params},
                        timeout=60).json()
        except Exception:
            time.sleep(2 ** attempt)
            continue
        time.sleep(0.15)
        if "error" in js:
            msg = str(js["error"].get("message", ""))
            if "limit" in msg.lower() or "too many" in msg.lower():
                time.sleep(2 + 2 * attempt)
                continue
            raise RuntimeError(f"nodereal: {msg}"[:160])
        return js["result"]
    raise RuntimeError("nodereal: retries exhausted")


_ts_cache = {}


def bsc_block_ts(n):
    if n not in _ts_cache:
        _ts_cache[n] = int(nr("eth_getBlockByNumber", [hex(n), False])["timestamp"], 16)
    return _ts_cache[n]


def bsc_block_at(ts):
    """Interpolation search: a few calls instead of ~27 for a plain binary search."""
    lo, hi = 1, int(nr("eth_blockNumber", []), 16)
    t_lo, t_hi = bsc_block_ts(lo), bsc_block_ts(hi)
    if ts >= t_hi:
        return hi
    if ts <= t_lo:
        return lo
    for _ in range(40):
        if hi - lo <= 1:
            break
        guess = lo + int((ts - t_lo) / max(t_hi - t_lo, 1) * (hi - lo))
        guess = min(max(guess, lo + 1), hi - 1)
        t_g = bsc_block_ts(guess)
        if t_g <= ts:
            lo, t_lo = guess, t_g
            if ts - t_g < 120:          # within two minutes is precise enough here
                return guess
        else:
            hi, t_hi = guess, t_g
    return lo


def bsc_creation_block(token):
    """Binary search on eth_getCode (archive node): first block in which the contract exists."""
    lo, hi = 0, int(nr("eth_blockNumber", []), 16)
    if nr("eth_getCode", [token, hex(hi)]) in ("0x", None):
        return None
    while hi - lo > 1:
        mid = (lo + hi) // 2
        if nr("eth_getCode", [token, hex(mid)]) in ("0x", None):
            lo = mid
        else:
            hi = mid
    return hi


def bsc_transfers(token, days, cap=MAX_TRANSFERS):
    """Transfers from contract creation until the first transfer + `days` days."""
    b0 = bsc_creation_block(token)
    if b0 is None:
        return [], False
    rows, start, b1 = [], b0, int(nr("eth_blockNumber", []), 16)
    t_stop = None
    while start <= b1:
        end = min(start + 1_999_999, b1)
        key = None
        while True:
            q = {"category": ["20"], "contractAddresses": [token], "fromBlock": hex(start),
                 "toBlock": hex(end), "order": "asc", "maxCount": hex(PAGE)}
            if key:
                q["pageKey"] = key
            res = nr("nr_getAssetTransfers", [q])
            rows += res.get("transfers") or []
            key = res.get("pageKey")
            if rows and t_stop is None:
                t_stop = int(rows[0]["blockTimeStamp"]) + days * 86400
                b1 = min(b1, bsc_block_at(t_stop) + 1)
            if not key or len(rows) > cap or (t_stop and int(rows[-1]["blockTimeStamp"]) > t_stop):
                break
        if len(rows) > cap or (t_stop and rows and int(rows[-1]["blockTimeStamp"]) > t_stop):
            break
        start = end + 1
    if t_stop is None:
        return [], False
    out = [{"ts": int(r["blockTimeStamp"]), "from": r["from"].lower(), "to": r["to"].lower(),
            "value": int(r["value"], 16) if r.get("value") else 0, "dec": int(r.get("decimal") or 0),
            "symbol": r.get("asset", ""), "name": r.get("name", "")} for r in rows]
    return [r for r in out if r["ts"] <= t_stop], len(rows) > cap


# ------------------------------------------------------------------ features
def features(tr, t_cut):
    sub = [r for r in tr if r["ts"] <= t_cut]
    if not sub:
        return {"transfers": 0, "holders": 0, "top10": np.nan, "hhi": np.nan, "gini": np.nan}
    bal = {}
    for r in sub:
        if r["from"] != ZERO:
            bal[r["from"]] = bal.get(r["from"], 0) - r["value"]
        bal[r["to"]] = bal.get(r["to"], 0) + r["value"]
    bal.pop(ZERO, None)
    v = np.array(sorted((b for b in bal.values() if b > 0), reverse=True), dtype=float)
    n = len(v)
    if n == 0:
        return {"transfers": len(sub), "holders": 0, "top10": np.nan, "hhi": np.nan, "gini": np.nan}
    sh = v / v.sum()
    s = np.sort(v)
    gini = 2 * np.sum(np.arange(1, n + 1) * s) / (n * s.sum()) - (n + 1) / n
    return {"transfers": len(sub), "holders": n, "top10": float(sh[:10].sum()),
            "hhi": float((sh ** 2).sum()), "gini": float(gini)}


BASE_COLS = ["key", "Project Title", "Blockchain", "class", "address", "status", "token_symbol",
             "token_name", "first_transfer", "days_first_vs_start", "capped", "candidates_tried"]
FEAT_COLS = [f"{k}_d{h}" for h in HORIZONS for k in ("transfers", "holders", "top10", "hhi", "gini")]
COLS = BASE_COLS + FEAT_COLS


def norm(s):
    return re.sub(r"[^a-z0-9]", "", str(s).lower())


def name_match(a, b):
    """Loose name match: one normalized name contains the other, or they share most characters."""
    from difflib import SequenceMatcher
    a, b = norm(a), norm(b)
    if not a or not b:
        return False
    return (a in b or b in a) and min(len(a), len(b)) >= 0.5 * max(len(a), len(b)) or SequenceMatcher(None, a, b).ratio() >= 0.7


def parse_symbol(title):
    m = re.match(r"^(.*?)\s*\(([^()]+)\)\s*$", str(title).replace("\xa0", " ").strip())
    return m.group(2).strip() if m else None


def probe(chain, addr):
    """First transfer only: (timestamp, symbol, name), or None if the token has no transfers."""
    if chain == "BSC":
        b0 = bsc_creation_block(addr)
        if b0 is None:
            return None
        latest = int(nr("eth_blockNumber", []), 16)
        for k in range(30):                       # up to 60M blocks after creation
            a = b0 + k * 2_000_000
            if a > latest:
                break
            res = nr("nr_getAssetTransfers", [{"category": ["20"], "contractAddresses": [addr],
                     "fromBlock": hex(a), "toBlock": hex(min(a + 1_999_999, latest)),
                     "order": "asc", "maxCount": hex(1)}])
            t = res.get("transfers") or []
            if t:
                return int(t[0]["blockTimeStamp"]), t[0].get("asset", ""), t[0].get("name", "")
        return None
    res = escan(ESCAN_CHAINS[chain], module="account", action="tokentx", contractaddress=addr,
                startblock=0, endblock=99999999, page=1, offset=1, sort="asc")
    if not res:
        return None
    return int(res[0]["timeStamp"]), res[0].get("tokenSymbol", ""), res[0].get("tokenName", "")


def identity(p, first, tok_sym, tok_name):
    sym = parse_symbol(p["Project Title"])
    pname = re.sub(r"\s*\([^()]*\)\s*$", "", str(p["Project Title"]))
    if sym is not None:
        ok_id = norm(sym) == norm(tok_sym)
    else:
        ok_id = name_match(pname, tok_name) or norm(pname) == norm(tok_sym)
    delta = (first - int(pd.Timestamp(p["project starting date"]).timestamp())) / 86400
    ok_late = delta <= WINDOW_BEFORE
    if not (ok_id and ok_late):
        return delta, "rejected: " + ",".join(n for n, ok in [("symbol", ok_id), ("later contract", ok_late)] if not ok)
    if delta < -WINDOW_BEFORE and not name_match(pname, tok_name):
        return delta, "rejected: name (old contract)"
    return delta, "ok" if delta >= -WINDOW_BEFORE else "start mismatch"


def check(p, chain, addr):
    """Verify one candidate from its first transfer; download the first month only if it passes."""
    try:
        pr = probe(chain, addr)
    except Exception as exc:
        return {"address": addr, "status": "error: " + str(exc)[:120]}
    if pr is None:
        return {"address": addr, "status": "rejected: no transfers"}
    delta, status = identity(p, *pr)
    if status.startswith("rejected"):
        return {"address": addr, "status": status, "token_symbol": pr[1], "token_name": pr[2] or None,
                "first_transfer": pd.Timestamp(pr[0], unit="s"), "days_first_vs_start": round(delta, 1)}
    return check_full(p, chain, addr)


def check_full(p, chain, addr, source=False):
    """Download and verify one candidate; return the output row fields.

    With source=True the address comes from the original collection (the explorer link in
    the source file). The identity checks are then recorded but do not reject the address,
    and features are computed whenever the token has transfers."""
    row = {"address": addr}
    start = pd.Timestamp(p["project starting date"])
    t0 = int(start.timestamp())
    try:
        if chain == "BSC":
            tr, capped = bsc_transfers(addr, MAX_DAYS)
        else:
            tr, capped = escan_transfers(ESCAN_CHAINS[chain], addr, MAX_DAYS)
    except Exception as exc:
        row["status"] = "error: " + str(exc)[:120]
        return row
    first = min((r["ts"] for r in tr), default=None)
    sym = parse_symbol(p["Project Title"])
    pname = re.sub(r"\s*\([^()]*\)\s*$", "", str(p["Project Title"]))
    tok_sym = tr[0]["symbol"] if tr else ""
    tok_name = tr[0]["name"] if tr else ""
    if sym is not None:
        ok_id = bool(tr) and norm(sym) == norm(tok_sym)
    else:
        ok_id = bool(tr) and (name_match(pname, tok_name) or norm(pname) == norm(tok_sym))
    delta = (first - t0) / 86400 if first else None
    ok_late = delta is not None and delta <= WINDOW_BEFORE
    row.update(token_symbol=tok_sym, token_name=tok_name or None,
               first_transfer=pd.Timestamp(first, unit="s") if first else None,
               days_first_vs_start=round(delta, 1) if delta is not None else None, capped=capped)
    row["symbol_match"] = ok_id
    if source:
        row["status"] = "no transfers" if not tr else "ok" if (ok_id and ok_late) else (
            "ok, " + ",".join(n for n, ok in [("symbol differs", ok_id), ("contract later than start", ok_late)] if not ok))
        if tr:
            for h in HORIZONS:
                f = features(tr, first + h * 86400)
                if capped and tr[-1]["ts"] < first + h * 86400:
                    f = {k: np.nan for k in f}
                for k, v in f.items():
                    row[f"{k}_d{h}"] = v
        return row
    if not tr:
        row["status"] = "rejected: no transfers"
    elif not (ok_id and ok_late):
        row["status"] = "rejected: " + ",".join(
            n for n, ok in [("symbol", ok_id), ("later contract", ok_late)] if not ok)
    elif delta < -WINDOW_BEFORE and not name_match(pname, tok_name):
        row["status"] = "rejected: name (old contract)"
    else:
        row["status"] = "ok" if delta >= -WINDOW_BEFORE else "start mismatch"
        for h in HORIZONS:
            f = features(tr, first + h * 86400)
            if capped and tr[-1]["ts"] < first + h * 86400:
                f = {k: np.nan for k in f}      # download stopped before this horizon
            for k, v in f.items():
                row[f"{k}_d{h}"] = v
    return row


def main(data_path, cand_paths, out):
    """cand_paths: one or more candidate CSVs separated by commas, tried in order."""
    data = pd.read_excel(data_path, sheet_name="data")
    data["key"] = data["Project Title"].astype(str) + "|" + data["Blockchain"].astype(str)
    cands = []
    for cp in cand_paths.split(","):
        c = pd.read_csv(cp)
        c["key"] = c["Project Title"].astype(str) + "|" + c["Blockchain"].astype(str)
        if "rank" in c:
            c = c.sort_values(["key", "rank"])
        cands.append(c.dropna(subset=["address"])[["key", "address"]])
    cand = pd.concat(cands)
    own = data.get("Contract address")
    done = set(pd.read_csv(out)["key"]) if os.path.exists(out) else set()
    for f in filter(None, os.environ.get("DONE_FILES", "").split(",")):   # keys finished elsewhere
        if os.path.exists(f):
            done |= set(pd.read_csv(f)["key"])
    only = os.environ.get("ONLY_CHAINS")          # e.g. "BSC" or "ETH,POLYGON,ARBI"
    only = set(only.split(",")) if only else None
    shard, nshard = map(int, os.environ.get("SHARD", "0/1").split("/"))   # parallel workers
    for i, p in data.iterrows():
        key, chain = p["key"], str(p["Blockchain"]).upper()
        if key in done or (only and chain not in only) or i % nshard != shard:
            continue
        addrs = []
        if own is not None and pd.notna(own.iloc[i]) and str(own.iloc[i]).strip().startswith("0x"):
            addrs.append(str(own.iloc[i]).strip().lower())
        for a in cand.loc[cand["key"] == key, "address"]:
            a = str(a).strip().lower()
            if a.startswith("0x") and a not in addrs:
                addrs.append(a)
        addrs = addrs[:MAX_CANDIDATES]
        base = {"key": key, "Project Title": p["Project Title"], "Blockchain": chain, "class": p["class"],
                "candidates_tried": 0}
        if pd.isna(pd.Timestamp(p["project starting date"])) or not addrs:
            row = {**base, "status": "no address"}
        elif chain not in ESCAN_CHAINS and chain != "BSC":
            row = {**base, "status": "chain not covered", "address": addrs[0]}
        else:
            row = None
            for n, a in enumerate(addrs, 1):
                r = {**base, **check(p, chain, a), "candidates_tried": n}
                if row is None or r["status"] in ("ok", "start mismatch"):
                    row = r
                if r["status"] in ("ok", "start mismatch"):
                    break
        pd.DataFrame([row]).reindex(columns=COLS).to_csv(out, mode="a", header=not os.path.exists(out), index=False)
        print(len(done) + 1, chain, str(p["Project Title"])[:35], row["status"], flush=True)
        done.add(key)


if __name__ == "__main__":
    if len(sys.argv) != 4:
        sys.exit(__doc__)
    main(*sys.argv[1:])
