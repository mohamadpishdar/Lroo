"""
Find candidate contract addresses for TM-RugPull projects through DexScreener and CoinGecko.

For every project on a chain we can read (Ethereum, BNB Smart Chain, Polygon, Arbitrum), the
script searches DexScreener by symbol and by name and keeps tokens on the project's chain whose
symbol matches the symbol in the project title. It also looks the symbol up in CoinGecko's
coin list (downloaded once to data/external/coingecko_coins_list.json).
Every candidate is only a candidate: scripts/onchain_horizons.py confirms or rejects it from
the token's own transfer history.

Output: one row per (project, candidate address), ranked so that the most plausible candidate
comes first (name match first, then the DEX pair created closest to the recorded start date).

Usage
    python scripts/find_addresses_dex.py data/TM-RugPull_v2.xlsx results/address_candidates.csv
"""
import os
import re
import sys
import time
from difflib import SequenceMatcher

import pandas as pd
import requests

CHAINS = {"ETH": ("ethereum", "ethereum"), "BSC": ("bsc", "binance-smart-chain"),
          "POLYGON": ("polygon", "polygon-pos"), "ARBI": ("arbitrum", "arbitrum-one")}
S = requests.Session()
S.headers["User-Agent"] = "TM-RugPull research (KU Leuven)"


def norm(s):
    return re.sub(r"[^a-z0-9]", "", str(s).lower())


def name_sim(a, b):
    a, b = norm(a), norm(b)
    if not a or not b:
        return 0.0
    return 1.0 if ((a in b or b in a) and min(len(a), len(b)) >= 0.5 * max(len(a), len(b))) else SequenceMatcher(None, a, b).ratio()


def split_title(title):
    t = str(title).replace("\xa0", " ").strip()
    m = re.match(r"^(.*?)\s*\(([^()]+)\)\s*$", t)
    return (m.group(1).strip(), m.group(2).strip()) if m else (t, None)


def get(url, params=None, pause=0.25):
    for attempt in range(5):
        try:
            r = S.get(url, params=params, timeout=25)
            if r.status_code == 429:
                time.sleep(10 * (attempt + 1))
                continue
            time.sleep(pause)
            return r.json() if r.ok else None
        except Exception:
            time.sleep(2 ** attempt)
    return None


def dex(q, chain_id):
    js = get("https://api.dexscreener.com/latest/dex/search", {"q": q}) or {}
    out = []
    for p in js.get("pairs") or []:
        if p.get("chainId") != chain_id:
            continue
        for side in ("baseToken", "quoteToken"):
            t = p.get(side) or {}
            out.append({"address": t.get("address", "").lower(), "tok_symbol": t.get("symbol", ""),
                        "tok_name": t.get("name", ""), "pair_created": p.get("pairCreatedAt")})
    return out


GECKO_LIST = os.path.join(os.path.dirname(__file__), "..", "data", "external", "coingecko_coins_list.json")
_gecko = None


def gecko(name, symbol, platform):
    """Local lookup in CoinGecko's coin list (one download, /coins/list?include_platform=true)."""
    global _gecko
    if _gecko is None:
        import json
        _gecko = json.load(open(GECKO_LIST)) if os.path.exists(GECKO_LIST) else []
    out = []
    for c in _gecko:
        a = (c.get("platforms") or {}).get(platform)
        if not a:
            continue
        if (symbol and norm(c["symbol"]) == norm(symbol)) or (not symbol and name_sim(name, c["name"]) >= 0.8):
            out.append({"address": a.lower(), "tok_symbol": c["symbol"], "tok_name": c["name"], "pair_created": None})
    return out


def main(data_path, out_path):
    x = pd.read_excel(data_path, sheet_name="data")
    done = set(pd.read_csv(out_path)["key"]) if os.path.exists(out_path) else set()
    for _, p in x.iterrows():
        chain = str(p["Blockchain"]).upper()
        key = f"{p['Project Title']}|{p['Blockchain']}"
        if chain not in CHAINS or key in done:
            continue
        name, sym = split_title(p["Project Title"])
        dex_chain, platform = CHAINS[chain]
        start = pd.Timestamp(p["project starting date"])
        cands = dex(sym, dex_chain) if sym else []
        cands += dex(name, dex_chain)
        good = [c for c in cands if c["address"].startswith("0x") and
                ((sym and norm(c["tok_symbol"]) == norm(sym)) or (not sym and name_sim(name, c["tok_name"]) >= 0.8))]
        source = "dexscreener"
        g = gecko(name, sym, platform)
        if g:
            good += g
            source = "dexscreener+coingecko" if len(good) > len(g) else "coingecko"
        rows = {}
        for c in good:
            a = c["address"]
            created = pd.to_datetime(c["pair_created"], unit="ms") if c["pair_created"] else pd.NaT
            gap = abs((created - start).days) if pd.notna(created) and pd.notna(start) else 99999
            sim = name_sim(name, c["tok_name"])
            r = rows.setdefault(a, {"key": key, "Project Title": p["Project Title"], "Blockchain": p["Blockchain"],
                                    "address": a, "tok_symbol": c["tok_symbol"], "tok_name": c["tok_name"],
                                    "name_sim": round(sim, 2), "first_pair": created, "gap_days": gap})
            if gap < r["gap_days"]:
                r.update(first_pair=created, gap_days=gap)
        ranked = sorted(rows.values(), key=lambda r: (-(r["name_sim"] >= 0.7), r["gap_days"]))[:5]
        if not ranked:
            ranked = [{"key": key, "Project Title": p["Project Title"], "Blockchain": p["Blockchain"]}]
        for i, r in enumerate(ranked):
            r["rank"] = i + 1
            r["source"] = source if "address" in r else "not found"
        pd.DataFrame(ranked).reindex(columns=["key", "Project Title", "Blockchain", "rank", "address", "tok_symbol",
                                              "tok_name", "name_sim", "first_pair", "gap_days", "source"]) \
            .to_csv(out_path, mode="a", header=not os.path.exists(out_path), index=False)
        done.add(key)
        print(len(done), key[:45], len(rows), flush=True)


if __name__ == "__main__":
    if len(sys.argv) != 3:
        sys.exit(__doc__)
    main(*sys.argv[1:])
