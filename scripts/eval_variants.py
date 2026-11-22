"""
LROO with reproducible features: archived web footprint and point-in-time on-chain features.

Three questions, all on the temporal hold-out with time-consistent training:

A vs B  (recorded start dates, all projects used in the paper)
    A  the paper's feature sets with Google result counts
    B  the same sets with the Google counts replaced by Internet Archive capture-day counts
    AB both together

C  (on-chain start date = first token transfer; projects whose contract was confirmed)
    C launch         chain + archived footprint up to the on-chain start
    C on-chain d30   chain + transfers, holders and concentration 30 days after the start
                     (no web data at all; every value recomputable from the chain)
    C day 30         C launch + on-chain d30
    C midpoint       C day 30 + archived footprint up to the midpoint
    Start, midpoint and the temporal split all use the on-chain start date. Only projects that
    lasted at least 30 days after their first transfer enter family C, so that no day-30
    statistic is computed after a collapse.

Small legitimate candidates (the 62 rows added in Nov 2026, never used for training) are
scored by every B and C model; the share flagged as rug pull (probability >= 0.5) is their
false-positive rate. They have no Google counts, so A cannot score them.

Usage
    [export TABPFN_TOKEN=...]
    python scripts/eval_variants.py data/TM-RugPull_v2.xlsx results/onchain_source.csv results
"""
import os
import sys
import tempfile

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import lroo_pipeline as L  # noqa: E402

CAT = L.CATEGORICAL
PRICE = ["Max Price Q1", "Max Price Q2"]
ARCH = {"start": ["Archived website days (recorded start)", "Archived X profile days (recorded start)"],
        "mid": ["Archived website days (recorded midpoint)", "Archived X profile days (recorded midpoint)"],
        "start_oc": ["Archived website days (on-chain start)", "Archived X profile days (on-chain start)"],
        "mid_oc": ["Archived website days (on-chain midpoint)", "Archived X profile days (on-chain midpoint)"]}
HZ = {"transfers_d30": "On-chain transfers (day 30)", "holders_d30": "On-chain holders (day 30)",
      "top10_d30": "On-chain top-10 share (day 30)", "hhi_d30": "On-chain HHI (day 30)",
      "gini_d30": "On-chain Gini (day 30)"}
H = pd.Timedelta(days=30)
GROUP, GROUP_OFFSET = np.array([]), 0
quiet = lambda *a, **k: None  # noqa: E731


def factory():
    if not os.environ.get("TABPFN_TOKEN"):
        return None
    from tabpfn_client import TabPFNClassifier, set_access_token
    set_access_token(os.environ["TABPFN_TOKEN"])
    return lambda: TabPFNClassifier(model_path="v2.5_real", random_state=L.SEED)


def evaluate(X, y, tr, te, extra, name, cols, fac):
    """Fit on tr, score te (+ extra rows); metrics on te, false-positive rate on extra."""
    idx = np.concatenate([te, extra]) if len(extra) else te
    res = L.run_experiment(X, y, tr, idx, name, [c for c in X.columns if c not in cols], fac, tune=False, log=print)
    rows, probs = [], {}
    for model, p in res["probs"].items():
        pt = p[:len(te)]
        r = {"Feature set": name, "Model": model, "n train": len(tr), "train rug pulls": int(y[tr].sum()),
             "n test": len(te), "test rug pulls": int(y[te].sum())}
        r.update(L.compute_metrics(y[te], pt))
        r.update(L.bootstrap_ci(y[te], pt))
        if len(extra):
            pe = p[len(te):]
            r["candidates"] = len(extra)
            r["candidates flagged"] = int((pe >= 0.5).sum())
            r["candidate FPR"] = float((pe >= 0.5).mean())
            r["test legit FPR"] = float((pt[y[te] == 0] >= 0.5).mean())
            g = GROUP[extra - GROUP_OFFSET]
            for keys, nice in ((("clean", "close to clean"), "clean"),
                               (("many", "close to many"), "many reports"), (("unclear",), "a few signs")):
                m = np.array([str(v).startswith(keys) for v in g])
                if m.any():
                    r[f"flagged: {nice}"] = f"{int((pe[m] >= 0.5).sum())}/{int(m.sum())}"
                    r[f"mean p: {nice}"] = float(pe[m].mean())
        rows.append(r)
        probs[model] = pt
    return rows, probs


def main(data_path, src_path, out_dir):
    fac = factory()
    X, y, meta, _ = L.load_and_clean(data_path, tempfile.mkdtemp())
    raw = pd.read_excel(data_path, sheet_name="data")
    raw.columns = [" ".join(str(c).split()) for c in raw.columns]
    raw["Excel row"] = raw.index + 2
    is_cand = raw["Candidate suitability"].notna()
    used = raw[raw["Use in training"].astype(str).str.lower() != "no"].reset_index(drop=True)
    assert len(used) == len(X) and (used["Project Title"].values == meta["title"].values).all()
    cand = raw[is_cand].reset_index(drop=True)

    src = pd.read_csv(src_path).drop_duplicates("Excel row").set_index("Excel row")
    src = src[src["status"].astype(str).str.startswith("ok")]

    def extras(frame):
        e = pd.DataFrame(index=frame.index)
        for cols in ARCH.values():
            for c in cols:
                e[c] = pd.to_numeric(frame[c], errors="coerce").values
        for k, nice in HZ.items():
            e[nice] = frame["Excel row"].map(src[k]).astype(float).values
        e["first_transfer"] = pd.to_datetime(frame["Excel row"].map(src["first_transfer"]), errors="coerce").values
        return e

    ex = extras(used)
    Xa = pd.concat([X.reset_index(drop=True), ex.drop(columns="first_transfer")], axis=1)

    # candidate rows: same columns, features that do not exist for them are missing
    exc = extras(cand)
    Xc_rows = pd.DataFrame(np.nan, index=cand.index, columns=X.columns)
    Xc_rows["Blockchain"] = cand["Blockchain"].astype(str).str.upper().replace({"FTM": "FANTOM"}).values
    Xc_rows["Consensus Mechanism"] = cand["Blockchain Type"].values
    Xc_rows = Xc_rows.astype({c: X[c].dtype for c in X.columns if c not in CAT}, errors="ignore")
    Xcand = pd.concat([Xc_rows, exc.drop(columns="first_transfer")], axis=1)
    X_all = pd.concat([Xa, Xcand], ignore_index=True)
    y_all = np.concatenate([y, np.zeros(len(cand), dtype=int)])
    cand_idx = np.arange(len(X), len(X_all))
    global GROUP, GROUP_OFFSET
    GROUP = cand["Manual review, two reviewers, consensus (Nov 2026)"].fillna("").astype(str).values
    GROUP_OFFSET = len(X)

    rows, tests = [], []

    # ---------------- A, B, AB: recorded start dates ----------------
    tr0, te = L.temporal_split(meta)
    cutoff = meta["start"].iloc[te].min()
    launch_g, mid_g = L.TIERS["launch"], L.TIERS["launch"] + L.TIERS["midpoint"]
    sets = {
        "A launch (Google)": (launch_g, "launch", False),
        "B launch (archive)": (CAT + ARCH["start"], "launch", True),
        "AB launch (both)": (launch_g + ARCH["start"], "launch", False),
        "A midpoint (Google)": (mid_g, "midpoint", False),
        "B midpoint (archive)": (CAT + PRICE + ARCH["start"] + ARCH["mid"], "midpoint", True),
        "AB midpoint (both)": (mid_g + ARCH["start"] + ARCH["mid"], "midpoint", False),
    }
    probs = {}
    for name, (cols, tier, score_cand) in sets.items():
        tr = L.admissible_train(tr0, meta, y, cutoff, tier)
        r, p = evaluate(X_all, y_all, tr, te, cand_idx if score_cand else np.array([], dtype=int), name, cols, fac)
        rows += r
        probs[name] = p
        print(f"{name:26s} n_train={len(tr):4d}  " + "  ".join(f"{x['Model'][:8]}={x['ROC AUC']:.3f}" for x in r), flush=True)
    for a, b in [("B launch (archive)", "A launch (Google)"), ("B midpoint (archive)", "A midpoint (Google)"),
                 ("AB launch (both)", "A launch (Google)"), ("AB midpoint (both)", "A midpoint (Google)")]:
        for m in probs[a]:
            if m in probs[b]:
                aa, ab, z, pv = L.delong_test(y[te], probs[a][m], probs[b][m])
                tests.append({"Setting": "recorded start", "With": a, "Against": b, "Model": m, "AUC with": aa,
                              "AUC against": ab, "Delta AUC": aa - ab, "DeLong p": pv})

    # ---------------- C: on-chain start date ----------------
    ft = pd.concat([ex["first_transfer"], exc["first_transfer"]], ignore_index=True)
    end = pd.concat([meta["end"].reset_index(drop=True), pd.to_datetime(cand["project end date"])], ignore_index=True)
    # Family C needs the day-30 statistics to precede the end of the project: for a project that
    # ended within 30 days of its first transfer they would include the collapse itself.
    has = ft.notna() & (ft + H <= end) & X_all[HZ["transfers_d30"]].notna()
    base = np.flatnonzero(has.values[:len(X)])                       # used projects with on-chain data
    cand_c = cand_idx[has.values[len(X):]]
    order = base[np.argsort(ft.iloc[base].values, kind="stable")]
    n_test = int(round(L.TEST_FRACTION * len(order)))
    tr0c, tec = order[:-n_test], order[-n_test:]
    cut_c = ft.iloc[tec].min()
    mid_c = ft + (end - ft) / 2
    obs = {"launch": ft, "d30": ft + H, "mid": pd.concat([mid_c, ft + H], axis=1).max(axis=1)}

    def admissible(tier):
        ok = (obs[tier].iloc[tr0c] < cut_c).values
        ok &= (y_all[tr0c] == 0) | (end.iloc[tr0c] + pd.Timedelta(days=L.LABEL_DELAY_DAYS) <= cut_c).values
        return tr0c[ok]

    hz = list(HZ.values())
    csets = {
        "C launch (archive, on-chain start)": (CAT + ARCH["start_oc"], "launch"),
        "C on-chain d30 only": (CAT + hz, "d30"),
        "C day 30 (archive + on-chain)": (CAT + ARCH["start_oc"] + hz, "d30"),
        "C midpoint (archive + on-chain)": (CAT + ARCH["start_oc"] + ARCH["mid_oc"] + hz, "mid"),
    }
    print(f"\non-chain start: {len(order)} projects ({int(y_all[order].sum())} rug pulls); test {len(tec)} "
          f"({int(y_all[tec].sum())} rug pulls); cutoff {cut_c.date()}; candidates scored {len(cand_c)}", flush=True)
    cprobs = {}
    for name, (cols, tier) in csets.items():
        tr = admissible(tier)
        r, p = evaluate(X_all, y_all, tr, tec, cand_c, name, cols, fac)
        rows += r
        cprobs[name] = p
        print(f"{name:36s} n_train={len(tr):4d}  " + "  ".join(f"{x['Model'][:8]}={x['ROC AUC']:.3f}" for x in r), flush=True)
    for a, b in [("C day 30 (archive + on-chain)", "C launch (archive, on-chain start)"),
                 ("C day 30 (archive + on-chain)", "C on-chain d30 only"),
                 ("C midpoint (archive + on-chain)", "C day 30 (archive + on-chain)")]:
        for m in cprobs[a]:
            if m in cprobs[b]:
                aa, ab, z, pv = L.delong_test(y_all[tec], cprobs[a][m], cprobs[b][m])
                tests.append({"Setting": "on-chain start", "With": a, "Against": b, "Model": m, "AUC with": aa,
                              "AUC against": ab, "Delta AUC": aa - ab, "DeLong p": pv})

    pd.DataFrame(rows).to_csv(os.path.join(out_dir, "variants_eval.csv"), index=False)
    pd.DataFrame(tests).to_csv(os.path.join(out_dir, "variants_tests.csv"), index=False)
    print(pd.DataFrame(tests).round(3).to_string(index=False))


if __name__ == "__main__":
    if len(sys.argv) != 4:
        sys.exit(__doc__)
    main(*sys.argv[1:])
