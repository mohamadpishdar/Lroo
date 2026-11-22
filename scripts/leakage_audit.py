"""
Leakage audit for the LROO temporal hold-out.

The script answers three questions about the 800/200 temporal split:

1. Does any project title appear in both partitions?
2. How well does each feature separate the classes on the test partition when
   used on its own? The start and end dates are included as references. They are
   not model inputs, but any feature that behaves like the end date is suspect,
   because rug pulls end when they collapse while legitimate projects end when
   observation stopped.
3. Is the drop in AUC under time-consistent training caused by the rule itself
   or simply by the smaller training set? For each feature set we draw random
   subsets of the unrestricted training partition with the same size and class
   ratio as the time-consistent training set and compare the two.

TabPFN is included in the matched-subsample control when TABPFN_TOKEN is set (it then
needs N_DRAWS extra calls to the hosted service per feature set).

Usage
    python scripts/leakage_audit.py data/TM-RugPull_v2.xlsx results/
"""
import os
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import lroo_pipeline as L  # noqa: E402

N_DRAWS = 10
SETS = ["midpoint_bounded", "launch_day", "full"]


def quiet(*_args, **_kwargs):
    pass


def factory():
    if not os.environ.get("TABPFN_TOKEN"):
        return None
    from tabpfn_client import TabPFNClassifier, set_access_token
    set_access_token(os.environ["TABPFN_TOKEN"])
    return lambda: TabPFNClassifier(model_path="v2.5_real", random_state=L.SEED)


def main(data_path, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    X, y, meta, _ = L.load_and_clean(data_path, out_dir)
    tr, te = L.temporal_split(meta)
    cutoff = meta["start"].iloc[te].min()

    # 1. Titles shared between partitions
    norm = meta["title"].str.strip().str.lower()
    shared = sorted(set(norm.iloc[te]) & set(norm.iloc[tr]))
    print(f"Titles present in both partitions: {len(shared)}")

    # 2. Single-feature AUC on the test partition
    rows = []
    yte = y[te]
    for c in X.columns:
        if c in L.CATEGORICAL:
            continue
        v = X[c].iloc[te]
        v = v.fillna(v.median())
        a = roc_auc_score(yte, v)
        rows.append({"Feature": c, "Tier": next(t for t, cols in L.TIERS.items() if c in cols),
                     "AUC": max(a, 1 - a), "Higher value means": "rug pull" if a >= 0.5 else "legitimate"})
    for name, col in [("Start date (not a feature)", "start"), ("End date (not a feature)", "end")]:
        a = roc_auc_score(yte, meta[col].iloc[te].astype("int64"))
        rows.append({"Feature": name, "Tier": "reference", "AUC": max(a, 1 - a),
                     "Higher value means": "rug pull" if a >= 0.5 else "legitimate"})
    single = pd.DataFrame(rows).sort_values("AUC", ascending=False)
    single.to_csv(os.path.join(out_dir, "leakage_single_feature_auc.csv"), index=False)
    print(single.round(3).to_string(index=False))

    # 3. Matched-subsample control
    fac = factory()
    rng = np.random.default_rng(L.SEED)
    pos, neg = tr[y[tr] == 1], tr[y[tr] == 0]
    out = []
    for fs in SETS:
        _, tier = L.FEATURE_SET_DEFS[fs]
        strict = L.admissible_train(tr, meta, y, cutoff, tier)
        n_pos, n_neg = int(y[strict].sum()), int((1 - y[strict]).sum())
        if n_pos < 6 or n_neg < 6:
            print(f"{fs}: skipped, {n_pos} rug pulls and {n_neg} legitimate projects admissible")
            continue
        strict_res = L.run_experiment(X, y, strict, te, fs, L.FEATURE_SETS[fs], fac, tune=False, log=quiet)
        draws = {}
        for _ in range(N_DRAWS):
            sub = np.concatenate([rng.choice(pos, n_pos, replace=False), rng.choice(neg, n_neg, replace=False)])
            sub = sub[np.argsort(meta["start"].iloc[sub].values, kind="stable")]
            res = L.run_experiment(X, y, sub, te, fs, L.FEATURE_SETS[fs], fac, tune=False, log=quiet)
            for r in res["rows"]:
                draws.setdefault(r["Model"], []).append(r["ROC AUC"])
        for r in strict_res["rows"]:
            m = r["Model"]
            out.append({"Feature set": fs, "Model": m, "n train": len(strict), "train rug pulls": n_pos,
                        "AUC time-consistent": r["ROC AUC"], "AUC matched random (mean)": np.mean(draws[m]),
                        "AUC matched random (sd)": np.std(draws[m], ddof=1),
                        "Difference": r["ROC AUC"] - np.mean(draws[m])})
    out = pd.DataFrame(out)
    out.to_csv(os.path.join(out_dir, "leakage_matched_subsample.csv"), index=False)
    print(out.round(3).to_string(index=False))


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "data/TM-RugPull_v2.xlsx",
         sys.argv[2] if len(sys.argv) > 2 else "results")
