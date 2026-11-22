"""
Sensitivity check: repeat the time-consistent hold-out without the rows that
were flagged for review while loading the data.

Excluded rows
  * end date before start date;
  * start date before the launch of the recorded chain;
  * both rows of every pair that shares name, chain and start date.
Rows whose only issue was a non-numeric cell are kept, because that cell is
simply treated as missing.

The cutoff date and the test period stay the same as in the main analysis, so
the two results can be compared directly. TabPFN is included when the
TABPFN_TOKEN environment variable is set.

A second check repeats the same evaluation with the original end dates of the
legitimate projects (column 'project end date (original)'), i.e. before the
end dates outside the data-collection window were reset.

Usage
    python scripts/sensitivity_flagged.py data/TM-RugPull_v2.xlsx results
"""
import os
import sys
import tempfile

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import lroo_pipeline as L  # noqa: E402

SETS = ["midpoint_bounded", "launch_day"]


def flagged_rows(meta, X, issues):
    rows = set(issues.loc[issues["column"].isin(["dates", "project starting date"]), "row"].astype(int))
    key = pd.DataFrame({"t": meta["title"].str.strip().str.lower(), "c": X["Blockchain"], "s": meta["start"]})
    rows |= set(np.flatnonzero(key.duplicated(keep=False).values))
    # rows whose on-chain activity around the recorded end date does not match the label
    rows |= set(np.flatnonzero(meta["flags"].str.contains("on-chain activity", regex=False).values))
    return rows


def main(data_path, out_dir):
    tmp = tempfile.mkdtemp()
    X, y, meta, _ = L.load_and_clean(data_path, tmp)
    issues = pd.read_csv(os.path.join(tmp, "data_quality_issues.csv"))
    drop = flagged_rows(meta, X, issues)
    tr, te = L.temporal_split(meta)
    cutoff = meta["start"].iloc[te].min()
    keep_tr = np.array([i for i in tr if i not in drop])
    keep_te = np.array([i for i in te if i not in drop])
    print(f"Excluded {len(drop)} rows: {len(tr) - len(keep_tr)} from training, {len(te) - len(keep_te)} from test")

    factory = None
    if os.environ.get("TABPFN_TOKEN"):
        from tabpfn_client import TabPFNClassifier, set_access_token
        set_access_token(os.environ["TABPFN_TOKEN"])
        factory = lambda: TabPFNClassifier(model_path="v2.5_real", random_state=L.SEED)  # noqa: E731

    main_res = pd.read_csv(os.path.join(out_dir, "results_holdout.csv"))
    main_res = main_res[main_res["Split"] == "temporal hold-out, time-consistent training"]
    rows = []
    for fs in SETS:
        _, tier = L.FEATURE_SET_DEFS[fs]
        tr_ok = L.admissible_train(keep_tr, meta, y, cutoff, tier)
        res = L.run_experiment(X, y, tr_ok, keep_te, fs, L.FEATURE_SETS[fs], factory, tune=False,
                               log=lambda *a, **k: None)
        for r in res["rows"]:
            ref = main_res[(main_res["Feature set"] == fs) & (main_res["Model"] == r["Model"])]
            rows.append({"Feature set": fs, "Model": r["Model"], "n excluded": len(drop),
                         "n train": len(tr_ok), "n test": len(keep_te),
                         "test rug pulls": int(y[keep_te].sum()),
                         "AUC without flagged rows": r["ROC AUC"], "CI low": r["ROC AUC low"],
                         "CI high": r["ROC AUC high"],
                         "AUC main analysis": ref["ROC AUC"].iloc[0] if len(ref) else np.nan})
    out = pd.DataFrame(rows)
    out["Difference"] = out["AUC without flagged rows"] - out["AUC main analysis"]
    out.to_csv(os.path.join(out_dir, "sensitivity_flagged_rows.csv"), index=False)
    print(out.round(3).to_string(index=False))

    # ---- original end dates of legitimate projects -------------------------------
    raw = pd.read_excel(data_path, sheet_name="data")
    orig = "project end date (original)"
    if orig in raw.columns:
        n_changed = int(raw[orig].notna().sum())
        raw["project end date"] = raw[orig].where(raw[orig].notna(), raw["project end date"])
        p = os.path.join(tmp, "original_end_dates.xlsx")
        raw.drop(columns=[orig]).to_excel(p, index=False)
        Xo, yo, mo, _ = L.load_and_clean(p, tempfile.mkdtemp())
        tr_o, te_o = L.temporal_split(mo)
        cut_o = mo["start"].iloc[te_o].min()
        rows = []
        for fs in SETS:
            _, tier = L.FEATURE_SET_DEFS[fs]
            tr_ok = L.admissible_train(tr_o, mo, yo, cut_o, tier)
            res = L.run_experiment(Xo, yo, tr_ok, te_o, fs, L.FEATURE_SETS[fs], factory, tune=False,
                                   log=lambda *a, **k: None)
            for r in res["rows"]:
                ref = main_res[(main_res["Feature set"] == fs) & (main_res["Model"] == r["Model"])]
                rows.append({"Feature set": fs, "Model": r["Model"], "n end dates restored": n_changed,
                             "n train": len(tr_ok), "n test": len(te_o),
                             "AUC original end dates": r["ROC AUC"], "CI low": r["ROC AUC low"],
                             "CI high": r["ROC AUC high"],
                             "AUC main analysis": ref["ROC AUC"].iloc[0] if len(ref) else np.nan})
        o = pd.DataFrame(rows)
        o["Difference"] = o["AUC original end dates"] - o["AUC main analysis"]
        o.to_csv(os.path.join(out_dir, "sensitivity_original_end_dates.csv"), index=False)
        print(o.round(3).to_string(index=False))

    # ---- every row, including the rows excluded by a recorded data decision ----------
    os.environ["LROO_USE_ALL_ROWS"] = "1"
    Xa, ya, ma, _ = L.load_and_clean(data_path, tempfile.mkdtemp())
    del os.environ["LROO_USE_ALL_ROWS"]
    tr_a, te_a = L.temporal_split(ma)
    cut_a = ma["start"].iloc[te_a].min()
    rows = []
    for fs in SETS:
        _, tier = L.FEATURE_SET_DEFS[fs]
        tr_ok = L.admissible_train(tr_a, ma, ya, cut_a, tier)
        res = L.run_experiment(Xa, ya, tr_ok, te_a, fs, L.FEATURE_SETS[fs], factory, tune=False,
                               log=lambda *a, **k: None)
        for r in res["rows"]:
            ref = main_res[(main_res["Feature set"] == fs) & (main_res["Model"] == r["Model"])]
            rows.append({"Feature set": fs, "Model": r["Model"], "n rows": len(ya), "n train": len(tr_ok),
                         "n test": len(te_a), "AUC all rows": r["ROC AUC"], "CI low": r["ROC AUC low"],
                         "CI high": r["ROC AUC high"],
                         "AUC main analysis": ref["ROC AUC"].iloc[0] if len(ref) else np.nan})
    a = pd.DataFrame(rows)
    a["Difference"] = a["AUC all rows"] - a["AUC main analysis"]
    a.to_csv(os.path.join(out_dir, "sensitivity_all_rows.csv"), index=False)
    print(a.round(3).to_string(index=False))


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "data/TM-RugPull_v2.xlsx",
         sys.argv[2] if len(sys.argv) > 2 else "results")
