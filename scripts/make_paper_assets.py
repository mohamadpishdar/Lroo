"""
Build every figure, table and in-text number of the LROO paper from the result
files, so that the manuscript cannot drift away from the experiments.

Inputs
    data/TM-RugPull_v2.xlsx          the dataset
    results/                         output of src/lroo_pipeline.py (with TabPFN)
                                     and of scripts/leakage_audit.py
Outputs
    paper/figures/*.pdf, *.png       figures (vector PDF for the paper, PNG for README)
    paper/generated/numbers.tex      one LaTeX macro per number quoted in the text
    paper/generated/tab_*.tex        tables

Usage
    python scripts/make_paper_assets.py data/TM-RugPull_v2.xlsx results paper
"""
import os
import re
import sys
import tempfile

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from sklearn.calibration import calibration_curve  # noqa: E402
from sklearn.metrics import roc_curve  # noqa: E402
from scipy.stats import mannwhitneyu  # noqa: E402
from statsmodels.stats.multitest import multipletests  # noqa: E402

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import lroo_pipeline as L  # noqa: E402

# ---------------------------------------------------------------------------
# Style. Categorical slots follow a fixed order and are validated for colour-
# vision deficiency; every series also gets its own marker or line style.
# ---------------------------------------------------------------------------
INK, INK2, MUTED, GRID = "#0b0b0b", "#52514e", "#898781", "#e1e0d9"
SLOT = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
MODELS = ["Logistic Regression", "Naive Bayes", "SVM", "XGBoost", "LightGBM", "TabPFN"]
TUNED = ["XGBoost (tuned)", "LightGBM (tuned)"]
COLOR = {m: SLOT[i] for i, m in enumerate(MODELS)}
COLOR.update({"XGBoost (tuned)": COLOR["XGBoost"], "LightGBM (tuned)": COLOR["LightGBM"]})
MARKER = {"Logistic Regression": "o", "Naive Bayes": "v", "SVM": "s", "XGBoost": "D",
          "LightGBM": "^", "TabPFN": "P", "XGBoost (tuned)": "d", "LightGBM (tuned)": "<"}
SHORT = {"Logistic Regression": "LR", "Naive Bayes": "NB", "SVM": "SVM", "XGBoost": "XGBoost",
         "LightGBM": "LightGBM", "TabPFN": "TabPFN", "XGBoost (tuned)": "XGBoost (tuned)",
         "LightGBM (tuned)": "LightGBM (tuned)"}
SET_LABEL = {"midpoint_bounded": "Midpoint-bounded", "midpoint_bounded_no_osint": "Midpoint-bounded, no OSINT",
             "launch_day": "Launch day", "fame_baseline": "Fame baseline", "full": "Full",
             "no_osint": "Full, no OSINT", "no_lifetime": "Full, no duration"}
TIER_COLOR = {"launch": SLOT[2], "midpoint": SLOT[3], "end": SLOT[6], "reference": MUTED}

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 8.5, "axes.edgecolor": MUTED, "axes.labelcolor": INK2,
    "axes.titlesize": 9, "axes.titleweight": "bold", "axes.titlecolor": INK, "axes.linewidth": 0.6,
    "xtick.color": MUTED, "ytick.color": MUTED, "xtick.labelcolor": INK2, "ytick.labelcolor": INK2,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.5, "axes.axisbelow": True,
    "axes.spines.top": False, "axes.spines.right": False, "legend.frameon": False,
    "legend.fontsize": 7.5, "figure.dpi": 150, "savefig.bbox": "tight", "pdf.fonttype": 42,
})

TC = "temporal hold-out, time-consistent training"
HO = "temporal hold-out"
RND = "random stratified"


def save(fig, name):
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(FIG, f"{name}.{ext}"), dpi=300 if ext == "png" else None)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Number formatting and macro registry
# ---------------------------------------------------------------------------
MACROS = {}


def f3(x):
    return "--" if pd.isna(x) else f"{x:.3f}"


def fp(p):
    if pd.isna(p):
        return "--"
    if p < 0.001:
        return "$<$0.001"
    return f"{p:.3f}"


def macro(name, value):
    assert re.fullmatch(r"[A-Za-z]+", name), name
    assert name not in MACROS or MACROS[name] == value, f"conflicting macro {name}"
    MACROS[name] = value


DIGITS = {"0": "Zero", "1": "One", "2": "Two", "3": "Three", "4": "Four", "5": "Five"}


def name_code(text):
    """'Max Price Q1' -> 'MaxPriceQOne', usable as part of a LaTeX macro name."""
    text = "".join(DIGITS.get(ch, ch) for ch in text)
    return "".join(w[:1].upper() + w[1:] for w in re.sub(r"[^A-Za-z ]", " ", text).split())


CODE_SET = {"midpoint_bounded": "Main", "midpoint_bounded_no_osint": "MainNoOsint", "launch_day": "Launch",
            "fame_baseline": "Fame", "full": "Full", "no_osint": "FullNoOsint", "no_lifetime": "FullNoDur"}
CODE_MODEL = {"Logistic Regression": "LR", "Naive Bayes": "NB", "SVM": "SVM", "XGBoost": "XGB",
              "LightGBM": "LGBM", "TabPFN": "TabPFN", "XGBoost (tuned)": "XGBt", "LightGBM (tuned)": "LGBMt"}
CODE_SPLIT = {TC: "tc", HO: "ho", RND: "rnd"}
METRICS = {"ROC AUC": "AUC", "ROC AUC low": "AUClo", "ROC AUC high": "AUChi", "Accuracy": "Acc",
           "Balanced Accuracy": "BalAcc", "MCC": "MCC", "F1 (weighted)": "Fone", "Brier": "Brier",
           "Recall (rug pull)": "RecRug", "Precision (rug pull)": "PrecRug",
           "FN": "FN", "FP": "FP", "TN": "TN", "TP": "TP"}


def main(data_path, res_dir, paper_dir):
    global FIG
    FIG = os.path.join(paper_dir, "figures")
    GEN = os.path.join(paper_dir, "generated")
    os.makedirs(FIG, exist_ok=True)
    os.makedirs(GEN, exist_ok=True)

    X, y, meta, _ = L.load_and_clean(data_path, tempfile.mkdtemp())
    tr, te = L.temporal_split(meta)
    cutoff = meta["start"].iloc[te].min()
    yte = y[te]

    res = pd.read_csv(os.path.join(res_dir, "results_holdout.csv"))
    pred = pd.read_csv(os.path.join(res_dir, "test_predictions_holdout.csv"))
    folds = pd.read_csv(os.path.join(res_dir, "results_rolling_folds.csv"))
    rpred = pd.read_csv(os.path.join(res_dir, "test_predictions_rolling.csv"))
    rsum = pd.read_csv(os.path.join(res_dir, "results_rolling_summary.csv"))
    paired = pd.read_csv(os.path.join(res_dir, "paired_tests.csv"))
    abl = pd.read_csv(os.path.join(res_dir, "ablation_deltas.csv"))
    cons = pd.read_csv(os.path.join(res_dir, "consensus_ranking.csv"), index_col=0)
    single = pd.read_csv(os.path.join(res_dir, "leakage_single_feature_auc.csv"))
    matched = pd.read_csv(os.path.join(res_dir, "leakage_matched_subsample.csv"))
    import json
    info = json.load(open(os.path.join(res_dir, "run_info.json")))
    assert (pred["row"].values == te).all(), "prediction rows do not match the temporal split"

    def probs(key):
        return {c.split(" | ", 1)[1]: pred[c].values for c in pred.columns if c.split(" | ")[0] == key}

    def row(split, fs, model):
        r = res[(res["Split"] == split) & (res["Feature set"] == fs) & (res["Model"] == model)]
        return None if r.empty else r.iloc[0]

    # ---------------------------------------------------------------- macros
    macro("NProjects", f"{len(y):,}".replace(",", "{,}"))
    macro("NRug", str(int(y.sum())))
    macro("NLegit", str(int((1 - y).sum())))
    late = meta["start"].dt.year >= 2022
    macro("PctRugLate", f"{100 * y[late.values].mean():.0f}\\%")
    dur = (meta["end"] - meta["start"]).dt.days.where(meta["end"] >= meta["start"])
    macro("MedDurRug", f"{dur[y == 1].median():.0f}")
    macro("MedDurLegit", f"{dur[y == 0].median():.0f}")
    macro("HoldoutCutoff", cutoff.strftime("%-d %B %Y"))
    sp = info["split"]
    macro("TrainRug", str(sp["train_rug_pulls"]))
    macro("NTrain", str(sp["n_train"]))
    global NTRAIN
    NTRAIN = sp["n_train"]
    macro("NTest", str(sp["n_test"]))
    macro("PctTest", f"{100 * sp['n_test'] / (sp['n_train'] + sp['n_test']):.0f}\\%")
    macro("TrainLegit", str(sp["train_legitimate"]))
    macro("TestRug", str(sp["test_rug_pulls"]))
    macro("TestLegit", str(sp["test_legitimate"]))
    macro("TrainEndAfterCutoff", str(sp["train_projects_ending_after_cutoff"]))
    for fs, code in [("midpoint_bounded", "Main"), ("launch_day", "Launch"), ("full", "Full")]:
        r = row(TC, fs, "TabPFN") if row(TC, fs, "TabPFN") is not None else row(TC, fs, "XGBoost")
        if r is None:          # set not estimable under time-consistent training
            continue
        macro(f"TCTrain{code}", str(int(r["n train"])))
        macro("TCTrainRug", str(int(r["train rug pulls"])))
    end_auc = single.loc[single["Feature"].str.startswith("End date"), "AUC"].iloc[0]
    macro("AUCEndDate", f3(end_auc))
    macro("AUCStartDate", f3(single.loc[single["Feature"].str.startswith("Start date"), "AUC"].iloc[0]))
    for _, r in single.iterrows():
        if r["Tier"] == "reference":
            continue
        macro(f"SingleAUC{name_code(r['Feature'])}", f3(r["AUC"]))

    for _, r in res.iterrows():
        if r["Split"] not in CODE_SPLIT or r["Feature set"] not in CODE_SET or r["Model"] not in CODE_MODEL:
            continue
        base = CODE_SPLIT[r["Split"]] + CODE_SET[r["Feature set"]] + CODE_MODEL[r["Model"]]
        for col, mc in METRICS.items():
            v = r[col]
            macro(f"{mc}{base}", str(int(v)) if mc in ("FN", "FP", "TN", "TP") else f3(v))

    for _, r in rsum.iterrows():
        base = CODE_SET[r["Feature set"]] + CODE_MODEL[r["Model"]]
        macro(f"RollMean{base}", f3(r["AUC mean"]))
        macro(f"RollSD{base}", f3(r["AUC sd"]))
        macro(f"RollPooled{base}", f3(r["ROC AUC"]))
        macro(f"RollETH{base}", f3(r["AUC within ETH"]))
        macro(f"RollBSC{base}", f3(r["AUC within BSC"]))
        macro(f"RollMin{base}", f3(r["AUC min"]))
        macro(f"RollMax{base}", f3(r["AUC max"]))
    cut_code = {c: "ABCD"[i] for i, c in enumerate(info["rolling_cutoffs"])}
    for _, r in folds.iterrows():
        macro(f"Roll{cut_code[r['Cutoff']]}{CODE_SET[r['Feature set']]}{CODE_MODEL[r['Model']]}", f3(r["ROC AUC"]))
    for c, g in folds.groupby("Cutoff"):
        for fs, h in g.groupby("Feature set"):
            macro(f"RollTrain{cut_code[c]}{CODE_SET[fs]}", str(int(h["n train"].iloc[0])))
        macro(f"RollTest{cut_code[c]}", str(int(g["n test"].iloc[0])))
        macro(f"RollTestRug{cut_code[c]}", str(int(g["test rug pulls"].iloc[0])))

    for _, r in abl.iterrows():
        tc = "[time-consistent]" in r["Reference set"]
        a = r["Reference set"].replace(" [time-consistent]", "")
        b = r["Compared set"].replace(" [time-consistent]", "")
        base = ("tc" if tc else "ho") + CODE_SET[a] + "vs" + CODE_SET[b] + CODE_MODEL[r["Model"]]
        macro(f"Delta{base}", f"{r['Delta AUC']:+.3f}")
        macro(f"DeltaAbs{base}", f"{abs(r['Delta AUC']):.3f}")
        macro(f"DeltaP{base}", fp(r["DeLong p"]))

    for _, r in paired.iterrows():
        tc = "[time-consistent]" in r["Feature set"]
        fs = r["Feature set"].replace(" [time-consistent]", "")
        base = ("tc" if tc else "ho") + CODE_SET[fs] + CODE_MODEL[r["Compared with"]]
        macro(f"PairRef{('tc' if tc else 'ho') + CODE_SET[fs]}", SHORT[r["Reference"]])
        macro(f"PairDLp{base}", fp(r["DeLong p (Holm)"]))
        macro(f"PairMNp{base}", fp(r["McNemar p (Holm)"]))
        macro(f"PairRefOnly{base}", str(int(r["Only reference correct"])))
        macro(f"PairOtherOnly{base}", str(int(r["Only other correct"])))

    for _, r in matched.iterrows():
        base = CODE_SET[r["Feature set"]] + CODE_MODEL[r["Model"]]
        macro(f"Matched{base}", f3(r["AUC matched random (mean)"]))
        macro(f"MatchedDiff{base}", f"{r['Difference']:+.3f}")
    for fs, g in matched.groupby("Feature set"):
        macro(f"MatchedDiffMean{CODE_SET[fs]}", f"{g['Difference'].mean():+.3f}")

    sens_path = os.path.join(res_dir, "sensitivity_flagged_rows.csv")
    if os.path.exists(sens_path):
        sens = pd.read_csv(sens_path)
        for _, r in sens.iterrows():
            macro(f"Sens{CODE_SET[r['Feature set']]}{CODE_MODEL[r['Model']]}", f3(r["AUC without flagged rows"]))
        macro("SensMaxAbsDiff", f"{sens['Difference'].abs().max():.3f}")
        macro("SensNTest", str(int(sens["n test"].iloc[0])))
        if "n excluded" in sens:
            macro("SensNExcluded", str(int(sens["n excluded"].iloc[0])))
    orig_path = os.path.join(res_dir, "sensitivity_original_end_dates.csv")
    if os.path.exists(orig_path):
        so = pd.read_csv(orig_path)
        for _, r in so.iterrows():
            macro(f"OrigEnd{CODE_SET[r['Feature set']]}{CODE_MODEL[r['Model']]}", f3(r["AUC original end dates"]))
        macro("OrigEndMaxAbsDiff", f"{so['Difference'].abs().max():.3f}")
        macro("OrigEndNRestored", str(int(so["n end dates restored"].iloc[0])))
        macro("OrigEndNTrainMain", str(int(so.loc[so["Feature set"] == "midpoint_bounded", "n train"].iloc[0])))

    all_path = os.path.join(res_dir, "sensitivity_all_rows.csv")
    if os.path.exists(all_path):
        sa = pd.read_csv(all_path)
        macro("AllRowsMaxAbsDiff", f"{sa['Difference'].abs().max():.3f}")

    # ---- data audit (rows kept but excluded, on-chain checks) ---------------------------
    raw = pd.read_excel(data_path, sheet_name="data")
    if "Candidate suitability" in raw.columns:      # small-project candidates are reported separately
        raw = raw[raw["Candidate suitability"].isna()]
    if "Use in training" in raw.columns:
        exc = raw[raw["Use in training"].astype(str).str.lower() == "no"]
        reason = exc["Exclusion reason"].astype(str)
        macro("NRowsFile", str(len(raw)))
        macro("NExcluded", str(len(exc)))
        macro("NExclLabel", str(int(reason.str.startswith("rug-pull label not supported").sum())))
        macro("NExclRepeat", str(int(reason.str.startswith("repeated collection").sum())))
        used = raw[raw["Use in training"].astype(str).str.lower() != "no"]
        fl = used["Quality flags"].astype(str)
        macro("NFlagOnchain", str(int(fl.str.contains("activity continued", regex=False).sum())))
        macro("NTxArtifact", str(int(used["the number of Transactions"].eq(25).sum())))
    hz_path = os.path.join(res_dir, "onchain_horizons.csv")
    if os.path.exists(hz_path):
        hz = pd.read_csv(hz_path).drop_duplicates("key")
        okk = hz[hz["status"].isin(["ok", "start mismatch"])]
        macro("NConfirmed", str(len(okk)))
        macro("NConfirmedRug", str(int(okk["class"].eq("scam").sum())))
        macro("NConfirmedLegit", str(int(okk["class"].eq("normal").sum())))
        macro("NRejectedCount", str(int(hz["status"].eq("rejected: count").sum())))
    for f, tag in [("collapse_audit_eth.csv", "Eth"), ("collapse_audit_bsc.csv", "Bsc")]:
        p = os.path.join(res_dir, f)
        if os.path.exists(p) and os.path.exists(hz_path):
            a = pd.read_csv(p)
            a = a[a["key"].isin(set(okk["key"])) & a["class"].eq("scam")]
            macro(f"AuditRug{tag}", str(len(a)))
            macro(f"AuditRugActive{tag}", str(int(a["flag"].astype(str).str.startswith("activity").sum())))
    lq_path = os.path.join(res_dir, "liquidity_check.csv")
    if os.path.exists(lq_path):
        lq = pd.read_csv(lq_path)
        r = lq[lq["class"].eq("scam")]
        for v, name in [("drained", "Drained"), ("frozen (no trading)", "Frozen"), ("still trading", "Trading"),
                        ("no V2 pool", "NoPool")]:
            macro(f"Liq{name}", str(int(r["verdict"].eq(v).sum())))
        macro("LiqMeasured", str(int(r["verdict"].isin(["drained", "frozen (no trading)", "still trading"]).sum())))

    # Low-visibility subgroup: test projects whose launch-day X-profile hits do not
    # exceed the upper quartile of rug pulls in the training partition.
    xcol = "Search Hits: X Profile (Launch Day)"
    thr = X[xcol].iloc[tr][y[tr] == 1].quantile(0.75)
    pm_ = pred["midpoint_bounded [time-consistent] | TabPFN"].values
    low = X[xcol].iloc[te].values <= thr
    leg = yte == 0
    macro("LowVisThreshold", f"{thr:.0f}")
    macro("LowVisLegitAll", str(int(((X[xcol] <= thr) & (y == 0)).sum())))
    macro("LowVisN", str(int(low.sum())))
    macro("LowVisLegitN", str(int((low & leg).sum())))
    macro("LowVisLegitFP", str(int((pm_[low & leg] > 0.5).sum())))
    macro("HighVisLegitN", str(int((~low & leg).sum())))
    macro("HighVisLegitFP", str(int((pm_[~low & leg] > 0.5).sum())))
    from sklearn.metrics import roc_auc_score
    yl, pl = yte[low], pm_[low]
    macro("LowVisAUC", f3(roc_auc_score(yl, pl)))
    rng = np.random.default_rng(L.SEED)
    boots = []
    for _ in range(1000):
        i = rng.integers(0, len(yl), len(yl))
        if len(set(yl[i])) == 2:
            boots.append(roc_auc_score(yl[i], pl[i]))
    macro("LowVisAUClo", f3(np.percentile(boots, 2.5)))
    macro("LowVisAUChi", f3(np.percentile(boots, 97.5)))

    cons = cons.sort_values("Consensus rank")
    for i, (feat, r) in enumerate(cons.iterrows(), 1):
        macro(f"Rank{name_code(feat)}", f"{r['Consensus rank']:.2f}")

    # ---------------------------------------------------------------- tables
    # Composition by chain
    chain_name = {"ETH": "Ethereum", "BSC": "BNB Smart Chain", "POLYGON": "Polygon", "ARBI": "Arbitrum",
                  "FANTOM": "Fantom", "BASE": "Base", "CRONO": "Cronos", "SNOW": "Avalanche"}
    ct = pd.crosstab(X["Blockchain"], y).rename(columns={0: "leg", 1: "rug"})
    ct["n"] = ct.sum(axis=1)
    ct = ct.sort_values("n", ascending=False)
    big = ct.loc[[c for c in ("ETH", "BSC") if c in ct.index], "n"].sum()
    macro("PctEthBsc", f"{100 * big / len(y):.0f}\\%")
    macro("NOtherChains", str(int(len(y) - big)))
    lines = [r"\begin{tabular}{@{}lrrrr@{}}", r"\toprule",
             r"Blockchain & Rug pulls & Legitimate & Total & Share \\", r"\midrule"]
    for k, r in ct.iterrows():
        lines.append(f"{chain_name.get(k, k)} & {r['rug']} & {r['leg']} & {r['n']} & {100 * r['n'] / len(y):.1f}\\% \\\\")
    lines += [r"\midrule", f"Total & {int(y.sum())} & {int((1 - y).sum())} & {len(y):,} & 100\\% \\\\".replace(",", "{,}"),
              r"\bottomrule", r"\end{tabular}"]
    open(os.path.join(GEN, "tab_composition.tex"), "w").write("\n".join(lines) + "\n")

    # Descriptive statistics, full dataset
    feats = ["Project Duration (days)", "Max Price Q1", "Max Price Q2", "Transaction Count",
             "Token Concentration Ratio", "Search Hits: Title (Launch Day)", "Search Hits: Website (Launch Day)",
             "Search Hits: X Profile (Launch Day)", "Search Hits: X Profile (Midpoint)"]
    tiers = {c: t for t, cols in L.TIERS.items() for c in cols}
    rows = []
    for c in feats:
        a, b = X.loc[y == 1, c].dropna(), X.loc[y == 0, c].dropna()
        rows.append((c, tiers[c], a.median(), b.median(), mannwhitneyu(a, b)[1]))
    pbh = multipletests([r[4] for r in rows], method="fdr_bh")[1]

    def num(v):
        if v == 0:
            return "0"
        if abs(v) >= 1e5 or abs(v) < 1e-2:
            m, e = f"{v:.2e}".split("e")
            return f"${m}\\times10^{{{int(e)}}}$"
        if abs(v) >= 100:
            return f"{v:,.0f}".replace(",", "{,}")
        return f"{v:.3g}"
    lines = [r"\begin{tabular}{@{}llrrr@{}}", r"\toprule",
             r"Feature & Tier & Median, rug pull & Median, legitimate & $p$ (BH) \\", r"\midrule"]
    for (c, t, ma, mb, _), p in zip(rows, pbh):
        lines.append(f"{c} & {t} & {num(ma)} & {num(mb)} & {fp(p)} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    open(os.path.join(GEN, "tab_descriptive.tex"), "w").write("\n".join(lines) + "\n")
    macro("MedTitleLaunchRug", num(rows[5][2])); macro("MedTitleLaunchLegit", num(rows[5][3]))
    macro("MedXLaunchRug", num(rows[7][2])); macro("MedXLaunchLegit", num(rows[7][3]))

    # Main results: time-consistent hold-out, three tiers
    def results_table(split, sets, models, path, label_tier=True):
        lines = [r"\begin{tabular}{@{}llcccccccc@{}}", r"\toprule",
                 r"Feature set & Model & ROC AUC [95\% CI] & Bal.\ acc. & MCC & Acc. & F1$_w$ & Brier & FN & FP \\",
                 r"\midrule"]
        for j, fs in enumerate(sets):
            sub = [row(split, fs, m) for m in models]
            sub = [r for r in sub if r is not None]
            label = SET_LABEL[fs]
            if not sub and split == TC:      # not estimable under time-consistent training
                sub = [r for r in (row(HO, fs, m) for m in models) if r is not None]
                label = SET_LABEL[fs] + r"$^{\dagger}$"
            best = max(r["ROC AUC"] for r in sub)
            for i, r in enumerate(sub):
                auc = f3(r["ROC AUC"])
                if abs(r["ROC AUC"] - best) < 1e-12:
                    auc = f"\\textbf{{{auc}}}"
                head = (f"\\multirow{{{len(sub)}}}{{*}}{{\\shortstack[l]{{{label}\\\\"
                        f"\\footnotesize $n_{{\\mathrm{{train}}}}={int(r['n train']) if not pd.isna(r['n train']) else NTRAIN}$}}}}"
                        if i == 0 else "")
                lines.append(f"{head} & {SHORT[r['Model']]} & {auc} [{f3(r['ROC AUC low'])}, {f3(r['ROC AUC high'])}] & "
                             f"{f3(r['Balanced Accuracy'])} & {f3(r['MCC'])} & {f3(r['Accuracy'])} & "
                             f"{f3(r['F1 (weighted)'])} & {f3(r['Brier'])} & {int(r['FN'])} & {int(r['FP'])} \\\\")
            if j < len(sets) - 1:
                lines.append(r"\midrule")
        lines += [r"\bottomrule", r"\end{tabular}"]
        open(os.path.join(GEN, path), "w").write("\n".join(lines) + "\n")

    results_table(TC, ["midpoint_bounded", "launch_day", "full"], MODELS, "tab_main_results.tex")

    # Unrestricted hold-out, AUC matrix (sets x models incl. tuned)
    sets = ["midpoint_bounded", "midpoint_bounded_no_osint", "launch_day", "fame_baseline",
            "full", "no_osint", "no_lifetime"]
    mods = MODELS[:5] + TUNED + ["TabPFN"]
    lines = [r"\begin{tabular}{@{}l" + "c" * len(mods) + "@{}}", r"\toprule",
             "Feature set & " + " & ".join(SHORT[m].replace(" (tuned)", r"\textsuperscript{t}") for m in mods) + r" \\",
             r"\midrule"]
    for fs in sets:
        vals = [row(HO, fs, m) for m in mods]
        best = max(v["ROC AUC"] for v in vals if v is not None)
        cells = []
        for v in vals:
            if v is None:
                cells.append("--")
            else:
                s = f3(v["ROC AUC"])
                cells.append(f"\\textbf{{{s}}}" if abs(v["ROC AUC"] - best) < 1e-12 else s)
        lines.append(f"{SET_LABEL[fs]} & " + " & ".join(cells) + r" \\")
        if fs == "fame_baseline":
            lines.append(r"\midrule")
    rr = [row(RND, "full", m) for m in mods]
    lines += [r"\midrule", "Full, random split & " + " & ".join("--" if v is None else f3(v["ROC AUC"]) for v in rr) + r" \\",
              r"\bottomrule", r"\end{tabular}"]
    open(os.path.join(GEN, "tab_unrestricted.tex"), "w").write("\n".join(lines) + "\n")

    # Paired tests, time-consistent main set
    p = paired[paired["Feature set"] == "midpoint_bounded [time-consistent]"]
    ref = p["Reference"].iloc[0]
    lines = [r"\begin{tabular}{@{}lccccc@{}}", r"\toprule",
             f"Compared with {SHORT[ref]} & AUC & $\\Delta$AUC & DeLong $p$ (Holm) & Discordant pairs & McNemar $p$ (Holm) \\\\",
             r"\midrule"]
    for _, r in p.sort_values("AUC other", ascending=False).iterrows():
        lines.append(f"{SHORT[r['Compared with']]} & {f3(r['AUC other'])} & {r['AUC reference'] - r['AUC other']:+.3f} & "
                     f"{fp(r['DeLong p (Holm)'])} & {int(r['Only reference correct'])} / {int(r['Only other correct'])} & "
                     f"{fp(r['McNemar p (Holm)'])} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    open(os.path.join(GEN, "tab_paired.tex"), "w").write("\n".join(lines) + "\n")
    macro("PairRefMain", SHORT[ref])

    # Rolling-origin summary
    rs_sets = ["midpoint_bounded", "midpoint_bounded_no_osint", "launch_day", "fame_baseline", "full"]
    lines = [r"\begin{tabular}{@{}l" + "c" * len(MODELS) + "@{}}", r"\toprule",
             "Feature set & " + " & ".join(SHORT[m] for m in MODELS) + r" \\", r"\midrule"]
    for fs in rs_sets:
        g = rsum[rsum["Feature set"] == fs].set_index("Model")
        if g.empty:
            lines.append(f"{SET_LABEL[fs]} & \\multicolumn{{{len(MODELS)}}}{{c}}{{\\emph{{not estimable (no admissible legitimate project)}}}} \\\\")
            continue
        best = g["AUC mean"].max()
        cells = []
        for m in MODELS:
            if m not in g.index:
                cells.append("--"); continue
            s = f"{g.loc[m, 'AUC mean']:.3f} $\\pm$ {g.loc[m, 'AUC sd']:.3f}"
            cells.append(f"\\textbf{{{s}}}" if abs(g.loc[m, "AUC mean"] - best) < 1e-12 else s)
        nf = int(g["Folds"].iloc[0])
        lines.append(f"{SET_LABEL[fs]}" + (f" ({nf} folds)" if nf != 4 else "") + " & " + " & ".join(cells) + r" \\")
    lines += [r"\midrule", r"\multicolumn{" + str(len(MODELS) + 1) + r"}{@{}l}{\emph{Pooled out-of-sample predictions, midpoint-bounded set}}\\"]
    g = rsum[rsum["Feature set"] == "midpoint_bounded"].set_index("Model")
    for lab, col in [("Pooled AUC", "ROC AUC"), ("Within Ethereum", "AUC within ETH"), ("Within BNB Smart Chain", "AUC within BSC")]:
        lines.append(lab + " & " + " & ".join(f3(g.loc[m, col]) if m in g.index else "--" for m in MODELS) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    open(os.path.join(GEN, "tab_rolling.tex"), "w").write("\n".join(lines) + "\n")

    # Time-consistent vs unrestricted vs matched
    lines = [r"\begin{tabular}{@{}llcccc@{}}", r"\toprule",
             r"Feature set & Model & Unrestricted & Time-consistent & Matched random & TC $-$ matched \\",
             f" & & ($n={NTRAIN}$) & & (mean of 10) & \\\\", r"\midrule"]
    ctl_sets = [fs for fs in ["midpoint_bounded", "launch_day", "full"] if row(TC, fs, "SVM") is not None]
    for j, fs in enumerate(ctl_sets):
        g = matched[matched["Feature set"] == fs].set_index("Model")
        for i, m in enumerate(MODELS):
            a, b = row(HO, fs, m), row(TC, fs, m)
            if b is None:
                continue
            head = (f"\\multirow{{{len(MODELS)}}}{{*}}{{\\shortstack[l]{{{SET_LABEL[fs]}\\\\\\footnotesize "
                    f"$n_{{\\mathrm{{TC}}}}={int(b['n train'])}$}}}}" if i == 0 else "")
            mt = f3(g.loc[m, "AUC matched random (mean)"]) if m in g.index else "--"
            dd = f"{g.loc[m, 'Difference']:+.3f}" if m in g.index else "--"
            lines.append(f"{head} & {SHORT[m]} & {f3(a['ROC AUC'])} & {f3(b['ROC AUC'])} & {mt} & {dd} \\\\")
        if j < len(ctl_sets) - 1:
            lines.append(r"\midrule")
    lines += [r"\bottomrule", r"\end{tabular}"]
    open(os.path.join(GEN, "tab_tc_control.tex"), "w").write("\n".join(lines) + "\n")

    # Hyperparameters (appendix)
    tp = info["tuned_params"]["midpoint_bounded"]
    lines = [r"\begin{tabular}{@{}llll@{}}", r"\toprule", r"Model & Hyperparameter & Search space & Selected \\", r"\midrule"]
    for name, (_, space) in L.TUNING_SPACES.items():
        keys = list(space)
        for i, k in enumerate(keys):
            kk = k.replace("clf__", "")
            sel = tp[name].get(kk, "--")
            head = name if i == 0 else ""
            lines.append(f"{head} & \\texttt{{{kk.replace('_', chr(92) + '_')}}} & "
                         f"\\{{{', '.join(str(v) for v in space[k])}\\}} & {sel} \\\\")
        lines.append(f" & \\emph{{CV ROC AUC}} & & {tp[name]['cv_roc_auc']:.3f} \\\\")
        lines.append(r"\midrule" if name != list(L.TUNING_SPACES)[-1] else r"\bottomrule")
    lines.append(r"\end{tabular}")
    open(os.path.join(GEN, "tab_hyper.tex"), "w").write("\n".join(lines) + "\n")

    # ---------------------------------------------------------------- figures
    # 1. Timeline
    yr = meta["start"].dt.year.clip(lower=2015)
    ctab = pd.crosstab(yr, y).reindex(range(2015, 2026), fill_value=0)
    fig, ax = plt.subplots(figsize=(6.8, 2.6))
    xs = np.arange(len(ctab))
    ax.bar(xs, ctab[0], width=0.72, color=SLOT[0], label="Legitimate", edgecolor="white", linewidth=1.0)
    ax.bar(xs, ctab[1], width=0.72, bottom=ctab[0], color=SLOT[1], label="Rug pull", edgecolor="white", linewidth=1.0)
    ax.set_xticks(xs, ["≤2015"] + [str(v) for v in ctab.index[1:]])
    ax.set_ylabel("Projects launched")
    ax.grid(axis="x", visible=False)

    def xpos(ts):
        return (ts.year - 2015) + (ts.dayofyear - 1) / 365.25 - 0.5
    for c in info["rolling_cutoffs"]:
        ax.axvline(xpos(pd.Timestamp(c)), color=INK2, lw=0.8, ls=(0, (3, 2)))
    ax.axvline(xpos(cutoff), color=INK, lw=1.2)
    ax.annotate("hold-out starts", xy=(xpos(cutoff), ctab.sum(axis=1).max() * 1.06),
                xytext=(xpos(cutoff) + 0.25, ctab.sum(axis=1).max() * 1.06), fontsize=7, color=INK, va="center",
                arrowprops=dict(arrowstyle="-", color=INK, lw=0.6))
    for i, v in enumerate(ctab.sum(axis=1)):
        if v:
            ax.text(i, v + 4, str(v), ha="center", fontsize=6.5, color=INK2)
    ax.set_ylim(0, ctab.sum(axis=1).max() * 1.14)
    ax.legend(loc="upper left")
    save(fig, "fig_timeline")

    # 2. ROC, time-consistent main set
    pm = probs("midpoint_bounded [time-consistent]")
    fig, ax = plt.subplots(figsize=(3.4, 3.3))
    ax.plot([0, 1], [0, 1], color=MUTED, lw=0.8, ls=(0, (3, 2)))
    for m in MODELS:
        if m not in pm:
            continue
        fpr, tpr, _ = roc_curve(yte, pm[m])
        auc = row(TC, "midpoint_bounded", m)["ROC AUC"]
        ax.plot(fpr, tpr, color=COLOR[m], lw=2 if m == "TabPFN" else 1.4, label=f"{SHORT[m]} ({auc:.3f})")
    ax.set_xlabel("False positive rate"); ax.set_ylabel("True positive rate")
    ax.set_xlim(-0.01, 1.01); ax.set_ylim(-0.01, 1.01); ax.set_aspect("equal")
    ax.legend(loc="lower right", title="Model (AUC)", title_fontsize=7.5)
    save(fig, "fig_roc_main")

    # 3. AUC with CIs across tiers (small multiples)
    tc_sets = ["launch_day", "midpoint_bounded", "full"]
    fig, axes = plt.subplots(1, 3, figsize=(6.8, 2.5), sharey=True, sharex=True)
    for ax, fs in zip(axes, tc_sets):
        sp_ = TC if row(TC, fs, "SVM") is not None else HO
        for i, m in enumerate(MODELS[::-1]):
            r = row(sp_, fs, m)
            if r is None:
                continue
            ax.plot([r["ROC AUC low"], r["ROC AUC high"]], [i, i], color=COLOR[m], lw=2, solid_capstyle="round")
            ax.plot(r["ROC AUC"], i, marker=MARKER[m], color=COLOR[m], ms=6, mec="white", mew=0.8)
        rr = row(sp_, fs, "SVM")
        n = int(rr["n train"]) if not pd.isna(rr["n train"]) else NTRAIN
        ax.set_title(f"{SET_LABEL[fs]}" + (" (unrestricted)" if sp_ == HO else "") + f"\n(n train = {n})", fontsize=8.5)
        ax.set_yticks(range(len(MODELS)), [SHORT[m] for m in MODELS[::-1]])
        ax.grid(axis="y", visible=False)
        ax.set_xlim(0.55, 1.0)
    axes[1].set_xlabel("ROC AUC with bootstrap 95% CI (time-consistent training, temporal hold-out)")
    fig.tight_layout()
    save(fig, "fig_auc_tiers")

    # 4. Rolling-origin, small multiples
    rs_plot = ["midpoint_bounded", "midpoint_bounded_no_osint", "launch_day", "fame_baseline"]
    fig, axes = plt.subplots(1, 4, figsize=(7.0, 2.4), sharey=True)
    labels_c = [str(pd.Timestamp(c).year) for c in info["rolling_cutoffs"]]
    for ax, fs in zip(axes, rs_plot):
        g = folds[folds["Feature set"] == fs]
        for m in MODELS:
            h = g[g["Model"] == m].sort_values("Cutoff")
            if h.empty:
                continue
            xi = [info["rolling_cutoffs"].index(c) for c in h["Cutoff"]]
            ax.plot(xi, h["ROC AUC"], color=COLOR[m], marker=MARKER[m], ms=4, lw=1.4 if m != "TabPFN" else 2,
                    mec="white", mew=0.5)
        ax.axhline(0.5, color=MUTED, lw=0.7, ls=(0, (3, 2)))
        ax.set_xticks(range(4), labels_c)
        ax.set_title(SET_LABEL[fs].replace(", ", ",\n"), fontsize=8)
        ax.set_ylim(0.4, 1.0)
        ax.grid(axis="x", visible=False)
    axes[0].set_ylabel("ROC AUC on the following year")
    fig.text(0.5, -0.02, "Training cutoff (1 January)", ha="center", color=INK2)
    handles = [Line2D([], [], color=COLOR[m], marker=MARKER[m], lw=1.4, ms=4, label=SHORT[m]) for m in MODELS]
    fig.legend(handles=handles, loc="upper center", ncol=6, bbox_to_anchor=(0.5, 1.10))
    fig.tight_layout()
    save(fig, "fig_rolling")

    # 5. Ablations under time-consistent training
    comps = [("midpoint_bounded", "midpoint_bounded_no_osint", "Remove OSINT\n(midpoint-bounded)"),
             ("midpoint_bounded", "fame_baseline", "Keep only name and\nwebsite hits"),
             ("midpoint_bounded", "launch_day", "Drop midpoint\nfeatures"),
             ("full", "midpoint_bounded", "Add end-tier\nfeatures")]
    # keep only comparisons that exist under time-consistent training (the full set does not)
    comps = [c for c in comps if ((abl["Reference set"] == f"{c[0]} [time-consistent]")
                                  & (abl["Compared set"] == f"{c[1]} [time-consistent]")).any()]
    fig, ax = plt.subplots(figsize=(6.8, 2.8))
    width = 0.13
    for k, (a, b, lab) in enumerate(comps):
        g = abl[(abl["Reference set"] == f"{a} [time-consistent]") & (abl["Compared set"] == f"{b} [time-consistent]")]
        for i, m in enumerate(MODELS):
            r = g[g["Model"] == m]
            if r.empty:
                continue
            r = r.iloc[0]
            d = r["Delta AUC"] if a == "full" else -r["Delta AUC"]   # sign: effect of the change on AUC
            xx = k + (i - 2.5) * width
            sig = r["DeLong p"] < 0.05
            ax.plot([xx, xx], [0, d], color=COLOR[m], lw=1.6)
            ax.plot(xx, d, marker=MARKER[m], ms=5.5, color=COLOR[m] if sig else "white", mec=COLOR[m], mew=1.2)
    ax.axhline(0, color=INK2, lw=0.8)
    ax.set_xticks(range(len(comps)), [c[2] for c in comps])
    ax.set_ylabel("Change in ROC AUC")
    ax.grid(axis="x", visible=False)
    handles = [Line2D([], [], color=COLOR[m], marker=MARKER[m], lw=0, ms=5, label=SHORT[m]) for m in MODELS]
    handles += [Line2D([], [], color=INK2, marker="o", lw=0, ms=5, mfc="white", label="DeLong p ≥ 0.05")]
    ax.legend(handles=handles, ncol=7, loc="upper center", bbox_to_anchor=(0.5, 1.16), fontsize=7)
    save(fig, "fig_ablation")

    # 6. Consensus importance
    c = cons.sort_values("Consensus rank", ascending=False)
    fig, ax = plt.subplots(figsize=(4.6, 3.0))
    tier_of = {f: t for t, cols in L.TIERS.items() for f in cols}
    for i, (feat, r) in enumerate(c.iterrows()):
        col = SLOT[1] if r["OSINT"] else (SLOT[0] if feat not in L.CATEGORICAL else MUTED)
        ax.barh(i, len(c) + 1 - r["Consensus rank"], color=col, height=0.66)
        ax.text(len(c) + 1 - r["Consensus rank"] + 0.1, i, f"{r['Consensus rank']:.2f}", va="center",
                fontsize=6.5, color=INK2)
    ax.set_yticks(range(len(c)), [f"{f} [{tier_of[f][0].upper()}]" for f in c.index])
    ax.set_xlabel("Importance score (11 − mean rank across four methods)")
    ax.grid(axis="y", visible=False)
    ax.set_xlim(0, len(c) + 1.4)
    ax.legend(handles=[plt.Rectangle((0, 0), 1, 1, color=SLOT[1], label="OSINT"),
                       plt.Rectangle((0, 0), 1, 1, color=SLOT[0], label="On-chain"),
                       plt.Rectangle((0, 0), 1, 1, color=MUTED, label="Context")], loc="lower right")
    save(fig, "fig_importance")

    # 7. Calibration, time-consistent main set
    fig, ax = plt.subplots(figsize=(3.4, 3.3))
    ax.plot([0, 1], [0, 1], color=MUTED, lw=0.8, ls=(0, (3, 2)))
    for m in ["Logistic Regression", "SVM", "XGBoost", "TabPFN"]:
        if m not in pm:
            continue
        fr, mp = calibration_curve(yte, pm[m], n_bins=8, strategy="quantile")
        br = row(TC, "midpoint_bounded", m)["Brier"]
        ax.plot(mp, fr, color=COLOR[m], marker=MARKER[m], ms=4, lw=1.4, label=f"{SHORT[m]} ({br:.3f})")
    ax.set_xlabel("Mean predicted probability"); ax.set_ylabel("Observed share of rug pulls")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1.02); ax.set_aspect("equal")
    ax.legend(loc="lower right", title="Model (Brier)", title_fontsize=7.5)
    save(fig, "fig_calibration")

    # 8. Single-feature AUC (leakage proxies)
    s = single.sort_values("AUC")
    fig, ax = plt.subplots(figsize=(4.8, 3.6))
    for i, (_, r) in enumerate(s.iterrows()):
        ax.barh(i, r["AUC"] - 0.5, left=0.5, color=TIER_COLOR[r["Tier"]], height=0.66,
                hatch="////" if r["Tier"] == "reference" else None, edgecolor="white" if r["Tier"] != "reference" else INK2,
                linewidth=0.0 if r["Tier"] != "reference" else 0.5)
        ax.text(r["AUC"] + 0.004, i, f"{r['AUC']:.3f}", va="center", fontsize=6.5, color=INK2)
    ax.set_yticks(range(len(s)), s["Feature"])
    ax.set_xlim(0.5, 1.0)
    ax.set_xlabel("Single-feature ROC AUC on the test partition")
    ax.grid(axis="y", visible=False)
    ax.legend(handles=[plt.Rectangle((0, 0), 1, 1, color=TIER_COLOR[t], label=t) for t in ["launch", "midpoint", "end"]]
              + [plt.Rectangle((0, 0), 1, 1, facecolor=MUTED, hatch="////", edgecolor=INK2, label="date (not a feature)")],
              loc="lower right", title="Tier", title_fontsize=7.5)
    save(fig, "fig_proxy")

    # ---------------------------------------------------------------- write macros
    with open(os.path.join(GEN, "numbers.tex"), "w") as f:
        f.write("% Generated by scripts/make_paper_assets.py. Do not edit by hand.\n")
        for k in sorted(MACROS):
            f.write(f"\\newcommand{{\\{k}}}{{{MACROS[k]}}}\n")
    print(f"{len(MACROS)} macros, tables and figures written to {paper_dir}")


if __name__ == "__main__":
    main(*(sys.argv[1:4] if len(sys.argv) > 3 else ["data/TM-RugPull_v2.xlsx", "results", "paper"]))
