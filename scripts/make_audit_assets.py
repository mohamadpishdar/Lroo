"""
Numbers and tables for the data audit and the reproducible-feature experiments.

Inputs
    data/TM-RugPull_v2.xlsx                 dataset (sheet "data")
    results/onchain_source.csv              on-chain evidence per row
    results/start_date_audit.csv            recorded start versus on-chain dates
    results/google_calibration.csv          recorded Google counts versus a re-run today
    results/wayback/wayback_counts*.csv     archive counts and the repeatability re-run
    results/variants_eval.csv, variants_tests.csv   output of scripts/eval_variants.py
Outputs
    paper/generated/audit_numbers.tex       one macro per number
    paper/generated/tab_variants.tex        AUC of every model on every feature set
    paper/generated/tab_candidates.tex      flag rates on the small legitimate projects

Usage
    python scripts/make_audit_assets.py data/TM-RugPull_v2.xlsx results paper
"""
import os
import sys

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

MODELS = [("Logistic Regression", "LR"), ("Naive Bayes", "NB"), ("SVM", "SVM"), ("XGBoost", "XGB"),
          ("LightGBM", "LGBM"), ("TabPFN", "TabPFN")]
SETS = [("A launch (Google)", "ALaunch", "A", "Launch, Google search hits"),
        ("B launch (archive)", "BLaunch", "B", "Launch, archive captures"),
        ("AB launch (both)", "ABLaunch", "AB", "Launch, both"),
        ("A midpoint (Google)", "AMid", "A", "Midpoint, Google search hits"),
        ("B midpoint (archive)", "BMid", "B", "Midpoint, archive captures"),
        ("AB midpoint (both)", "ABMid", "AB", "Midpoint, both"),
        ("C launch (archive, on-chain start)", "CLaunch", "C", "Launch, archive captures"),
        ("C on-chain d30 only", "COnchain", "C", "Day 30, on-chain only"),
        ("C day 30 (archive + on-chain)", "CDay", "C", "Day 30, archive and on-chain"),
        ("C midpoint (archive + on-chain)", "CMid", "C", "Midpoint, archive and on-chain")]
CODE = {s: c for s, c, _, _ in SETS}
MCODE = dict(MODELS)
out = []


def macro(name, value):
    out.append(f"\\newcommand{{\\{name}}}{{{value}}}")


def pct(x):
    return f"{100 * x:.0f}\\%"


def main(data_path, res, paper):
    d = pd.read_excel(data_path, sheet_name="data")
    d.columns = [" ".join(str(c).split()) for c in d.columns]
    d["Excel row"] = d.index + 2
    cand = d["Candidate suitability"].notna()
    o, c = d[~cand], d[cand]
    use = o["Use in training"].astype(str).str.lower() != "no"
    macro("NOrig", len(o))
    macro("NCand", len(c))
    macro("NUsed", int(use.sum()))

    # ---- contract addresses and on-chain confirmation ----------------------------------
    src = o["Address source"].astype(str)
    macro("NAddrLink", int(src.eq("original explorer link").sum()))
    macro("NAddrWithdrawn", int(src.str.startswith("withdrawn").sum()))
    macro("NAddrNone", int((~src.eq("original explorer link") & ~src.str.startswith("withdrawn")).sum()))
    s = pd.read_csv(os.path.join(res, "onchain_source.csv")).drop_duplicates("Excel row")
    m = o.merge(s, on="Excel row", how="left", suffixes=("", "_s"))
    ok = m["status"].astype(str).str.startswith("ok")
    macro("NOnchainOK", int(ok.sum()))
    macro("PctOnchainOK", pct(ok.mean()))
    macro("NOnchainOKRug", int((ok & m["class"].eq("scam")).sum()))
    macro("NOnchainOKLegit", int((ok & m["class"].eq("normal")).sum()))
    macro("NLaterContract", int(m["status"].astype(str).str.contains("contract later").sum()))
    macro("NSymbolDiff", int(m["status"].astype(str).str.contains("symbol differs").sum()))
    cl = pd.read_excel(data_path, sheet_name="Change log")
    bc = cl[cl["Column"].astype(str).eq("Blockchain")]
    fixed = set(bc.loc[bc["Reason"].astype(str).str.startswith("the contract recorded"), "Row (Excel)"])
    fixed -= set(bc.loc[bc["Reason"].astype(str).str.startswith("chain correction withdrawn"), "Row (Excel)"])
    macro("NChainFixed", len(fixed))

    # ---- liquidity collapse of the rug pulls -------------------------------------------
    r = m[ok & m["class"].eq("scam")]
    v = r["liq_verdict"].astype(str)
    meas = v.isin(["drained", "frozen (no trading)", "pool empty before recorded end", "still trading"])
    active = v.eq("still trading") & r["Use in training"].astype(str).str.lower().eq("no")
    for key, name in [("drained", "Drained"), ("frozen (no trading)", "Frozen"),
                      ("pool empty before recorded end", "Empty"), ("still trading", "Trading"), ("no V2 pool", "NoPool")]:
        macro(f"Lq{name}", int(v.eq(key).sum()))
    macro("LqMeasured", int(meas.sum()))
    macro("LqActive", int(active.sum()))
    macro("LqThin", int((v.eq("still trading") & ~active).sum()))
    macro("PctCollapse", pct(1 - active.sum() / meas.sum()))
    lg = m[ok & m["class"].eq("normal")]["liq_verdict"].astype(str)
    macro("LqLegitLive", int(lg.eq("pool with liquidity today").sum()))
    macro("LqLegitNoPool", int(lg.eq("no V2 pool").sum()))

    # ---- exclusions ----------------------------------------------------------------------
    ex = o[~use]
    why = ex["Exclusion reason"].astype(str) + " " + ex["Quality flags"].astype(str)
    groups = {"ExclCollapse": why.str.contains("no liquidity collapse"),
              "ExclRepeat": why.str.contains("repeated collection"),
              "ExclConflict": why.str.contains("both labels|once as rug pull and once"),
              "ExclReview": ex["Exclusion reason"].astype(str).str.contains("second review"),
              "ExclDoubt": why.str.contains("label doubtful")}
    macro("NExclTotal", len(ex))
    for k, g in groups.items():
        macro("N" + k + ("Audit" if k == "ExclRepeat" else ""), int(g.sum()))
    rest = ~np.logical_or.reduce(list(groups.values()))
    macro("NExclOther", int(rest.sum()))

    # ---- start date -------------------------------------------------------------------------
    a = pd.read_csv(os.path.join(res, "start_date_audit.csv"))
    a = a[a["Excel row"].isin(o["Excel row"])]
    lag = -a["d_first_transfer"].dropna()            # days from first transfer to recorded start
    macro("StartN", len(lag))
    macro("StartMedLag", f"{lag.median():.0f}")
    macro("StartWithinSeven", pct((lag.abs() <= 7).mean()))
    macro("StartWithinThirty", pct((lag.abs() <= 30).mean()))
    macro("StartBeforeFirst", pct((lag < -30).mean()))
    pl = -a["d_pool"].dropna()
    macro("StartPoolMedLag", f"{pl.median():.0f}")

    # ---- Google counts are not reproducible ------------------------------------------------
    g = pd.read_csv(os.path.join(res, "google_calibration.csv")).dropna()
    rho = spearmanr(g["recorded"], g["today"]).correlation
    macro("GoogleN", len(g))
    macro("GoogleRho", f"{rho:.2f}")
    ratio = (g["today"] / g["recorded"].clip(lower=1))
    macro("GoogleRatioLo", f"{ratio.min():.2f}")
    macro("GoogleRatioHi", f"{ratio.max():.0f}")

    # ---- archive counts --------------------------------------------------------------------
    w = pd.read_csv(os.path.join(res, "wayback", "wayback_counts.csv")).drop_duplicates("row", keep="last")
    wu = w[w["row"].isin(o.loc[use, "Excel row"])]
    macro("WbSiteN", int(wu["wb_site_mid"].notna().sum()))
    macro("WbXN", int(wu["wb_x_mid"].notna().sum()))
    macro("WbDate", str(w["collected_at"].dropna().max()))
    rc = pd.read_csv(os.path.join(res, "wayback", "wayback_counts_recheck.csv"))
    j = rc.merge(w, on="row", suffixes=("_r", ""))
    vals = []
    for col in ["wb_site_start", "wb_site_mid", "wb_x_start", "wb_x_mid"]:
        q = j[[col, col + "_r"]].dropna()
        vals.append((q[col] == q[col + "_r"]).values)
    allv = np.concatenate(vals)
    macro("WbRecheckN", len(rc))
    macro("WbRepeat", pct(allv.mean()))
    # zero counts in the used rows, by class
    cls = o.set_index("Excel row")["class"]
    wu = wu.assign(cls=wu["row"].map(cls))
    for col, name in [("wb_x_start", "XStart"), ("wb_site_start", "SiteStart")]:
        for k, nm in [("scam", "Rug"), ("normal", "Legit")]:
            x = wu.loc[wu["cls"].eq(k), col].dropna()
            macro(f"WbZero{name}{nm}", pct((x == 0).mean()))
            macro(f"WbMed{name}{nm}", f"{x.median():.0f}")

    # ---- small legitimate projects ----------------------------------------------------------
    rv = c["Manual review, two reviewers, consensus (Nov 2026)"].astype(str)
    macro("CandClean", int(rv.str.startswith(("clean", "close to clean")).sum()))
    macro("CandCleanStrict", int(rv.str.startswith("clean").sum()))
    macro("CandFew", int(rv.str.startswith("unclear").sum()))
    macro("CandMany", int(rv.str.startswith(("many", "close to many")).sum()))
    macro("CandManyStrict", int(rv.str.startswith("many").sum()))
    macro("CandFully", int(c["Candidate suitability"].astype(str).str.startswith("fully").sum()))
    macro("CandETH", int(c["Blockchain"].astype(str).str.upper().eq("ETH").sum()))
    macro("CandBSC", int(c["Blockchain"].astype(str).str.upper().eq("BSC").sum()))
    macro("NDisputed", 4)

    # ---- model results -----------------------------------------------------------------------
    ev = pd.read_csv(os.path.join(res, "variants_eval.csv"))
    if "flagged: clean" not in ev.columns:     # candidate groups from the run without TabPFN
        extra = pd.read_csv(os.path.join(res, "variants_eval_owner_review.csv"))
        keep = ["Feature set", "Model"] + [k for k in extra.columns if k.startswith(("flagged:", "mean p:"))]
        ev = ev.merge(extra[keep], on=["Feature set", "Model"], how="left")
    have = [mm for mm, _ in MODELS if mm in set(ev["Model"])]
    main_model = "TabPFN" if "TabPFN" in have else "XGBoost"
    macro("VMainModel", "Real-TabPFN" if main_model == "TabPFN" else main_model)
    for _, row in ev.iterrows():
        if row["Feature set"] not in CODE:
            continue
        k = CODE[row["Feature set"]] + MCODE[row["Model"]]
        macro("V" + k, f"{row['ROC AUC']:.3f}")
        macro("V" + k + "lo", f"{row['ROC AUC low']:.3f}")
        macro("V" + k + "hi", f"{row['ROC AUC high']:.3f}")
    for name, code in [("A launch (Google)", "AB"), ("C launch (archive, on-chain start)", "C")]:
        q = ev[ev["Feature set"] == name].iloc[0]
        macro(f"VNTest{code}", int(q["n test"]))
        macro(f"VNTestRug{code}", int(q["test rug pulls"]))
    for name in ["C launch (archive, on-chain start)", "C day 30 (archive + on-chain)", "C midpoint (archive + on-chain)",
                 "B launch (archive)", "B midpoint (archive)"]:
        q = ev[ev["Feature set"] == name].iloc[0]
        macro("VNTrain" + CODE[name], int(q["n train"]))
    best = ev.groupby("Feature set")["ROC AUC"].max()
    for s_, code in CODE.items():
        if s_ in best:
            macro("VBest" + code, f"{best[s_]:.3f}")
    t = pd.read_csv(os.path.join(res, "variants_tests.csv"))
    for _, row in t.iterrows():
        if row["With"] in CODE and row["Against"] in CODE:
            k = CODE[row["With"]] + "vs" + CODE[row["Against"]] + MCODE[row["Model"]]
            macro("VD" + k, f"${row['Delta AUC']:+.3f}$")
            p = row["DeLong p"]
            macro("VP" + k, "<0.001" if p < 0.001 else f"{p:.3f}")
    for _, row in t.iterrows():
        pass
    # small projects: flag rates for the main model and the mean over models
    for s_, code, fam, _ in SETS:
        if fam not in ("B", "C"):
            continue
        sub = ev[ev["Feature set"] == s_]
        q = sub[sub["Model"] == main_model]
        if q.empty or "flagged: clean" not in q:
            continue
        q = q.iloc[0]
        macro("VTestFPR" + code, pct(q["test legit FPR"]))
        for col, nm in [("clean", "Clean"), ("a few signs", "Few"), ("many reports", "Many")]:
            f = str(q.get("flagged: " + col, ""))
            if "/" in f:
                n, dd = map(int, f.split("/"))
                macro(f"VFlag{nm}{code}", f"{n}/{dd}")
            macro(f"VMeanP{nm}{code}", f"{sub['mean p: ' + col].mean():.2f}")

    # ---- tables -------------------------------------------------------------------------------
    cols = [mm for mm in have]
    lines = ["\\begin{tabular}{@{}ll" + "c" * len(cols) + "@{}}", "\\toprule",
             "Start & Feature set & " + " & ".join("Real-TabPFN" if x == "TabPFN" else x.replace("Logistic Regression", "LR").replace("Naive Bayes", "NB") for x in cols) + " \\\\",
             "\\midrule"]
    last = None
    for s_, code, fam, label in SETS:
        sub = ev[ev["Feature set"] == s_].set_index("Model")
        if sub.empty:
            continue
        if last is not None and (fam == "C") != (last == "C"):
            lines.append("\\midrule")
        if fam == "C" and last != "C":
            lines.append(f"\\multicolumn{{{2 + len(cols)}}}{{@{{}}l}}{{\\emph{{On-chain start date: {int(sub['n test'].iloc[0])} test projects, {int(sub['test rug pulls'].iloc[0])} rug pulls}}}}\\\\")
        if fam != "C" and last is None:
            lines.append(f"\\multicolumn{{{2 + len(cols)}}}{{@{{}}l}}{{\\emph{{Recorded start date: {int(sub['n test'].iloc[0])} test projects, {int(sub['test rug pulls'].iloc[0])} rug pulls}}}}\\\\")
        top = sub["ROC AUC"].max()
        cells = []
        for mm in cols:
            x = sub.loc[mm, "ROC AUC"] if mm in sub.index else np.nan
            cells.append("--" if np.isnan(x) else (f"\\textbf{{{x:.3f}}}" if abs(x - top) < 5e-4 else f"{x:.3f}"))
        start = "recorded" if fam != "C" else "on-chain"
        lines.append(f"{start} & {label} ({fam}) & " + " & ".join(cells) + " \\\\")
        last = fam
    lines += ["\\bottomrule", "\\end{tabular}"]
    open(os.path.join(paper, "generated", "tab_variants.tex"), "w").write("\n".join(lines) + "\n")

    lines = ["\\begin{tabular}{@{}lccccc@{}}", "\\toprule",
             f"Feature set & AUC & Test legitimate flagged & \\multicolumn{{3}}{{c}}{{Small projects flagged}} \\\\",
             "\\cmidrule(l){4-6}",
             " & & & clean & a few signs & many reports \\\\", "\\midrule"]
    for s_, code, fam, label in SETS:
        if fam not in ("B", "C"):
            continue
        q = ev[(ev["Feature set"] == s_) & (ev["Model"] == main_model)]
        if q.empty or "flagged: clean" not in q.columns:
            continue
        q = q.iloc[0]
        lines.append(f"{label} ({fam}) & {q['ROC AUC']:.3f} & {pct(q['test legit FPR'])} & "
                     f"{q.get('flagged: clean', '--')} & {q.get('flagged: a few signs', '--')} & {q.get('flagged: many reports', '--')} \\\\")
    lines += ["\\bottomrule", "\\end{tabular}"]
    open(os.path.join(paper, "generated", "tab_candidates.tex"), "w").write("\n".join(lines) + "\n")

    with open(os.path.join(paper, "generated", "audit_numbers.tex"), "w") as f:
        f.write("% generated by scripts/make_audit_assets.py\n" + "\n".join(out) + "\n")
    print(len(out), "macros; main model for candidate rates:", main_model)


if __name__ == "__main__":
    main(*(sys.argv[1:4] if len(sys.argv) > 3 else ["data/TM-RugPull_v2.xlsx", "results", "paper"]))
