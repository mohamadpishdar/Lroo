"""
LROO: rug-pull classification from on-chain and OSINT signals.

This module runs every experiment reported in the paper on the TM-RugPull
dataset and writes the results, predictions and diagnostic figures to one
folder.

Design in brief
  * Every feature carries an observation tier (launch, midpoint or end), the
    moment at which its value could first have been known. See TIERS.
  * The main feature set uses launch and midpoint features only. Sets with
    end-tier features are reported as upper references, not as results.
  * The hold-out is temporal: the 800 earliest projects by start date train,
    the 200 latest test. A second version of the hold-out admits a training
    project only if its features and (for rug pulls) its confirmed label
    existed at the cutoff date; see admissible_train().
  * A rolling-origin evaluation repeats this at four yearly cutoffs.
  * All preprocessing, SMOTE and hyperparameter search happen inside the
    training partition. Tuning uses forward-chaining folds.
  * Uncertainty: bootstrap intervals, DeLong and exact McNemar tests with Holm
    correction. Every test-set probability is saved.

Usage
    python src/lroo_pipeline.py data/TM-RugPull_v2.xlsx results
TabPFN is optional; scripts/run_with_tabpfn.py passes a model factory in.
"""

import json
import os
import warnings

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats
from scipy.stats import mannwhitneyu, shapiro, spearmanr, ttest_ind
from sklearn.base import clone
from sklearn.calibration import calibration_curve
from sklearn.compose import ColumnTransformer
from sklearn.feature_selection import mutual_info_classif
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, brier_score_loss,
                             cohen_kappa_score, confusion_matrix, f1_score,
                             matthews_corrcoef, precision_score, recall_score,
                             roc_auc_score, roc_curve)
from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit, train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler
from sklearn.svm import SVC
from statsmodels.stats.contingency_tables import mcnemar
from statsmodels.stats.multitest import multipletests
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from xgboost import XGBClassifier
import lightgbm as lgb

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
SEED = 42
TEST_FRACTION = 0.20
N_BOOTSTRAP = 1000
N_PERMUTATION_REPEATS = 15
N_TUNING_ITER = 30
N_TIME_SPLITS = 5
LOG_TRANSFORM = True          # signed log1p on numeric features (heavy-tailed data)
CHAIN_LAUNCH = {              # earliest plausible start date per chain
    "ETH": "2015-07-30", "BSC": "2020-09-01", "POLYGON": "2020-05-30",
    "ARBI": "2021-05-28", "FANTOM": "2019-12-27", "CRONO": "2021-11-08",
    "BASE": "2023-07-13", "SNOW": "2020-09-21",
}

LABEL_RAW = "class"
POSITIVE_CLASS = "scam"
START_RAW, END_RAW = "project starting date", "project end date"

# Raw column name -> clean name used in the paper.
# OSINT names keep the meaning of the raw columns; check that "(Launch Day)" and
# "(Midpoint)" describe how the counts were actually collected.
RENAME = {
    "MaxPrice (Quarter 1)": "Max Price Q1",
    "MaxPrice (Quarter 2)": "Max Price Q2",
    "MaxPrice (Quarter 3)": "Max Price Q3",
    "MaxPrice (Quarter 4)": "Max Price Q4",
    "Blockchain": "Blockchain",
    "the number of Transactions": "Transaction Count",
    "Token concentration ratio per holder": "Token Concentration Ratio",
    "Total Variance": "Holder Variance (All)",
    "Variance of holders with more than 1% tokens": "Holder Variance (Top 1%)",
    "Token balance": "Initial Token Supply",
    "first deposits": "Initial Deposit Count",
    "Blockchain Type": "Consensus Mechanism",
    "Google results for project title (first day)": "Search Hits: Title (Launch Day)",
    "Google results for project title (project duration/2)": "Search Hits: Title (Midpoint)",
    "Google results for project website (first day)": "Search Hits: Website (Launch Day)",
    "Google results for project website (duration/2)": "Search Hits: Website (Midpoint)",
    "Google results for project x profile (first days)": "Search Hits: X Profile (Launch Day)",
    "Google results for project x profile (duration/2)": "Search Hits: X Profile (Midpoint)",
}
DROP_RAW = {
    "Project Title": "identifier",
    "Sign": "identifier (token symbol)",
    "website": "identifier (URL)",
    "x profile": "identifier (URL)",
    "Smart Contract (online)": "constant (998 of 1,000 identical, differs only in letter case)",
    "smart Contract (offline)": "constant (single value)",
}
CATEGORICAL = ["Blockchain", "Consensus Mechanism"]
OSINT = [v for k, v in RENAME.items() if k.startswith("Google results")]
DURATION = "Project Duration (days)"
# Features whose measurement window extends past the project midpoint.
# Single-value on-chain columns were recorded at the end of the project, i.e. at or
# after the collapse for rug pulls.
END_SNAPSHOT = ["Transaction Count", "Token Concentration Ratio", "Holder Variance (All)",
                "Holder Variance (Top 1%)", "Initial Token Supply", "Initial Deposit Count"]
BEYOND_MIDPOINT = [DURATION, "Max Price Q3", "Max Price Q4"] + END_SNAPSHOT

# Feature tiers by the moment at which a value can first be observed.
OSINT_LAUNCH = [c for c in OSINT if "Launch Day" in c]
OSINT_MID = [c for c in OSINT if "Midpoint" in c]
TIERS = {
    "launch": OSINT_LAUNCH + CATEGORICAL,                       # known at T_start
    "midpoint": ["Max Price Q1", "Max Price Q2"] + OSINT_MID,   # known at T_mid
    "end": BEYOND_MIDPOINT,                                     # known at T_end
}
ALL_FEATURES = TIERS["launch"] + TIERS["midpoint"] + TIERS["end"]
FAME = [c for c in OSINT if "Title" in c or "Website" in c]     # general popularity proxies

# Each feature set lists the columns it KEEPS and the latest tier it needs.
FEATURE_SET_DEFS = {
    "full":                      (ALL_FEATURES, "end"),
    "no_osint":                  ([c for c in ALL_FEATURES if c not in OSINT], "end"),
    "no_lifetime":               ([c for c in ALL_FEATURES if c not in (DURATION, "Max Price Q3", "Max Price Q4")], "end"),
    "midpoint_bounded":          (TIERS["launch"] + TIERS["midpoint"], "midpoint"),   # main model
    "midpoint_bounded_no_osint": (CATEGORICAL + ["Max Price Q1", "Max Price Q2"], "midpoint"),
    "launch_day":                (TIERS["launch"], "launch"),
    "fame_baseline":             (FAME, "midpoint"),
}
FEATURE_SETS = {k: [c for c in ALL_FEATURES if c not in keep] for k, (keep, _) in FEATURE_SET_DEFS.items()}
MAIN_SET = "midpoint_bounded"
TIME_CONSISTENT_SETS = [MAIN_SET, "midpoint_bounded_no_osint", "launch_day", "fame_baseline", "full"]
ROLLING_SETS = ["midpoint_bounded", "midpoint_bounded_no_osint", "launch_day", "fame_baseline", "full"]
ROLLING_CUTOFFS = ["2021-01-01", "2022-01-01", "2023-01-01", "2024-01-01"]
LABEL_DELAY_DAYS = 3   # a collapse is confirmed after 72 hours


# ---------------------------------------------------------------------------
# Data loading and cleaning
# ---------------------------------------------------------------------------
def to_number(series):
    s = (series.astype(str).str.replace("\xa0", "", regex=False)
         .str.replace(",", "", regex=False).str.strip())
    s = s.replace({"-": np.nan, "nan": np.nan, "None": np.nan, "": np.nan})
    return pd.to_numeric(s, errors="coerce")


def load_and_clean(path, out_dir):
    raw = pd.read_excel(path) if str(path).endswith((".xlsx", ".xls")) else pd.read_csv(path)
    raw.columns = [" ".join(str(c).split()) for c in raw.columns]
    issues = []
    # Rows excluded by a recorded data decision (see the Review list) stay in the file but are
    # not used. Set LROO_USE_ALL_ROWS=1 to use every row.
    # Small-project candidates (added Nov 2026) have no Google counts and are never trained on.
    if "Candidate suitability" in raw.columns:
        raw = raw[raw["Candidate suitability"].isna()].reset_index(drop=True)
    if "Use in training" in raw.columns and os.environ.get("LROO_USE_ALL_ROWS") != "1":
        raw = raw[raw["Use in training"].astype(str).str.strip().str.lower() != "no"].reset_index(drop=True)
    extra_drop = os.environ.get("LROO_EXCLUDE_FLAG")          # sensitivity: drop rows with this flag
    if extra_drop and "Quality flags" in raw.columns:
        raw = raw[~raw["Quality flags"].astype(str).str.contains(extra_drop, regex=False)].reset_index(drop=True)

    missing = [c for c in list(RENAME) + list(DROP_RAW) + [LABEL_RAW, START_RAW, END_RAW]
               if c not in raw.columns]
    if missing:
        raise ValueError(f"Columns not found in dataset: {missing}")

    df = pd.DataFrame(index=raw.index)
    for src, dst in RENAME.items():
        if dst in CATEGORICAL:
            df[dst] = raw[src].astype(str).str.strip().str.upper()
        else:
            before = raw[src].isna().sum()
            df[dst] = to_number(raw[src])
            for idx in raw.index[df[dst].isna() & raw[src].notna()]:
                issues.append({"row": int(idx), "title": raw.loc[idx, "Project Title"],
                               "column": src, "problem": f"non-numeric value '{raw.loc[idx, src]}' set to missing"})
    df["Blockchain"] = df["Blockchain"].replace({"FTM": "FANTOM"})
    # 25 is the explorer's default page size, not a transaction count (see the Review list)
    tx = RENAME["the number of Transactions"]
    df.loc[raw["the number of Transactions"].eq(25).values, tx] = np.nan

    start = pd.to_datetime(raw[START_RAW], errors="coerce")
    end = pd.to_datetime(raw[END_RAW], errors="coerce")
    duration = (end - start).dt.total_seconds() / 86400.0
    for idx in raw.index[duration < 0]:
        issues.append({"row": int(idx), "title": raw.loc[idx, "Project Title"], "column": "dates",
                       "problem": f"end date {end[idx].date()} before start date {start[idx].date()}; duration set to missing"})
    duration[duration < 0] = np.nan
    df[DURATION] = duration

    for idx in raw.index:
        launch = CHAIN_LAUNCH.get(df.loc[idx, "Blockchain"])
        if launch is not None and pd.notna(start[idx]) and start[idx] < pd.Timestamp(launch):
            issues.append({"row": int(idx), "title": raw.loc[idx, "Project Title"], "column": START_RAW,
                           "problem": f"start date {start[idx].date()} is before {df.loc[idx, 'Blockchain']} launched ({launch})"})

    dup = raw[raw.duplicated(["Project Title", "Blockchain", START_RAW], keep=False)]
    for title, grp in dup.groupby("Project Title"):
        labels = sorted(grp[LABEL_RAW].unique())
        issues.append({"row": ";".join(map(str, grp.index)), "title": title, "column": "duplicate",
                       "problem": f"same title, chain and start date appear {len(grp)} times; labels={labels}"})

    y = (raw[LABEL_RAW].astype(str).str.strip().str.lower() == POSITIVE_CLASS).astype(int).values
    meta = pd.DataFrame({"title": raw["Project Title"], "start": start, "end": end,
                         "key": raw["Project Title"].astype(str) + "|" + raw["Blockchain"].astype(str),
                         "flags": raw["Quality flags"].astype(str) if "Quality flags" in raw.columns else ""})
    pd.DataFrame(issues).to_csv(os.path.join(out_dir, "data_quality_issues.csv"), index=False)
    return df, y, meta, issues


# ---------------------------------------------------------------------------
# Splits
# ---------------------------------------------------------------------------
def temporal_split(meta, test_fraction=TEST_FRACTION):
    order = np.argsort(meta["start"].fillna(pd.Timestamp.min).values, kind="stable")
    n_test = int(round(test_fraction * len(meta)))
    # Keep both partitions in chronological order: TimeSeriesSplit relies on it.
    train_idx, test_idx = order[:-n_test], order[-n_test:]
    return train_idx, test_idx


def random_split(y, test_fraction=TEST_FRACTION):
    idx = np.arange(len(y))
    return train_test_split(idx, test_size=test_fraction, random_state=SEED, stratify=y)


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------
def signed_log1p(x):
    return np.sign(x) * np.log1p(np.abs(x))


def make_preprocessor(numeric_cols, categorical_cols):
    num_steps = [("impute", SimpleImputer(strategy="median"))]
    if LOG_TRANSFORM:
        num_steps.append(("log", FunctionTransformer(signed_log1p, feature_names_out="one-to-one")))
    num_steps.append(("scale", StandardScaler()))
    return ColumnTransformer([
        ("num", Pipeline(num_steps), numeric_cols),
        ("cat", Pipeline([("impute", SimpleImputer(strategy="most_frequent")),
                          ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))]),
         categorical_cols),
    ])


def base_estimators():
    return {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=SEED),
        "Naive Bayes": GaussianNB(),
        "SVM": SVC(kernel="rbf", probability=True, random_state=SEED),
        "XGBoost": XGBClassifier(eval_metric="logloss", random_state=SEED, n_jobs=1),
        "LightGBM": lgb.LGBMClassifier(random_state=SEED, verbose=-1, n_jobs=1),
    }


TUNING_SPACES = {
    "XGBoost (tuned)": (XGBClassifier(eval_metric="logloss", random_state=SEED, n_jobs=1), {
        "clf__n_estimators": [100, 200, 400, 800],
        "clf__max_depth": [2, 3, 4, 6, 8],
        "clf__learning_rate": [0.01, 0.03, 0.05, 0.1, 0.2],
        "clf__subsample": [0.6, 0.8, 1.0],
        "clf__colsample_bytree": [0.6, 0.8, 1.0],
        "clf__min_child_weight": [1, 3, 5],
        "clf__reg_lambda": [0.1, 1.0, 10.0],
    }),
    "LightGBM (tuned)": (lgb.LGBMClassifier(random_state=SEED, verbose=-1, n_jobs=1), {
        "clf__n_estimators": [100, 200, 400, 800],
        "clf__num_leaves": [7, 15, 31, 63],
        "clf__learning_rate": [0.01, 0.03, 0.05, 0.1, 0.2],
        "clf__min_child_samples": [5, 10, 20, 40],
        "clf__subsample": [0.6, 0.8, 1.0],
        "clf__subsample_freq": [1],
        "clf__colsample_bytree": [0.6, 0.8, 1.0],
        "clf__reg_lambda": [0.0, 1.0, 10.0],
    }),
}


def build_pipeline(estimator, numeric_cols, categorical_cols, use_smote=True):
    steps = [("prep", make_preprocessor(numeric_cols, categorical_cols))]
    if use_smote:
        steps.append(("smote", SMOTE(random_state=SEED)))
    steps.append(("clf", estimator))
    return ImbPipeline(steps)


# ---------------------------------------------------------------------------
# Metrics and statistics
# ---------------------------------------------------------------------------
def compute_metrics(y_true, y_prob, threshold=0.5):
    y_pred = (y_prob >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return {
        "Accuracy": accuracy_score(y_true, y_pred),
        "Precision (weighted)": precision_score(y_true, y_pred, average="weighted", zero_division=0),
        "Recall (weighted)": recall_score(y_true, y_pred, average="weighted", zero_division=0),
        "F1 (weighted)": f1_score(y_true, y_pred, average="weighted"),
        "F1 (macro)": f1_score(y_true, y_pred, average="macro"),
        "Balanced Accuracy": balanced_accuracy_score(y_true, y_pred),
        "MCC": matthews_corrcoef(y_true, y_pred),
        "Kappa": cohen_kappa_score(y_true, y_pred),
        "ROC AUC": roc_auc_score(y_true, y_prob),
        "Brier": brier_score_loss(y_true, y_prob),
        "Recall (rug pull)": tp / (tp + fn) if tp + fn else np.nan,
        "Precision (rug pull)": tp / (tp + fp) if tp + fp else np.nan,
        "TN": int(tn), "FP": int(fp), "FN": int(fn), "TP": int(tp),
    }


def bootstrap_ci(y_true, y_prob, n=N_BOOTSTRAP, seed=SEED):
    rng = np.random.default_rng(seed)
    y_pred = (y_prob >= 0.5).astype(int)
    rows = []
    for _ in range(n):
        i = rng.integers(0, len(y_true), len(y_true))
        yt = y_true[i]
        row = {"Accuracy": accuracy_score(yt, y_pred[i]),
               "F1 (weighted)": f1_score(yt, y_pred[i], average="weighted"),
               "MCC": matthews_corrcoef(yt, y_pred[i])}
        row["ROC AUC"] = roc_auc_score(yt, y_prob[i]) if len(np.unique(yt)) == 2 else np.nan
        rows.append(row)
    b = pd.DataFrame(rows)
    out = {}
    for m in b.columns:
        out[f"{m} low"] = np.nanpercentile(b[m], 2.5)
        out[f"{m} high"] = np.nanpercentile(b[m], 97.5)
    return out


def _midrank(x):
    order = np.argsort(x)
    xs = x[order]
    n = len(x)
    t = np.zeros(n)
    i = 0
    while i < n:
        j = i
        while j < n and xs[j] == xs[i]:
            j += 1
        t[i:j] = 0.5 * (i + j - 1) + 1
        i = j
    out = np.empty(n)
    out[order] = t
    return out


def delong_test(y_true, prob_a, prob_b):
    """Two-sided DeLong test for two correlated ROC AUCs (DeLong et al., 1988)."""
    pos, neg = y_true == 1, y_true == 0
    m, n = pos.sum(), neg.sum()
    preds = np.vstack([prob_a, prob_b])
    k = preds.shape[0]
    tx = np.array([_midrank(p[pos]) for p in preds])
    ty = np.array([_midrank(p[neg]) for p in preds])
    tz = np.array([_midrank(np.concatenate([p[pos], p[neg]])) for p in preds])
    aucs = tz[:, :m].sum(axis=1) / m / n - (m + 1.0) / 2.0 / n
    v01 = (tz[:, :m] - tx) / n
    v10 = 1.0 - (tz[:, m:] - ty) / m
    cov = np.cov(v01) / m + np.cov(v10) / n
    diff = aucs[0] - aucs[1]
    var = cov[0, 0] + cov[1, 1] - 2 * cov[0, 1]
    z = diff / np.sqrt(var) if var > 0 else 0.0
    return aucs[0], aucs[1], z, 2 * stats.norm.sf(abs(z))


def mcnemar_test(y_true, pred_a, pred_b):
    a_ok, b_ok = pred_a == y_true, pred_b == y_true
    table = [[np.sum(a_ok & b_ok), np.sum(a_ok & ~b_ok)],
             [np.sum(~a_ok & b_ok), np.sum(~a_ok & ~b_ok)]]
    res = mcnemar(table, exact=True)
    return int(table[0][1]), int(table[1][0]), res.pvalue


# ---------------------------------------------------------------------------
# Experiment runner
# ---------------------------------------------------------------------------
def run_experiment(X, y, train_idx, test_idx, feature_set_name, drop_cols,
                   tabpfn_factory=None, tune=True, log=print):
    cols = [c for c in X.columns if c not in drop_cols]
    num_cols = [c for c in cols if c not in CATEGORICAL]
    cat_cols = [c for c in cols if c in CATEGORICAL]
    Xtr, Xte = X.iloc[train_idx][cols], X.iloc[test_idx][cols]
    ytr, yte = y[train_idx], y[test_idx]

    fitted, probs, rows, tuned_params = {}, {}, [], {}
    estimators = base_estimators()
    for name, est in estimators.items():
        pipe = build_pipeline(est, num_cols, cat_cols)
        pipe.fit(Xtr, ytr)
        fitted[name], probs[name] = pipe, pipe.predict_proba(Xte)[:, 1]

    if tune:
        cv = TimeSeriesSplit(n_splits=N_TIME_SPLITS)   # train rows are in time order
        for name, (est, space) in TUNING_SPACES.items():
            search = RandomizedSearchCV(build_pipeline(est, num_cols, cat_cols), space,
                                        n_iter=N_TUNING_ITER, scoring="roc_auc", cv=cv,
                                        random_state=SEED, n_jobs=-1, refit=True, error_score="raise")
            search.fit(Xtr, ytr)
            fitted[name], probs[name] = search.best_estimator_, search.best_estimator_.predict_proba(Xte)[:, 1]
            tuned_params[name] = {k.replace("clf__", ""): v for k, v in search.best_params_.items()}
            tuned_params[name]["cv_roc_auc"] = search.best_score_

    if tabpfn_factory is not None:
        try:
            pipe = build_pipeline(tabpfn_factory(), num_cols, cat_cols, use_smote=False)
            pipe.fit(Xtr, ytr)
            fitted["TabPFN"], probs["TabPFN"] = pipe, pipe.predict_proba(Xte)[:, 1]
        except Exception as exc:
            log(f"TabPFN failed on '{feature_set_name}': {exc}")

    for name, p in probs.items():
        row = {"Feature set": feature_set_name, "Model": name}
        row.update(compute_metrics(yte, p))
        row.update(bootstrap_ci(yte, p))
        rows.append(row)
    return {"rows": rows, "probs": probs, "fitted": fitted, "cols": cols,
            "num_cols": num_cols, "cat_cols": cat_cols, "Xtr": Xtr, "Xte": Xte,
            "ytr": ytr, "yte": yte, "tuned_params": tuned_params}


def paired_tests(yte, probs, reference):
    out = []
    ref_p = probs[reference]
    for name, p in probs.items():
        if name == reference:
            continue
        auc_a, auc_b, z, p_auc = delong_test(yte, ref_p, p)
        b, c, p_mc = mcnemar_test(yte, (ref_p >= 0.5).astype(int), (p >= 0.5).astype(int))
        out.append({"Reference": reference, "Compared with": name,
                    "AUC reference": auc_a, "AUC other": auc_b, "DeLong z": z, "DeLong p": p_auc,
                    "Only reference correct": b, "Only other correct": c, "McNemar exact p": p_mc})
    df = pd.DataFrame(out)
    if len(df):
        df["DeLong p (Holm)"] = multipletests(df["DeLong p"], method="holm")[1]
        df["McNemar p (Holm)"] = multipletests(df["McNemar exact p"], method="holm")[1]
    return df


# ---------------------------------------------------------------------------
# Feature importance
# ---------------------------------------------------------------------------
def batched_permutation_importance(model, Xte, yte, cols, n_repeats=N_PERMUTATION_REPEATS, seed=SEED):
    """Mean drop in ROC AUC when one raw feature is shuffled.

    All permuted copies are scored in a single predict_proba call, which keeps the
    number of requests to the hosted TabPFN service at one per model.
    """
    rng = np.random.default_rng(seed)
    base = roc_auc_score(yte, model.predict_proba(Xte)[:, 1])
    blocks, keys = [], []
    for c in cols:
        for _ in range(n_repeats):
            Xp = Xte.copy()
            Xp[c] = rng.permutation(Xp[c].values)
            blocks.append(Xp)
            keys.append(c)
    big = pd.concat(blocks, ignore_index=True)
    chunk = 10000   # stay well below per-request limits of the hosted service
    probs = np.concatenate([model.predict_proba(big.iloc[i:i + chunk])[:, 1]
                            for i in range(0, len(big), chunk)])
    n = len(Xte)
    drops = {}
    for k, c in enumerate(keys):
        drops.setdefault(c, []).append(base - roc_auc_score(yte, probs[k * n:(k + 1) * n]))
    return pd.Series({c: np.mean(v) for c, v in drops.items()}).reindex(cols)

def feature_importance(res, out_dir, log=print):
    Xtr, Xte, ytr, yte = res["Xtr"], res["Xte"], res["ytr"], res["yte"]
    num_cols, cat_cols, cols = res["num_cols"], res["cat_cols"], res["cols"]
    ranks = {}

    # 1. Mutual information (training partition)
    Xmi = Xtr.copy()
    for c in num_cols:
        Xmi[c] = signed_log1p(Xmi[c].fillna(Xmi[c].median()))
    for c in cat_cols:
        Xmi[c] = Xmi[c].astype("category").cat.codes
    mi = mutual_info_classif(Xmi[cols], ytr, discrete_features=[c in cat_cols for c in cols],
                             random_state=SEED)
    mi = pd.Series(mi, index=cols)
    ranks["Mutual information"] = mi.rank(ascending=False)

    # 2. Spearman correlation and class-difference tests (training partition, numeric only)
    sp_rows = []
    for c in num_cols:
        v = Xtr[c]
        ok = v.notna()
        rho, p_rho = spearmanr(v[ok], ytr[ok.values])
        a, b = v[ok & (ytr == 1)], v[ok & (ytr == 0)]
        normal = (len(a) >= 3 and len(b) >= 3 and shapiro(a.sample(min(len(a), 500), random_state=SEED))[1] > 0.05
                  and shapiro(b.sample(min(len(b), 500), random_state=SEED))[1] > 0.05)
        test = "Welch t" if normal else "Mann-Whitney U"
        p_diff = ttest_ind(a, b, equal_var=False)[1] if normal else mannwhitneyu(a, b)[1]
        sp_rows.append({"Feature": c, "Spearman rho": rho, "Spearman p": p_rho,
                        "Median (rug pull)": a.median(), "Median (legitimate)": b.median(),
                        "Test": test, "Test p": p_diff})
    sp = pd.DataFrame(sp_rows)
    sp["Spearman p (BH)"] = multipletests(sp["Spearman p"], method="fdr_bh")[1]
    sp["Test p (BH)"] = multipletests(sp["Test p"], method="fdr_bh")[1]
    sp.to_csv(os.path.join(out_dir, "feature_statistics.csv"), index=False)
    ranks["Spearman |rho|"] = sp.set_index("Feature")["Spearman rho"].abs().rank(ascending=False)

    # 3. Permutation importance (test partition, every model, raw features)
    perm = {}
    for name, model in res["fitted"].items():
        # Fewer repeats for the hosted TabPFN service to stay within API quotas.
        repeats = 5 if name == "TabPFN" else N_PERMUTATION_REPEATS
        perm[name] = batched_permutation_importance(model, Xte, yte, cols, n_repeats=repeats)
    perm_df = pd.DataFrame(perm)
    perm_df.to_csv(os.path.join(out_dir, "permutation_importance.csv"))
    ranks["Permutation"] = perm_df.rank(ascending=False).mean(axis=1)

    # 4. SHAP for tree models, aggregated back to raw features
    try:
        import shap
        shap_rank = {}
        for name in ["XGBoost", "LightGBM"]:
            model = res["fitted"][name]
            prep = model.named_steps["prep"]
            Z = prep.transform(Xte)
            names = list(prep.get_feature_names_out())
            sv = shap.TreeExplainer(model.named_steps["clf"]).shap_values(Z)
            sv = sv[1] if isinstance(sv, list) else sv
            imp = pd.Series(np.abs(sv).mean(axis=0), index=names)
            raw = {}
            for n_, v in imp.items():
                base = n_.split("__", 1)[1]
                key = next((c for c in cat_cols if base.startswith(c + "_")), base)
                raw[key] = raw.get(key, 0.0) + v
            shap_rank[name] = pd.Series(raw).reindex(cols).rank(ascending=False)
        ranks["SHAP"] = pd.DataFrame(shap_rank).mean(axis=1)
    except Exception as exc:
        log(f"SHAP skipped: {exc}")

    rank_df = pd.DataFrame(ranks).reindex(cols)
    rank_df["Consensus rank"] = rank_df.mean(axis=1, skipna=True)
    rank_df["Methods"] = rank_df.drop(columns="Consensus rank").notna().sum(axis=1)
    rank_df["OSINT"] = [c in OSINT for c in rank_df.index]
    rank_df["Extends past midpoint"] = [c in BEYOND_MIDPOINT for c in rank_df.index]
    rank_df = rank_df.sort_values("Consensus rank")
    rank_df.to_csv(os.path.join(out_dir, "consensus_ranking.csv"))
    return rank_df, sp


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------
PALETTE = ["#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B3", "#937860", "#DA8BC3", "#8C8C8C"]


def plot_roc(yte, probs, path, title):
    plt.figure(figsize=(6.5, 5.5))
    for i, (name, p) in enumerate(probs.items()):
        fpr, tpr, _ = roc_curve(yte, p)
        plt.plot(fpr, tpr, color=PALETTE[i % len(PALETTE)], lw=1.8,
                 label=f"{name} (AUC = {roc_auc_score(yte, p):.3f})")
    plt.plot([0, 1], [0, 1], "k--", lw=0.8)
    plt.xlabel("False positive rate"); plt.ylabel("True positive rate"); plt.title(title)
    plt.legend(fontsize=8, loc="lower right"); plt.tight_layout(); plt.savefig(path, dpi=200); plt.close()


def plot_calibration(yte, probs, path):
    plt.figure(figsize=(6.5, 5.5))
    for i, (name, p) in enumerate(probs.items()):
        pt, pp = calibration_curve(yte, p, n_bins=10, strategy="uniform")
        plt.plot(pp, pt, marker="o", ms=3, color=PALETTE[i % len(PALETTE)],
                 label=f"{name} (Brier = {brier_score_loss(yte, p):.3f})")
    plt.plot([0, 1], [0, 1], "k:", lw=0.8)
    plt.xlabel("Mean predicted probability"); plt.ylabel("Fraction of rug pulls")
    plt.title("Reliability diagram"); plt.legend(fontsize=8); plt.tight_layout()
    plt.savefig(path, dpi=200); plt.close()


def plot_ci(results, path, feature_set="full"):
    d = results[results["Feature set"] == feature_set].sort_values("ROC AUC")
    plt.figure(figsize=(6.5, 4))
    y = np.arange(len(d))
    plt.errorbar(d["ROC AUC"], y, xerr=[d["ROC AUC"] - d["ROC AUC low"], d["ROC AUC high"] - d["ROC AUC"]],
                 fmt="o", color="#4C72B0", capsize=3)
    plt.yticks(y, d["Model"]); plt.xlabel("ROC AUC (bootstrap 95% CI)"); plt.grid(axis="x", alpha=0.3)
    plt.tight_layout(); plt.savefig(path, dpi=200); plt.close()


def plot_confusions(yte, probs, path):
    n = len(probs)
    cols = 4
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(3.2 * cols, 3 * rows))
    axes = np.atleast_1d(axes).ravel()
    for ax, (name, p) in zip(axes, probs.items()):
        cm = confusion_matrix(yte, (p >= 0.5).astype(int), labels=[0, 1])
        ax.imshow(cm, cmap="Blues")
        for (i, j), v in np.ndenumerate(cm):
            ax.text(j, i, v, ha="center", va="center", color="white" if v > cm.max() / 2 else "black")
        ax.set_xticks([0, 1]); ax.set_xticklabels(["legit", "rug pull"])
        ax.set_yticks([0, 1]); ax.set_yticklabels(["legit", "rug pull"])
        ax.set_title(name, fontsize=9); ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    for ax in axes[n:]:
        ax.axis("off")
    plt.tight_layout(); plt.savefig(path, dpi=200); plt.close()


def plot_ablation(results, path):
    d = results.pivot_table(index="Model", columns="Feature set", values="ROC AUC")
    order = [c for c in FEATURE_SETS if c in d.columns]
    d = d[order]
    ax = d.plot(kind="bar", figsize=(8, 4.5), color=PALETTE[:len(order)], width=0.8)
    ax.set_ylabel("ROC AUC"); ax.set_ylim(max(0.5, d.min().min() - 0.05), 1.0)
    ax.set_xlabel(""); ax.legend(fontsize=8); plt.xticks(rotation=30, ha="right")
    plt.tight_layout(); plt.savefig(path, dpi=200); plt.close()


def plot_ranking(rank_df, path, top=15):
    d = rank_df.head(top).iloc[::-1]
    colors = ["#DD8452" if o else ("#C44E52" if b else "#4C72B0")
              for o, b in zip(d["OSINT"], d["Extends past midpoint"])]
    plt.figure(figsize=(7, 5))
    plt.barh(d.index, d["Consensus rank"], color=colors)
    plt.xlabel("Mean rank across methods (lower = more important)")
    from matplotlib.patches import Patch
    plt.legend(handles=[Patch(color="#4C72B0", label="On-chain"), Patch(color="#DD8452", label="OSINT"),
                        Patch(color="#C44E52", label="Extends past midpoint")], fontsize=8)
    plt.tight_layout(); plt.savefig(path, dpi=200); plt.close()



def plot_rolling(fold_df, path, feature_set=None):
    feature_set = feature_set or MAIN_SET
    d = fold_df[fold_df["Feature set"] == feature_set]
    plt.figure(figsize=(7, 4.2))
    for i, (m, g) in enumerate(d.groupby("Model")):
        plt.plot(g["Cutoff"], g["ROC AUC"], marker="o", color=PALETTE[i % len(PALETTE)], label=m)
    plt.ylabel("ROC AUC"); plt.xlabel("Cutoff (test = projects launched in the following year)")
    plt.title(f"Rolling-origin evaluation ({feature_set})"); plt.grid(alpha=0.3)
    plt.legend(fontsize=7, ncol=2); plt.tight_layout(); plt.savefig(path, dpi=200); plt.close()


# ---------------------------------------------------------------------------
# Time-consistent training sets
# ---------------------------------------------------------------------------
def observable_time(meta, tier):
    """Time at which every feature of a tier is observable for each project."""
    s, e = meta["start"], meta["end"]
    e = e.where(e >= s)                       # invalid end dates -> unknown
    return {"launch": s, "midpoint": s + (e - s) / 2, "end": e}[tier]


def admissible_train(idx, meta, y, cutoff, tier):
    """Projects that could have been in a training set built at `cutoff`.

    A project qualifies when (a) all features of the needed tier were observable
    before the cutoff and (b) its label was known: a rug pull only once its
    collapse had been confirmed (end date + 72 hours) before the cutoff.
    """
    cutoff = pd.Timestamp(cutoff)
    obs = observable_time(meta, tier).iloc[idx]
    ok = (obs < cutoff).values
    end_ok = (meta["end"].iloc[idx] + pd.Timedelta(days=LABEL_DELAY_DAYS) <= cutoff).values
    ok &= (y[idx] == 0) | end_ok
    keep = idx[ok]
    return keep[np.argsort(meta["start"].iloc[keep].values, kind="stable")]


def rolling_origin(X, y, meta, tabpfn_factory, log=print):
    rows, preds = [], []
    s = meta["start"]
    for k, c in enumerate(ROLLING_CUTOFFS):
        c0 = pd.Timestamp(c)
        c1 = pd.Timestamp(ROLLING_CUTOFFS[k + 1]) if k + 1 < len(ROLLING_CUTOFFS) else pd.Timestamp.max
        test_idx = np.where((s >= c0) & (s < c1))[0]
        test_idx = test_idx[np.argsort(s.iloc[test_idx].values, kind="stable")]
        for fs in ROLLING_SETS:
            keep, tier = FEATURE_SET_DEFS[fs]
            train_idx = admissible_train(np.arange(len(y)), meta, y, c0, tier)
            if y[train_idx].sum() < 6 or (1 - y[train_idx]).sum() < 6:
                log(f"  skip {c} {fs}: too few training examples")
                continue
            res = run_experiment(X, y, train_idx, test_idx, fs, FEATURE_SETS[fs],
                                 tabpfn_factory, tune=False, log=log)
            for r in res["rows"]:
                r.update({"Split": "rolling-origin", "Cutoff": c, "n train": len(train_idx),
                          "train rug pulls": int(y[train_idx].sum()), "n test": len(test_idx),
                          "test rug pulls": int(y[test_idx].sum())})
                rows.append(r)
            for m, p in res["probs"].items():
                preds.append(pd.DataFrame({"Cutoff": c, "Feature set": fs, "Model": m, "row": test_idx,
                                           "y_true": y[test_idx], "prob": p,
                                           "Blockchain": X["Blockchain"].iloc[test_idx].values,
                                           "start year": s.iloc[test_idx].dt.year.values}))
            line = " ".join(f"{r['Model'].split(' ')[0]}={r['ROC AUC']:.3f}" for r in res["rows"])
            log(f"  cutoff {c} | {fs:26s} train={len(train_idx)} ({int(y[train_idx].sum())} rug) "
                f"test={len(test_idx)} | {line}")
    return pd.DataFrame(rows), pd.concat(preds, ignore_index=True)


def summarize_rolling(folds, preds):
    agg = (folds.groupby(["Feature set", "Model"])["ROC AUC"]
           .agg(["mean", "std", "min", "max", "count"]).reset_index())
    agg.columns = ["Feature set", "Model", "AUC mean", "AUC sd", "AUC min", "AUC max", "Folds"]
    pooled = []
    for (fs, m), g in preds.groupby(["Feature set", "Model"]):
        yt, pr = g["y_true"].values, g["prob"].values
        row = {"Feature set": fs, "Model": m, "Pooled n": len(g)}
        row.update(compute_metrics(yt, pr))
        row.update(bootstrap_ci(yt, pr))
        for chain in ["ETH", "BSC"]:
            h = g[g["Blockchain"] == chain]
            row[f"AUC within {chain}"] = (roc_auc_score(h["y_true"], h["prob"])
                                          if h["y_true"].nunique() == 2 else np.nan)
        pooled.append(row)
    pooled = pd.DataFrame(pooled)
    keep = ["Feature set", "Model", "Pooled n", "ROC AUC", "ROC AUC low", "ROC AUC high", "Accuracy",
            "Balanced Accuracy", "MCC", "Brier", "TN", "FP", "FN", "TP", "AUC within ETH", "AUC within BSC"]
    return agg.merge(pooled[keep], on=["Feature set", "Model"])


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main(data_path, out_dir="results", tabpfn_factory=None, tune=True, log=print,
         permutation_for_tabpfn=True):
    os.makedirs(out_dir, exist_ok=True)
    np.random.seed(SEED)
    X, y, meta, issues = load_and_clean(data_path, out_dir)
    log(f"Loaded {len(X)} projects, {X.shape[1]} features, {y.sum()} rug pulls / {(1 - y).sum()} legitimate.")
    log(f"Data-quality issues written to data_quality_issues.csv: {len(issues)}")

    # ---- 1. Single temporal hold-out ------------------------------------
    train_idx, test_idx = temporal_split(meta)
    cutoff = meta["start"].iloc[test_idx].min()
    split_info = {
        "split": "temporal, ordered by project start date",
        "n_train": int(len(train_idx)), "n_test": int(len(test_idx)),
        "first_test_start": str(cutoff.date()),
        "last_train_start": str(meta["start"].iloc[train_idx].max().date()),
        "train_rug_pulls": int(y[train_idx].sum()), "train_legitimate": int((1 - y[train_idx]).sum()),
        "test_rug_pulls": int(y[test_idx].sum()), "test_legitimate": int((1 - y[test_idx]).sum()),
        "train_projects_ending_after_cutoff": int((meta["end"].iloc[train_idx] > cutoff).sum()),
    }
    log(json.dumps(split_info, indent=2))
    meta.assign(rug_pull=y, partition=np.where(np.isin(np.arange(len(y)), test_idx), "test", "train")) \
        .to_csv(os.path.join(out_dir, "split_assignment.csv"))

    all_rows, all_probs, res_by_set, tuned = [], {}, {}, {}
    for fs, drop in FEATURE_SETS.items():
        log(f"\n=== Hold-out | {fs} ({X.shape[1] - len(drop)} features) ===")
        res = run_experiment(X, y, train_idx, test_idx, fs, drop, tabpfn_factory, tune, log)
        for r in res["rows"]:
            r["Split"] = "temporal hold-out"
        all_rows += res["rows"]; all_probs[fs] = res["probs"]; tuned[fs] = res["tuned_params"]
        res_by_set[fs] = res
        for r in res["rows"]:
            log(f"  {r['Model']:20s} AUC={r['ROC AUC']:.3f} [{r['ROC AUC low']:.3f}-{r['ROC AUC high']:.3f}] "
                f"Acc={r['Accuracy']:.3f} MCC={r['MCC']:.3f} FN={r['FN']} FP={r['FP']}")

    # ---- 2. Same hold-out, training restricted to label-known projects --
    log("\n=== Hold-out with time-consistent training set (label and features known at cutoff) ===")
    skipped_tc = {}
    for fs in TIME_CONSISTENT_SETS:
        keep, tier = FEATURE_SET_DEFS[fs]
        tr_ok = admissible_train(train_idx, meta, y, cutoff, tier)
        n_rug, n_leg = int(y[tr_ok].sum()), int((1 - y[tr_ok]).sum())
        if n_rug < 6 or n_leg < 6:
            # Same rule as the rolling evaluation. For end-tier sets this happens because a
            # legitimate project's end-tier features are only observable at data collection.
            log(f"  {fs:18s} skipped: {n_rug} rug pulls and {n_leg} legitimate projects admissible")
            skipped_tc[fs] = {"train rug pulls": n_rug, "train legitimate": n_leg}
            continue
        res = run_experiment(X, y, tr_ok, test_idx, fs, FEATURE_SETS[fs], tabpfn_factory, tune=False, log=log)
        for r in res["rows"]:
            r.update({"Split": "temporal hold-out, time-consistent training",
                      "n train": len(tr_ok), "train rug pulls": int(y[tr_ok].sum())})
            log(f"  {fs:18s} {r['Model']:20s} train={len(tr_ok)} AUC={r['ROC AUC']:.3f}")
        all_rows += res["rows"]
        for m, p in res["probs"].items():
            all_probs.setdefault(f"{fs} [time-consistent]", {})[m] = p

    # ---- 3. Random split, for comparison with the earlier notebook ------
    log("\n=== Sensitivity: random stratified split (full features, default models) ===")
    r_tr, r_te = random_split(y)
    rres = run_experiment(X, y, r_tr, r_te, "full", [], tabpfn_factory, tune=False, log=log)
    for r in rres["rows"]:
        r["Split"] = "random stratified"
        log(f"  {r['Model']:20s} AUC={r['ROC AUC']:.3f} Acc={r['Accuracy']:.3f}")
    all_rows += rres["rows"]

    results = pd.DataFrame(all_rows)
    first = ["Split", "Feature set", "Model"]
    results = results[first + [c for c in results.columns if c not in first]]
    results.to_csv(os.path.join(out_dir, "results_holdout.csv"), index=False)

    yte = y[test_idx]
    pred = pd.DataFrame({"row": test_idx, "title": meta["title"].iloc[test_idx].values, "y_true": yte})
    for fs, probs in all_probs.items():
        for m, p in probs.items():
            pred[f"{fs} | {m}"] = p
    pred.to_csv(os.path.join(out_dir, "test_predictions_holdout.csv"), index=False)

    # ---- 4. Rolling-origin evaluation -----------------------------------
    log("\n=== Rolling-origin evaluation (time-consistent training sets) ===")
    folds, rpreds = rolling_origin(X, y, meta, tabpfn_factory, log)
    folds.to_csv(os.path.join(out_dir, "results_rolling_folds.csv"), index=False)
    rpreds.to_csv(os.path.join(out_dir, "test_predictions_rolling.csv"), index=False)
    rsum = summarize_rolling(folds, rpreds)
    rsum.to_csv(os.path.join(out_dir, "results_rolling_summary.csv"), index=False)
    log(rsum[["Feature set", "Model", "AUC mean", "AUC sd", "ROC AUC", "AUC within ETH", "AUC within BSC"]]
        .round(3).to_string(index=False))

    # ---- 5. Paired tests and ablations on the hold-out ------------------
    hold = results[results["Split"] == "temporal hold-out"]
    tests = []
    for fs in [MAIN_SET, "full"]:
        ref = hold[hold["Feature set"] == fs].sort_values("ROC AUC", ascending=False)["Model"].iloc[0]
        t = paired_tests(yte, all_probs[fs], ref); t.insert(0, "Feature set", fs); tests.append(t)
    tc = results[results["Split"] == "temporal hold-out, time-consistent training"]
    for fs in [MAIN_SET, "full"]:
        if f"{fs} [time-consistent]" not in all_probs:
            continue
        ref = tc[tc["Feature set"] == fs].sort_values("ROC AUC", ascending=False)["Model"].iloc[0]
        t = paired_tests(yte, all_probs[f"{fs} [time-consistent]"], ref)
        t.insert(0, "Feature set", f"{fs} [time-consistent]"); tests.append(t)
    tests = pd.concat(tests, ignore_index=True)
    tests.to_csv(os.path.join(out_dir, "paired_tests.csv"), index=False)
    log(f"\nPaired tests:\n{tests.round(4).to_string(index=False)}")

    comparisons = [("full", "no_osint"), ("full", "no_lifetime"), ("full", MAIN_SET),
                   (MAIN_SET, "midpoint_bounded_no_osint"), (MAIN_SET, "launch_day"),
                   (MAIN_SET, "fame_baseline")]
    deltas = []
    for a_fs, b_fs in comparisons:
        for m in all_probs[a_fs]:
            if m not in all_probs[b_fs]:
                continue
            a = hold[(hold["Feature set"] == a_fs) & (hold["Model"] == m)].iloc[0]
            b = hold[(hold["Feature set"] == b_fs) & (hold["Model"] == m)].iloc[0]
            _, _, _, p = delong_test(yte, all_probs[a_fs][m], all_probs[b_fs][m])
            deltas.append({"Reference set": a_fs, "Compared set": b_fs, "Model": m,
                           "AUC reference": a["ROC AUC"], "AUC compared": b["ROC AUC"],
                           "Delta AUC": a["ROC AUC"] - b["ROC AUC"], "DeLong p": p,
                           "FN reference": a["FN"], "FN compared": b["FN"]})
    tc_comparisons = [("full", MAIN_SET), (MAIN_SET, "midpoint_bounded_no_osint"),
                      (MAIN_SET, "launch_day"), (MAIN_SET, "fame_baseline")]
    for a_fs, b_fs in tc_comparisons:
        if f"{a_fs} [time-consistent]" not in all_probs or f"{b_fs} [time-consistent]" not in all_probs:
            continue
        pa, pb = all_probs[f"{a_fs} [time-consistent]"], all_probs[f"{b_fs} [time-consistent]"]
        for m in pa:
            if m not in pb:
                continue
            a = tc[(tc["Feature set"] == a_fs) & (tc["Model"] == m)].iloc[0]
            b = tc[(tc["Feature set"] == b_fs) & (tc["Model"] == m)].iloc[0]
            _, _, _, p = delong_test(yte, pa[m], pb[m])
            deltas.append({"Reference set": f"{a_fs} [time-consistent]", "Compared set": f"{b_fs} [time-consistent]",
                           "Model": m, "AUC reference": a["ROC AUC"], "AUC compared": b["ROC AUC"],
                           "Delta AUC": a["ROC AUC"] - b["ROC AUC"], "DeLong p": p,
                           "FN reference": a["FN"], "FN compared": b["FN"]})
    pd.DataFrame(deltas).to_csv(os.path.join(out_dir, "ablation_deltas.csv"), index=False)

    # ---- 6. Feature importance for the main model -----------------------
    main_res = res_by_set[MAIN_SET]
    if not permutation_for_tabpfn:
        main_res = dict(main_res, fitted={k: v for k, v in main_res["fitted"].items() if k != "TabPFN"})
    rank_df, _ = feature_importance(main_res, out_dir, log)
    log(f"\nConsensus ranking ({MAIN_SET}):\n{rank_df[['Consensus rank', 'Methods']].round(2).to_string()}")

    # ---- 7. Figures ------------------------------------------------------
    plot_roc(yte, all_probs[MAIN_SET], os.path.join(out_dir, "fig_roc_main.png"),
             "ROC curves, midpoint-bounded features (temporal hold-out)")
    plot_roc(yte, all_probs["full"], os.path.join(out_dir, "fig_roc_full.png"),
             "ROC curves, all features incl. end-of-project (upper bound)")
    plot_calibration(yte, all_probs[MAIN_SET], os.path.join(out_dir, "fig_calibration_main.png"))
    plot_ci(hold, os.path.join(out_dir, "fig_auc_ci_main.png"), MAIN_SET)
    plot_confusions(yte, all_probs[MAIN_SET], os.path.join(out_dir, "fig_confusion_main.png"))
    plot_ablation(hold, os.path.join(out_dir, "fig_ablation_auc.png"))
    plot_rolling(folds, os.path.join(out_dir, "fig_rolling_main.png"))
    plot_ranking(rank_df, os.path.join(out_dir, "fig_consensus_ranking_main.png"))

    import sklearn, xgboost, imblearn
    env = {"python": __import__("sys").version.split()[0], "numpy": np.__version__, "pandas": pd.__version__,
           "scikit-learn": sklearn.__version__, "xgboost": xgboost.__version__, "lightgbm": lgb.__version__,
           "imbalanced-learn": imblearn.__version__, "seed": SEED, "log_transform": LOG_TRANSFORM,
           "n_bootstrap": N_BOOTSTRAP, "n_permutation_repeats": N_PERMUTATION_REPEATS,
           "n_tuning_iter": N_TUNING_ITER, "n_time_splits": N_TIME_SPLITS}
    with open(os.path.join(out_dir, "run_info.json"), "w") as f:
        json.dump({"split": split_info, "environment": env, "tuned_params": tuned,
                   "dropped_columns": DROP_RAW, "feature_tiers": TIERS,
                   "feature_sets": {k: {"keeps": v[0], "latest_tier": v[1]} for k, v in FEATURE_SET_DEFS.items()},
                   "main_feature_set": MAIN_SET, "rolling_cutoffs": ROLLING_CUTOFFS,
                   "label_delay_days": LABEL_DELAY_DAYS,
                   "time_consistent_skipped": skipped_tc,
                   "time_consistent_training_rule": admissible_train.__doc__.strip()},
                  f, indent=2, default=str)
    return results, rsum


if __name__ == "__main__":
    import sys
    main(sys.argv[1] if len(sys.argv) > 1 else "data/TM-RugPull_v2.xlsx",
         sys.argv[2] if len(sys.argv) > 2 else "results")
