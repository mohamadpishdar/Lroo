# Results

Output of `scripts/run_with_tabpfn.py` and `scripts/leakage_audit.py` on
`data/TM-RugPull_v2.xlsx`. Every table and number in the paper is computed
from these files by `scripts/make_paper_assets.py`.

| File | Content |
|---|---|
| `results_holdout.csv` | All metrics for every split, feature set and model. `Split` is `temporal hold-out` (763 earlier projects for training), `temporal hold-out, time-consistent training` (only projects whose features and labels were known at the cutoff) or `random stratified`. |
| `test_predictions_holdout.csv` | Predicted probability of every test project for every configuration. Columns are named `<feature set> | <model>`; `[time-consistent]` marks the restricted training set. |
| `results_rolling_folds.csv`, `test_predictions_rolling.csv` | Rolling-origin evaluation, per fold and per project. |
| `results_rolling_summary.csv` | Mean and standard deviation across folds, pooled AUC, and AUC within Ethereum and BNB Smart Chain. |
| `paired_tests.csv` | DeLong and exact McNemar tests against the best model, with Holm correction. |
| `ablation_deltas.csv` | AUC differences between feature sets with DeLong tests. |
| `consensus_ranking.csv`, `permutation_importance.csv`, `feature_statistics.csv` | Feature importance for the midpoint-bounded set. |
| `leakage_single_feature_auc.csv`, `leakage_matched_subsample.csv` | Leakage audit. |
| `sensitivity_flagged_rows.csv` | Time-consistent hold-out repeated without the 28 rows flagged by the on-chain audit, next to the main result. |
| `sensitivity_original_end_dates.csv` | Time-consistent hold-out repeated with the original end dates of the 107 legitimate projects whose end date was reset. |
| `data_quality_issues.csv` | Rows flagged while loading the data. |
| `split_assignment.csv` | Partition of every project in the temporal hold-out. |
| `run_info.json`, `tabpfn_info.json` | Package versions, seeds, tuned hyperparameters, feature tiers and TabPFN settings. |

## On-chain evidence and reproducible feature sets (November 2026)

| File | Content |
|---|---|
| `address_candidates.csv` | Candidate contract addresses from DexScreener and CoinGecko, ranked per project |
| `onchain_raw/hz2_*.csv` | Raw output of `onchain_horizons.py` per chain and worker |
| `onchain_horizons.csv` | Merged result after the transfer-count check: status of every project's address and, for confirmed ones, transfers, holders, top-10 share, HHI and Gini 7, 14 and 30 days after the first transfer |
| `collapse_audit_eth.csv`, `collapse_audit_bsc.csv` | Transfers in the 30 days before and after the recorded end date, with a flag when they do not match the label |
| `liquidity_check.csv` | Main V2 pool and its quote-token balance before and after the recorded end (rug pulls), or today (legitimate projects) |
| `label_review.csv` | Rows whose on-chain evidence does not match the label, in three groups |
| `onchain_source.csv` | Point-in-time on-chain features computed from each token's transfer history |
| `start_date_audit.csv` | Recorded start date against the first transfer on chain |
| `wayback/`, `wayback_targets.csv` | Raw Internet Archive capture lists and the URLs queried |
| `variants_eval.csv`, `variants_tests.csv` | Reproducible feature sets (archive counts, day-30 on-chain statistics) with time-consistent training, flag rates on the 62 small legitimate projects, and DeLong tests between sets |
| `wash_trading_check.csv` | Wash-trading check used in the label audit |
| `sensitivity_all_rows.csv` | Time-consistent evaluation with every row, including the 43 excluded ones |

## Label audit and small legitimate projects

| File | Content |
|---|---|
| `label_check_original_contracts.xlsx` | On-chain label check per project, grouped into inconsistent rug pulls, inactive legitimate projects, chain corrections and projects that cannot be checked |
| `second_review_output.csv` | Second review of disputed labels against public incident reports: verdict, confidence, reason and source URL per row |
| `small_legit_candidates.csv` | Small-cap CoinGecko candidates (`scripts/find_small_legit.py`); the 62 kept are rows 999 to 1060 of the dataset |
| `google_calibration.csv` | Google counts recorded at collection time next to the same queries today, for 16 projects; shows why the counts cannot be reproduced |

## Figures

The pipeline also writes diagnostic plots (`fig_*.png`) here when it runs. The
figures of the paper are in `../figures` (PNG) and `../paper/figures` (PDF).
