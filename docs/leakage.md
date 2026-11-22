# Leakage: what we found and what the pipeline does about it

This note records how the evaluation protocol in the paper came about. It is
written for reviewers and for anyone who wants to reuse TM-RugPull for a
different task.

## The problem in one paragraph

Every project in TM-RugPull has a start date and an end date. For a rug pull,
the end date is the day it collapsed. For a legitimate project, it is the day
observation stopped, which for every legitimate project is the day its data
were collected (November 2024 to January 2025). On the 191 most recent projects,
the end date alone therefore separates the two classes perfectly (ROC AUC
1.000), better than any model we trained. The end date is not a model input, but
anything derived from it can inherit this signal. Project duration is the
obvious case. Less obvious are the descriptors read from a block-explorer page
(transaction count, holder concentration, the variances of holder balances,
supply and initial deposits): the explorer shows current values, the
collection notes do not record a point-in-time value, and for a rug pull
"current" means after the collapse.

## Two separate issues

1. **Feature timing.** Some features may be recorded after the event they are
   meant to explain.
2. **Label timing.** With a split by start date, 514 of the 763 training
   projects were still running when the first test project launched. Many rug
   pulls in the training set collapsed after test projects had already
   started, so their labels could not have been known when a real classifier
   would have been trained.

## What the pipeline does

- **Feature tiers.** Every feature is tagged with the moment it becomes
  observable: `launch`, `midpoint` or `end` (see `TIERS` in
  `src/lroo_pipeline.py` and the column dictionary in the dataset). The main
  model (`midpoint_bounded`) uses only launch and midpoint features. End-tier
  features appear only in the `full` set, which we report as an upper
  reference and not as a result.
- **Time-consistent training.** For a cutoff date `c`, a training project is
  admitted only if the features of the set's tier were observable before `c`
  and, for a rug pull, if `end + 72 h <= c`. See `admissible_train()`.
- **Rolling-origin evaluation.** The same rule is applied at four yearly
  cutoffs (2021 to 2024); each cutoff is tested on the following year.
- **Matched-subsample control.** `scripts/leakage_audit.py` draws random
  training subsets of the same size and class ratio as the time-consistent set.
  If the time-consistent model scored much lower than these, the gap would
  point to look-ahead in the unrestricted training set rather than to the
  smaller sample.
- **Single-feature AUCs.** The same script reports how well each feature
  separates the classes on its own, next to the start and end dates.
- **Fame baseline.** Legitimate projects were selected among well-rated,
  well-known tokens. The `fame_baseline` set keeps only the search hits for
  the project name and website, to see how much a "famous versus obscure"
  shortcut explains.

## What the pipeline cannot fix

- The midpoint is defined from the end date, so even the main feature set
  assumes that the end of the project is known. That is fine for forensic
  classification after a collapse and wrong for early warning.
- Point-in-time on-chain features exist only for projects whose contract
  could be confirmed on chain. They are computed from each token's transfer
  history 7, 14 and 30 days after its first transfer
  (`results/onchain_source.csv`). A rug pull that collapsed within 30 days
  would carry its collapse in the day-30 statistics, so every feature set that
  uses them is restricted to projects that lasted at least 30 days after their
  first transfer (`scripts/eval_variants.py`).
- Google result counts cannot be measured again. Internet Archive capture
  counts can, and the reproducible feature sets in the paper use them instead.
- The end date of a legitimate project marks the end of observation, that
  is, the day its data were collected (column `End date meaning`).

## Consequence for end-of-project features

Because a legitimate project's end-of-project features only become observable
when its data are collected, after every cutoff used in the evaluation, the
time-consistent rule admits no legitimate project for any feature set that
contains them. Such sets (`full` and its ablations) are therefore reported
only with unrestricted training, as an upper reference.
