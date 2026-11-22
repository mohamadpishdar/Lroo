# Reproducing the results

## Environment

The results in `results/` were produced with Python 3.13 and the package
versions pinned in `requirements.txt` (scikit-learn 1.9.0, XGBoost 3.4.1,
LightGBM 4.7.0, imbalanced-learn 0.14.2, tabpfn-client 0.7.1). The random seed
is 42 for every stochastic step: SMOTE, the models, the bootstrap, permutation
importance and the matched-subsample draws.

## Steps

```bash
pip install -r requirements.txt

# 1. Main experiments without TabPFN (about 30 minutes on a laptop)
python src/lroo_pipeline.py data/TM-RugPull_v2.xlsx results_local

# 2. The same with Real-TabPFN. The key is read from the environment only.
export TABPFN_TOKEN=...            # from https://ux.priorlabs.ai
python scripts/run_with_tabpfn.py data/TM-RugPull_v2.xlsx results

# 3. Leakage audit (single-feature AUCs, matched-subsample control)
python scripts/leakage_audit.py data/TM-RugPull_v2.xlsx results

# 3b. Sensitivity checks: without the rows flagged for review, with the original
#     end dates of legitimate projects, and with every row of the file
#     (TabPFN runs here if the key is set)
python scripts/sensitivity_flagged.py data/TM-RugPull_v2.xlsx results

# 3c. Reproducible feature sets: Internet Archive counts and day-30 on-chain
#     statistics (projects that lasted at least 30 days after their first
#     transfer), plus flag rates on the 62 small legitimate projects
python scripts/eval_variants.py data/TM-RugPull_v2.xlsx results/onchain_source.csv results

# 4. Figures, tables and the numbers quoted in the paper
python scripts/make_paper_assets.py data/TM-RugPull_v2.xlsx results paper
python scripts/make_audit_assets.py data/TM-RugPull_v2.xlsx results paper

# 5. The paper
cd paper && latexmk -pdf main.tex
```

Step 2 sends the training and test features (never the project names) to the
Prior Labs API. Permutation importance with TabPFN alone needs more API tokens
than a monthly free quota allows, so `scripts/run_with_tabpfn.py` computes
permutation importance for the local models only, as in the paper. The
sensitivity checks include TabPFN for both feature sets.

## Rows used

The data sheet has a `Use in training` column. Rows marked `no` stay in the file
with their `Exclusion reason` and are skipped by the pipeline; set
`LROO_USE_ALL_ROWS=1` to use every row. A transaction count of exactly 25 is
the explorer's default page size, not a count, and is read as missing.

## Rebuilding the on-chain evidence (optional)

The files `results/onchain_source.csv`, `results/onchain_horizons.csv`, `results/collapse_audit_*.csv`, `results/wayback/` and
`results/liquidity_check.csv` are released, so the steps above do not need any
blockchain access. To rebuild them you need an Etherscan key (Ethereum,
Polygon, Arbitrum) and a NodeReal endpoint (BNB Smart Chain, archive access),
both read from the environment only:

```bash
export ETHERSCAN_API_KEY=...  NODEREAL_URL=...
# candidate addresses from DexScreener and CoinGecko
python scripts/find_addresses_dex.py data/TM-RugPull_v2.xlsx results/address_candidates.csv
# confirm each candidate from its transfer history and compute day-7/14/30 features
python scripts/onchain_horizons.py data/TM-RugPull_v2.xlsx results/address_candidates.csv results/hz_raw.csv
# merge and apply the transfer-count identity check
python scripts/finalize_horizons.py data/TM-RugPull_v2.xlsx results/onchain_horizons.csv results/hz_raw.csv
# transfer activity and pool liquidity around the recorded end date
python scripts/audit_collapse.py data/TM-RugPull_v2.xlsx results/onchain_horizons.csv results/collapse_audit_eth.csv
python scripts/liquidity_check.py data/TM-RugPull_v2.xlsx results/onchain_horizons.csv results/liquidity_check.csv
# integrity checks
python scripts/verify_dataset.py data/TM-RugPull_v2.xlsx results/onchain_horizons.csv
```

`onchain_horizons.py` can be split across workers with `SHARD=i/n` and
restricted to chains with `ONLY_CHAINS`. Free API tiers make the full run take
several hours, mostly for large legitimate tokens.

## Determinism

The local models are deterministic for a fixed seed and package versions.
TabPFN is called with a fixed `random_state`, and repeated calls during our
runs returned identical probabilities, but the hosted service may change the
model behind a checkpoint name. `results/tabpfn_info.json` records the client
version and parameters used.

## Google Colab

`notebooks/LROO_colab.ipynb` runs steps 1 to 3 in Colab. Store the TabPFN key
as a Colab secret named `TABPFN_TOKEN`; never paste it into a cell.
