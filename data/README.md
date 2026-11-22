# TM-RugPull, version 2

`TM-RugPull_v2.xlsx` is the dataset used in the LROO paper. The pipeline reads
the first sheet, `data`; the other sheets document it.

| Sheet | Content |
|---|---|
| `data` | 1059 rows, one per project. 954 are used in the paper; the column `Use in training` says which and `Exclusion reason` why. Rows 999 to 1060 are the 62 small legitimate projects described below. |
| `README` | Short description of the workbook. |
| `Collection protocol` | How a new project should be collected, step by step. |
| `Column dictionary` | Every column, its name in the paper, when it is measured and its observation tier. |
| `Review list` | Every known issue, with the decision taken (see below). |
| `Change log` | Every cell changed relative to version 1. |
| `Horizon features` | Template for point-in-time on-chain features. The values used in the paper are in `results/onchain_source.csv`; `results/onchain_horizons.csv` holds an earlier collection. |
| `New projects (template)` | Template for adding projects, with one example row. |

## Changes from version 1

- 69 numeric values stored as text (thousand separators, non-breaking spaces)
  were converted to numbers.
- Cells holding a dash, a project name or a date in a numeric column were
  emptied.
- `FTM` and `FANTOM` were merged into `FANTOM`.
- A misspelt column header was corrected.
- Four columns were added: `Contract address` (filled in later, see On-chain checks below),
  `Label confirmation date` (end date plus 72 hours for rug pulls),
  `End date meaning` (collapse date for rug pulls; end of observation, i.e. the
  data collection date, for legitimate projects) and `Selection source`.

The data owner confirmed that the end date of a legitimate project is the end
of observation. 294 of the 401 legitimate projects fall between 1 November 2024
and 31 January 2025. The other 107 had end dates outside that window (103
earlier, 4 later); they were reset to 8 December 2024 (the window median). The
original values are kept in the last column of the `data` sheet
(`project end date (original)`) and every change is listed in `Change log`.
Two rows were removed: FARM Reward Token and qiibeeToken each appeared twice with
conflicting labels (rug pull and legitimate). Etherscan transfer histories
(checked 6 November 2026) show that FARM never had a gap of 72 hours or more
before November 2026 and that QBX faded gradually rather than collapsing, so the
rug-pull rows were removed and the legitimate rows kept. Kizuna AI was also removed: its recorded end date preceded its start, and on
chain the token still has liquidity and daily trading, so it does not meet the
rug-pull definition. NOVAMIND's end date, which also preceded its start, was
corrected to 3 May 2024 from its on-chain history. No other label was
changed. Because of the reset, results computed on version 2 are
not identical to those on version 1.

## Rows excluded from the experiments (November 2026)

No row was deleted. 43 of the original rows are kept but marked
`Use in training = no`, with the reason in `Exclusion reason`:

- 21 rug-pull rows that the source database lists as rug pulls, but whose
  contract shows no liquidity collapse on chain;
- 16 repeated collections of the same project (same website and X account or
  same contract, same chain and label). The copy with the latest collection
  is used;
- 2 contracts recorded once as a rug pull and once as legitimate;
- 3 rug-pull labels that a second review of public incident reports does not
  support, for example the official Contentos token, whose pool kept trading
  after the recorded end;
- 1 legitimate label judged doubtful: COS, which the review found likely to be
  the COSS exchange token, with rug-pull signs.

Four labels that the reviews disputed were decided by the authors: Frontfanz,
Arken and Nabox keep their rug-pull label, and COS is excluded as above.

Set `LROO_USE_ALL_ROWS=1` to use every original row. The paper reports this
as a sensitivity analysis.

## Small legitimate projects (rows 999 to 1060)

The legitimate class leans toward large, well-known projects. To test whether
the models learned "obscure" rather than "fraudulent", 62 small-cap tokens
(37 on Ethereum, 25 on BNB Smart Chain) were sampled from CoinGecko in
November 2026. They have no Google counts, so they are never used for
training (`Candidate suitability` is filled only for them, and the pipeline
drops them by that column). Two reviewers searched each one for scam and
rug-pull reports and agreed on a verdict, recorded in
`Manual review, two reviewers, consensus (Nov 2026)`:

| Verdict | Projects |
|---|---:|
| clean: no scam or rug-pull reports found | 40 |
| close to clean: almost no signs | 1 |
| unclear: a few warning signs | 15 |
| close to many reports: more signs than unclear | 1 |
| many scam or rug-pull reports | 5 |

The paper pools each "close to" verdict with its neighbour (41 clean, 15 with
a few signs, 6 with many reports) and reports how often the reproducible
models flag each group.

## Internet Archive captures

For every project, the number of days on which the Internet Archive captured
the website and the X profile was counted up to the recorded start date and
midpoint, and up to the on-chain start date (first transfer) and its midpoint.
The columns start with `Archived`, the date of collection is in
`Archive counts collected on`, and `Archive measurement note` explains
individual cases. Unlike Google counts, these values can be measured again
(`scripts/wayback_collect.py`).

## On-chain checks

Contract addresses were searched on DexScreener and CoinGecko and confirmed
from each token's transfer history (symbol, launch time, and a transfer count
that fits the row). Many rug pulls copy the name of a famous token; the count
check rejected matches with the original token. The confirmed addresses, the
first on-chain transfer and the gap between it and the recorded start date are
in the `data` sheet; 95% of the contracts were confirmed from their own transfer
history. Transfer, holder and concentration features 7, 14 and 30 days after the
first transfer are in `results/onchain_source.csv`.

For confirmed rug pulls, `results/collapse_audit_*.csv` compares transfers in the
month before and after the recorded end date, and `results/liquidity_check.csv`
measures the token's main V2 pool. The verdicts are hints, not labels; rows
whose activity continued are flagged in `Quality flags` and left out in a
sensitivity analysis. `results/label_review.csv` lists them.

## Known issues

The `Review list` sheet records:

- 4 projects whose end date preceded their start date (all resolved: two
  legitimate projects through the end-date reset, NOVAMIND corrected, Kizuna AI
  removed);
- 27 projects whose start date precedes the launch of the chain they are
  recorded on; 21 are on BNB Smart Chain, and about a third of those are
  Binance-Peg tokens whose start date refers to the original token on Ethereum;
- 29 rows whose transaction count is 25, the explorer's default page size
  rather than a count; the pipeline reads it as missing;
- the repeated projects and contested labels above.

Google result counts cannot be collected again in the same form, because
search indexes change. The exact query settings used at collection time should
be documented by the data owners (see `docs/DATASHEET.md`).

## Observation tiers

| Tier | Features |
|---|---|
| launch | Search hits for name, website and X account on the launch day; blockchain; consensus mechanism |
| midpoint | Max price in quarters 1 and 2; search hits for name, website and X account at the midpoint |
| end | Project duration; max price in quarters 3 and 4; transaction count; token concentration; holder variances; initial supply; initial deposits |

End-tier values were read from block-explorer pages at collection time and,
for rug pulls, describe the token after its collapse. They are used only in
the reference configurations of the paper.
