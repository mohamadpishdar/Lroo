# Datasheet for TM-RugPull (v2)

This datasheet follows the questions of Gebru et al., *Datasheets for Datasets* (CACM 2021).

## Motivation

**Why was the dataset created?** To study forensic classification of token projects after they
have collapsed: given what was observable about a project, was it a rug pull or a legitimate
project? It was built for the LROO study, which asks which features stay predictive once
information that is only known after the collapse is removed.

**Who created it?** Mohammad Pishdar and Bert Lagaisse (DistriNet, KU Leuven, Belgium), and
Fatemeh Shoaei, Mozafar Bag-Mohammadi and Mojtaba Karami (Department of Computer Science, Ilam
University, Iran). Contact: mohammad.pishdar@kuleuven.be.

## Composition

**What does an instance represent?** One token project on one EVM-compatible blockchain.

**How many instances are there?** 1059 rows. 954 are used for training and evaluation. 43 stay in
the file and carry the reason they are not used (column `Exclusion reason`, see below). The
remaining 62 are small legitimate projects added for an external false-alarm check and never
used for training.

| Chain | Legitimate | Rug pull | Total |
|---|---:|---:|---:|
| Ethereum | 282 | 309 | 591 |
| BNB Smart Chain | 92 | 226 | 318 |
| Polygon | 9 | 15 | 24 |
| Arbitrum | 5 | 7 | 12 |
| Fantom | 0 | 5 | 5 |
| Base, Cronos, Avalanche | 1 | 3 | 4 |
| **Total used** | **389** | **565** | **954** |

**Is it a sample?** Yes, and not a random one. Rug pulls were collected from incident databases
and explorers; legitimate projects were chosen among tokens with high security ratings and no
recorded incident. The legitimate class therefore leans toward large, well-known projects.

**What data does each instance contain?** Project title and symbol, chain, contract address
(when known), start and end date, maximum price per quarter of the observed lifetime, explorer
descriptors (transaction count, holder concentration, balance variances, supply, first
deposits), date-restricted Google result counts for the name, website and X account on the
launch day and at the lifetime midpoint, website and X URLs, and the label. The column
dictionary sheet defines every column. The released version adds point-in-time on-chain
features (transfers, holders, top-10 share, HHI and Gini at 7, 14 and 30 days after the first
transfer) for the projects whose contract could be confirmed on chain
(`results/onchain_source.csv`), and Internet Archive capture counts for the website and the
X profile at the start and the midpoint of each project.

**Is there a label?** Yes: `scam` (rug pull) or `normal` (legitimate).

**Is any information missing?** Contract addresses are missing for most rug pulls, because the
tokens are dead and no longer listed by search services. Point-in-time on-chain features exist
only where the address could be confirmed. In 29 rows the transaction count is 25, which is the
default page size of the explorer and not a real count; the value is kept and the pipeline
treats it as missing.

**Are there recommended splits?** A temporal hold-out on the start date (the latest 20% of
projects form the test set) and rolling-origin evaluation; random splits mix time periods and
are not recommended (see `docs/leakage.md`).

**Are there errors, noise or redundancies?** Known issues are listed one by one in the Review
list sheet, and every edit to the original file is in the Change log sheet. In short:

* 43 rows are not used for training: 21 rug-pull rows without a liquidity collapse on chain,
  16 repeated collections of the same project (the latest collection is used), 2 contracts
  recorded with both labels, 3 rug-pull labels not supported by a second review of public
  incident reports, and 1 legitimate label judged doubtful.
* 28 rows in use are flagged by the on-chain audit; a sensitivity analysis leaves them out.
* 62 small legitimate projects (rows 999 to 1060) were reviewed for scam and rug-pull reports
  by two reviewers, who agreed on a verdict for each (column
  `Manual review, two reviewers, consensus (Nov 2026)`). They are used only to check false
  alarms.
* 27 start dates fall before the launch of their chain; they are kept and recorded.
* Contract addresses that the project's own row could not support were rejected by an identity
  check (symbol, name, launch time and transfer count must agree with the row).

**Is the dataset self-contained?** Yes. Google result counts cannot be collected again in the
same form: search engines change their index and their counts over time. The Internet Archive
counts and the on-chain features can be measured again.

**Does it contain personal or confidential data?** No. Projects are identified by public token
names, contract addresses, websites and X accounts.

## Collection process

**How was the data acquired?** Rug-pull candidates came from CertiK, De.Fi, Rekt.news and block
explorers. Legitimate projects were chosen among tokens with high security ratings on the same
sources and no recorded incident. On-chain values were read from Etherscan, BscScan and the
other explorers. The Google counts are the number of results returned for date-restricted
queries on the project's name, its website and its X account, each run twice: restricted to the
launch day and to the midpoint of the project's lifetime. They were collected retrospectively,
so they approximate rather than reproduce what a search on those days would have returned.

**Labeling.** A project is labeled a rug pull when its pools hold close to zero liquidity,
there is no meaningful on-chain activity and its price can no longer be determined, all three for
at least 72 hours, and two computer-science experts agree that the collapse was deliberate.
The two experts confirmed each label independently; no agreement coefficient is reported. The
62 small legitimate projects added in November 2026 were reviewed by two reviewers who agreed
on one verdict per project; their separate verdicts were not recorded.

**On-chain checks of the labels (Nov 2026).** For rows with a confirmed contract, transfer
activity around the recorded end date (`results/collapse_audit_*.csv`) and the balance of the
token's main DEX pool before and after it (`results/liquidity_check.csv`) were measured. These
checks do not replace the labels; they flag rows for review.

**Time frame.** Projects started between 2015 and January 2025. Data for legitimate projects
was collected between November 2024 and January 2025, which is also their recorded end date.

## Preprocessing

Numeric cells stored as text are parsed (thousands separators and non-breaking spaces
removed); non-numeric entries in numeric columns become missing. No values were imputed in the
file. The original end dates of legitimate projects are kept in `project end date (original)`.

## Uses

**Intended use.** Research on rug-pull classification, especially on which signals are
available at which time. Every feature carries the time at which it becomes observable, so
studies can respect the time of prediction.

**Uses to avoid.** Do not use it as a list of accusations against specific projects: some labels
are contested (see the Review list), and a few legitimate-looking names in the rug-pull class
are copycat tokens of real projects. Do not train on end-tier features if the goal is early
warning.

## Distribution and maintenance

**License.** The dataset is released under the Creative Commons Attribution 4.0 International
license (CC BY 4.0); the code is released under the MIT License.

**Where is it distributed?** In this repository, together with the code that produces every
result in the paper.

**Maintenance.** Corrections are added to the Change log sheet with date and reason; rows are
excluded rather than deleted, so earlier results can be reproduced with
`LROO_USE_ALL_ROWS=1`.
