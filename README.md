# OpenPopBench

OpenPopBench is a training and evaluation harness and a continuous benchmark for predicting who joins a spreading cascade next, whether that participant is new to the platform, and when. It is built for platform conditions:
- populations of up to 13.7 million accounts;
- every observable account is a candidate;
- only the past is input;
- accounts keep arriving after training.

| | |
|---|---|
| Paper | *OpenPopBench: A Benchmark for Diffusion Prediction in Open Populations* (KDD 2027 Datasets and Benchmarks, under review) |
| Code | https://github.com/instabaines/openpopbench (MIT) |
| Data | `Baines/openpopbench-bluesky` (gated, source-export terms), `Baines/openpopbench-telegram` (CC BY 4.0), `Baines/openpopbench-reddit` (rebuild files only) on Hugging Face |
| Contact, takedown requests | raamure@ualr.edu |

## What the harness gives you

1. **Data.** Nine tiers from three platforms, in one schema:

   | platform | tiers |
   |---|---|
   | Bluesky 2025 | S, Mid, M, L |
   | Telegram 2024 | L |
   | Reddit 2012–13 | S, Mid, M, L |

   Each tier comes with fixed query files, newcomer labels and the Task 2a and Task 3 baselines.
2. **Methods.** 19 adapters, each running its authors' released code under one calling convention:
   - 15 of the paper's 16 published methods, from four families: diffusion, temporal graph, marked point process and sequential recommendation;
   - MS-HGAT and DyHGCN, the legacy diffusion references;
   - the two controls: the learned feature ranker and the exposure Hawkes model.

   `opb run` trains or re-scores any of them on any tier. TopoLSTM has no adapter: its released Theano code, run on CPU, needs more than 2.7 h per training step even on the smallest tier.
3. **Evaluation.** `opb evaluate` scores a submission with the estimators of the paper: full-candidate metrics, cluster-bootstrap intervals, seed variation, both slices, and the sanity gate. On the v1.0 result files it reproduces every leaderboard value of the paper (`python -m openpopbench.tests.check_against_paper`).
4. **The benchmark.** A versioned leaderboard per tier, with new test windows in later versions (see *Versions and the continuous benchmark*).

## Install

```bash
pip install -e .                    # the harness: numpy, pandas, pyarrow, scikit-learn, scipy, duckdb
opb tiers                           # tiers and where their files are
opb methods                         # registered methods
```

Data: download a dataset from Hugging Face and point the harness at it with `export OPB_DATA=/path/to/openpopbench-v1.0`. The Bluesky dataset needs you to accept its access agreement first. Reddit is rebuilt locally from the public source with `scripts/reddit/` (see the dataset card), because Reddit's terms do not permit redistribution.

Each published method needs its own environment, because its released code pins old library versions. `bash environments/setup_envs.sh` creates them with the exact versions used for the paper (see `environments/README.md`). `bash scripts/setup_methods.sh` then fetches each method's released code at its pinned commit and applies our adapter and patch from `methods/`.

## Tasks

Notation: an event e = (c, v, t) records that account v joins cascade c at time t.

| Task | Input | Output | Metrics |
|---|---|---|---|
| 1. Next participant | cascade c up to time t, all platform history before t | a ranking of every candidate account | Hits@10, Hits@100, MRR, Hits@0.1%, Hits@1% on all events and on no-tie events (no earlier tie to the origin) |
| 2a. Is the next participant new? | as Task 1 | P(newcomer) | log loss, Brier, ECE, AUC |
| 2b. Which newcomer? | as Task 1 | the Task 1 ranking | Hits@100, Hits@1%, MRR on newcomer events |
| 3. When? | cascade c up to its previous event | a density for the next gap | NLL, median NLL, log MAE of the median, KS of the PIT |

**Newcomer:** the participant has no event in any training cascade of the tier.

**Comparison and scale tiers.** Bluesky S and Mid and Reddit S and Mid (50k and 150k accounts) are the comparison tiers: methods train with three seeds, and the method comparison is made there. Bluesky M and L, Telegram and Reddit M and L are scale tiers.

## Rules

A result counts for the leaderboard only if it follows the protocol of the paper:

- **Chronological splits.** Train on training cascades only. Select checkpoints and hyperparameters on validation queries only, never on test.
- **Only the past.** A prediction for an event at time t may use only events, edges and first-seen times before t. Edge creation times must be respected.
- **Full candidate set.** Rank the target against every candidate in `n_candidates`; do not sample negatives at evaluation. Ties get the mid-rank.
- **Pairwise scorers** (methods that score one candidate per forward pass) may use the fixed 10% query subsample (`queries_sub10.parquet`) and are marked §.
- **Report the compute** (GPU type, hours per seed) and the hyperparameter search in `meta.json`. v1.0 used 8 GPU-hours per seed on comparison tiers and 24 on scale tiers.
- **Sanity gate.** A method must reach the validation Hits@100 of the popularity heuristic, and of every heuristic whose signal it uses as an input. Results that fail are listed with †.

## Adding a method

Write an adapter: a command with this interface.

```
<adapter> TABLES QUERIES OUT SEED BUDGET_H
```

- **TABLES:** the folder with `events`, `cascades`, `edges` and `participants`.
- **QUERIES:** `queries.parquet` (or `queries_sub10.parquet`).
- **OUT:** the output folder.

The adapter:
1. trains on the training split;
2. selects on validation;
3. writes `OUT/<key>_seed<SEED>.parquet` with `event_row` and `rank_<key>` for every row of QUERIES, validation and test;
4. with `SCORE_ONLY` set, loads its checkpoint from OUT instead of training. Re-scoring and the leakage test use this mode.

Register the method in `openpopbench/registry.py` with its family and command template. Then:

```bash
opb run mymethod bluesky_S --seed 1 --gpu 0 --out runs/bluesky_S/mymethod
opb collect mymethod bluesky_S runs/bluesky_S/mymethod --into submissions/mymethod
opb evaluate submissions/mymethod
```

The adapters in `methods/` are worked examples. Each comes with the upstream URL and commit, a patch of our edits, and a `BENCH_CHANGES.md` that lists every change made to the released code. Permitted changes:
- version-compatibility fixes;
- data adapters;
- data-pipeline bug fixes;
- numerically exact memory-saving ports;
- validation-based checkpoint selection.

Architectures and losses are not changed.

## Submitting a result

A submission is a folder:

```
meta.json                        name, family, code URL, queries ("full" | "sub10"), heuristic_inputs, training, track
<tier>/task1_seed<s>.parquet     event_row, rank
<tier>/task2a_seed<s>.parquet    event_row, p_new                  (optional)
<tier>/task3_seed<s>.parquet     event_row, nll, pit, median_gap   (optional)
```

`opb validate` checks the format, and `opb evaluate` writes `results.json`. Open a pull request to `leaderboard/` with the folder and a link to your code.

There are two tracks:
- **Files.** Your rank files are scored as submitted, and the entry is marked self-reported.
- **Verified (✓).** We run your adapter through the harness, by retraining or by re-scoring a checkpoint you provide, and confirm the files. We also run the leakage test (below). Verified entries are the only ones ranked on hidden test windows.

## Checks you can run

- **Leakage test.** `opb leakage cut` deletes everything at or after a cut time inside the test window; `opb leakage test` re-scores the same checkpoint on both worlds. Outputs for queries before the cut must not change (paper §6.1).
- **Reproduction.** `opb run METHOD TIER --score-only` re-scores a saved checkpoint and must reproduce the rank file it wrote.
- **Paper check.** `python -m openpopbench.tests.check_against_paper` compares the results with the paper's leaderboards (`leaderboard/v1.0/paper_values.json`). It matches 1,248 of 1,248 values.

## Versions and the continuous benchmark

- **v1.0** is frozen to the paper. Its numbers will not change.
- **Releases every six months.** The Bluesky tables are rebuilt from the newest source export, so removals made upstream carry over, and accounts named in takedown requests are dropped. Each release adds a new test window after the previous one. Its query files are published without the answers, and verified entries are scored on it by the maintainers. The answers are published with the next release. Results on public windows stay on the leaderboard and are marked by version.
- **Changelog.** Every release has a changelog, and its leaderboards are re-scored from the maintainers' checkpoints.

## Names in the code

The scripts and adapters were written before the task names of the paper were fixed. In code and file names:
- T1 is Task 1 and T2a/T2b are Tasks 2a/2b; T4 is Task 3 (the `T4=1` switch, `*_t4_seed*.parquet` files);
- floors are the heuristics (`floors_eval.py` computes the four heuristic ranks of `queries.parquet`);
- `novel` is the no-tie slice, and `unseen_train` the newcomer label;
- b1 is the learned feature ranker and b6 the exposure Hawkes model.

## Citation

Cite the paper and the data sources:
- Balduf et al., IMC 2024, for Bluesky;
- the Harvard Dataverse election corpus, for Telegram;
- Hessel, Tan and Lee, ICWSM 2016, for Reddit.
