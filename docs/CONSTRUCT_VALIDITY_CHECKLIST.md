# Construct validity checklist (Bean et al., "Measuring what Matters", arXiv 2511.04703, NeurIPS 2025 D&B), filled for OpenPopBench v1.0

The checklist was written for LLM benchmarks. The items specific to LLMs (format constraints, output parsing, training-corpus contamination, human raters) are answered with their closest analogue or marked n/a with the reason. Sections refer to the paper.

## 1. Define the phenomenon

| Item | Answer |
|---|---|
| Precise, operational definition | Spread of content through a platform population: which account joins a cascade next (Task 1), whether it is new to the platform (Task 2), and when (Task 3). Each is defined over the event stream (§2). |
| Scope and excluded aspects | Excluded:<br>- content (no text or media);<br>- cascade-size prediction;<br>- what accounts saw (impressions are not observed);<br>- accounts that never act. |
| Sub-components measured separately | Repeat versus no-tie participation (two slices of every Task 1 score); newcomers as their own task; timing as its own task. |

## 2. Measure only the phenomenon

| Item | Answer |
|---|---|
| Control for unrelated tasks | Four heuristics separate habit (recency, affinity) and popularity from spread (exposure). The no-tie slice removes repeat interactions. The learned feature ranker is a learned control. |
| Impact of format constraints | Analogue: the candidate set and negative sampling. Every method ranks the full candidate set. Training with sampled negatives is measured separately (§7: 10.2% vs 36.6% Hits@100). |
| Validate automated output parsing | Analogue: score paths. Each adapter's scores were checked against the model's own computation; each check is recorded in the adapter's `BENCH_CHANGES.md`. Ties get mid-ranks (§2.2), so ties cannot raise a score. In 13 of the 15 method-tier cells with the most ties, no tied target falls in the top 100 (measured on the v1.0 result files). |

## 3. Representative dataset

| Item | Answer |
|---|---|
| Sampling strategy | Queries are sampled uniformly from all test events (20,000 per slice), with cascade-cluster bootstrap CIs. The tiers are Forest Fire samples selected by pre-registered rules. The L tiers are complete populations. |
| Quality of items | Integrity checks pass on every tier, with exact reconciliation of 1,000 cascades and 100 edges against each source (Appendix A). |
| Known sensitivities | Sensitivities tested:<br>- population size (four tiers per platform);<br>- platform (three platforms);<br>- metric scale (Hits@k and Hits@percentile);<br>- training budget and hyperparameters (Appendix F). |

## 4. Reusing datasets

| Item | Answer |
|---|---|
| Adapts earlier data? | Yes:<br>- Bluesky: the Balduf et al. export;<br>- Telegram: the Dataverse election corpus;<br>- Reddit: Hessel et al. |
| Strengths and limitations of the prior work | Discussed in §3 and the datasheets: the Reddit deleted-submitter drop; Telegram covers only crawled chats; the Bluesky window is six weeks. |
| Performance on the new vs the original | MS-HGAT reproduced on its own data (Appendix C). On the legacy datasets, MS-HGAT reaches 155%, 108%, 116% and 93% of the best heuristic; on OpenPopBench it falls below the best heuristic on every tier where it runs (Figure 2a, Appendix C). |
| Modifications and how they help | Added:<br>- chronological splits;<br>- full-candidate ranking;<br>- timestamped networks;<br>- newcomer labels;<br>- size tiers;<br>- the leakage test.<br>Each answers a failure of the legacy protocols (§1, Appendix C). |

## 5. Contamination

| Item | Answer |
|---|---|
| Tests for contamination | Analogue: temporal leakage. The black-box cut test passes on all nine tiers and seven trained methods (§6.1). The shuffled-timing control passes (Appendix F.7). |
| Held-out set | The test split is held out by time. Later snapshots (v1.1 onward) give fresh test windows. |
| Pre-exposure in training corpora | n/a for models trained from scratch on the benchmark. For pretrained models, the Bluesky and Reddit sources are public, so later test windows should be preferred. |

## 6. Statistics

| Item | Answer |
|---|---|
| Sample size and power | 20,000 queries per slice; the CI half-width on Hits@100 is ≤ 0.01. |
| Uncertainty for primary scores | Every leaderboard cell has a 95% cluster-bootstrap CI and the sd over three seeds. Paired bootstrap tests of every comparison in the body are in Appendix F.5. |
| Human raters | n/a: no human labels. |
| Variability of subjective labels | n/a: labels are observed events. |

## 7. Error analysis

| Item | Answer |
|---|---|
| Failure modes | Failure modes found:<br>- one-vector models score 0 on newcomers;<br>- implementations fail at scale (probe-measured causes, Appendix D);<br>- the sanity gate catches degenerate scorers;<br>- timing models are miscalibrated. |
| Correlation with non-targeted phenomena | Scores by slice show that most gains come from repeat ties. Drift by test day shows no single day drives the gap (Appendix F.6). |
| Scoring biases | Hits@k depends on pool size, so Hits@percentile is reported (Appendix F). |
| Directions to improve scores | §7: pair-level structural features, inductive account representations, candidate generation that keeps training close to full ranking, and calibrated timing. |

## 8. Justify construct validity

| Item | Answer |
|---|---|
| Real-world relevance | Platform feeds and moderation act on populations of millions with arriving accounts (§1). |
| Rationale for tasks and metrics | The three tasks map to who, whether new, and when. Full-candidate Hits@k and MRR match ranking use; NLL and KS cover timing likelihood and calibration (§2). |
| Comparison with existing evaluations | Related work (§8) and the legacy-heuristic experiment (Appendix C). |
| Limitations and trade-offs | Limitations covered:<br>- sampled tiers cut network neighbours;<br>- pairwise methods are scored on a 10% subsample;<br>- only one snapshot per platform;<br>- Reddit is released as scripts. |
