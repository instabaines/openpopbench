# Benchmark changes to the official HyperEvent release (jianjianGJ/HyperEvent @ 148fca7)

The G-repro run used the unmodified `train_evaluation_tgb.py --dataset tgbl-wiki` and scored test MRR 0.8098 against the leaderboard's 0.810.

**Benchmark adapter** `hyperevent_bench.py`. The released modules are imported unchanged: `get_segment`, `get_x_y`, `load_ns_eval`, `SequenceClassifier` and the train/eval steps.

1. **Graph:** an edge cascade → participant for every event, in the floors' event order.
2. **Config rule**, fixed before any result: each tier uses the `config_tgb.yaml` block of the TGB dataset nearest in edge count, on a log scale. Bluesky S maps to tgbl-review, Telegram to tgbl-coin. The tgbl-wiki block (one validation segment) made every validation pass on S take 360k sequential steps.
3. **Validation negatives:** 20 uniform participants per validation edge, fixed seed. 20 is `load_ns_eval`'s count for a dataset it does not know, and TGB ships no negative files for this data.
4. **Scoring:**
   - The adjacency table is streamed with the released update rule. Each run asserts it is identical to the released validation snapshot; this passed on S.
   - Each query uses the state after all earlier events in stream order. `get_x_y` builds the released features for every candidate, and the two rows it updates in place are restored after each chunk.
   - Queries follow the 10% query subsample (paper, Section 3.4).
5. **Environment:** duckdb and pyarrow were added to bm-tncn.

## (4) SCORE_ONLY and (5) optional logit ranking, with a tie check (2026-10-30)

- `SCORE_ONLY=1` scores the saved best checkpoint (`hyperevent_seed<s>.pth`) without training. `TRAIN_HOURS` carries the original training time.
- `LOGIT=1` ranks by the classifier logit instead of the sigmoid output. The order is the same; it only matters if float32 saturates the sigmoid. It is off by default.
- **Tie check:** 200 test queries, Bluesky S, seed 1, run with `EVAL_PROBE`.
  - 35% of queries have at least one candidate tied with the target under the logit, and also 35% under the sigmoid.
  - No candidate scores sigmoid = 1.0.
  - So the ties are not numerical. HyperEvent gives identical features to candidates with no neighbourhood around the cascade, and so gives them identical scores. Ties count half, as for every method.
  - No rescoring needed.

- **Tuning check (2026-11-13; paper, Appendix F.3).** Environment overrides of the hyperparameters listed there; when unset, every value equals the released run.
