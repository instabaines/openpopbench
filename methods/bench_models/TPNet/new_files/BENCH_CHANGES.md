# Benchmark changes to the official TPNet release (lxd99/TPNet @ 7dd0cf4)

The G-repro run used the unmodified README command (Wikipedia, AP 99.31 ± 0.03 against the paper's 99.32).

**Benchmark adapter** `tpnet_bench.py`. The released modules are imported unchanged.
- **Graph:** a bipartite temporal graph with an edge cascade → participant for every event, origin included (protocol F2). There are no features, so they are zero-padded to 172 dimensions as DyGLib does.
- **Training, validation and early stopping:**
  - These replicate the released loop for TPNet with `--use_random_projection --load_best_configs`.
  - Train edges are the train-split events.
  - Validation uses the validation-window events with the released random-negative AP/AUC criterion, patience 20 and 100 epochs, plus the 8 GPU-h budget.
  - The released per-epoch test and new-node evaluations are skipped, so test data is never touched during training.
- **Full-candidate scoring (the addition):**
  - The random projections are reset, as at the start of every released epoch.
  - Every event is streamed in time order through the released `update`, in batches of 200.
  - At each floors query the released backbone and link predictor score (cascade, v) for every candidate v, using only edges strictly before t.
- **Neighbour retrieval for scoring** uses a vectorised version of the released `recent` strategy. Each run asserts that it is identical to the released `NeighborSampler` on 5,000 random (node, time) pairs. It passed on Bluesky S on 2026-10-16.
- **Compatibility:** `CUBLAS_WORKSPACE_CONFIG=:16:8` is set as in the released script. duckdb 1.1.3 and pyarrow 12.0.1 were added to bm-tpnet for reading the tables.
- **`EVAL_PROBE`:** times full-candidate scoring on 200 test queries after one training epoch, then extrapolates to all floors queries. This is used to decide feasibility on the large tiers.
- **`SCORE_ONLY=1` (2026-11-08, leakage test).** Skips training and scores the checkpoint already in OUT. Before scoring it calls `torch.manual_seed(SEED)`, so every score-only run uses the same random-projection draw (the released code redraws it at each reset). Used by the leakage test (`opb leakage`). Released results are unaffected.
- **`NODES=n` (2026-11-08, leakage test).** Pads the node table to n rows. TPNet has one node per cascade, so a benchmark cut at time T_c has fewer nodes than the trained checkpoint. The padded nodes have no events. Used only by the leakage test (`opb leakage`).
