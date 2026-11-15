# SASRec (kang205/SASRec @ e373896): changes for G-repro and the benchmark

1. **Python 2 to 3 syntax** via the standard `lib2to3` converter (`git diff` shows every change): `print` statements, `xrange`, and dict iteration. Model and training logic are unchanged.
    - Reason: the only TensorFlow 1 build with A40 (sm_86) kernels is NVIDIA's nvidia-tensorflow 1.15.5, which requires Python 3.
2. **Environment:** bm-tf1 (Python 3.8, nvidia-tensorflow 1.15.5, numpy 1.21.6).
3. **Integer division preserved.** Python 2 `/` on integers floored the result; Python 3 needs `//` for the same behaviour:
    - `main.py:39`: `num_batch = len(user_train) // args.batch_size`. Without it, `range()` would get a float.
    - `modules.py:16`: the positional-encoding exponent `2*i//dim`.

    All other divisions are on floats or tensors, so their behaviour is unchanged.
4. **Evaluation hook, benchmark only** (`model.py`, 2 lines after `test_logits`): exposes `bench_last_seq` (the last-position sequence vector, `self.seq[:, -1, :]`) and `bench_item_table`. The released code scores 1 positive against 100 sampled negatives. The benchmark ranks all eligible candidates with the same dot-product scorer. Training is unchanged.
5. **Benchmark adapter** `sasrec_bench.py`:
   - Uses the released `Model`, `WarpSampler`, BCE loss and default hyperparameters.
   - Each train cascade is a user and each participant is an item.
   - Selection: validation NDCG@10 under full ranking, on 5,000 sampled validation queries. Evaluation runs every 20 epochs as released, plus every hour and at the end of training.
   - Ranks are computed in the graph. The run aborts unless the candidate count equals the floors' `n_candidates` on every query. In the smoke test this held on all 77,249 S queries.
- **`SCORE_ONLY=1` (2026-11-08, leakage test).** Skips training and scores the checkpoint already in OUT. Used by the leakage test (`opb leakage`) to score one checkpoint on the original and the cut world. Released results are unaffected.
