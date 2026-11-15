# Benchmark use of EasyTPP 0.3.0 (official library; RMTPP, S2P2, WSMTHP, AttNHP)

The library code is unchanged. The clone at tag 0.3.0 is identical to `f1_bench/pydeps_easytpp` (checked with `diff -rq`, 2026-10-16). model-env also has an older `easy_tpp` in site-packages. Every run imports the 0.3.0 copy, from the clone's directory or from PYTHONPATH.

**Benchmark adapter** `easytpp_bench.py`:
- **Released pieces:**
  - the model classes;
  - the TPPRunner loop: log-likelihood loss, selection on validation log-likelihood, best checkpoint;
  - each model's `<MODEL>_train` block from `examples/configs/experiment_config.yaml`;
  - `rescale_time: true`, the library's option for large inter-event times.
- **Mapping:** mark = participant (N types); time is seconds since the cascade start.
- **Changes:**
  1. Sequences are cut into chunks of at most 64 events, because the non-event integral costs batch × length × MC samples × N.
  2. The runner's per-epoch "test" set is 500 validation sequences, so benchmark test data is never seen.
  3. A time budget; one validation always runs after it expires, so a checkpoint exists.
  4. `BATCH` comes from the feasibility probe when the released batch does not fit.
  5. Validation side-metrics are off (`model.event_sampler = None`). These are acc/RMSE from thinning, which needs 500 draws × N marks: 188 GB for RMTPP on S. Selection by log-likelihood is unchanged.
- **Scoring:**
  - The released `compute_intensities_at_sample_times` gives λ_k(t) for every mark at each query's true event time, over the last 63 earlier events. The last-step sample time is relative to the previous event; AttNHP takes absolute sample times, per its API.
  - Candidates are ranked by λ_k(t).
  - **Check (2026-10-16):** for RMTPP the scoring path reproduces the model's own event intensity (`forward()`, used in the log-likelihood) to within 2×10⁻⁶ relative error, over 200 validation sequences.

**Change (6), 2026-10-26: split validation batches.** With `VAL_SPLIT=k`, used for WSMTHP at scale, each validation batch is cut into pieces of k sequences.
- **Why:** WSMTHP trains with a score-matching loss but validates with the full log-likelihood, whose intensity covers every mark. Its validation pass therefore needs far more memory than a training step. It failed on Bluesky M after a 54 h epoch, and on Reddit L at batch 1.
- **Why it is exact:** the released epoch loop sums loss and event counts over batches and divides once.
- **Check:** RMTPP on Bluesky S, untrained model with the same seed. Validation log-likelihood is −49640.92508 unsplit against −49640.92510 split, over 304,063 events: only floating-point summation order.

**Change (7), 2026-10-27: scoring overrides.**
- **`QB`:** the scoring batch size, which affects memory only. The default sizes batches by 5e8 / (N·4) and ignores sequence length; at Bluesky M it ran out of memory.
- **`SCORE_ONLY=<checkpoint>`:** scores a saved best checkpoint without retraining. Used once: RMTPP on Bluesky M completed a 56.8 h epoch and its validation, then failed in our scoring step. The data export and time scale are rebuilt deterministically, as in training.

## (8) T4 event-time scoring of a saved checkpoint (2026-10-28)

`T4=1` with `SCORE_ONLY=<checkpoint>` loads the released model from its saved best T1 checkpoint and, instead of T1 ranking, scores the next-event gap for each all-events-sample floors query. It uses the same released `compute_intensities_at_sample_times`, the same ≤ 63-event history and `compute_last_step_only`. The total intensity (sum over all marks) is evaluated on a 96-point log grid from 0 to max(gap, 30 days), with the true gap inserted. The integral is trapezoidal. Outputs `<model>_t4_seed<s>.parquet`: gap, time NLL in nats with the gap in seconds (`-log λ(d) + Λ(d) + log(time_scale)`), PIT `1 - exp(-Λ(d))`, the median gap (Λ = ln 2) and a flag when the median lies past the grid end. No training or released code is changed. `T4_LIMIT=n` scores only the first n queries (smoke tests). The runs went on CPU (`CUDA_VISIBLE_DEVICES=`).

## (9) T1 ranking by log intensity in float64 (2026-10-30)

**Problem.** The released intensity heads compute λ_k(t) in float32:
- AttNHP and S2P2: `ScaledSoftplus`, log1p(exp(βx))/β.
- WSMTHP: `Softplus`.
- RMTPP: exp(logit).

When the argument is very negative, these round to exactly 0. Many candidates then tie with the target, and a tie gets the mid-rank. AttNHP on Bluesky M (learned β ≈ 37, biases ≈ −2) gave λ = 0 for every one of the 500k candidates, so every query scored rank (n+1)/2 and Hits@100 was 0.000.

**Change.** For T1 ranking only, the same activations are evaluated in log space in float64: log softplus_β(x) = log(log1p(exp(βx))) − log β, which equals βx − log β below βx = −30; for RMTPP, the clamped logit. log is monotone, so the order is the true order of λ_k(t). Nothing in training, model selection or T4 changes. `LOGSCORE=0` restores the old scoring.

**Check (Bluesky S, seed 1, where float32 ties were rare).** Hits@100 is identical to three decimals on val/test × all/novel for RMTPP and AttNHP. Ranks differ only on queries with float32 ties: 3.2% of RMTPP queries and 0.08% of AttNHP queries.

**Applied to** all 43 EasyTPP T1 rank files, from the saved best checkpoints.

## (10) Default T1 scoring batch under change (9) (2026-11-02)

Memory only; scores unchanged. Change (9) ranks in float64, which doubles the scoring tensor, but the automatic scoring batch (QB) still assumed float32 and allowed up to 256 queries per batch. On Reddit Mid, RMTPP seed 1 trained for 8.06 h and then ran out of GPU memory while scoring (one 17.7 GiB allocation). With LOGSCORE=1 the default is now min(32, 5e8 / (N x 8)). 32 is the batch the 10-30 re-scoring used on every tier. A QB set in the environment still overrides it. The failed cell was scored from its saved checkpoint with SCORE_ONLY. Its training history is only in the log of that run, because the crash came before the JSON was written.

- **Tuning check (2026-11-13; paper, Appendix F.3).** Environment overrides of the hyperparameters listed there; when unset, every value equals the released run.
