# Benchmark changes to the official PMRCA release

1. **Environment**: `torch_geometric==2.6.1` is installed with `--no-deps` into `next_project/pydeps_pyg` and put on PYTHONPATH (model-env, torch 2.9.1). The code is unchanged.
2. **GPU**: run with `-gpu $CUDA_VISIBLE_DEVICES`, because `run.py:37` overwrites `CUDA_VISIBLE_DEVICES` with this flag (default 0, which belongs to another user).
3. **Benchmark adapter** `pmrca_bench.py`:
   - **Released pieces:** the loader, model, `train_epoch` and `test_epoch`, the released patience of 10, and the k-means `e_step` before every batch.
   - **Data:** train cascades followed by 2,000 validation cascades. The released time-ordered ratio split then gives exactly train / validation.
   - **Scoring:** the released forward pass on prefixes padded by one slot, because forward drops the last input position.
   - **Budget:** time budget, stopping the training loader mid-epoch.
   - **Smoke test (S, 2026-10-19):** all 77,249 queries; candidate counts equal the floors' n_candidates.
4. **Release cached GPU memory after validation (2026-11-04).** `pmrca_bench.py` calls `torch.cuda.empty_cache()` after each validation pass.
   - **Why:** on Reddit Mid seed 1 the released faiss GPU k-means (`e_step`, run before every batch) failed to allocate on the first batch after the epoch-0 validation pass (`StandardGpuResources: alloc fail`). It had run before every one of the epoch's batches without trouble. PyTorch's caching allocator still held the validation pass's blocks, which faiss cannot use.
   - **Why it is exact:** it only returns unused cached blocks to the driver. No tensor, random state or computation changes.
5. **Score-only mode and a smaller scoring batch (2026-11-07).** `SCORE_ONLY=<log of the training run>` skips training, loads the saved best checkpoint and reads the training history back from that log. `SCORE_QB` sets how many queries are scored per forward pass (default 32, as before).
   - **Why:** on Reddit M seed 1 (500k users) training and validation finished (one epoch, 24.9 h, checkpoint saved), then scoring 32 queries at once ran out of memory: the logits are 32 × 199 × 500k floats, and it needed 11.9 GiB more with 4.8 GiB free.
   - **Why it is exact:** the forward pass is computed row by row. The one term that spans the batch is the softmax shift by the batch maximum in the intention attention, and a softmax gives the same result for any constant shift, so the ranks agree up to floating-point rounding. The checkpoint, candidate sets and rank formula are unchanged.
