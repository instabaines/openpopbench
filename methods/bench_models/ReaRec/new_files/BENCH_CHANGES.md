# Benchmark changes to the official ReaRec release (TangJiakai/ReaRec @ 7f46432)

1. **Data:**
   - Software comes from Amazon-Reviews-2023 (Hugging Face, McAuley-Lab), using the raw reviews, the metadata and the official 5-core timestamp splits.
   - The released notebooks `dataset/src/Software.ipynb` and `amazon_remap.ipynb` were run as scripts, with their code cells in order.
   - The only edit: `amazon_remap.ipynb`'s domain list became `['Software']`, because the other three domains are not downloaded.
2. **G-repro command:** `main.py --model_name PRL --dataset Software --path ../dataset/processed --gpu $CUDA_VISIBLE_DEVICES`, with released defaults otherwise. `--gpu` sets `CUDA_VISIBLE_DEVICES`; its default is "" (CPU).
3. **Target:** paper Table 2 (arXiv 2503.22675), ID-based SASRec+PRL on Software: N@10 0.0739, N@20 0.0949, R@10 0.1488, R@20 0.2324.
4. **User ids:** the released remap notebooks leave Amazon `user_id` as strings, but `BaseReader` converts the column to an integer tensor (TypeError). `bench_encode_users.py` encodes users as consecutive integers. The models and runner never read `user_id`, so no computation changes.
5. **Release bug** (`src/models/PRL.py`, `loss`): `cl_loss` is assigned only when the batch has a second, noise-augmented view. During the warm-up epochs (`epoch <= warmup`, default 2) the noise factor is 0, so no second view exists and the first epoch crashed with UnboundLocalError. The fix initialises `cl_loss = 0`. Epochs with two views are unchanged, and warm-up epochs get no contrastive term, as the gating intends.

**Change (2), 2026-10-25: chunked validation.** With `CHUNKED_EVAL=1`, used from Bluesky M upward, validation runs batch by batch.
- **Why:** the released `BaseRunner.predict` stacks the full [validation rows × items] score matrix on the GPU. That is 26 GiB on Bluesky M, and hundreds of GB on Reddit L.
- **What it does:** each eval batch goes through the same predictions and the same metric functions (`utils.metrics`). Per-row sums are accumulated, and the released 4-decimal rounding is applied once at the end.
- **Check:** on random scores (3,000 × 5,000) it is identical to `evaluate_method` on the full matrix.
- **Also added:** `EVAL_BATCH` (memory only) and `BATCH` / `MAX_EPOCHS` for the one-epoch rule at scale (paper, Section 4.4).

## (3) Score-only from a saved checkpoint, and the scoring batch (2026-10-29)

`SCORE_ONLY=1` skips `runner.train` and loads the released best checkpoint (`model_path`). This is used when training and validation completed but our scoring step failed. `TRAIN_HOURS` records the hours of the original run. `QB` sets the scoring batch (memory only; default 256). On Reddit L, 256 queries × 6.9 M item logits ran out of memory after a 61 h epoch.

- **Tuning check (2026-11-13; paper, Appendix F.3).** Environment overrides of the hyperparameters listed there; when unset, every value equals the released run.
