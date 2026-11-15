# MIM (CZ-TAO12/MIM @ e075dd5): changes for G-repro and the benchmark

1. **Environment.** bm-mim has Python 3.10, torch 2.1.1, torch_scatter 2.1.2, torch_sparse 0.6.18, torch_cluster 1.6.3 and torch_geometric 2.8.1, as in the repository's requirements. numpy is pinned below 2, because the code uses `np.Inf`, which NumPy 2.0 removed.
2. **`output_result/` is created before running.** The released code saves a checkpoint there but never creates the folder.
3. **Command** verbatim from the README: `python run.py -data_name=twitter`, on the bundled Twitter data.
4. **Released reporting bug (not modified; corrected when reading the log).** In `run.py` the loop that scales test scores ×100 sits inside the loop over the 6 validation metrics, so printed test scores are scaled by 100⁶ = 10¹².
    - It also records the test score of the **last** epoch, not of the best-validation epoch.
    - G-repro reads the printed numbers ÷ 10¹².
    - The benchmark adapter computes metrics itself, so it is unaffected.
5. **Benchmark adapter** `mim_bench.py`:
   - **Released pieces:** `run.model_training`, `run.model_testing`, EarlyStopping (patience 6) and the graph builders.
   - **Hypergraphs:** built from TRAIN cascades only. The released `Read_all_cascade` uses train + valid; it is patched in the adapter.
   - **Validation:** 2,000 validation cascades.
   - **Scoring:** `model_prediction` logits after the last real event, over the full candidate set.
   - **Budget:** time budget, stopping the training loader mid-epoch.
