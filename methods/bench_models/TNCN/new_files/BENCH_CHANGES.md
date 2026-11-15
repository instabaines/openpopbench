# TNCN (GraphPKU/TNCN @ 4c93a2a): G-repro notes

1. **The README example command runs one epoch only.** The vendored TGB argument parser defaults to `patience=0`. That run gave test MRR 0.347, against 0.718 on the leaderboard, and is not a valid reproduction attempt.
2. **Rerun with the paper's own tgbl-wiki hyperparameters** (arXiv 2406.07926, Table 5): `--num_neighbors 15 --num_epoch 20 --patience 5 --mem_dim 184 --emb_dim 184 --time_dim 100`. This is no tuning: the values are the authors'.
3. **Data:** the TGB v2 archive from TGB's current host, symlinked into the vendored `tgb/datasets/tgbl_wiki`.
4. **Benchmark adapter** `tncn_bench.py`:
   - **Released pieces:** the TNCN modules and `train()`, copied verbatim, because the example script runs at import.
   - **Hyperparameters:** the paper's Table 5 row for the TGB dataset nearest in edge count. Bluesky S maps to Review, Telegram to Coin.
   - **Messages:** a zero 1-dim message, since there are no edge features.
   - **Validation:** the released one-vs-many MRR with 20 uniform negatives, on 5,000 evenly spaced validation edges. Every validation edge still updates memory and neighbours.
   - **Edge ids:** the run asserts that train edges form a prefix of the stream, so the neighbour loader's insertion ids equal global edge ids.
   - **Scoring:** the released `test()` path, for every candidate, on the 10% query subsample (paper, Section 3.4).
