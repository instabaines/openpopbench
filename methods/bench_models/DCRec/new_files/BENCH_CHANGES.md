# Benchmark changes to the official DCRec release (HKUDS/DCRec @ 2338233)

The G-repro run used the unmodified `run_DCRec.py --dataset=reddit`, with `--gpu_id=$CUDA_VISIBLE_DEVICES` added so that RecBole stays on the assigned GPU. It scored HR@10 0.5036 and NDCG@10 0.3301, against the paper's 0.5057 and 0.3358.

**Benchmark adapter** `dcrec_bench.py`. The released modules are imported unchanged: the model, RecBole trainer, `build_adj_graph`, `build_sim_graph` and `sequential_augmentation`.

1. **Data:**
   - Each train cascade with at least 2 events becomes a session. The row's input is the last 50 earlier events and its target is the final event; the released augmentation then expands the row to every prefix.
   - `bench.item` lists every participant, so the item vocabulary covers the whole population.
2. **Configuration:** the release's `config_dict` with its tuned Reddit setting (batch 128, graphcl_coefficient 1, weight_mean 0.5, sim_group 4, kl_weight 1). The release's default branch lacks `sim_group`, `weight_mean` and `kl_weight` and raises KeyError, so it cannot run on a new dataset. Reddit is the reproduced setting and the closest domain.
3. **Test-time graphs:** `adj_graph_test` and `sim_graph_test` are the training graphs. The release builds them from the test sequences, which here would leak later events into earlier queries.
4. **Model selection:** the released `valid_metric`, Recall@1, under full ranking on 5,000 validation queries (released: pop100 sampled evaluation on the test file). Also the released `stopping_step` of 20, plus the time budget.
5. **Scoring:** the full candidate set, ranked with the released fused scorer from `fast_predict` (sequence, adjacency-graph and similarity-graph views, then a dot product with the item table).
6. **Environment:** duckdb 1.1.3 and pyarrow 12.0.1 were added to bm-dgl for reading the benchmark tables; numpy stays at 1.23.5.

**Known scaling limit (released code, unchanged):** `build_sim_graph` calls `cosine_similarity` on the session-item matrix. Its output is dense, an N × N float64 item-item matrix: 20 GB for Bluesky S (50k items), 2 TB for M (500k) and 784 GB for Telegram (313k). Tiers above S are therefore infeasible as released (host-memory bound).
