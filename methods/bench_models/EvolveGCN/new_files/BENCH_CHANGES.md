# Benchmark changes to the official EvolveGCN release (IBM/EvolveGCN)

1. **Compatibility** (`utils.py:136`): `yaml.load(f)` became `yaml.load(f, Loader=yaml.FullLoader)`, because PyYAML ≥ 6 requires the `Loader` argument. No behavioural change.
2. Created the empty `log/` directory that `logger.py` writes to; the release does not ship it.
3. Compatibility (`logger.py:237`): `np.float` became `np.float64`, the same type; the alias was removed in NumPy 1.24.
4. **Benchmark adapter** `evolvegcn_bench.py`:
   - **Released pieces:** EGCN-O, `Link_Pred_Tasker`, Classifier, `Cross_Entropy`, `Trainer.run_epoch`, and the reproduced config `parameters_sbm_egcn_o.yaml`.
   - **Snapshots:** aggr_time = max(1 h, train window / 100), fixed before any result.
   - **Graph:** each event gives both directed edges. With one direction only, every participant has out-degree 0.
   - **Training:** samples come only from training-window snapshots.
   - **Selection:** the released target measure (MAP, which equals MRR with one target) under full ranking on 5,000 validation queries. The released dev pass scores all N² pairs.
   - **Scoring:** a query in snapshot k uses embeddings from snapshots up to k−1 only. The score is logit(link) − logit(no link).

**Change (3), 2026-10-21: empty label snapshots.** Training samples whose label snapshot window holds no edge are dropped.
- **Why:** this happens on the sparse Reddit S tier. The released `get_non_existing_edges` draws its negatives with `np.random.choice` over the label edges, which fails on an empty array.
- **Effect:** no other sample changes. The number dropped is logged.

**Change (4), 2026-10-22: empty label snapshots at scoring.** At scoring time only the node embeddings of `get_sample` are used, not its label negatives. When the label snapshot after a query is empty (late test queries on Reddit S), the released negative sampler fails, so it returns an empty negative set there. Training is unaffected.
