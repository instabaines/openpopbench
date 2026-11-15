# FEARec (sudaada/FEARec @ 5e98f45): changes for G-repro and the benchmark

1. **Incomplete release.** The bundled `recbole/model/sequential_recommender/__init__.py` imports `cl4srec.CL4SRec`, but `cl4srec.py` is not in the repository, so the code cannot start. The import line is commented out. FEARec (`fearec.py`) does not use CL4SRec.
2. **Data.** `dataset/Amazon_Beauty/Amazon_Beauty.inter` was rebuilt from the Amazon 2014 `ratings_Beauty.csv`, because RecBole's S3 bucket returns AccessDenied. See `GREPRO_PLAN.md`, data notes.
3. **Command** taken verbatim from `fearec.sh`. It includes the release's own `lmd=0.1` token without `--`.
4. **seaborn and matplotlib installed.** `quick_start.run_recbole` imports them to plot item embeddings *before* the test evaluation, so the first full run (149 epochs) crashed without a test result.
5. **Benchmark adapter** `fearec_bench.py`:
   - **Released pieces:** the vendored RecBole 0.2.1 pipeline and trainer, with `seq.yaml`, `FEARec.yaml` and the `fearec.sh` flags.
   - **Data:**
     - Each train cascade is a user.
     - Two hold-out events (token BENCH_HOLDOUT) are appended to each cascade, so the released leave-one-out split takes only them and every real event is a training target.
     - The 5-core filters are off, and `bench.item` lists every participant.
     - Each run gets its own data_path, because `semantic_augmentation.npy` is cached there.
   - **Selection:** Recall@10 under full ranking on 5,000 validation queries.
   - **Scoring:** `full_sort_predict` over the full candidate set.
   - The embedding plot in quick_start is skipped.
