# Benchmark construction

The scripts that built the v1.0 tables, tiers, queries and task files. Users of the benchmark do not need them: the released tiers on Hugging Face are their output. `scripts/reddit/rebuild.sh` runs the same steps for Reddit, in order, and checks the result against the v1.0 content hashes.

Working folders are set by environment variables: `OPB_WORK` (default `./work`) for derived files, and `BLUESKY_RAW`/`BLUESKY_OUT`, `TELEGRAM_SRC`/`TELEGRAM_OUT`, `REDDIT_PARSED`/`REDDIT_OUT` for the per-source builds.

| step | script | paper |
|---|---|---|
| 1. Source tables | `build_bluesky.py`, `build_telegram.py`, `parse_reddit.py` + `build_reddit.py` (+ `reddit_id_perm.py`, the permutation file) | §3.1, Appendix A.1 |
| 2. Integrity checks | `check_integrity.py` | §3.5, Appendix A |
| 3. Size tiers (Forest Fire) | `ff_tiers.py`; selection of p_f on validation: `validate_grid.py`, `validate_tiers.py` | §3.3, Appendix A.3 |
| 4. Queries and heuristic ranks | `floors_eval.py` (with `bench_state.py`); 10% subsample: `make_query_subsample.py` | §3.4, §4.3, Appendix B.1 |
| 5. Task 2 and Task 3 files | `t2_flags.py` (newcomer labels), `t2a_floors.py` (Task 2a baselines), `t4_time.py` (Task 3 baselines) | §2.2, Appendix B |
| 6. Controls | `b1_ranker.py` (learned feature ranker with `MODEL=b1`, exposure Hawkes with `MODEL=b6`) | Appendix B.2, B.3 |
| 7. Leaderboards | `leaderboard.py` (Task 1), `t2b_newcomer.py` (Task 2b), `t4_time.py board` (Task 3); `openpopbench` reproduces them | §5 |
| Checksums | `content_hash.py` | §6.2 |
| Feasibility probe | `probe_util.py`, used by the adapters | Appendix D |
| Leakage test | `leakage/`, driven by `opb leakage` | §6.1, Appendix F.1 |

Names: the scripts use the internal names of the README's *Names in the code* (T1, T4, floors).
