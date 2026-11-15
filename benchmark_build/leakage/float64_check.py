"""Diagnostic for the leakage test: recompute the shared features (bench_state.Stream) in float64 on the original and the
cut world and report the largest relative difference. Summation-order noise is ~1e-15; information would be far larger.
    python float64_check.py ORIG_DIR CUT_DIR"""
import sys, json, duckdb, numpy as np, pandas as pd, torch
src = open("benchmark_build/bench_state.py").read()
assert src.count("], 1).float()") == 1; src = src.replace("], 1).float()", "], 1)")      # keep float64
ns = {}; exec(compile(src, "bench_state_f64", "exec"), ns); Stream = ns["Stream"]
O, C = sys.argv[1:3]
ss = int(duckdb.sql(f"SELECT min(t0) FROM read_parquet('{O}/cascades.parquet') WHERE split='test'").fetchone()[0])
q = pd.read_parquet(f"{C}/QUERIES.parquet").event_row.values
S = [Stream(d, torch.device("cuda:0"), self_exc_scales=(3600.0, 86400.0)) for d in (O, C)]
rel = 0.0; nd = 0
for j in q:
    (_, c1, X1), (_, c2, X2) = S[0].query(int(j), ss), S[1].query(int(j), ss)
    assert bool((c1 == c2).all()); m = c1
    d = (X1[m] - X2[m]).abs(); r = float((d / X1[m].abs().clamp(min=1e-300)).max()); nd += int(d.max() > 0); rel = max(rel, r)
res = dict(orig=O, queries=len(q), queries_with_any_float64_difference=nd, max_relative_difference=rel)
json.dump(res, open(f"{C}/RESULT_float64.json", "w"), indent=1); print(res)
