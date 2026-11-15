"""Leakage test, step 2: the sanity floors (floors_eval.py) and the shared feature code (bench_state.Stream, used by the
learned rankers) on the original and the cut world, same queries. A leakage-free pipeline gives identical outputs; the
oracle-leak floor exposure_leak (edges without their creation time) is the positive control and must differ.

    python test_floors_stream.py ORIG_DIR CUT_DIR DEVICE
"""
import json, os, subprocess, sys
import numpy as np, pandas as pd, torch
BB = os.path.join(os.environ.get("OPB_ROOT", "."), "benchmark_build"); sys.path.insert(0, BB)
from bench_state import Stream, FEATURES

ORIG, CUT, DEV = sys.argv[1], sys.argv[2], sys.argv[3]
Q = f"{CUT}/QUERIES.parquet"; res = dict(orig=ORIG, cut=CUT, queries=Q)
env = dict(os.environ, QUERIES=Q)
for tag, d in (("orig", ORIG), ("cut", CUT)):
    subprocess.run([sys.executable, "-u", f"{BB}/floors_eval.py", d, f"{CUT}/floors_{tag}.parquet", "0", DEV], env=env, check=True)
a, b = (pd.read_parquet(f"{CUT}/floors_{t}.parquet") for t in ("orig", "cut"))
q = pd.read_parquet(Q)
assert np.array_equal(a.event_row.values, q.event_row.values) and np.array_equal(b.event_row.values, q.event_row.values)
fl = {}
for c in [c for c in a.columns if c != "event_row"]:
    same = (a[c].values == b[c].values) if a[c].dtype != float else np.isclose(a[c].values, b[c].values, rtol=0, atol=0)
    fl[c] = dict(identical=int(same.sum()), of=len(a))
fl["stored_floor_file_rank_popularity_identical"] = int((a.rank_popularity.values == q.rank_popularity.values).sum())
res["floors"] = fl

# Stream: candidate mask and every feature (plus two self-excitation scales) per query, exact comparison
S = [Stream(d, torch.device(DEV), self_exc_scales=(3600.0, 86400.0)) for d in (ORIG, CUT)]
names = FEATURES + ["self1h", "self1d"]; mx = np.zeros(len(names)); ncd = nbad = 0; split_start = int(__import__("duckdb").sql(f"SELECT min(t0) FROM read_parquet('{ORIG}/cascades.parquet') WHERE split = 'test'").fetchone()[0])
for j in q.event_row.values:
    (y1, c1, X1), (y2, c2, X2) = S[0].query(int(j), split_start), S[1].query(int(j), split_start)
    assert y1 == y2
    ncd += int((c1 != c2).sum()); m = c1 | c2
    d = (X1[m] - X2[m]).abs().amax(0).cpu().numpy(); mx = np.maximum(mx, d); nbad += int(d.max() > 0)
res["stream"] = dict(queries=len(q), candidate_mask_differences=ncd, queries_with_any_feature_difference=nbad,
                     max_abs_difference=dict(zip(names, map(float, mx))))
json.dump(res, open(f"{CUT}/RESULT_floors_stream.json", "w"), indent=1); print(json.dumps(res, indent=1))
