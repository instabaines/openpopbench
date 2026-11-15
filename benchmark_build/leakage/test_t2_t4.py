"""Leakage test, step 2b: the T2a newcomer floors (t2a_floors.py) and the T4 timing floors (t4_time.py floors) on the
original and the cut world. The released scripts run unchanged; only their tier list and output folder (both read from
t2_flags) are pointed at the two worlds. Queries: the floors rows of validation (T2a floors are fitted there) plus test
rows before the cut, all-events sample. Labels (unseen_train) are computed as in t2_flags.py.

    python test_t2_t4.py ORIG_DIR CUT_DIR FLOORS.parquet
"""
import json, os, runpy, sys, types
import duckdb, numpy as np, pandas as pd
BB = os.path.join(os.environ.get("OPB_ROOT", "."), "benchmark_build"); sys.path.insert(0, BB)
ORIG, CUT, FL = (os.path.abspath(x) for x in sys.argv[1:4])
TC = json.load(open(f"{CUT}/CUT.json"))["t_cut"]; R = f"{CUT}/t2t4"
for d in ("eval/t2", "eval/t4", "baselines"): os.makedirs(f"{R}/{d}", exist_ok=True)
Q = f"{R}/QUERIES_val_test.parquet"
con = duckdb.connect()
con.execute(f"COPY (SELECT * FROM read_parquet('{FL}') WHERE in_all_sample AND (split = 'val' OR (split = 'test' AND t < {TC})) ORDER BY event_row) TO '{Q}' (FORMAT parquet)")
tiers = {"orig": (ORIG, Q), "cut": (CUT, Q)}
for tier, (B, _) in tiers.items():   # t2_flags.py query (labels)
    con.execute(f"""COPY (WITH ev AS (SELECT e.participant_id, e.t, c.split FROM read_parquet('{B}/events.parquet') e JOIN read_parquet('{B}/cascades.parquet') c USING (cascade_id)),
                 tr AS (SELECT DISTINCT participant_id FROM ev WHERE split = 'train'),
                 fp AS (SELECT participant_id, min(t) t0 FROM ev GROUP BY participant_id),
                 fl AS (SELECT event_row, participant_id, t, split, in_all_sample, in_novel_sample, novel FROM read_parquet('{Q}'))
            SELECT fl.event_row, fl.split, fl.in_all_sample, fl.in_novel_sample, fl.novel,
                   tr.participant_id IS NULL AS unseen_train, fp.t0 >= fl.t AS first_part
            FROM fl LEFT JOIN tr USING (participant_id) LEFT JOIN fp USING (participant_id) ORDER BY event_row) TO '{R}/eval/t2/flags_{tier}.parquet' (FORMAT parquet)""")
sys.modules["t2_flags"] = types.SimpleNamespace(BB=R, TIERS=tiers)
runpy.run_path(f"{BB}/t2a_floors.py", run_name="__main__")
sys.argv = ["t4_time.py", "floors"]; runpy.run_path(f"{BB}/t4_time.py", run_name="__main__")

res = dict(queries=Q, n=dict(zip(*np.unique(pd.read_parquet(Q).split, return_counts=True))))
res["n"] = {k: int(v) for k, v in res["n"].items()}
for name, f in (("t2a", "eval/t2/t2a_{}.parquet"), ("t2_flags", "eval/t2/flags_{}.parquet"), ("t4", "eval/t4/floors_{}.parquet")):
    a, b = (pd.read_parquet(f"{R}/{f.format(w)}") for w in ("orig", "cut"))
    assert np.array_equal(a.event_row.values, b.event_row.values)
    res[name] = {c: dict(identical=int((a[c].values == b[c].values).sum()), of=len(a), max_abs_diff=(float(np.nanmax(np.abs(a[c].values - b[c].values))) if a[c].dtype.kind == "f" else None))
                 for c in a.columns if c not in ("event_row", "cascade_id", "split")}
json.dump(res, open(f"{CUT}/RESULT_t2_t4.json", "w"), indent=1); print(json.dumps(res, indent=1))
