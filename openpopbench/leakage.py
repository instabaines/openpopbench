"""Black-box leakage test (paper Section 6.1): a pipeline that uses only the past gives identical outputs, for every query
before a cut time T_c, on the full data and on the data with everything at or after T_c deleted.

    opb leakage cut TIER DEST [--tc mid|<unix seconds>]
        writes the cut world DEST/tables/ and DEST/QUERIES.parquet, DEST/QUERIES_sub10.parquet (test queries with t < T_c)
    opb leakage test METHOD TIER --run RUN_DIR --cut DEST [--seed 1] [--gpu 0] [--repeat] [--train-log FILE]
        scores the checkpoint saved in RUN_DIR (by `opb run`) on both worlds, same queries, through the adapter's
        score-only mode, and compares the ranks. --repeat scores the full world twice to measure run-to-run GPU noise.

Cut world: events with t < T_c; cascades with t0 < T_c (n_events recounted, whole-cascade summaries set to NULL);
participants all kept (so checkpoints load) with first-seen times >= T_c set to NULL; edges created before T_c, plus
edges without a creation time. The deleted rows are a suffix of the event-row order, so query row ids are unchanged
(asserted). Verdict: PASS when every query has the same rank in both worlds, or when the differences are no larger than
the run-to-run noise measured with --repeat.
"""
import glob, json, os, shutil, subprocess
import numpy as np, pandas as pd

from . import data, registry


def cut(tier, dst, tc="mid"):
    import duckdb
    src = data.tables_dir(tier); T = os.path.join(dst, "tables"); os.makedirs(T, exist_ok=True)
    con = duckdb.connect(); con.execute("SET preserve_insertion_order=true")
    r = lambda f: f"read_parquet('{src}/{f}.parquet')"
    TC = int(tc) if tc != "mid" else int(con.execute(f"SELECT (min(t0) + max(t0)) // 2 FROM {r('cascades')} WHERE split = 'test'").fetchone()[0])
    pcols = [c[0] for c in con.execute(f"DESCRIBE SELECT * FROM {r('participants')}").fetchall()]
    ccols = [c[0] for c in con.execute(f"DESCRIBE SELECT * FROM {r('cascades')}").fetchall()]
    con.execute(f"COPY (SELECT * FROM {r('events')} WHERE t < {TC}) TO '{T}/events.parquet' (FORMAT parquet, COMPRESSION zstd)")
    sel = []
    for c in ccols:
        if c == "n_events": sel.append("coalesce(k.n, 0)::BIGINT n_events")
        elif c in ("l_n_events", "retained_share"): sel.append(f"NULL::{dict(l_n_events='BIGINT', retained_share='DOUBLE')[c]} {c}")
        else: sel.append(f"c.{c}")
    con.execute(f"""COPY (SELECT {', '.join(sel)} FROM {r('cascades')} c
                    LEFT JOIN (SELECT cascade_id, count(*) n FROM read_parquet('{T}/events.parquet') GROUP BY 1) k USING (cascade_id)
                    WHERE c.t0 < {TC}) TO '{T}/cascades.parquet' (FORMAT parquet, COMPRESSION zstd)""")
    sel = []
    for c in pcols:
        if c in ("first_seen_t", "first_part_t", "first_post_t"): sel.append(f"CASE WHEN {c} < {TC} THEN {c} END {c}")
        elif c == "is_active": sel.append(f"coalesce(first_seen_t < {TC}, false) is_active")
        else: sel.append(c)
    con.execute(f"COPY (SELECT {', '.join(sel)} FROM {r('participants')}) TO '{T}/participants.parquet' (FORMAT parquet, COMPRESSION zstd)")
    con.execute(f"COPY (SELECT * FROM {r('edges')} WHERE t_created < {TC} OR t_created IS NULL) TO '{T}/edges.parquet' (FORMAT parquet, COMPRESSION zstd)")
    o = """SELECT row_number() OVER (ORDER BY e.t, e.cascade_id, e."position") - 1 rw, e.cascade_id, e."position" FROM read_parquet('{}/events.parquet') e"""
    bad = con.execute(f"SELECT count(*) FROM ({o.format(src)}) a JOIN ({o.format(T)}) b USING (rw) WHERE a.cascade_id <> b.cascade_id OR a.\"position\" <> b.\"position\"").fetchone()[0]
    mx, nc = con.execute(f"SELECT max(cascade_id), count(*) FROM read_parquet('{T}/cascades.parquet')").fetchone()
    mn = con.execute(f"SELECT min(cascade_id) FROM {r('cascades')} WHERE t0 >= {TC}").fetchone()[0]
    assert bad == 0 and (mn is None or mn > mx) and mx == nc - 1, "cut is not a suffix of the event and cascade order"
    st = dict(tier=tier, t_cut=TC)
    for name, q in (("QUERIES", "queries"), ("QUERIES_sub10", "queries_sub10")):
        f = data.path(tier, q)
        if not os.path.exists(f): continue
        x = pd.read_parquet(f); x = x[(x.split == "test") & (x.t < TC)].sort_values("event_row")
        x.to_parquet(os.path.join(dst, name + ".parquet"), index=False); st[name] = len(x)
    json.dump(st, open(os.path.join(dst, "CUT.json"), "w"), indent=1)
    return st


def _score(method, tables, queries, out, seed, gpu, run_dir, train_log):
    """Copy the saved checkpoint of RUN_DIR into a fresh OUT and run the adapter in score-only mode."""
    m = registry.METHODS[method]; so = m.get("score_only")
    if not so: raise SystemExit(f"{method} has no score-only mode; the leakage test needs one")
    shutil.rmtree(out, ignore_errors=True); os.makedirs(out)
    pat = so["checkpoint"].format(SEED=seed); src = glob.glob(os.path.join(run_dir, pat))
    if not src: raise SystemExit(f"no checkpoint {pat} in {run_dir}")
    if "{CKPT}" in so["env"]:
        env = so["env"].format(CKPT=os.path.abspath(src[0]), TRAINLOG="")     # adapters that take the checkpoint path directly
    else:
        for f in src:
            d = os.path.join(out, os.path.relpath(f, run_dir)); os.makedirs(os.path.dirname(d), exist_ok=True)
            (shutil.copytree if os.path.isdir(f) else shutil.copy)(f, d)
        env = so["env"].format(CKPT="", TRAINLOG=train_log or os.path.join(run_dir, f"run_seed{seed}.log"))
    cmd = f"export CUDA_VISIBLE_DEVICES={gpu} OPB_ROOT={registry.NP} {env}; " + registry.render(method, tables, queries, out, seed, 8)
    rc = subprocess.run(["bash", "-c", cmd], stdout=open(os.path.join(out, "score.log"), "w"), stderr=subprocess.STDOUT).returncode
    if rc: raise SystemExit(f"adapter failed (rc {rc}); see {out}/score.log")
    x = pd.read_parquet(registry.output_file(method, out, seed))
    return x.set_index("event_row")[f"rank_{m['key']}"]


def test(method, tier, run_dir, cut_dir, seed=1, gpu="0", repeat=False, train_log=None):
    m = registry.METHODS[method]
    q = os.path.join(cut_dir, "QUERIES_sub10.parquet" if m["queries"] == "sub10" else "QUERIES.parquet")
    W = os.path.join(cut_dir, "runs", method)
    a = _score(method, data.tables_dir(tier), q, os.path.join(W, "full"), seed, gpu, run_dir, train_log)
    b = _score(method, os.path.join(cut_dir, "tables"), q, os.path.join(W, "cut"), seed, gpu, run_dir, train_log).reindex(a.index)
    d = (a - b).abs(); res = dict(method=method, tier=tier, queries=len(a), identical=int((d == 0).sum()), max_rank_diff=float(d.max()))
    if repeat:
        a2 = _score(method, data.tables_dir(tier), q, os.path.join(W, "full_repeat"), seed, gpu, run_dir, train_log).reindex(a.index)
        n = (a - a2).abs(); res.update(noise_identical=int((n == 0).sum()), noise_max_rank_diff=float(n.max()))
    ok = res["identical"] == res["queries"] or (repeat and res["max_rank_diff"] <= res["noise_max_rank_diff"]
                                                 and res["queries"] - res["identical"] <= 2 * max(1, res["queries"] - res["noise_identical"]))
    res["verdict"] = "PASS" if ok else ("CHECK: differences with no noise measurement; rerun with --repeat" if not repeat else "FAIL")
    json.dump(res, open(os.path.join(W, "RESULT.json"), "w"), indent=1)
    return res
