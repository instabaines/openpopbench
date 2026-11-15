"""Leakage test, step 1: build the "cut world" of a benchmark directory. Everything at or after the cut time T_c is
deleted, as if the data had been collected up to T_c:

  events        t < T_c
  cascades      t0 < T_c; n_events recounted; whole-cascade columns from the full stream (l_n_events, retained_share) set to NULL
  participants  every row kept (ids and vocabulary sizes unchanged, so trained checkpoints still load); first_seen_t,
                first_part_t, first_post_t set to NULL when >= T_c; is_active recomputed (seen before T_c)
  edges         t_created < T_c (edges without a creation time are kept, flagged as in the full release)

Queries at t < T_c then have the same past in both worlds, so a leakage-free pipeline must give them identical outputs.
Event rows (order t, cascade_id, position) and cascade ids before T_c are unchanged, because the deleted rows are a suffix
of both orders; this is asserted. Also writes QUERIES.parquet: the rows of FLOORS with split == test and t < T_c.

    python make_cut.py SRC_DIR DST_DIR T_C|mid FLOORS.parquet [FLOORS2.parquet ...]
"""
import json, os, shutil, sys
import duckdb

SRC, DST, TC = sys.argv[1], sys.argv[2], sys.argv[3]; FLS = sys.argv[4:]
os.makedirs(DST, exist_ok=True)
con = duckdb.connect(); con.execute("SET threads=32; SET preserve_insertion_order=true")   # keep the source row order
TC = int(TC) if TC != "mid" else int(con.execute(f"SELECT (min(t0) + max(t0)) // 2 FROM read_parquet('{SRC}/cascades.parquet') WHERE split = 'test'").fetchone()[0])   # mid = middle of the test window
r = lambda f: f"read_parquet('{SRC}/{f}.parquet')"
pcols = [c[0] for c in con.execute(f"DESCRIBE SELECT * FROM {r('participants')}").fetchall()]
ccols = [c[0] for c in con.execute(f"DESCRIBE SELECT * FROM {r('cascades')}").fetchall()]

con.execute(f"COPY (SELECT * FROM {r('events')} WHERE t < {TC}) TO '{DST}/events.parquet' (FORMAT parquet, COMPRESSION zstd)")
sel = []
for c in ccols:
    if c == "n_events": sel.append("coalesce(k.n, 0)::BIGINT n_events")
    elif c in ("l_n_events", "retained_share"): sel.append(f"NULL::{dict(l_n_events='BIGINT', retained_share='DOUBLE')[c]} {c}")
    else: sel.append(f"c.{c}")
con.execute(f"""COPY (SELECT {', '.join(sel)} FROM {r('cascades')} c
                LEFT JOIN (SELECT cascade_id, count(*) n FROM read_parquet('{DST}/events.parquet') GROUP BY 1) k USING (cascade_id)
                WHERE c.t0 < {TC}) TO '{DST}/cascades.parquet' (FORMAT parquet, COMPRESSION zstd)""")
sel = []
for c in pcols:
    if c in ("first_seen_t", "first_part_t", "first_post_t"): sel.append(f"CASE WHEN {c} < {TC} THEN {c} END {c}")
    elif c == "is_active": sel.append(f"coalesce(first_seen_t < {TC}, false) is_active")
    else: sel.append(c)
con.execute(f"COPY (SELECT {', '.join(sel)} FROM {r('participants')}) TO '{DST}/participants.parquet' (FORMAT parquet, COMPRESSION zstd)")
con.execute(f"COPY (SELECT * FROM {r('edges')} WHERE t_created < {TC} OR t_created IS NULL) TO '{DST}/edges.parquet' (FORMAT parquet, COMPRESSION zstd)")
for f in os.listdir(SRC):
    if f.endswith(".json"): shutil.copy(f"{SRC}/{f}", f"{DST}/{f}")

# the deleted rows must be a suffix of the event-row order and of the cascade ids
o = f"""SELECT row_number() OVER (ORDER BY e.t, e.cascade_id, e."position") - 1 rw, e.cascade_id, e."position", e.t FROM read_parquet('{{}}/events.parquet') e"""
bad = con.execute(f"SELECT count(*) FROM ({o.format(SRC)}) a JOIN ({o.format(DST)}) b USING (rw) WHERE a.cascade_id <> b.cascade_id OR a.\"position\" <> b.\"position\"").fetchone()[0]
n_src, n_dst = (con.execute(f"SELECT count(*) FROM read_parquet('{d}/events.parquet')").fetchone()[0] for d in (SRC, DST))
mx, nc = con.execute(f"SELECT max(cascade_id), count(*) FROM read_parquet('{DST}/cascades.parquet')").fetchone()
mn = con.execute(f"SELECT min(cascade_id) FROM {r('cascades')} WHERE t0 >= {TC}").fetchone()[0]
assert bad == 0 and (mn is None or mn > mx) and mx == nc - 1, (bad, mn, mx, nc)
st = dict(src=SRC, t_cut=TC, events=[n_src, n_dst], cascades_kept=nc,
          edges=[con.execute(f"SELECT count(*) FROM read_parquet('{d}/edges.parquet')").fetchone()[0] for d in (SRC, DST)],
          participants_seen_before_cut=con.execute(f"SELECT count(first_seen_t) FROM read_parquet('{DST}/participants.parquet')").fetchone()[0])
for i, fl in enumerate(FLS):
    q = f"{DST}/QUERIES{'' if i == 0 else i + 1}.parquet"
    con.execute(f"COPY (SELECT * FROM read_parquet('{fl}') WHERE split = 'test' AND t < {TC} ORDER BY event_row) TO '{q}' (FORMAT parquet)")
    st[os.path.basename(q)] = dict(floors=fl, queries=con.execute(f"SELECT count(*), count(*) FILTER (WHERE in_all_sample) FROM '{q}'").fetchone())
json.dump(st, open(f"{DST}/CUT.json", "w"), indent=1); print(json.dumps(st, indent=1))
