"""Build the Reddit 2012-13 benchmark tables (paper, Section 3; release = IDs + build scripts only).

Input : $REDDIT_PARSED/lines/*.parquet (parse_reddit.py on Hessel et al. ICWSM 2016 v1.1, 5,693 subreddits).
Output: $REDDIT_OUT/{events,events_all,cascades,edges,participants}.parquet, private/{id_map,cascade_map,event_map}.parquet, splits.json.

Cascade = a submission created in a split window by a non-deleted account; event = a user's first comment in the thread
within [t0, t0 + H), H = 72 h; participant = user; origin = submitter. [deleted] authors and the submitter's own comments
are excluded. Splits by t0 (gap = H; stream end 2013-01-14): train [2012-12-03, 12-28), val [12-31, 2013-01-04),
test [01-07, 01-11), UTC. Participants = every non-deleted user with any submission or comment before the stream end
(whole history from 2008), first_seen_t = earliest line. Edges = reply ties: v -> u at the first time u replied to v
(v's content reached u), any subreddit, before the stream end, among accounts active in window cascades.

Rebuild by others (scripts/reddit/README.md): REDDIT_PARSED=<parse> REDDIT_OUT=<dir> ID_PERM=reddit_id_perm.parquet python build_reddit.py.
ID_PERM numbers participants by a key-free canonical order and the released permutation, so the tables do not depend on
the private hash key of the parse. Without it (our own build) the order is by the keyed hash, as before (2026-11-08).
"""
import json
import os
import duckdb

PARSED = os.environ.get("REDDIT_PARSED", "parsed")
OUT = os.environ.get("REDDIT_OUT", "v0.1")
ID_PERM = os.environ.get("ID_PERM")   # released reddit_id_perm.parquet (reddit_id_perm.py): reproduces our participant ids from a parse made with any key
H = 72 * 3600
D = lambda s: int(duckdb.sql(f"SELECT epoch(TIMESTAMPTZ '{s} 00:00:00+00')").fetchone()[0])
W = dict(train=(D("2012-12-03"), D("2012-12-28")), val=(D("2012-12-31"), D("2013-01-04")), test=(D("2013-01-07"), D("2013-01-11")))
T_END = D("2013-01-14")

os.makedirs(f"{OUT}/private", exist_ok=True)
con = duckdb.connect(f"{OUT}/../build_tmp.duckdb" if not os.environ.get("REDDIT_OUT") else f"{OUT}_build_tmp.duckdb")
con.execute(f"SET threads=48; SET memory_limit='300GB'; SET preserve_insertion_order=false; SET temp_directory='{OUT}/../duck_tmp'")
X = con.execute
step = lambda s: print(f"[build] {s}", flush=True)
lo, hi = W["train"][0], W["test"][1]
X(f"CREATE OR REPLACE VIEW ln AS SELECT * FROM read_parquet('{PARSED}/lines/*.parquet')")

step("cascades")
split = " ".join(f"WHEN t0 >= {a} AND t0 < {b} THEN '{n}'" for n, (a, b) in W.items())
X(f"""CREATE OR REPLACE TABLE casc0 AS
      SELECT thread, any_value(sub) sub, any_value(u) s, min(t) t0, count(*) n_records FROM ln WHERE depth = 0 AND t >= {lo} AND t < {hi} GROUP BY thread""")
(n_all, n_del, n_dup), = X("SELECT count(*), count(*) FILTER (WHERE s = 0), count(*) FILTER (WHERE n_records > 1) FROM casc0").fetchall()
X(f"CREATE OR REPLACE TABLE casc0 AS SELECT *, CASE {split} END split FROM casc0 WHERE s <> 0")
X("DELETE FROM casc0 WHERE split IS NULL")   # submissions created in the gaps between splits
step("events")
X(f"""CREATE OR REPLACE TABLE ev_all AS
      SELECT l.thread, l.u p, l.t, l.cid FROM ln l JOIN casc0 c USING (thread)
      WHERE l.depth > 0 AND l.u <> 0 AND l.u <> c.s AND l.t >= c.t0 AND l.t < c.t0 + {H}""")
(early,), = X("SELECT count(*) FROM ln l JOIN casc0 c USING (thread) WHERE l.depth > 0 AND l.t < c.t0").fetchall()
X("CREATE OR REPLACE TABLE ev AS SELECT thread, p, min(t) t, arg_min(cid, (t, cid)) cid FROM ev_all GROUP BY 1, 2")
X("""CREATE OR REPLACE TABLE casc AS SELECT c.*, n, row_number() OVER (ORDER BY c.t0, c.thread) - 1 cascade_id
      FROM casc0 c JOIN (SELECT thread, count(*) n FROM ev GROUP BY 1) USING (thread)""")

step("participants and id map")
X(f"""CREATE OR REPLACE TABLE fs AS SELECT u c, min(t) first_seen_t, min(t) FILTER (WHERE depth > 0) first_part_t, min(t) FILTER (WHERE depth = 0) first_post_t
      FROM ln WHERE u <> 0 AND t < {T_END} GROUP BY 1""")
if ID_PERM:   # key-free canonical order (first line: t, is_comment, id), then the released permutation to our ids
    X(f"""CREATE OR REPLACE TABLE idm AS WITH k AS (SELECT u, min(struct_pack(t := t, is_comment := depth > 0, id := cid)) k FROM ln WHERE u <> 0 AND t < {T_END} GROUP BY u),
          r AS (SELECT u, row_number() OVER (ORDER BY k) - 1 rk FROM k),
          p AS (SELECT canonical_rank rk, participant_id FROM read_parquet('{ID_PERM}'))
          SELECT r.u, p.participant_id::BIGINT participant_id FROM r JOIN p USING (rk)""")
    assert X("SELECT count(*) FROM idm").fetchone()[0] == X("SELECT count(*) FROM fs").fetchone()[0], "permutation does not cover every participant"
else:
    X("CREATE OR REPLACE TABLE idm AS SELECT c u, row_number() OVER (ORDER BY hash(c), c) - 1 participant_id FROM fs")
X("""CREATE OR REPLACE TABLE part AS SELECT i.participant_id, i.u, f.first_seen_t, f.first_part_t, f.first_post_t,
      i.u IN (SELECT p FROM ev UNION SELECT s FROM casc) is_active FROM idm i JOIN fs f ON f.c = i.u""")

step("edges (reply ties among active accounts)")
X(f"""CREATE OR REPLACE TABLE edg AS
      WITH a AS (SELECT u, participant_id FROM part WHERE is_active)
      SELECT av.participant_id src, au.participant_id dst, min(l.t) t_created, 'reply' edge_type
      FROM ln l JOIN a au ON au.u = l.u JOIN a av ON av.u = l.pu
      WHERE l.depth > 0 AND l.pu <> l.u AND l.t < {T_END} GROUP BY 1, 2""")

step("write")
X(f"""COPY (SELECT cascade_id, o.participant_id origin_id, t0, split, n n_events FROM casc c JOIN idm o ON o.u = c.s ORDER BY cascade_id)
      TO '{OUT}/cascades.parquet' (FORMAT PARQUET, COMPRESSION zstd)""")
X(f"""COPY (WITH e AS (SELECT c.cascade_id, i.participant_id, e.t, 'comment' event_type FROM ev e JOIN casc c USING (thread) JOIN idm i ON i.u = e.p
                   UNION ALL SELECT c.cascade_id, i.participant_id, c.t0, 'origin' FROM casc c JOIN idm i ON i.u = c.s)
            SELECT e.*, row_number() OVER (PARTITION BY e.cascade_id ORDER BY (e.event_type <> 'origin'), e.t, e.participant_id) - 1 "position" FROM e
            ORDER BY e.cascade_id, "position")
      TO '{OUT}/events.parquet' (FORMAT PARQUET, COMPRESSION zstd, ROW_GROUP_SIZE 1000000)""")
X(f"""COPY (SELECT c.cascade_id, i.participant_id, e.t FROM ev_all e JOIN casc c USING (thread) JOIN idm i ON i.u = e.p ORDER BY 1, 3)
      TO '{OUT}/events_all.parquet' (FORMAT PARQUET, COMPRESSION zstd, ROW_GROUP_SIZE 1000000)""")
X(f"COPY (SELECT * EXCLUDE (u) FROM part ORDER BY participant_id) TO '{OUT}/participants.parquet' (FORMAT PARQUET, COMPRESSION zstd)")
X(f"COPY (SELECT * FROM edg ORDER BY t_created) TO '{OUT}/edges.parquet' (FORMAT PARQUET, COMPRESSION zstd, ROW_GROUP_SIZE 4000000)")
X(f"COPY idm TO '{OUT}/private/id_map.parquet' (FORMAT PARQUET, COMPRESSION zstd)")
X(f"COPY (SELECT cascade_id, sub, thread submission_id36 FROM casc) TO '{OUT}/private/cascade_map.parquet' (FORMAT PARQUET, COMPRESSION zstd)")
X(f"""COPY (SELECT c.cascade_id, i.participant_id, e.cid comment_id36 FROM ev e JOIN casc c USING (thread) JOIN idm i ON i.u = e.p)
      TO '{OUT}/private/event_map.parquet' (FORMAT PARQUET, COMPRESSION zstd)""")
json.dump(dict(version="0.1", horizon_s=H, stream_end=T_END, windows_t0={n: list(w) for n, w in W.items()},
               submissions_in_window=n_all, deleted_submitter_dropped=n_del, duplicate_submission_records=n_dup, comments_before_t0_dropped=early,
               cascade="submission created in a split window (non-deleted submitter); t0 = submission time", event="first comment by a user within [t0, t0+H)",
               edges="reply tie v->u at the first time u replied to v, among accounts active in window cascades",
               ids="submission and comment ids (base-36 values) in private maps; release = ids + build scripts"),
          open(f"{OUT}/splits.json", "w"), indent=1)
print("[build] DONE", X("SELECT split, count(*), sum(n) FROM casc GROUP BY 1 ORDER BY 1").fetchall(), flush=True)
