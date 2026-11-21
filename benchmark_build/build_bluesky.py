"""Build the Bluesky 2025 benchmark tables (paper, Section 3).

Input : $BLUESKY_RAW/{reposts,posts,follows}.parquet (the anonymized export of Balduf et al., IMC 2024,
        snapshot of 2025-04-14: https://bsky.leobalduf.com/datasets.html).
Output: $BLUESKY_OUT/{events,events_all,cascades,edges,participants}.parquet, private/{id_map,cascade_map}.parquet, splits.json.

Cascade = a post created in a split window; event = the first repost of it by an account within [t0, t0 + H), H = 72 h;
participant = account; origin = post author. Self-reposts are dropped. Splits by t0 (gap = H; the stream ends 2025-04-14):
train [Mar 3, Mar 28), val [Mar 31, Apr 4), test [Apr 7, Apr 11), all 2025 UTC.
Participants = every account with a post or a repost before the stream end (the T1 universe); first_seen_t = earliest of
the two (whole history, so a newcomer is an account acting for the first time). Edges = follows, oriented followed -> follower
(the direction content flows), restricted to accounts that take part in or originate a window cascade; t_created = follow
creation time, null when implausible (before 2023-01-01 or after the stream end). Timestamps are client-set: events
before t0 are dropped and counted in QA. Deleted records (reposts, follows) are absent from the snapshot.
"""
import json
import os
import duckdb

RAW = os.environ["BLUESKY_RAW"]
OUT = os.environ["BLUESKY_OUT"]
ID_PERM = os.environ.get("ID_PERM")   # released bluesky_id_perm.parquet: reproduces our participant ids without DuckDB's hash() (2026-11-20)
H = 72 * 3600
D = lambda s: int(duckdb.sql(f"SELECT epoch(TIMESTAMPTZ '{s} 00:00:00+00')").fetchone()[0])
W = dict(train=(D("2025-03-03"), D("2025-03-28")), val=(D("2025-03-31"), D("2025-04-04")), test=(D("2025-04-07"), D("2025-04-11")))
T_END, T_MIN = D("2025-04-14"), D("2023-01-01")

os.makedirs(f"{OUT}/private", exist_ok=True)
con = duckdb.connect(f"{OUT}/../build_tmp.duckdb")
con.execute(f"SET threads=64; SET memory_limit='400GB'; SET preserve_insertion_order=false; SET temp_directory='{OUT}/../duck_tmp'")
X = con.execute
lo, hi = W["train"][0], W["test"][1]
step = lambda s: print(f"[build] {s}", flush=True)

step("reposts in the stream window")
X(f"""CREATE OR REPLACE TABLE rp AS
      SELECT did_id p, subject.did_id s, subject.rkey k, epoch(created_at)::BIGINT t FROM read_parquet('{RAW}/reposts.parquet')
      WHERE subject.collection = 'app.bsky.feed.post' AND created_at >= to_timestamp({lo}) AND created_at < to_timestamp({T_END})
        AND did_id <> subject.did_id""")
step("cascades (posts created in a split window that were reposted)")
split = " ".join(f"WHEN t0 >= {a} AND t0 < {b} THEN '{n}'" for n, (a, b) in W.items())
X(f"""CREATE OR REPLACE TABLE casc0 AS
      WITH ps AS (SELECT did_id s, rkey k, epoch(created_at)::BIGINT t0, reply IS NOT NULL AND reply.parent.did_id IS NOT NULL is_reply
                  FROM read_parquet('{RAW}/posts.parquet') WHERE created_at >= to_timestamp({lo}) AND created_at < to_timestamp({hi})),
           rk AS (SELECT DISTINCT s, k FROM rp)
      SELECT ps.*, CASE {split} END split FROM ps SEMI JOIN rk USING (s, k)""")
X("DELETE FROM casc0 WHERE split IS NULL")
X("CREATE OR REPLACE TABLE casc0 AS SELECT s, k, min(t0) t0, bool_or(is_reply) is_reply, min(split) split FROM casc0 GROUP BY 1, 2")  # duplicate post records
step("events")
X(f"""CREATE OR REPLACE TABLE ev_all AS SELECT r.s, r.k, r.p, r.t FROM rp r JOIN casc0 c USING (s, k) WHERE r.t >= c.t0 AND r.t < c.t0 + {H}""")
X("CREATE OR REPLACE TABLE ev AS SELECT s, k, p, min(t) t FROM ev_all GROUP BY 1, 2, 3")
X("""CREATE OR REPLACE TABLE casc AS SELECT c.*, n, row_number() OVER (ORDER BY c.t0, c.s, c.k) - 1 cascade_id
      FROM casc0 c JOIN (SELECT s, k, count(*) n FROM ev GROUP BY 1, 2) USING (s, k)""")
(early,), = X("""SELECT count(*) FROM rp r JOIN casc0 c USING (s, k) WHERE r.t < c.t0""").fetchall()

step("participants (whole history) and id map")
X(f"""CREATE OR REPLACE TABLE first_part AS SELECT did_id c, min(epoch(created_at))::BIGINT t FROM read_parquet('{RAW}/reposts.parquet')
      WHERE created_at >= to_timestamp({T_MIN}) AND created_at < to_timestamp({T_END}) GROUP BY 1""")
X(f"""CREATE OR REPLACE TABLE first_post AS SELECT did_id c, min(epoch(created_at))::BIGINT t FROM read_parquet('{RAW}/posts.parquet')
      WHERE created_at >= to_timestamp({T_MIN}) AND created_at < to_timestamp({T_END}) GROUP BY 1""")
if ID_PERM:   # canonical order (ascending did_id), then the released permutation to our ids (bluesky_id_perm.py)
    X(f"""CREATE OR REPLACE TABLE idm AS WITH u AS (SELECT c FROM first_part UNION SELECT c FROM first_post),
          r AS (SELECT c, row_number() OVER (ORDER BY c) - 1 rk FROM u)
          SELECT r.c did_id, p.participant_id::BIGINT participant_id FROM r JOIN read_parquet('{ID_PERM}') p ON p.canonical_rank = r.rk""")
    assert X("SELECT count(*) FROM idm").fetchone()[0] == X("SELECT count(*) FROM (SELECT c FROM first_part UNION SELECT c FROM first_post)").fetchone()[0]
else:
    X("""CREATE OR REPLACE TABLE idm AS WITH u AS (SELECT c FROM first_part UNION SELECT c FROM first_post)
      SELECT c did_id, row_number() OVER (ORDER BY hash(c), c) - 1 participant_id FROM u""")
X("""CREATE OR REPLACE TABLE part AS
      SELECT i.participant_id, i.did_id, fp.t first_part_t, fo.t first_post_t, least(fp.t, fo.t) first_seen_t,
             i.did_id IN (SELECT p FROM ev UNION SELECT s FROM casc) is_active
      FROM idm i LEFT JOIN first_part fp ON fp.c = i.did_id LEFT JOIN first_post fo ON fo.c = i.did_id""")

step("edges (follows among active accounts)")
X(f"""CREATE OR REPLACE TABLE edg AS
      WITH a AS (SELECT did_id, participant_id FROM part WHERE is_active)
      SELECT fo.participant_id src, fr.participant_id dst,
             min(CASE WHEN f.created_at >= to_timestamp({T_MIN}) AND f.created_at < to_timestamp({T_END}) THEN epoch(f.created_at)::BIGINT END) t_created,
             'follow' edge_type
      FROM read_parquet('{RAW}/follows.parquet') f JOIN a fr ON fr.did_id = f.did_id JOIN a fo ON fo.did_id = f.subject_id
      WHERE f.did_id <> f.subject_id AND (f.created_at < to_timestamp({T_END}) OR f.created_at >= TIMESTAMPTZ '2100-01-01') GROUP BY 1, 2""")
# follows created after the stream end are excluded; far-future (corrupt) times are kept with a null t_created

step("write")
X(f"""COPY (SELECT cascade_id, o.participant_id origin_id, t0, split, n n_events, is_reply FROM casc c JOIN idm o ON o.did_id = c.s ORDER BY cascade_id)
      TO '{OUT}/cascades.parquet' (FORMAT PARQUET, COMPRESSION zstd)""")
X(f"""COPY (WITH e AS (SELECT c.cascade_id, i.participant_id, e.t, 'repost' event_type FROM ev e JOIN casc c USING (s, k) JOIN idm i ON i.did_id = e.p
                   UNION ALL SELECT c.cascade_id, i.participant_id, c.t0, 'origin' FROM casc c JOIN idm i ON i.did_id = c.s)
            SELECT e.*, row_number() OVER (PARTITION BY e.cascade_id ORDER BY (e.event_type <> 'origin'), e.t, e.participant_id) - 1 "position" FROM e
            ORDER BY e.cascade_id, "position")
      TO '{OUT}/events.parquet' (FORMAT PARQUET, COMPRESSION zstd, ROW_GROUP_SIZE 1000000)""")
X(f"""COPY (SELECT c.cascade_id, i.participant_id, e.t FROM ev_all e JOIN casc c USING (s, k) JOIN idm i ON i.did_id = e.p ORDER BY 1, 3)
      TO '{OUT}/events_all.parquet' (FORMAT PARQUET, COMPRESSION zstd, ROW_GROUP_SIZE 1000000)""")
X(f"COPY (SELECT * EXCLUDE (did_id) FROM part ORDER BY participant_id) TO '{OUT}/participants.parquet' (FORMAT PARQUET, COMPRESSION zstd)")
X(f"COPY (SELECT * FROM edg ORDER BY t_created NULLS FIRST) TO '{OUT}/edges.parquet' (FORMAT PARQUET, COMPRESSION zstd, ROW_GROUP_SIZE 4000000)")
X(f"COPY idm TO '{OUT}/private/id_map.parquet' (FORMAT PARQUET, COMPRESSION zstd)")
X(f"COPY (SELECT cascade_id, s author_did_id, k rkey FROM casc) TO '{OUT}/private/cascade_map.parquet' (FORMAT PARQUET, COMPRESSION zstd)")
json.dump(dict(version="0.1", horizon_s=H, stream_end=T_END, windows_t0={n: list(w) for n, w in W.items()}, reposts_before_t0_dropped=early,
               cascade="post created in a split window and reposted; t0 = post created_at", event="first repost by an account within [t0, t0+H)",
               edges="follow, followed -> follower, among accounts active in window cascades; t_created null if implausible"),
          open(f"{OUT}/splits.json", "w"), indent=1)
print("[build] DONE", con.execute("SELECT split, count(*), sum(n) FROM casc GROUP BY 1 ORDER BY 1").fetchall(), flush=True)
