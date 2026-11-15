"""Build the Telegram 2024 benchmark tables (paper, Section 3).

Input : telegram_1b_compact/data_*.parquet (converted Dataverse corpus doi:10.7910/DVN/1M5KHX).
Output: $TELEGRAM_OUT/{events,events_all,cascades,edges,participants}.parquet, private/id_map.parquet, splits.json.

Cascade = an original post (source chat, post id) that is forwarded; event = the first forward of it by a chat within
[t0, t0 + H); participant = chat. t0 = the original post time (fwd_orig_date). Self-forwards are dropped.
Splits by t0 (gap = H): train [Jul 1, Sep 24), val [Oct 1, Oct 20), test [Oct 27, Nov 23), all 2024 UTC, so every
cascade's horizon ends before the next period starts. Edges = first forward from chat a seen in chat b (a -> b),
timestamped by first occurrence, built from all forwards in the corpus (any date, with or without a post id).
"""
import json
import os
import duckdb

SRC = os.environ["TELEGRAM_SRC"]
OUT = os.environ["TELEGRAM_OUT"]
H = 7 * 86400
D = lambda s: int(duckdb.sql(f"SELECT epoch(TIMESTAMPTZ '{s} 00:00:00+00')").fetchone()[0])
W = dict(train=(D("2024-07-01"), D("2024-09-24")), val=(D("2024-10-01"), D("2024-10-20")), test=(D("2024-10-27"), D("2024-11-23")))

os.makedirs(f"{OUT}/private", exist_ok=True)
con = duckdb.connect(f"{OUT}/../build_tmp.duckdb")
con.execute(f"SET threads=48; SET memory_limit='300GB'; SET preserve_insertion_order=false; SET temp_directory='{OUT}/../duck_tmp'")
X = con.execute
X(f"CREATE OR REPLACE VIEW m AS SELECT * FROM read_parquet('{SRC}/data_*.parquet')")
lo, hi = W["train"][0], W["test"][1]

print("[build] forwards", flush=True)
X(f"""CREATE OR REPLACE TABLE fw AS
      SELECT chat_id p, fwd_src_chat s, fwd_src_post k, date t, fwd_orig_date od FROM m
      WHERE fwd_src_post IS NOT NULL AND fwd_src_chat IS NOT NULL AND date IS NOT NULL AND chat_id <> fwd_src_chat""")

print("[build] cascades and events", flush=True)
split = " ".join(f"WHEN t0 >= {a} AND t0 < {b} THEN '{n}'" for n, (a, b) in W.items())
X(f"""CREATE OR REPLACE TABLE casc0 AS
      SELECT s, k, min(od) t0, CASE {split.replace('t0', 'min(od)')} END split FROM fw GROUP BY s, k
      HAVING min(od) >= {lo} AND min(od) < {hi}""")
X("DELETE FROM casc0 WHERE split IS NULL")
X(f"""CREATE OR REPLACE TABLE ev_all AS
      SELECT f.s, f.k, f.p, f.t FROM fw f JOIN casc0 c USING (s, k) WHERE f.t >= c.t0 AND f.t < c.t0 + {H}""")
X("""CREATE OR REPLACE TABLE ev AS SELECT s, k, p, min(t) t FROM ev_all GROUP BY s, k, p""")
X("""CREATE OR REPLACE TABLE casc AS SELECT c.*, n, row_number() OVER (ORDER BY c.t0, c.s, c.k) - 1 cascade_id
      FROM casc0 c JOIN (SELECT s, k, count(*) n FROM ev GROUP BY 1, 2) USING (s, k)""")

print("[build] participants and id map", flush=True)
X("""CREATE OR REPLACE TABLE first_msg AS SELECT chat_id c, min(date) t FROM m WHERE date IS NOT NULL GROUP BY 1""")
X("""CREATE OR REPLACE TABLE first_part AS SELECT p c, min(t) t FROM fw GROUP BY 1""")          # first forward of any post, whole history
X("""CREATE OR REPLACE TABLE first_src AS SELECT s c, min(od) t FROM fw GROUP BY 1""")
X("""CREATE OR REPLACE TABLE idm AS
      WITH u AS (SELECT c FROM first_msg UNION SELECT c FROM first_src UNION SELECT s FROM casc)
      SELECT c chat_id, row_number() OVER (ORDER BY hash(c), c) - 1 participant_id FROM u""")
X("""CREATE OR REPLACE TABLE part AS
      SELECT i.participant_id, i.chat_id, fm.t IS NOT NULL is_crawled, fm.t first_msg_t, fp.t first_part_t, fs.t first_origin_t,
             least(fm.t, fp.t, fs.t) first_seen_t
      FROM idm i LEFT JOIN first_msg fm ON fm.c = i.chat_id LEFT JOIN first_part fp ON fp.c = i.chat_id LEFT JOIN first_src fs ON fs.c = i.chat_id""")

print("[build] edges", flush=True)
X("""CREATE OR REPLACE TABLE edg AS
      SELECT a.participant_id src, b.participant_id dst, min(t) t_created, 'forward' edge_type
      FROM (SELECT fwd_src_chat s, chat_id p, date t FROM m WHERE fwd_src_chat IS NOT NULL AND date IS NOT NULL AND chat_id <> fwd_src_chat) f
      JOIN idm a ON a.chat_id = f.s JOIN idm b ON b.chat_id = f.p GROUP BY 1, 2""")

print("[build] write", flush=True)
X(f"""COPY (SELECT cascade_id, o.participant_id origin_id, t0, split, n n_events FROM casc c JOIN idm o ON o.chat_id = c.s ORDER BY cascade_id)
      TO '{OUT}/cascades.parquet' (FORMAT PARQUET, COMPRESSION zstd)""")
X(f"""COPY (WITH e AS (SELECT c.cascade_id, i.participant_id, e.t, 'forward' event_type FROM ev e JOIN casc c USING (s, k) JOIN idm i ON i.chat_id = e.p
                   UNION ALL SELECT c.cascade_id, i.participant_id, c.t0, 'origin' FROM casc c JOIN idm i ON i.chat_id = c.s)
            SELECT e.*, row_number() OVER (PARTITION BY e.cascade_id ORDER BY (e.event_type <> 'origin'), e.t, e.participant_id) - 1 "position",
                   e.event_type = 'forward' AND NOT coalesce(g.t_created < e.t, false) novel_source   -- never forwarded from the origin before
            FROM e JOIN casc c USING (cascade_id) JOIN idm o ON o.chat_id = c.s LEFT JOIN edg g ON g.src = o.participant_id AND g.dst = e.participant_id
            ORDER BY e.cascade_id, "position")
      TO '{OUT}/events.parquet' (FORMAT PARQUET, COMPRESSION zstd, ROW_GROUP_SIZE 1000000)""")
X(f"""COPY (SELECT c.cascade_id, i.participant_id, e.t FROM ev_all e JOIN casc c USING (s, k) JOIN idm i ON i.chat_id = e.p ORDER BY 1, 3)
      TO '{OUT}/events_all.parquet' (FORMAT PARQUET, COMPRESSION zstd, ROW_GROUP_SIZE 1000000)""")
X(f"COPY (SELECT * EXCLUDE (chat_id) FROM part ORDER BY participant_id) TO '{OUT}/participants.parquet' (FORMAT PARQUET, COMPRESSION zstd)")
X(f"COPY (SELECT * FROM edg ORDER BY t_created) TO '{OUT}/edges.parquet' (FORMAT PARQUET, COMPRESSION zstd)")
X(f"COPY idm TO '{OUT}/private/id_map.parquet' (FORMAT PARQUET, COMPRESSION zstd)")
X(f"""COPY (SELECT c.cascade_id, c.s src_chat, c.k src_post FROM casc c) TO '{OUT}/private/cascade_map.parquet' (FORMAT PARQUET, COMPRESSION zstd)""")
json.dump(dict(version="0.1", horizon_s=H, windows_t0={n: list(w) for n, w in W.items()},
               cascade="(source chat, post) forwarded; t0 = original post time", event="first forward by a chat within [t0, t0+H)",
               edges="a->b at first forward from a seen in b (all corpus forwards)"), open(f"{OUT}/splits.json", "w"), indent=1)
print("[build] DONE", con.execute("SELECT split, count(*), sum(n) FROM casc GROUP BY 1 ORDER BY 1").fetchall(), flush=True)
