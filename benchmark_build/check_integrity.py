"""Data-integrity tests (paper, Appendix A). Every check prints PASS or FAIL with its evidence.

    python check_integrity.py BENCH_DIR {telegram|bluesky|tier}

A4 and A5 compare against the raw source data, so they run only for telegram and bluesky (L), not for derived tiers.
"""
import json
import sys
import duckdb

B, KIND = sys.argv[1], sys.argv[2]
con = duckdb.connect(); con.execute("SET threads=48; SET memory_limit='300GB'")
for t in ("events", "cascades", "participants", "edges"):
    con.execute(f"CREATE VIEW {t} AS SELECT * FROM read_parquet('{B}/{t}.parquet')")
q1 = lambda s: con.execute(s).fetchone()
meta = json.load(open(f"{B}/splits.json")) if KIND != "tier" else json.load(open(f"{B}/tier.json"))["splits"]
H, W = meta["horizon_s"], meta["windows_t0"]
results = []


def check(name, ok, evidence):
    results.append((name, bool(ok), evidence)); print(f"[{'PASS' if ok else 'FAIL'}] {name}: {evidence}", flush=True)


# A1 ids
n, mx, nd = q1("SELECT count(*), max(participant_id), count(DISTINCT participant_id) FROM participants")
check("A1 participant ids dense and unique", n == mx + 1 == nd, f"{n:,} rows, max id {mx:,}, distinct {nd:,}")
n, mx, nd = q1("SELECT count(*), max(cascade_id), count(DISTINCT cascade_id) FROM cascades")
check("A1 cascade ids dense and unique", n == mx + 1 == nd, f"{n:,} rows, max id {mx:,}")
cols = {r[0] for r in con.execute("DESCRIBE SELECT * FROM participants").fetchall()} | {r[0] for r in con.execute("DESCRIBE SELECT * FROM events").fetchall()}
check("A1 no raw platform ids in released tables", not cols & {"chat_id", "did_id", "rkey", "src_chat", "src_post"}, f"columns {sorted(cols)}")
if KIND != "tier":
    raw = {"telegram": "chat_id", "bluesky": "did_id", "reddit": "u"}[KIND]
    a, b, c_ = q1(f"SELECT count(*), count(DISTINCT participant_id), count(DISTINCT {raw}) FROM read_parquet('{B}/private/id_map.parquet')")
    n_part = q1("SELECT count(*) FROM participants")[0]
    check("A1 private id map bijective and complete", a == b == c_ == n_part, f"{a:,} rows, {b:,} ids, {c_:,} raw ids, {n_part:,} participants")

# A2 events
bad = q1(f"SELECT count(*) FROM events e JOIN cascades c USING (cascade_id) WHERE e.t < c.t0 OR e.t >= c.t0 + {H}")[0]
check("A2 every event in [t0, t0+H)", bad == 0, f"{bad:,} violations")
bad = q1("""SELECT count(*) FROM (SELECT cascade_id, count(*) n, max("position") mp, count(*) FILTER (WHERE "position" = 0) n0,
            count(*) FILTER (WHERE "position" = 0 AND event_type = 'origin') o0, count(*) FILTER (WHERE event_type = 'origin') n_or FROM events GROUP BY 1)
            WHERE mp <> n - 1 OR n0 <> 1 OR o0 <> 1 OR n_or <> 1""")[0]
check("A2 positions contiguous; exactly one origin, at position 0", bad == 0, f"{bad:,} bad cascades")
bad = q1("SELECT count(*) - count(DISTINCT (cascade_id, participant_id)) FROM events")[0]
check("A2 each participant at most once per cascade (origin included)", bad == 0, f"{bad:,} duplicates")
bad = q1("SELECT count(*) FROM events e JOIN cascades c USING (cascade_id) WHERE e.event_type <> 'origin' AND e.participant_id = c.origin_id")[0]
check("A2 no self-participation", bad == 0, f"{bad:,}")
bad = q1("""SELECT count(*) FROM (SELECT cascade_id, "position", t, lag(t) OVER (PARTITION BY cascade_id ORDER BY "position") pt FROM events) WHERE t < pt""")[0]
check("A2 positions follow time order", bad == 0, f"{bad:,}")
bad = q1("SELECT count(*) FROM events e JOIN cascades c USING (cascade_id) WHERE e.event_type = 'origin' AND (e.participant_id <> c.origin_id OR e.t <> c.t0)")[0]
check("A2 origin event matches cascades.origin_id and t0", bad == 0, f"{bad:,}")

# A3 splits
for s, (a, b) in W.items():
    bad = q1(f"SELECT count(*) FROM cascades WHERE split = '{s}' AND (t0 < {a} OR t0 >= {b})")[0]
    check(f"A3 {s} t0 inside its window", bad == 0, f"{bad:,} outside [{a}, {b})")
check("A3 horizons end before the next split starts", W["train"][1] + H <= W["val"][0] and W["val"][1] + H <= W["test"][0],
      f"train end + H = {W['train'][1] + H}, val start {W['val'][0]}; val end + H = {W['val'][1] + H}, test start {W['test'][0]}")
bad = q1(f"""SELECT count(*) FROM events e JOIN cascades c USING (cascade_id)
             WHERE (c.split = 'train' AND e.t >= {W['val'][0]}) OR (c.split = 'val' AND e.t >= {W['test'][0]})""")[0]
check("A3 no train event after val start, no val event after test start", bad == 0, f"{bad:,}")

# A4 counts
bad = q1("SELECT count(*) FROM cascades c JOIN (SELECT cascade_id, count(*) - 1 n FROM events GROUP BY 1) e USING (cascade_id) WHERE c.n_events <> e.n")[0]
check("A4 n_events equals event count", bad == 0, f"{bad:,} mismatches")
if KIND == "bluesky":
    RAW = os.environ["BLUESKY_RAW"]
    con.execute(f"""CREATE TABLE smp AS SELECT * FROM (SELECT m.*, c.t0, c.n_events FROM read_parquet('{B}/private/cascade_map.parquet') m JOIN cascades c USING (cascade_id))
                    USING SAMPLE reservoir(1000 ROWS) REPEATABLE (7)""")
    bad, tot = q1(f"""WITH r AS (SELECT s.cascade_id, count(DISTINCT x.did_id) n FROM smp s JOIN read_parquet('{RAW}/reposts.parquet') x
                      ON x.subject.did_id = s.author_did_id AND x.subject.rkey = s.rkey AND x.subject.collection = 'app.bsky.feed.post'
                      WHERE epoch(x.created_at)::BIGINT >= s.t0 AND epoch(x.created_at)::BIGINT < s.t0 + {H} AND x.created_at < to_timestamp({meta['stream_end']}) AND x.did_id <> s.author_did_id
                      GROUP BY 1)
                     SELECT count(*) FILTER (WHERE coalesce(r.n, 0) <> s.n_events), count(*) FROM smp s LEFT JOIN r USING (cascade_id)""")
    check("A4 reconciliation with raw reposts (1,000 random cascades)", bad == 0, f"{bad} of {tot} differ")
    bad, tot = q1(f"""WITH p AS (SELECT s.cascade_id, min(epoch(x.created_at)::BIGINT) t0 FROM smp s JOIN read_parquet('{RAW}/posts.parquet') x
                      ON x.did_id = s.author_did_id AND x.rkey = s.rkey GROUP BY 1)
                     SELECT count(*) FILTER (WHERE p.t0 IS DISTINCT FROM s.t0), count(*) FROM smp s LEFT JOIN p USING (cascade_id)""")
    check("A4 t0 equals the raw post created_at (1,000 cascades)", bad == 0, f"{bad} of {tot} differ")
    con.execute(f"""CREATE TABLE es AS SELECT g.*, a.did_id src_did, b.did_id dst_did FROM (SELECT * FROM edges USING SAMPLE reservoir(100 ROWS) REPEATABLE (7)) g
                    JOIN read_parquet('{B}/private/id_map.parquet') a ON a.participant_id = g.src JOIN read_parquet('{B}/private/id_map.parquet') b ON b.participant_id = g.dst""")
    bad, tot = q1(f"""SELECT count(*) FILTER (WHERE NOT ok), count(*) FROM (SELECT e.src, e.dst, bool_or(f.did_id IS NOT NULL AND (e.t_created IS NULL OR epoch(f.created_at)::BIGINT = e.t_created)) ok
                      FROM es e LEFT JOIN read_parquet('{RAW}/follows.parquet') f ON f.did_id = e.dst_did AND f.subject_id = e.src_did GROUP BY 1, 2)""")
    check("A5 edge direction (followed -> follower) and time match raw follows (100 edges)", bad == 0, f"{bad} of {tot} unmatched")
if KIND == "telegram":
    SRC = os.environ["TELEGRAM_SRC"]
    con.execute(f"""CREATE TABLE smp AS SELECT * FROM (SELECT m.*, c.t0, c.n_events FROM read_parquet('{B}/private/cascade_map.parquet') m JOIN cascades c USING (cascade_id))
                    USING SAMPLE reservoir(1000 ROWS) REPEATABLE (7)""")
    bad, tot = q1(f"""WITH f AS (SELECT x.* FROM read_parquet('{SRC}/data_*.parquet') x SEMI JOIN smp s ON x.fwd_src_chat = s.src_chat AND x.fwd_src_post = s.src_post),
                      r AS (SELECT s.cascade_id, min(f.fwd_orig_date) FILTER (WHERE f.chat_id <> s.src_chat AND f.date IS NOT NULL) t0, count(DISTINCT f.chat_id) FILTER (WHERE f.date >= s.t0 AND f.date < s.t0 + {H} AND f.chat_id <> s.src_chat) n
                            FROM smp s JOIN f ON f.fwd_src_chat = s.src_chat AND f.fwd_src_post = s.src_post GROUP BY 1)
                     SELECT count(*) FILTER (WHERE r.n IS DISTINCT FROM s.n_events OR r.t0 IS DISTINCT FROM s.t0), count(*) FROM smp s LEFT JOIN r USING (cascade_id)""")
    check("A4 reconciliation with the converted corpus (1,000 cascades: n_events and t0)", bad == 0, f"{bad} of {tot} differ")
    con.execute(f"""CREATE TABLE es AS SELECT g.*, a.chat_id src_chat, b.chat_id dst_chat FROM (SELECT * FROM edges USING SAMPLE reservoir(100 ROWS) REPEATABLE (7)) g
                    JOIN read_parquet('{B}/private/id_map.parquet') a ON a.participant_id = g.src JOIN read_parquet('{B}/private/id_map.parquet') b ON b.participant_id = g.dst""")
    bad, tot = q1(f"""WITH f AS (SELECT x.chat_id, x.fwd_src_chat, min(x.date) t FROM read_parquet('{SRC}/data_*.parquet') x
                      SEMI JOIN es e ON x.chat_id = e.dst_chat AND x.fwd_src_chat = e.src_chat GROUP BY 1, 2)
                     SELECT count(*) FILTER (WHERE f.t IS DISTINCT FROM e.t_created), count(*) FROM es e LEFT JOIN f ON f.chat_id = e.dst_chat AND f.fwd_src_chat = e.src_chat""")
    check("A5 edge direction (source -> forwarding chat) and first-occurrence time (100 edges)", bad == 0, f"{bad} of {tot} differ")

if KIND == "reddit":
    PARSED = "parsed"
    RAWD = "Hessel_ICWSM_2016/only_meta_and_stats"
    con.execute(f"""CREATE TABLE smp AS SELECT * FROM (SELECT m.*, c.t0, c.n_events FROM read_parquet('{B}/private/cascade_map.parquet') m JOIN cascades c USING (cascade_id))
                    USING SAMPLE reservoir(1000 ROWS) REPEATABLE (7)""")
    bad, tot = q1(f"""WITH l AS (SELECT x.* FROM read_parquet('{PARSED}/lines/*.parquet') x SEMI JOIN smp s ON x.thread = s.submission_id36),
                      sb AS (SELECT thread, min(t) FILTER (WHERE depth = 0) t0, any_value(u) FILTER (WHERE depth = 0) s FROM l GROUP BY 1),
                      r AS (SELECT s.cascade_id, sb.t0, count(DISTINCT l.u) FILTER (WHERE l.depth > 0 AND l.u <> 0 AND l.u <> sb.s AND l.t >= sb.t0 AND l.t < sb.t0 + {H}) n
                            FROM smp s JOIN sb ON sb.thread = s.submission_id36 JOIN l ON l.thread = s.submission_id36 GROUP BY 1, 2)
                     SELECT count(*) FILTER (WHERE r.n IS DISTINCT FROM s.n_events OR r.t0 IS DISTINCT FROM s.t0), count(*) FROM smp s LEFT JOIN r USING (cascade_id)""")
    check("A4 reconciliation with parsed lines (1,000 cascades: n_events and t0)", bad == 0, f"{bad} of {tot} differ")
    import numpy as np
    subs = json.load(open(f"{PARSED}/subs.json"))
    rows = con.execute("SELECT sub, submission_id36, t0 FROM smp USING SAMPLE reservoir(50 ROWS) REPEATABLE (3)").fetchall()
    def b36(n):
        a = "0123456789abcdefghijklmnopqrstuvwxyz"; o = ""
        while n: n, r = divmod(n, 36); o = a[r] + o
        return o or "0"
    miss = 0
    by_sub = {}
    for sb, th, t0 in rows: by_sub.setdefault(sb, {})[f"t3_{b36(th)}"] = t0
    for sb, want in by_sub.items():
        found = {}
        for line in open(f"{RAWD}/{subs[sb]}.jsonlist-meta", errors="surrogateescape"):
            if line[0] != " ":
                p_ = line.rstrip("\n").rsplit(" ", 2)
                if len(p_) == 3 and p_[2] in want: found[p_[2]] = int(float(p_[1]))
        miss += sum(found.get(k) != v for k, v in want.items())
    check("A4 submission id and time found in the raw Hessel file (50 cascades)", miss == 0, f"{miss} of {len(rows)} missing or different")

# A5 structure
bad = q1("SELECT count(*) FROM edges WHERE src = dst")[0]
check("A5 no self-loops", bad == 0, f"{bad:,}")
bad = q1("SELECT count(*) - count(DISTINCT (src, dst)) FROM edges")[0]
check("A5 (src, dst) unique", bad == 0, f"{bad:,} duplicates")
nn, ne = q1("SELECT count(*) FILTER (WHERE t_created IS NULL), count(*) FROM edges")
check("A5 null t_created is rare and flagged", nn / ne < 1e-3, f"{nn:,} of {ne:,}")

# A6 first_seen
bad = q1("""SELECT count(*) FROM (SELECT participant_id, min(t) ft FROM events GROUP BY 1) e JOIN participants p USING (participant_id)
            WHERE p.first_seen_t IS NULL OR p.first_seen_t > e.ft""")[0]
check("A6 first_seen_t ≤ first event of the participant", bad == 0, f"{bad:,} violations")

nf = sum(not ok for _, ok, _ in results)
L = [f"# Integrity checks: `{B}` ({KIND})", "", f"{len(results) - nf} of {len(results)} passed.", "", "| check | result | evidence |", "|---|---|---|"]
L += [f"| {n} | {'PASS' if ok else '**FAIL**'} | {e} |" for n, ok, e in results]
open(f"{B}/INTEGRITY.md", "w").write("\n".join(L) + "\n")
print(f"{len(results) - nf}/{len(results)} passed", flush=True)
