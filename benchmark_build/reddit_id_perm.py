"""Reddit participant-id permutation (release file; paper, Section 6.2).

build_reddit.py numbers participants by a keyed hash of the username. The key is secret (pseudonymization), so a
rebuild by anyone else numbers accounts differently, and the tier sampler, query sample and every Reddit number change.
This script defines a KEY-FREE canonical order of participants that anyone can compute from the public source, and
writes the permutation from that order to our participant ids:

  canonical key of an account = its first line in the source before the stream end (t, is_comment, base-36 id value),
  i.e. arg_min over (t, depth > 0, cid). Two accounts cannot share a key (each line has one author).
  canonical rank = row_number() over that key, from 0.
  release file: reddit_id_perm.parquet, columns canonical_rank, participant_id (int32), one row per participant.

The file is a permutation of 0..N-1: it contains no usernames, ids or times. build_reddit.py with ID_PERM=<file> uses
it instead of the keyed-hash order, and reproduces our participant ids from a parse made with any key.

    python reddit_id_perm.py PARSED_DIR ID_MAP.parquet OUT.parquet      (we run this once, on our own parse)
"""
import sys
import duckdb

PARSED, IDMAP, OUT = sys.argv[1:4]
T_END = int(duckdb.sql("SELECT epoch(TIMESTAMPTZ '2013-01-14 00:00:00+00')").fetchone()[0])   # = build_reddit.T_END
con = duckdb.connect(); con.execute("SET threads=48; SET memory_limit='300GB'")
KEYS = f"SELECT u, min(struct_pack(t := t, is_comment := depth > 0, id := cid)) k FROM read_parquet('{PARSED}/lines/*.parquet') WHERE u <> 0 AND t < {T_END} GROUP BY u"
con.execute(f"CREATE TABLE canon AS SELECT u, k.t kt, k.is_comment kc, k.id kid, row_number() OVER (ORDER BY k) - 1 canonical_rank FROM ({KEYS})")
n_c, n_k = con.execute("SELECT count(*), count(DISTINCT (kt, kc, kid)) FROM canon").fetchone()   # keys must be unique
n_i = con.execute(f"SELECT count(*) FROM read_parquet('{IDMAP}')").fetchone()[0]
assert n_c == n_k == n_i, (n_c, n_k, n_i)
con.execute(f"""COPY (SELECT c.canonical_rank::INTEGER canonical_rank, i.participant_id::INTEGER participant_id FROM canon c JOIN read_parquet('{IDMAP}') i USING (u) ORDER BY c.canonical_rank)
                TO '{OUT}' (FORMAT parquet)""")
chk = con.execute(f"SELECT count(*), count(DISTINCT participant_id), min(participant_id), max(participant_id) FROM read_parquet('{OUT}')").fetchone()
assert chk == (n_c, n_c, 0, n_c - 1), chk
print(f"[perm] {n_c:,} participants; permutation written to {OUT}")
