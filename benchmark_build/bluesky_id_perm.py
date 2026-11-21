"""Bluesky participant-id permutation (release file, 2026-11-20; the Reddit procedure of reddit_id_perm.py).

build_bluesky.py numbers participants by DuckDB's hash() of the export's account id (did_id), which is not guaranteed
stable across DuckDB versions. This script defines a canonical order that anyone with the source export can compute,
the ascending did_id (the export's own random integer per account), and writes the permutation from that order to our
participant ids:
  release file: bluesky_id_perm.parquet, columns canonical_rank, participant_id (int32), one row per participant.
The file is a permutation of 0..N-1. build_bluesky.py with ID_PERM=<file> uses it instead of the hash order.

    python bluesky_id_perm.py ID_MAP.parquet OUT.parquet      (we run this once, on our own build's private id map)
"""
import sys
import duckdb

IDMAP, OUT = sys.argv[1:3]
con = duckdb.connect(); con.execute("SET threads=32")
con.execute(f"""COPY (SELECT (row_number() OVER (ORDER BY did_id) - 1)::INTEGER canonical_rank, participant_id::INTEGER participant_id
                FROM read_parquet('{IDMAP}') ORDER BY canonical_rank) TO '{OUT}' (FORMAT parquet)""")
n = con.execute(f"SELECT count(*) FROM read_parquet('{IDMAP}')").fetchone()[0]
chk = con.execute(f"SELECT count(*), count(DISTINCT participant_id), min(participant_id), max(participant_id) FROM read_parquet('{OUT}')").fetchone()
assert chk == (n, n, 0, n - 1), chk
print(f"[perm] {n:,} participants; permutation written to {OUT}")
