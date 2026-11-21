"""Bluesky rebuild, last steps: write the rebuilt tiers in the release layout, and compare content hashes.

    python assemble.py OUT                      writes OUT/release/<tier>/ (layout of openpopbench/data.py)
    python assemble.py --check MINE.tsv REF.tsv compares two content-hash manifests (content_hash.py --manifest)
"""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
from openpopbench import data  # noqa: E402

if sys.argv[1] == "--check":
    rd = lambda f: {l.split("\t")[2].strip(): l.split("\t")[0] for l in open(f) if not l.startswith("content_sha256")}
    a, b = rd(sys.argv[2]), rd(sys.argv[3]); bad = [k for k in b if a.get(k) != b[k]]
    print(f"[bluesky] {len(b) - len(bad)} of {len(b)} release files match the v1.0 content hashes")
    for k in bad: print("  differs or missing:", k)
    sys.exit(1 if bad else 0)
OUT = sys.argv[1]
TB = {"bluesky_S": "tiers/tier_S", "bluesky_Mid": "tiers/tier_Mid", "bluesky_M": "tiers/tier_M", "bluesky_L": "v0.1"}
for t, tb in TB.items():
    src = {n: f"{OUT}/{tb}/{n}.parquet" for n in ("events", "cascades", "edges", "participants")}
    src |= {"queries": f"{OUT}/eval/floors_{t}.parquet", "queries_sub10": f"{OUT}/eval/sub10/floors_{t}.parquet",
            "newcomer": f"{OUT}/eval/t2/flags_{t}.parquet", "task2a": f"{OUT}/eval/t2/t2a_{t}.parquet", "task3": f"{OUT}/eval/t4/floors_{t}.parquet"}
    print(t, data.export(src, f"{OUT}/release/{t}"))
