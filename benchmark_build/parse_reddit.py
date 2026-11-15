"""Parse the Hessel et al. (ICWSM 2016, v1.1) Reddit discussion trees into compact parquet, one file per subreddit.

    python parse_reddit.py OUT_DIR

Line format: (depth spaces) username utc_timestamp id; usernames may contain spaces, so the last two tokens are time and id.
Columns: sub (int32 index into subs.json), thread (int64, base-36 value of the submission's t3 id), cid (int64, base-36
value of this line's t3/t1 id), depth (int16), u (int64 keyed blake2b hash of the username; 0 for [deleted]), t (int64 s),
pu (parent's u; -1 for submissions), pcid (parent's cid; -1 for submissions). Usernames are never written; the hash key
is stored in OUT_DIR/private/hash_key.bin (not released).
"""
import glob
import hashlib
import json
import os
import sys
from multiprocessing import Pool
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

SRC = os.environ.get("HESSEL_SRC", "Hessel_ICWSM_2016/only_meta_and_stats")   # the extracted v1.1 archive
OUT = sys.argv[1]
os.makedirs(f"{OUT}/private", exist_ok=True)
KEYF = f"{OUT}/private/hash_key.bin"
if not os.path.exists(KEYF): open(KEYF, "wb").write(os.urandom(32))
KEY = open(KEYF, "rb").read()


def h(name, cache):
    if name == "[deleted]": return 0
    v = cache.get(name)
    if v is None:
        v = int.from_bytes(hashlib.blake2b(name.encode("utf-8", "surrogateescape"), digest_size=8, key=KEY).digest(), "little") >> 1
        cache[name] = v = v or 1
    return v


def flush(cols, k, part):
    n = len(cols["t"])
    if n == 0: return part, 0
    tb = pa.table(dict(sub=np.full(n, k, np.int32), thread=np.array(cols["thread"], np.int64), cid=np.array(cols["cid"], np.int64),
                       depth=np.array(cols["depth"], np.int16), u=np.array(cols["u"], np.int64), t=np.array(cols["t"], np.int64),
                       pu=np.array(cols["pu"], np.int64), pcid=np.array(cols["pcid"], np.int64)))
    pq.write_table(tb, f"{OUT}/lines/{k:05d}_{part:03d}.parquet", compression="zstd")
    for c in cols: cols[c].clear()
    return part + 1, n


def work(args):
    k, path = args
    cols = {c: [] for c in ("thread", "cid", "depth", "u", "t", "pu", "pcid")}
    stack, cache, thread, part, tot = [], {}, -1, 0, 0
    bad = dict(short=0, value=0, d0_not_submission=0, orphan=0)
    with open(path, "r", encoding="utf-8", errors="surrogateescape") as fh:
        for line in fh:
            s = line.rstrip("\n"); body = s.lstrip(" "); d = len(s) - len(body); parts = body.rsplit(" ", 2)
            if len(parts) < 3: bad["short"] += 1; continue
            name, ts, pid = parts
            try:
                t = int(float(ts)); cid = int(pid[3:], 36)
            except ValueError:
                bad["value"] += 1; continue
            if d == 0:
                if not pid.startswith("t3_"): bad["d0_not_submission"] += 1; continue
                thread = cid; stack = []
            elif thread < 0 or len(stack) < d:
                bad["orphan"] += 1; continue             # orphan line: no parent at depth d - 1
            u = h(name, cache)
            del stack[d:]
            pu, pc = (stack[d - 1] if d > 0 else (-1, -1))
            stack.append((u, cid))
            for c, v in zip(cols, (thread, cid, d, u, t, pu, pc)): cols[c].append(v)
            if len(cols["t"]) >= 5_000_000:
                part, n = flush(cols, k, part); tot += n
    part, n = flush(cols, k, part); tot += n
    return k, tot, bad


if __name__ == "__main__":
    os.makedirs(f"{OUT}/lines", exist_ok=True)
    files = sorted(glob.glob(f"{SRC}/*.jsonlist-meta"))
    subs = [os.path.basename(f).replace(".jsonlist-meta", "") for f in files]
    json.dump(subs, open(f"{OUT}/subs.json", "w"))
    order = sorted(range(len(files)), key=lambda i: -os.path.getsize(files[i]))
    tot = 0; bad = {}
    with Pool(40) as p:
        for n, (k, m, b) in enumerate(p.imap_unordered(work, [(i, files[i]) for i in order])):
            tot += m; bad = {r: bad.get(r, 0) + v for r, v in b.items()}
            if sum(b.values()) > 0.01 * max(m, 1): print(f"[parse] high skip share in {subs[k]}: {b} of {m:,}", flush=True)
            if n % 500 == 0: print(f"[parse] {n:,}/{len(files):,} files, {tot:,} lines, skipped {bad}", flush=True)
    json.dump(dict(files=len(files), lines=tot, skipped=bad), open(f"{OUT}/parse_stats.json", "w"))
    print(f"[parse] DONE {len(files):,} files, {tot:,} lines, skipped {bad}", flush=True)
