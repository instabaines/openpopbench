"""T1 sanity floors on any benchmark directory (released tables only), GPU. Writes one row per sampled query with the
rank of the target under every floor, so the summariser can compute metrics, bootstrap CIs and slices.

    python floors_eval.py BENCH_DIR OUT.parquet [NQ=20000] [DEVICE=cuda:4]
    QUERIES=Q.parquet python floors_eval.py ...   scores the event rows of Q instead of sampling (leakage test)

Queries: for val and test separately, NQ events sampled from all eligible events and NQ from the novel-source slice.
Eligible = non-origin event whose participant is observable before t (first_seen_t < t) and can participate
(is_crawled where the table has it). Novel source = the participant had no prior tie to the origin: no earlier event in a
cascade of the same origin within the stream, and no edge origin -> participant created before t.
Candidates = eligible participants observable before t, minus the cascade's earlier participants (origin included).

Floors (popularity breaks ties for every floor except popularity itself):
  popularity        train-split participations
  recency           latest participation before t
  affinity          earlier participations in cascades of the same origin (EdgeBank-style memory)
  exposure          earlier participants of this cascade (origin included) with an edge to the candidate created before t
  exposure_leak     the same, ignoring edge creation times (oracle-leak demonstration; not a valid heuristic)
"""
import os
import sys
import time
import duckdb
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch

B, OUT = sys.argv[1], sys.argv[2]
NQ = int(sys.argv[3]) if len(sys.argv) > 3 else 20000
dev = torch.device(sys.argv[4] if len(sys.argv) > 4 else "cuda:4")
log = lambda s: print(f"[floors {time.strftime('%H:%M:%S')}] {s}", flush=True)
con = duckdb.connect(); con.execute("SET threads=48; SET memory_limit='300GB'")
P = f"read_parquet('{B}/participants.parquet')"
cols = [r[0] for r in con.execute(f"DESCRIBE SELECT * FROM {P}").fetchall()]

log("load")
ev = con.execute(f"""SELECT e.cascade_id, e.participant_id, e.t, e."position", c.origin_id,
                            (CASE c.split WHEN 'train' THEN 0 WHEN 'val' THEN 1 ELSE 2 END)::TINYINT split
                     FROM read_parquet('{B}/events.parquet') e JOIN read_parquet('{B}/cascades.parquet') c USING (cascade_id)
                     ORDER BY e.t, e.cascade_id, e."position" """).fetchnumpy()
pa_ = con.execute(f"SELECT participant_id, coalesce(first_seen_t, 9e18)::BIGINT f{', is_crawled' if 'is_crawled' in cols else ''} FROM {P} ORDER BY participant_id").fetchnumpy()
N = len(pa_["f"]); assert pa_["participant_id"][-1] == N - 1, "participant ids must be dense"
first = pa_["f"]; elig = pa_["is_crawled"].astype(bool) if "is_crawled" in pa_ else np.ones(N, bool)
eg = con.execute(f"SELECT src, dst, coalesce(t_created, 9e18)::BIGINT t FROM read_parquet('{B}/edges.parquet') ORDER BY src, dst").fetchnumpy()
E = len(eg["src"]); log(f"{len(ev['t']):,} events, {N:,} participants, {E:,} edges")
non0 = ev["position"] > 0

log("novel-source flags")
key = ev["origin_id"].astype(np.int64) * N + ev["participant_id"]
idx = np.flatnonzero(non0); o = idx[np.lexsort((ev["t"][idx], key[idx]))]
firstk = np.ones(len(o), bool); firstk[1:] = key[o][1:] != key[o][:-1]
no_prior_event = np.zeros(len(ev["t"]), bool); no_prior_event[o[firstk]] = True
ekey = eg["src"].astype(np.int64) * N + eg["dst"]
j = np.searchsorted(ekey, key); j = np.minimum(j, E - 1)
edge_before = (ekey[j] == key) & (eg["t"][j] < ev["t"])
novel = non0 & no_prior_event & ~edge_before
del ekey, j, key

log("sample queries")
rng = np.random.default_rng(0)
ok = non0 & elig[ev["participant_id"]] & (first[ev["participant_id"]] < ev["t"])
q_all, q_nov = np.zeros(len(ok), bool), np.zeros(len(ok), bool)
pool = {}
SPLIT = dict(train=0, val=1, test=2)
for s in ("val", "test"):
    for name, m, arr in (("all", ok & (ev["split"] == SPLIT[s]), q_all), ("novel", ok & novel & (ev["split"] == SPLIT[s]), q_nov)):
        c = np.flatnonzero(m); pool[f"{s}_{name}"] = len(c)
        arr[rng.choice(c, min(NQ, len(c)), replace=False)] = True
if os.environ.get("QUERIES"):   # leakage test: score a fixed query list instead of sampling (2026-11-08)
    fq = pq.read_table(os.environ["QUERIES"], columns=["event_row", "in_all_sample", "in_novel_sample"]).to_pandas()
    q_all[:] = q_nov[:] = False; q_all[fq.event_row.values] = fq.in_all_sample.values; q_nov[fq.event_row.values] = fq.in_novel_sample.values
qi = np.flatnonzero(q_all | q_nov)   # already in time order
log(f"{len(qi):,} queries; eligible pools {pool}")

log("GPU structures")
order = np.lexsort((ev["position"], ev["cascade_id"])); cstart = np.searchsorted(ev["cascade_id"][order], np.arange(ev["cascade_id"].max() + 2))
ai = np.flatnonzero(non0); ai = ai[np.lexsort((ev["t"][ai], ev["origin_id"][ai]))]
a_t, a_p = ev["t"][ai], ev["participant_id"][ai]; astart = np.searchsorted(ev["origin_id"][ai], np.arange(N + 1))
rowptr = torch.from_numpy(np.searchsorted(eg["src"], np.arange(N + 1)).astype(np.int64)).to(dev)
g_dst = torch.from_numpy(eg["dst"].astype(np.int32)).to(dev); g_t = torch.from_numpy(eg["t"]).to(dev); del eg
pop = torch.from_numpy(np.bincount(ev["participant_id"][(ev["split"] == 0) & non0], minlength=N).astype(np.float64)).to(dev)
tieb = pop / (pop.max() + 1)
first_g, elig_g = torch.from_numpy(first).to(dev), torch.from_numpy(elig).to(dev)
ev_p = torch.from_numpy(ev["participant_id"].astype(np.int64)).to(dev); ev_t = torch.from_numpy(ev["t"].astype(np.float64)).to(dev)
last = torch.full((N,), -1.0, device=dev, dtype=torch.float64); ptr = 0


def exposure(prefix, t):
    s, e = rowptr[prefix], rowptr[prefix + 1]; ln = e - s; tot = int(ln.sum())
    x, xl = torch.zeros(N, device=dev, dtype=torch.float64), torch.zeros(N, device=dev, dtype=torch.float64)
    if tot:
        base = torch.repeat_interleave(s - torch.cumsum(ln, 0) + ln, ln) + torch.arange(tot, device=dev)
        d = g_dst[base].long(); keep = g_t[base] < t
        xl.index_add_(0, d, torch.ones(tot, device=dev, dtype=torch.float64))
        x.index_add_(0, d[keep], torch.ones(int(keep.sum()), device=dev, dtype=torch.float64))
    return x, xl


NAMES = ("popularity", "recency", "affinity", "exposure", "exposure_leak")
R = {m: np.zeros(len(qi)) for m in NAMES}; ncand = np.zeros(len(qi), np.int64); exp_pos = np.zeros(len(qi), bool)
t0 = time.time()
for n, jx in enumerate(qi):
    t, c, y, pos, o = int(ev["t"][jx]), ev["cascade_id"][jx], int(ev["participant_id"][jx]), ev["position"][jx], ev["origin_id"][jx]
    nxt = int(np.searchsorted(ev["t"], t, side="left"))
    if nxt > ptr: last.scatter_reduce_(0, ev_p[ptr:nxt], ev_t[ptr:nxt], "amax"); ptr = nxt
    ix = order[cstart[c]:cstart[c + 1]]; prev = torch.from_numpy(ev["participant_id"][ix[ev["position"][ix] < pos]].astype(np.int64)).to(dev)
    cand = elig_g & (first_g < t); cand[prev] = False; cand[y] = True
    a, b = astart[o], astart[o + 1]; b = a + np.searchsorted(a_t[a:b], t, side="left")
    aff = tieb.clone(); aff.index_add_(0, torch.from_numpy(a_p[a:b].astype(np.int64)).to(dev), torch.ones(b - a, device=dev, dtype=torch.float64))
    ex, exl = exposure(prev, t); exp_pos[n] = bool(ex[y] > 0)
    ncand[n] = int(cand.sum())
    for m, sc in zip(NAMES, (pop, last + tieb * 1e-3, aff, ex + tieb, exl + tieb)):
        s_ = sc[cand]; v = sc[y]
        R[m][n] = 1 + float((s_ > v).sum()) + 0.5 * (float((s_ == v).sum()) - 1)
    if n % 5000 == 0: log(f"{n:,}/{len(qi):,} queries ({time.time() - t0:.0f} s)")

tb = pa.table(dict(event_row=qi, cascade_id=ev["cascade_id"][qi], participant_id=ev["participant_id"][qi], t=ev["t"][qi],
                   split=np.array(["train", "val", "test"])[ev["split"][qi]], position=ev["position"][qi], in_all_sample=q_all[qi], in_novel_sample=q_nov[qi], novel=novel[qi],
                   exposure_pos=exp_pos, n_candidates=ncand, **{f"rank_{m}": R[m] for m in NAMES}))
pq.write_table(tb, OUT, compression="zstd")
pq.write_table(pa.table(dict(pool=list(pool), n=list(pool.values()), novel_share_test=[float(novel[ok & (ev["split"] == 2)].mean())] * len(pool))),
               OUT.replace(".parquet", "_pools.parquet"))
log(f"DONE {OUT}")
