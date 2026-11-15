"""Forest Fire size tiers (paper, Section 3.3 and Appendix A.3).

    python ff_tiers.py BENCH_L OUT_ROOT [P_F=0.7] [SEED=0]       writes OUT_ROOT/tier_S and OUT_ROOT/tier_M (TIERS env overrides the names and sizes)
    P_F and SEED may be comma-separated lists; with more than one run, each goes to OUT_ROOT/pf<P_F>_s<SEED>/tier_{S,M}
    (the graph is loaded once).

Sampler (Leskovec & Faloutsos, KDD 2006): the network graph is treated as undirected. A fire starts at a uniformly random
active account (one that takes part in or originates a cascade). Each burning node burns x ~ Geometric unvisited
neighbours, with mean p_f / (1 - p_f), which then burn in turn. When the fire dies, it re-ignites from a fresh random seed. The fire
is snapshotted at 50k accounts (S) and continues to 500k (M), so S is nested in M.
Restriction: a cascade is kept when its origin is sampled; its events are restricted to sampled accounts; cascades left
with no non-origin event are dropped. Ids are re-densified; l_participant_id and l_cascade_id link back to L.
Tier membership uses the whole-period graph (as legacy datasets do); features inside a tier still obey the time rules.
"""
import json
import os
import sys
import time
from collections import deque
import duckdb
import numpy as np
import pyarrow as pa

BL, OUT = sys.argv[1], sys.argv[2]
PFS = [float(x) for x in (sys.argv[3] if len(sys.argv) > 3 else "0.7").split(",")]
SEEDS = [int(x) for x in (sys.argv[4] if len(sys.argv) > 4 else "0").split(",")]
os.makedirs(OUT, exist_ok=True)
TARGETS = [("S", 50_000), ("M", 500_000)]
if os.environ.get("TIERS"):   # e.g. TIERS=S:50000,Mid:150000 (the Mid tier; paper, Section 3.3); the fire is seeded per run, so S is unchanged
    TARGETS = [(k, int(v)) for k, v in (x.split(":") for x in os.environ["TIERS"].split(","))]
log = lambda s: print(f"[ff {time.strftime('%H:%M:%S')}] {s}", flush=True)
con = duckdb.connect(); con.execute(f"SET threads=48; SET memory_limit='300GB'; SET temp_directory='{OUT}/duck_tmp'")
for t in ("events", "cascades", "participants", "edges"):
    con.execute(f"CREATE VIEW {t} AS SELECT * FROM read_parquet('{BL}/{t}.parquet')")
N = con.execute("SELECT count(*) FROM participants").fetchone()[0]

log("undirected adjacency")
adj = con.execute("SELECT u, v FROM (SELECT src u, dst v FROM edges UNION ALL SELECT dst, src FROM edges) ORDER BY u, v").fetchnumpy()
rowptr = np.searchsorted(adj["u"], np.arange(N + 1)); nbr = adj["v"].astype(np.int32); del adj
active = np.unique(con.execute("SELECT DISTINCT participant_id FROM events").fetchnumpy()["participant_id"])
log(f"{N:,} participants, {len(active):,} active, {len(nbr):,} adjacency entries")

RUNS = [(pf, sd) for pf in PFS for sd in SEEDS]
for PF, SEED in RUNS:
    ROOT = OUT if len(RUNS) == 1 else f"{OUT}/pf{PF}_s{SEED}"
    log("forest fire")
    rng = np.random.default_rng(SEED)
    visited = np.zeros(N, bool); count = 0; queue = deque(); snaps = {}; reignite = 0
    while len(snaps) < len(TARGETS):
        if not queue:
            while True:
                s = active[rng.integers(len(active))]
                if not visited[s]: break
            visited[s] = True; count += 1; queue.append(s); reignite += 1
        v = queue.popleft()
        nb = nbr[rowptr[v]:rowptr[v + 1]]
        x = rng.geometric(1 - PF) - 1
        if x > 0 and len(nb):
            cand = nb[rng.integers(0, len(nb), size=min(len(nb), 4 * x + 8))]
            cand = np.unique(cand[~visited[cand]])
            if len(cand) < x and len(nb) <= 5000:
                cand = np.unique(nb[~visited[nb]])
            if len(cand) > x: cand = rng.choice(cand, x, replace=False)
            visited[cand] = True; count += len(cand); queue.extend(cand.tolist())
        for name, tgt in TARGETS:
            if name not in snaps and count >= tgt:
                snaps[name] = (np.flatnonzero(visited), reignite); log(f"{name}: {count:,} accounts, {reignite:,} ignitions")
    
    for name, _ in TARGETS:
        ids, ign = snaps[name]; D = f"{ROOT}/tier_{name}"; os.makedirs(D, exist_ok=True)
        log(f"restrict tier {name}")
        try: con.unregister("samp")
        except Exception: pass
        con.register("samp", pa.table(dict(l_participant_id=ids, participant_id=np.arange(len(ids)))))
        con.execute("CREATE OR REPLACE TEMP TABLE sm AS SELECT * FROM samp")
        con.execute("""CREATE OR REPLACE TEMP TABLE ev AS
            SELECT e.cascade_id l_cascade_id, s.participant_id, e.t, e.event_type, e."position" l_position
            FROM events e JOIN cascades c USING (cascade_id) JOIN sm s ON s.l_participant_id = e.participant_id
            WHERE c.origin_id IN (SELECT l_participant_id FROM sm)""")
        con.execute("""CREATE OR REPLACE TEMP TABLE cs AS
            SELECT row_number() OVER (ORDER BY c.t0, c.cascade_id) - 1 cascade_id, s.participant_id origin_id, c.t0, c.split, k.n n_events,
                   k.n / c.n_events retained_share, c.cascade_id l_cascade_id, c.n_events l_n_events
            FROM cascades c JOIN (SELECT l_cascade_id, count(*) - 1 n FROM ev GROUP BY 1 HAVING count(*) > 1) k ON k.l_cascade_id = c.cascade_id
            JOIN sm s ON s.l_participant_id = c.origin_id""")
        con.execute(f"""COPY (SELECT c.cascade_id, e.participant_id, e.t, e.event_type,
                             row_number() OVER (PARTITION BY c.cascade_id ORDER BY e.l_position) - 1 "position"
                             FROM ev e JOIN cs c USING (l_cascade_id) ORDER BY 1, 5) TO '{D}/events.parquet' (FORMAT PARQUET, COMPRESSION zstd)""")
        con.execute(f"COPY (SELECT * FROM cs ORDER BY cascade_id) TO '{D}/cascades.parquet' (FORMAT PARQUET, COMPRESSION zstd)")
        con.execute(f"""COPY (SELECT s.participant_id, p.* EXCLUDE (participant_id), s.l_participant_id FROM sm s JOIN participants p ON p.participant_id = s.l_participant_id
                        ORDER BY 1) TO '{D}/participants.parquet' (FORMAT PARQUET, COMPRESSION zstd)""")
        con.execute(f"""COPY (SELECT a.participant_id src, b.participant_id dst, g.t_created, g.edge_type FROM edges g
                        JOIN sm a ON a.l_participant_id = g.src JOIN sm b ON b.l_participant_id = g.dst ORDER BY g.t_created NULLS FIRST)
                        TO '{D}/edges.parquet' (FORMAT PARQUET, COMPRESSION zstd)""")
        st = con.execute("""SELECT count(*), sum(n_events), median(retained_share), avg(retained_share), sum(n_events) FILTER (WHERE split = 'train'),
                                   count(*) FILTER (WHERE split = 'train'), count(*) FILTER (WHERE split = 'val'), count(*) FILTER (WHERE split = 'test') FROM cs""").fetchone()
        (orig_share,), = con.execute("SELECT avg((origin_id IN (SELECT l_participant_id FROM sm))::INT) FROM cascades").fetchall()
        (ne,), = con.execute(f"SELECT count(*) FROM read_parquet('{D}/edges.parquet')").fetchall()
        (act,), = con.execute(f"SELECT count(DISTINCT participant_id) FROM read_parquet('{D}/events.parquet')").fetchall()
        meta = json.load(open(f"{BL}/splits.json"))
        info = dict(tier=name, sampler=dict(method="forest fire, undirected", p_f=PF, seed=SEED, ignitions=int(ign), source=BL),
                    accounts=int(len(ids)), accounts_with_events=int(act), cascades=int(st[0]), events=int(st[1]),
                    events_per_account=float(st[1] / len(ids)), train_events_per_account=float(st[4] / len(ids)),
                    retained_share_median=float(st[2]), retained_share_mean=float(st[3]), cascades_by_split=dict(train=st[5], val=st[6], test=st[7]),
                    L_cascades_with_origin_sampled=float(orig_share), edges=int(ne),
                    splits=dict(horizon_s=meta["horizon_s"], windows_t0=meta["windows_t0"]))
        json.dump(info, open(f"{D}/tier.json", "w"), indent=1); log(json.dumps(info))
log("DONE")
