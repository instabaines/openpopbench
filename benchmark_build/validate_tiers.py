"""Tier representativeness (paper, Appendix A).

    python validate_tiers.py BENCH_L TIER_ROOT EVAL_DIR LABEL      (EVAL_DIR holds floors_<label>_{L,S,M}.parquet)

E3 compares tier against L with the two-sample KS D-statistic (Leskovec & Faloutsos 2006), on up to 2M values per side.
Degree bias compares the L-graph degree of the sampled accounts against all active L accounts (Kurant et al. 2010).
"""
import json
import sys
from itertools import combinations
import duckdb
import numpy as np
import pandas as pd

BL, TR, EV, LABEL = sys.argv[1:5]
con = duckdb.connect(); con.execute("SET threads=48; SET memory_limit='300GB'")
rng = np.random.default_rng(0)
VALID = ["popularity", "recency", "affinity", "exposure"]


def sub(x, n=2_000_000):
    x = np.asarray(x, float); return x if len(x) <= n else rng.choice(x, n, replace=False)


def ks(a, b):
    a, b = np.sort(sub(a)), np.sort(sub(b)); g = np.concatenate([a, b])
    return float(np.abs(np.searchsorted(a, g, side="right") / len(a) - np.searchsorted(b, g, side="right") / len(b)).max())


def dists(D):
    r = {}
    r["cascade size"] = con.execute(f"SELECT n_events FROM read_parquet('{D}/cascades.parquet')").fetchnumpy()["n_events"]
    r["account activity (events per active account)"] = con.execute(
        f"""SELECT count(*) n FROM read_parquet('{D}/events.parquet') WHERE event_type <> 'origin' GROUP BY participant_id""").fetchnumpy()["n"]
    r["inter-event time (s)"] = con.execute(f"""SELECT t - lag(t) OVER (PARTITION BY cascade_id ORDER BY "position") d
        FROM read_parquet('{D}/events.parquet') QUALIFY d IS NOT NULL""").fetchnumpy()["d"]
    r["time since t0 (s)"] = con.execute(f"""SELECT e.t - c.t0 d FROM read_parquet('{D}/events.parquet') e JOIN read_parquet('{D}/cascades.parquet') c USING (cascade_id)
        WHERE e.event_type <> 'origin'""").fetchnumpy()["d"]
    return r


def unseen(D):
    return con.execute(f"""WITH e AS (SELECT e.participant_id, e."position", c.split FROM read_parquet('{D}/events.parquet') e JOIN read_parquet('{D}/cascades.parquet') c USING (cascade_id)),
        tr AS (SELECT DISTINCT participant_id FROM e WHERE split = 'train')
        SELECT avg((participant_id NOT IN (SELECT * FROM tr))::INT) FROM e WHERE "position" > 0 AND split = 'test'""").fetchone()[0]


def tau(a, b):
    p = list(combinations(range(len(a)), 2)); return sum(np.sign(a[i] - a[j]) * np.sign(b[i] - b[j]) for i, j in p) / len(p)


def h100(path, sl, split="test"):
    d = pd.read_parquet(path); d = d[(d.split == split) & d[f"in_{sl}_sample"]]
    return np.array([(d[f"rank_{m}"] <= 100).mean() for m in VALID]), len(d)


L = [f"# Tier representativeness: {LABEL}", "", "Pre-registered criteria are in paper, Appendix A.3.", ""]
dL = dists(BL); uL = unseen(BL)
degL = con.execute(f"""WITH a AS (SELECT DISTINCT participant_id FROM read_parquet('{BL}/events.parquet')),
        o AS (SELECT src p, count(*) k FROM read_parquet('{BL}/edges.parquet') GROUP BY 1), i AS (SELECT dst p, count(*) k FROM read_parquet('{BL}/edges.parquet') GROUP BY 1)
        SELECT a.participant_id, coalesce(o.k, 0) followers, coalesce(i.k, 0) followees FROM a LEFT JOIN o ON o.p = a.participant_id LEFT JOIN i ON i.p = a.participant_id""").fetchnumpy()
deg_idx = np.argsort(degL["participant_id"])
checks = []
summ = {}
for name in ("S", "M"):
    D = f"{TR}/tier_{name}"; info = json.load(open(f"{D}/tier.json"))
    L += [f"## Tier {name}", "", f"- Accounts {info['accounts']:,} ({info['accounts_with_events']:,} with events); cascades {info['cascades']:,} "
          f"(train {info['cascades_by_split']['train']:,}, val {info['cascades_by_split']['val']:,}, test {info['cascades_by_split']['test']:,}); events {info['events']:,}; edges {info['edges']:,}.",
          f"- **E1 density:** {info['events_per_account']:.2f} events per sampled account (train {info['train_events_per_account']:.2f}); the random-cascade sample had 1.1.",
          f"- **E2 truncation:** median share of a cascade's L events retained {info['retained_share_median']:.2f} (mean {info['retained_share_mean']:.2f}); "
          f"{info['L_cascades_with_origin_sampled']:.2%} of L cascades have their origin sampled. Forest Fire ignitions: {info['sampler']['ignitions']:,}.", ""]
    summ[f"density_{name}"] = info["events_per_account"]
    if name == "S": checks.append(("E1 S density ≥ 5.5 events per account", f"{info['events_per_account']:.2f}", info["events_per_account"] >= 5.5))
    dT = dists(D)
    L += ["| E3 distribution | KS D (tier vs L) | tier median | L median |", "|---|---|---|---|"]
    for k in dL:
        Dk = ks(dT[k], dL[k]); L.append(f"| {k} | {Dk:.3f} | {np.median(dT[k]):.0f} | {np.median(dL[k]):.0f} |")
        if k == "inter-event time (s)": checks.append((f"E3 {name} inter-event time KS D ≤ 0.2", f"{Dk:.3f}", Dk <= 0.2))
    lid = con.execute(f"SELECT l_participant_id FROM read_parquet('{D}/participants.parquet')").fetchnumpy()["l_participant_id"]
    pos = np.searchsorted(degL["participant_id"], lid, sorter=deg_idx); pos = deg_idx[np.minimum(pos, len(deg_idx) - 1)]
    hit = degL["participant_id"][pos] == lid
    for k in ("followers", "followees"):
        a, b = degL[k][pos[hit]], degL[k]
        L.append(f"| L-graph {k} of sampled vs all active accounts (hub bias) | {ks(a, b):.3f} | {np.median(a):.0f} | {np.median(b):.0f} |")
    uT = unseen(D)
    L += ["", f"- Unseen-in-train share of test events: tier {uT:.1%} vs L {uL:.1%}.", ""]
    try:
        hT, nT = h100(f"{EV}/floors_{LABEL}_{name}.parquet", "all"); hL, _ = h100(f"{EV}/floors_{LABEL}_L.parquet", "all")
        nvT, _ = h100(f"{EV}/floors_{LABEL}_{name}.parquet", "novel"); nvL, _ = h100(f"{EV}/floors_{LABEL}_L.parquet", "novel")
        L += ["| E4 test Hits@100 | " + " | ".join(VALID) + " |", "|---|" + "---|" * len(VALID),
              f"| tier {name}, all | " + " | ".join(f"{v:.3f}" for v in hT) + " |", "| L, all | " + " | ".join(f"{v:.3f}" for v in hL) + " |",
              f"| tier {name}, novel | " + " | ".join(f"{v:.3f}" for v in nvT) + " |", "| L, novel | " + " | ".join(f"{v:.3f}" for v in nvL) + " |", ""]
        for sl, a, b in (("all", hT, hL), ("novel", nvT, nvL)):
            t = tau(a, b); checks.append((f"E4 {name} floor ranking agrees with L (τ ≥ 0.8; {sl})", f"τ = {t:.2f}", t >= 0.8)); summ[f"test_tau_{name}_{sl}"] = float(t)
        for sl in ("all", "novel"):
            a, _ = h100(f"{EV}/floors_{LABEL}_{name}.parquet", sl, "val"); b, _ = h100(f"{EV}/floors_{LABEL}_L.parquet", sl, "val")
            summ[f"val_tau_{name}_{sl}"] = float(tau(a, b))
    except FileNotFoundError as e:
        L += [f"E4 pending: {e}", ""]
L += ["## Checks", "", "| check | value | result |", "|---|---|---|"] + [f"| {c} | {v} | {'PASS' if p else '**FAIL**'} |" for c, v, p in checks]
L += ["", "Degree and size statistics are reported with the direction of the bias; only E1, the inter-event-time part of E3, and E4 have pass criteria (plan §E)."]
json.dump(summ, open(f"{TR}/validation_tiers.json", "w"), indent=1)
open(f"{TR}/VALIDATION_tiers.md", "w").write("\n".join(L) + "\n"); print("\n".join(L))
