"""Forest Fire grid validation and selection, the pre-registered tier-selection rule (paper, Appendix A.3) (validation split only).

    python validate_grid.py BENCH_L GRID_ROOT EVAL_GRID_DIR OUT.md

GRID_ROOT/pf<p>_s<seed>/tier_{S,M}; EVAL_GRID_DIR/floors_pf<p>_s<seed>_{S,M}.parquet and EVAL_GRID_DIR/../floors_bluesky_L.parquet.
Test-split statistics are NOT computed here; the selected M setting is tested once, separately.
"""
import glob
import json
import os
import re
import sys
from itertools import combinations
import duckdb
import numpy as np
import pandas as pd

BL, GR, EVG, OUT = sys.argv[1:5]
con = duckdb.connect(); con.execute("SET threads=48; SET memory_limit='300GB'")
rng = np.random.default_rng(0)
VALID = ["popularity", "recency", "affinity", "exposure"]
DISTS = ["cascade size", "account activity", "inter-event time", "time since t0", "L-followers", "L-followees"]
sub = lambda x, n=2_000_000: (np.asarray(x, float) if len(x) <= n else rng.choice(np.asarray(x, float), n, replace=False))


def ks(a, b):
    a, b = np.sort(sub(a)), np.sort(sub(b)); g = np.concatenate([a, b])
    return float(np.abs(np.searchsorted(a, g, side="right") / len(a) - np.searchsorted(b, g, side="right") / len(b)).max())


def dists(D):
    q = lambda s: con.execute(s).fetchnumpy()["x"]
    return {"cascade size": q(f"SELECT n_events x FROM read_parquet('{D}/cascades.parquet')"),
            "account activity": q(f"SELECT count(*) x FROM read_parquet('{D}/events.parquet') WHERE event_type <> 'origin' GROUP BY participant_id"),
            "inter-event time": q(f"""SELECT t - lag(t) OVER (PARTITION BY cascade_id ORDER BY "position") x FROM read_parquet('{D}/events.parquet') QUALIFY x IS NOT NULL"""),
            "time since t0": q(f"""SELECT e.t - c.t0 x FROM read_parquet('{D}/events.parquet') e JOIN read_parquet('{D}/cascades.parquet') c USING (cascade_id) WHERE e.event_type <> 'origin'""")}


tau = lambda a, b: float(np.mean([np.sign(a[i] - a[j]) * np.sign(b[i] - b[j]) for i, j in combinations(range(len(a)), 2)]))
h100 = lambda d, sl: np.array([(d[(d.split == "val") & d[f"in_{sl}_sample"]][f"rank_{m}"] <= 100).mean() for m in VALID])

print("[grid] L statistics", flush=True)
dL = dists(BL)
deg = con.execute(f"""WITH a AS (SELECT DISTINCT participant_id p FROM read_parquet('{BL}/events.parquet')),
      o AS (SELECT src p, count(*) k FROM read_parquet('{BL}/edges.parquet') GROUP BY 1), i AS (SELECT dst p, count(*) k FROM read_parquet('{BL}/edges.parquet') GROUP BY 1)
      SELECT a.p, coalesce(o.k, 0) fo, coalesce(i.k, 0) fe FROM a LEFT JOIN o USING (p) LEFT JOIN i USING (p) ORDER BY a.p""").fetchnumpy()
dL["L-followers"], dL["L-followees"] = deg["fo"], deg["fe"]
Lf = pd.read_parquet(f"{EVG}/../floors_bluesky_L.parquet"); hL = {sl: h100(Lf, sl) for sl in ("all", "novel")}

rows = []
for run in sorted(glob.glob(f"{GR}/pf*_s*")):
    pf, sd = re.search(r"pf([\d.]+)_s(\d+)", run).groups()
    for t in ("S", "M"):
        D = f"{run}/tier_{t}"; info = json.load(open(f"{D}/tier.json")); dT = dists(D)
        lid = con.execute(f"SELECT l_participant_id FROM read_parquet('{D}/participants.parquet')").fetchnumpy()["l_participant_id"]
        pos = np.searchsorted(deg["p"], lid); ok = (pos < len(deg["p"])) & (deg["p"][np.minimum(pos, len(deg["p"]) - 1)] == lid)
        dT["L-followers"], dT["L-followees"] = deg["fo"][pos[ok]], deg["fe"][pos[ok]]
        r = dict(p_f=float(pf), seed=int(sd), tier=t, density=info["events_per_account"], ignitions=info["sampler"]["ignitions"],
                 retained_median=info["retained_share_median"], **{f"D {k}": ks(dT[k], dL[k]) for k in DISTS})
        r["D_max"] = max(r[f"D {k}"] for k in DISTS)
        fl = f"{EVG}/floors_pf{pf}_s{sd}_{t}.parquet"
        if os.path.exists(fl):
            f = pd.read_parquet(fl); r.update({f"val_tau_{sl}": tau(h100(f, sl), hL[sl]) for sl in ("all", "novel")})
        rows.append(r); print(f"[grid] {run} {t}: D_max {r['D_max']:.3f} density {r['density']:.1f}", flush=True)
g = pd.DataFrame(rows); g.to_csv(OUT.replace(".md", ".csv"), index=False)

sel = g[g.seed > 0]
S = sel[sel.tier == "S"].groupby("p_f").agg(D_max=("D_max", "mean"), spread=("D_max", "std"), min_density=("density", "min"))
S_ok = S[S.min_density >= 5.5].sort_values(["D_max"])
best_S = None
if len(S_ok):
    top = S_ok[S_ok.D_max <= S_ok.D_max.min() + 0.01]; best_S = float(top.sort_values("spread").index[0])
M = sel[sel.tier == "M"].assign(val_tau=lambda d: (d.val_tau_all + d.val_tau_novel) / 2).groupby("p_f").agg(val_tau=("val_tau", "mean"), D_max=("D_max", "mean"))
topM = M[M.val_tau >= M.val_tau.max() - 0.05]; best_M = float(topM.sort_values("D_max").index[0])
L = ["# Forest Fire grid: pre-registered selection (validation only)", "",
     "Seeds 1–3 are used for selection; seed 0 is shown for reference only. Test statistics were not computed.", "",
     "## S (scale tier): lowest mean D_max with density ≥ 5.5 on every seed", "", S.round(3).to_markdown(), "", f"**Selected S: p_f = {best_S}**", "",
     "## M (ranking tier): highest mean validation τ (all, novel); ties within 0.05 go to the lower D_max", "", M.round(3).to_markdown(), "", f"**Selected M: p_f = {best_M}**", "",
     "## All runs", "", g.round(3).to_markdown(index=False)]
json.dump(dict(selected_S=best_S, selected_M=best_M), open(OUT.replace(".md", ".json"), "w"))
open(OUT, "w").write("\n".join(L) + "\n"); print("\n".join(L[:14]), flush=True)
