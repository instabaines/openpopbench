"""T2b 'which newcomer?' scores from saved T1 ranks (no model runs).

    python t2b_newcomer.py            writes $OPB_WORK/eval/t2/T2B_LEADERBOARD.md (needs t2_flags.py output)

Newcomer query: the participant has no event in any TRAIN cascade of the tier (unseen_train, t2_flags.py). The rank is the
T1 rank against the full candidate set, so a newcomer must beat every eligible account.
Estimator (stratified, so both T1 samples are used): newcomer queries are split into novel-source and habit (not novel).
The novel stratum uses every newcomer row of the novel-source sample (uniform over novel events); the habit stratum uses
the newcomer rows of the all-events sample that are not novel; they are combined with p = the novel share among newcomers
in the all-events sample: H = p * H_novel + (1 - p) * H_habit. 95% CI: cluster bootstrap over cascades (1,000 resamples)
of the seed-averaged outcome, p held fixed. Pairwise scorers (*_sub10 dirs) are evaluated on their 10% query subsample.
"""
import glob
import os
import re

import numpy as np
import pandas as pd

BB = os.environ.get("OPB_WORK", os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "work"))
sys_path = os.path.dirname(os.path.abspath(__file__))
import sys; sys.path.insert(0, sys_path)
from t2_flags import TIERS  # noqa: E402  (importing runs nothing heavy: guarded below)

FLOORS = ["popularity", "recency", "affinity", "exposure"]
METRICS = [("Hits@100", lambda r, n: r <= 100), ("Hits@1%", lambda r, n: r / n <= 0.01), ("MRR", lambda r, n: 1 / r)]
rng = np.random.default_rng(0)


def load(tier, FL):
    fl = pd.read_parquet(FL).sort_values("event_row").reset_index(drop=True)
    fg = pd.read_parquet(f"{BB}/eval/t2/flags_{tier}.parquet").sort_values("event_row").reset_index(drop=True)
    assert (fg.event_row.values == fl.event_row.values).all()
    fl["newcomer"] = fg.unseen_train.values
    M = {m: ("", [fl[f"rank_{m}"].to_numpy()]) for m in FLOORS}
    for d, tag in ((f"{BB}/baselines/{tier}", ""), (f"{BB}/baselines/{tier}_sub10", "sub10")):
        for f in sorted(glob.glob(f"{d}/*_seed*.parquet")):
            base = re.match(r"(.+)_seed\d+\.parquet", f.split("/")[-1]).group(1)
            if base.endswith("_t4") or base.startswith("t2a_"): continue   # T4 and T2a outputs share the file pattern (fix 2026-11-08)
            if tag and base in M and not M[base][0]: continue          # full-query version exists
            x = pd.read_parquet(f, columns=["event_row", f"rank_{base}"]).set_index("event_row")[f"rank_{base}"]
            r = x.reindex(fl.event_row.values).to_numpy()                 # NaN where a subsample method did not score
            if not tag: assert not np.isnan(r).any(), f
            M.setdefault(base, (tag, []))[1].append(r)
    return fl, M


def table(fl, M, split):
    s = fl.split.values == split; nc = fl.newcomer.values; nov = fl.novel.values
    hab = s & nc & ~nov & fl.in_all_sample.values; nvl = s & nc & nov & fl.in_novel_sample.values
    a = s & nc & fl.in_all_sample.values; p = nov[a].mean() if a.any() else 0.0
    rows = []
    for name, (tag, runs) in M.items():
        R = np.stack(runs); ok = ~np.isnan(R).any(0)
        h, v = hab & ok, nvl & ok
        cid_h, cid_v = pd.factorize(fl.cascade_id.values[h])[0], pd.factorize(fl.cascade_id.values[v])[0]
        Wh = rng.poisson(1.0, (1000, cid_h.max() + 1 if h.any() else 1)).astype(np.float32)[:, cid_h]
        Wv = rng.poisson(1.0, (1000, cid_v.max() + 1 if v.any() else 1)).astype(np.float32)[:, cid_v]
        cells = []
        for k, f in METRICS:
            per = lambda m: np.stack([f(r[m], fl.n_candidates.values[m]).astype(float) for r in R])   # seeds x queries
            ph, pv = per(h), per(v)
            seedv = p * pv.mean(1) + (1 - p) * ph.mean(1)
            bs = p * (Wv @ pv.mean(0)) / np.maximum(Wv.sum(1), 1) + (1 - p) * (Wh @ ph.mean(0)) / np.maximum(Wh.sum(1), 1)
            sd = f" ± {seedv.std(ddof=1):.3f}" if len(R) > 1 else ""
            cells.append(f"{seedv.mean():.3f}{sd} [{np.percentile(bs, 2.5):.3f}, {np.percentile(bs, 97.5):.3f}]")
        rows.append((name + (" §" if tag else ""), len(R), int(h.sum()), int(v.sum()), cells))
    return p, rows


if __name__ == "__main__":
    L = ["# T2b: which newcomer? (from saved T1 ranks)", "", __doc__.split("\n\n", 1)[1].strip(), "",
         "§ = pairwise scorer on the 10% query subsample (few newcomer queries; wide CIs).", ""]
    for tier, (B, FL) in TIERS.items():
        if not os.path.exists(f"{BB}/eval/t2/flags_{tier}.parquet"): continue
        fl, M = load(tier, FL)
        for split in ("test",):
            p, rows = table(fl, M, split)
            L += [f"## {tier}, {split} (newcomer share of all events {fl[(fl.split == split) & fl.in_all_sample].newcomer.mean():.3f}; novel share among newcomers p = {p:.2f})", "",
                  "| method | seeds | n habit | n novel | " + " | ".join(k for k, _ in METRICS) + " |", "|---|---|---|---|" + "---|" * len(METRICS)]
            rows.sort(key=lambda r: -float(r[4][0].split()[0]))
            L += [f"| {n} | {s} | {nh} | {nv} | " + " | ".join(c) + " |" for n, s, nh, nv, c in rows] + [""]
        print(tier, "done", flush=True)
    open(f"{BB}/eval/t2/T2B_LEADERBOARD.md", "w").write("\n".join(L) + "\n")
