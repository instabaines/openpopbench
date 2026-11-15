"""T1 leaderboard: floors plus trained methods on the same queries (paper, Section 4).

    python leaderboard.py LABEL FLOORS.parquet OUT.md METHOD_DIR [METHOD_DIR ...]

Each METHOD_DIR holds <method>_seed<s>.parquet with a rank_<method> column (same event_row set as FLOORS). A cell reports
the mean over seeds, ± sd over seeds when there are several, and a 95% cluster-bootstrap CI (cascades, 1,000
resamples) of the seed-averaged per-query outcome.
SUBSET=1: FLOORS is a query subsample (make_query_subsample.py; paper, Section 3.4); method files that cover more queries are
restricted to its rows, so every method is compared on the same queries. Without SUBSET the rows must match exactly.
"""
import os
import glob
import re
import sys
import numpy as np
import pandas as pd

LABEL, FL, OUT = sys.argv[1:4]
fl = pd.read_parquet(FL).sort_values("event_row").reset_index(drop=True)
FLOORS = ["popularity", "recency", "affinity", "exposure"]
methods = {m: [fl[f"rank_{m}"].to_numpy()] for m in FLOORS}
for arg in sys.argv[4:]:
    tag, d = (arg.split(":", 1) if ":" in arg else ("", arg))     # "tag:dir" keeps configurations apart (e.g. bs128:dir)
    for f in sorted(g for g in glob.glob(f"{d}/*_seed*.parquet") if not re.search(r"(^|/)t\d+[ab]?_|_t\d+_seed", g)):   # T2/T4 task outputs share the pattern (2026-10-30)
        base = re.match(r"(.+)_seed\d+\.parquet", f.split("/")[-1]).group(1); name = base + (f"[{tag}]" if tag else "")
        x = pd.read_parquet(f).sort_values("event_row").reset_index(drop=True)
        if os.environ.get("SUBSET"):
            x = x[x.event_row.isin(fl.event_row.values)].reset_index(drop=True)
        assert (x.event_row.values == fl.event_row.values).all(), f"{f}: query rows differ from floors"
        methods.setdefault(name, []).append(x[f"rank_{base}"].to_numpy())
METRICS = [("Hits@10", lambda r, n: r <= 10), ("Hits@100", lambda r, n: r <= 100), ("MRR", lambda r, n: 1 / r),
           ("Hits@0.1%", lambda r, n: r / n <= 0.001), ("Hits@1%", lambda r, n: r / n <= 0.01)]
rng = np.random.default_rng(0)
L = [f"# T1 leaderboard: {LABEL}", "", "Each cell: mean over seeds (± sd if several seeds) and a 95% cluster-bootstrap CI. Floors are deterministic.", ""]
for split in ("val", "test"):
    for sl in ("all", "novel"):
        m_ = (fl.split.values == split) & fl[f"in_{sl}_sample"].values
        if not m_.any(): continue
        cid = pd.factorize(fl.cascade_id.values[m_])[0]; W = rng.poisson(1.0, (1000, cid.max() + 1)).astype(np.float32)[:, cid]
        n = fl.n_candidates.values[m_]
        L += [f"## {split}, {sl} events (n = {m_.sum():,})", "", "| method | seeds | " + " | ".join(k for k, _ in METRICS) + " |", "|---|---|" + "---|" * len(METRICS)]
        for name, runs in methods.items():
            cells = []
            for k, f in METRICS:
                per = np.stack([f(r[m_], n).astype(float) for r in runs])        # seeds x queries
                vals = per.mean(1); avg = per.mean(0); bs = (W @ avg) / W.sum(1)
                sd = f" ± {vals.std(ddof=1):.3f}" if len(runs) > 1 else ""
                cells.append(f"{vals.mean():.3f}{sd} [{np.percentile(bs, 2.5):.3f}, {np.percentile(bs, 97.5):.3f}]")
            L.append(f"| {name} | {len(runs)} | " + " | ".join(cells) + " |")
        L.append("")
open(OUT, "w").write("\n".join(L) + "\n"); print("\n".join(L))
