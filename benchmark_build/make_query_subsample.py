"""Query subsample (paper, Section 3.4): fixed query subsample for methods whose full-candidate scoring is pairwise.

    python make_query_subsample.py FLOORS.parquet OUT.parquet [K=2000] [SEED=0]

From each split (val, test) draw K rows uniformly at random from the all-events sample (in_all_sample) and K from the
novel-source sample (in_novel_sample) of FLOORS. The output keeps every FLOORS column for the union of drawn rows, with
in_all_sample / in_novel_sample redefined as membership in the drawn sets. Every candidate is still ranked for each
query: only the number of queries shrinks. Drawn once, before any F2 result exists, with a fixed seed.
"""
import sys

import numpy as np
import pandas as pd

FL, OUT = sys.argv[1:3]
K = int(sys.argv[3]) if len(sys.argv) > 3 else 2000
SEED = int(sys.argv[4]) if len(sys.argv) > 4 else 0
fl = pd.read_parquet(FL).sort_values("event_row").reset_index(drop=True)
rng = np.random.default_rng(SEED)
keep_all = np.zeros(len(fl), bool); keep_nov = np.zeros(len(fl), bool)
for split in ("val", "test"):
    for col, keep in (("in_all_sample", keep_all), ("in_novel_sample", keep_nov)):
        idx = np.flatnonzero((fl.split.values == split) & fl[col].values)
        keep[rng.choice(idx, min(K, len(idx)), replace=False)] = True
sub = fl[keep_all | keep_nov].copy()
sub["in_all_sample"] = keep_all[keep_all | keep_nov]; sub["in_novel_sample"] = keep_nov[keep_all | keep_nov]
sub.to_parquet(OUT, index=False)
print(f"{FL}: {len(fl):,} rows -> {len(sub):,} rows; "
      + ", ".join(f"{s}/{c}={int(((sub.split == s) & sub[c]).sum())}" for s in ("val", "test") for c in ("in_all_sample", "in_novel_sample"))
      + f"; candidate pairs {fl.n_candidates.sum():.2e} -> {sub.n_candidates.sum():.2e}")
