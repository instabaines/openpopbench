"""Submission format and validation.

A submission is a folder:

    meta.json
    <tier>/task1_seed<s>.parquet     event_row, rank        (Task 1; Task 2b is derived from it)
    <tier>/task2a_seed<s>.parquet    event_row, p_new       (optional)
    <tier>/task3_seed<s>.parquet     event_row, nll, pit, median_gap   (optional)

Task 1 `rank` is the rank of the true participant among all n_candidates accounts of the query (1 = best), with ties at
their mid-rank (a tie of the target with k others at positions r..r+k gives r + k/2). Rows: every row of the tier's
queries.parquet (validation and test), or of queries_sub10.parquet if meta.json declares "queries": "sub10".
Task 2a `p_new`: probability that the participant of the event is a newcomer. Task 3: negative log-likelihood (nats,
gap in seconds) of the observed gap d = t_p - t_{p-1}, the predicted CDF at d (PIT), and the predicted median gap, for the
Task 3 queries (rows of task3.parquet).

meta.json fields:
    name            display name
    family          diffusion | temporal_graph | point_process | sequential_recommender | feature_ranker | other
    code            public URL of the code that produced the files
    queries         "full" (default) or "sub10"
    heuristic_inputs  list of heuristics whose signals the method uses as inputs (sanity gate; may be empty)
    training        free text: data used, budget (GPU-h), hyperparameter search, checkpoint selection rule
    track           "files" (rank files; leaderboard mark: self-reported) or "verified" (code re-run by the maintainers)
    flags           optional {tier: reason}, set by the maintainers (e.g. a failed reproduction gate); flagged results get †
"""
import glob, json, os, re
import numpy as np, pandas as pd

from . import data

FAMILIES = {"diffusion", "temporal_graph", "point_process", "sequential_recommender", "feature_ranker", "other"}
REQUIRED_META = ["name", "family", "code", "training", "track"]


class Invalid(Exception):
    pass


def read_meta(sub):
    p = os.path.join(sub, "meta.json")
    if not os.path.exists(p): raise Invalid("meta.json missing")
    m = json.load(open(p))
    miss = [k for k in REQUIRED_META if k not in m]
    if miss: raise Invalid(f"meta.json lacks {miss}")
    if m["family"] not in FAMILIES: raise Invalid(f"family must be one of {sorted(FAMILIES)}")
    if m["track"] not in ("files", "verified"): raise Invalid("track must be 'files' or 'verified'")
    m.setdefault("queries", "full"); m.setdefault("heuristic_inputs", [])
    bad = [h for h in m["heuristic_inputs"] if h not in data.HEURISTICS]
    if bad: raise Invalid(f"unknown heuristic_inputs {bad}")
    return m


def files(sub, tier, task):
    out = {}
    for f in glob.glob(os.path.join(sub, tier, f"{task}_seed*.parquet")):
        out[int(re.search(r"_seed(\d+)\.parquet$", f).group(1))] = f
    return dict(sorted(out.items()))


def load_task1(sub, tier, meta):
    """Queries frame and {seed: rank array aligned to it}. Raises Invalid on any format problem."""
    q = pd.read_parquet(data.path(tier, "queries_sub10" if meta["queries"] == "sub10" else "queries")).sort_values("event_row").reset_index(drop=True)
    runs = {}
    for s, f in files(sub, tier, "task1").items():
        x = pd.read_parquet(f)
        if not {"event_row", "rank"} <= set(x.columns): raise Invalid(f"{f}: needs columns event_row, rank")
        if x.event_row.duplicated().any(): raise Invalid(f"{f}: duplicate event_row")
        x = x.set_index("event_row")
        missing = np.setdiff1d(q.event_row.values, x.index.values)
        if len(missing): raise Invalid(f"{f}: {len(missing):,} query rows missing (e.g. {missing[:3].tolist()})")
        r = x["rank"].reindex(q.event_row.values).to_numpy(dtype=float)
        n = q.n_candidates.to_numpy()
        if np.isnan(r).any() or (r < 1).any() or (r > n).any(): raise Invalid(f"{f}: ranks must lie in [1, n_candidates]")
        if ((r * 2) % 1 != 0).any(): raise Invalid(f"{f}: ranks must be whole or half numbers (mid-rank ties)")
        runs[s] = r
    return q, runs


def load_task2a(sub, tier):
    t = pd.read_parquet(data.path(tier, "task2a")).sort_values("event_row").reset_index(drop=True); runs = {}
    for s, f in files(sub, tier, "task2a").items():
        x = pd.read_parquet(f).set_index("event_row")["p_new"].reindex(t.event_row.values).to_numpy(dtype=float)
        if np.isnan(x).any() or (x < 0).any() or (x > 1).any(): raise Invalid(f"{f}: p_new missing or outside [0, 1]")
        runs[s] = x
    return t, runs


def load_task3(sub, tier):
    t = pd.read_parquet(data.path(tier, "task3")).sort_values("event_row").reset_index(drop=True); runs = {}
    for s, f in files(sub, tier, "task3").items():
        x = pd.read_parquet(f).set_index("event_row").reindex(t.event_row.values)
        if not {"nll", "pit", "median_gap"} <= set(x.columns): raise Invalid(f"{f}: needs nll, pit, median_gap")
        if x[["nll", "pit", "median_gap"]].isna().any().any(): raise Invalid(f"{f}: rows missing")
        runs[s] = (x.nll.to_numpy(float), x.pit.to_numpy(float), x.median_gap.to_numpy(float))
    return t, runs


def validate(sub):
    """Check a submission folder; returns (meta, {tier: [tasks present]})."""
    meta = read_meta(sub); found = {}
    for tier in data.TIERS:
        tasks = [t for t in ("task1", "task2a", "task3") if files(sub, tier, t)]
        if not tasks: continue
        if "task1" in tasks: load_task1(sub, tier, meta)
        if "task2a" in tasks: load_task2a(sub, tier)
        if "task3" in tasks: load_task3(sub, tier)
        found[tier] = tasks
    if not found: raise Invalid("no result files found (expected <tier>/task1_seed<s>.parquet)")
    return meta, found
