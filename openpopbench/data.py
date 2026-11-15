"""Where a tier's files live, in the release layout or in the development tree.

Release layout (what is uploaded to Hugging Face; `opb export` writes it):

    <root>/<tier>/events.parquet        cascade_id, participant_id, t, position
    <root>/<tier>/cascades.parquet      cascade_id, origin, t0, split, ...
    <root>/<tier>/edges.parquet         src, dst, t_created
    <root>/<tier>/participants.parquet  participant_id, first_seen_t, ...
    <root>/<tier>/queries.parquet       the evaluated events: event_row, cascade_id, participant_id, t, split, position,
                                        in_all_sample, in_novel_sample, novel, n_candidates, rank_<heuristic> x 4
    <root>/<tier>/queries_sub10.parquet the fixed 10% query subsample (pairwise scorers)
    <root>/<tier>/newcomer.parquet      event_row, unseen_train (Task 2 label: no event in any training cascade)
    <root>/<tier>/task2a.parquet        event_row, cascade_id, split, y, p_const, p_roll, p_logit (Task 2a baselines)
    <root>/<tier>/task3.parquet         event_row, cascade_id, split, gap, nll_/pit_/med_ {exp, lognorm, lognorm_x}

`event_row` is the row of the event in events.parquet sorted by (t, cascade_id, position); every file uses it as the key.
Set OPB_DATA=<root> to the folder of a downloaded release.
"""
import os

VERSION = "v1.0"
TIERS = ["bluesky_S", "bluesky_Mid", "bluesky_M", "bluesky_L", "telegram_L", "reddit_S", "reddit_Mid", "reddit_M", "reddit_L"]
COMPARISON = ["bluesky_S", "bluesky_Mid", "reddit_S", "reddit_Mid"]
HEURISTICS = ["popularity", "recency", "affinity", "exposure"]
FILES = ["events", "cascades", "edges", "participants", "queries", "queries_sub10", "newcomer", "task2a", "task3"]

def path(tier, name):
    """Absolute path of one file of a tier (name in FILES)."""
    if tier not in TIERS: raise KeyError(f"unknown tier {tier}; tiers: {', '.join(TIERS)}")
    root = os.environ.get("OPB_DATA")
    if root: return os.path.join(root, tier, name + ".parquet")
    try:   # the maintainers' working tree (not part of the public repository)
        from ._devtree import path as dev_path
        return dev_path(tier, name)
    except ImportError:
        raise FileNotFoundError("no data: set OPB_DATA to the folder holding the downloaded tiers "
                                "(e.g. export OPB_DATA=~/openpopbench-v1.0; see README, Install)") from None


def tables_dir(tier):
    """The folder holding the four tables (what every adapter takes as its data argument)."""
    return os.path.dirname(path(tier, "events"))


def export(src, dst):
    """Write one tier in the release layout. src: {name: path} for the names in FILES; returns {name: rows}."""
    import shutil
    import pandas as pd, pyarrow.parquet as pq
    os.makedirs(dst, exist_ok=True); rows = {}
    for name in FILES:
        f = src.get(name)
        if not f or not os.path.exists(f): continue
        out = os.path.join(dst, name + ".parquet")
        if name not in ("queries", "queries_sub10", "newcomer"):   # released as built (byte copy)
            shutil.copyfile(f, out); rows[name] = pq.ParquetFile(out).metadata.num_rows; continue
        x = pd.read_parquet(f)
        if name in ("queries", "queries_sub10"): x = x.drop(columns=[c for c in ("rank_exposure_leak", "exposure_pos") if c in x.columns])
        if name == "newcomer": x = x[["event_row", "unseen_train"]]
        x.to_parquet(out, index=False); rows[name] = len(x)
    return rows
