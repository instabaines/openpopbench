"""Scoring of a validated submission. Same estimators as the paper's leaderboards:

Task 1  Hits@10, Hits@100, MRR, Hits@0.1%, Hits@1% per seed (mean ± sd over seeds); 95% CI from 1,000 Poisson-bootstrap
        resamples of cascades of the seed-averaged per-query outcome. Slices: all events (in_all_sample), no-tie events
        (in_novel_sample). Sanity gate: validation Hits@100 (all events) at least that of popularity and of every heuristic
        named in meta.heuristic_inputs; results that fail are flagged.
Task 2b From the Task 1 ranks of newcomer events, stratified: H = p * H_no-tie + (1 - p) * H_repeat, p = no-tie share among
        newcomers in the all-events sample.
Task 2a log loss, Brier, ECE (10 bins) and AUC per seed, on test.
Task 3  mean and median NLL, log-scale MAE of the predicted median, KS distance of the PIT from uniform, on test.
"""
import json, os
import numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score

from . import data, submission

T1_METRICS = [("hits@10", lambda r, n: r <= 10), ("hits@100", lambda r, n: r <= 100), ("mrr", lambda r, n: 1 / r),
              ("hits@0.1%", lambda r, n: r / n <= 0.001), ("hits@1%", lambda r, n: r / n <= 0.01)]
T2B_METRICS = [("hits@100", lambda r, n: r <= 100), ("hits@1%", lambda r, n: r / n <= 0.01), ("mrr", lambda r, n: 1 / r)]


def _boot(cid, R=1000, seed=0):
    c = pd.factorize(cid)[0]
    return np.random.default_rng(seed).poisson(1.0, (R, c.max() + 1 if len(c) else 1)).astype(np.float32)[:, c]


def _cell(per_seed, W):
    """per_seed: seeds x queries outcome -> dict(mean, sd, ci)."""
    vals = per_seed.mean(1); bs = (W @ per_seed.mean(0)) / W.sum(1)
    return dict(mean=float(vals.mean()), sd=float(vals.std(ddof=1)) if len(vals) > 1 else None,
                ci=[float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))])


def task1(q, runs):
    R = np.stack(list(runs.values())); n = q.n_candidates.to_numpy(); out = {}
    for split in ("val", "test"):
        for sl in ("all", "novel"):
            m = (q.split.values == split) & q[f"in_{sl}_sample"].values.astype(bool)
            if not m.any(): continue
            W = _boot(q.cascade_id.values[m])
            out[f"{split}/{sl}"] = {k: _cell(np.stack([f(r[m], n[m]).astype(float) for r in R]), W) for k, f in T1_METRICS} | {"n": int(m.sum())}
    return out


def gate(res, heur, meta):
    need = ["popularity"] + list(meta.get("heuristic_inputs", []))
    got = res["val/all"]["hits@100"]["mean"]
    fails = [h for h in need if got < heur[h]["val/all"]["hits@100"]["mean"]]
    return {"passed": not fails, "below": fails}


def newcomer_share(q_full, newcomer):
    """p: no-tie share among newcomer events of the full all-events test sample (also used for subsample scorers)."""
    f = q_full.merge(newcomer[["event_row", "unseen_train"]], on="event_row", how="left")
    a = (f.split.values == "test") & f.unseen_train.fillna(False).values.astype(bool) & f.in_all_sample.values.astype(bool)
    return float(f.novel.values.astype(bool)[a].mean()) if a.any() else 0.0


def task2b(q, runs, newcomer, p):
    fl = q.merge(newcomer[["event_row", "unseen_train"]], on="event_row", how="left")
    s = fl.split.values == "test"; nc = fl.unseen_train.fillna(False).values.astype(bool); nov = fl.novel.values.astype(bool)
    hab = s & nc & ~nov & fl.in_all_sample.values.astype(bool); nvl = s & nc & nov & fl.in_novel_sample.values.astype(bool)
    R = np.stack(list(runs.values())); n = fl.n_candidates.to_numpy(); Wh, Wv = _boot(fl.cascade_id.values[hab]), _boot(fl.cascade_id.values[nvl], seed=1)
    out = {"p_no_tie": p, "n_repeat": int(hab.sum()), "n_no_tie": int(nvl.sum())}
    for k, f in T2B_METRICS:
        ph = np.stack([f(r[hab], n[hab]).astype(float) for r in R]); pv = np.stack([f(r[nvl], n[nvl]).astype(float) for r in R])
        seedv = p * (pv.mean(1) if nvl.any() else 0) + (1 - p) * (ph.mean(1) if hab.any() else 0)
        bs = p * (Wv @ pv.mean(0)) / np.maximum(Wv.sum(1), 1) + (1 - p) * (Wh @ ph.mean(0)) / np.maximum(Wh.sum(1), 1)
        out[k] = dict(mean=float(np.mean(seedv)), sd=float(np.std(seedv, ddof=1)) if len(R) > 1 else None,
                      ci=[float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))])
    return out


def _t2a_metrics(y, p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    ll = -(y * np.log(p) + (1 - y) * np.log(1 - p)); br = (p - y) ** 2
    b = np.minimum((p * 10).astype(int), 9)
    ece = sum(abs(p[b == i].mean() - y[b == i].mean()) * (b == i).mean() for i in range(10) if (b == i).any())
    return dict(log_loss=float(ll.mean()), brier=float(br.mean()), ece=float(ece), auc=float(roc_auc_score(y, p)) if 0 < y.sum() < len(y) else None)


def task2a(t, runs):
    te = t.split.values == "test"; y = t.y.values.astype(float)[te]
    per = [_t2a_metrics(y, r[te]) for r in runs.values()]
    agg = lambda k: dict(mean=float(np.mean([x[k] for x in per])), sd=float(np.std([x[k] for x in per], ddof=1)) if len(per) > 1 else None)
    return {k: agg(k) for k in per[0]} | {"n": int(te.sum()), "newcomer_rate": float(y.mean())}


def task3(t, runs):
    te = t.split.values == "test"; yq = np.log1p(t.gap.values[te])
    nll = np.stack([r[0][te] for r in runs.values()]); mae = np.stack([np.abs(np.log1p(r[2][te]) - yq) for r in runs.values()])
    ks = [float(np.max(np.abs(np.sort(r[1][te]) - (np.arange(te.sum()) + 0.5) / te.sum()))) for r in runs.values()]
    W = _boot(t.cascade_id.values[te])
    return {"nll": _cell(nll, W), "median_nll": float(np.median(nll.mean(0))), "log_mae": _cell(mae, W),
            "ks": dict(mean=float(np.mean(ks)), sd=float(np.std(ks, ddof=1)) if len(ks) > 1 else None), "n": int(te.sum())}


def evaluate(sub):
    """Validate and score a submission folder; returns a JSON-serialisable result and writes <sub>/results.json."""
    meta, found = submission.validate(sub)
    res = {"meta": meta, "version": data.VERSION, "tiers": {}}
    for tier, tasks in found.items():
        r = {"seeds": {}}
        if "task1" in tasks:
            q, runs = submission.load_task1(sub, tier, meta); r["seeds"]["task1"] = sorted(runs)
            r["task1"] = task1(q, runs)
            heur = {h: task1(q, {0: q[f"rank_{h}"].to_numpy(float)}) for h in data.HEURISTICS}
            r["sanity_gate"] = gate(r["task1"], heur, meta)
            nw = pd.read_parquet(data.path(tier, "newcomer")); qf = pd.read_parquet(data.path(tier, "queries"))
            # Task 2b uses the slice flags of the full query file, also for subsample scorers (as in the paper)
            f = qf.set_index("event_row").loc[q.event_row.values, ["in_all_sample", "in_novel_sample"]].to_numpy()
            q2 = q.copy(); q2[["in_all_sample", "in_novel_sample"]] = f
            r["task2b"] = task2b(q2, runs, nw, newcomer_share(qf, nw))
        if "task2a" in tasks:
            t, runs = submission.load_task2a(sub, tier); r["seeds"]["task2a"] = sorted(runs); r["task2a"] = task2a(t, runs)
        if "task3" in tasks:
            t, runs = submission.load_task3(sub, tier); r["seeds"]["task3"] = sorted(runs); r["task3"] = task3(t, runs)
        res["tiers"][tier] = r
    json.dump(res, open(os.path.join(sub, "results.json"), "w"), indent=1)
    return res
