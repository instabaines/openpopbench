"""EvolveGCN (AAAI 2020, IBM/EvolveGCN, official release) on benchmark tables. Released EGCN-O model, Link_Pred_Tasker
(snapshot adjacencies, 1-hot degree features, smart negative sampling), Classifier, class-weighted Cross_Entropy and the
Trainer's training epoch; configuration experiments/parameters_sbm_egcn_o.yaml (the reproduced one).

    model-env/bin/python evolvegcn_bench.py BENCH FLOORS.parquet OUTDIR SEED [BUDGET_HOURS=8]

Snapshot rule (fixed before any EvolveGCN result, 2026-10-16): aggr_time = max(1 h, train window / 100), so
the training window has about 100 snapshots, the scale of the released datasets (88-138 steps). Bluesky S: 24,191 s.
Graph (protocol F2: a timed link between a participant and a cascade, undirected as in the TGB/DyGLib samplers):
participant p is node p, cascade c is node N+c; every event gives the two directed edges cascade->participant and
participant->cascade in its snapshot (one direction alone leaves every participant with out-degree 0, so all candidate
embeddings would coincide).
Training: the released Trainer.run_epoch over samples idx = min_time + num_hist_steps .. last snapshot whose label snapshot
idx+1 lies wholly before the validation window (train events only).
Changes (BENCH_CHANGES.md): (1) model selection: the released dev evaluation scores all N^2 node pairs (all_edges=True,
2.3e11 pairs on S); instead the released target measure (MAP; with one target per query MAP = MRR) under full ranking on
5,000 benchmark validation queries, every epoch after eval_after_epochs, released early_stop_patience, plus the time budget;
the best gcn/classifier state is kept (the release reports test at the best-valid epoch); (2) scoring: a query at time t in
snapshot k uses the embeddings computed from snapshots k-1-num_hist_steps .. k-1 (all closed before t; the release's
"history up to idx predicts idx+1"); score = classifier logit(link) - logit(no link) for (cascade, v), every candidate v.
Candidates as floors_eval.py; ties count half; candidate counts must equal the floors' n_candidates.
"""
import json
import os
import sys
import time

import duckdb
import numpy as np
import pandas as pd
import torch

B, FL, OUT, SEED = sys.argv[1:5]
SEED = int(SEED); BUDGET = float(sys.argv[5]) if len(sys.argv) > 5 else 8.0
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE); os.chdir(HERE)
sys.argv = [sys.argv[0], "--config_file", "experiments/parameters_sbm_egcn_o.yaml"]
import utils as u  # noqa: E402
import link_pred_tasker as lpt  # noqa: E402
import models as mls  # noqa: E402
import Cross_Entropy as ce  # noqa: E402
import trainer as tr  # noqa: E402
import splitter as sp  # noqa: E402
from run_exp import build_random_hyper_params, build_gcn, build_classifier  # noqa: E402

args = u.parse_args(u.create_parser())
args.device = "cuda"; args.use_cuda = True; args.seed = SEED; args.rank = 0; args.wsize = 1; args.use_logfile = False
np.random.seed(SEED); torch.manual_seed(SEED); torch.cuda.manual_seed_all(SEED); import random; random.seed(SEED)
args = build_random_hyper_params(args)
os.makedirs(OUT, exist_ok=True)
log = lambda s: print(f"[evolvegcn {time.strftime('%H:%M:%S')}] {s}", flush=True)

con = duckdb.connect(); con.execute("SET threads=32; SET memory_limit='100GB'")
P = f"read_parquet('{B}/participants.parquet')"; cols = [r[0] for r in con.execute(f"DESCRIBE SELECT * FROM {P}").fetchall()]
ev = con.execute(f"""SELECT e.cascade_id, e.participant_id, e.t, (CASE c.split WHEN 'train' THEN 0 WHEN 'val' THEN 1 ELSE 2 END)::TINYINT split
                     FROM read_parquet('{B}/events.parquet') e JOIN read_parquet('{B}/cascades.parquet') c USING (cascade_id)
                     ORDER BY e.t, e.cascade_id, e."position" """).fetchnumpy()
pa_ = con.execute(f"SELECT coalesce(first_seen_t, 9e18)::BIGINT f{', is_crawled' if 'is_crawled' in cols else ''} FROM {P} ORDER BY participant_id").fetchnumpy()
N = len(pa_["f"]); C = int(ev["cascade_id"].max()) + 1; E = len(ev["t"])
tmin = int(ev["t"].min())
val_start = int(con.execute(f"SELECT min(t0) FROM read_parquet('{B}/cascades.parquet') WHERE split = 'val'").fetchone()[0])
tr_start = int(con.execute(f"SELECT min(t0) FROM read_parquet('{B}/cascades.parquet') WHERE split = 'train'").fetchone()[0])
AGGR = max(3600, (val_start - tr_start) // 100)
k_ev = (ev["t"].astype(np.int64) - tmin) // AGGR
s_ = torch.from_numpy((N + ev["cascade_id"]).astype(np.int64)); d_ = torch.from_numpy(ev["participant_id"].astype(np.int64)); kk = torch.from_numpy(k_ev)


class BenchDataset:
    """The attributes Link_Pred_Tasker reads (as sbm_dataset provides them)."""
    def __init__(self):
        idx = torch.cat([torch.stack([s_, d_, kk], 1), torch.stack([d_, s_, kk], 1)])      # both directions
        self.edges = {"idx": idx, "vals": torch.ones(idx.size(0))}
        self.num_nodes = N + C; self.max_time = kk.max(); self.min_time = kk.min()
        self.feats_per_node = 1; self.nodes_feats = None; self.num_classes = 2


dataset = BenchDataset()
tasker = lpt.Link_Pred_Tasker(args, dataset)
last_train_label = (val_start - tmin) // AGGR - 1                  # last snapshot wholly before the validation window
train_split = sp.data_split(tasker, int(dataset.min_time) + args.num_hist_steps, last_train_label, test=False)
# change (3), 2026-10-21: drop training samples whose label snapshot window holds no edge (sparse Reddit S): the released
# get_non_existing_edges draws negatives with np.random.choice over the label edges and fails on an empty window
w_lab = int(getattr(args, "adj_mat_time_window", 1) or 1); n_k = np.bincount(k_ev, minlength=int(dataset.max_time) + 2)
s0 = int(dataset.min_time) + args.num_hist_steps
keep = [i for i in range(len(train_split)) if n_k[max(0, s0 + i + 1 - w_lab + 1):s0 + i + 2].sum() > 0]
if len(keep) < len(train_split): log(f"dropped {len(train_split) - len(keep)} of {len(train_split)} training samples with an empty label snapshot")
train_split = torch.utils.data.Subset(train_split, keep)
train_loader = torch.utils.data.DataLoader(train_split, **args.data_loading_params)
log(f"{N:,} participants, {C:,} cascades, {E:,} events; aggr_time {AGGR} s, snapshots {int(dataset.max_time) + 1}, train samples {len(train_split)}; "
    f"1-hot degree feats {tasker.feats_per_node}; lr {args.learning_rate}, hist {args.num_hist_steps}, gcn {args.gcn_parameters}")
gcn = build_gcn(args, tasker); classifier = build_classifier(args, tasker).to(args.device)
loss = ce.Cross_Entropy(args, dataset).to(args.device)
splitter = u.Namespace({"tasker": tasker, "train": train_loader, "dev": [], "test": []})
trainer = tr.Trainer(args, splitter=splitter, gcn=gcn, classifier=classifier, comp_loss=loss, dataset=dataset, num_classes=2)
trainer.tr_step = 0                                    # set in the released Trainer.train(), which this loop replaces
if os.environ.get("PROBE"):   # feasibility probe: released run_epoch on one training sample (one snapshot window) per step
    sys.path.insert(0, os.path.join(os.environ.get("OPB_ROOT", "."), "benchmark_build")); from probe_util import probe
    _it = [iter(train_loader)]
    def _step(bs):
        try: smp = next(_it[0])
        except StopIteration: _it[0] = iter(train_loader); smp = next(_it[0])
        trainer.run_epoch([smp], 0, "TRAIN", grad=True)
    probe(_step, len(train_loader), 1, os.environ["PROBE"], method="EvolveGCN", bench=B, users=N, note="batch = one snapshot window (the release has no batch size); steps per epoch = training samples")

fl = pd.read_parquet(FL).sort_values("event_row").reset_index(drop=True)
first = torch.from_numpy(pa_["f"]).cuda(); elig = torch.from_numpy(pa_["is_crawled"].astype(bool) if "is_crawled" in pa_ else np.ones(N, bool)).cuda()
prefix = pd.DataFrame({"c": ev["cascade_id"], "p": ev["participant_id"], "r": np.arange(E)}).groupby("c")
val_rows = np.flatnonzero(fl.split.values == "val")
val_sel = np.sort(np.random.RandomState(SEED).choice(val_rows, min(5000, len(val_rows)), replace=False))


_gne = lpt.tu.get_non_existing_edges


def _gne_scoring(*a, **k):
    """Change (4): at scoring time only the embeddings are used, not the sample's label negatives; when the label
    snapshot after a query is empty the released sampler fails, so return an empty negative set instead."""
    try: return _gne(*a, **k)
    except IndexError: return {"idx": torch.zeros((0, 2), dtype=torch.long), "vals": torch.zeros(0, dtype=torch.long)}


@torch.no_grad()
def embeddings(idx):
    """Released forward: node embeddings from the snapshots idx-num_hist_steps .. idx (tasker.get_sample inputs)."""
    lpt.tu.get_non_existing_edges = _gne_scoring
    try: raw = torch.utils.data.dataloader.default_collate([tasker.get_sample(idx, test=True)])      # the DataLoader's batch-of-1 format
    finally: lpt.tu.get_non_existing_edges = _gne
    s = trainer.prepare_sample(raw)
    return gcn(s.hist_adj_list, s.hist_ndFeats_list, s.node_mask_list)


@torch.no_grad()
def ranks(rows):
    gcn.eval(); classifier.eval(); Rk = np.zeros(len(rows)); nc = np.zeros(len(rows)); cache = {}
    for k, r in enumerate(fl.event_row.values[rows]):
        r = int(r); tq = int(ev["t"][r]); c = int(ev["cascade_id"][r]); y = int(ev["participant_id"][r]); idx = int(k_ev[r]) - 1
        if idx not in cache: cache.clear(); cache[idx] = embeddings(idx)
        Z = cache[idx]
        g = prefix.get_group(c); earlier = g.p.values[g.r.values < r]
        cand = elig & (first < tq); cand[torch.from_numpy(earlier.astype(np.int64)).cuda()] = False; cand[y] = True
        cv = torch.nonzero(cand).squeeze(1); nc[k] = len(cv)
        x = torch.cat([Z[N + c].expand(len(cv), -1), Z[cv]], 1); out = classifier(x); sc = out[:, 1] - out[:, 0]
        v = sc[int(torch.searchsorted(cv, torch.tensor([y], device=cv.device)))]
        Rk[k] = 1 + float((sc > v).sum()) + 0.5 * (float((sc == v).sum()) - 1)
    gcn.train(); classifier.train()
    assert np.array_equal(nc, fl.n_candidates.values[rows]), "candidate set differs from floors"
    return Rk


assert int(k_ev[fl.event_row.values].min()) - 1 - args.num_hist_steps >= int(dataset.min_time), "query snapshots need num_hist_steps of history"
best, bad, hist, t0 = -1.0, 0, [], time.time(); ck = f"{OUT}/evolvegcn_seed{SEED}.pt"
for e in range(args.num_epochs):
    trainer.run_epoch(train_loader, e, "TRAIN", grad=True)
    if e > args.eval_after_epochs or (time.time() - t0) / 3600 > BUDGET:
        te = time.time(); r = ranks(val_sel); mrr = float((1 / r).mean())
        hist.append(dict(epoch=e, val_mrr=mrr, val_hits100=float((r <= 100).mean()), train_hours=(time.time() - t0) / 3600, eval_s=time.time() - te)); log(f"epoch {hist[-1]}")
        if mrr > best: best, bad = mrr, 0; torch.save({"gcn": [p.detach().clone() for p in gcn.parameters()], "cls": classifier.state_dict()}, ck)
        else: bad += 1
        if bad > args.early_stop_patience: break
    if (time.time() - t0) / 3600 > BUDGET: break
train_hours = (time.time() - t0) / 3600
st = torch.load(ck); classifier.load_state_dict(st["cls"])
with torch.no_grad():          # EGCN replaces self._parameters with a ParameterList, so state_dict() fails: restore tensors in order
    for p, q in zip(gcn.parameters(), st["gcn"]): p.copy_(q)
a = time.time(); Rk = ranks(np.arange(len(fl))); log(f"scored {len(fl):,} queries in {time.time() - a:.0f} s")
out = fl[["event_row", "cascade_id", "participant_id", "t", "split", "position", "in_all_sample", "in_novel_sample", "novel", "n_candidates"]].copy()
out["rank_evolvegcn"] = Rk; out.to_parquet(f"{OUT}/evolvegcn_seed{SEED}.parquet", index=False)
json.dump(dict(seed=SEED, hist=hist, best_val_mrr=best, train_hours=train_hours, aggr_time=AGGR, budget_h=BUDGET,
               args={k: str(v) for k, v in vars(args).items()}), open(f"{OUT}/evolvegcn_seed{SEED}.json", "w"), indent=1)
log("DONE")
