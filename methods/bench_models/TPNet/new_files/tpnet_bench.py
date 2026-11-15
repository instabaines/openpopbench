"""TPNet (NeurIPS 2024, lxd99/TPNet, official release) on benchmark tables. Released model, loss, training loop,
validation and early stopping; released TPNet defaults with --use_random_projection (the reproduced README command).

    bm-tpnet/bin/python tpnet_bench.py BENCH FLOORS.parquet OUTDIR SEED [BUDGET_HOURS=8]
    EVAL_PROBE=out.json ...   measure full-candidate scoring throughput on 200 queries after 1 training epoch, then exit

Temporal graph (next participant = a timed link between a participant and a cascade):
participant p is node p+1, cascade c is node N+1+c; every event (origin included) is an edge cascade -> participant at
its time; edge ids follow the floors' event_row order (t, cascade_id, position), +1. Features: none, zero-padded to 172
dimensions, as DyGLib does for featureless datasets. Train edges = train-split events (all precede the validation
window); validation edges = validation-window events (the released AP/AUC early stopping on random negatives, patience
20, 100 epochs, plus the time budget). Test edges are never used for training or selection.
Scoring (the only addition, BENCH_CHANGES.md): reset the random projections as at the start of every released epoch,
stream every event in time order through the released update, and at each floors query (time t, cascade c) score the
pair (c, v) for every candidate v with the released backbone and link predictor, using all edges strictly before t
(the released full-graph neighbour sampler semantics). Neighbour retrieval for scoring uses a vectorised implementation
of the released 'recent' strategy, checked equal to the released sampler on 5,000 random (node, time) pairs every run.
"""
import json
import logging
import os
import sys
import time

import duckdb
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

os.environ['CUBLAS_WORKSPACE_CONFIG'] = ":16:8"     # as the released train_link_prediction.py (deterministic algorithms)
B, FL, OUT, SEED = sys.argv[1:5]
SEED = int(SEED); BUDGET = float(sys.argv[5]) if len(sys.argv) > 5 else 8.0
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE); os.chdir(HERE); sys.argv = [sys.argv[0]]
from models.TPNet import TPNet, RandomProjectionModule  # noqa: E402
from models.modules import LinkPredictor_v1  # noqa: E402
from utils.utils import set_random_seed, convert_to_gpu, create_optimizer, get_neighbor_sampler, NegativeEdgeSampler  # noqa: E402
from utils.DataLoader import Data, get_idx_data_loader  # noqa: E402
from utils.EarlyStopping import EarlyStopping  # noqa: E402
from utils.metrics import get_link_prediction_metrics  # noqa: E402
from utils.load_configs import get_link_prediction_args  # noqa: E402
from evaluate_models_utils import evaluate_model_link_prediction  # noqa: E402

sys.argv += ["--model_name", "TPNet", "--use_random_projection", "--load_best_configs", "--dataset_name", "wikipedia"]
args = get_link_prediction_args(is_evaluation=False)            # released defaults (dataset-specific branches do not touch TPNet)
args.device = "cuda:0"; args.seed = SEED
os.makedirs(OUT, exist_ok=True)
log = lambda s: print(f"[tpnet {time.strftime('%H:%M:%S')}] {s}", flush=True)
logger = logging.getLogger("tpnet_bench"); logger.setLevel(logging.WARNING)

con = duckdb.connect(); con.execute("SET threads=32; SET memory_limit='100GB'")
P = f"read_parquet('{B}/participants.parquet')"; cols = [r[0] for r in con.execute(f"DESCRIBE SELECT * FROM {P}").fetchall()]
ev = con.execute(f"""SELECT e.cascade_id, e.participant_id, e.t, (CASE c.split WHEN 'train' THEN 0 WHEN 'val' THEN 1 ELSE 2 END)::TINYINT split
                     FROM read_parquet('{B}/events.parquet') e JOIN read_parquet('{B}/cascades.parquet') c USING (cascade_id)
                     ORDER BY e.t, e.cascade_id, e."position" """).fetchnumpy()           # = floors event_row order
pa_ = con.execute(f"SELECT coalesce(first_seen_t, 9e18)::BIGINT f{', is_crawled' if 'is_crawled' in cols else ''} FROM {P} ORDER BY participant_id").fetchnumpy()
N = len(pa_["f"]); C = int(ev["cascade_id"].max()) + 1; E = len(ev["t"]); NN = N + C + 1
NN0 = NN   # nodes that exist in the data
if os.environ.get("NODES"): NN = max(NN, int(os.environ["NODES"]))   # leakage test (2026-11-08): pad the node table to a checkpoint's size; padded cascade nodes have no events
src = (N + 1 + ev["cascade_id"]).astype(np.longlong); dst = (ev["participant_id"] + 1).astype(np.longlong)
ts = ev["t"].astype(np.float64); eid = np.arange(1, E + 1, dtype=np.longlong)
val_start = float(con.execute(f"SELECT min(t0) FROM read_parquet('{B}/cascades.parquet') WHERE split = 'val'").fetchone()[0])
test_start = float(con.execute(f"SELECT min(t0) FROM read_parquet('{B}/cascades.parquet') WHERE split = 'test'").fetchone()[0])
trm = ev["split"] == 0; vam = (ts >= val_start) & (ts < test_start)
assert ts[trm].max() < val_start, "train events must precede the validation window"
mk = lambda m: Data(src_node_ids=src[m], dst_node_ids=dst[m], node_interact_times=ts[m], edge_ids=eid[m], labels=np.zeros(int(m.sum())))
full_data, train_data, val_data = mk(np.ones(E, bool)), mk(trm), mk(vam)
node_raw_features = np.zeros((NN, 172), np.float32); edge_raw_features = np.zeros((E + 1, 172), np.float32)
log(f"{N:,} participants, {C:,} cascades, {E:,} edges (train {trm.sum():,}, val window {vam.sum():,}); rp dim factor {args.rp_dim_factor}")

train_neighbor_sampler = get_neighbor_sampler(data=train_data, sample_neighbor_strategy=args.sample_neighbor_strategy, time_scaling_factor=args.time_scaling_factor, seed=0)
full_neighbor_sampler = get_neighbor_sampler(data=full_data, sample_neighbor_strategy=args.sample_neighbor_strategy, time_scaling_factor=args.time_scaling_factor, seed=1)
train_neg_edge_sampler = NegativeEdgeSampler(src_node_ids=train_data.src_node_ids, dst_node_ids=train_data.dst_node_ids)
val_neg_edge_sampler = NegativeEdgeSampler(src_node_ids=full_data.src_node_ids, dst_node_ids=full_data.dst_node_ids, seed=0)
train_idx_data_loader = get_idx_data_loader(indices_list=list(range(len(train_data.src_node_ids))), batch_size=args.batch_size, shuffle=False)
val_idx_data_loader = get_idx_data_loader(indices_list=list(range(len(val_data.src_node_ids))), batch_size=args.batch_size, shuffle=False)


class RecentSampler:
    """Vectorised released 'recent' strategy over the full graph: the last num_neighbors edges strictly before t,
    right-aligned, ordered by (time, edge id) as the released stable sort of the edge-ordered adjacency lists."""
    def __init__(self):
        node = np.concatenate([src, dst]); nb = np.concatenate([dst, src]); e = np.concatenate([eid, eid]); t = np.concatenate([ts, ts])
        o = np.lexsort((e, t, node)); self.node, self.nb, self.e, self.t = node[o], nb[o], e[o], t[o]
        self.t0 = float(ts.min()); self.R = float(ts.max() - ts.min() + 2)
        self.key = self.node.astype(np.float64) * self.R + (self.t - self.t0)
        assert self.key.max() < 2 ** 53, "time key exceeds float64 integer precision"
        self.ptr = np.searchsorted(self.node, np.arange(NN + 1))
        self.sample_neighbor_strategy = "recent"; self.seed = None

    def get_historical_neighbors(self, node_ids, node_interact_times, num_neighbors=20):
        i = np.searchsorted(self.key, node_ids.astype(np.float64) * self.R + (node_interact_times - self.t0), side="left")
        idx = i[:, None] - num_neighbors + np.arange(num_neighbors)[None, :]; ok = idx >= self.ptr[node_ids][:, None]
        idx = np.where(ok, idx, 0)
        return (np.where(ok, self.nb[idx], 0).astype(np.longlong), np.where(ok, self.e[idx], 0).astype(np.longlong), np.where(ok, self.t[idx], 0.0))


fast = RecentSampler(); rs = np.random.RandomState(SEED)
chk_n = rs.randint(1, NN0, 5000); chk_t = ts[rs.randint(0, E, 5000)]
for a_, b_ in zip(full_neighbor_sampler.get_historical_neighbors(chk_n, chk_t, args.num_neighbors), fast.get_historical_neighbors(chk_n, chk_t, args.num_neighbors)):
    assert np.array_equal(a_, b_), "vectorised sampler differs from the released sampler"
log("vectorised 'recent' sampler equals the released sampler on 5,000 random (node, time) pairs")
if os.environ.get("DRYRUN"): sys.exit(0)                # CPU check of the export and the sampler equivalence

set_random_seed(seed=SEED, deterministic_alg=True)
random_projections = RandomProjectionModule(node_num=NN, edge_num=E + 1, dim_factor=args.rp_dim_factor, num_layer=args.rp_num_layer,
                                            time_decay_weight=args.rp_time_decay_weight, device=args.device, use_matrix=args.rp_use_matrix,
                                            beginning_time=train_data.node_interact_times[0], not_scale=args.rp_not_scale, enforce_dim=args.enforce_dim)
backbone = TPNet(node_raw_features=node_raw_features, edge_raw_features=edge_raw_features, neighbor_sampler=train_neighbor_sampler,
                 time_feat_dim=args.time_feat_dim, random_projections=None if args.encode_not_rp else random_projections,
                 num_neighbors=args.num_neighbors, num_layers=args.num_layers, dropout=args.dropout, device=args.device)
link_predictor = LinkPredictor_v1(input_dim1=172, input_dim2=172, hidden_dim=172, output_dim=1,
                                  random_projections=None if args.decode_not_rp else random_projections, not_encode=args.not_encode)
model = convert_to_gpu(nn.Sequential(backbone, link_predictor), device=args.device)
optimizer = create_optimizer(model=model, optimizer_name=args.optimizer, learning_rate=args.learning_rate, weight_decay=args.weight_decay)
ck = f"{OUT}/tpnet_seed{SEED}.pkl"
early_stopping = EarlyStopping(patience=args.patience, save_model_path=ck, logger=logger, model_name=args.model_name)
loss_func = nn.BCEWithLogitsLoss()
fl = pd.read_parquet(FL).sort_values("event_row").reset_index(drop=True)
first = torch.from_numpy(pa_["f"]).cuda(); elig = torch.from_numpy(pa_["is_crawled"].astype(bool) if "is_crawled" in pa_ else np.ones(N, bool)).cuda()
cstart_ev = pd.DataFrame({"c": ev["cascade_id"], "p": ev["participant_id"], "r": np.arange(E)})       # earlier participants per query


@torch.no_grad()
def score_all(rows, chunk=int(os.environ.get("SCORE_CHUNK", 8192))):
    """Released update stream from a reset; full-candidate rank of each floors query in `rows` (ascending event_row)."""
    model.eval(); model[0].set_neighbor_sampler(fast); random_projections.reset_random_projections()
    Rk = np.zeros(len(rows)); nc = np.zeros(len(rows)); ptr = 0; pending = []
    prefix = cstart_ev.groupby("c")
    for k, r in enumerate(rows):
        r = int(r); tq = ts[r]; c = int(ev["cascade_id"][r]); y = int(ev["participant_id"][r])
        stop = np.searchsorted(ts, tq, side="left")                                 # edges strictly before t
        while ptr < stop:
            b = min(ptr + args.batch_size, stop); random_projections.update(src[ptr:b], dst[ptr:b], ts[ptr:b]); ptr = b
        g = prefix.get_group(c); earlier = g.p.values[g.r.values < r]
        cand = elig & (first < int(tq)); cand[torch.from_numpy(earlier.astype(np.int64)).cuda()] = False; cand[y] = True
        cv = torch.nonzero(cand).squeeze(1).cpu().numpy(); nc[k] = len(cv)
        sc = []
        for a in range(0, len(cv), chunk):
            d = (cv[a:a + chunk] + 1).astype(np.longlong); s = np.full(len(d), N + 1 + c, np.longlong); tt = np.full(len(d), tq)
            se, de = model[0].compute_src_dst_node_temporal_embeddings(src_node_ids=s, dst_node_ids=d, node_interact_times=tt)
            sc.append(model[1](src_node_ids=s, dst_node_ids=d, src_node_embeddings=se, dst_node_embeddings=de).squeeze(-1))
        sc = torch.cat(sc); v = sc[int(np.searchsorted(cv, y))]
        Rk[k] = 1 + float((sc > v).sum()) + 0.5 * (float((sc == v).sum()) - 1)
    return Rk, nc


hist = []; t0 = time.time()
if os.environ.get("MAX_EPOCHS"): args.num_epochs = int(os.environ["MAX_EPOCHS"])   # one full epoch at scale (paper, Section 4.4)
for epoch in range(0 if os.environ.get("SCORE_ONLY") else args.num_epochs):   # SCORE_ONLY=1: score the saved checkpoint in OUT (leakage test, 2026-11-08)
    model.train(); model[0].set_neighbor_sampler(train_neighbor_sampler); random_projections.reset_random_projections(); losses = []
    for batch_idx, ix in enumerate(train_idx_data_loader):
        ix = ix.numpy(); bs, bd, bt = train_data.src_node_ids[ix], train_data.dst_node_ids[ix], train_data.node_interact_times[ix]
        _, nd = train_neg_edge_sampler.sample(size=len(bs))
        se, de = model[0].compute_src_dst_node_temporal_embeddings(src_node_ids=bs, dst_node_ids=bd, node_interact_times=bt)
        nse, nde = model[0].compute_src_dst_node_temporal_embeddings(src_node_ids=bs, dst_node_ids=nd, node_interact_times=bt)
        pos = model[1](src_node_ids=bs, dst_node_ids=bd, src_node_embeddings=se, dst_node_embeddings=de).squeeze(dim=-1)
        neg = model[1](src_node_ids=bs, dst_node_ids=nd, src_node_embeddings=nse, dst_node_embeddings=nde).squeeze(dim=-1)
        random_projections.update(src_node_ids=bs, dst_node_ids=bd, node_interact_times=bt)
        predicts = torch.cat([pos, neg]); labels = torch.cat([torch.ones_like(pos), torch.zeros_like(neg)])
        loss = loss_func(input=predicts, target=labels); optimizer.zero_grad(); loss.backward(); optimizer.step(); losses.append(loss.item())
        if (time.time() - t0) / 3600 > BUDGET: break
        if os.environ.get("EVAL_PROBE") and os.environ.get("PROBE_TRAIN_STEPS") and batch_idx + 1 >= int(os.environ["PROBE_TRAIN_STEPS"]): break
    if os.environ.get("EVAL_PROBE"):                     # scoring-throughput probe: 200 evenly spaced test queries
        te = np.flatnonzero(fl.split.values == "test"); sel = fl.event_row.values[te[np.linspace(0, len(te) - 1, 200).astype(int)]]
        torch.cuda.synchronize(); a = time.time(); _, nc = score_all(np.sort(sel)); torch.cuda.synchronize(); sec = time.time() - a
        pairs_s = nc.sum() / sec; tot = float(fl.n_candidates.sum())
        res = dict(method="TPNet", bench=B, score_chunk=int(os.environ.get("SCORE_CHUNK", 8192)), train_steps_before_probe=batch_idx + 1, epoch_hours=(time.time() - t0) / 3600, probe_queries=200, probe_pairs=float(nc.sum()), probe_sec=sec,
                   pairs_per_sec=pairs_s, total_pairs=tot, est_scoring_hours=tot / pairs_s / 3600, note="probe includes the update stream up to each query")
        json.dump(res, open(os.environ["EVAL_PROBE"], "w"), indent=1); log(f"[probe] {res}"); sys.exit(0)
    _, val_metrics = evaluate_model_link_prediction(model_name=args.model_name, model=model, neighbor_sampler=full_neighbor_sampler,
                                                    evaluate_idx_data_loader=val_idx_data_loader, evaluate_neg_edge_sampler=val_neg_edge_sampler,
                                                    evaluate_data=val_data, loss_func=loss_func, num_neighbors=args.num_neighbors,
                                                    time_gap=args.time_gap, random_projections=random_projections)
    vm = {k: float(np.mean([m[k] for m in val_metrics])) for k in val_metrics[0]}
    hist.append(dict(epoch=epoch, train_loss=float(np.mean(losses)), train_hours=(time.time() - t0) / 3600, **{f"val_{k}": v for k, v in vm.items()}))
    log(f"epoch {epoch} {hist[-1]}")
    if early_stopping.step([(k, v, True) for k, v in vm.items()], model, args) or (time.time() - t0) / 3600 > BUDGET: break
train_hours = (time.time() - t0) / 3600
model.load_state_dict(torch.load(ck, map_location=args.device)["model"])
if os.environ.get("SCORE_ONLY"): torch.manual_seed(SEED)   # same random-projection draw in every score-only run
a = time.time(); Rk, nc = score_all(fl.event_row.values)
assert np.array_equal(nc, fl.n_candidates.values), "candidate set differs from floors"
log(f"scored {len(fl):,} queries ({nc.sum():.3e} pairs) in {time.time() - a:.0f} s")
out = fl[["event_row", "cascade_id", "participant_id", "t", "split", "position", "in_all_sample", "in_novel_sample", "novel", "n_candidates"]].copy()
out["rank_tpnet"] = Rk; out.to_parquet(f"{OUT}/tpnet_seed{SEED}.parquet", index=False)
json.dump(dict(seed=SEED, hist=hist, train_hours=train_hours, scoring_hours=(time.time() - a) / 3600, budget_h=BUDGET, args=vars(args)), open(f"{OUT}/tpnet_seed{SEED}.json", "w"), indent=1, default=str)
log("DONE")
