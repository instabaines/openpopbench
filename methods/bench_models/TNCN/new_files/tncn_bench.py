"""TNCN (GraphPKU/TNCN @ 4c93a2a, official release) on benchmark tables. Released modules (TGNMemory, GraphAttention
embedding, NCNPredictor, LastNeighborLoader, EarlyStopMonitor), the released train() (copied verbatim from
examples/linkproppred/tgbl-dataset/TNCN.py, which runs at import and cannot be imported), released defaults plus the
paper's Table 5 hyperparameters of the TGB dataset nearest in edge count (log scale; fixed before any result: Bluesky S ->
Review, Telegram -> Coin; the G-repro used the Wiki row).

    bm-tncn/bin/python tncn_bench.py BENCH FLOORS.parquet OUTDIR SEED [BUDGET_HOURS=8]
    EVAL_PROBE=out.json: after one epoch, time scoring on 200 test queries of FLOORS and exit

Temporal graph (protocol F2): participant p is node p, cascade c is node N+c; every event is an edge cascade -> participant
in the floors' event order; msg = a zero 1-dim feature (no edge features). Train edges = train-split events; validation
edges = validation-window events.
Changes (BENCH_CHANGES.md): (1) validation MRR with 20 uniform participant negatives (TGB ships none for this data),
computed on 5,000 evenly spaced validation edges while every validation edge updates memory and neighbours (the released
per-edge loop over all 360k validation edges of Bluesky S would take about an hour per epoch); (2) the time budget;
(3) scoring: memory and neighbour loader are reset and every event before the query (event order) is streamed through
update_state / insert in batches of bs; at each floors query, the released test() path scores (cascade, v) for every
candidate v, in chunks. FLOORS should be the 10% query subsample (paper, Section 3.4). Candidates as floors_eval.py; ties count half.
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
TABLE5 = {"tgbl-wiki": (157_474, 15, 20, 5, 184, 184, 100), "tgbl-review": (4_873_540, 15, 10, 3, 184, 184, 100),       # paper Table 5
          "tgbl-coin": (22_809_486, 10, 5, 3, 100, 100, 100), "tgbl-comment": (44_314_507, 10, 3, 2, 100, 100, 100)}
E_bench = duckdb.connect().execute(f"SELECT count(*) FROM read_parquet('{B}/events.parquet')").fetchone()[0]
TGB_NAME = min(TABLE5, key=lambda k: abs(np.log(TABLE5[k][0] / E_bench)))   # config rule: nearest TGB dataset by edge count (log scale)
_, nn_, ne_, pa_t, md_, ed_, td_ = TABLE5[TGB_NAME]
sys.argv = [sys.argv[0], "--num_neighbors", str(nn_), "--num_epoch", str(ne_), "--patience", str(pa_t), "--mem_dim", str(md_), "--emb_dim", str(ed_),
            "--time_dim", str(td_), "--seed", str(SEED)]
from torch_geometric.data import TemporalData  # noqa: E402
from torch_geometric.loader import TemporalDataLoader  # noqa: E402
from tgb.utils.utils import get_args, set_random_seed  # noqa: E402
from tgb.linkproppred.evaluate import Evaluator  # noqa: E402
from modules.emb_module import GraphAttentionEmbedding  # noqa: E402
from modules.msg_func import IdentityMessage  # noqa: E402
from modules.msg_agg import LastAggregator  # noqa: E402
from modules.neighbor_loader import LastNeighborLoader  # noqa: E402
from modules.memory_module import TGNMemory  # noqa: E402
from modules.early_stopping import EarlyStopMonitor  # noqa: E402
from modules.NCNDecoder.NCNPred import NCNPredictor  # noqa: E402

args, _ = get_args()
assert (args.msg_func, args.emb_func, args.agg_func, args.nei_sampler) == ("identity", "GraphAttention", "last", "l")
LR, BATCH_SIZE, NUM_EPOCH, MEM_DIM, TIME_DIM, EMB_DIM = args.lr, args.bs, args.num_epoch, args.mem_dim, args.time_dim, args.emb_dim
TOLERANCE, PATIENCE, NUM_NEIGHBORS, HOP_NUM, NCN_MODE = args.tolerance, args.patience, args.num_neighbors, args.hop_num, args.NCN_mode
CN_TIME_DECAY = False; device = torch.device("cuda")
os.makedirs(OUT, exist_ok=True)
log = lambda s: print(f"[tncn {time.strftime('%H:%M:%S')}] {s}", flush=True)

con = duckdb.connect(); con.execute("SET threads=32; SET memory_limit='100GB'")
P = f"read_parquet('{B}/participants.parquet')"; cols = [r[0] for r in con.execute(f"DESCRIBE SELECT * FROM {P}").fetchall()]
ev = con.execute(f"""SELECT e.cascade_id, e.participant_id, e.t, (CASE c.split WHEN 'train' THEN 0 WHEN 'val' THEN 1 ELSE 2 END)::TINYINT split
                     FROM read_parquet('{B}/events.parquet') e JOIN read_parquet('{B}/cascades.parquet') c USING (cascade_id)
                     ORDER BY e.t, e.cascade_id, e."position" """).fetchnumpy()
pa_ = con.execute(f"SELECT coalesce(first_seen_t, 9e18)::BIGINT f{', is_crawled' if 'is_crawled' in cols else ''} FROM {P} ORDER BY participant_id").fetchnumpy()
N = len(pa_["f"]); E = len(ev["t"])
val_start = int(con.execute(f"SELECT min(t0) FROM read_parquet('{B}/cascades.parquet') WHERE split = 'val'").fetchone()[0])
test_start = int(con.execute(f"SELECT min(t0) FROM read_parquet('{B}/cascades.parquet') WHERE split = 'test'").fetchone()[0])
data = TemporalData(src=torch.from_numpy((N + ev["cascade_id"]).astype(np.int64)), dst=torch.from_numpy(ev["participant_id"].astype(np.int64)),
                    t=torch.from_numpy(ev["t"].astype(np.int64)), msg=torch.zeros(E, 1)).to(device)
train_mask = torch.from_numpy(ev["split"] == 0).to(device); val_mask = (data.t >= val_start) & (data.t < test_start)
train_data, val_data = data[train_mask], data[val_mask]
ntr = int(train_mask.sum())    # neighbour-loader edge ids are insertion counters: they equal global edge ids only if train is a prefix, val next
assert bool(train_mask[:ntr].all()) and bool(val_mask[ntr:ntr + int(val_mask.sum())].all()), "train must be a prefix of the stream, validation right after"
train_loader = TemporalDataLoader(train_data, batch_size=BATCH_SIZE); val_loader = TemporalDataLoader(val_data, batch_size=BATCH_SIZE)
min_dst_idx, max_dst_idx = int(data.dst.min()), int(data.dst.max())
neighbor_loader = LastNeighborLoader(data.num_nodes, size=NUM_NEIGHBORS, device=device)
torch.manual_seed(SEED); set_random_seed(SEED)
memory = TGNMemory(data.num_nodes, data.msg.size(-1), MEM_DIM, TIME_DIM, message_module=IdentityMessage(data.msg.size(-1), MEM_DIM, TIME_DIM),
                   aggregator_module=LastAggregator(), t_enc_grad=True).to(device)
gnn = GraphAttentionEmbedding(in_channels=MEM_DIM, out_channels=EMB_DIM, msg_dim=data.msg.size(-1), time_enc=memory.time_enc).to(device)
link_pred = NCNPredictor(in_channels=EMB_DIM, hidden_channels=256, out_channels=1, NCN_mode=NCN_MODE).to(device)
model = {"memory": memory, "gnn": gnn, "link_pred": link_pred}
optimizer = torch.optim.Adam(set(memory.parameters()) | set(gnn.parameters()) | set(link_pred.parameters()), lr=LR)
criterion = torch.nn.BCEWithLogitsLoss()
assoc = torch.empty(data.num_nodes, dtype=torch.long, device=device)
evaluator = Evaluator(name="tgbl-wiki"); metric = "mrr"
log(f"config: Table 5 row {TGB_NAME} (nearest TGB dataset by edge count to {E_bench:,})")
log(f"{N:,} participants, {data.num_nodes:,} nodes, {E:,} edges (train {train_data.num_events:,}, val {val_data.num_events:,}); args {vars(args)}")


def find_neighbor(neighbor_loader, n_id, k=1):      # released, verbatim
    for i in range(k-1):
        n_id, _, _ = neighbor_loader(n_id)
    neighbor_info = neighbor_loader(n_id)
    return neighbor_info


def train():                                          # released, verbatim (tqdm removed)
    model['memory'].train()
    model['gnn'].train()
    model['link_pred'].train()

    model['memory'].reset_state()  # Start with a fresh memory.
    neighbor_loader.reset_state()  # Start with an empty graph.

    total_loss = 0

    for batch in train_loader:
        batch = batch.to(device)
        optimizer.zero_grad()

        src, pos_dst, t, msg = batch.src, batch.dst, batch.t, batch.msg

        # Sample negative destination nodes.
        neg_dst = torch.randint(
            min_dst_idx,
            max_dst_idx + 1,
            (src.size(0),),
            dtype=torch.long,
            device=device,
        )

        n_id = torch.cat([src, pos_dst, neg_dst]).unique()
        n_id, edge_index, e_id = find_neighbor(neighbor_loader, n_id, HOP_NUM)
        assoc[n_id] = torch.arange(n_id.size(0), device=device)

        z, last_update = model['memory'](n_id)
        z = model['gnn'](
            z,
            last_update,
            edge_index,
            data.t[e_id].to(device),
            data.msg[e_id].to(device),
        )

        src_re = assoc[src]
        pos_re = assoc[pos_dst]
        neg_re = assoc[neg_dst]

        time_info = (last_update, t)
        pos_out = model['link_pred'](z, edge_index, torch.stack([src_re,pos_re]), NCN_MODE, cn_time_decay=CN_TIME_DECAY, time_info=time_info)
        neg_out = model['link_pred'](z, edge_index, torch.stack([src_re,neg_re]), NCN_MODE, cn_time_decay=CN_TIME_DECAY, time_info=time_info)
        loss = criterion(pos_out, torch.ones_like(pos_out))
        loss += criterion(neg_out, torch.zeros_like(neg_out))

        model['memory'].update_state(src, pos_dst, t, msg)
        neighbor_loader.insert(src, pos_dst)

        loss.backward()
        optimizer.step()
        model['memory'].detach()
        total_loss += float(loss) * batch.num_events
        if (time.time() - t0) / 3600 > BUDGET: break

    return total_loss / train_data.num_events


@torch.no_grad()
def score_pairs(s_node, dsts, t_q):
    """The released test() scoring path for one source against the destinations `dsts` (first = positive)."""
    src = torch.full((len(dsts),), s_node, device=device)
    n_id = torch.cat([src, dsts]).unique()
    n_id, edge_index, e_id = find_neighbor(neighbor_loader, n_id, HOP_NUM)
    assoc[n_id] = torch.arange(n_id.size(0), device=device)
    z, last_update = model['memory'](n_id)
    z = model['gnn'](z, last_update, edge_index, data.t[e_id].to(device), data.msg[e_id].to(device))
    time_info = (last_update, t_q.repeat(len(src)))
    return model['link_pred'](z, edge_index, torch.stack([assoc[src], assoc[dsts]]), NCN_MODE, cn_time_decay=CN_TIME_DECAY, time_info=time_info).squeeze(-1)


@torch.no_grad()
def validate():
    """Released one-vs-many MRR (20 uniform negatives) on 5,000 evenly spaced validation edges; all edges update state."""
    model['memory'].eval("val"); model['gnn'].eval(); model['link_pred'].eval()
    rng = np.random.default_rng(0); pick = set(np.linspace(0, val_data.num_events - 1, min(5000, val_data.num_events)).astype(int).tolist())
    perf, base = [], 0
    for pos_batch in val_loader:
        for idx in range(pos_batch.num_events):
            if base + idx in pick:
                y = int(pos_batch.dst[idx]); neg = rng.choice(N - 1, 20, replace=False); neg[neg >= y] += 1
                out = score_pairs(int(pos_batch.src[idx]), torch.tensor(np.concatenate([[y], neg]), device=device), pos_batch.t[idx])
                perf.append(evaluator.eval({"y_pred_pos": np.array([out[0].cpu()]), "y_pred_neg": np.array(out[1:].cpu()), "eval_metric": [metric]})[metric])
        model['memory'].update_state(pos_batch.src, pos_batch.dst, pos_batch.t, pos_batch.msg); neighbor_loader.insert(pos_batch.src, pos_batch.dst)
        base += pos_batch.num_events
    return float(np.mean(perf))


fl = pd.read_parquet(FL).sort_values("event_row").reset_index(drop=True)
first = torch.from_numpy(pa_["f"]).to(device); elig = torch.from_numpy(pa_["is_crawled"].astype(bool) if "is_crawled" in pa_ else np.ones(N, bool)).to(device)
prefix = pd.DataFrame({"c": ev["cascade_id"], "p": ev["participant_id"], "r": np.arange(E)}).groupby("c")
CH = int(os.environ.get("SCORE_CHUNK", 8192))


@torch.no_grad()
def score_all(rows):
    model['memory'].eval("val"); model['gnn'].eval(); model['link_pred'].eval()
    model['memory'].reset_state(); neighbor_loader.reset_state(); ptr = 0
    Rk = np.zeros(len(rows)); nc = np.zeros(len(rows))
    for k, r in enumerate(rows):
        r = int(r)
        while ptr < r:
            b = min(ptr + BATCH_SIZE, r)
            model['memory'].update_state(data.src[ptr:b], data.dst[ptr:b], data.t[ptr:b], data.msg[ptr:b]); neighbor_loader.insert(data.src[ptr:b], data.dst[ptr:b]); ptr = b
        c = int(ev["cascade_id"][r]); y = int(ev["participant_id"][r]); tq = int(ev["t"][r])
        g = prefix.get_group(c); earlier = g.p.values[g.r.values < r]
        cand = elig & (first < tq); cand[torch.from_numpy(earlier.astype(np.int64)).to(device)] = False; cand[y] = True
        others = torch.nonzero(cand).squeeze(1); others = others[others != y]; nc[k] = 1 + len(others)
        yt = torch.tensor([y], device=device); sc = []
        for a in range(0, max(len(others), 1), CH - 1):
            sc.append(score_pairs(N + c, torch.cat([yt, others[a:a + CH - 1]]), data.t[r]))
        v = sc[0][0]; allsc = torch.cat([z[1:] for z in sc])
        Rk[k] = 1 + float((allsc > v).sum()) + 0.5 * float((allsc == v).sum())
        if k % 1000 == 0: log(f"scored {k:,}/{len(rows):,}")
    return Rk, nc


early_stopper = EarlyStopMonitor(save_model_dir=f"{OUT}/tncn_ckpt/", save_model_id=f"tncn_seed{SEED}", tolerance=TOLERANCE, patience=PATIENCE)
hist, t0 = [], time.time()
for epoch in (range(1, NUM_EPOCH + 1) if not os.environ.get("DEBUG_NEG") else []):     # DEBUG_NEG: diagnostic on the saved checkpoint
    loss = train()
    if os.environ.get("EVAL_PROBE"):
        te = np.flatnonzero(fl.split.values == "test"); sel = np.sort(fl.event_row.values[te[np.linspace(0, len(te) - 1, 200).astype(int)]])
        torch.cuda.synchronize(); a = time.time(); _, nc = score_all(sel); torch.cuda.synchronize(); sec = time.time() - a
        res = dict(method="TNCN", bench=B, floors=FL, epoch_hours=(time.time() - t0 - sec) / 3600, probe_queries=200, probe_pairs=float(nc.sum()), probe_sec=sec,
                   pairs_per_sec=nc.sum() / sec, total_pairs=float(fl.n_candidates.sum()), est_scoring_hours=fl.n_candidates.sum() / (nc.sum() / sec) / 3600,
                   peak_mem_gb=torch.cuda.max_memory_allocated() / 2 ** 30, note="probe includes the update stream up to each query")
        json.dump(res, open(os.environ["EVAL_PROBE"], "w"), indent=1); log(f"[probe] {res}"); sys.exit(0)
    mrr = validate(); hist.append(dict(epoch=epoch, loss=loss, val_mrr=mrr, train_hours=(time.time() - t0) / 3600)); log(f"epoch {hist[-1]}")
    if early_stopper.step_check(mrr, model) or (time.time() - t0) / 3600 > BUDGET: break
train_hours = (time.time() - t0) / 3600
early_stopper.load_checkpoint(model)
if os.environ.get("DEBUG_NEG"):
    # G-sanity diagnostic: at 300 test queries, rank of the target among (a) all candidates, (b) 20 uniform participants
    # (the validation protocol), (c) 20 uniform floors candidates; same streamed state and scorer as score_all
    model['memory'].reset_state(); neighbor_loader.reset_state()          # empty store first: eval() flushes it when called in train mode
    model['memory'].eval("val"); model['gnn'].eval(); model['link_pred'].eval()
    te = np.flatnonzero((fl.split.values == "test") & fl.in_all_sample.values); sel = np.sort(fl.event_row.values[te[np.linspace(0, len(te) - 1, 300).astype(int)]])
    rng = np.random.default_rng(0); ptr = 0; res = {"full": [], "uniform20": [], "cand20": []}
    rk = lambda sc: 1 + float((sc[1:] > sc[0]).sum()) + 0.5 * float((sc[1:] == sc[0]).sum())
    with torch.no_grad():
        for r in sel:
            r = int(r)
            while ptr < r:
                b = min(ptr + BATCH_SIZE, r); model['memory'].update_state(data.src[ptr:b], data.dst[ptr:b], data.t[ptr:b], data.msg[ptr:b]); neighbor_loader.insert(data.src[ptr:b], data.dst[ptr:b]); ptr = b
            c = int(ev["cascade_id"][r]); y = int(ev["participant_id"][r]); tq = int(ev["t"][r])
            g = prefix.get_group(c); earlier = g.p.values[g.r.values < r]
            cand = elig & (first < tq); cand[torch.from_numpy(earlier.astype(np.int64)).to(device)] = False; cand[y] = True
            others = torch.nonzero(cand).squeeze(1); others = others[others != y]; yt = torch.tensor([y], device=device)
            full = torch.cat([score_pairs(N + c, torch.cat([yt, others[a:a + CH - 1]]), data.t[r])[1:] for a in range(0, len(others), CH - 1)] )
            v = score_pairs(N + c, yt, data.t[r])[:1]
            res["full"].append(1 + float((full > v).sum()) + 0.5 * float((full == v).sum()))
            u = rng.choice(N - 1, 20, replace=False); u[u >= y] += 1
            res["uniform20"].append(rk(score_pairs(N + c, torch.cat([yt, torch.from_numpy(u).to(device)]), data.t[r])))
            cc = others[torch.from_numpy(rng.choice(len(others), 20, replace=False)).to(device)]
            res["cand20"].append(rk(score_pairs(N + c, torch.cat([yt, cc]), data.t[r])))
    summ = {k: dict(mrr=float(np.mean(1 / np.array(v))), median_rank=float(np.median(v))) for k, v in res.items()}
    json.dump(dict(seed=SEED, n=len(sel), **summ), open(os.environ["DEBUG_NEG"], "w"), indent=1); log(f"[debug] {summ}"); sys.exit(0)
a = time.time(); Rk, nc = score_all(fl.event_row.values)
assert np.array_equal(nc, fl.n_candidates.values), "candidate set differs from floors"
log(f"scored {len(fl):,} queries ({nc.sum():.3e} pairs) in {time.time() - a:.0f} s")
out = fl[["event_row", "cascade_id", "participant_id", "t", "split", "position", "in_all_sample", "in_novel_sample", "novel", "n_candidates"]].copy()
out["rank_tncn"] = Rk; out.to_parquet(f"{OUT}/tncn_seed{SEED}.parquet", index=False)
json.dump(dict(seed=SEED, hist=hist, train_hours=train_hours, scoring_hours=(time.time() - a) / 3600, budget_h=BUDGET, args={k: str(v) for k, v in vars(args).items()}),
          open(f"{OUT}/tncn_seed{SEED}.json", "w"), indent=1)
log("DONE")
