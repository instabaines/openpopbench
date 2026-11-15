"""DCRec (WWW 2023, official release) on benchmark tables. Released model, losses, graphs, trainer and default configuration.

    bm-dgl/bin/python dcrec_bench.py BENCH FLOORS.parquet OUTDIR SEED [BUDGET_HOURS=8]

Mapping: each TRAIN cascade with >= 2 events is a session; participant p is item token p. As in the released files, a
session row holds the input list (the last 50 earlier events) and the final event as target; the released
sequential_augmentation expands it to every prefix. Every participant is listed in bench.item so that the item
vocabulary is the full benchmark population.
Configuration: the release's config_dict with its tuned Reddit setting (the reproduced one; the release's default branch
lacks sim_group, weight_mean and kl_weight and raises KeyError), seed = SEED.
Changes (BENCH_CHANGES.md): (1) the test-time graphs (adj_graph_test, sim_graph_test) are the TRAINING graphs; the release
builds them from the test sequences, which here would leak later events into earlier queries; (2) model selection uses
the released valid_metric, Recall@1, under FULL ranking on 5,000 validation queries (released: pop100 sampled evaluation
on the test file); released stopping_step 20, plus the time budget; (3) scoring ranks the full candidate set with the
released fused scorer (fast_predict's fusion of sequence, adjacency-graph and similarity-graph views, dot product with
the item table).
Evaluation: the query rows of FLOORS.parquet; candidates as floors_eval.py; ties count half; the candidate count must
equal the floors' n_candidates on every query.
"""
import json
import os
import sys
import time

import duckdb
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

B, FL, OUT, SEED = sys.argv[1:5]
SEED = int(SEED); BUDGET = float(sys.argv[5]) if len(sys.argv) > 5 else 8.0
GPU = os.environ.get("CUDA_VISIBLE_DEVICES", "")
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE); sys.argv = [sys.argv[0]]
from recbole.config import Config  # noqa: E402
from recbole.data import create_dataset  # noqa: E402
from recbole.data.utils import get_dataloader, create_samplers  # noqa: E402
from recbole.utils import init_logger, init_seed, get_model, get_trainer  # noqa: E402
from run_DCRec import build_adj_graph, build_sim_graph, sequential_augmentation  # noqa: E402

os.makedirs(OUT, exist_ok=True)
log = lambda s: print(f"[dcrec {time.strftime('%H:%M:%S')}] {s}", flush=True)
con = duckdb.connect(); con.execute("SET threads=32; SET memory_limit='100GB'")
P = f"read_parquet('{B}/participants.parquet')"; cols = [r[0] for r in con.execute(f"DESCRIBE SELECT * FROM {P}").fetchall()]
ev = con.execute(f"""SELECT e.cascade_id, e.participant_id, e.t, e."position", (CASE c.split WHEN 'train' THEN 0 WHEN 'val' THEN 1 ELSE 2 END)::TINYINT split
                     FROM read_parquet('{B}/events.parquet') e JOIN read_parquet('{B}/cascades.parquet') c USING (cascade_id)
                     ORDER BY e.cascade_id, e."position" """).fetchnumpy()
pa_ = con.execute(f"SELECT coalesce(first_seen_t, 9e18)::BIGINT f{', is_crawled' if 'is_crawled' in cols else ''} FROM {P} ORDER BY participant_id").fetchnumpy()
N = len(pa_["f"])
cstart = np.searchsorted(ev["cascade_id"], np.arange(ev["cascade_id"].max() + 2)); csplit = ev["split"][cstart[:-1]]
pid = ev["participant_id"].astype(np.int64)
fl = pd.read_parquet(FL).sort_values("event_row").reset_index(drop=True)
erow = con.execute(f"""SELECT cascade_id, "position" FROM read_parquet('{B}/events.parquet') ORDER BY t, cascade_id, "position" """).fetchnumpy()
qc, qp = erow["cascade_id"][fl.event_row.values], erow["position"][fl.event_row.values]
val_rows = np.flatnonzero(fl.split.values == "val")
val_sel = np.sort(np.random.RandomState(SEED).choice(val_rows, min(5000, len(val_rows)), replace=False))

# ---- export in the released RecBole benchmark-file format
D = f"{OUT}/recbole_data_seed{SEED}/bench"; os.makedirs(D, exist_ok=True)
tr = [c for c in np.flatnonzero(csplit == 0) if cstart[c + 1] - cstart[c] >= 2]
with open(f"{D}/bench.train.inter", "w") as f:
    f.write("session_id:token\titem_id_list:token_seq\titem_id:token\n")
    for s, c in enumerate(tr):
        x = pid[cstart[c]:cstart[c + 1]]; f.write(f"{s}\t{' '.join(map(str, x[:-1][-50:]))}\t{x[-1]}\n")
with open(f"{D}/bench.test.inter", "w") as f:       # the validation-selection queries (the trainer's valid_data)
    f.write("session_id:token\titem_id_list:token_seq\titem_id:token\n")
    for k, i in enumerate(val_sel):
        s0 = cstart[qc[i]]; j = s0 + qp[i]; f.write(f"{len(tr) + k}\t{' '.join(map(str, pid[s0:j][-50:]))}\t{pid[j]}\n")
with open(f"{D}/bench.item", "w") as f:
    f.write("item_id:token\n"); f.write("\n".join(map(str, range(N))) + "\n")
log(f"{N:,} items, {len(tr):,} train sessions, {len(val_sel):,} validation queries exported")

# ---- released configuration (default branch of run_DCRec.py), our paths and seed
config_dict = {
    "seed": SEED, "reproducibility": 0, 'USER_ID_FIELD': 'session_id', 'load_col': None, 'neg_sampling': None,
    'benchmark_filename': ['train', 'test'], 'alias_of_item_id': ['item_id_list'], 'topk': [1, 5, 10],
    'metrics': ['Recall', 'NDCG'], 'valid_metric': 'Recall@1', 'eval_args': {'mode': 'pop100', 'order': 'TO'},
    'gpu_id': GPU, "MAX_ITEM_LIST_LENGTH": 50, "train_batch_size": 512, "eval_batch_size": 256, "stopping_step": 20,
    "fast_sample_eval": 1, "hidden_dropout_prob": 0.3, "attn_dropout_prob": 0.3, "graph_dropout_prob": 0.3,
    "graphcl_enable": 1, "graphcl_coefficient": 1e-4, "cl_ablation": "full", "graph_view_fusion": 1, "cl_temp": 1,
    # released "reddit" BEST SETTINGS: the default branch lacks sim_group / weight_mean / kl_weight and cannot run
    "train_batch_size": 128, "graphcl_coefficient": 1, "weight_mean": 0.5, "sim_group": 4, "kl_weight": 1,
    "data_path": f"{OUT}/recbole_data_seed{SEED}", "checkpoint_dir": f"{OUT}/dcrec_ckpt_seed{SEED}", "show_progress": False}
config = Config(model='DCRec', dataset='bench', config_dict=config_dict)
init_seed(config["seed"], config["reproducibility"]); init_logger(config, 0)          # as released: args.log default 0
dataset = create_dataset(config)
train_dataset, test_dataset = dataset.build()
t_g = time.time()
adj_graph, user_edges = build_adj_graph(train_dataset)
sim_graph = build_sim_graph(train_dataset, config_dict["sim_group"])
log(f"graphs built in {time.time() - t_g:.0f} s")
sequential_augmentation(train_dataset)
train_sampler, _, test_sampler = create_samplers(config, dataset, [train_dataset, test_dataset])
train_data = get_dataloader(config, 'train')(config, train_dataset, train_sampler, shuffle=True)
test_data = get_dataloader(config, 'test')(config, test_dataset, test_sampler, shuffle=False)
external_data = {"adj_graph": adj_graph, "sim_graph": sim_graph, "user_edges": user_edges,
                 "adj_graph_test": adj_graph, "sim_graph_test": sim_graph}           # change (1): training graphs at test time
model = get_model('DCRec')(config, train_data.dataset, external_data).to(config['device'])
dev = config['device']
tok2id = dataset.field2token_id['item_id']
iid = torch.tensor([tok2id[str(p)] for p in range(N)], dtype=torch.long, device=dev)      # internal id of participant p
first = torch.from_numpy(pa_["f"]).to(dev); elig = torch.from_numpy(pa_["is_crawled"].astype(bool) if "is_crawled" in pa_ else np.ones(N, bool)).to(dev)


@torch.no_grad()
def ranks(idx, QB=64):
    """Released fused scorer (fast_predict) over every item; rank of the target among the floors' candidate set."""
    model.eval(); G_adj = model.gcn_forward(model.adj_graph_test); G_sim = model.gcn_forward(model.sim_graph_test)
    W = model.item_embedding.weight; Rk = np.zeros(len(idx)); nc = np.zeros(len(idx))
    for b0 in range(0, len(idx), QB):
        bi = idx[b0:b0 + QB]; S = torch.zeros(len(bi), 50, dtype=torch.long, device=dev); lens = []
        for r, i in enumerate(bi):
            s0 = cstart[qc[i]]; w = pid[s0:s0 + qp[i]][-50:]; S[r, :len(w)] = iid[torch.from_numpy(w).to(dev)]; lens.append(len(w))
        ln = torch.tensor(lens, device=dev)
        so = model.forward(S, ln); last = S[torch.arange(len(bi), device=dev), ln - 1]
        mixed = torch.stack((so, G_adj[last], G_sim[last]), 0)
        sc = F.softmax((torch.matmul(mixed, model.attn_weights.unsqueeze(0)) * model.attn).sum(-1), 0).unsqueeze(-1)
        lg = ((mixed * sc).sum(0) @ W.T)[:, iid]                                                  # [Q, N] in participant order
        for r, i in enumerate(bi):
            s0 = cstart[qc[i]]; j = s0 + qp[i]; y = int(pid[j]); tq = int(ev["t"][j])
            cand = elig & (first < tq); cand[torch.from_numpy(pid[s0:j]).to(dev)] = False; cand[y] = True
            v = lg[r, y]; c = lg[r][cand]; Rk[b0 + r] = 1 + float((c > v).sum()) + 0.5 * (float((c == v).sum()) - 1); nc[b0 + r] = float(cand.sum())
    ref = fl.n_candidates.values[idx]
    assert np.array_equal(nc, ref), f"candidate set differs from floors: {nc[:5]} vs {ref[:5]}"
    return Rk


trainer = get_trainer(config['MODEL_TYPE'], config["model"])(config, model)
t0 = time.time(); hist = []


def valid_epoch(valid_data, show_progress=False):
    """Change (2): released valid_metric (Recall@1) under full ranking on the validation queries; time budget."""
    te = time.time(); r = ranks(val_sel); score = float((r <= 1).mean())
    res = {"recall@1": score, "ndcg@10": float(np.where(r <= 10, 1 / np.log2(r + 1), 0).mean()), "hits@100": float((r <= 100).mean())}
    hist.append(dict(train_hours=(te - t0) / 3600, eval_s=time.time() - te, **res)); log(f"valid {hist[-1]}")
    if (time.time() - t0) / 3600 > BUDGET:              # budget reached: keep this checkpoint if best, then stop
        if score > trainer.best_valid_score: trainer.best_valid_score = score; trainer._save_checkpoint(-1)
        trainer.stopping_step = -1; return -1.0, res
    return score, res


trainer._valid_epoch = valid_epoch
trainer.fit(train_data, test_data, saved=True, show_progress=False)
train_hours = (time.time() - t0) / 3600
model.load_state_dict(torch.load(trainer.saved_model_file, map_location=dev)["state_dict"])
t1 = time.time(); Rk = ranks(np.arange(len(fl))); log(f"scored {len(fl):,} queries in {time.time() - t1:.0f} s")
out = fl[["event_row", "cascade_id", "participant_id", "t", "split", "position", "in_all_sample", "in_novel_sample", "novel", "n_candidates"]].copy()
out["rank_dcrec"] = Rk; out.to_parquet(f"{OUT}/dcrec_seed{SEED}.parquet", index=False)
json.dump(dict(seed=SEED, hist=hist, best_valid_recall1=trainer.best_valid_score, train_hours=train_hours, budget_h=BUDGET), open(f"{OUT}/dcrec_seed{SEED}.json", "w"), indent=1)
log("DONE")
