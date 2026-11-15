"""MIM (TKDE 2025, CZ-TAO12/MIM, official release) on benchmark tables. Released model (MIMIDP), graphs (ConRelationGraph,
ConHypergraph), datasets class, training step (run.model_training), validation (run.model_testing) and EarlyStopping
(patience 6, criterion -sum of validation Hits/MAP@10,50,100); released defaults (run's parser).

    bm-mim/bin/python mim_bench.py BENCH FLOORS.parquet OUTDIR SEED [BUDGET_HOURS=8]
    PROBE=out.json: feasibility probe (probe_util.py), then exit

Data (released text format, written to OUTDIR/mim_root_seed<s>/data/bench/): user index u2idx = {'<blank>': 0, '</s>': 1,
str(p): p + 2}; cascadetrain.txt = TRAIN cascades, cascadevalid.txt = 2,000 validation cascades (released selection),
cascadetest.txt = the same 2,000 (the release reads it; never used for selection here); edges.txt = social edges created
before the validation window. Changes (BENCH_CHANGES.md): (1) hypergraphs from TRAIN cascades only (released
Read_all_cascade uses train + valid); (2) the time budget (the training loader stops mid-epoch when it is spent).
Scoring: released model_prediction on the last <= 199 earlier events of each floors query; logits at the last position;
candidates as floors_eval.py; ties count half; candidate counts must equal the floors' n_candidates.
"""
import json
import os
import pickle
import sys
import time

import duckdb
import numpy as np
import pandas as pd
import torch

B, FL, OUT, SEED = sys.argv[1:5]
SEED = int(SEED); BUDGET = float(sys.argv[5]) if len(sys.argv) > 5 else 8.0
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
ROOT = os.path.abspath(f"{OUT}/mim_root_seed{SEED}"); D = f"{ROOT}/data/bench"; os.makedirs(D, exist_ok=True)
os.makedirs(f"{ROOT}/output_result", exist_ok=True)
log = lambda s: print(f"[mim {time.strftime('%H:%M:%S')}] {s}", flush=True)

con = duckdb.connect(); con.execute("SET threads=32; SET memory_limit='100GB'")
P = f"read_parquet('{B}/participants.parquet')"; cols = [r[0] for r in con.execute(f"DESCRIBE SELECT * FROM {P}").fetchall()]
ev = con.execute(f"""SELECT e.cascade_id, e.participant_id, e.t, e."position", (CASE c.split WHEN 'train' THEN 0 WHEN 'val' THEN 1 ELSE 2 END)::TINYINT split
                     FROM read_parquet('{B}/events.parquet') e JOIN read_parquet('{B}/cascades.parquet') c USING (cascade_id)
                     ORDER BY e.cascade_id, e."position" """).fetchnumpy()
pa_ = con.execute(f"SELECT coalesce(first_seen_t, 9e18)::BIGINT f{', is_crawled' if 'is_crawled' in cols else ''} FROM {P} ORDER BY participant_id").fetchnumpy()
N = len(pa_["f"])
cstart = np.searchsorted(ev["cascade_id"], np.arange(ev["cascade_id"].max() + 2)); csplit = ev["split"][cstart[:-1]]
pid = ev["participant_id"].astype(np.int64); tsec = ev["t"].astype(np.float64)
users = pid + 2
rng = np.random.default_rng(SEED)
tr = [c for c in np.flatnonzero(csplit == 0) if cstart[c + 1] - cstart[c] >= 2]
va = rng.choice([c for c in np.flatnonzero(csplit == 1) if cstart[c + 1] - cstart[c] >= 2], 2000, replace=False)
line = lambda c: " ".join(f"{u},{t}" for u, t in zip(users[cstart[c]:cstart[c + 1]], tsec[cstart[c]:cstart[c + 1]])) + "\n"
for name, cs in (("cascadetrain", tr), ("cascadevalid", va), ("cascadetest", va)):
    with open(f"{D}/{name}.txt", "w") as f: f.writelines(line(c) for c in cs)
u2idx = {"<blank>": 0, "</s>": 1, **{str(p + 2): p + 2 for p in range(N)}}       # MIM reads ids already indexed (users[] = p+2)
pickle.dump(u2idx, open(f"{D}/u2idx.pickle", "wb")); pickle.dump(list(u2idx), open(f"{D}/idx2u.pickle", "wb"))
val_start = int(con.execute(f"SELECT min(t0) FROM read_parquet('{B}/cascades.parquet') WHERE split = 'val'").fetchone()[0])
eg = con.execute(f"SELECT src, dst FROM read_parquet('{B}/edges.parquet') WHERE t_created < {val_start}").fetchnumpy()
with open(f"{D}/edges.txt", "w") as f: f.writelines(f"{u + 2},{v + 2}\n" for u, v in zip(eg["src"], eg["dst"]))
log(f"{N:,} users, {len(tr):,} train cascades, 2,000 validation cascades, {len(eg['src']):,} social edges exported")
# guard: the released ConRelationGraph forms A.dot(A) of the social matrix; its nonzeros are at most the number of two-hop paths
two_hop = int((np.bincount(eg["src"], minlength=N).astype(np.float64) * np.bincount(eg["dst"], minlength=N)).sum())
log(f"two-hop paths in the social graph: {two_hop:,} (upper bound on nnz of the released A.dot(A))")
if os.environ.get("PROBE") and two_hop > 3e9:
    json.dump(dict(method="MIM", bench=B, users=N, social_edges=int(len(eg["src"])), two_hop_paths=two_hop,
                   verdict=f"infeasible: the released relation graph squares the social matrix; up to {two_hop:.2e} nonzeros (> 3e9, i.e. > ~70 GB as float32 COO plus scipy intermediates) before any GPU step"),
              open(os.environ["PROBE"], "w"), indent=1)
    sys.exit("infeasible: two-hop social matrix too large")

os.chdir(ROOT); sys.argv = [sys.argv[0], "-seed", str(SEED), "-save_path", f"{ROOT}/output_result/mim_seed{SEED}.pt"]
if os.environ.get("BATCH"): sys.argv += ["-batch_size", os.environ["BATCH"]]     # largest batch that fits (probe), when the released 48 does not
import run as R  # noqa: E402  (released module: parses the released defaults at import)
import dataLoader  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402
from utils.graphConstruct import ConRelationGraph, ConHypergraph  # noqa: E402
from utils.EarlyStopping import EarlyStopping  # noqa: E402
from utils.util import trans_to_cuda  # noqa: E402
from model.model import MIMIDP  # noqa: E402

opt = R.opt; opt.data_name = "bench"; opt.preprocess = False
_read_all = dataLoader.Read_all_cascade
def train_only(data_name, with_EOS=False):                                  # change (1): hypergraphs from TRAIN cascades only
    user_size, _, _ = _read_all(data_name, with_EOS)
    tr_c, tr_t = dataLoader._readFromFile(dataLoader.Options(data_name).train_data, with_EOS)
    return user_size, tr_c, tr_t
dataLoader.Read_all_cascade = train_only
import utils.graphConstruct as GC; GC.Read_all_cascade = train_only

R.init_seeds(SEED)
train, valid, test, user_size = dataLoader.Read_data("bench")
train_loader = DataLoader(dataset=dataLoader.datasets(train, opt.max_lenth), batch_size=opt.batch_size, shuffle=True, num_workers=8)
val_loader = DataLoader(dataset=dataLoader.datasets(valid, opt.max_lenth), batch_size=opt.batch_size, shuffle=False, num_workers=8)
opt.n_node = user_size
t_g = time.time(); relation_graph = ConRelationGraph("bench"); HG_Item, HG_User = ConHypergraph("bench", opt.n_node, opt.window)
log(f"graphs built in {time.time() - t_g:.0f} s")
model = trans_to_cuda(MIMIDP(graph=relation_graph, hypergraphs=[HG_Item, HG_User], opt=opt, reverse=True, dropout=opt.dropout))
R.model = model; R.opt = opt

if os.environ.get("PROBE"):   # feasibility probe: real released training steps
    sys.path.insert(0, os.path.join(os.environ.get("OPB_ROOT", "."), "benchmark_build")); from probe_util import probe
    import Constants
    _it = {}
    def _step(bs):
        if bs not in _it: _it[bs] = iter(DataLoader(dataset=dataLoader.datasets(train, opt.max_lenth), batch_size=bs, shuffle=True, num_workers=4))
        try: batch = next(_it[bs])
        except StopIteration: _it.pop(bs); return _step(bs)
        cascade_item, label, cascade_time, label_time, _ = batch
        model.zero_grad()
        ci, tar, ct, lt = (trans_to_cuda(x.long()) for x in (cascade_item, label, cascade_time, label_time))
        op, fo, pdist, fdist, pe, fe = model(ci, tar, ct, lt)
        loss = R.get_performance(model.loss_function, op, tar)[0] + opt.beta * R.get_performance(model.loss_function, fo, ci)[0] \
            + opt.beta2 * model.kl_loss(pdist, fdist) + opt.beta3 * model.seq2seqloss(pe, fe, ci)
        loss.backward(); model.optimizer.step(); model.optimizer.update_learning_rate()
    probe(_step, len(tr), int(opt.batch_size), os.environ["PROBE"], method="MIM", bench=B, users=N)


class Budgeted:
    """Training loader stopped mid-epoch once the time budget is spent (change 2)."""
    def __init__(self, dl): self.dl = dl
    def __len__(self): return len(self.dl)
    def __iter__(self):
        for b in self.dl:
            if (time.time() - t0) / 3600 > BUDGET: return
            yield b


early = EarlyStopping(patience=opt.patience, verbose=False, path=opt.save_path)
hist, t0 = [], time.time()
for epoch in range(opt.epoch):
    loss, _ = R.model_training(model, Budgeted(train_loader), epoch)
    vs, _ = R.model_testing(model, val_loader)
    hist.append(dict(epoch=epoch, loss=float(loss), train_hours=(time.time() - t0) / 3600, **{k: float(v) for k, v in vs.items()})); log(f"epoch {hist[-1]}")
    early(-sum(vs.values()), model)
    if early.early_stop or (time.time() - t0) / 3600 > BUDGET: break
train_hours = (time.time() - t0) / 3600
model.load_state_dict(torch.load(opt.save_path)); model.eval()

fl = pd.read_parquet(FL).sort_values("event_row").reset_index(drop=True)
erow = con.execute(f"""SELECT cascade_id, "position" FROM read_parquet('{B}/events.parquet') ORDER BY t, cascade_id, "position" """).fetchnumpy()
qc, qp = erow["cascade_id"][fl.event_row.values], erow["position"][fl.event_row.values]
first = torch.from_numpy(pa_["f"]).cuda(); elig = torch.from_numpy(pa_["is_crawled"].astype(bool) if "is_crawled" in pa_ else np.ones(N, bool)).cuda()
L = opt.max_lenth - 1; Rk = np.zeros(len(fl)); nc = np.zeros(len(fl)); QB = 32; t1 = time.time()
with torch.no_grad():
    for b0 in range(0, len(fl), QB):
        idx = list(range(b0, min(b0 + QB, len(fl)))); T = np.zeros((len(idx), L), np.int64); last = []
        for r, i in enumerate(idx):
            s0 = cstart[qc[i]]; seq = users[s0 + max(0, qp[i] - L):s0 + qp[i]]; T[r, :len(seq)] = seq; last.append(len(seq) - 1)
        lg = model.model_prediction(torch.from_numpy(T).cuda(), None).view(len(idx), L, -1)
        for r, i in enumerate(idx):
            s0 = cstart[qc[i]]; j = s0 + qp[i]; y = int(pid[j]); tq = int(ev["t"][j])
            sc = lg[r, last[r], 2:N + 2]
            cand = elig & (first < tq); cand[torch.from_numpy(pid[s0:j]).cuda()] = False; cand[y] = True
            v = sc[y]; c = sc[cand]; Rk[i] = 1 + float((c > v).sum()) + 0.5 * (float((c == v).sum()) - 1); nc[i] = float(cand.sum())
assert np.array_equal(nc, fl.n_candidates.values), "candidate set differs from floors"
log(f"scored {len(fl):,} queries in {time.time() - t1:.0f} s")
out = fl[["event_row", "cascade_id", "participant_id", "t", "split", "position", "in_all_sample", "in_novel_sample", "novel", "n_candidates"]].copy()
out["rank_mim"] = Rk; out.to_parquet(f"{OUT}/mim_seed{SEED}.parquet", index=False)
json.dump(dict(seed=SEED, hist=hist, train_hours=train_hours, budget_h=BUDGET, opt={k: str(v) for k, v in vars(opt).items()}), open(f"{OUT}/mim_seed{SEED}.json", "w"), indent=1)
log("DONE")
