"""PMRCA (SIGIR 2025, HeyWeCome/PMRCA, official release) on benchmark tables. Released model, BaseLoader (user index,
length filter 1..500, time-ordered split), DataLoader, BaseRunner.train_epoch (loss incl. SSL and prototype terms; faiss
k-means e_step before every batch, as released) and BaseRunner.test_epoch (validation metrics); released defaults and
patience 10 on the sum of validation Hits/MAP@10,20,50,100.

    model-env/bin/python pmrca_bench.py BENCH FLOORS.parquet OUTDIR SEED [BUDGET_HOURS=8]   (PYTHONPATH=pydeps_pyg)
    PROBE=out.json: feasibility probe (probe_util.py), then exit

Data (released format, OUTDIR/pmrca_root_seed<s>/data/bench/): cascades.txt = TRAIN cascades then 2,000 validation
cascades ('user ts,user ts,...'); u2idx {'<blank>': 0, '</s>': 1, str(p): p + 2}. The released split sorts cascades by
timestamps and cuts by ratio; train cascades all precede validation cascades, so train_rate = n_train / total gives exactly
train / validation (test empty: the benchmark never uses it). The model's cascade-user adjacency is built from the train
split only (released). Changes (BENCH_CHANGES.md): (1) the time budget (the training loader stops mid-epoch when spent).
Scoring: released forward in eval mode on the last <= 199 earlier events of each floors query (padded by one slot: forward
drops the last input position); logits after the last real event; candidates as floors_eval.py; ties count half.
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
import torch.nn as nn

B, FL, OUT, SEED = sys.argv[1:5]
SEED = int(SEED); BUDGET = float(sys.argv[5]) if len(sys.argv) > 5 else 8.0
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
ROOT = os.path.abspath(f"{OUT}/pmrca_root_seed{SEED}"); D = f"{ROOT}/data/bench"; os.makedirs(D, exist_ok=True)
log = lambda s: print(f"[pmrca {time.strftime('%H:%M:%S')}] {s}", flush=True)

con = duckdb.connect(); con.execute("SET threads=32; SET memory_limit='100GB'")
P = f"read_parquet('{B}/participants.parquet')"; cols = [r[0] for r in con.execute(f"DESCRIBE SELECT * FROM {P}").fetchall()]
ev = con.execute(f"""SELECT e.cascade_id, e.participant_id, e.t, e."position", (CASE c.split WHEN 'train' THEN 0 WHEN 'val' THEN 1 ELSE 2 END)::TINYINT split
                     FROM read_parquet('{B}/events.parquet') e JOIN read_parquet('{B}/cascades.parquet') c USING (cascade_id)
                     ORDER BY e.cascade_id, e."position" """).fetchnumpy()
pa_ = con.execute(f"SELECT coalesce(first_seen_t, 9e18)::BIGINT f{', is_crawled' if 'is_crawled' in cols else ''} FROM {P} ORDER BY participant_id").fetchnumpy()
N = len(pa_["f"])
cstart = np.searchsorted(ev["cascade_id"], np.arange(ev["cascade_id"].max() + 2)); csplit = ev["split"][cstart[:-1]]
pid = ev["participant_id"].astype(np.int64); tsec = ev["t"].astype(np.float64)
rng = np.random.default_rng(SEED)
ok = lambda c: 2 <= cstart[c + 1] - cstart[c] <= 500                       # released length filter (filter_num 1 .. 500), >= 2 to have a target
tr = [c for c in np.flatnonzero(csplit == 0) if ok(c)]
va = rng.choice([c for c in np.flatnonzero(csplit == 1) if ok(c)], 2000, replace=False)
with open(f"{D}/cascades.txt", "w") as f:
    for c in list(tr) + list(va):
        f.write(",".join(f"{p} {t}" for p, t in zip(pid[cstart[c]:cstart[c + 1]], tsec[cstart[c]:cstart[c + 1]])) + "\n")
u2idx = {"<blank>": 0, "</s>": 1, **{str(p): p + 2 for p in range(N)}}
pickle.dump(u2idx, open(f"{D}/u2idx.pickle", "wb")); pickle.dump(list(u2idx), open(f"{D}/idx2u.pickle", "wb"))
open(f"{D}/edges.txt", "w").close()                                          # the released model does not read the social network
log(f"{N:,} users, {len(tr):,} train + 2,000 validation cascades exported")

os.chdir(ROOT); os.makedirs("saved", exist_ok=True)
import importlib  # noqa: E402
import argparse  # noqa: E402
run = importlib.import_module("run")
from helpers.BaseLoader import BaseLoader, DataLoader  # noqa: E402
from helpers.BaseRunner import BaseRunner  # noqa: E402
from utils.Optim import ScheduledOptim  # noqa: E402
from utils import Constants, Utils  # noqa: E402
PM = importlib.import_module("models.PMRCA").PMRCA
parser = PM.parse_model_args(run.parse_global_args(argparse.ArgumentParser()))
args, _ = parser.parse_known_args(["-data_name", "bench", "-gpu", os.environ.get("CUDA_VISIBLE_DEVICES", "0")]
                                  + (["-batch_size", os.environ["BATCH"]] if os.environ.get("BATCH") else []))   # largest batch that fits (probe)
args.device = torch.device("cuda"); args.model_path = f"{ROOT}/saved/pmrca_seed{SEED}.pt"
Utils.init_seed(SEED)
torch.manual_seed(SEED); np.random.seed(SEED)
total = len(tr) + 2000
loader = BaseLoader(args)
user_size, total_cascades, timestamps, train, valid, test = loader.split_data((len(tr) + 0.5) / total, (2000 + 0.0) / total, load_dict=True)
assert len(train[0]) == len(tr) and len(valid[0]) == 2000, (len(train[0]), len(valid[0]))
train_data = DataLoader(train, batch_size=args.batch_size, load_dict=True, cuda=False)
valid_data = DataLoader(valid, batch_size=args.batch_size, load_dict=True, cuda=False)
model = PM(args, loader).to(args.device)
runner = BaseRunner(model)
loss_func = nn.CrossEntropyLoss(reduction="sum", ignore_index=Constants.PAD).to(args.device)
optimizer = ScheduledOptim(torch.optim.Adam(model.parameters(), betas=(0.9, 0.98), eps=1e-09), args.d_model, args.n_warmup_steps)
log(f"model built: {user_size:,} users, {loader.cas_num:,} cascades; args {vars(args)}")

if os.environ.get("PROBE"):
    sys.path.insert(0, os.path.join(os.environ.get("OPB_ROOT", "."), "benchmark_build")); from probe_util import probe
    _it = {}
    def _step(bs):
        if bs not in _it: _it[bs] = iter(DataLoader(train, batch_size=bs, load_dict=True, cuda=False))
        try: batch = next(_it[bs])
        except StopIteration: _it.pop(bs); return _step(bs)
        hs, ht, hi = (x.cuda() for x in batch); model.train(); optimizer.zero_grad(); model.before_epoch()
        loss, _ = model.get_performance(hs, ht, hi, loss_func, hs[:, 1:]); loss.backward(); optimizer.step(); optimizer.update_learning_rate()
    probe(_step, len(tr), int(args.batch_size), os.environ["PROBE"], method="PMRCA", bench=B, users=N)


class Budgeted:
    """Training loader stopped mid-epoch once the time budget is spent (change 1)."""
    def __init__(self, dl): self.dl = dl
    def __len__(self): return len(self.dl)
    def __iter__(self):
        for b in self.dl:
            if (time.time() - t0) / 3600 > BUDGET: return
            yield b


best, bad, hist, t0 = -1.0, 0, [], time.time()
if os.environ.get("MAX_EPOCHS"): args.epoch = int(os.environ["MAX_EPOCHS"])   # one full epoch at scale (paper, Section 4.4)
if os.environ.get("SCORE_ONLY"):   # change 5: score a saved checkpoint; the training history is read back from that run's log
    import ast
    hist = [ast.literal_eval(l.split("] epoch ", 1)[1]) for l in open(os.environ["SCORE_ONLY"], errors="replace") if "] epoch {" in l]
    args.epoch = 0; log(f"score only: checkpoint {args.model_path}, {len(hist)} epoch(s) from the log")
for epoch in range(args.epoch):
    tl, ta = runner.train_epoch(model, Budgeted(train_data), loss_func, optimizer)
    vs = runner.test_epoch(model, valid_data); s = float(sum(vs.values()))
    torch.cuda.empty_cache()   # change 4: free the validation pass's cached blocks so the next batch's faiss GPU k-means can allocate
    hist.append(dict(epoch=epoch, loss=float(tl), train_hours=(time.time() - t0) / 3600, **{k: float(v) for k, v in vs.items()})); log(f"epoch {hist[-1]}")
    if s > best: best, bad = s, 0; torch.save(model.state_dict(), args.model_path)
    else: bad += 1
    if bad >= runner.patience or (time.time() - t0) / 3600 > BUDGET: break
train_hours = hist[-1]["train_hours"] if os.environ.get("SCORE_ONLY") else (time.time() - t0) / 3600
model.load_state_dict(torch.load(args.model_path)); model.eval()

fl = pd.read_parquet(FL).sort_values("event_row").reset_index(drop=True)
erow = con.execute(f"""SELECT cascade_id, "position" FROM read_parquet('{B}/events.parquet') ORDER BY t, cascade_id, "position" """).fetchnumpy()
qc, qp = erow["cascade_id"][fl.event_row.values], erow["position"][fl.event_row.values]
first = torch.from_numpy(pa_["f"]).cuda(); elig = torch.from_numpy(pa_["is_crawled"].astype(bool) if "is_crawled" in pa_ else np.ones(N, bool)).cuda()
L = 199; Rk = np.zeros(len(fl)); nc = np.zeros(len(fl)); QB = int(os.environ.get("SCORE_QB", 32)); t1 = time.time()
with torch.no_grad():
    for b0 in range(0, len(fl), QB):
        idx = list(range(b0, min(b0 + QB, len(fl)))); T = np.zeros((len(idx), L + 1), np.int64); last = []
        for r, i in enumerate(idx):
            s0 = cstart[qc[i]]; seq = pid[s0 + max(0, qp[i] - L):s0 + qp[i]] + 2; T[r, :len(seq)] = seq; last.append(len(seq) - 1)
        Tt = torch.from_numpy(T).cuda()
        lg = model(Tt, torch.zeros_like(Tt).float(), torch.zeros(len(idx), dtype=torch.long).cuda()).view(len(idx), L, -1)
        for r, i in enumerate(idx):
            s0 = cstart[qc[i]]; j = s0 + qp[i]; y = int(pid[j]); tq = int(ev["t"][j])
            sc = lg[r, last[r], 2:N + 2]
            cand = elig & (first < tq); cand[torch.from_numpy(pid[s0:j]).cuda()] = False; cand[y] = True
            v = sc[y]; c = sc[cand]; Rk[i] = 1 + float((c > v).sum()) + 0.5 * (float((c == v).sum()) - 1); nc[i] = float(cand.sum())
assert np.array_equal(nc, fl.n_candidates.values), "candidate set differs from floors"
log(f"scored {len(fl):,} queries in {time.time() - t1:.0f} s")
out = fl[["event_row", "cascade_id", "participant_id", "t", "split", "position", "in_all_sample", "in_novel_sample", "novel", "n_candidates"]].copy()
out["rank_pmrca"] = Rk; out.to_parquet(f"{OUT}/pmrca_seed{SEED}.parquet", index=False)
json.dump(dict(seed=SEED, hist=hist, train_hours=train_hours, budget_h=BUDGET, args={k: str(v) for k, v in vars(args).items()}), open(f"{OUT}/pmrca_seed{SEED}.json", "w"), indent=1)
log("DONE")
