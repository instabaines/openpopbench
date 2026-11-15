"""HyperEvent (official release, jianjianGJ/HyperEvent) on benchmark tables. Released model, feature construction
(get_x_y), adjacency tables, segment-parallel training loop, BCE loss and validation-MRR model selection.

    model-env/bin/python hyperevent_bench.py BENCH FLOORS.parquet OUTDIR SEED [BUDGET_HOURS=8]

Temporal graph (protocol F2): participant p is node p, cascade c is node N+c; every event (origin included) is an edge
cascade -> participant, in the floors' event order (t, cascade_id, position). Train edges = train-split events (all precede
the validation window); validation edges = validation-window events; the rest is the test region.
Configuration: config_tgb.yaml 'default' + the block of the TGB dataset nearest in edge count (log scale; fixed before
any result: Bluesky S -> tgbl-review, Telegram -> tgbl-coin); one run per seed (released: num_runs runs, seeds 0..4).
Changes (BENCH_CHANGES.md): (1) validation negatives: TGB ships none for this data; load_ns_eval uses 20 negatives for a
dataset it does not know, drawn here uniformly from participants (excluding the positive), fixed seed 0; (2) the time
budget; (3) scoring, below. FLOORS should be the query subsample (paper, Section 3.4; queries_sub10.parquet).
Scoring: the adjacency table is streamed in event order with the released update rule (precalculate_adjtable_for_segments:
shift left, append) and checked equal to the released snapshot at the start of the validation region. At each floors
query (event row r: state after events < r, the released streaming semantics) get_x_y builds the released features for
(cascade, v) for every candidate v, and the released model scores them; the two table rows get_x_y updates in place are
restored after each chunk. Candidates as floors_eval.py; ties count half; candidate counts must equal the floors'.
"""
import json
import os
import sys
import time

import duckdb
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import yaml
import zlib

B, FL, OUT, SEED = sys.argv[1:5]
SEED = int(SEED); BUDGET = float(sys.argv[5]) if len(sys.argv) > 5 else 8.0
HERE = os.path.dirname(os.path.abspath(__file__)); os.chdir(HERE); sys.path.insert(0, HERE); sys.argv = [sys.argv[0]]
from utils_data_tgb import get_segment, load_ns_eval, get_batch, get_x_y  # noqa: E402
from model import SequenceClassifier  # noqa: E402
from train_evaluation_tgb import train_one_batch, evaluate  # noqa: E402
from tgb.linkproppred.evaluate import Evaluator  # noqa: E402

TGB_EDGES = {"tgbl-wiki": 157_474, "tgbl-review": 4_873_540, "tgbl-coin": 22_809_486, "tgbl-comment": 44_314_507, "tgbl-flight": 67_169_570}
E_bench = duckdb.connect().execute(f"SELECT count(*) FROM read_parquet('{B}/events.parquet')").fetchone()[0]
TGB_NAME = os.environ.get("TGB_CONFIG") or min(TGB_EDGES, key=lambda k: abs(np.log(TGB_EDGES[k] / E_bench)))   # TGB_CONFIG: tuning check (2026-11-13); unset = nearest rule   # config rule: nearest TGB dataset by edge count (log scale)
cfg = yaml.safe_load(open("config_tgb.yaml")); params = {**cfg["default"], **cfg[TGB_NAME]}
os.makedirs(OUT, exist_ok=True); dev = torch.device("cuda")
log = lambda s: print(f"[hyperevent {time.strftime('%H:%M:%S')}] {s}", flush=True)
con = duckdb.connect(); con.execute("SET threads=32; SET memory_limit='100GB'")
P = f"read_parquet('{B}/participants.parquet')"; cols = [r[0] for r in con.execute(f"DESCRIBE SELECT * FROM {P}").fetchall()]
ev = con.execute(f"""SELECT e.cascade_id, e.participant_id, e.t, (CASE c.split WHEN 'train' THEN 0 WHEN 'val' THEN 1 ELSE 2 END)::TINYINT split
                     FROM read_parquet('{B}/events.parquet') e JOIN read_parquet('{B}/cascades.parquet') c USING (cascade_id)
                     ORDER BY e.t, e.cascade_id, e."position" """).fetchnumpy()
pa_ = con.execute(f"SELECT coalesce(first_seen_t, 9e18)::BIGINT f{', is_crawled' if 'is_crawled' in cols else ''} FROM {P} ORDER BY participant_id").fetchnumpy()
N = len(pa_["f"]); C = int(ev["cascade_id"].max()) + 1; E = len(ev["t"]); NN = N + C
src = torch.from_numpy((N + ev["cascade_id"]).astype(np.int64)); dst = torch.from_numpy(ev["participant_id"].astype(np.int64)); ts = torch.from_numpy(ev["t"].astype(np.int64))
val_start = int(con.execute(f"SELECT min(t0) FROM read_parquet('{B}/cascades.parquet') WHERE split = 'val'").fetchone()[0])
test_start = int(con.execute(f"SELECT min(t0) FROM read_parquet('{B}/cascades.parquet') WHERE split = 'test'").fetchone()[0])
trm = torch.from_numpy(ev["split"] == 0); vam = (ts >= val_start) & (ts < test_start)
assert int(torch.nonzero(trm).max()) < int(torch.nonzero(vam).min()), "train edges must precede the validation edges"


class RandomNS:
    """20 uniform participant negatives per validation edge (load_ns_eval's count for an unknown dataset), fixed seed."""
    def __init__(self): self.rng = np.random.default_rng(0)
    def query_batch(self, s, d, t, split_mode):
        out = []
        for di in d.tolist():
            x = self.rng.choice(N - 1, 20, replace=False); x[x >= di] += 1; out.append(x.tolist())
        return out


class BenchDataset:
    def __init__(self):
        self.name = "bench_" + os.path.basename(os.path.normpath(B)) + f"_{zlib.crc32(os.path.abspath(B).encode())}"
        self.src, self.dst, self.ts, self.num_edges = src, dst, ts, E
        self.train_mask, self.val_mask = trm, vam; self.negative_sampler = RandomNS()
    def load_val_ns(self): self.negative_sampler.rng = np.random.default_rng(0)    # same negatives every validation pass


dataset = BenchDataset()
log(f"config block {TGB_NAME} (nearest TGB dataset by edge count to {E_bench:,})")
log(f"{N:,} participants, {C:,} cascades, {E:,} edges (train {int(trm.sum()):,}, val {int(vam.sum()):,}); params {params}")
torch.manual_seed(SEED); evaluator = Evaluator(name="tgbl-wiki")          # evaluator: MRR over one positive and the negatives
input_dim = 12 if params["big_hood"] else 4
model = SequenceClassifier(input_dim=input_dim, d_model=params["d_model"], nhead=params["nhead"], num_layers=params["num_layers"],
                           dim_feedforward=params["dim_feedforward"], dropout=params["dropout"]).to(dev)
opt = torch.optim.Adam(model.parameters(), lr=params["learning_rate"]); criterion = nn.BCELoss()
ck = f"{OUT}/hyperevent_seed{SEED}.pth"
segs = (params["num_segment_train"], params["num_segment_val"], params["num_segment_test"], params["num_neighbor"])
need_gb = params["num_segment_train"] * (NN + 1) * params["num_neighbor"] * 8 / 2 ** 30   # released train adjacency table (int64), moved to the GPU
if need_gb > 40: sys.exit(f"infeasible: the released training adjacency table needs {need_gb:,.0f} GiB on the GPU (44 GiB A40); not precomputed")
t_a = time.time()
s_tr_c, d_tr_c, t_tr_c, a_tr_c = get_segment(dataset, "train", *segs)
s_va_c, d_va_c, t_va_c, a_va_c = get_segment(dataset, "val", *segs)
log(f"released adjacency tables ready in {time.time() - t_a:.0f} s")
s_tr, d_tr, t_tr, a_tr = s_tr_c.to(dev), d_tr_c.to(dev), t_tr_c.to(dev), a_tr_c.to(dev)
s_va, d_va, t_va, a_va = s_va_c.to(dev), d_va_c.to(dev), t_va_c.to(dev), a_va_c.to(dev)
min_dst_idx, max_dst_idx = int(dst.min()), int(dst.max())
best, hist, t0 = 0.0, [], time.time()
SCORE_ONLY = os.environ.get("SCORE_ONLY") == "1"   # change (4), 2026-10-30: score the saved best checkpoint, no training
for epoch in range(0 if SCORE_ONLY else params["num_epochs"]):
    s_tr[:], d_tr[:], t_tr[:], a_tr[:] = s_tr_c, d_tr_c, t_tr_c, a_tr_c; losses = []
    for i in range(s_tr.size(1)):
        src_i, dst_i, ts_i = get_batch(s_tr, d_tr, t_tr, i)
        neg = torch.randint(min_dst_idx, max_dst_idx, (src_i.size(0), 1), device=dev)
        x, y = get_x_y(a_tr, src_i, dst_i, torch.hstack([dst_i, neg]), params["num_latest"], params["big_hood"])
        losses.append(train_one_batch(model, x, y, criterion, opt))
        if (time.time() - t0) / 3600 > BUDGET: break
    dataset.load_val_ns(); s_va[:], d_va[:], t_va[:], a_va[:] = s_va_c, d_va_c, t_va_c, a_va_c; outs = []
    for i in range(s_va.size(1)):
        src_i, dst_i, ts_i = get_batch(s_va, d_va, t_va, i)
        q = torch.hstack([dst_i, load_ns_eval(dataset, "val", src_i, dst_i, ts_i)])
        x, y = get_x_y(a_va, src_i, dst_i, q, params["num_latest"], params["big_hood"])
        outs.append(evaluate(model, x, y, criterion)[1].view(-1, q.size(1)).cpu())
    outs = np.vstack(outs); mrr = float(evaluator.eval({"y_pred_pos": outs[:, 0], "y_pred_neg": outs[:, 1:], "eval_metric": ["mrr"]})["mrr"])
    hist.append(dict(epoch=epoch, train_loss=float(np.mean(losses)), val_mrr=mrr, train_hours=(time.time() - t0) / 3600)); log(f"epoch {hist[-1]}")
    if mrr > best: best = mrr; torch.save(model.state_dict(), ck)
    if (time.time() - t0) / 3600 > BUDGET: break
train_hours = float(os.environ.get("TRAIN_HOURS", "nan")) if SCORE_ONLY else (time.time() - t0) / 3600
if SCORE_ONLY: hist = json.load(open(f"{OUT}/hyperevent_seed{SEED}.json")).get("hist", []) + [dict(phase="score_only")]
del s_tr, d_tr, t_tr, a_tr, s_va, d_va, t_va, a_va; torch.cuda.empty_cache()
model.load_state_dict(torch.load(ck)); model.eval()
# change (5), 2026-10-30: rank by the classifier logit (released head minus its final Sigmoid). Same order; float32
# sigmoid could round large logits to 1.0. Off by default: the 2026-10-30 probe found no saturation (see BENCH_CHANGES).
if os.environ.get("LOGIT", "0") == "1": model.classifier[-1] = nn.Identity(); log("change (5): ranking by logit")

# ---- scoring: stream the adjacency table (released update rule), checked against the released validation snapshot
fl = pd.read_parquet(FL).sort_values("event_row").reset_index(drop=True)
if os.environ.get("EVAL_PROBE"):                         # throughput probe: 200 evenly spaced test queries
    te = np.flatnonzero(fl.split.values == "test"); fl = fl.iloc[np.sort(te[np.linspace(0, len(te) - 1, 200).astype(int)])].reset_index(drop=True)
rows = fl.event_row.values.astype(np.int64); Kn = params["num_neighbor"]
A = np.full((NN + 1, Kn), -1, np.int64); S_, D_ = src.numpy(), dst.numpy()
first_val = int(torch.nonzero(trm).max()) + 1                    # released snapshot point: id_train_last + 1
first = torch.from_numpy(pa_["f"]).to(dev); elig = torch.from_numpy(pa_["is_crawled"].astype(bool) if "is_crawled" in pa_ else np.ones(N, bool)).to(dev)
prefix = pd.DataFrame({"c": ev["cascade_id"], "p": ev["participant_id"], "r": np.arange(E)}).groupby("c")
Rk = np.zeros(len(rows)); nc = np.zeros(len(rows)); ptr = 0; CH = int(os.environ.get("SCORE_CHUNK", 16384)); t1 = time.time()


def advance(stop):
    global ptr
    while ptr < stop:
        s, d = S_[ptr], D_[ptr]
        A[s, :-1] = A[s, 1:]; A[s, -1] = d; A[d, :-1] = A[d, 1:]; A[d, -1] = s; ptr += 1


advance(first_val)
assert np.array_equal(A, a_va_c[0].numpy()), "streamed adjacency table differs from the released snapshot"
log("streamed adjacency table equals the released validation snapshot")
tie_stats = []
with torch.no_grad():
    for k, r in enumerate(rows):
        advance(int(r)); At = torch.from_numpy(A).to(dev)[None]
        c = int(ev["cascade_id"][r]); y = int(ev["participant_id"][r]); tq = int(ev["t"][r]); s_node = N + c
        g = prefix.get_group(c); earlier = g.p.values[g.r.values < r]
        cand = elig & (first < tq); cand[torch.from_numpy(earlier.astype(np.int64)).to(dev)] = False; cand[y] = True
        others = torch.nonzero(cand).squeeze(1); others = others[others != y]; nc[k] = 1 + len(others)
        src_i = torch.tensor([[s_node]], device=dev); dst_i = torch.tensor([[y]], device=dev)
        keep_s, keep_y = At[0, s_node].clone(), At[0, y].clone(); sc = []
        for a in range(0, len(others), CH - 1):
            q = torch.cat([dst_i[0], others[a:a + CH - 1]])[None]
            x, _ = get_x_y(At, src_i, dst_i, q, params["num_latest"], params["big_hood"])
            sc.append(model(x).squeeze(-1)); At[0, s_node], At[0, y] = keep_s, keep_y       # undo get_x_y's in-place update
        v = sc[0][0]; allsc = torch.cat([sc[0][1:]] + [z[1:] for z in sc[1:]])
        Rk[k] = 1 + float((allsc > v).sum()) + 0.5 * float((allsc == v).sum())
        if os.environ.get("EVAL_PROBE"):
            sg = torch.sigmoid(allsc.float()) if os.environ.get("LOGIT", "0") == "1" else allsc; vs = sg.new_tensor(float(torch.sigmoid(v.float()))) if os.environ.get("LOGIT", "0") == "1" else v
            tie_stats.append((float((allsc == v).sum()), float((sg == vs).sum()), float((sg == 1.0).sum()), len(allsc)))
        if k % 1000 == 0: log(f"scored {k:,}/{len(rows):,} ({time.time() - t1:.0f} s)")
assert np.array_equal(nc, fl.n_candidates.values), "candidate set differs from floors"
log(f"scored {len(rows):,} queries ({nc.sum():.3e} pairs) in {time.time() - t1:.0f} s")
if os.environ.get("EVAL_PROBE"):
    full = pd.read_parquet(FL, columns=["n_candidates"]).n_candidates.sum(); sec = time.time() - t1
    res = dict(method="HyperEvent", bench=B, floors=FL, epoch_hours=train_hours, probe_queries=len(rows), probe_pairs=float(nc.sum()), probe_sec=sec,
               pairs_per_sec=nc.sum() / sec, total_pairs=float(full), est_scoring_hours=full / (nc.sum() / sec) / 3600, peak_mem_gb=torch.cuda.max_memory_allocated() / 2 ** 30)
    ts = np.array(tie_stats)   # per query: candidates tied with the target (logit), tied with it after sigmoid, candidates at sigmoid == 1.0, candidates
    res.update(queries_with_tie_logit=float((ts[:, 0] > 0).mean()), queries_with_tie_sigmoid=float((ts[:, 1] > 0).mean()), mean_frac_at_sigmoid_1=float((ts[:, 2] / ts[:, 3]).mean()))
    json.dump(res, open(os.environ["EVAL_PROBE"], "w"), indent=1); log(f"[probe] {res}"); sys.exit(0)
out = fl[["event_row", "cascade_id", "participant_id", "t", "split", "position", "in_all_sample", "in_novel_sample", "novel", "n_candidates"]].copy()
out["rank_hyperevent"] = Rk; out.to_parquet(f"{OUT}/hyperevent_seed{SEED}.parquet", index=False)
json.dump(dict(seed=SEED, hist=hist, best_val_mrr=best, train_hours=train_hours, scoring_hours=(time.time() - t1) / 3600, budget_h=BUDGET, params=params),
          open(f"{OUT}/hyperevent_seed{SEED}.json", "w"), indent=1)
log("DONE")
