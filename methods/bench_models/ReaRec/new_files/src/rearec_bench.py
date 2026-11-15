"""ReaRec (PRL, ID-based SASRec backbone; TangJiakai/ReaRec @ 7f46432 with the two documented fixes) on benchmark tables.
Released BaseReader, PRL model and Dataset, BaseRunner (training, full-ranking validation on the released main_metric
NDCG@20, early stop 10, best-model saving); released defaults except the G-repro-selected PRL temperatures
(--temperature 2.0, --temp_scale 1.0: the validation-selected configuration of the approved retry).

    model-env/bin/python rearec_bench.py BENCH FLOORS.parquet OUTDIR SEED [BUDGET_HOURS=8]

Data (released remap-CSV format, OUTDIR/rearec_data_seed<s>/bench/): one row per prediction instance, as in the released
Amazon files: user_id = cascade index, item_id = target participant p, history_id = earlier participants (the reader keeps
the last 50). train = every non-origin event of the TRAIN cascades; valid = every non-origin event of 2,000 validation
cascades (released selection); test = the same rows (read by the release, never used for selection).
Changes (BENCH_CHANGES.md): (1) the time budget: once spent, the next fit() returns NaN, which the released loop treats
as a stop ("Loss is Nan. Stop training") before reloading the best saved model.
Scoring: the released forward (stage 'infer') on the last <= 50 earlier events of each floors query; prediction logits
over the item table; candidates as floors_eval.py; ties count half; candidate counts must equal the floors' n_candidates.
"""
import argparse
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
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
DP = os.path.abspath(f"{OUT}/rearec_data_seed{SEED}"); D = f"{DP}/bench"; os.makedirs(D, exist_ok=True)
log = lambda s: print(f"[rearec {time.strftime('%H:%M:%S')}] {s}", flush=True)

con = duckdb.connect(); con.execute("SET threads=32; SET memory_limit='100GB'")
P = f"read_parquet('{B}/participants.parquet')"; cols = [r[0] for r in con.execute(f"DESCRIBE SELECT * FROM {P}").fetchall()]
ev = con.execute(f"""SELECT e.cascade_id, e.participant_id, e.t, e."position", (CASE c.split WHEN 'train' THEN 0 WHEN 'val' THEN 1 ELSE 2 END)::TINYINT split
                     FROM read_parquet('{B}/events.parquet') e JOIN read_parquet('{B}/cascades.parquet') c USING (cascade_id)
                     ORDER BY e.cascade_id, e."position" """).fetchnumpy()
pa_ = con.execute(f"SELECT coalesce(first_seen_t, 9e18)::BIGINT f{', is_crawled' if 'is_crawled' in cols else ''} FROM {P} ORDER BY participant_id").fetchnumpy()
N = len(pa_["f"])
cstart = np.searchsorted(ev["cascade_id"], np.arange(ev["cascade_id"].max() + 2)); csplit = ev["split"][cstart[:-1]]
pid = ev["participant_id"].astype(np.int64)
rng = np.random.default_rng(SEED)
tr = [c for c in np.flatnonzero(csplit == 0) if cstart[c + 1] - cstart[c] >= 2]
va = rng.choice([c for c in np.flatnonzero(csplit == 1) if cstart[c + 1] - cstart[c] >= 2], 2000, replace=False)


def rows(cs, f):
    f.write("user_id,item_id,history_id\n")
    for c in cs:
        x = pid[cstart[c]:cstart[c + 1]]
        for j in range(1, len(x)): f.write(f"{c},{x[j]},{' '.join(map(str, x[max(0, j - 50):j]))}\n")


for name, cs in (("train", tr), ("valid", va), ("test", va)):
    with open(f"{D}/bench.{name}.remap.csv", "w") as f: rows(cs, f)
# the reader sizes the item table from the largest id seen; make the last participant appear (as a history item of one extra row)
with open(f"{D}/bench.train.remap.csv", "a") as f: f.write(f"{tr[0]},{pid[cstart[tr[0]] + 1]},{N - 1}\n")
log(f"{N:,} items, {len(tr):,} train cascades, 2,000 validation cascades exported")

from helpers.BaseReader import BaseReader  # noqa: E402
from helpers.BaseRunner import BaseRunner  # noqa: E402
from models.PRL import PRL  # noqa: E402
from utils import utils  # noqa: E402
from utils.constants import ITEM_SEQ, ITEM_SEQ_LEN, ITEM_ID, MAX_ITEM_SEQ_LEN  # noqa: E402
import main as M  # noqa: E402

parser = PRL.parse_model_args(BaseRunner.parse_runner_args(BaseReader.parse_data_args(M.parse_global_args(argparse.ArgumentParser()))))
args, _ = parser.parse_known_args(["--dataset", "bench", "--path", DP, "--random_seed", str(SEED), "--temperature", os.environ.get("TAU", "2.0"), "--temp_scale", os.environ.get("ALPHA", "1.0"),   # TAU/ALPHA: tuning check (2026-11-13); defaults = benchmark config
                                  
                                   "--gpu", os.environ.get("CUDA_VISIBLE_DEVICES", "0")])
if os.environ.get("MAX_EPOCHS"): args.epoch = int(os.environ["MAX_EPOCHS"])   # one full epoch at scale (paper, Section 4.4)
if os.environ.get("BATCH"): args.batch_size = int(os.environ["BATCH"])
if os.environ.get("EVAL_BATCH"): args.eval_batch_size = int(os.environ["EVAL_BATCH"])  # validation batch: memory only, metrics unchanged          # largest batch that fits (probe), when the released one does not
args.device = torch.device("cuda"); args.log_file = f"{OUT}/rearec_log_seed{SEED}/run.log"; args.model_path = f"{OUT}/rearec_ckpt_seed{SEED}/PRL.pt"
args.model_name = "PRL"
for p in (args.log_file, args.model_path): os.makedirs(os.path.dirname(p), exist_ok=True)
import logging as _lg   # 2026-10-30: the released runner logs each validation via logging; attach the log file so the history is kept
_lg.basicConfig(filename=args.log_file, level=_lg.INFO, format="%(asctime)s %(message)s", force=True)
utils.init_seed(args.random_seed)
_evl = utils.eval_list_columns
def _int_lists(df):
    """The released reader parses history lists as floats; the item count then becomes a float whenever the largest id occurs
    only in a history (seen from M upward). Same values, integer dtype (the reader casts the lists to long tensors anyway)."""
    df = _evl(df)
    for c in df.columns:
        if df[c].dtype == object and len(df) and isinstance(df[c].iloc[0], np.ndarray): df[c] = df[c].apply(lambda x: x.astype(np.int64))
    return df
utils.eval_list_columns = _int_lists
import helpers.BaseReader as _BR; _BR.utils.eval_list_columns = _int_lists
corpus = BaseReader(args)
assert corpus.n_items - 1 >= N, f"item table {corpus.n_items} smaller than the population {N}"
model = PRL(args, corpus).to(args.device)
data = {ph: PRL.Dataset(model, corpus, ph) for ph in ("train", "valid", "test")}
for ph in data: data[ph].prepare()
runner = BaseRunner(args)


def _chunked_evaluate(dataset, topks, metrics):
    """Change (2), 2026-10-25: the released predict() stacks the full [validation rows x items] score matrix on the GPU
    (26 GiB on Bluesky M). Same predictions, same evaluate_method, applied per eval batch; every metric is a mean over
    rows, so the row-weighted average of the per-batch results equals the released value exactly."""
    from torch.utils.data import DataLoader as _DL
    import helpers.BaseRunner as _BRm
    dataset.model.eval(); tot, n = {}, 0
    for batch in _DL(dataset, batch_size=runner.eval_batch_size, shuffle=False, num_workers=runner.num_workers, collate_fn=dataset.collate_batch):
        with torch.no_grad():
            pred = dataset.model(utils.batch_to_gpu(batch, dataset.model.device), stage="infer")["prediction"].detach()
        pred[:, -1] = -torch.inf
        _, top = torch.topk(pred, k=max(topks), dim=1)                     # as evaluate_method, per batch, unrounded sums
        pos = top.cpu().numpy() == batch[ITEM_ID].cpu().numpy().reshape(-1, 1)        # batch_to_gpu moves the batch in place
        for mt in metrics:
            for k, r in zip(topks, getattr(_BRm, mt)(pos, topks)): tot[f"{mt}@{k}"] = tot.get(f"{mt}@{k}", 0.0) + float(r.sum())
        n += pos.shape[0]
    return {k: round(v / n, 4) for k, v in tot.items()}            # released rounding, applied once


if os.environ.get("CHUNKED_EVAL"): runner.evaluate = _chunked_evaluate
if os.environ.get("PROBE"):   # feasibility probe: the released training step (BaseRunner.fit body) at decreasing batch sizes
    sys.path.insert(0, os.path.join(os.environ.get("OPB_ROOT", "."), "benchmark_build")); from probe_util import probe
    from torch.utils.data import DataLoader
    model.optimizer = runner._build_optimizer(model); data["train"].actions_before_epoch(); model.train(); _it = {}
    def _step(bs):
        if bs not in _it: _it[bs] = iter(DataLoader(data["train"], batch_size=bs, shuffle=True, num_workers=4, collate_fn=data["train"].collate_batch))
        try: batch = next(_it[bs])
        except StopIteration: _it.pop(bs); return _step(bs)
        batch = utils.batch_to_gpu(batch, model.device); model.optimizer.zero_grad()
        loss = model.loss(model(batch, epoch=1)); loss.backward(); model.optimizer.step()
    probe(_step, len(data["train"]), int(args.batch_size), os.environ["PROBE"], method="ReaRec", bench=B, users=N)
t0 = time.time(); _fit = runner.fit


def fit(dataset, epoch=-1):
    """Change (1): once the budget is spent, return NaN so the released loop stops and reloads the best model."""
    if (time.time() - t0) / 3600 > BUDGET: return float("nan")
    return _fit(dataset, epoch=epoch)


runner.fit = fit
if os.environ.get("SCORE_ONLY"):   # change (3), 2026-10-29: score the saved best checkpoint (training ended; our scoring step had failed)
    train_hours = float(os.environ.get("TRAIN_HOURS", "nan"))
else:
    runner.train(data)                                   # released: trains, validates (NDCG@20), saves best, reloads it
    train_hours = (time.time() - t0) / 3600
model.load_model(); model.eval()

fl = pd.read_parquet(FL).sort_values("event_row").reset_index(drop=True)
erow = con.execute(f"""SELECT cascade_id, "position" FROM read_parquet('{B}/events.parquet') ORDER BY t, cascade_id, "position" """).fetchnumpy()
qc, qp = erow["cascade_id"][fl.event_row.values], erow["position"][fl.event_row.values]
first = torch.from_numpy(pa_["f"]).cuda(); elig = torch.from_numpy(pa_["is_crawled"].astype(bool) if "is_crawled" in pa_ else np.ones(N, bool)).cuda()
PAD = corpus.n_items - 1; Rk = np.zeros(len(fl)); nc = np.zeros(len(fl)); QB = int(os.environ.get("QB", 256)); t1 = time.time()   # QB: scoring batch (memory only)
with torch.no_grad():
    for b0 in range(0, len(fl), QB):
        idx = list(range(b0, min(b0 + QB, len(fl)))); S = np.full((len(idx), MAX_ITEM_SEQ_LEN), PAD, np.int64); lens = []
        for r, i in enumerate(idx):
            s0 = cstart[qc[i]]; w = pid[s0:s0 + qp[i]][-MAX_ITEM_SEQ_LEN:]; S[r, MAX_ITEM_SEQ_LEN - len(w):] = w; lens.append(len(w))   # left padding, as the reader
        fd = {ITEM_SEQ: torch.from_numpy(S).cuda(), ITEM_SEQ_LEN: torch.tensor(lens).cuda(), ITEM_ID: torch.zeros(len(idx), dtype=torch.long).cuda()}
        lg = model(fd, stage="infer")["prediction"][:, :N]
        for r, i in enumerate(idx):
            s0 = cstart[qc[i]]; j = s0 + qp[i]; y = int(pid[j]); tq = int(ev["t"][j])
            cand = elig & (first < tq); cand[torch.from_numpy(pid[s0:j]).cuda()] = False; cand[y] = True
            v = lg[r, y]; c = lg[r][cand]; Rk[i] = 1 + float((c > v).sum()) + 0.5 * (float((c == v).sum()) - 1); nc[i] = float(cand.sum())
assert np.array_equal(nc, fl.n_candidates.values), "candidate set differs from floors"
log(f"scored {len(fl):,} queries in {time.time() - t1:.0f} s")
out = fl[["event_row", "cascade_id", "participant_id", "t", "split", "position", "in_all_sample", "in_novel_sample", "novel", "n_candidates"]].copy()
out["rank_rearec"] = Rk; out.to_parquet(f"{OUT}/rearec_seed{SEED}.parquet", index=False)
json.dump(dict(seed=SEED, train_hours=train_hours, budget_h=BUDGET, args={k: str(v) for k, v in vars(args).items()}), open(f"{OUT}/rearec_seed{SEED}.json", "w"), indent=1)
log("DONE")
