"""FEARec (SIGIR 2023, sudaada/FEARec @ 5e98f45, official release with its vendored RecBole 0.2.1) on benchmark tables.
Released model, losses (CE + contrastive + semantic augmentation), RecBole data pipeline and trainer; configuration =
seq.yaml + FEARec.yaml + the flags of the released fearec.sh (the reproduced command).

    bm-dgl/bin/python fearec_bench.py BENCH FLOORS.parquet OUTDIR SEED [BUDGET_HOURS=8]

Mapping: each TRAIN cascade with >= 2 events is a RecBole user; participant p is item token p; the interaction order is
the cascade order (timestamp = position). Every participant is listed in bench.item, so the vocabulary is the population.
Changes (BENCH_CHANGES.md): (1) two hold-out events (item token BENCH_HOLDOUT) are appended to every train cascade, so the
released leave-one-out split (TO_LS) takes them as its internal valid/test targets and every real event is a training
target; the hold-out item is never a candidate; (2) the released user/item 5-core filters are off (min_user_inter_num and
min_item_inter_num 0): T1 ranks every participant and cascades are short; (3) model selection by the released
valid_metric, Recall@10, under full ranking on 5,000 benchmark validation queries (instead of the internal hold-out),
released stopping_step 40, plus the time budget; (4) the released quick_start's post-training embedding plot is skipped.
Scoring: the released full_sort_predict (sequence representation . item table) on the last <= 50 earlier events of each
floors query; candidates as floors_eval.py; ties count half; candidate counts must equal the floors' n_candidates.
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
GPU = os.environ.get("CUDA_VISIBLE_DEVICES", "")
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE); os.chdir(HERE); sys.argv = [sys.argv[0]]
from recbole.config import Config  # noqa: E402
from recbole.data import create_dataset, data_preparation  # noqa: E402
from recbole.utils import init_logger, get_model, get_trainer, init_seed  # noqa: E402

os.makedirs(OUT, exist_ok=True)
log = lambda s: print(f"[fearec {time.strftime('%H:%M:%S')}] {s}", flush=True)
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

# ---- export: RecBole atomic files (the semantic-augmentation cache lives in data_path, so one directory per run)
DP = f"{OUT}/recbole_fearec_seed{SEED}"; D = f"{DP}/bench"; os.makedirs(D, exist_ok=True)
tr = [c for c in np.flatnonzero(csplit == 0) if cstart[c + 1] - cstart[c] >= 2]
with open(f"{D}/bench.inter", "w") as f:
    f.write("user_id:token\titem_id:token\ttimestamp:float\n")
    for u, c in enumerate(tr):
        x = pid[cstart[c]:cstart[c + 1]]
        for k, p in enumerate(x): f.write(f"{u}\t{p}\t{k}\n")
        f.write(f"{u}\tBENCH_HOLDOUT\t{len(x)}\n{u}\tBENCH_HOLDOUT\t{len(x) + 1}\n")       # change (1)
with open(f"{D}/bench.item", "w") as f:
    f.write("item_id:token\n" + "\n".join(map(str, range(N))) + "\nBENCH_HOLDOUT\n")
log(f"{N:,} items, {len(tr):,} train cascades exported")

# ---- configuration: seq.yaml + FEARec.yaml + fearec.sh flags (released), our data and seed
config_dict = {"train_batch_size": 256, "lmd_sem": 0.1, "contrast": "us_x", "sim": "dot", "tau": 1, "global_ratio": 0.6, "dual_domain": True,
               "spatial_ratio": 0.1, "std": True,                                                    # fearec.sh
               "data_path": DP, "load_col": {"inter": ["user_id", "item_id", "timestamp"], "item": ["item_id"]},
               "min_user_inter_num": 0, "min_item_inter_num": 0,                                    # change (2)
               "seed": SEED, "gpu_id": GPU, "show_progress": False}
config = Config(model="FEARec", dataset="bench", config_file_list=["seq.yaml"], config_dict=config_dict)
init_seed(config["seed"], config["reproducibility"]); init_logger(config)
import logging  # noqa: E402
config["log_dir"] = f"{OUT}/fearec_ckpt_seed{SEED}"; os.makedirs(config["log_dir"], exist_ok=True)   # trainer saves model.pth here (released: the log folder)
dataset = create_dataset(config)
t_a = time.time(); train_data, valid_data, test_data = data_preparation(config, dataset); log(f"data preparation (incl. semantic augmentation) {time.time() - t_a:.0f} s")
model = get_model("FEARec")(config, train_data).to(config["device"]); dev = config["device"]
tok = dataset.field2token_id["item_id"]
iid = torch.tensor([tok[str(p)] for p in range(N)], dtype=torch.long, device=dev)
first = torch.from_numpy(pa_["f"]).to(dev); elig = torch.from_numpy(pa_["is_crawled"].astype(bool) if "is_crawled" in pa_ else np.ones(N, bool)).to(dev)
L = config["MAX_ITEM_LIST_LENGTH"]


@torch.no_grad()
def ranks(idx, QB=256):
    """Released full_sort_predict over every item; rank of the target among the floors' candidates."""
    model.eval(); Rk = np.zeros(len(idx)); nc = np.zeros(len(idx))
    for b0 in range(0, len(idx), QB):
        bi = idx[b0:b0 + QB]; S = torch.zeros(len(bi), L, dtype=torch.long, device=dev); lens = []
        for r, i in enumerate(bi):
            s0 = cstart[qc[i]]; w = pid[s0:s0 + qp[i]][-L:]; S[r, :len(w)] = iid[torch.from_numpy(w).to(dev)]; lens.append(len(w))
        inter = {model.ITEM_SEQ: S, model.ITEM_SEQ_LEN: torch.tensor(lens, device=dev)}
        lg = model.full_sort_predict(inter)[:, iid]                                                # [Q, N] in participant order
        for r, i in enumerate(bi):
            s0 = cstart[qc[i]]; j = s0 + qp[i]; y = int(pid[j]); tq = int(ev["t"][j])
            cand = elig & (first < tq); cand[torch.from_numpy(pid[s0:j]).to(dev)] = False; cand[y] = True
            v = lg[r, y]; c = lg[r][cand]; Rk[b0 + r] = 1 + float((c > v).sum()) + 0.5 * (float((c == v).sum()) - 1); nc[b0 + r] = float(cand.sum())
    ref = fl.n_candidates.values[idx]
    assert np.array_equal(nc, ref), f"candidate set differs from floors: {nc[:5]} vs {ref[:5]}"
    return Rk


trainer = get_trainer(config["MODEL_TYPE"], config["model"])(config, model)
t0 = time.time(); hist = []


def valid_epoch(valid_data, show_progress=False):
    """Change (3): released valid_metric (Recall@10) under full ranking on benchmark validation queries; time budget."""
    te = time.time(); r = ranks(val_sel); score = float((r <= 10).mean())
    res = {"recall@10": score, "ndcg@10": float(np.where(r <= 10, 1 / np.log2(r + 1), 0).mean()), "hits@100": float((r <= 100).mean())}
    hist.append(dict(train_hours=(te - t0) / 3600, eval_s=time.time() - te, **res)); log(f"valid {hist[-1]}")
    if (time.time() - t0) / 3600 > BUDGET:
        if score > trainer.best_valid_score: trainer.best_valid_score = score; trainer._save_checkpoint(-1)
        trainer.stopping_step = -1; return -1.0, res
    return score, res


trainer._valid_epoch = valid_epoch
trainer.fit(train_data, valid_data, saved=True, show_progress=False)
train_hours = (time.time() - t0) / 3600
model.load_state_dict(torch.load(trainer.saved_model_file, map_location=dev)["state_dict"])
t1 = time.time(); Rk = ranks(np.arange(len(fl))); log(f"scored {len(fl):,} queries in {time.time() - t1:.0f} s")
out = fl[["event_row", "cascade_id", "participant_id", "t", "split", "position", "in_all_sample", "in_novel_sample", "novel", "n_candidates"]].copy()
out["rank_fearec"] = Rk; out.to_parquet(f"{OUT}/fearec_seed{SEED}.parquet", index=False)
json.dump(dict(seed=SEED, hist=hist, best_valid_recall10=trainer.best_valid_score, train_hours=train_hours, budget_h=BUDGET), open(f"{OUT}/fearec_seed{SEED}.json", "w"), indent=1)
log("DONE")
