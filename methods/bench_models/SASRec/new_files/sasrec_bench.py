"""SASRec (ICDM 2018, kang205/SASRec, official TF1 release) on benchmark tables. Released model, sampler, loss and hyperparameters.

    bm-tf1/bin/python sasrec_bench.py BENCH FLOORS.parquet OUTDIR SEED [BUDGET_HOURS=8] [EPOCHS=201]

Mapping: each TRAIN cascade with >= 2 events is a "user"; participant p is item p+1 (0 = released padding). The released
WarpSampler (uniform user, last maxlen+1 events, one uniform negative per position) and released BCE objective train the
model; released defaults maxlen 50, d 50, 2 blocks, 1 head, dropout 0.5, lr 1e-3, batch 128, 201 epochs of
len(users)//128 steps.
Changes (BENCH_CHANGES.md): (1) evaluation ranks the FULL candidate set (released: 1 positive vs 100 sampled negatives);
(2) model selection by the released metric, NDCG@10, on VALIDATION queries (released: reported on val and test, no
selection); evaluated every 20 epochs as released, and also after each hour of training and at the budget end so that
selection has checkpoints when 20 epochs exceed the 8 GPU-h seed budget, and after the last epoch.
Evaluation: the query rows of FLOORS.parquet; input = the last 50 earlier events of the cascade; score = the released
dot product of the last-position representation with the item table; candidates as floors_eval.py (eligible, first seen
before t, minus every earlier participant of the cascade, the target always included); ties count half.
"""
import argparse
import json
import os
import sys
import time

import duckdb
import numpy as np
import pandas as pd
import tensorflow as tf

B, FL, OUT, SEED = sys.argv[1:5]
SEED = int(SEED); BUDGET = float(sys.argv[5]) if len(sys.argv) > 5 else 8.0; EPOCHS = int(sys.argv[6]) if len(sys.argv) > 6 else 201
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from model import Model  # noqa: E402
from sampler import WarpSampler  # noqa: E402

args = argparse.Namespace(batch_size=128, lr=0.001, maxlen=50, hidden_units=50, num_blocks=2, num_epochs=EPOCHS, num_heads=1, dropout_rate=0.5, l2_emb=0.0)
np.random.seed(SEED); tf.set_random_seed(SEED)
os.makedirs(OUT, exist_ok=True)
log = lambda s: print(f"[sasrec {time.strftime('%H:%M:%S')}] {s}", flush=True)
con = duckdb.connect(); con.execute("SET threads=32; SET memory_limit='100GB'")
P = f"read_parquet('{B}/participants.parquet')"; cols = [r[0] for r in con.execute(f"DESCRIBE SELECT * FROM {P}").fetchall()]
ev = con.execute(f"""SELECT e.cascade_id, e.participant_id, e.t, e."position", (CASE c.split WHEN 'train' THEN 0 WHEN 'val' THEN 1 ELSE 2 END)::TINYINT split
                     FROM read_parquet('{B}/events.parquet') e JOIN read_parquet('{B}/cascades.parquet') c USING (cascade_id)
                     ORDER BY e.cascade_id, e."position" """).fetchnumpy()
pa_ = con.execute(f"SELECT coalesce(first_seen_t, 9e18)::BIGINT f{', is_crawled' if 'is_crawled' in cols else ''} FROM {P} ORDER BY participant_id").fetchnumpy()
N = len(pa_["f"]); itemnum = N
first = np.concatenate([[np.iinfo(np.int64).max], pa_["f"]]).astype(np.int64)                  # item index space (0 = pad)
elig = np.concatenate([[False], pa_["is_crawled"].astype(bool) if "is_crawled" in pa_ else np.ones(N, bool)])
cstart = np.searchsorted(ev["cascade_id"], np.arange(ev["cascade_id"].max() + 2)); csplit = ev["split"][cstart[:-1]]
items = (ev["participant_id"] + 1).astype(np.int32)
tr = [c for c in np.flatnonzero(csplit == 0) if cstart[c + 1] - cstart[c] >= 2]
user_train = {u + 1: items[cstart[c]:cstart[c + 1]].tolist() for u, c in enumerate(tr)}; usernum = len(user_train)
num_batch = len(user_train) // args.batch_size
log(f"{N:,} items, {usernum:,} train cascades (users), {num_batch:,} steps per released epoch")

# query rows, ordered as the floors' event_row (events sorted by t, cascade_id, position)
fl = pd.read_parquet(FL).sort_values("event_row").reset_index(drop=True)
erow = con.execute(f"""SELECT cascade_id, "position" FROM read_parquet('{B}/events.parquet') ORDER BY t, cascade_id, "position" """).fetchnumpy()
qc, qp = erow["cascade_id"][fl.event_row.values], erow["position"][fl.event_row.values]


def query_arrays(idx):
    """Released right-aligned input sequence, target item, query time and deduplicated earlier participants (minus target)."""
    S = np.zeros((len(idx), args.maxlen), np.int32); y = np.zeros(len(idx), np.int32); tq = np.zeros(len(idx), np.int64); prefs = []
    for r, i in enumerate(idx):
        s0 = cstart[qc[i]]; j = s0 + qp[i]; pre = items[s0:j]
        w = pre[-args.maxlen:]; S[r, args.maxlen - len(w):] = w; y[r] = items[j]; tq[r] = ev["t"][j]
        prefs.append(np.setdiff1d(pre, [y[r]]))
    Pm = np.zeros((len(idx), max(1, max(len(p) for p in prefs))), np.int32)
    for r, p in enumerate(prefs): Pm[r, :len(p)] = p
    return S, y, tq, Pm


model = Model(usernum, itemnum, args)
# full-candidate rank in the graph (benchmark evaluation; uses the two exposed tensors, see BENCH_CHANGES.md)
with tf.variable_scope("bench_eval"):
    first_ph, elig_ph = tf.placeholder(tf.int64, [N + 1]), tf.placeholder(tf.bool, [N + 1])
    first_v = tf.Variable(first_ph, trainable=False, collections=[]); elig_v = tf.Variable(elig_ph, trainable=False, collections=[])
    q_y, q_t, q_pref = tf.placeholder(tf.int32, [None]), tf.placeholder(tf.int64, [None]), tf.placeholder(tf.int32, [None, None])
    logits = tf.matmul(model.bench_last_seq, model.bench_item_table, transpose_b=True)                    # [Q, N+1]
    base = tf.logical_and(elig_v[None, :], first_v[None, :] < q_t[:, None])                              # candidates before prefix removal
    rows = tf.range(tf.shape(q_y)[0])
    v = tf.gather_nd(logits, tf.stack([rows, q_y], 1))[:, None]
    gt = tf.reduce_sum(tf.cast(tf.logical_and(base, logits > v), tf.float64), 1)
    eq = tf.reduce_sum(tf.cast(tf.logical_and(base, tf.equal(logits, v)), tf.float64), 1)
    pl = tf.gather(logits, q_pref, batch_dims=1); pb = tf.logical_and(tf.gather(base, q_pref, batch_dims=1), q_pref > 0)
    gt -= tf.reduce_sum(tf.cast(tf.logical_and(pb, pl > v), tf.float64), 1)
    eq -= tf.reduce_sum(tf.cast(tf.logical_and(pb, tf.equal(pl, v)), tf.float64), 1)
    y_in = tf.cast(tf.gather_nd(base, tf.stack([rows, q_y], 1)), tf.float64)
    rank = 1 + gt + 0.5 * (eq + (1 - y_in) - 1)
    ncand = tf.reduce_sum(tf.cast(base, tf.float64), 1) - tf.reduce_sum(tf.cast(pb, tf.float64), 1) + (1 - y_in)   # must equal floors n_candidates

config = tf.ConfigProto(); config.gpu_options.allow_growth = True; config.allow_soft_placement = True
sess = tf.Session(config=config)
sess.run(tf.initialize_all_variables())
sess.run([first_v.initializer, elig_v.initializer], {first_ph: first, elig_ph: elig})
saver = tf.train.Saver(max_to_keep=1)
QB = int(max(1, min(256, 2e8 // (N + 1))))


def ranks(idx):
    Rk = np.zeros(len(idx))
    for b0 in range(0, len(idx), QB):
        S, y, tq, Pm = query_arrays(idx[b0:b0 + QB])
        Rk[b0:b0 + len(y)], nc = sess.run([rank, ncand], {model.input_seq: S, model.is_training: False, q_y: y, q_t: tq, q_pref: Pm})
        ref = fl.n_candidates.values[idx[b0:b0 + QB]]
        assert np.array_equal(nc, ref), f"candidate set differs from floors: {nc[:5]} vs {ref[:5]}"
    return Rk


val_rows = np.flatnonzero(fl.split.values == "val"); rs = np.random.RandomState(SEED)
val_sel = np.sort(rs.choice(val_rows, min(5000, len(val_rows)), replace=False))
ndcg10 = lambda r: float(np.where(r <= 10, 1 / np.log2(r + 1), 0).mean())
sampler = WarpSampler(user_train, usernum, itemnum, batch_size=args.batch_size, maxlen=args.maxlen, n_workers=3)
ck = f"{OUT}/sasrec_seed{SEED}.ckpt"; best, hist, t0, last_eval = -1.0, [], time.time(), time.time()
try:
    for epoch in (range(0) if os.environ.get("SCORE_ONLY") else range(1, args.num_epochs + 1)):   # SCORE_ONLY=1: score the saved checkpoint in OUT (leakage test, 2026-11-08)
        for step in range(num_batch):
            u, seq, pos, neg = sampler.next_batch()
            auc, loss, _ = sess.run([model.auc, model.loss, model.train_op], {model.u: u, model.input_seq: seq, model.pos: pos, model.neg: neg, model.is_training: True})
            over = (time.time() - t0) / 3600 > BUDGET
            if over or time.time() - last_eval > 3600 or (step == num_batch - 1 and (epoch % 20 == 0 or epoch == args.num_epochs)):
                te = time.time(); r = ranks(val_sel); v_ = ndcg10(r)
                hist.append(dict(epoch=epoch, step=step + 1, train_hours=(te - t0) / 3600, loss=float(loss), auc=float(auc), val_ndcg10=v_, val_hits100=float((r <= 100).mean())))
                log(f"epoch {epoch} step {step + 1:,} {hist[-1]} (eval {time.time() - te:.0f} s)")
                if v_ > best: best = v_; saver.save(sess, ck)
                t0 += time.time() - te; last_eval = time.time()        # evaluation time does not count against the training budget
            if over: break
        if (time.time() - t0) / 3600 > BUDGET: break
        if epoch % 20 == 0: log(f"epoch {epoch} done, loss {float(loss):.4f}")
finally:
    sampler.close()
train_hours = hist[-1]["train_hours"] if hist else 0.0
saver.restore(sess, ck); t1 = time.time()
Rk = ranks(np.arange(len(fl))); log(f"scored {len(fl):,} queries in {time.time() - t1:.0f} s")
out = fl[["event_row", "cascade_id", "participant_id", "t", "split", "position", "in_all_sample", "in_novel_sample", "novel", "n_candidates"]].copy()
out["rank_sasrec"] = Rk; out.to_parquet(f"{OUT}/sasrec_seed{SEED}.parquet", index=False)
json.dump(dict(seed=SEED, hist=hist, best_val_ndcg10=best, train_hours=train_hours, budget_h=BUDGET, args=vars(args)), open(f"{OUT}/sasrec_seed{SEED}.json", "w"), indent=1)
log("DONE")
