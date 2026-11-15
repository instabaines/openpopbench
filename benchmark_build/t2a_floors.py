"""T2a 'is the next participant new?' floors (CPU).

    python t2a_floors.py           writes $OPB_WORK/eval/t2/t2a_<tier>.parquet (per-query probabilities) and T2A_LEADERBOARD.md

Query: an all-events-sample floors row (validation or test). Label: the participant has no event in any TRAIN cascade of
the tier (unseen_train, t2_flags.py). Training-split events cannot have this label, so floors are FITTED ON VALIDATION
and evaluated on test (the validation rows are reported for reference only).
Features, all observable before the query time t:
  pos      log1p(position in the cascade)            age     log1p(seconds since the cascade's first event)
  gap      log1p(seconds since the cascade's previous event)
  pnew     share of the cascade's earlier participants that are unseen in training
  onew     1 if the cascade's origin is unseen in training
  (days since the end of training was tried and dropped: fitted on validation, it extrapolates badly on test)
  roll     share of newcomers among the last 1,000 non-origin events of the whole stream before t
Floors: const (validation rate), roll (the rolling stream share itself, no fitting), logit (logistic regression on all
features, fitted on validation). Metrics: log loss, Brier, ECE (10 equal-width bins), AUC; 95% CIs by cluster bootstrap
over cascades (1,000 resamples).
"""
import os
import sys

import duckdb
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from t2_flags import BB, TIERS  # noqa: E402

OUT = f"{BB}/eval/t2"; FEAT = ["pos", "age", "gap", "pnew", "onew", "roll"]   # "days" dropped (2026-10-28): fitted on validation it extrapolates on test (ECE 0.2-0.4)
WIN = 1000
rng = np.random.default_rng(0)


def features(B, FL, tier):
    con = duckdb.connect(); con.execute("SET threads=32; SET memory_limit='100GB'")
    ev = con.execute(f"""SELECT e.cascade_id, e.participant_id, e.t, e."position", c.split FROM read_parquet('{B}/events.parquet') e
                         JOIN read_parquet('{B}/cascades.parquet') c USING (cascade_id) ORDER BY e.cascade_id, e."position" """).df()
    seen = np.zeros(int(ev.participant_id.max()) + 1, bool); seen[ev.participant_id[ev.split == "train"].values] = True
    ev["new"] = ~seen[ev.participant_id.values]
    train_end = ev.t[ev.split == "train"].max()
    cs = np.searchsorted(ev.cascade_id.values, np.arange(ev.cascade_id.max() + 2))
    erow = con.execute(f"""SELECT cascade_id, "position" FROM read_parquet('{B}/events.parquet') ORDER BY t, cascade_id, "position" """).fetchnumpy()
    fg = pd.read_parquet(f"{OUT}/flags_{tier}.parquet").sort_values("event_row").reset_index(drop=True)
    fl = pd.read_parquet(FL, columns=["event_row", "cascade_id", "split", "in_all_sample", "t"]).sort_values("event_row").reset_index(drop=True)
    fl["y"] = fg.unseen_train.values; fl = fl[fl.in_all_sample].reset_index(drop=True)
    qc, qp = erow["cascade_id"][fl.event_row.values], erow["position"][fl.event_row.values]
    assert (qc == fl.cascade_id.values).all()
    j = cs[qc] + qp; s0 = cs[qc]; t = ev.t.values; new = ev.new.values.astype(float)
    cnew = np.concatenate([[0], np.cumsum(new)])
    # stream rolling share: non-origin events in time order
    # stable order (t, cascade_id, position) = the floors event order. Fix 2026-11-08 (leakage test): sort_values("t") is a
    # quicksort, so the order of tied timestamps, and with it the 1,000-event window, depended on rows after the query time.
    no = ev[ev.position > 0].sort_values(["t", "cascade_id", "position"], kind="stable"); nt = no.t.values; cn = np.concatenate([[0], np.cumsum(no.new.values.astype(float))])
    k = np.searchsorted(nt, fl.t.values, side="left"); lo = np.maximum(k - WIN, 0)
    X = pd.DataFrame(dict(pos=np.log1p(qp), age=np.log1p(t[j - 1] - t[s0] + (t[j] - t[j - 1]) * 0), gap=np.log1p(fl.t.values - t[j - 1]),
                          pnew=(cnew[j] - cnew[s0 + 1]) / np.maximum(qp - 1, 1), onew=new[s0], days=(fl.t.values - train_end) / 86400,
                          roll=(cn[k] - cn[lo]) / np.maximum(k - lo, 1)))
    X["age"] = np.log1p(fl.t.values - t[s0])          # time since cascade start, at the query time (known: the query time is given)
    return fl, X


def metrics(y, p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    ll = -(y * np.log(p) + (1 - y) * np.log(1 - p)); br = (p - y) ** 2
    b = np.minimum((p * 10).astype(int), 9); ece = sum(abs(p[b == i].mean() - y[b == i].mean()) * (b == i).mean() for i in range(10) if (b == i).any())
    return ll, br, ece, (roc_auc_score(y, p) if 0 < y.sum() < len(y) else np.nan)


if __name__ == "__main__":
    L = ["# T2a: is the next participant new? (floors)", "", __doc__.split("\n\n", 1)[1].strip(), ""]
    for tier, (B, FL) in TIERS.items():
        if not os.path.exists(f"{OUT}/flags_{tier}.parquet"): continue
        fl, X = features(B, FL, tier)
        va, te = (fl.split == "val").values, (fl.split == "test").values
        y = fl.y.values.astype(float)
        Z = X[FEAT].values; mu, sd = Z[va].mean(0), Z[va].std(0) + 1e-9; Z = (Z - mu) / sd
        lr = LogisticRegression(max_iter=2000).fit(Z[va], y[va])
        P = dict(const=np.full(len(fl), y[va].mean()), roll=np.clip(X.roll.values, 1e-4, 1 - 1e-4), logit=lr.predict_proba(Z)[:, 1])
        out = fl[["event_row", "cascade_id", "split", "y"]].copy()
        for k, v in P.items(): out[f"p_{k}"] = v
        out.to_parquet(f"{OUT}/t2a_{tier}.parquet", index=False)
        # model-implied probabilities: softmax mass on newcomer candidates (b1_ranker.py t2a), unfitted. Since 2026-11-11 the
        # metrics are computed per seed and reported as mean ± sd over seeds (paper standard); earlier boards scored the seed average.
        import glob, re
        for g in sorted(glob.glob(f"{BB}/baselines/{tier}/t2a_*_seed*.parquet")):
            m = re.match(r"t2a_(.+)_seed\d+\.parquet", g.split("/")[-1]).group(1)
            x = pd.read_parquet(g).set_index("event_row").p_new.reindex(fl.event_row.values).values
            P.setdefault(m + " (model mass)", []).append(x)
        P = {k: (np.stack(v) if isinstance(v, list) else v[None]) for k, v in P.items()}
        L += [f"## {tier}, test (n = {te.sum():,}; newcomer rate {y[te].mean():.3f}; validation rate {y[va].mean():.3f})", "",
              "| method | log loss | Brier | ECE | AUC | mean p |", "|---|---|---|---|---|---|"]
        cid = pd.factorize(fl.cascade_id.values[te])[0]; W = rng.poisson(1.0, (1000, cid.max() + 1)).astype(np.float32)[:, cid]
        for k, v in P.items():
            per = [metrics(y[te], vs[te]) for vs in v]                     # one tuple per seed
            ll, br = np.mean([r[0] for r in per], 0), np.mean([r[1] for r in per], 0)
            ci = lambda x: np.percentile((W @ x) / W.sum(1), [2.5, 97.5])
            a, b_ = ci(ll); c, d = ci(br)
            sd = lambda xs, f: f" ± {np.std(xs, ddof=1):{f}}" if len(xs) > 1 else ""
            lls, brs, eces, aucs, mps = [r[0].mean() for r in per], [r[1].mean() for r in per], [r[2] for r in per], [r[3] for r in per], [vs[te].mean() for vs in v]
            L.append(f"| {k} | {np.mean(lls):.4f}{sd(lls, '.4f')} [{a:.4f}, {b_:.4f}] | {np.mean(brs):.4f}{sd(brs, '.4f')} [{c:.4f}, {d:.4f}] | {np.mean(eces):.4f}{sd(eces, '.4f')} | {np.mean(aucs):.3f}{sd(aucs, '.3f')} | {np.mean(mps):.4f}{sd(mps, '.4f')} |")
        L += ["", "logit coefficients (standardized features): " + ", ".join(f"{f} {c:+.3f}" for f, c in zip(FEAT, lr.coef_[0])), ""]
        print(tier, "done", flush=True)
    open(f"{OUT}/T2A_LEADERBOARD.md", "w").write("\n".join(L) + "\n")
