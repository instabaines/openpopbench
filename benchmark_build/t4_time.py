"""T4 event timing: floors and leaderboard (CPU).

    python t4_time.py floors       writes $OPB_WORK/eval/t4/floors_<tier>.parquet
    python t4_time.py board        writes $OPB_WORK/eval/t4/T4_LEADERBOARD.md (floors + $OPB_WORK/baselines/<tier>/<method>_t4_seed<s>.parquet)

Query: an all-events-sample floors row (cascade c, position p >= 1; the same rows as T1). Target: the gap d = t_p - t_{p-1}
in seconds, given the cascade up to and including event p-1 and the platform history. All methods give a density for d.
Metrics: time NLL (nats per event, d in seconds; lower is better), log-scale MAE |log1p(median gap) - log1p(d)| of the
predicted median, and KS = max |ECDF(PIT) - uniform| of the probability integral transform u = F(d) (calibration; 0 is
perfect). Median NLL (of the seed-averaged per-query NLL) is shown because a model whose intensity grows with the gap
(RMTPP: exp(w * gap)) can put a few queries at NLL in the millions, which dominates the mean. 95% CIs (NLL, MAE): cluster bootstrap over cascades, 1,000 resamples, of the seed-averaged per-query value.
Floors, fitted on TRAIN-split events (position >= 1):
  exp        exponential, one global rate (a homogeneous Poisson process)
  lognorm    log1p(d) ~ Normal(mu, sigma), global
  lognorm_x  log1p(d) ~ Normal(x . w, sigma); x = [1, log1p(p), log1p(t_{p-1} - t_0), log1p(t_{p-1} - t_{p-2})] (least squares)
Point-process models (RMTPP, S2P2, WSMTHP, AttNHP): saved T1 checkpoints re-scored by easytpp_bench.py (T4=1, change 8).
"""
import glob
import os
import re
import sys

import duckdb
import numpy as np
import pandas as pd
from scipy.stats import norm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from t2_flags import BB, TIERS  # noqa: E402

OUT = f"{BB}/eval/t4"; os.makedirs(OUT, exist_ok=True)
rng = np.random.default_rng(0)


def gaps(B):
    con = duckdb.connect(); con.execute("SET threads=32; SET memory_limit='100GB'")
    ev = con.execute(f"""SELECT e.cascade_id, e.t, e."position", c.split FROM read_parquet('{B}/events.parquet') e
                         JOIN read_parquet('{B}/cascades.parquet') c USING (cascade_id) ORDER BY e.cascade_id, e."position" """).fetchnumpy()
    erow = con.execute(f"""SELECT cascade_id, "position" FROM read_parquet('{B}/events.parquet') ORDER BY t, cascade_id, "position" """).fetchnumpy()
    cs = np.searchsorted(ev["cascade_id"], np.arange(ev["cascade_id"].max() + 2)); t = ev["t"].astype(np.float64)
    j = np.arange(len(t)); s0 = cs[ev["cascade_id"]]; p = ev["position"]
    ok = p >= 1; jp = np.maximum(j - 1, 0)
    X = np.stack([np.ones(len(t)), np.log1p(p), np.log1p(t[jp] - t[s0]), np.log1p(np.where(p >= 2, t[jp] - t[np.maximum(j - 2, 0)], 0))], 1)
    d = np.where(ok, t - t[jp], np.nan)
    return ev, erow, cs, X, d, ok


def floors():
    for tier, (B, FL) in TIERS.items():
        if not os.path.exists(FL): continue
        ev, erow, cs, X, d, ok = gaps(B)
        tr = ok & (ev["split"] == "train")
        fl = pd.read_parquet(FL, columns=["event_row", "cascade_id", "split", "in_all_sample"]).sort_values("event_row").reset_index(drop=True)
        fl = fl[fl.in_all_sample].reset_index(drop=True)
        qc, qp = erow["cascade_id"][fl.event_row.values], erow["position"][fl.event_row.values]; j = cs[qc] + qp
        dq, Xq = d[j], X[j]; out = fl[["event_row", "cascade_id", "split"]].copy(); out["gap"] = dq
        r = 1 / d[tr].mean()                                                           # exp
        out["nll_exp"], out["pit_exp"], out["med_exp"] = -np.log(r) + r * dq, 1 - np.exp(-r * dq), np.log(2) / r
        y = np.log1p(d[tr]); yq = np.log1p(dq)
        for name, mu_tr, mu_q in (("lognorm", np.full(tr.sum(), y.mean()), np.full(len(dq), y.mean())),
                                  ("lognorm_x",) + (lambda w: (X[tr] @ w, Xq @ w))(np.linalg.lstsq(X[tr], y, rcond=None)[0])):
            sd = (y - mu_tr).std()
            out[f"nll_{name}"] = -norm.logpdf(yq, mu_q, sd) + yq                         # density of d: N(log1p d) / (1 + d)
            out[f"pit_{name}"], out[f"med_{name}"] = norm.cdf(yq, mu_q, sd), np.expm1(mu_q)
        out.to_parquet(f"{OUT}/floors_{tier}.parquet", index=False); print(tier, "floors done", flush=True)


def board():
    L = ["# T4: event timing (next-event gap)", "", __doc__.split("\n\n", 1)[1].strip(), ""]
    for tier in TIERS:
        f = f"{OUT}/floors_{tier}.parquet"
        if not os.path.exists(f): continue
        fl = pd.read_parquet(f); M = {k: [(fl[f"nll_{k}"].values, fl[f"pit_{k}"].values, fl[f"med_{k}"].values)] for k in ("exp", "lognorm", "lognorm_x")}
        for g in sorted(glob.glob(f"{BB}/baselines/{tier}/*_t4_seed*.parquet")):
            m = re.match(r"(.+)_t4_seed\d+\.parquet", g.split("/")[-1]).group(1)
            x = pd.read_parquet(g).set_index("event_row").reindex(fl.event_row.values)
            M.setdefault(m, []).append((x.nll_time.values, x.pit.values, x.median_gap.values))
        for split in ("test",):
            s = (fl.split == split).values
            L += [f"## {tier}, {split}", "", "| method | seeds | n | time NLL (nats) | median NLL | log MAE | KS (PIT) |", "|---|---|---|---|---|---|---|"]
            rows = []
            for name, runs in M.items():
                ok = s & np.all([~np.isnan(r[0]) for r in runs], 0)
                cid = pd.factorize(fl.cascade_id.values[ok])[0]; W = rng.poisson(1.0, (1000, cid.max() + 1)).astype(np.float32)[:, cid]
                yq = np.log1p(fl.gap.values[ok])
                nll = np.stack([r[0][ok] for r in runs]); mae = np.stack([np.abs(np.log1p(r[2][ok]) - yq) for r in runs])
                ks = [np.max(np.abs(np.sort(r[1][ok]) - (np.arange(ok.sum()) + 0.5) / ok.sum())) for r in runs]
                ci = lambda a: np.percentile((W @ a.mean(0)) / W.sum(1), [2.5, 97.5])
                sd = lambda a: f" ± {a.mean(1).std(ddof=1):.3f}" if len(runs) > 1 else ""
                (a, b), (c, d) = ci(nll), ci(mae)
                ks_sd = f" ± {np.std(ks, ddof=1):.3f}" if len(ks) > 1 else ""
                rows.append((nll.mean(), f"| {name} | {len(runs)} | {ok.sum():,} | {nll.mean():.3f}{sd(nll)} [{a:.3f}, {b:.3f}] | {np.median(nll.mean(0)):.3f} | {mae.mean():.3f}{sd(mae)} [{c:.3f}, {d:.3f}] | {np.mean(ks):.3f}{ks_sd} |"))
            L += [r for _, r in sorted(rows)] + [""]
    open(f"{OUT}/T4_LEADERBOARD.md", "w").write("\n".join(L) + "\n"); print("\n".join(L))


if __name__ == "__main__":
    {"floors": floors, "board": board}[sys.argv[1]]()
