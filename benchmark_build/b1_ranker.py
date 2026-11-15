"""B1: learned ranker over time-respecting floor features; B6: exposure Hawkes (paper, Appendix B). MODEL=b1|b6 (env).

B6 intensity: lambda_u(t) = exp(theta . [pop, never, dout, din] + c) + a1 exp1h + a2 exp1d + a3 exp + a4 aff + a5 self1h + a6 self1d
(raw counts; a_k = softplus >= 0): a baseline rate plus exposure excitation with exponential kernels (1 h, 1 d, no decay),
pair excitation (earlier participations with the origin) and self-excitation from the candidate's own participations
(1 h, 1 d kernels; added 2026-10-13 after the first B6 tuning omitted it, selected on validation only). The next participant is ranked by lambda_u, i.e. the mark distribution lambda_u / sum_v lambda_v.

    python b1_ranker.py tune  BENCH FLOORS.parquet OUTDIR DEVICE          8 configs on validation; writes OUTDIR/b1_config.json
    python b1_ranker.py final BENCH FLOORS.parquet OUTDIR DEVICE CONFIG   3 seeds; writes OUTDIR/b1_seed<s>.parquet (val + test ranks)

Training: NTRAIN train-split events (eligible, non-origin), sampled uniformly. neg="full" (paper, Appendix B.2,
2026-10-14; the configuration used for the leaderboard): the exact softmax cross-entropy of the target over the FULL
candidate set at each query (the evaluation objective), streamed in time order, 32 queries per step, 3 epochs.
neg="uniform"/"hard" (superseded, kept for the sampling-bias appendix): target vs NNEG sampled candidates (uniform, or
7/8 uniform + 1/8 hardest by affinity + exposure). Evaluation: the same query rows as FLOORS.parquet (floors_eval.py),
scored over the FULL candidate set. Selection metric (pre-registered): mean validation MRR over the all and novel slices.
"""
import json
import os
import sys
import time
import numpy as np
import pandas as pd
import torch
from bench_state import Stream, FEATURES

MODE, B, FL, OUT, DEV = sys.argv[1:6]
dev = torch.device(DEV); os.makedirs(OUT, exist_ok=True)
NTRAIN, NNEG = 20000, 2048
MODEL = os.environ.get("MODEL", "b1")
NEG = os.environ.get("NEG", "full")
if NEG == "full":
    CONFIGS = ([dict(hidden=h, lr=lr, neg="full") for h in (0, 32) for lr in (1e-2, 1e-3)] if MODEL == "b1"
               else [dict(hidden=-1, lr=lr, neg="full") for lr in (1e-2, 1e-3)])
else:
    CONFIGS = ([dict(hidden=h, lr=lr, neg=n) for h in (0, 32) for lr in (1e-2, 1e-3) for n in ("uniform", "hard")] if MODEL == "b1"
               else [dict(hidden=-1, lr=lr, neg=n) for lr in (1e-2, 1e-3) for n in ("uniform", "hard")])
TAG = "" if NEG == "full" else "_sampled"
log = lambda s: print(f"[b1 {time.strftime('%H:%M:%S')}] {s}", flush=True)
SPLIT_START = None


def split_starts(st):
    sp, t = st.ev["split"], st.ev["t"]
    return {k: int(t[sp == k].min()) for k in (0, 1, 2)}


def collect(st, neg_mode, seed):
    path = f"{OUT}/train_{neg_mode}_s{seed}{'_self' if st.scales else ''}.pt"
    if os.path.exists(path): return torch.load(path)
    rng = np.random.default_rng(seed); ev = st.ev
    ok = np.flatnonzero((ev["split"] == 0) & st.non0 & st.elig[ev["participant_id"]] & (st.first[ev["participant_id"]] < ev["t"]))
    qi = np.sort(rng.choice(ok, min(NTRAIN, len(ok)), replace=False)); st.reset()
    gen = torch.Generator(device=dev); gen.manual_seed(seed)
    Xp, Xn = [], []
    for n, j in enumerate(qi):
        y, cand, X = st.query(j, SPLIT_START[0])
        cand[y] = False; ci = torch.nonzero(cand).squeeze(1)
        if neg_mode == "uniform":
            neg = ci[torch.randint(len(ci), (NNEG,), device=dev, generator=gen)]
        else:
            k = NNEG // 8; hard = ci[torch.topk(X[ci, 3] + X[ci, 4], min(k, len(ci))).indices]
            neg = torch.cat([hard, ci[torch.randint(len(ci), (NNEG - len(hard),), device=dev, generator=gen)]])
        Xp.append(X[y].cpu()); Xn.append(X[neg].cpu())
        if n % 5000 == 0: log(f"collect {neg_mode} {n:,}/{len(qi):,}")
    d = dict(Xp=torch.stack(Xp), Xn=torch.stack(Xn)); torch.save(d, path); return d


class Scorer(torch.nn.Module):
    def __init__(self, hidden, mu, sd):
        super().__init__(); F = len(FEATURES)
        self.register_buffer("mu", mu); self.register_buffer("sd", sd)
        self.f = torch.nn.Linear(F, 1) if hidden == 0 else torch.nn.Sequential(torch.nn.Linear(F, hidden), torch.nn.ReLU(), torch.nn.Linear(hidden, 1))

    def forward(self, X): return self.f((X - self.mu) / self.sd).squeeze(-1)


class Hawkes(torch.nn.Module):
    """log lambda_u = log(exp(baseline) + sum_k a_k * excitation_k)."""
    BASE = [0, 2, 7, 8]          # pop, never, dout, din; excitation: exp1h, exp1d, exp, aff, self1h, self1d

    def __init__(self, mu, sd):
        super().__init__()
        self.register_buffer("mu", mu[self.BASE]); self.register_buffer("sd", sd[self.BASE])
        self.base = torch.nn.Linear(len(self.BASE), 1); self.a = torch.nn.Parameter(torch.full((6,), -2.0))

    def forward(self, X):
        b = self.base((X[..., self.BASE] - self.mu) / self.sd).squeeze(-1)
        E = torch.stack([X[..., 5], X[..., 6], torch.expm1(X[..., 4]), torch.expm1(X[..., 3]), X[..., 9], X[..., 10]], -1)
        exc = (E * torch.nn.functional.softplus(self.a)).sum(-1)
        return torch.logaddexp(b, torch.log(exc.clamp(min=1e-30)))


def fit(data, cfg, seed):
    torch.manual_seed(seed)
    Xp, Xn = data["Xp"].to(dev), data["Xn"].to(dev)
    allx = torch.cat([Xp, Xn.reshape(-1, Xn.shape[-1])[::64]]); mu, sd = allx.mean(0), allx.std(0).clamp(min=1e-3)
    m = (Scorer(cfg["hidden"], mu, sd) if cfg["hidden"] >= 0 else Hawkes(mu, sd)).to(dev); opt = torch.optim.Adam(m.parameters(), lr=cfg["lr"])
    n = len(Xp); idx = torch.arange(n, device=dev)
    for ep in range(5):
        perm = idx[torch.randperm(n, device=dev)]; tot = 0.0
        for b in range(0, n, 256):
            k = perm[b:b + 256]
            logits = torch.cat([m(Xp[k]).unsqueeze(1), m(Xn[k])], 1)
            loss = torch.nn.functional.cross_entropy(logits, torch.zeros(len(k), dtype=torch.long, device=dev))
            opt.zero_grad(); loss.backward(); opt.step(); tot += float(loss) * len(k)
        log(f"fit {cfg} seed {seed} epoch {ep} loss {tot / n:.4f}")
    return m.eval()


def train_rows(st, seed):
    rng = np.random.default_rng(seed); ev = st.ev
    ok = np.flatnonzero((ev["split"] == 0) & st.non0 & st.elig[ev["participant_id"]] & (st.first[ev["participant_id"]] < ev["t"]))
    return np.sort(rng.choice(ok, min(NTRAIN, len(ok)), replace=False))


def fit_full(st, cfg, seed, epochs=3, qbatch=32):
    """Exact full-candidate softmax training (paper, Appendix B.2)."""
    torch.manual_seed(seed); qi = train_rows(st, seed)
    st.reset(); acc = []
    for j in qi[:: max(1, len(qi) // 300)]:                       # feature standardization from ~300 training queries
        y, cand, X = st.query(j, SPLIT_START[0]); ci = torch.nonzero(cand).squeeze(1)
        acc.append(X[ci[torch.randint(len(ci), (min(len(ci), 4096),), device=dev)]])
    allx = torch.cat(acc); mu, sd = allx.mean(0), allx.std(0).clamp(min=1e-3)
    m = (Scorer(cfg["hidden"], mu, sd) if cfg["hidden"] >= 0 else Hawkes(mu, sd)).to(dev); opt = torch.optim.Adam(m.parameters(), lr=cfg["lr"])
    for ep in range(epochs):
        st.reset(); tot = 0.0; opt.zero_grad()
        for n, j in enumerate(qi):
            y, cand, X = st.query(j, SPLIT_START[0])
            s_ = m(X); loss = torch.logsumexp(s_[cand], 0) - s_[y]
            (loss / qbatch).backward(); tot += float(loss.detach())
            if (n + 1) % qbatch == 0: opt.step(); opt.zero_grad()
        opt.step(); opt.zero_grad()
        log(f"fit_full {cfg} seed {seed} epoch {ep} mean full-softmax NLL {tot / len(qi):.4f}")
    return m.eval()


def get_model(st, cfg, seed):
    return fit_full(st, cfg, seed) if cfg["neg"] == "full" else fit(collect(st, cfg["neg"], seed), cfg, seed)


@torch.no_grad()
def evaluate(st, models, rows):
    """models: {name: Scorer}; rows: query event rows in time order. Returns {name: ranks}."""
    st.reset(); R = {k: np.zeros(len(rows)) for k in models}; t0 = time.time()
    for n, j in enumerate(rows):
        y, cand, X = st.query(j, SPLIT_START[int(st.ev["split"][j])])
        for k, m in models.items():
            s = m(X); v = s[y]; sc = s[cand]
            R[k][n] = 1 + float((sc > v).sum()) + 0.5 * (float((sc == v).sum()) - 1)
        if n % 5000 == 0: log(f"eval {n:,}/{len(rows):,} ({time.time() - t0:.0f} s)")
    return R


def mrr_sel(meta, r):
    v = meta.split.values == "val"
    return float(np.mean([np.mean(1 / r[v & meta[c].values]) for c in ("in_all_sample", "in_novel_sample")]))


st = Stream(B, dev, self_exc_scales=(3600.0, 86400.0) if MODEL == "b6" else ()); SPLIT_START = split_starts(st)
fl = pd.read_parquet(FL); fl = fl.sort_values("event_row").reset_index(drop=True)
assert (np.diff(st.ev["t"][fl.event_row.values]) >= 0).all(), "query rows must be in time order"
if MODE == "t2a":
    # T2a (2026-10-28): P(next participant is new) = the softmax mass the model puts on candidates with no event in any
    # TRAIN cascade. Same configuration and seeds as `final`; all-events-sample rows only.
    #   python b1_ranker.py t2a BENCH FLOORS.parquet OUTDIR DEVICE CONFIG     writes OUTDIR/t2a_<model>_seed<s>.parquet
    cfg = json.load(open(sys.argv[6]))["best"]; meta = fl[fl.in_all_sample].reset_index(drop=True); rows = meta.event_row.values
    newm = torch.ones(st.N, dtype=torch.bool, device=dev); newm[torch.from_numpy(st.ev["participant_id"][st.ev["split"] == 0].astype(np.int64)).to(dev)] = False
    models = {s: get_model(st, cfg, s) for s in (1, 2, 3)}
    P = {s: np.zeros(len(rows)) for s in models}; st.reset()
    with torch.no_grad():
        for n, j in enumerate(rows):
            y, cand, X = st.query(j, SPLIT_START[int(st.ev["split"][j])])
            for s, m in models.items():
                lp = torch.log_softmax(m(X)[cand].double(), 0); P[s][n] = float(torch.exp(lp[newm[cand]]).sum())
            if n % 5000 == 0: log(f"t2a {n:,}/{len(rows):,}")
    for s in models:
        o = meta[["event_row", "cascade_id", "split"]].copy(); o["p_new"] = P[s]; o.to_parquet(f"{OUT}/t2a_{MODEL}_seed{s}.parquet", index=False)
    log("T2A DONE"); sys.exit(0)
if MODE == "tune":
    meta = fl[fl.split == "val"].reset_index(drop=True); rows = meta.event_row.values
    models = {}
    for i, cfg in enumerate(CONFIGS):
        models[i] = get_model(st, cfg, 0)
    R = evaluate(st, models, rows)
    res = [dict(cfg=CONFIGS[i], val_sel=mrr_sel(meta, R[i]), val_hits100_all=float((R[i][meta.in_all_sample.values] <= 100).mean())) for i in models]
    best = max(res, key=lambda r: r["val_sel"])
    json.dump(dict(results=res, best=best["cfg"], rule="max mean val MRR over all and novel slices"), open(f"{OUT}/{MODEL}{TAG}_config.json", "w"), indent=1)
    log(f"tuning results: {json.dumps(res)}"); log(f"best {best}")
else:
    cfg = json.load(open(sys.argv[6]))["best"]; rows = fl.event_row.values
    models = {s: get_model(st, cfg, s) for s in (1, 2, 3)}
    R = evaluate(st, models, rows)
    for s in models:
        out = fl[["event_row", "cascade_id", "participant_id", "t", "split", "position", "in_all_sample", "in_novel_sample", "novel", "n_candidates"]].copy()
        out[f"rank_{MODEL}{TAG}"] = R[s]; out.to_parquet(f"{OUT}/{MODEL}{TAG}_seed{s}.parquet", index=False)
    if MODEL == "b6": json.dump({s: dict(a=torch.nn.functional.softplus(m.a).tolist(), base=m.base.weight.tolist()) for s, m in models.items()}, open(f"{OUT}/b6{TAG}_params.json", "w"), indent=1)
    json.dump(dict(config=cfg, seeds=[1, 2, 3], ntrain=NTRAIN, nneg=NNEG if NEG != "full" else "full"), open(f"{OUT}/{MODEL}{TAG}_final.json", "w"), indent=1)
    log("DONE")
