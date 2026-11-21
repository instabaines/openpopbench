"""Decision-time check (AI review 2026-11-20, point 1). Rescores the Task 1 heuristics and the learned feature ranker with
the history cut at a decision time tau instead of the target's own time t.

    python decision_time_check.py TIER DECISION OUTDIR DEVICE [ranker]
      TIER      bluesky_S | bluesky_Mid | reddit_S | reddit_Mid  (tables and query rows from openpopbench.data.path)
      DECISION  event: tau = t (the benchmark's definition; must reproduce the released floor ranks exactly)
                prev:  tau = time of the previous event of the same cascade (position p-1; the origin for p = 1)

Same query rows as the released floors file. What changes with tau: recency (latest participation before tau), affinity
(earlier participations with the origin before tau), exposure (edges created before tau from the cascade's earlier
participants), and every ranker feature (bench_state features evaluated at tau). The cascade prefix is the same (all
its events precede the target by position). Popularity (train-split counts) does not change. The candidate set is kept
identical to the benchmark's (observable before t, minus earlier participants) so that ranks stay comparable with the
published methods; the share of targets first seen in [tau, t) is reported.
The ranker (`ranker` flag) is retrained with features at tau (training queries also use their own tau), with the
released configuration (hidden 32, lr 1e-2, full softmax, 20,000 training events, 3 epochs, seeds 1-3).
Writes OUTDIR/<TIER>_<DECISION>.parquet (one row per query, ranks) and OUTDIR/<TIER>_<DECISION>.json (summary).
"""
import json
import os
import sys
import time
import numpy as np
import pandas as pd
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE); sys.path.insert(0, os.path.dirname(HERE))
from bench_state import Stream, FEATURES  # noqa: E402
from openpopbench.data import path as data_path  # noqa: E402  (OPB_DATA release, or the maintainers' tree)

TIER, DEC, OUT, DEV = sys.argv[1:5]
RANKER = len(sys.argv) > 5 and sys.argv[5] == "ranker"
dev = torch.device(DEV); os.makedirs(OUT, exist_ok=True)
log = lambda s: print(f"[dtc {time.strftime('%H:%M:%S')}] {s}", flush=True)
B = os.path.dirname(data_path(TIER, "events")); FL = data_path(TIER, "queries")


class StreamD(Stream):
    def query_at(self, j, split_start, tau):
        """bench_state.Stream.query with every history cut at tau (<= t); candidates as the benchmark (first_seen < t)."""
        ev, dev, N = self.ev, self.dev, self.N
        t, c, y, pos, o = int(ev["t"][j]), ev["cascade_id"][j], int(ev["participant_id"][j]), ev["position"][j], ev["origin_id"][j]
        self.advance(tau)
        ix = self.order[self.cstart[c]:self.cstart[c + 1]]; pre = ix[ev["position"][ix] < pos]
        prev = torch.from_numpy(ev["participant_id"][pre].astype(np.int64)).to(dev)
        prev_t = torch.from_numpy(ev["t"][pre].astype(np.float64)).to(dev)
        cand = self.elig_g & (self.first_g < t); cand[prev] = False; cand[y] = True
        a, b = self.astart[o], self.astart[o + 1]; b = a + np.searchsorted(self.a_t[a:b], tau, side="left")
        aff = torch.zeros(N, device=dev, dtype=torch.float64)
        aff.index_add_(0, torch.from_numpy(self.a_p[a:b].astype(np.int64)).to(dev), torch.ones(b - a, device=dev, dtype=torch.float64))
        s, e = self.rowptr[prev], self.rowptr[prev + 1]; ln = e - s; tot = int(ln.sum())
        ex, e1h, e1d = (torch.zeros(N, device=dev, dtype=torch.float64) for _ in range(3))
        if tot:
            base = torch.repeat_interleave(s - torch.cumsum(ln, 0) + ln, ln) + torch.arange(tot, device=dev)
            src_t = torch.repeat_interleave(prev_t, ln)
            keep = self.g_t[base] < tau; d = self.g_dst[base[keep]].long(); dt = (tau - src_t[keep])
            ex.index_add_(0, d, torch.ones(len(d), device=dev, dtype=torch.float64))
            e1h.index_add_(0, d, torch.exp(-dt / 3600.0)); e1d.index_add_(0, d, torch.exp(-dt / 86400.0))
        dout, din = self.degrees(split_start)
        never = (self.last < 0).double()
        rec = torch.where(self.last < 0, torch.full_like(self.last, 12.0), torch.log1p((tau - self.last).clamp(min=0) / 3600.0))
        X = torch.stack([torch.log1p(self.pop), rec, never, torch.log1p(aff), torch.log1p(ex), e1h, e1d, torch.log1p(dout), torch.log1p(din)], 1).float()
        return y, cand, X, prev


st = StreamD(B, dev)
ev = st.ev; N = st.N
sp, tt = ev["split"], ev["t"]
SPLIT_START = {k: int(tt[sp == k].min()) for k in (0, 1, 2)}


def taus(rows):
    out = np.empty(len(rows), np.int64)
    for n, j in enumerate(rows):
        if DEC == "event": out[n] = tt[j]; continue
        c, pos = ev["cascade_id"][j], ev["position"][j]
        ix = st.order[st.cstart[c]:st.cstart[c + 1]]
        out[n] = tt[ix[ev["position"][ix] == pos - 1][0]]
    return out


fl = pd.read_parquet(FL).sort_values("event_row").reset_index(drop=True)
rows = fl.event_row.values; tau = taus(rows)
assert (tau <= tt[rows]).all()
order = np.lexsort((rows, tau))          # process in decision-time order (the stream state only moves forward)
log(f"{TIER} {DEC}: {len(rows):,} queries; B={B}")

# ---- heuristics, exactly as floors_eval.py with t -> tau (own state: `last` over all events, train-split popularity)
non0 = ev["position"] > 0
pop = torch.from_numpy(np.bincount(ev["participant_id"][(sp == 0) & non0], minlength=N).astype(np.float64)).to(dev)
tieb = pop / (pop.max() + 1)
last = torch.full((N,), -1.0, device=dev, dtype=torch.float64); ptr = 0
ev_p = torch.from_numpy(ev["participant_id"].astype(np.int64)).to(dev); ev_t = torch.from_numpy(tt.astype(np.float64)).to(dev)
NAMES = ("popularity", "recency", "affinity", "exposure")
R = {m: np.zeros(len(rows)) for m in NAMES}; ncand = np.zeros(len(rows), np.int64); late = np.zeros(len(rows), bool)
t0 = time.time()
with torch.no_grad():
    for k, n in enumerate(order):
        j, ta = rows[n], int(tau[n])
        nxt = int(np.searchsorted(tt, ta, side="left"))
        if nxt > ptr: last.scatter_reduce_(0, ev_p[ptr:nxt], ev_t[ptr:nxt], "amax"); ptr = nxt
        st.advance(ta)
        y, cand, X, prev = st.query_at(j, SPLIT_START[int(sp[j])], ta)
        o = ev["origin_id"][j]; a, b = st.astart[o], st.astart[o + 1]; b = a + np.searchsorted(st.a_t[a:b], ta, side="left")
        aff = tieb.clone(); aff.index_add_(0, torch.from_numpy(st.a_p[a:b].astype(np.int64)).to(dev), torch.ones(b - a, device=dev, dtype=torch.float64))
        ex = torch.expm1(X[:, 4].double()).round()          # exposure count before tau (float32 log1p round trip)
        ncand[n] = int(cand.sum()); late[n] = st.first[y] >= ta
        for m, sc in zip(NAMES, (pop, last + tieb * 1e-3, aff, ex + tieb)):
            s_ = sc[cand]; v = sc[y]
            R[m][n] = 1 + float((s_ > v).sum()) + 0.5 * (float((s_ == v).sum()) - 1)
        if k % 10000 == 0: log(f"heuristics {k:,}/{len(rows):,} ({time.time() - t0:.0f} s)")

res = fl[["event_row", "cascade_id", "participant_id", "t", "split", "position", "in_all_sample", "in_novel_sample", "novel", "n_candidates"]].copy()
res["tau"] = tau; res["first_seen_after_tau"] = late
for m in NAMES: res[f"rank_{m}"] = R[m]
assert (ncand == fl.n_candidates.values).all(), "candidate sets differ from the benchmark"
if DEC == "event":   # reproduction check against the released floors
    for m in NAMES:
        d = np.abs(R[m] - fl[f"rank_{m}"].values)
        log(f"reproduce {m}: max |diff| {d.max():.3g}, rows differing {(d > 0.5).sum()}")
        res[f"released_rank_{m}"] = fl[f"rank_{m}"].values


# ---- learned feature ranker (retrained at tau)
class Scorer(torch.nn.Module):
    def __init__(self, hidden, mu, sd):
        super().__init__(); F = len(FEATURES)
        self.register_buffer("mu", mu); self.register_buffer("sd", sd)
        self.f = torch.nn.Linear(F, 1) if hidden == 0 else torch.nn.Sequential(torch.nn.Linear(F, hidden), torch.nn.ReLU(), torch.nn.Linear(hidden, 1))

    def forward(self, X): return self.f((X - self.mu) / self.sd).squeeze(-1)


def train_rows(seed):
    rng = np.random.default_rng(seed)
    ok = np.flatnonzero((sp == 0) & st.non0 & st.elig[ev["participant_id"]] & (st.first[ev["participant_id"]] < tt))
    return np.sort(rng.choice(ok, min(20000, len(ok)), replace=False))


def fit_full(seed, hidden=32, lr=1e-2, epochs=3, qbatch=32):
    torch.manual_seed(seed); qi = train_rows(seed); qt = taus(qi); od = np.lexsort((qi, qt)); qi, qt = qi[od], qt[od]
    st.reset(); acc = []
    sub = np.arange(0, len(qi), max(1, len(qi) // 300))
    for n in sub:
        y, cand, X, _ = st.query_at(qi[n], SPLIT_START[0], int(qt[n])); ci = torch.nonzero(cand).squeeze(1)
        acc.append(X[ci[torch.randint(len(ci), (min(len(ci), 4096),), device=dev)]])
    allx = torch.cat(acc); mu, sd = allx.mean(0), allx.std(0).clamp(min=1e-3)
    m = Scorer(hidden, mu, sd).to(dev); opt = torch.optim.Adam(m.parameters(), lr=lr)
    for ep in range(epochs):
        st.reset(); tot = 0.0; opt.zero_grad()
        for n in range(len(qi)):
            y, cand, X, _ = st.query_at(qi[n], SPLIT_START[0], int(qt[n]))
            s_ = m(X); loss = torch.logsumexp(s_[cand], 0) - s_[y]
            (loss / qbatch).backward(); tot += float(loss.detach())
            if (n + 1) % qbatch == 0: opt.step(); opt.zero_grad()
        opt.step(); opt.zero_grad()
        log(f"ranker seed {seed} epoch {ep} full-softmax NLL {tot / len(qi):.4f}")
    return m.eval()


if RANKER:
    for seed in (1, 2, 3):
        m = fit_full(seed); st.reset(); r = np.zeros(len(rows))
        with torch.no_grad():
            for k, n in enumerate(order):
                y, cand, X, _ = st.query_at(rows[n], SPLIT_START[int(sp[rows[n]])], int(tau[n]))
                s = m(X); v = s[y]; sc = s[cand]
                r[n] = 1 + float((sc > v).sum()) + 0.5 * (float((sc == v).sum()) - 1)
        res[f"rank_b1_seed{seed}"] = r
        log(f"ranker seed {seed} evaluated")

res.to_parquet(f"{OUT}/{TIER}_{DEC}.parquet", index=False)
summ = {}
for spl in ("val", "test"):
    for sl, col in (("all", "in_all_sample"), ("novel", "in_novel_sample")):
        msk = (res.split.values == spl) & res[col].values
        cols = [f"rank_{m}" for m in NAMES] + ([f"rank_b1_seed{s}" for s in (1, 2, 3)] if RANKER else [])
        summ[f"{spl}/{sl}"] = {c: float((res[c].values[msk] <= 100).mean()) for c in cols}
        summ[f"{spl}/{sl}"]["n"] = int(msk.sum())
summ["share_first_seen_after_tau_test_all"] = float(late[(res.split.values == "test") & res.in_all_sample.values].mean())
json.dump(summ, open(f"{OUT}/{TIER}_{DEC}.json", "w"), indent=1)
log(json.dumps(summ)); log("DONE")
