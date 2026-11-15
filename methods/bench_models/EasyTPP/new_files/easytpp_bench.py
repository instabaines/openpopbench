"""F3 point-process methods (RMTPP, S2P2, WSMTHP, AttNHP) through EasyTPP 0.3.0 (official library) on benchmark tables.

    PYTHONPATH=f1_bench/pydeps_easytpp model-env/bin/python easytpp_bench.py MODEL BENCH FLOORS.parquet OUTDIR SEED [BUDGET_HOURS=8]
    BATCH=b    training batch size (default: the released experiment config's); PROBE=out.json: feasibility probe, then exit

Released pieces: the model classes, the TPPRunner training loop (log-likelihood loss, validation log-likelihood model
selection, saving the best checkpoint) and the per-model `<MODEL>_train` block of EasyTPP/examples/configs/
experiment_config.yaml (architecture, optimiser, learning rate, epochs). data_specs.rescale_time = true (EasyTPP's option
"recommended when raw inter-event times are large": times are divided by the mean training inter-event time).
Mapping: a cascade is an event sequence; mark = participant id (num_event_types = N, pad = N); time = seconds since the
cascade's first event. Changes (BENCH_CHANGES.md): (1) sequences are cut into consecutive chunks of at most 64 events
(time still measured from the cascade start; each chunk's first inter-event time is 0, as EasyTPP's format requires),
because the non-event integral costs batch x length x MC samples x N marks; (2) the 'test' set handed to the runner is
500 validation sequences (the runner evaluates it every epoch; benchmark test data is never seen in training);
(3) the training time budget; (4) BATCH from the feasibility probe when the released batch does not fit; (5) the
validation side-metrics acc/RMSE (thinning sampler: 500 draws x N marks per event) are switched off; selection is unchanged.
Scoring: each floors query (cascade c, position p, time t) gets the released compute_intensities_at_sample_times on the
last <= 63 earlier events with the sample time at t (relative to the last event; absolute for AttNHP, whose API takes
absolute sample times), compute_last_step_only; candidates (floors_eval.py definition) are ranked by their mark
intensity lambda_k(t); ties count half; the candidate count must equal the floors' n_candidates.
"""
import copy
import json
import os
import sys
import time

import duckdb
import numpy as np
import pandas as pd
import torch
import yaml

M, B, FL, OUT, SEED = sys.argv[1:6]
SEED = int(SEED); BUDGET = float(sys.argv[6]) if len(sys.argv) > 6 else 8.0
HERE = os.path.dirname(os.path.abspath(__file__)); CHUNK = 64
from easy_tpp.config_factory import Config  # noqa: E402
from easy_tpp.runner import Runner  # noqa: E402
from easy_tpp.utils import RunnerPhase  # noqa: E402
from easy_tpp.runner.tpp_runner import TPPRunner  # noqa: E402

os.makedirs(OUT, exist_ok=True)
log = lambda s: print(f"[{M.lower()} {time.strftime('%H:%M:%S')}] {s}", flush=True)
con = duckdb.connect(); con.execute("SET threads=32; SET memory_limit='100GB'")
P = f"read_parquet('{B}/participants.parquet')"; cols = [r[0] for r in con.execute(f"DESCRIBE SELECT * FROM {P}").fetchall()]
ev = con.execute(f"""SELECT e.cascade_id, e.participant_id, e.t, e."position", (CASE c.split WHEN 'train' THEN 0 WHEN 'val' THEN 1 ELSE 2 END)::TINYINT split
                     FROM read_parquet('{B}/events.parquet') e JOIN read_parquet('{B}/cascades.parquet') c USING (cascade_id)
                     ORDER BY e.cascade_id, e."position" """).fetchnumpy()
pa_ = con.execute(f"SELECT coalesce(first_seen_t, 9e18)::BIGINT f{', is_crawled' if 'is_crawled' in cols else ''} FROM {P} ORDER BY participant_id").fetchnumpy()
N = len(pa_["f"])
cstart = np.searchsorted(ev["cascade_id"], np.arange(ev["cascade_id"].max() + 2)); csplit = ev["split"][cstart[:-1]]
pid = ev["participant_id"].astype(np.int64); tsec = ev["t"].astype(np.float64)

# ---- export in EasyTPP's json format
D = f"{OUT}/easytpp_data_{M}_seed{SEED}"; os.makedirs(D, exist_ok=True)


def seqs(cids):
    out = []
    for c in cids:
        a, b = cstart[c], cstart[c + 1]
        if b - a < 2: continue
        t = tsec[a:b] - tsec[a]; k = pid[a:b]
        for s in range(0, b - a, CHUNK):
            tt, kk = t[s:s + CHUNK], k[s:s + CHUNK]
            if len(tt) < 2: continue
            dt = np.diff(tt, prepend=tt[0])
            out.append(dict(dim_process=N, seq_idx=len(out), seq_len=len(tt), time_since_start=tt.tolist(), time_since_last_event=dt.tolist(), type_event=kk.tolist()))
    return out


rng = np.random.default_rng(SEED)
tr, va = seqs(np.flatnonzero(csplit == 0)), seqs(np.flatnonzero(csplit == 1))
for name, x in (("train", tr), ("dev", va), ("test", [va[i] for i in rng.choice(len(va), min(500, len(va)), replace=False)])):
    json.dump(x, open(f"{D}/{name}.json", "w"))
log(f"{N:,} marks; exported {len(tr):,} train / {len(va):,} validation sequences (chunks of <= {CHUNK})")

# ---- released experiment block, benchmark data
off = yaml.safe_load(open(f"{HERE}/examples/configs/experiment_config.yaml"))
exp = copy.deepcopy(off[f"{M}_train"])
exp["base_config"].update(dataset_id="bench", base_dir=f"{OUT}/easytpp_ckpt_{M}_seed{SEED}/")
exp["trainer_config"].update(seed=SEED, gpu=0 if torch.cuda.is_available() else -1)
if os.environ.get("BATCH"): exp["trainer_config"]["batch_size"] = int(os.environ["BATCH"])
if os.environ.get("MODEL_CFG"):   # tuning check (2026-11-13): JSON overrides of model_config (model_specs merged); unset = released
    import json as _j; _o = _j.loads(os.environ["MODEL_CFG"]); _ms = _o.pop("model_specs", {}); exp["model_config"].update(_o); exp["model_config"].setdefault("model_specs", {}).update(_ms)
if os.environ.get("MAX_EPOCHS"): exp["trainer_config"]["max_epoch"] = int(os.environ["MAX_EPOCHS"])   # one full epoch at scale (paper, Section 4.4)
cfg = {"pipeline_config_id": off.get("pipeline_config_id", "runner_config"),
       "data": {"bench": {"data_format": "json", "train_dir": f"{D}/train.json", "valid_dir": f"{D}/dev.json", "test_dir": f"{D}/test.json",
                          "data_specs": {"num_event_types": N, "pad_token_id": N, "padding_side": "right", "truncation_side": "right", "rescale_time": True}}},
       f"{M}_bench": exp}
yaml.safe_dump(cfg, open(f"{D}/config.yaml", "w"))
config = Config.build_from_yaml_file(f"{D}/config.yaml", experiment_id=f"{M}_bench")
runner = Runner.build_from_config(config)
model = runner.model; dev = next(model.parameters()).device
model.event_sampler = None     # change (5): no thinning-based acc/RMSE during validation (500 draws x N marks); selection is by loglike, as released
log(f"released config: {exp['trainer_config']} | {exp['model_config']}")

if os.environ.get("PROBE"):   # feasibility probe: real training steps through the released loss
    sys.path.insert(0, os.path.join(os.environ.get("OPB_ROOT", "."), "benchmark_build")); from probe_util import probe
    from easy_tpp.preprocess import TPPDataLoader
    loader_cache = {}
    opt = torch.optim.Adam(model.parameters(), lr=exp["trainer_config"]["learning_rate"])

    def _step(bs):
        if bs not in loader_cache:
            kw = {**config.trainer_config.get_yaml_config(), "batch_size": bs}
            loader_cache[bs] = iter(TPPDataLoader(data_config=config.data_config, backend=config.base_config.backend, **kw).train_loader())
        try: batch = next(loader_cache[bs])
        except StopIteration: loader_cache.pop(bs); return _step(bs)
        batch = {k: (v.to(dev) if torch.is_tensor(v) else v) for k, v in batch.items()}
        loss, _ = model.loglike_loss(batch=batch)
        opt.zero_grad(); loss.backward(); opt.step()
    probe(_step, len(tr), int(exp["trainer_config"]["batch_size"]), os.environ["PROBE"], method=M, bench=B, marks=N)

# ---- time budget on top of the released loop: once exceeded, remaining epochs are skipped
t0 = time.time(); hist = []; final_val = [False]
_orig = TPPRunner.run_one_epoch


class Budgeted:
    """The training loader, stopped mid-epoch once the time budget is spent."""
    def __init__(self, dl): self.dl = dl
    def __len__(self): return len(self.dl)
    def __getattr__(self, k): return getattr(self.dl, k)
    def __iter__(self):
        for b in self.dl:
            if (time.time() - t0) / 3600 > BUDGET: return
            yield b


class SplitBatches:
    """Change (6), 2026-10-26: validation batches cut into pieces of VAL_SPLIT sequences (same padding length). The released
    epoch loop sums loss and event counts over batches and divides once, so the validation log-likelihood is unchanged;
    only peak memory drops (the released validation intensity pass needs more memory than a training step)."""
    def __init__(self, dl, k): self.dl, self.k = dl, k
    def __len__(self): return len(self.dl)
    def __getattr__(self, a): return getattr(self.dl, a)
    def __iter__(self):
        for b in self.dl:
            n = next(iter(b.values())).shape[0]
            for i in range(0, n, self.k): yield type(b)({key: v[i:i + self.k] for key, v in b.items()})


def run_one_epoch(self, data_loader, phase):
    over = (time.time() - t0) / 3600 > BUDGET
    if over and phase == RunnerPhase.TRAIN: return {"loglike": float("nan"), "num_events": 0}
    if over and phase == RunnerPhase.VALIDATE:            # one validation pass after the budget runs out, then none
        if final_val[0]: return {"loglike": -float("inf"), "num_events": 0}
        final_val[0] = True
    if phase == RunnerPhase.TRAIN: data_loader = Budgeted(data_loader)
    if phase == RunnerPhase.VALIDATE and os.environ.get("VAL_SPLIT"): data_loader = SplitBatches(data_loader, int(os.environ["VAL_SPLIT"]))
    r = _orig(self, data_loader, phase); hist.append(dict(phase=str(phase), hours=(time.time() - t0) / 3600, **{k: float(v) for k, v in r.items() if np.isscalar(v)}))
    return r


TPPRunner.run_one_epoch = run_one_epoch
if os.environ.get("SCORE_ONLY"):   # change (7), 2026-10-27: score a saved best checkpoint (training already completed; scoring had failed)
    ckpt = os.environ["SCORE_ONLY"]; train_hours = float(os.environ.get("TRAIN_HOURS", "nan"))
    _old = f"{OUT}/{M.lower()}_seed{SEED}.json"   # keep the training history of the run being re-scored (2026-10-30)
    if os.path.exists(_old): hist.extend(h for h in json.load(open(_old)).get("hist", []))
    hist.append(dict(phase="score_only", ckpt=ckpt))
else:
    runner.run()
    train_hours = (time.time() - t0) / 3600
    ckpt = runner.runner_config.base_config.specs["saved_model_dir"]
model.load_state_dict(torch.load(ckpt, map_location=dev)); model.eval()
scale = float(runner._data_loader.time_scale) if getattr(runner._data_loader, "time_scale", None) else 1.0
log(f"trained {train_hours:.2f} h; best checkpoint {ckpt}; time scale {scale:.3f} s")

if os.environ.get("T4"):   # change (8), 2026-10-28: T4 event-time scoring of a saved checkpoint (use with SCORE_ONLY)
    # Query = an all-events-sample floors row (cascade c, position p). Target: the gap d = t_p - t_{p-1} (seconds). The
    # model's total intensity lam(tau) = sum over all marks of lambda_k(t_{p-1} + tau), from the same released
    # compute_intensities_at_sample_times and the same <= 63-event history as T1, on a log grid tau in [0, max(d, 30 days)]
    # with d inserted. Lam(tau) = trapezoid integral. Time NLL (nats, seconds) = -log lam(d) + Lam(d) + log(scale)
    # (rescale_time divides times by `scale`). PIT = 1 - exp(-Lam(d)). Median gap: Lam(tau) = ln 2 (linear interpolation);
    # set to the grid end and flagged when not reached.
    fl = pd.read_parquet(FL).sort_values("event_row").reset_index(drop=True); fl = fl[fl.in_all_sample].reset_index(drop=True)
    if os.environ.get("T4_LIMIT"): fl = fl.iloc[:int(os.environ["T4_LIMIT"])].reset_index(drop=True)   # smoke test only
    erow = con.execute(f"""SELECT cascade_id, "position" FROM read_parquet('{B}/events.parquet') ORDER BY t, cascade_id, "position" """).fetchnumpy()
    qc, qp = erow["cascade_id"][fl.event_row.values], erow["position"][fl.event_row.values]
    G = 96; HMAX = 30 * 86400.0; L_ = np.minimum(qp, CHUNK - 1)
    QB = int(os.environ.get("QB") or max(1, min(256, int(1.5e9 // (N * (G + 1) * 4)))))
    nll, pit, med, cap, gap = (np.zeros(len(fl)) for _ in range(5)); t1 = time.time()
    with torch.no_grad():
        for Lq in np.unique(L_):
            ids = np.flatnonzero(L_ == Lq)
            for b0 in range(0, len(ids), QB):
                bi = ids[b0:b0 + QB]; T = np.zeros((len(bi), Lq)); K = np.zeros((len(bi), Lq), np.int64); S = np.zeros((len(bi), Lq, G + 1)); di = np.zeros(len(bi), np.int64)
                for r, i in enumerate(bi):
                    s0 = cstart[qc[i]]; j = s0 + qp[i]; a = j - Lq; d = tsec[j] - tsec[j - 1]; gap[i] = d
                    T[r] = (tsec[a:j] - tsec[s0]) / scale; K[r] = pid[a:j]
                    g = np.expm1(np.log1p(max(d, HMAX)) * np.arange(G) / (G - 1)); g = np.sort(np.append(g, d)); di[r] = np.searchsorted(g, d)
                    S[r, -1] = g / scale + (T[r, -1] if M == "AttNHP" else 0.0)
                Tt = torch.tensor(T, dtype=torch.float32, device=dev); DT = torch.diff(Tt, dim=1, prepend=Tt[:, :1])
                lam = model.compute_intensities_at_sample_times(Tt, DT, torch.from_numpy(K).to(dev), torch.tensor(S, dtype=torch.float32, device=dev),
                                                                compute_last_step_only=True)[:, -1, :, :N].double().sum(-1).cpu().numpy()   # [b, G+1]
                if (b0 // QB) % 50 == 0: log(f"T4 progress: length {Lq}, batch {b0 // QB} ({time.time() - t1:.0f} s)")
                tau = S[:, -1] - (T[:, -1:] if M == "AttNHP" else 0.0)                                                               # scaled units
                Lam = np.concatenate([np.zeros((len(bi), 1)), np.cumsum(0.5 * (lam[:, 1:] + lam[:, :-1]) * np.diff(tau, axis=1), 1)], 1)
                for r, i in enumerate(bi):
                    k = di[r]; nll[i] = -np.log(max(lam[r, k], 1e-300)) + Lam[r, k] + np.log(scale); pit[i] = 1 - np.exp(-Lam[r, k])
                    h = np.searchsorted(Lam[r], np.log(2))
                    if h >= len(Lam[r]): med[i], cap[i] = tau[r, -1] * scale, 1
                    else:
                        x0, x1, y0, y1 = tau[r, h - 1], tau[r, h], Lam[r, h - 1], Lam[r, h]
                        med[i] = (x0 + (np.log(2) - y0) / max(y1 - y0, 1e-300) * (x1 - x0)) * scale
    log(f"T4 scored {len(fl):,} queries in {time.time() - t1:.0f} s")
    out = fl[["event_row", "cascade_id", "split", "position"]].copy()
    out["gap"], out["nll_time"], out["pit"], out["median_gap"], out["median_capped"] = gap, nll, pit, med, cap
    out.to_parquet(f"{OUT}/{M.lower()}_t4_seed{SEED}.parquet", index=False); log("T4 DONE"); sys.exit(0)

# ---- change (9), 2026-10-30: rank by log lambda_k(t), computed in float64, instead of lambda_k(t) in float32.
# Same quantity and same order; it only stops exact ties where float32 rounds small intensities to 0
# (AttNHP on Bluesky M: every intensity was 0, so all 500k candidates tied). LOGSCORE=0 restores the old scoring.
if os.environ.get("LOGSCORE", "1") == "1":
    import math as _m
    import types as _t
    from easy_tpp.model.baselayer import ScaledSoftplus as _SS

    def _log_softplus(z):                         # log(log1p(exp(z))) for float64 z; equals z to 1e-13 below z = -30
        return torch.where(z < -30, z, torch.log(torch.log1p(torch.exp(z.clamp(min=-30)))))

    def _ss_log(self, x):                         # ScaledSoftplus.forward (AttNHP, S2P2), in log space
        x = x.double(); beta = self.log_beta.double().exp(); bx = beta * x
        low = _log_softplus(bx.clamp(max=_m.log(1e5))) - torch.log(beta)
        return torch.where(bx <= self.threshold, low, torch.log(x.clamp(min=1e-300)))
    _SS.forward = _ss_log
    if M == "RMTPP":                              # released: exp(min(logit, log 1e5)) -> log = min(logit, log 1e5)
        def _rmtpp_log(self, right_hiddens_BNH, dts_BNG):
            p = self.hidden_to_intensity_logits(right_hiddens_BNH[..., None, :])
            return (p + self.w_t[None, None, :] * dts_BNG[..., None] + self.b_t[None, None, :]).double().clamp(max=_m.log(1e5))
        model.evolve_and_get_intentsity = _t.MethodType(_rmtpp_log, model)
    if M == "WSMTHP":                             # released nn.Softplus(beta=1, threshold=20)
        class _LogSP(torch.nn.Module):
            def forward(self, x):
                x = x.double(); return torch.where(x > 20, torch.log(x.clamp(min=1e-300)), _log_softplus(x))
        assert not getattr(model, "with_survival", False), "survival correction rescales by a per-row factor; check before log scoring"
        model.intensity_act = _LogSP()
    log("change (9): ranking by log intensity in float64")

# ---- full-candidate scoring of the floors queries
fl = pd.read_parquet(FL).sort_values("event_row").reset_index(drop=True)
erow = con.execute(f"""SELECT cascade_id, "position" FROM read_parquet('{B}/events.parquet') ORDER BY t, cascade_id, "position" """).fetchnumpy()
qc, qp = erow["cascade_id"][fl.event_row.values], erow["position"][fl.event_row.values]
first = torch.from_numpy(pa_["f"]).to(dev); elig = torch.from_numpy(pa_["is_crawled"].astype(bool) if "is_crawled" in pa_ else np.ones(N, bool)).to(dev)
L_ = np.minimum(qp, CHUNK - 1); Rk = np.zeros(len(fl)); nc = np.zeros(len(fl)); t1 = time.time()
_LS = os.environ.get("LOGSCORE", "1") == "1"   # change (9) scores in float64: twice the memory, so cap at QB 32 (the batch the 10-30 re-scoring used on every tier)
QB = int(os.environ.get("QB") or max(1, min(32 if _LS else 256, 5e8 // (N * (8 if _LS else 4)))))   # QB: scoring batch override (memory only)
with torch.no_grad():
    for Lq in np.unique(L_):                                  # equal-length batches: the last position is each row's last event
        ids = np.flatnonzero(L_ == Lq)
        for b0 in range(0, len(ids), QB):
            bi = ids[b0:b0 + QB]; T = np.zeros((len(bi), Lq)); K = np.zeros((len(bi), Lq), np.int64); S = np.zeros((len(bi), Lq, 1))
            for r, i in enumerate(bi):
                s0 = cstart[qc[i]]; j = s0 + qp[i]; a = j - Lq
                T[r] = (tsec[a:j] - tsec[s0]) / scale; K[r] = pid[a:j]
                S[r, -1, 0] = (tsec[j] - tsec[s0]) / scale if M == "AttNHP" else (tsec[j] - tsec[j - 1]) / scale
            Tt = torch.tensor(T, dtype=torch.float32, device=dev); DT = torch.diff(Tt, dim=1, prepend=Tt[:, :1])
            lam = model.compute_intensities_at_sample_times(Tt, DT, torch.from_numpy(K).to(dev), torch.tensor(S, dtype=torch.float32, device=dev),
                                                            compute_last_step_only=True)[:, -1, 0, :N]
            for r, i in enumerate(bi):
                s0 = cstart[qc[i]]; j = s0 + qp[i]; y = int(pid[j])
                cand = elig & (first < int(tsec[j])); cand[torch.from_numpy(pid[s0:j]).to(dev)] = False; cand[y] = True
                v = lam[r, y]; c = lam[r][cand]; Rk[i] = 1 + float((c > v).sum()) + 0.5 * (float((c == v).sum()) - 1); nc[i] = float(cand.sum())
assert np.array_equal(nc, fl.n_candidates.values), "candidate set differs from floors"
log(f"scored {len(fl):,} queries in {time.time() - t1:.0f} s")
key = M.lower()
out = fl[["event_row", "cascade_id", "participant_id", "t", "split", "position", "in_all_sample", "in_novel_sample", "novel", "n_candidates"]].copy()
out[f"rank_{key}"] = Rk; out.to_parquet(f"{OUT}/{key}_seed{SEED}.parquet", index=False)
json.dump(dict(seed=SEED, model=M, hist=hist, train_hours=train_hours, budget_h=BUDGET, time_scale=scale, config=exp), open(f"{OUT}/{key}_seed{SEED}.json", "w"), indent=1, default=str)
log("DONE")
