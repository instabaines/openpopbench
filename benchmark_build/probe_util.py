"""Feasibility probe shared by the method adapters (paper, Appendix D: infeasibility is measured, not assumed).

probe(step_fn, n_train_seqs, batch0, out_json, **meta): for batch sizes batch0, batch0/2, ..., 1, time `step_fn(bs)` (one
full training step: forward, backward, optimizer) for 2 warm-up plus 8 timed steps. Record peak GPU memory, seconds per step,
steps per epoch (n_train_seqs / bs) and the extrapolated hours per epoch. Stop at the largest batch that fits; record each
out-of-memory attempt. Writes out_json, compares with the 8 GPU-h seed budget, and exits the process.
"""
import json
import sys
import time

import torch


def probe(step_fn, n_train_seqs, batch0, out_json, budget_h=8.0, **meta):
    res = dict(meta, n_train_seqs=int(n_train_seqs), attempts=[], gpu=torch.cuda.get_device_name())
    bs = int(batch0)
    while bs >= 1:
        try:
            torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
            for _ in range(2): step_fn(bs)
            torch.cuda.synchronize(); t = time.time()
            for _ in range(8): step_fn(bs)
            torch.cuda.synchronize(); sec = (time.time() - t) / 8
            steps = -(-n_train_seqs // bs); hours = sec * steps / 3600
            res["attempts"].append(dict(batch=bs, ok=True, sec_per_step=round(sec, 4), peak_mem_gb=round(torch.cuda.max_memory_allocated() / 2**30, 2)))
            res.update(feasible_batch=bs, sec_per_step=round(sec, 4), steps_per_epoch=int(steps), est_hours_per_epoch=round(hours, 2),
                       epochs_within_budget=round(budget_h / hours, 3) if hours > 0 else None,
                       verdict=("fits: >= 1 epoch per seed budget" if hours <= budget_h else f"over budget: {hours / budget_h:.1f}x the {budget_h:.0f} GPU-h seed budget per epoch"))
            break
        except torch.cuda.OutOfMemoryError as e:
            res["attempts"].append(dict(batch=bs, ok=False, error="CUDA out of memory", detail=str(e)[:200]))
            torch.cuda.empty_cache(); bs //= 2
    else:
        res.update(feasible_batch=None, verdict="infeasible: out of memory even at batch 1")
    json.dump(res, open(out_json, "w"), indent=1)
    print("[probe]", json.dumps(res), flush=True)
    sys.exit(0)
