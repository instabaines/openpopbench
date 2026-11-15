"""Check: the harness results reproduce the paper's leaderboards.

    python -m openpopbench.tests.check_against_paper [SUBDIR ...]

Compares each submission's results.json with the paper's values in leaderboard/v1.0/paper_values.json: every Task 1
metric on test (all events and no-tie events; mean and sd over seeds), Task 2b Hits@100, and Task 3 NLL and KS. Values
must agree to the printed precision (3 decimals). Without arguments it checks every folder of leaderboard/v1.0.

To check the evaluator itself rather than the shipped results.json, download the v1.0 result files (Hugging Face,
results/v1.0/<method>/), run `opb evaluate` on each folder, and pass the folders here. The folder name must be the
method key (`opb methods`).
"""
import glob, json, os, sys
from .. import registry

LB = os.path.join(registry.NP, "leaderboard", "v1.0")


def lookup(r, k):
    """Value of a paper key in a results.json tier entry: (mean, sd)."""
    if k.startswith("task1/"):
        _, split, sl, metric = k.split("/"); c = r["task1"][f"{split}/{sl}"][metric]
    elif k == "task2b/hits@100":
        c = r["task2b"]["hits@100"]
    else:
        c = r["task3"][k.split("/")[1]]
    return c["mean"], c.get("sd")


def main(subs):
    paper = json.load(open(os.path.join(LB, "paper_values.json")))
    subs = subs or sorted(os.path.dirname(f) for f in glob.glob(os.path.join(LB, "*", "results.json")))
    n = bad = 0
    for sub in subs:
        meth = os.path.basename(sub.rstrip("/")); res = json.load(open(os.path.join(sub, "results.json")))
        if meth not in paper: print(f"  {meth}: not in the paper's leaderboards"); continue
        for tier, vals in paper[meth].items():
            r = res["tiers"].get(tier)
            if r is None: print(f"  MISSING {meth} {tier}"); bad += 1; n += 1; continue
            for k, (m0, s0) in vals.items():
                m, s = lookup(r, k); n += 1
                ok = abs(round(m, 3) - m0) < 1.5e-3 and (s0 is None or (s is not None and abs(round(s, 3) - s0) < 1.5e-3))
                if not ok: bad += 1; print(f"  MISMATCH {meth} {tier} {k}: paper {m0} ± {s0}, harness {m:.4f} ± {s}")
    print(f"{n} values compared, {bad} mismatches")
    return 1 if bad or n == 0 else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
