"""OpenPopBench command line.

    opb tiers                                       list tiers and where their files are
    opb methods                                     list registered methods
    opb run METHOD TIER --seed S [--budget H] [--gpu G] [--out DIR] [--score-only] [--dry-run]
                                                    train (or re-score) a registered method through its adapter
    opb collect METHOD TIER OUT --into SUB [--seeds 1 2 3]
                                                    convert adapter outputs to the submission format
    opb validate SUB                                check a submission folder
    opb evaluate SUB                                score it (writes SUB/results.json)
    opb leaderboard DIR [--tier T] [--out FILE]     render the leaderboard from every DIR/*/results.json (heuristic rows
                                                    from DIR/heuristics.json, or from the data)
    opb export TIER DEST                            write a tier in the release layout (maintainers)
    opb leakage cut TIER DEST [--tc mid]            build the cut world for the leakage test
    opb leakage test METHOD TIER --run DIR --cut DEST [--seed S] [--gpu G] [--repeat]
                                                    score a saved checkpoint on both worlds and compare
"""
import argparse, glob, json, os, shutil, subprocess, sys
import numpy as np, pandas as pd

from . import data, registry, submission, evaluate as ev


def cmd_tiers(a):
    for t in data.TIERS:
        role = "comparison" if t in data.COMPARISON else "scale"
        try: where = data.tables_dir(t)
        except FileNotFoundError: where = "(no data: set OPB_DATA)"
        print(f"{t:12s} {role:10s} {where}")


def cmd_methods(a):
    for k, m in registry.METHODS.items():
        print(f"{k:11s} {m['name']:24s} {m['family']:22s} queries={m['queries']:5s} tasks={','.join(m['tasks'])}"
              f"{'  score-only' if m.get('score_only') else ''}")


def cmd_run(a):
    m = registry.METHODS[a.method]
    q = data.path(a.tier, "queries_sub10" if m["queries"] == "sub10" else "queries")
    out = os.path.abspath(a.out or os.path.join("runs", a.tier, a.method)); os.makedirs(out, exist_ok=True)
    cmd = registry.render(a.method, data.tables_dir(a.tier), q, out, a.seed, a.budget)
    env = f"export CUDA_VISIBLE_DEVICES={a.gpu} OPB_ROOT={registry.NP}; "
    if a.score_only:
        so = m.get("score_only")
        if not so: sys.exit(f"{a.method} has no score-only mode; re-run training instead")
        ck = registry.checkpoint(a.method, out, a.seed)
        if not ck: sys.exit(f"no checkpoint for seed {a.seed} in {out} (expected {so['checkpoint']})")
        env += "export " + so["env"].format(CKPT=ck, TRAINLOG=os.path.join(out, "train.log")) + "; "
    if a.task3: env += "export T4=1; "
    full = env + cmd
    print(full)
    if a.dry_run: return
    log = os.path.join(out, f"run_seed{a.seed}{'_score' if a.score_only else ''}.log")
    rc = subprocess.run(["bash", "-c", full], stdout=open(log, "w"), stderr=subprocess.STDOUT).returncode
    print(f"rc {rc}; log {log}; output {registry.output_file(a.method, out, a.seed)}")
    sys.exit(rc)


def cmd_collect(a):
    m = registry.METHODS[a.method]; dst = os.path.join(a.into, a.tier); os.makedirs(dst, exist_ok=True); key = m["key"]
    seeds = a.seeds or sorted(int(f.rsplit("_seed", 1)[1][:-8]) for f in glob.glob(os.path.join(a.out, f"{key}_seed*.parquet")))
    for s in seeds:
        x = pd.read_parquet(os.path.join(a.out, f"{key}_seed{s}.parquet"), columns=["event_row", f"rank_{key}"])
        x.rename(columns={f"rank_{key}": "rank"}).to_parquet(os.path.join(dst, f"task1_seed{s}.parquet"), index=False)
        t4 = os.path.join(a.out, f"{key}_t4_seed{s}.parquet")
        if os.path.exists(t4):
            y = pd.read_parquet(t4, columns=["event_row", "nll_time", "pit", "median_gap"]).rename(columns={"nll_time": "nll"})
            y.to_parquet(os.path.join(dst, f"task3_seed{s}.parquet"), index=False)
        t2 = os.path.join(a.out, f"t2a_{key}_seed{s}.parquet")
        if os.path.exists(t2): pd.read_parquet(t2, columns=["event_row", "p_new"]).to_parquet(os.path.join(dst, f"task2a_seed{s}.parquet"), index=False)
    meta_p = os.path.join(a.into, "meta.json")
    if not os.path.exists(meta_p):
        json.dump(dict(name=m["name"], family=m["family"], code="https://github.com/instabaines/openpopbench",
                       queries=m["queries"], heuristic_inputs=m["heuristic_inputs"],
                       training="OpenPopBench v1.0 protocol: released hyperparameters, validation checkpoint selection, "
                                "8 GPU-h per seed on comparison tiers, 24 GPU-h on scale tiers", track="verified"),
                  open(meta_p, "w"), indent=1)
    print(f"collected seeds {seeds} into {dst}")


def cmd_validate(a):
    try:
        meta, found = submission.validate(a.sub)
    except submission.Invalid as e:
        sys.exit(f"INVALID: {e}")
    print(f"valid: {meta['name']} ({meta['track']}); " + "; ".join(f"{t}: {', '.join(v)}" for t, v in found.items()))


def cmd_evaluate(a):
    try:
        r = ev.evaluate(a.sub)
    except submission.Invalid as e:
        sys.exit(f"INVALID: {e}")
    for t, x in r["tiers"].items():
        if "task1" in x:
            h = x["task1"]["test/all"]["hits@100"]; n = x["task1"]["test/novel"]["hits@100"]
            print(f"{t:12s} Task 1 test Hits@100 all {h['mean']:.3f} no-tie {n['mean']:.3f}; "
                  f"Task 2b {x['task2b']['hits@100']['mean']:.3f}; gate {'pass' if x['sanity_gate']['passed'] else 'FAIL ' + str(x['sanity_gate']['below'])}")
    print(f"wrote {os.path.join(a.sub, 'results.json')}")


def _fmt(c):
    return "–" if c is None else f"{100 * c['mean']:.1f}" + (f" ± {100 * c['sd']:.1f}" if c.get("sd") else "")


def cmd_leaderboard(a):
    res = [json.load(open(f)) for f in sorted(glob.glob(os.path.join(a.dir, "*", "results.json")))]
    tiers = [a.tier] if a.tier else [t for t in data.TIERS if any(t in r["tiers"] for r in res)]
    L = [f"# OpenPopBench {data.VERSION} leaderboard", "",
         "Test Hits@100 (%), mean ± sd over seeds. Heuristic rows are deterministic. "
         "† fails the sanity gate, or is flagged by the maintainers (meta.json \"flags\", e.g. a failed reproduction gate). § scored on the 10% query subsample. ✓ verified (re-run by the maintainers).", ""]
    hp = os.path.join(a.dir, "heuristics.json")   # heuristic rows: shipped with each version, recomputed from the data if absent
    hj = json.load(open(hp)) if os.path.exists(hp) else {}
    for t in tiers:
        if t not in hj:
            q = pd.read_parquet(data.path(t, "queries")); nw = pd.read_parquet(data.path(t, "newcomer")); p = ev.newcomer_share(q, nw)
            hj[t] = {}
            for h in data.HEURISTICS:
                r1 = ev.task1(q, {0: q[f"rank_{h}"].to_numpy(float)})
                hj[t][h] = {"all": r1["test/all"]["hits@100"], "no-tie": r1["test/novel"]["hits@100"],
                            "task2b": ev.task2b(q, {0: q[f"rank_{h}"].to_numpy(float)}, nw, p)["hits@100"]}
            if a.save_heuristics: json.dump(hj, open(hp, "w"), indent=1)
        rows = [(h, None, hj[t][h]["all"], hj[t][h]["no-tie"], hj[t][h]["task2b"], "") for h in data.HEURISTICS]
        for r in res:
            x = r["tiers"].get(t)
            if not x or "task1" not in x: continue
            mk = ("†" if not x["sanity_gate"]["passed"] or t in r["meta"].get("flags", {}) else "") + ("§" if r["meta"].get("queries") == "sub10" else "") + ("✓" if r["meta"]["track"] == "verified" else "")
            rows.append((r["meta"]["name"], r["meta"]["family"], x["task1"]["test/all"]["hits@100"], x["task1"]["test/novel"]["hits@100"], x["task2b"]["hits@100"], mk))
        rows.sort(key=lambda z: -z[2]["mean"])
        L += [f"## {t}", "", "| method | family | all events | no-tie events | newcomers (Task 2b) |", "|---|---|---|---|---|"]
        L += [f"| {n}{mk} | {f or 'heuristic'} | {_fmt(h1)} | {_fmt(h2)} | {_fmt(nb)} |" for n, f, h1, h2, nb, mk in rows] + [""]
    txt = "\n".join(L) + "\n"
    if a.out: open(a.out, "w").write(txt)
    print(txt)


def cmd_export(a):
    rows = data.export({n: data.path(a.tier, n) for n in data.FILES}, os.path.join(a.dest, a.tier))
    for n, k in rows.items(): print(f"{n}: {k:,} rows")


def cmd_leakage(a):
    from . import leakage
    if a.action == "cut":
        print(json.dumps(leakage.cut(a.tier, a.dest, a.tc), indent=1))
    else:
        r = leakage.test(a.method, a.tier, a.run, a.cut, a.seed, a.gpu, a.repeat, a.train_log)
        print(json.dumps(r, indent=1)); sys.exit(0 if r["verdict"] == "PASS" else 1)


def main(argv=None):
    p = argparse.ArgumentParser(prog="opb", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    s = p.add_subparsers(dest="cmd", required=True)
    s.add_parser("tiers").set_defaults(f=cmd_tiers); s.add_parser("methods").set_defaults(f=cmd_methods)
    r = s.add_parser("run"); r.add_argument("method", choices=registry.METHODS); r.add_argument("tier", choices=data.TIERS)
    r.add_argument("--seed", type=int, default=1); r.add_argument("--budget", type=float, default=8); r.add_argument("--gpu", default="0")
    r.add_argument("--out"); r.add_argument("--score-only", action="store_true"); r.add_argument("--task3", action="store_true")
    r.add_argument("--dry-run", action="store_true"); r.set_defaults(f=cmd_run)
    c = s.add_parser("collect"); c.add_argument("method", choices=registry.METHODS); c.add_argument("tier", choices=data.TIERS)
    c.add_argument("out"); c.add_argument("--into", required=True); c.add_argument("--seeds", type=int, nargs="*"); c.set_defaults(f=cmd_collect)
    for n, f in (("validate", cmd_validate), ("evaluate", cmd_evaluate)):
        x = s.add_parser(n); x.add_argument("sub"); x.set_defaults(f=f)
    l = s.add_parser("leaderboard"); l.add_argument("dir"); l.add_argument("--tier"); l.add_argument("--out")
    l.add_argument("--save-heuristics", action="store_true", help="write DIR/heuristics.json (maintainers)"); l.set_defaults(f=cmd_leaderboard)
    e = s.add_parser("export"); e.add_argument("tier", choices=data.TIERS); e.add_argument("dest"); e.set_defaults(f=cmd_export)
    k = s.add_parser("leakage"); ks = k.add_subparsers(dest="action", required=True)
    kc = ks.add_parser("cut"); kc.add_argument("tier", choices=data.TIERS); kc.add_argument("dest"); kc.add_argument("--tc", default="mid")
    kt = ks.add_parser("test"); kt.add_argument("method", choices=registry.METHODS); kt.add_argument("tier", choices=data.TIERS)
    kt.add_argument("--run", required=True); kt.add_argument("--cut", required=True); kt.add_argument("--seed", type=int, default=1)
    kt.add_argument("--gpu", default="0"); kt.add_argument("--repeat", action="store_true"); kt.add_argument("--train-log")
    k.set_defaults(f=cmd_leakage)
    a = p.parse_args(argv)
    try:
        a.f(a)
    except FileNotFoundError as e:   # missing data or result files: one line, no traceback
        sys.exit(f"opb: {e}")


if __name__ == "__main__":
    main()
