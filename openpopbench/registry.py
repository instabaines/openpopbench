"""Registered methods: fifteen of the paper's sixteen published methods (all but TopoLSTM), MS-HGAT and DyHGCN, and the
two controls, each wrapped by an adapter.

Adapter contract (what a new method must implement to join the harness):

    <adapter> TABLES QUERIES OUT SEED BUDGET_H [extra args]

    TABLES   folder with events, cascades, edges, participants (data.tables_dir)
    QUERIES  the tier's queries.parquet (or queries_sub10.parquet for pairwise scorers)
    OUT      output folder
    SEED     integer seed
    BUDGET_H training budget in GPU-hours (8 on comparison tiers, 24 on scale tiers in v1.0)

The adapter trains on training cascades only, selects its checkpoint on validation queries only, and writes
OUT/<key>_seed<SEED>.parquet with event_row and rank_<key> for every row of QUERIES (validation and test), ranking the true
participant against all n_candidates accounts with mid-rank ties. With SCORE_ONLY set it loads the saved checkpoint
from OUT instead of training (re-scoring and the leakage test use this). Task 3 adapters write <key>_t4_seed<SEED>.parquet
(event_row, nll_time, pit, median_gap) when T4=1.

Templates use {NP} (repository root), {TABLES}, {QUERIES}, {OUT}, {SEED}, {BUDGET} and {PY:<env>} (python of a conda
environment; see environments/README.md, or set OPB_ENV_<ENV> to override). The method code itself is fetched by
scripts/setup_methods.sh into bench_models/ and f1_bench/.
"""
import glob, os

_DGL_LIB = "{ENVROOT:bm-dgl}/lib/python3.9/site-packages/nvidia"
_TPP = ("cd {NP}/bench_models/EasyTPP && BATCH={batch} PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True "
        "PYTHONPATH={NP}/f1_bench/pydeps_easytpp {PY:model-env} -u easytpp_bench.py {model} {TABLES} {QUERIES} {OUT} {SEED} {BUDGET}")


def _tpp(model, batch):
    return dict(family="point_process", key=model.lower(), queries="full", tasks=["task1", "task3"],
                cmd=_TPP.replace("{model}", model).replace("{batch}", str(batch)),
                score_only=dict(env="SCORE_ONLY={CKPT}", checkpoint=f"easytpp_ckpt_{model}_seed{{SEED}}/*/models/saved_model"))


METHODS = {
    # controls
    "b1": dict(name="Learned feature ranker", family="feature_ranker", key="b1", queries="full", tasks=["task1", "task2a"],
               heuristic_inputs=["popularity", "recency", "affinity", "exposure"], seeds_internal=True,
               cmd="cd {NP}/benchmark_build && MODEL=b1 {PY:model-env} -u b1_ranker.py final {TABLES} {QUERIES} {OUT} cuda:0 {NP}/configs/b1_config.json"),
    "b6": dict(name="Exposure Hawkes", family="other", key="b6", queries="full", tasks=["task1"], heuristic_inputs=["exposure"], seeds_internal=True,
               cmd="cd {NP}/benchmark_build && MODEL=b6 {PY:model-env} -u b1_ranker.py final {TABLES} {QUERIES} {OUT} cuda:0 {NP}/configs/b6_config.json"),
    # diffusion
    "mshgat": dict(name="MS-HGAT", family="diffusion", key="b2", queries="full", tasks=["task1"],
                   cmd="cd {NP}/f1_bench/mshgat_sparse && BATCH=128 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True MSHGAT_SPARSE=1 PYTHONPATH={NP}/f1_repro/pydeps {PY:model-env} -u mshgat_bench.py {TABLES} {QUERIES} {OUT} cuda:0 {SEED} {BUDGET}"),
    "dyhgcn": dict(name="DyHGCN", family="diffusion", key="b3", queries="full", tasks=["task1"],
                   cmd="cd {NP}/f1_bench/dyhgcn_port && BATCH=64 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True PYTHONPATH={NP}/f1_repro/pydeps {PY:model-env} -u dyhgcn_bench.py {TABLES} {QUERIES} {OUT} cuda:0 {SEED} {BUDGET}"),
    "minds": dict(name="MINDS", family="diffusion", key="b4", queries="full", tasks=["task1"],
                  cmd="cd {NP}/f1_bench/minds_port && BATCH=64 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True PYTHONPATH=../pydeps_dhg {PY:model-env} -u minds_bench.py {TABLES} {QUERIES} {OUT} cuda:0 {SEED} {BUDGET}",
                  score_only=dict(env="SCORE_ONLY=1", checkpoint="b4_seed{SEED}.pt")),
    "pmrca": dict(name="PMRCA", family="diffusion", key="pmrca", queries="full", tasks=["task1"],
                  cmd="cd {NP}/bench_models/PMRCA && PYTHONPATH={NP}/pydeps_pyg {PY:model-env} -u pmrca_bench.py {TABLES} {QUERIES} {OUT} {SEED} {BUDGET}",
                  score_only=dict(env="SCORE_ONLY={TRAINLOG}", checkpoint="pmrca_root_seed{SEED}/saved/pmrca_seed{SEED}.pt")),
    "mim": dict(name="MIM", family="diffusion", key="mim", queries="full", tasks=["task1"],
                cmd="cd {NP}/bench_models/MIM && PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True {PY:bm-mim} -u mim_bench.py {TABLES} {QUERIES} {OUT} {SEED} {BUDGET}"),
    # temporal graph
    "evolvegcn": dict(name="EvolveGCN", family="temporal_graph", key="evolvegcn", queries="full", tasks=["task1"],
                      cmd="cd {NP}/bench_models/EvolveGCN && {PY:model-env} -u evolvegcn_bench.py {TABLES} {QUERIES} {OUT} {SEED} {BUDGET}"),
    "tpnet": dict(name="TPNet", family="temporal_graph", key="tpnet", queries="sub10", tasks=["task1"],
                  cmd="cd {NP}/bench_models/TPNet && {PY:bm-tpnet} -u tpnet_bench.py {TABLES} {QUERIES} {OUT} {SEED} {BUDGET}",
                  score_only=dict(env="SCORE_ONLY=1", checkpoint="tpnet_seed{SEED}.pkl")),
    "hyperevent": dict(name="HyperEvent", family="temporal_graph", key="hyperevent", queries="sub10", tasks=["task1"],
                       cmd="cd {NP}/bench_models/HyperEvent && {PY:bm-tncn} -u hyperevent_bench.py {TABLES} {QUERIES} {OUT} {SEED} {BUDGET}",
                       score_only=dict(env="SCORE_ONLY=1", checkpoint="hyperevent_seed{SEED}.pth")),
    "tncn": dict(name="TNCN", family="temporal_graph", key="tncn", queries="sub10", tasks=["task1"],
                 cmd="cd {NP}/bench_models/TNCN && {PY:bm-tncn} -u tncn_bench.py {TABLES} {QUERIES} {OUT} {SEED} {BUDGET}"),
    # marked point processes
    "rmtpp": _tpp("RMTPP", 16), "s2p2": _tpp("S2P2", 32), "wsmthp": _tpp("WSMTHP", 64), "attnhp": _tpp("AttNHP", 32),
    # sequential recommenders
    "sasrec": dict(name="SASRec", family="sequential_recommender", key="sasrec", queries="full", tasks=["task1"],
                   cmd="cd {NP}/bench_models/SASRec && {PY:bm-tf1} -u sasrec_bench.py {TABLES} {QUERIES} {OUT} {SEED} {BUDGET} 201",
                   score_only=dict(env="SCORE_ONLY=1", checkpoint="sasrec_seed{SEED}.ckpt.*")),
    "rearec": dict(name="ReaRec", family="sequential_recommender", key="rearec", queries="full", tasks=["task1"],
                   cmd="cd {NP}/bench_models/ReaRec/src && {PY:model-env} -u rearec_bench.py {TABLES} {QUERIES} {OUT} {SEED} {BUDGET}",
                   score_only=dict(env="SCORE_ONLY=1", checkpoint="rearec_ckpt_seed{SEED}/PRL.pt")),
    "fearec": dict(name="FEARec", family="sequential_recommender", key="fearec", queries="full", tasks=["task1"],
                   cmd="cd {NP}/bench_models/FEARec && export LD_LIBRARY_PATH=" + _DGL_LIB + "/cuda_runtime/lib:" + _DGL_LIB + "/cusparse/lib && {PY:bm-dgl} -u fearec_bench.py {TABLES} {QUERIES} {OUT} {SEED} {BUDGET}"),
    "dcrec": dict(name="DCRec", family="sequential_recommender", key="dcrec", queries="full", tasks=["task1"],
                  cmd="cd {NP}/bench_models/DCRec && export LD_LIBRARY_PATH=" + _DGL_LIB + "/cuda_runtime/lib:" + _DGL_LIB + "/cusparse/lib && {PY:bm-dgl} -u dcrec_bench.py {TABLES} {QUERIES} {OUT} {SEED} {BUDGET}"),
}
for _k, _v in METHODS.items():
    _v.setdefault("name", {"rmtpp": "RMTPP", "s2p2": "S2P2", "wsmthp": "WSMTHP", "attnhp": "AttNHP"}.get(_k, _k))
    _v.setdefault("heuristic_inputs", [])

NP = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_ENV_ROOTS = [os.path.expanduser("~/.conda/envs"), os.path.expanduser("~/miniconda3/envs"), os.path.expanduser("~/anaconda3/envs")]


def env_root(env):
    o = os.environ.get("OPB_ENV_" + env.upper().replace("-", "_"))
    if o: return o
    for r in _ENV_ROOTS:
        if os.path.isdir(os.path.join(r, env)): return os.path.join(r, env)
    raise FileNotFoundError(f"conda environment {env} not found; create it (environments/setup_envs.sh) or set OPB_ENV_{env.upper().replace('-', '_')}")


def render(method, tables, queries, out, seed, budget):
    import re
    cmd = METHODS[method]["cmd"]
    cmd = re.sub(r"\{PY:([\w-]+)\}", lambda m: os.path.join(env_root(m.group(1)), "bin", "python"), cmd)
    cmd = re.sub(r"\{ENVROOT:([\w-]+)\}", lambda m: env_root(m.group(1)), cmd)
    return cmd.format(NP=NP, TABLES=tables, QUERIES=queries, OUT=out, SEED=seed, BUDGET=budget)


def output_file(method, out, seed):
    return os.path.join(out, f"{METHODS[method]['key']}_seed{seed}.parquet")


def checkpoint(method, out, seed):
    so = METHODS[method].get("score_only")
    if not so: return None
    g = glob.glob(os.path.join(out, so["checkpoint"].format(SEED=seed)))
    return g[0] if g else None
