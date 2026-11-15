# Method environments

The released code of the published methods pins different, often old, library versions, so each family runs in its own conda environment. `python_versions.tsv` gives each environment's Python version, and `<env>.txt` its exact package versions (`pip freeze` of the environments used for the paper).

```bash
bash environments/setup_envs.sh            # all environments
bash scripts/setup_methods.sh              # released code at pinned commits, our patches, vendored packages
```

| environment | methods |
|---|---|
| model-env | controls, MS-HGAT, DyHGCN, MINDS, PMRCA, EvolveGCN, ReaRec, RMTPP, S2P2, WSMTHP, AttNHP |
| bm-tpnet | TPNet |
| bm-tncn | TNCN, HyperEvent |
| bm-mim | MIM |
| bm-dgl | FEARec, DCRec |
| bm-tf1 | SASRec (NVIDIA's TensorFlow 1.15 build) |

The harness finds an environment under `~/.conda/envs`, `~/miniconda3/envs` or `~/anaconda3/envs`. Set `OPB_ENV_<NAME>` (for example `OPB_ENV_BM_TF1=/opt/envs/bm-tf1`) to point elsewhere.

Notes from building them:
- **bm-dgl.** Install the CUDA build of DGL (`dgl==2.1.0+cu118` from the DGL wheel index), not the CPU wheel from PyPI. Keep numpy at 1.23.5, because RecBole uses aliases that numpy 1.24 removed.
- **bm-mim.** Keep numpy below 2, because MIM uses `np.Inf`.
- **FEARec and DCRec (RecBole).** RecBole sets `CUDA_VISIBLE_DEVICES` from its `gpu_id` option. The adapters pass the assigned GPU explicitly.
- **bm-tf1.** Do not install `nvidia-pyindex`: it rewrites global pip configuration files. Use the extra index in `setup_envs.sh` instead.
