#!/usr/bin/env bash
# Create the conda environments of the released method code, with the exact package versions used for the paper.
#   bash environments/setup_envs.sh [ENV ...]      (default: all)
# GPUs used for v1.0: NVIDIA A40 (sm_86), CUDA 11.7/11.8 wheels (12.6 for model-env).
set -euo pipefail
cd "$(dirname "$0")"
extra() {   # package indexes needed by each environment's pinned wheels
  case $1 in
    model-env) echo "--extra-index-url https://download.pytorch.org/whl/cu126" ;;
    bm-tpnet|bm-mim) echo "--extra-index-url https://download.pytorch.org/whl/cu118 -f https://data.pyg.org/whl/torch-2.1.0+cu118.html" ;;
    bm-tncn) echo "--extra-index-url https://download.pytorch.org/whl/cu117 -f https://data.pyg.org/whl/torch-2.0.0+cu117.html" ;;
    bm-dgl) echo "--extra-index-url https://download.pytorch.org/whl/cu118 -f https://data.dgl.ai/wheels/cu118/repo.html" ;;
    bm-tf1) echo "--extra-index-url https://pypi.nvidia.com" ;;
  esac
}
for e in ${@:-$(cut -f1 python_versions.tsv)}; do
  v=$(awk -v e="$e" '$1 == e {print $2}' python_versions.tsv)
  conda env list | grep -q "^$e " || conda create -y -q -n "$e" "python=$v"
  conda run -n "$e" pip install -q $(extra "$e") -r "$e.txt"
  echo "[envs] $e ready (python $v)"
done
