#!/usr/bin/env bash
# Fetch the authors' released code of every registered method at the pinned commit, apply our changes, and install the
# vendored packages some adapters load from a folder. Run from the repository root:  bash scripts/setup_methods.sh
#
# methods/<path>/UPSTREAM.txt   origin URL and commit of the released code
# methods/<path>/changes.patch  our edits to released files (if any; every change is listed in BENCH_CHANGES.md)
# methods/<path>/new_files/     our adapter and documentation files, copied into the clone
# methods/<path>/upstream/      the released code itself, only where the upstream repository is no longer public
# methods/f1_bench/<port>/port.patch  ports of MS-HGAT, DyHGCN and MINDS (memory-saving and compatibility changes,
#                                     validation-based selection), applied to a fresh copy of the released code
#                                     (UPSTREAM.txt fields archive and subdir: the released code is in a zip archive of the clone)
set -euo pipefail
ROOT=$(pwd)
field() { awk -v k="$1" -F'\t' '$1 == k {print $2}' "$2"; }
for up in $(find methods -name UPSTREAM.txt | sort); do
  d=$(dirname "$up"); path=$(field path "$up"); url=$(field url "$up"); commit=$(field commit "$up")
  if [ -e "$path/.opb_ready" ]; then echo "[setup] $path ready"; continue; fi
  echo "[setup] $path <- $url @ $commit"
  tmp=$(mktemp -d)
  if [ -d "$d/upstream" ]; then cp -r "$d/upstream" "$tmp/src"            # vendored: upstream no longer public (see its NOTICE)
  else git clone -q "$url" "$tmp/src"; git -C "$tmp/src" checkout -q "$commit"; fi
  rm -rf "$path"; mkdir -p "$(dirname "$path")"
  if [ -f "$d/port.patch" ]; then
    src="$tmp/src"; arc=$(field archive "$up")
    if [ -n "$arc" ]; then
      python -c 'import sys, zipfile; zipfile.ZipFile(sys.argv[1]).extractall(sys.argv[2])' "$tmp/src/$arc" "$tmp/arc"
      src="$tmp/arc/$(field subdir "$up")"
    fi
    rsync -a --exclude .git --exclude .idea --exclude __pycache__ "$src/" "$path/"; (cd "$path" && patch -s -E -p1 < "$ROOT/$d/port.patch")
  else
    mv "$tmp/src" "$path"
    if [ -f "$d/changes.patch" ]; then (cd "$path" && git apply "$ROOT/$d/changes.patch"); fi
    if [ -d "$d/new_files" ]; then cp -r "$d/new_files/." "$path/"; fi
  fi
  rm -rf "$tmp"; touch "$path/.opb_ready"
done
# vendored packages (installed into folders, loaded through PYTHONPATH by the adapters; versions as used for the paper)
PY=${OPB_PY_MODEL_ENV:-python}
$PY -m pip install -q --no-deps --target f1_bench/pydeps_easytpp easy_tpp==0.3.0
$PY -m pip install -q --no-deps --target f1_bench/pydeps_dhg dhg==0.9.4 optuna==5.0.0 alembic==1.20.0 colorlog==6.12.0 mako==1.4.3 sqlalchemy==2.1.4
$PY -m pip install -q --no-deps --target f1_repro/pydeps torch_geometric==2.8.0.post1 markupsafe==3.0.4 typing_extensions==4.16.0
$PY -m pip install -q --no-deps --target pydeps_pyg torch_geometric==2.6.1
echo "[setup] methods ready"
