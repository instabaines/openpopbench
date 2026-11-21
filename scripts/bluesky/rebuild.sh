#!/usr/bin/env bash
# Rebuild the Bluesky tiers of OpenPopBench v1.0 from your copy of the source export, in the release layout.
#   bash scripts/bluesky/rebuild.sh /path/to/export /path/to/openpopbench-bluesky OUT [GPU]
# /path/to/export holds follows.parquet, posts.parquet and reposts.parquet of the Balduf et al. export (snapshot 2025-04-14).
# /path/to/openpopbench-bluesky is the rebuild package (Hugging Face dataset Baines/openpopbench-bluesky).
# Then: export OPB_DATA=OUT/release
# Needs about 400 GB of RAM, 300 GB of disk and one GPU; several hours on 64 cores. See scripts/bluesky/README.md.
set -euo pipefail
SRC=$(readlink -f "$1"); PKG=$(readlink -f "$2"); OUT=$(mkdir -p "$3" && readlink -f "$3"); GPU=${4:-0}
ROOT=$(cd "$(dirname "$0")/../.." && pwd); B=$ROOT/benchmark_build; PY=${OPB_PY:-python}
cd "$SRC" && sha256sum --quiet -c --ignore-missing "$PKG/SOURCE_SHA256.txt" && echo "[bluesky] source verified"
cd "$B"
BLUESKY_RAW=$SRC BLUESKY_OUT=$OUT/v0.1 ID_PERM=$PKG/bluesky_id_perm.parquet $PY build_bluesky.py
for k in S Mid M; do
  MEMBERS=$k=$PKG/tier_${k}_members.parquet $PY ff_tiers.py "$OUT/v0.1" "$OUT/tiers" 0.35 1
done
declare -A TB=([bluesky_S]=$OUT/tiers/tier_S [bluesky_Mid]=$OUT/tiers/tier_Mid [bluesky_M]=$OUT/tiers/tier_M [bluesky_L]=$OUT/v0.1)
mkdir -p "$OUT/eval/sub10" "$OUT/eval/t2" "$OUT/eval/t4"
J="{"
for t in bluesky_S bluesky_Mid bluesky_M bluesky_L; do
  $PY floors_eval.py "${TB[$t]}" "$OUT/eval/floors_$t.parquet" 20000 cuda:$GPU
  $PY make_query_subsample.py "$OUT/eval/floors_$t.parquet" "$OUT/eval/sub10/floors_$t.parquet"
  J="$J\"$t\": [\"${TB[$t]}\", \"$OUT/eval/floors_$t.parquet\"],"
done
export OPB_WORK=$OUT OPB_TIERS="${J%,}}"
$PY t2_flags.py && $PY t2a_floors.py && $PY t4_time.py floors
$PY "$ROOT/scripts/bluesky/assemble.py" "$OUT"
$PY content_hash.py --manifest "$OUT/MANIFEST_rebuilt.tsv" "$OUT/release" $(find "$OUT/release" -name '*.parquet' | sort)
$PY "$ROOT/scripts/bluesky/assemble.py" --check "$OUT/MANIFEST_rebuilt.tsv" "$PKG/MANIFEST_RELEASE.tsv"
