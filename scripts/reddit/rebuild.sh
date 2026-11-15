#!/usr/bin/env bash
# Rebuild the Reddit tiers of OpenPopBench v1.0 from the public source, in the release layout.
#   bash scripts/reddit/rebuild.sh /path/to/Hessel_ICWSM_2016 /path/to/reddit_id_perm.parquet OUT [GPU]
# Then: export OPB_DATA=OUT/release   (the Bluesky and Telegram tiers from Hugging Face can be placed in the same folder)
# Needs about 200 GB of disk and one GPU (floors), about 1-2 hours on 40 cores. See scripts/reddit/README.md.
set -euo pipefail
SRC=$(readlink -f "$1"); PERM=$(readlink -f "$2"); OUT=$(mkdir -p "$3" && readlink -f "$3"); GPU=${4:-0}
ROOT=$(cd "$(dirname "$0")/../.." && pwd); B=$ROOT/benchmark_build; PY=${OPB_PY:-python}
cd "$SRC" && sha256sum --quiet -c "$ROOT/scripts/reddit/SOURCE_SHA256.txt" && echo "[reddit] source verified"
cd "$B"
HESSEL_SRC=$SRC/only_meta_and_stats $PY parse_reddit.py "$OUT/parsed"
REDDIT_PARSED=$OUT/parsed REDDIT_OUT=$OUT/v0.1 ID_PERM=$PERM $PY build_reddit.py
TIERS=S:50000,Mid:150000 $PY ff_tiers.py "$OUT/v0.1" "$OUT/v0.1_ladderE/pf0.35_s1" 0.35 1
TIERS=M:500000 $PY ff_tiers.py "$OUT/v0.1" "$OUT/v0.1_ladderE/pf0.7_s1" 0.7 1
declare -A TB=([reddit_S]=$OUT/v0.1_ladderE/pf0.35_s1/tier_S [reddit_Mid]=$OUT/v0.1_ladderE/pf0.35_s1/tier_Mid
               [reddit_M]=$OUT/v0.1_ladderE/pf0.7_s1/tier_M [reddit_L]=$OUT/v0.1)
mkdir -p "$OUT/eval/sub10" "$OUT/eval/t2" "$OUT/eval/t4"
J="{"
for t in reddit_S reddit_Mid reddit_M reddit_L; do
  $PY floors_eval.py "${TB[$t]}" "$OUT/eval/floors_$t.parquet" 20000 cuda:$GPU
  $PY make_query_subsample.py "$OUT/eval/floors_$t.parquet" "$OUT/eval/sub10/floors_$t.parquet"
  J="$J\"$t\": [\"${TB[$t]}\", \"$OUT/eval/floors_$t.parquet\"],"
done
export OPB_WORK=$OUT OPB_TIERS="${J%,}}"
$PY t2_flags.py && $PY t2a_floors.py && $PY t4_time.py floors
$PY "$ROOT/scripts/reddit/assemble.py" "$OUT"
$PY content_hash.py --manifest "$OUT/MANIFEST_rebuilt.tsv" "$OUT/release" $(find "$OUT/release" -name '*.parquet' | sort)
$PY "$ROOT/scripts/reddit/assemble.py" --check "$OUT/MANIFEST_rebuilt.tsv" "$ROOT/scripts/reddit/MANIFEST_RELEASE.tsv"
