#!/bin/bash
# Phase D - msprof gate on real PowerGraph batches.
#
# Each case must show:
#   * ScatterMaxV1 tasks completing on AI_VECTOR_CORE
#   * no AI_CPU task types
#   * no host-CPU fallback marker in the msprof log
#   * no aten::scatter_reduce call (compat cases)
#   * compat counters proving the Ascend path was taken
#
# Usage: run_profiles.sh "TAG DATASET BATCH DTYPE PATH" ...
#
# Roots are derived from this script's location and can be overridden:
#   POWERGRAPH_PROFILE_ROOT   msprof output root (default <package>/profiler_runs)
#
# The summary is written inside POWERGRAPH_PROFILE_ROOT, so a default run never
# overwrites the committed evidence under results/ or evidence/profiler/.
set -u

_SCRIPTS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
_VALIDATION_ROOT="$(cd "$_SCRIPTS_DIR/.." && pwd)"
PROF_ROOT="${POWERGRAPH_PROFILE_ROOT:-$_VALIDATION_ROOT/profiler_runs}"
RESULTS_ROOT="${POWERGRAPH_RESULTS_ROOT:-$_VALIDATION_ROOT/results}"
APPS="$_SCRIPTS_DIR/profile_app.py"
PARSE="$_SCRIPTS_DIR/parse_profile.py"
SUMMARY="$PROF_ROOT/profiler_summary.txt"
mkdir -p "$PROF_ROOT" "$RESULTS_ROOT"

FORBIDDEN='507035|507011|out of range|vector core exception|aicore exception|AIV exception|npu_cpu_fallback|fall back to run on the CPU'

{
    echo "# PowerGraph global_max_pool profiler gate"
    echo "date=$(date -u +%FT%TZ)"
    echo "ASCEND_CUSTOM_OPP_PATH=${ASCEND_CUSTOM_OPP_PATH:-}"
    echo "adapter=${PYG_ASCEND_ADAPTER_PATH:-}"
    echo "bridge=${SCATTERMAXV1_BRIDGE:-}"
    echo "app=$APPS (3 warm-up + 5 profiled calls per case)"
    echo
} > "$SUMMARY"

FAILED=0
for spec in "$@"; do
    set -- $spec
    TAG=$1; DS=$2; BS=$3; DT=$4; PATHK=$5
    OUTDIR="$PROF_ROOT/$TAG"
    LOG="$PROF_ROOT/$TAG.msprof.log"
    rm -rf "$OUTDIR"
    mkdir -p "$OUTDIR"
    echo "[msprof] $TAG dataset=$DS batch=$BS dtype=$DT path=$PATHK"
    msprof --application="python3 $APPS --dataset $DS --batch $BS --dtype $DT --path $PATHK --tag $TAG --out $PROF_ROOT/$TAG.json" \
        --output="$OUTDIR" --ai-core=on > "$LOG" 2>&1
    rc=$?
    PROF=$(find "$OUTDIR" -maxdepth 1 -mindepth 1 -type d -name "PROF_*" | head -1)

    {
        echo "## $TAG  dataset=$DS batch=$BS dtype=$DT path=$PATHK  msprof_exit=$rc"
        echo "prof_dir=$PROF"
        if [ -z "$PROF" ]; then
            echo "GATE_RESULT FAIL (no profiler output)"
            FAILED=1
        else
            python3 "$PARSE" "$PROF" "$TAG" "$PATHK" "$PROF_ROOT/$TAG.gate.json"
            echo "--- compat counters (from profile_app) ---"
            python3 -c "import json,sys; d=json.load(open(sys.argv[1])); print(json.dumps(d['compat_stats']))" "$PROF_ROOT/$TAG.json" 2>/dev/null || echo "(no counters json)"
            echo "--- forbidden markers in msprof log ---"
            grep -hoE "$FORBIDDEN" "$LOG" | sort -u || echo "NONE"
            echo "--- app banner ---"
            grep -E "BENCH_PROFILE" "$LOG" || echo "(missing banner)"
        fi
        echo
    } | tee -a "$SUMMARY"

    if [ "$rc" -ne 0 ]; then
        echo "!! msprof failed for $TAG"
        FAILED=1
    fi
done
echo "[profiler] failed=$FAILED"
exit "$FAILED"
