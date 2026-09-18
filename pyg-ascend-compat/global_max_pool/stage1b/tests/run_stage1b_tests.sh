#!/bin/bash
# Stage 1B test driver: T1..T5 for DrivingSDK ScatterMaxV1 (FP32 + INT32) on 910B3.
#
# Environment used (only for this shell, nothing is persisted globally):
#   ASCEND_CUSTOM_OPP_PATH  -> isolated custom OPP vendor dir
#   LD_LIBRARY_PATH         -> isolated op_api lib dir (per vendor set_env.bash)
#
# Usage: bash run_stage1b_tests.sh [--skip-t2-...]   (no options needed normally)

set -u

HERE=$(cd "$(dirname "$0")" && pwd)
STAGE1B=$(cd "$HERE/.." && pwd)
RUNNER="$STAGE1B/runner/scattermaxv1_runner"
OPP="${STAGE1B_OPP:-/root/zyg/build/scattermax_runtime_opp/vendors/customize}"
LOGS="${STAGE1B_LOGS:-/root/zyg/logs}"
DUMP="$STAGE1B/results"
TOL="${STAGE1B_TOL:-0}"

mkdir -p "$LOGS" "$DUMP"

# shellcheck disable=SC1091
if [ -f "$OPP/bin/set_env.bash" ]; then
    # vendor script references ${ASCEND_CUSTOM_OPP_PATH} unconditionally
    export ASCEND_CUSTOM_OPP_PATH="${ASCEND_CUSTOM_OPP_PATH:-}"
    set +u
    source "$OPP/bin/set_env.bash"
    set -u
else
    export ASCEND_CUSTOM_OPP_PATH="$OPP"
    export LD_LIBRARY_PATH="$OPP/op_api/lib:$LD_LIBRARY_PATH"
fi

if [ ! -x "$RUNNER" ]; then
    echo "[ERROR] runner not built: run ../runner/build_runner.sh first" >&2
    exit 1
fi

FAILED=0
SUMMARY="$LOGS/stage1b_test_summary.txt"
: > "$SUMMARY"

run_case() {
    local tag="$1"; shift
    local log="$LOGS/stage1b_${tag}.log"
    echo "=================================================================="
    echo "[RUN ] $tag : $*"
    "$RUNNER" --dump "$DUMP/${tag}.txt" "$@" > "$log" 2>&1
    local rc=$?
    grep -E "^(CASE|N=|workspace_bytes|empty_rows|max_abs_diff|expected_row0|actual_row0|RESULT|ACL_ERROR|ACLNN_ERROR|RUNNER_)" "$log" | sed 's/^/    /'
    local verdict
    verdict=$(grep -E "^RESULT " "$log" | awk '{print $2}')
    if [ "$rc" -ne 0 ] || [ "$verdict" != "PASS" ]; then
        FAILED=1
        echo "    --> FAIL (exit=$rc)"
        printf "%-28s %-6s exit=%s log=%s\n" "$tag" "FAIL" "$rc" "$log" >> "$SUMMARY"
    else
        echo "    --> PASS"
        printf "%-28s %-6s exit=%s log=%s\n" "$tag" "PASS" "$rc" "$log" >> "$SUMMARY"
    fi
}

# ---------------------------------------------------------------- T1
# minimal aligned baseline: F=8 (32B aligned), duplicates, +/-, 4 groups
run_case t1_aligned_baseline \
    --case t1 --n 8 --f 8 --size 4 --index 0,1,0,2,1,2,0,3 --pattern baseline --tol "$TOL"

# second baseline with F=16 and more rows
run_case t1b_aligned_baseline_f16 \
    --case t1b --n 32 --f 16 --size 8 \
    --index 3,0,7,7,2,5,1,4,6,3,2,0,7,1,5,4,2,6,0,3,7,7,1,2,4,5,6,0,3,2,1,7 \
    --pattern baseline --tol "$TOL"

# ---------------------------------------------------------------- T2
# repeated index stress: every row writes into group 0
run_case t2_repeated_index \
    --case t2 --n 16 --f 8 --size 1 \
    --index 0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0 --pattern distinct --tol "$TOL"

# ---------------------------------------------------------------- T3
# negative-only group: result must be the most negative max, never 0
run_case t3_negative_only \
    --case t3 --n 8 --f 8 --size 2 --index 0,0,0,0,1,1,1,1 --pattern negative --tol "$TOL"

# ---------------------------------------------------------------- T4
# caller-provided output / explicit size: size=5 but max(index)+1=3
run_case t4_explicit_size \
    --case t4 --n 4 --f 8 --size 5 --index 0,2,0,1 --pattern baseline --tol "$TOL"

# ---------------------------------------------------------------- T5
# sparse groups: size=8, only groups 1,2,5 are written
run_case t5_sparse_groups \
    --case t5 --n 5 --f 8 --size 8 --index 1,1,5,2,5 --pattern baseline --tol "$TOL"

echo "=================================================================="
cat "$SUMMARY"
echo "[DONE] failed=$FAILED"
exit $FAILED
