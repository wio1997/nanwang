#!/bin/bash
# Stage 3C: largeTail runtime matrix LT0..LT6, ONE PROCESS PER CASE, stop on first failure.
#
# Uses the repaired package (kernel entries _0 and _1) installed in an isolated dir; the frozen
# Stage 1A/2/6 packages are untouched.
set -u

LOGS=${STAGE3C_LOGS:-/root/zyg/logs/stage3c}
OPP=${STAGE3C_OPP:-/root/zyg/build/stage3c_opp_fixed/vendors/customize}
RUNNER=${STAGE3C_RUNNER:-/root/zyg/global_max_pool/stage1b/runner/scattermaxv1_runner}
mkdir -p "$LOGS"

export ASCEND_CUSTOM_OPP_PATH="${ASCEND_CUSTOM_OPP_PATH:-}"
set +u
source "$OPP/bin/set_env.bash"
set -u

SUMMARY="$LOGS/largetail_matrix_summary.txt"
echo "# Stage 3C largeTail runtime matrix (one process per case, stop on first failure)" > "$SUMMARY"
echo "# OPP=$OPP" >> "$SUMMARY"

FAILED=0
run_case() {
    local tag="$1" n="$2" f="$3" s="$4" pattern="$5"
    local log="$LOGS/${tag}.log"
    local idx
    idx=$(python3 -c "print(','.join(str(i % $s) for i in range($n)))")
    echo "[RUN ] $tag : N=$n F=$f S=$s pattern=$pattern"
    "$RUNNER" --case "$tag" --n "$n" --f "$f" --size "$s" --index "$idx" \
        --pattern "$pattern" --tol 0 > "$log" 2>&1
    local rc=$?
    local verdict tkey
    verdict=$(grep -h '^RESULT ' "$log" | awk '{print $2}')
    tkey=$(grep -ho 'TILING_KEY=[A-Z_]*(.)' "$log" | head -1)
    printf "%-28s exit=%s verdict=%-5s %s\n" "$tag" "$rc" "${verdict:-none}" "${tkey:-tiling-unknown}"
    printf "%-28s exit=%s verdict=%-5s %s\n" "$tag" "$rc" "${verdict:-none}" \
        "${tkey:-tiling-unknown}" >> "$SUMMARY"
    if [ "$rc" -ne 0 ]; then
        FAILED=1
        echo "!! STOPPING series at $tag (exit=$rc) - no further cases attempted"
        echo "STOPPED_AT=$tag" >> "$SUMMARY"
        return 1
    fi
    return 0
}

run_case LT0_control_N40_F48824_S8     40  48824 8 baseline || true
if [ "$FAILED" -eq 0 ]; then run_case LT1_first_largetail_N40_F48825_S8  40  48825 8 baseline || true; fi
if [ "$FAILED" -eq 0 ]; then run_case LT2_non_aligned_N40_F48826_S8      40  48826 8 distinct || true; fi
if [ "$FAILED" -eq 0 ]; then run_case LT3_largetail_leftsrc_N41_F48825_S8 41 48825 8 baseline || true; fi
if [ "$FAILED" -eq 0 ]; then run_case LT4_N80_F48825_S8                  80  48825 8 baseline || true; fi
if [ "$FAILED" -eq 0 ]; then run_case LT5_largetail_leftsrc_N81_F48825_S8 81 48825 8 distinct || true; fi
if [ "$FAILED" -eq 0 ]; then run_case LT6_N320_F48825_S8                320  48825 8 negative || true; fi

echo
cat "$SUMMARY"
echo "[DONE] failed=$FAILED"
