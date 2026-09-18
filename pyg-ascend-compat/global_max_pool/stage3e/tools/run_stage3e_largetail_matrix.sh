#!/bin/bash
# Stage 3E / Task F: largeTail runtime matrix against the FORMAL delivery OPP.
#
#   LT0  N=40  F=48824 S=8   SMALL_TAIL control (last small-tail F for N=40)
#   LT1  N=40  F=48825 S=8   first LARGE_TAIL F
#   LT2  N=40  F=48826 S=8   LARGE_TAIL + non-aligned F
#   LT3  N=41  F=48825 S=8   LARGE_TAIL + leftSrc (1 remainder row)
#   LT4  N=80  F=48825 S=8   LARGE_TAIL, 2 rows/core
#   LT5  N=81  F=48825 S=8   LARGE_TAIL + leftSrc
#   LT6  N=320 F=48825 S=8   LARGE_TAIL + negative-only data
#   LTA1 N=40  F=48832 S=8   32 B aligned F
#   LTA2 N=40  F=48960 S=8   512 B aligned F
#
# ONE SHAPE = ONE PROCESS. The series stops at the first failure and never hides it.
set -u

LOGS=${STAGE3E_LOGS:-/root/zyg/logs/stage3e}
OPP=${STAGE3E_OPP:-/root/zyg/build/scattermax_runtime_opp/vendors/customize}
RUNNER=${STAGE3E_RUNNER:-/root/zyg/global_max_pool/stage1b/runner/scattermaxv1_runner}
SUMMARY="$LOGS/04_largetail_matrix_summary.txt"
mkdir -p "$LOGS"

export ASCEND_CUSTOM_OPP_PATH="${ASCEND_CUSTOM_OPP_PATH:-}"
set +u
source "$OPP/bin/set_env.bash"
set -u

FORBIDDEN='507035|507011|out of range|vector core exception|aicore exception|AIV exception|npu_cpu_fallback|fall back to run on the CPU'

{
    echo "# Stage 3E largeTail runtime matrix - FORMAL delivery OPP"
    echo "# OPP=$OPP"
    echo "# ASCEND_CUSTOM_OPP_PATH=$ASCEND_CUSTOM_OPP_PATH"
    echo "# runner=$RUNNER"
    echo "# one shape = one process; series stops at first failure"
    echo
    printf "%-34s %-6s %-8s %s\n" case exit verdict detail
} > "$SUMMARY"

FAILED=0
run_case() {
    local tag="$1" n="$2" f="$3" s="$4" pattern="$5"
    local log="$LOGS/04_${tag}.log"
    local idx
    idx=$(python3 -c "print(','.join(str(i % $s) for i in range($n)))")
    echo "[RUN ] $tag N=$n F=$f S=$s pattern=$pattern"
    local t0 t1
    t0=$(date +%s)
    "$RUNNER" --case "$tag" --n "$n" --f "$f" --size "$s" --index "$idx" \
        --pattern "$pattern" --tol 0 > "$log" 2>&1
    local rc=$?
    t1=$(date +%s)
    local verdict diff bad
    verdict=$(grep -h '^RESULT ' "$log" | awk '{print $2}')
    diff=$(grep -ho 'max_abs_diff=[^ ]*' "$log" | head -1)
    bad=$(grep -hoE "$FORBIDDEN" "$log" | sort -u | paste -sd, -)
    printf "%-34s exit=%-3s verdict=%-5s %s %s %s(s)\n" "$tag" "$rc" "${verdict:-none}" \
        "${diff:-no-diff-line}" "${bad:+FORBIDDEN=$bad}" "$((t1 - t0))"
    printf "%-34s %-6s %-8s %s %s\n" "$tag" "$rc" "${verdict:-none}" \
        "${diff:-no-diff-line}" "${bad:+FORBIDDEN=$bad}" >> "$SUMMARY"
    if [ "$rc" -ne 0 ] || [ "$verdict" != "PASS" ] || [ -n "$bad" ]; then
        FAILED=1
        echo "!! STOP: $tag failed (exit=$rc verdict=${verdict:-none} forbidden=${bad:-none})"
        echo "STOPPED_AT=$tag" >> "$SUMMARY"
        return 1
    fi
    return 0
}

run_case LT0_N40_F48824_S8    40  48824 8 baseline || true
[ "$FAILED" -eq 0 ] && { run_case LT1_N40_F48825_S8    40  48825 8 baseline || true; }
[ "$FAILED" -eq 0 ] && { run_case LT2_N40_F48826_S8    40  48826 8 distinct || true; }
[ "$FAILED" -eq 0 ] && { run_case LT3_N41_F48825_S8    41  48825 8 baseline || true; }
[ "$FAILED" -eq 0 ] && { run_case LT4_N80_F48825_S8    80  48825 8 baseline || true; }
[ "$FAILED" -eq 0 ] && { run_case LT5_N81_F48825_S8    81  48825 8 distinct || true; }
[ "$FAILED" -eq 0 ] && { run_case LT6_N320_F48825_S8  320  48825 8 negative || true; }
[ "$FAILED" -eq 0 ] && { run_case LTA1_N40_F48832_S8   40  48832 8 baseline || true; }
[ "$FAILED" -eq 0 ] && { run_case LTA2_N40_F48960_S8   40  48960 8 baseline || true; }

echo
cat "$SUMMARY"
echo "[DONE] failed=$FAILED"
exit "$FAILED"
