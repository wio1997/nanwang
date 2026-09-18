#!/bin/bash
# Stage 3A controlled RAW (non-padded) non-aligned feature probe.
#
# Safety rules honoured here:
#   * only run because the source audit (kernel_operator_data_copy_impl.h: DataCopyPadUB2GMImpl /
#     DataCopyPadGm2UBImpl) shows the GM address is unconstrained and the align-copy handles it
#   * minimal sizes: N=2, size=2, one process per F
#   * every case in its own process with its own log and exit code
#   * the series stops at the first failure
set -u

OPP=${STAGE3A_OPP:-/root/zyg/build/scattermax_runtime_opp/vendors/customize}
RUNNER=${STAGE3A_RUNNER:-/root/zyg/global_max_pool/stage1b/runner/scattermaxv1_runner}
LOG_DIR=${STAGE3A_LOGS:-/root/zyg/logs}
SUMMARY="$LOG_DIR/stage3a_raw_probe_summary.txt"

# shellcheck disable=SC1091
export ASCEND_CUSTOM_OPP_PATH="${ASCEND_CUSTOM_OPP_PATH:-}"
set +u
source "$OPP/bin/set_env.bash"
set -u

echo "# Stage 3A controlled RAW non-aligned probe (one process per F, stop on first failure)" > "$SUMMARY"

# shellcheck disable=SC2086  # word-splitting is intentional (space separated F list)
for F in ${STAGE3A_F_LIST:-1 7 9 17}; do
    LOG="$LOG_DIR/stage3a_raw_f${F}.log"
    echo "=== F=$F (separate process) ==="
    "$RUNNER" --case "raw_f${F}" --n 2 --f "$F" --size 2 --index 0,1 --pattern baseline --tol 0 > "$LOG" 2>&1
    rc=$?
    echo "exit=$rc"
    grep -E "^(CASE|N=|max_abs_diff|expected_row0|actual_row0|RESULT|ACL_ERROR|ACLNN_ERROR|RUNNER_)" "$LOG" | sed 's/^/    /'
    verdict=$(grep -h '^RESULT ' "$LOG" | awk '{print $2}')
    printf "F=%-3s exit=%s %s\n" "$F" "$rc" "${verdict:-NO_VERDICT}" >> "$SUMMARY"
    if [ "$rc" -ne 0 ]; then
        echo "!! STOPPING raw non-aligned series at F=$F (exit=$rc); no further F is attempted"
        echo "STOPPED_AT_F=$F" >> "$SUMMARY"
        break
    fi
done

echo
cat "$SUMMARY"
