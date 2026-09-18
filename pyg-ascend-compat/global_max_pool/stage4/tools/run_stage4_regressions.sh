#!/bin/bash
# Stage 4 — full regression: forward suites (must be unchanged) + Stage 4 backward suites.
set -u

LOGS=${STAGE4_LOGS:-/root/zyg/logs/stage4}
OPP=${STAGE4_OPP:-/root/zyg/build/scattermax_runtime_opp/vendors/customize}
GMP=/root/zyg/global_max_pool
mkdir -p "$LOGS"

export STAGE6_OPP="$OPP"
export PYG_ASCEND_ADAPTER_PATH="${PYG_ASCEND_ADAPTER_PATH:-$GMP/stage2/python/global_max_pool_ascend.py}"
export SCATTERMAXV1_BRIDGE="${SCATTERMAXV1_BRIDGE:-/root/zyg/build/stage2_ext/scattermaxv1_bridge.so}"
set +u
source "$GMP/stage6/env.sh" > "$LOGS/stage4_env.txt" 2>&1
set -u

FAILED=0
run() {  # label log command...
    local label="$1" log="$2"; shift 2
    echo "[regr] $label"
    "$@" > "$log" 2>&1
    local rc=$?
    printf "%-22s exit=%-3s %s\n" "$label" "$rc" \
        "$(grep -hE '^TOTAL |^RESULT: |AFTER_HOST_CPU_FALLBACK' "$log" | tr '\n' ' ')"
    [ "$rc" -ne 0 ] && FAILED=1
}

run stage3a "$LOGS/stage4_regr_stage3a.log" python3 "$GMP/stage3a/tests/run_stage3a_tests.py"
run stage3b "$LOGS/stage4_regr_stage3b.log" python3 "$GMP/stage3b/tests/run_stage3b_boundary_tests.py"
run stage3d "$LOGS/stage4_regr_stage3d.log" python3 "$GMP/stage3d/tests/run_stage3d_largetail_tests.py"
run stage3e_lt "$LOGS/stage4_regr_stage3e.log" python3 /root/zyg/stage3e/tests/run_stage3e_largetail_tests.py
run stage4_bwd "$LOGS/stage4_backward_tests.log" python3 /root/zyg/stage4/tests/run_stage4_backward_tests.py
run stage4_e2e "$LOGS/stage4_pyg_e2e.log" python3 /root/zyg/stage4/tests/run_stage4_pyg_e2e.py
run stage6_demo "$LOGS/stage4_regr_stage6_demo.log" python3 "$GMP/stage6/demo_global_max_pool_ascend.py"
run stage6_tests "$LOGS/stage4_regr_stage6_tests.log" python3 "$GMP/stage6/tests/run_stage6_tests.py"
run stage2_tests "$LOGS/stage4_regr_stage2.log" python3 "$GMP/stage2/tests/run_stage2_tests.py"

echo
echo "=== gates ==="
gate=0
chk() {
    if grep -qE "$3" "$2"; then echo "PASS  $1 : $(grep -hE "$3" "$2" | head -1)"; else
        echo "FAIL  $1 : pattern '$3' not found"; gate=1; fi
}
chk "Stage 3A 34/34"       "$LOGS/stage4_regr_stage3a.log" '^TOTAL 34  PASS 34  FAIL 0'
chk "Stage 3B 45/45"       "$LOGS/stage4_regr_stage3b.log" '^TOTAL 45  PASS 45  FAIL 0'
chk "Stage 3D 9/9"         "$LOGS/stage4_regr_stage3d.log" '^TOTAL 9  PASS 9  FAIL 0'
chk "Stage 3E largeTail"   "$LOGS/stage4_regr_stage3e.log" '^TOTAL 13  PASS 13  FAIL 0'
chk "Stage 4 backward"     "$LOGS/stage4_backward_tests.log" '^TOTAL [0-9]+  PASS [0-9]+  FAIL 0'
chk "Stage 4 PyG E2E"      "$LOGS/stage4_pyg_e2e.log" '^TOTAL [0-9]+  PASS [0-9]+  FAIL 0'
chk "Stage 6 demo PASS"    "$LOGS/stage4_regr_stage6_demo.log" '^RESULT: PASS'
chk "Stage 6 20/20"        "$LOGS/stage4_regr_stage6_tests.log" '^TOTAL 20  PASS 20  FAIL 0'
chk "Stage 6 fallback"     "$LOGS/stage4_regr_stage6_tests.log" 'BEFORE_HOST_CPU_FALLBACK=YES AFTER_HOST_CPU_FALLBACK=NO'
chk "Stage 4 no fallback"  "$LOGS/stage4_backward_tests.log" 'fallback=0'

echo
echo "[DONE] script_failed=$FAILED gate_failed=$gate"
[ "$FAILED" -ne 0 ] || [ "$gate" -ne 0 ] && exit 1
exit 0
