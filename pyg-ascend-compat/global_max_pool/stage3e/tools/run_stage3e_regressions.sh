#!/bin/bash
# Stage 3E / Task J: full regression against the FORMAL delivery OPP.
#
#   Stage 3A  34/34   non-aligned feature dims / padding / F=0 / transitions
#   Stage 3B  45/45   boundary matrix (N/core boundary, leftSrc, large N, index+budget guards)
#   Stage 3D   9/9    the original Stage 3D largeTail suite (re-run on the promoted package)
#   Stage 3E   n/n    the Stage 3E tail-attack suite (superset of 3D)
#   Stage 6   demo    real PyG API demo
#   Stage 6   20/20   real PyG API matrix incl. before/after fallback + CPU parity
set -u

LOGS=${STAGE3E_LOGS:-/root/zyg/logs/stage3e}
OPP=${STAGE3E_OPP:-/root/zyg/build/scattermax_runtime_opp/vendors/customize}
GMP=/root/zyg/global_max_pool
mkdir -p "$LOGS"

export STAGE6_OPP="$OPP"
export PYG_ASCEND_ADAPTER_PATH="${PYG_ASCEND_ADAPTER_PATH:-$GMP/stage2/python/global_max_pool_ascend.py}"
export SCATTERMAXV1_BRIDGE="${SCATTERMAXV1_BRIDGE:-/root/zyg/build/stage2_ext/scattermaxv1_bridge.so}"
set +u
source "$GMP/stage6/env.sh" > "$LOGS/07_env.txt" 2>&1
set -u

mkdir -p "$LOGS/stage3d_pre_stage3e_backup"
cp -f /root/zyg/logs/stage3d/largetail_correctness.json \
      "$LOGS/stage3d_pre_stage3e_backup/largetail_correctness.json" 2>/dev/null || true

FAILED=0
run() {  # label logfile command...
    local label="$1" log="$2"; shift 2
    echo "[regr] $label"
    "$@" > "$log" 2>&1
    local rc=$?
    printf "%-16s exit=%-3s %s\n" "$label" "$rc" "$(grep -hE '^TOTAL |^RESULT: |AFTER_HOST_CPU_FALLBACK' "$log" | tr '\n' ' ')"
    if [ "$rc" -ne 0 ]; then FAILED=1; fi
}

run stage3a "$LOGS/07_stage3a.log" python3 "$GMP/stage3a/tests/run_stage3a_tests.py"
run stage3b "$LOGS/07_stage3b.log" python3 "$GMP/stage3b/tests/run_stage3b_boundary_tests.py"
run stage3d "$LOGS/07_stage3d.log" python3 "$GMP/stage3d/tests/run_stage3d_largetail_tests.py"
run stage3e "$LOGS/07_stage3e_largetail.log" python3 /root/zyg/stage3e/tests/run_stage3e_largetail_tests.py
run stage6_demo "$LOGS/07_stage6_demo.log" python3 "$GMP/stage6/demo_global_max_pool_ascend.py"
run stage6_tests "$LOGS/07_stage6_tests.log" python3 "$GMP/stage6/tests/run_stage6_tests.py"

echo
echo "=== gates ==="
gate=0
chk() { # label file pattern
    if grep -qE "$3" "$2"; then echo "PASS  $1 : $(grep -hE "$3" "$2" | head -1)"; else echo "FAIL  $1 : pattern '$3' not found"; gate=1; fi
}
chk "Stage 3A 34/34"      "$LOGS/07_stage3a.log"       '^TOTAL 34  PASS 34  FAIL 0'
chk "Stage 3B 45/45"      "$LOGS/07_stage3b.log"       '^TOTAL 45  PASS 45  FAIL 0'
chk "Stage 3D 9/9"        "$LOGS/07_stage3d.log"       '^TOTAL 9  PASS 9  FAIL 0'
chk "Stage 3E tail attack" "$LOGS/07_stage3e_largetail.log" '^TOTAL [0-9]+  PASS [0-9]+  FAIL 0'
chk "Stage 6 demo PASS"   "$LOGS/07_stage6_demo.log"   '^RESULT: PASS'
chk "Stage 6 20/20"       "$LOGS/07_stage6_tests.log"  '^TOTAL 20  PASS 20  FAIL 0'
chk "Stage 6 fallback"    "$LOGS/07_stage6_tests.log"  'BEFORE_HOST_CPU_FALLBACK=YES AFTER_HOST_CPU_FALLBACK=NO'

echo
echo "[DONE] script_failed=$FAILED gate_failed=$gate"
if [ "$FAILED" -ne 0 ] || [ "$gate" -ne 0 ]; then exit 1; fi
