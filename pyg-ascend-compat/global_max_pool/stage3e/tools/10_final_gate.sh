#!/bin/bash
# Stage 3E: consolidated hard-gate check over all Stage 3E evidence files.
set -u

LOGS=${STAGE3E_LOGS:-/root/zyg/logs/stage3e}
OPP=${STAGE3E_OPP:-/root/zyg/build/scattermax_runtime_opp/vendors/customize}
OUT="$LOGS/10_final_gate.txt"
FAIL=0

exec > >(tee "$OUT") 2>&1
echo "###############################################################"
echo "# STAGE 3E FINAL GATE  $(date -u +%FT%TZ)"
echo "###############################################################"

gate() { # description, command...
    local desc="$1"; shift
    if "$@" >/dev/null 2>&1; then
        printf "PASS  %s\n" "$desc"
    else
        printf "FAIL  %s\n" "$desc"
        FAIL=1
    fi
}

echo
echo "## formal OPP content"
grep -q "'ScatterMaxV1_7d55161965c898907fdb3028d01c7c76_1'" /dev/null 2>/dev/null
gate "kernelList contains _0 and _1" bash -c "python3 -c \"
import json,glob;
ok=False
for p in glob.glob('$OPP/op_impl/ai_core/tbe/kernel/ascend910b/scatter_max_v1/ScatterMaxV1_*.json'):
    d=json.load(open(p)); n=[k['kernelName'] for k in d['kernelList']]
    ok = len(n)==2 and n[0].endswith('_0') and n[1].endswith('_1') and d['supportInfo']['tilingKey']==['0','1']
raise SystemExit(0 if ok else 1)\""
gate "promoted source contains TILING_KEY_IS(1)" grep -q "TILING_KEY_IS(1)" "$OPP/op_impl/ai_core/tbe/customize_impl/dynamic/scatter_max_v1.cpp"
gate "promoted source contains GetValue(k)" grep -q "GetValue(k)" "$OPP/op_impl/ai_core/tbe/customize_impl/dynamic/scatter_max_v1.h"
gate "promoted source contains write offset + n*_srcBatchNum" grep -q "n \* _srcBatchNum\], _srcLocal" "$OPP/op_impl/ai_core/tbe/customize_impl/dynamic/scatter_max_v1.h"
gate "index GM over-read removed (4 byte-exact index loads, 0 legacy)" bash -c "[ \$(grep -c 'DataCopyPad(_idxLocal,' $OPP/op_impl/ai_core/tbe/customize_impl/dynamic/scatter_max_v1.h) -eq 4 ] && ! grep -q 'DataCopy(_idxLocal,' $OPP/op_impl/ai_core/tbe/customize_impl/dynamic/scatter_max_v1.h"
gate "probe OPPs absent from /root/zyg/build" bash -c "! find /root/zyg/build -maxdepth 1 -type d \\( -name 'stage3b_*opp*' -o -name 'stage3c_*opp*' -o -name 'stage3d_opp*' \\) | grep -q ."

echo
echo "## runtime provenance (Task D)"
gate "runtime opened only the formal OPP kernel binary" grep -q "PROVENANCE: the runtime opened kernel metadata/binary from the FORMAL delivery OPP" "$LOGS/03_runtime_opp_provenance.txt"
gate "no probe OPP path opened" grep -q "PROBE LEAK: none" "$LOGS/03_runtime_opp_provenance.txt"

echo
echo "## largeTail matrix (Task F)"
for tag in LT0_N40_F48824_S8 LT1_N40_F48825_S8 LT2_N40_F48826_S8 LT3_N41_F48825_S8 LT4_N80_F48825_S8 \
           LT5_N81_F48825_S8 LT6_N320_F48825_S8 LTA1_N40_F48832_S8 LTA2_N40_F48960_S8; do
    gate "$tag PASS max_diff=0" grep -qE "^$tag +0 +PASS +max_abs_diff=0" "$LOGS/04_largetail_matrix_summary.txt"
done
gate "one-shape-per-process series completed (failed=0)" bash -c "! grep -q 'STOPPED_AT=' $LOGS/04_largetail_matrix_summary.txt"

echo
echo "## tail attack + PyG E2E (Task G/H)"
gate "tail-attack suite 13/13" grep -q "^TOTAL 13  PASS 13  FAIL 0" "$LOGS/05_largetail_tests.log"
gate "PyG E2E_LT1/2/3 PASS max_diff=0" bash -c "[ \$(grep -c 'E2E_LT[123].*PASS.*max_diff=0.0' $LOGS/05_largetail_tests.log) -eq 3 ]"
gate "PyG ascend_calls=3 original_calls=0" grep -q "ascend_calls=3 original_calls=0" "$LOGS/05_largetail_tests.log"

echo
echo "## profiler (Task I)"
gate "3 profiled cases completed on AI_VECTOR_CORE" bash -c "[ \$(grep -c 'GATE ScatterMaxV1 completed tasks: PASS' $LOGS/06_profiler_summary.txt) -eq 3 ]"
gate "profiled kernel name carries entry _1" bash -c "[ \$(grep -c 'ScatterMaxV1_7d55161965c898907fdb3028d01c7c76_1' $LOGS/06_profiler_summary.txt) -ge 3 ]"
gate "AI_CPU tasks = 0 in all profiled runs" bash -c "! grep -q 'AI_CPU_tasks=[^0]' $LOGS/06_profiler_summary.txt"
gate "aten::scatter_reduce not called (0 api_statistic hits)" bash -c "! grep -q 'occurrences in api_statistic = [1-9]' $LOGS/06_profiler_summary.txt"

echo
echo "## regressions (Task J)"
gate "Stage 3A 34/34" grep -q "^TOTAL 34  PASS 34  FAIL 0" "$LOGS/07_stage3a.log"
gate "Stage 3B 45/45" grep -q "^TOTAL 45  PASS 45  FAIL 0" "$LOGS/07_stage3b.log"
gate "Stage 3D 9/9"   grep -q "^TOTAL 9  PASS 9  FAIL 0"   "$LOGS/07_stage3d.log"
gate "Stage 6 demo PASS" grep -q "^RESULT: PASS" "$LOGS/07_stage6_demo.log"
gate "Stage 6 20/20" grep -q "^TOTAL 20  PASS 20  FAIL 0" "$LOGS/07_stage6_tests.log"
gate "Stage 6 host fallback BEFORE=YES AFTER=NO" grep -q "BEFORE_HOST_CPU_FALLBACK=YES AFTER_HOST_CPU_FALLBACK=NO" "$LOGS/07_stage6_tests.log"

echo
echo "## failure-marker sweep over every Stage 3E run log"
MARKERS='507035|507011|MTE instruction is out of range|vector core exception|aicore exception|AIV exception|npu_cpu_fallback|fall back to run on the CPU'
hits=$(grep -lE "$MARKERS" "$LOGS"/04_*.log "$LOGS"/05_*.log "$LOGS"/06_*.log "$LOGS"/07_*.log 2>/dev/null || true)
if [ -z "$hits" ]; then
    echo "PASS  507035 / 507011 / MTE OOB / AIV exception / host fallback: NONE in any Stage 3E runtime log"
else
    echo "FAIL  markers found in: $hits"; FAIL=1
fi

echo
echo "## extreme combined shape"
gate "N=163800 F=44736 SMALL_TAIL / F=44737 LARGE_TAIL (shape-only)" bash -c "grep -q 'N=163800 F=44736 .*TILING_KEY=SMALL_TAIL(0)' $LOGS/08_extreme_shape_tiling.txt && grep -q 'N=163800 F=44737 .*TILING_KEY=LARGE_TAIL(1)' $LOGS/08_extreme_shape_tiling.txt"

echo
echo "## git"
REPO=/root/zyg/nanwang
gate "branch feat/global-max-pool-scattermax-zyg" bash -c "[ \$(git -C $REPO rev-parse --abbrev-ref HEAD) = feat/global-max-pool-scattermax-zyg ]"
gate "main unchanged (5816ef6ae20759cf101db21834802fd33f2f7fff)" bash -c "[ \$(git -C $REPO rev-parse main) = 5816ef6ae20759cf101db21834802fd33f2f7fff ]"
gate "history preserves 5816ef6 -> 1a8df65 -> 382301e -> 028112b" bash -c "git -C $REPO merge-base --is-ancestor 5816ef6 1a8df65 && git -C $REPO merge-base --is-ancestor 1a8df65 382301e && git -C $REPO merge-base --is-ancestor 382301e 028112b"
gate "git diff --check clean" bash -c "git -C $REPO diff --check"

echo
if [ "$FAIL" -eq 0 ]; then
    echo "STAGE3E: PASS"
else
    echo "STAGE3E: PARTIAL / FAIL"
fi
exit "$FAIL"
