#!/bin/bash
# Stage 4 — consolidated hard-gate check (section 24) over all Stage 4 evidence.
set -u

LOGS=${STAGE4_LOGS:-/root/zyg/logs/stage4}
REPO=/root/zyg/nanwang
OPP=${STAGE4_OPP:-/root/zyg/build/scattermax_runtime_opp/vendors/customize}
OUT="$LOGS/stage4_final_gate.txt"
FAIL=0

exec > >(tee "$OUT") 2>&1
echo "###############################################################"
echo "# STAGE 4 FINAL GATE  $(date -u +%FT%TZ)"
echo "###############################################################"

gate() {
    local desc="$1"; shift
    if "$@" >/dev/null 2>&1; then printf "PASS  %s\n" "$desc"; else
        printf "FAIL  %s\n" "$desc"; FAIL=1; fi
}

echo
echo "## CPU oracle / frozen contract"
gate "cpu_gradient_oracle.md exists and documents G0-G10 + batch=None" bash -c "grep -q 'G0 unique max' /root/zyg/nanwang/pyg-ascend-compat/global_max_pool/stage4_cpu_gradient_oracle.md && grep -q 'batch=None' /root/zyg/nanwang/pyg-ascend-compat/global_max_pool/stage4_cpu_gradient_oracle.md"
gate "contract model exact on 300/300 randomized cases" grep -q "exact match with the real PyG API on all 300 random cases" "$LOGS/cpu_contract_model.log"
gate "oracle measured with real PyG 2.8.0.post1 / torch 2.9.0" bash -c "grep -q 'torch_geometric : 2.8.0.post1' $LOGS/cpu_gradient_oracle.log && grep -q 'torch      : 2.9.0+cpu' $LOGS/cpu_gradient_oracle.log"
gate "self-slot quirk proven by controlled experiment" grep -q "init=-1000    include_self=False out=\[\[0.0\]\] grad=\[\[0.5\], \[0.5\]\]" "$LOGS/cpu_tie_quirk_probe.log"
gate "batch=None: CPU == NPU forward+backward, native, compat delegates" bash -c "grep -q 'RESULT PASS' $LOGS/batch_none_probe.log && grep -q 'delegated to original PyG for batch=None: True' $LOGS/batch_none_probe.log"

echo
echo "## NPU backward correctness vs CPU oracle (35 cases)"
gate "backward matrix 35/35 PASS" grep -q "^TOTAL 35  PASS 35  FAIL 0" "$LOGS/stage4_backward_tests.log"
gate "max ULP over all cases <= 1" bash -c "grep -q 'max ULP over all cases = 1' $LOGS/stage4_backward_tests.log || grep -q 'max ULP over all cases = 0' $LOGS/stage4_backward_tests.log"
for c in B1_unique_max B2_two_way_tie B3_three_way_tie B4_per_feature_ties B5_weighted_upstream \
         B6_multi_group_repeat B7_explicit_size_empty B8_negative_only B9a_all_neg_inf \
         B10a_pm_zero B11a_one_nan B_LT_N40_F48825 B_N41; do
    gate "$c PASS" grep -qE "^$c +PASS" "$LOGS/stage4_backward_tests.log"
done
gate "non-aligned F=1/7/8/9/17/33 all PASS" bash -c "[ \$(grep -cE '^B_F(1|7|8|9|17|33)_non_aligned +PASS' $LOGS/stage4_backward_tests.log) -eq 6 ]"
gate "N=39/40/41/4097 all PASS" bash -c "[ \$(grep -cE '^B_N(39|40|41|4097) +PASS' $LOGS/stage4_backward_tests.log) -eq 4 ]"
gate "largeTail backward N=40/N=41 F=48825 PASS" bash -c "[ \$(grep -cE '^B_LT_N(40|41)_F48825 +PASS' $LOGS/stage4_backward_tests.log) -eq 2 ]"
gate "gradient invariants clean in every case" bash -c "! grep -E '^B.*FAIL' $LOGS/stage4_backward_tests.log && ! grep -q 'invariants(nonwinner!=0:[1-9]' $LOGS/stage4_backward_tests.log && ! grep -q 'sum:[1-9]' $LOGS/stage4_backward_tests.log"

echo
echo "## real PyG API end-to-end backward"
gate "PyG E2E 11/11 PASS" grep -q "^TOTAL 11  PASS 11  FAIL 0" "$LOGS/stage4_pyg_e2e.log"
gate "counters ascend_calls=10 original_calls=0 autograd_calls=10" grep -q "counters: ascend_calls=10 original_calls=0 autograd_calls=10 -> PASS" "$LOGS/stage4_pyg_e2e.log"

echo
echo "## profiler / fallback gates (forward + loss + backward)"
gate "4 profiled cases, ScatterMaxV1 forward AI_VECTOR_CORE" bash -c "[ \$(grep -c 'GATE forward ScatterMaxV1 on AI_VECTOR_CORE: PASS' $LOGS/stage4_profiler_summary.txt) -eq 4 ]"
gate "no AI_CPU tasks in any profiled run" bash -c "[ \$(grep -c 'no AI_CPU tasks anywhere in the profiled forward+backward: PASS' $LOGS/stage4_profiler_summary.txt) -eq 4 ] && ! grep -q 'AI_CPU_tasks=[1-9]' $LOGS/stage4_profiler_summary.txt"
gate "aten::scatter_reduce not called in any profiled run" bash -c "! grep -q 'occurrences=[1-9]' $LOGS/stage4_profiler_summary.txt"
gate "primitive audit: no CPU task type" grep -q "GATE no CPU task type in any profiled forward+backward operator: PASS" "$LOGS/stage4_primitive_audit.txt"
gate "no host-fallback text in any Stage 4 log" bash -c "! grep -lE 'fall back to run on the CPU|npu_cpu_fallback' $LOGS/stage4_backward_tests.log $LOGS/stage4_pyg_e2e.log $LOGS/stage4_msprof_P-BWD*.log 2>/dev/null | grep -q ."

echo
echo "## forward regressions (unchanged)"
gate "Stage 3A 34/34" grep -q "^TOTAL 34  PASS 34  FAIL 0" "$LOGS/stage4_regr_stage3a.log"
gate "Stage 3B 45/45" grep -q "^TOTAL 45  PASS 45  FAIL 0" "$LOGS/stage4_regr_stage3b.log"
gate "Stage 3D 9/9" grep -q "^TOTAL 9  PASS 9  FAIL 0" "$LOGS/stage4_regr_stage3d.log"
gate "Stage 3E largeTail 13/13" grep -q "^TOTAL 13  PASS 13  FAIL 0" "$LOGS/stage4_regr_stage3e.log"
gate "Stage 6 demo PASS" grep -q "^RESULT: PASS" "$LOGS/stage4_regr_stage6_demo.log"
gate "Stage 6 20/20" grep -q "^TOTAL 20  PASS 20  FAIL 0" "$LOGS/stage4_regr_stage6_tests.log"
gate "Stage 6 before/after fallback" grep -q "BEFORE_HOST_CPU_FALLBACK=YES AFTER_HOST_CPU_FALLBACK=NO" "$LOGS/stage4_regr_stage6_tests.log"
gate "Stage 2 adapter suite 16/16" grep -q "^TOTAL 16  PASS 16  FAIL 0" "$LOGS/stage4_regr_stage2.log"
gate "delivery OPP unchanged (kernel _0/_1)" bash -c "python3 -c \"
import glob,json
ok=False
for p in glob.glob('$OPP/op_impl/ai_core/tbe/kernel/ascend910b/scatter_max_v1/ScatterMaxV1_*.json'):
    d=json.load(open(p)); n=[k['kernelName'] for k in d['kernelList']]
    ok = len(n)==2 and d['supportInfo']['tilingKey']==['0','1']
raise SystemExit(0 if ok else 1)\""

echo
echo "## git"
MAIN_PRE=$(sed -n '/^--- main ---$/{n;p;}' /root/zyg/logs/stage3e/00_pre_state.txt)
gate "branch feat/global-max-pool-scattermax-zyg" bash -c "[ \$(git -C $REPO rev-parse --abbrev-ref HEAD) = feat/global-max-pool-scattermax-zyg ]"
gate "main unchanged ($MAIN_PRE)" bash -c "[ \$(git -C $REPO rev-parse main) = $MAIN_PRE ]"
gate "d506208 is an ancestor of HEAD (Stage 4 continues the chain)" bash -c "git -C $REPO merge-base --is-ancestor d506208 HEAD"
gate "no Stage 3E commit rewritten (430bf81, d506208 intact)" bash -c "git -C $REPO merge-base --is-ancestor 430bf81 HEAD && git -C $REPO merge-base --is-ancestor d506208 HEAD"
gate "git status clean" bash -c "[ -z \"\$(git -C $REPO status --porcelain)\" ]"
gate "git diff --check clean" bash -c "git -C $REPO diff --check"

echo
if [ "$FAIL" -eq 0 ]; then echo "STAGE4: PASS"; else echo "STAGE4: PARTIAL / FAIL"; fi
exit "$FAIL"
