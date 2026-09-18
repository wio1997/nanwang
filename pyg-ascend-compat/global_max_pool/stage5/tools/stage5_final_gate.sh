#!/bin/bash
# Stage 5 — consolidated hard gate for FP16 / BF16 (and the FP32 hard-gate regression).
set -u

LOGS=${STAGE5_LOGS:-/root/zyg/logs/stage5}
REPO=/root/zyg/nanwang
OPP=${STAGE5_OPP:-/root/zyg/build/scattermax_runtime_opp/vendors/customize}
OUT="$LOGS/stage5_final_gate.txt"
FAIL=0

exec > >(tee "$OUT") 2>&1
echo "###############################################################"
echo "# STAGE 5 FINAL GATE  $(date -u +%FT%TZ)"
echo "###############################################################"

gate() { local d="$1"; shift; if "$@" >/dev/null 2>&1; then printf "PASS  %s\n" "$d";
    else printf "FAIL  %s\n" "$d"; FAIL=1; fi; }

echo
echo "## Stage 4 remote freeze (gate 0)"
gate "remote feature HEAD == a586e48 (Stage 4 freeze pushed)" bash -c "[ \"\$(git -C $REPO rev-parse refs/remotes/origin/feat/global-max-pool-scattermax-zyg)\" = a586e489d64909d437499438b8da1b64ba2b16ec ]"
gate "remote main == a3c9ed18 (unchanged)" bash -c "[ \"\$(git -C $REPO rev-parse refs/remotes/origin/main)\" = a3c9ed18cba65cb713e26b334f94d70184960de4 ]"
gate "a586e48 is an ancestor of HEAD" bash -c "git -C $REPO merge-base --is-ancestor a586e48 HEAD"

echo
echo "## dtype capability evidence"
gate "OPP rejects fp16/bf16, accepts fp32 (status 0/161002/161002)" bash -c "
  grep -q 'fp32(control) status=0' $LOGS/scattermaxv1_dtype_probe.txt &&
  grep -q 'fp16         status=161002' $LOGS/scattermaxv1_dtype_probe.txt &&
  grep -q 'bf16         status=161002' $LOGS/scattermaxv1_dtype_probe.txt"
gate "authored op def declares DT_FLOAT only" bash -c "grep -q 'DataType({ge::DT_FLOAT})' /root/zyg/nanwang/pyg-ascend-compat/global_max_pool/stage5_fp16_bf16.md"
gate "CPU oracle: fp16+bf16 supported, dtype preserved" bash -c "
  grep -q 'fp16: 26 cases OK, out_dtype=\[.torch.float16.\], grad_dtype=\[.torch.float16.\]' $LOGS/cpu_dtype_oracle.log &&
  grep -q 'bf16: 26 cases OK' $LOGS/cpu_dtype_oracle.log"

echo
echo "## internal arithmetic determination"
gate "count = integer rounded to dtype (fp16 21/21)" grep -q "'M_round/div_dtype': '21/21'" "$LOGS/cpu_dtype_count_model.log"
gate "count = integer rounded to dtype (bf16 22/22)" grep -q "'M_round/div_dtype': '22/22'" "$LOGS/cpu_dtype_count_model.log"
gate "no dtype-vs-fp32 division discriminating pair for fp16" bash -c "! grep -q 'grad=' $LOGS/cpu_dtype_div_decide.log"
gate "primitive audit: no fallback for any fp16/bf16 primitive" grep -q "GATE no fallback and dtype preserved for all audited primitives: PASS" "$LOGS/npu_primitive_dtype_audit.log"

echo
echo "## FP16 + BF16 correctness (62 cases)"
gate "stage5 dtype matrix 62/62 PASS" grep -q "^TOTAL 62  PASS 62  FAIL 0" "$LOGS/stage5_dtype_tests.log"
gate "fp16 31/31 with zero ULP" grep -q "^fp16: TOTAL 31 PASS 31 FAIL 0 | gradient bit-exact 31/31 | max ULP 0" "$LOGS/stage5_dtype_tests.log"
gate "bf16 31/31 with zero ULP" grep -q "^bf16: TOTAL 31 PASS 31 FAIL 0 | gradient bit-exact 31/31 | max ULP 0" "$LOGS/stage5_dtype_tests.log"
gate "no mismatch/fallback in any Stage 5 dtype case" bash -c "! grep -E '\[(fp16|bf16)\] .* FAIL' $LOGS/stage5_dtype_tests.log && ! grep -q 'fallback=[1-9]' $LOGS/stage5_dtype_tests.log"
gate "real PyG E2E 25/25 PASS" grep -q "^TOTAL 25  PASS 25  FAIL 0" "$LOGS/stage5_pyg_e2e.log"
gate "E2E counters: ascend 24, original 0, dtype16 auto 22 / fwd 2" grep -q "ascend_calls=24 original_calls=0" "$LOGS/stage5_pyg_e2e.log"
gate "batch=None fp16/bf16 native + delegated" grep -q "BATCH_NONE_DTYPE: PASS" "$LOGS/batch_none_dtype_probe.log"

echo
echo "## profiler (8 runs: 4 cases x fp16/bf16, forward+loss+backward)"
gate "8x ScatterMaxV1 on device core" bash -c "[ \$(grep -c 'GATE forward ScatterMaxV1 on device core: PASS' $LOGS/stage5_profiler_summary.txt) -eq 8 ]"
gate "8x no AI_CPU tasks" bash -c "[ \$(grep -c 'GATE no AI_CPU tasks: PASS' $LOGS/stage5_profiler_summary.txt) -eq 8 ]"
gate "8x aten::scatter_reduce not called" bash -c "[ \$(grep -c 'GATE aten::scatter_reduce not called: PASS' $LOGS/stage5_profiler_summary.txt) -eq 8 ]"
gate "no fallback text in any Stage 5 profiler log" bash -c "! grep -lE 'fall back to run on the CPU|npu_cpu_fallback' $LOGS/stage5_msprof_*.log 2>/dev/null | grep -q ."

echo
echo "## FP32 regression (hard gate)"
gate "Stage 3A 34/34" grep -q "^TOTAL 34  PASS 34  FAIL 0" "$LOGS/stage5_regr_stage3a.log"
gate "Stage 3B 45/45" grep -q "^TOTAL 45  PASS 45  FAIL 0" "$LOGS/stage5_regr_stage3b.log"
gate "Stage 3D 9/9" grep -q "^TOTAL 9  PASS 9  FAIL 0" "$LOGS/stage5_regr_stage3d.log"
gate "Stage 3E 13/13" grep -q "^TOTAL 13  PASS 13  FAIL 0" "$LOGS/stage5_regr_stage3e.log"
gate "Stage 6 demo PASS" grep -q "^RESULT: PASS" "$LOGS/stage5_regr_stage6_demo.log"
gate "Stage 6 20/20 + fallback contract" bash -c "grep -q '^TOTAL 20  PASS 20  FAIL 0' $LOGS/stage5_regr_stage6_tests.log && grep -q 'BEFORE_HOST_CPU_FALLBACK=YES AFTER_HOST_CPU_FALLBACK=NO' $LOGS/stage5_regr_stage6_tests.log"
gate "Stage 2 16/16" grep -q "^TOTAL 16  PASS 16  FAIL 0" "$LOGS/stage5_regr_stage2.log"
gate "Stage 4 backward 35/35, ULP unchanged (<=1)" bash -c "grep -q '^TOTAL 35  PASS 35  FAIL 0' $LOGS/stage5_regr_stage4_bwd.log && grep -qE 'max ULP over all cases = [01]' $LOGS/stage5_regr_stage4_bwd.log"
gate "Stage 4 E2E 11/11" grep -q "^TOTAL 11  PASS 11  FAIL 0" "$LOGS/stage5_regr_stage4_e2e.log"
gate "Stage 4 profiler sanity" grep -q "\[DONE\] failed=0" "$LOGS/stage5_regr_stage4_profiler.log"
gate "delivery OPP unchanged (_0/_1, tilingKey 0,1)" bash -c "python3 -c \"
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
gate "Stage 3 + Stage 4 commits intact (d506208, a586e48 ancestors)" bash -c "git -C $REPO merge-base --is-ancestor d506208 HEAD && git -C $REPO merge-base --is-ancestor a586e48 HEAD"
gate "git status clean" bash -c "[ -z \"\$(git -C $REPO status --porcelain)\" ]"
gate "git diff --check clean" bash -c "git -C $REPO diff --check"

echo
if [ "$FAIL" -eq 0 ]; then echo "STAGE5: PASS"; else echo "STAGE5: PARTIAL / FAIL"; fi
exit "$FAIL"
