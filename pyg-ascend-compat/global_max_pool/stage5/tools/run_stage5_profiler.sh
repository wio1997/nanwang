#!/bin/bash
# Stage 5 / profiler gate: forward + loss + backward for FP16 and BF16.
#
#   <dtype>-1 unique aligned   N=64 F=8     S=4
#   <dtype>-2 tie + N=41       N=41 F=33    S=8   (leftSrc + non-aligned)
#   <dtype>-3 non-aligned F    N=64 F=17    S=4
#   <dtype>-4 largeTail        N=40 F=48825 S=8
set -u

LOGS=${STAGE5_LOGS:-/root/zyg/logs/stage5}
OPP=${STAGE5_OPP:-/root/zyg/build/scattermax_runtime_opp/vendors/customize}
APP=${STAGE5_PROFILE_APP:-/root/zyg/stage5/tests/profile_stage5_case.py}
PROF_ROOT=${STAGE5_PROF_ROOT:-/root/zyg/profiler/stage5}
SUMMARY="$LOGS/stage5_profiler_summary.txt"
mkdir -p "$LOGS" "$PROF_ROOT"

export ASCEND_CUSTOM_OPP_PATH="${ASCEND_CUSTOM_OPP_PATH:-}"
set +u
source "$OPP/bin/set_env.bash"
set -u
export PYG_ASCEND_ADAPTER_PATH="${PYG_ASCEND_ADAPTER_PATH:-/root/zyg/global_max_pool/stage2/python/global_max_pool_ascend.py}"
export SCATTERMAXV1_BRIDGE="${SCATTERMAXV1_BRIDGE:-/root/zyg/build/stage2_ext/scattermaxv1_bridge.so}"

FORBIDDEN='507035|507011|out of range|vector core exception|aicore exception|AIV exception|npu_cpu_fallback|fall back to run on the CPU'

{
    echo "# Stage 5 profiler summary (forward + loss + backward) — formal delivery OPP"
    echo "date=$(date -u +%FT%TZ)"
    echo "OPP=$OPP"
    echo "ASCEND_CUSTOM_OPP_PATH=$ASCEND_CUSTOM_OPP_PATH"
    echo "app=$APP (2 warm-up + 3 profiled forward/loss/backward iterations per case)"
    echo
} > "$SUMMARY"

FAILED=0
for dt in fp16 bf16; do
    for spec in "1 64 8 4 unique_aligned" "2 41 33 8 tie_n41" \
                "3 64 17 4 non_aligned" "4 40 48825 8 largeTail"; do
        set -- $spec
        IDX=$1; N=$2; F=$3; S=$4; DESC=$5
        TAG="P5-${dt}-${IDX}_${DESC}"
        OUTDIR="$PROF_ROOT/$TAG"
        LOG="$LOGS/stage5_msprof_${TAG}.log"
        rm -rf "$OUTDIR"; mkdir -p "$OUTDIR"
        echo "[msprof] $TAG N=$N F=$F S=$S dtype=$dt"
        msprof --application="python3 $APP --n $N --f $F --s $S --dtype $dt --tag $TAG" \
              --output="$OUTDIR" --ai-core=on > "$LOG" 2>&1
        rc=$?
        PROF=$(find "$OUTDIR" -maxdepth 1 -mindepth 1 -type d -name "PROF_*" | head -1)
        {
            echo "## $TAG ($dt) N=$N F=$F S=$S msprof_exit=$rc"
            echo "prof_dir=$PROF"
            if [ -z "$PROF" ]; then
                echo "FAIL no profiler output"
            else
                python3 - "$PROF" <<'PY'
import csv, glob, os, sys
prof = sys.argv[1]
rows = []
for p in glob.glob(os.path.join(prof, "mindstudio_profiler_output", "op_summary_*.csv")):
    rows += list(csv.DictReader(open(p)))
sc = [r for r in rows if (r.get("OP Type") or "").strip() == "ScatterMaxV1"]
cores = sorted({(r.get("Task Type") or "").strip() for r in sc})
ai_cpu = [r for r in rows if "CPU" in (r.get("Task Type") or "").upper()]
print("GATE forward ScatterMaxV1 on device core:",
      "PASS" if (sc and all(c in ("AI_VECTOR_CORE", "MIX_AIV") for c in cores)) else "FAIL")
print(f"forward ScatterMaxV1 tasks={len(sc)} core_types={cores}")
by_op = {}
for r in rows:
    by_op.setdefault((r.get("OP Type") or "").strip(), {}).setdefault(
        (r.get("Task Type") or "").strip(), 0)
    by_op[(r.get("OP Type") or "").strip()][(r.get("Task Type") or "").strip()] += 1
print("GATE no AI_CPU tasks:", "PASS" if not ai_cpu else "FAIL")
print(f"AI_CPU_tasks={len(ai_cpu)}")
for op in sorted(by_op):
    print(f"   {op:26s} {by_op[op]}")
apicnt = 0
for p in glob.glob(os.path.join(prof, "mindstudio_profiler_output", "api_statistic_*.csv")):
    for r in csv.DictReader(open(p)):
        if "scatter_reduce" in str(r.get("API Name") or r.get("Name") or ""):
            apicnt += 1
print("GATE aten::scatter_reduce not called:", "PASS" if apicnt == 0 else "FAIL")
PY
                echo "--- app banner ---"
                grep -E "STAGE5_PROFILE" "$LOG" || echo "(no banner)"
                echo "--- forbidden markers ---"
                grep -hoE "$FORBIDDEN" "$LOG" | sort -u || echo "NONE"
            fi
            echo
        } | tee -a "$SUMMARY"
        if [ "$rc" -ne 0 ] || ! grep -q "STAGE5_PROFILE $TAG OK" "$LOG"; then
            echo "!! msprof/app failed for $TAG"; FAILED=1; break 2
        fi
        if ! tail -40 "$SUMMARY" | grep -q "GATE forward ScatterMaxV1 on device core: PASS"; then
            echo "!! $TAG forward kernel not on device"; FAILED=1; break 2
        fi
        if ! tail -40 "$SUMMARY" | grep -q "GATE no AI_CPU tasks: PASS"; then
            echo "!! $TAG has AI_CPU tasks"; FAILED=1; break 2
        fi
    done
done

echo
echo "=== aggregate ==="
grep -cE "GATE forward ScatterMaxV1 on device core: PASS" "$SUMMARY" | xargs echo "forward-on-device gates:"
grep -cE "GATE no AI_CPU tasks: PASS" "$SUMMARY" | xargs echo "no-AI_CPU gates:"
grep -cE "GATE aten::scatter_reduce not called: PASS" "$SUMMARY" | xargs echo "no-scatter_reduce gates:"
echo "[DONE] failed=$FAILED"
exit "$FAILED"
