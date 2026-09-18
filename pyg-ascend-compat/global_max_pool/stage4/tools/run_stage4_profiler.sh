#!/bin/bash
# Stage 4 / profiler gate + NPU primitive audit.
#
#   P-BWD1 unique max   N=64   F=8     S=4
#   P-BWD2 tie case     N=41   F=33    S=8   (leftSrc + non-aligned F)
#   P-BWD3 non-aligned  N=64   F=17    S=4
#   P-BWD4 largeTail    N=40   F=48825 S=8
#
# Each run profiles forward + loss + backward and must show
#   * ScatterMaxV1 forward tasks on AI_VECTOR_CORE
#   * every backward primitive on a device core (no AI_CPU, no host fallback)
#   * no aten::scatter_reduce
set -u

LOGS=${STAGE4_LOGS:-/root/zyg/logs/stage4}
OPP=${STAGE4_OPP:-/root/zyg/build/scattermax_runtime_opp/vendors/customize}
APP=${STAGE4_PROFILE_APP:-/root/zyg/stage4/tests/profile_stage4_backward.py}
PROF_ROOT=${STAGE4_PROF_ROOT:-/root/zyg/profiler/stage4}
SUMMARY="$LOGS/stage4_profiler_summary.txt"
mkdir -p "$LOGS" "$PROF_ROOT"

export ASCEND_CUSTOM_OPP_PATH="${ASCEND_CUSTOM_OPP_PATH:-}"
set +u
source "$OPP/bin/set_env.bash"
set -u
export PYG_ASCEND_ADAPTER_PATH="${PYG_ASCEND_ADAPTER_PATH:-/root/zyg/global_max_pool/stage2/python/global_max_pool_ascend.py}"
export SCATTERMAXV1_BRIDGE="${SCATTERMAXV1_BRIDGE:-/root/zyg/build/stage2_ext/scattermaxv1_bridge.so}"

FORBIDDEN='507035|507011|out of range|vector core exception|aicore exception|AIV exception|npu_cpu_fallback|fall back to run on the CPU'

{
    echo "# Stage 4 profiler summary (forward + loss + backward) — formal delivery OPP"
    echo "date=$(date -u +%FT%TZ)"
    echo "OPP=$OPP"
    echo "ASCEND_CUSTOM_OPP_PATH=$ASCEND_CUSTOM_OPP_PATH"
    echo "app=$APP (2 warm-up + 3 profiled forward/loss/backward iterations)"
    echo
} > "$SUMMARY"

FAILED=0
for spec in "P-BWD1 64 8 4 unique_max" "P-BWD2 41 33 8 tie_leftsrc" \
            "P-BWD3 64 17 4 non_aligned" "P-BWD4 40 48825 8 largeTail"; do
    set -- $spec
    TAG=$1; N=$2; F=$3; S=$4; DESC=$5
    OUTDIR="$PROF_ROOT/${TAG}_${DESC}"
    LOG="$LOGS/stage4_msprof_${TAG}.log"
    rm -rf "$OUTDIR"; mkdir -p "$OUTDIR"
    echo "[msprof] $TAG N=$N F=$F S=$S ($DESC)"
    msprof --application="python3 $APP --n $N --f $F --s $S --tag $TAG" \
          --output="$OUTDIR" --ai-core=on > "$LOG" 2>&1
    rc=$?
    PROF=$(find "$OUTDIR" -maxdepth 1 -mindepth 1 -type d -name "PROF_*" | head -1)

    {
        echo "## $TAG N=$N F=$F S=$S ($DESC) msprof_exit=$rc"
        echo "prof_dir=$PROF"
        if [ -z "$PROF" ]; then
            echo "FAIL no profiler output"
        else
            python3 - "$PROF" <<'PY'
import csv, glob, json, os, sys
prof = sys.argv[1]
rows = []
for p in glob.glob(os.path.join(prof, "mindstudio_profiler_output", "op_summary_*.csv")):
    rows += list(csv.DictReader(open(p)))
sc = [r for r in rows if (r.get("OP Type") or "").strip() == "ScatterMaxV1"]
cores = sorted({(r.get("Task Type") or "").strip() for r in sc})
ai_cpu = [r for r in rows if "CPU" in (r.get("Task Type") or "").upper()]
dur = [float(r["Task Duration(us)"]) for r in sc if r.get("Task Duration(us)")]
print("GATE forward ScatterMaxV1 on AI_VECTOR_CORE:",
      "PASS" if (sc and cores == ["AI_VECTOR_CORE"]) else "FAIL")
print(f"forward ScatterMaxV1 tasks={len(sc)} core_types={cores}"
      + (f" avg={sum(dur)/len(dur):.2f}us" if dur else ""))

by_op = {}
for r in rows:
    op = (r.get("OP Type") or "").strip()
    core = (r.get("Task Type") or "").strip()
    by_op.setdefault(op, {}).setdefault(core, 0)
    by_op[op][core] += 1
print("GATE no AI_CPU tasks anywhere in the profiled forward+backward:",
      "PASS" if not ai_cpu else "FAIL")
print(f"AI_CPU_tasks={len(ai_cpu)}")
print("device ops seen (forward + backward):")
for op in sorted(by_op):
    print(f"   {op:28s} {by_op[op]}")

apicnt = 0
apis = set()
for p in glob.glob(os.path.join(prof, "mindstudio_profiler_output", "api_statistic_*.csv")):
    for r in csv.DictReader(open(p)):
        api = str(r.get("API Name") or r.get("Name") or "")
        apis.add(api)
        if "scatter_reduce" in api:
            apicnt += 1
print("GATE aten::scatter_reduce not called:", "PASS" if apicnt == 0 else "FAIL")
print(f"aten::scatter_reduce occurrences={apicnt}")
PY
            echo "--- app output ---"
            grep -E "STAGE4_PROFILE" "$LOG" || echo "(no app banner)"
            echo "--- forbidden markers ---"
            grep -hoE "$FORBIDDEN" "$LOG" | sort -u || echo "NONE"
        fi
        echo
    } | tee -a "$SUMMARY"

    if [ "$rc" -ne 0 ] || ! grep -q "STAGE4_PROFILE $TAG OK" "$LOG"; then
        echo "!! msprof/app failed for $TAG"; FAILED=1; break
    fi
    if ! tail -40 "$SUMMARY" | grep -q "GATE forward ScatterMaxV1 on AI_VECTOR_CORE: PASS"; then
        echo "!! $TAG: forward kernel not AI_VECTOR_CORE"; FAILED=1; break
    fi
    if ! tail -60 "$SUMMARY" | grep -q "GATE no AI_CPU tasks anywhere in the profiled forward+backward: PASS"; then
        echo "!! $TAG: AI_CPU tasks present"; FAILED=1; break
    fi
done

echo
echo "=== aggregate gates ==="
grep -E "GATE |AI_CPU_tasks=|occurrences=" "$SUMMARY"
echo "[DONE] failed=$FAILED"
exit "$FAILED"
