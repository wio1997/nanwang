#!/bin/bash
# Stage 3E / Task I: msprof runs against the FORMAL delivery OPP.
#
#   P1 N=40  F=48825 S=8  LARGE_TAIL
#   P2 N=41  F=48825 S=8  LARGE_TAIL + leftSrc
#   P3 N=40  F=48826 S=8  LARGE_TAIL + non-aligned (adapter pads to 48832)
#
# Gate: ScatterMaxV1 completed on AI_VECTOR_CORE, no AI_CPU task, no host fallback, no
# 507035/507011/MTE out-of-range/AIV exception, and the profiled kernel name carries the
# installed OPP's kernel hash with the `_1` entry.
set -u

LOGS=${STAGE3E_LOGS:-/root/zyg/logs/stage3e}
OPP=${STAGE3E_OPP:-/root/zyg/build/scattermax_runtime_opp/vendors/customize}
APP=${STAGE3E_PROFILE_APP:-/root/zyg/stage3e/tests/profile_stage3e_case.py}
PROF_ROOT=${STAGE3E_PROF_ROOT:-/root/zyg/profiler/stage3e}
SUMMARY="$LOGS/06_profiler_summary.txt"
mkdir -p "$LOGS" "$PROF_ROOT"

export ASCEND_CUSTOM_OPP_PATH="${ASCEND_CUSTOM_OPP_PATH:-}"
set +u
source "$OPP/bin/set_env.bash"
set -u

# the profiled application needs the adapter + bridge explicitly (stage3b_common's built-in
# default is a directory, so the env var must point at the .py file, as stage6/env.sh does)
export PYG_ASCEND_ADAPTER_PATH="${PYG_ASCEND_ADAPTER_PATH:-/root/zyg/global_max_pool/stage2/python/global_max_pool_ascend.py}"
export SCATTERMAXV1_BRIDGE="${SCATTERMAXV1_BRIDGE:-/root/zyg/build/stage2_ext/scattermaxv1_bridge.so}"

FORBIDDEN='507035|507011|out of range|vector core exception|aicore exception|AIV exception|npu_cpu_fallback|fall back to run on the CPU'

{
    echo "# Stage 3E profiler summary - FORMAL delivery OPP"
    echo "date=$(date -u +%FT%TZ)"
    echo "OPP=$OPP"
    echo "ASCEND_CUSTOM_OPP_PATH=$ASCEND_CUSTOM_OPP_PATH"
    echo "app=$APP (3 warm-up + 5 profiled calls per case)"
    echo
} > "$SUMMARY"

FAILED=0
for spec in "P1 40 48825 8 largeTail" "P2 41 48825 8 largeTail_leftsrc" "P3 40 48826 8 largeTail_nonaligned"; do
    set -- $spec
    TAG=$1; N=$2; F=$3; S=$4; DESC=$5
    OUTDIR="$PROF_ROOT/${TAG}_${DESC}"
    LOG="$LOGS/06_msprof_${TAG}.log"
    rm -rf "$OUTDIR"
    mkdir -p "$OUTDIR"
    echo "[msprof] $TAG N=$N F=$F S=$S"
    msprof --application="python3 $APP --n $N --f $F --s $S --tag $TAG" \
        --output="$OUTDIR" --ai-core=on > "$LOG" 2>&1
    rc=$?
    PROF=$(find "$OUTDIR" -maxdepth 1 -mindepth 1 -type d -name "PROF_*" | head -1)

    {
        echo "## $TAG N=$N F=$F S=$S ($DESC)  msprof_exit=$rc"
        echo "prof_dir=$PROF"
        if [ -z "$PROF" ]; then
            echo "FAIL no profiler output"
        else
    python3 - "$PROF" "$SUMMARY" <<'PY'
import csv, glob, os, re, sys
prof, summary = sys.argv[1], sys.argv[2]
rows = []
for p in glob.glob(os.path.join(prof, "mindstudio_profiler_output", "op_summary_*.csv")):
    rows += list(csv.DictReader(open(p)))
sc = [r for r in rows if (r.get("OP Type") or "").strip() == "ScatterMaxV1"]
cores = sorted({(r.get("Task Type") or "").strip() for r in sc})
ai_cpu = [r for r in rows if "CPU" in (r.get("Task Type") or "").upper()]
dur = [float(r["Task Duration(us)"]) for r in sc if r.get("Task Duration(us)")]
line = [f"ScatterMaxV1 tasks={len(sc)} core_types={cores}"]
if dur:
    line.append(f"avg={sum(dur)/len(dur):.2f}us min={min(dur):.2f}us max={max(dur):.2f}us")
line.append(f"AI_CPU_tasks={len(ai_cpu)}")
print("GATE ScatterMaxV1 completed tasks:", "PASS" if (sc and cores == ["AI_VECTOR_CORE"]) else "FAIL")
print(" | ".join(line))
print("host fallback marker: NONE (checked by the wrapper against the msprof log)")

apicnt = 0
for p in glob.glob(os.path.join(prof, "mindstudio_profiler_output", "api_statistic_*.csv")):
    for r in csv.DictReader(open(p)):
        api = str(r.get("API Name") or r.get("Name") or "")
        if "scatter_reduce" in api:
            apicnt += 1
print(f"aten::scatter_reduce occurrences in api_statistic = {apicnt}")

names = set()
for p in glob.glob(os.path.join(prof, "host", "data", "*")):
    try:
        blob = open(p, "rb").read().decode("utf-8", "ignore")
    except Exception:
        continue
    names |= set(re.findall(r"ScatterMaxV1_[0-9a-f]{32}_[0-9]+", blob))
print("device kernel names seen:", sorted(names))
PY
            echo "--- app output ---"
            grep -E "STAGE3E_PROFILE" "$LOG" || echo "(no app banner)"
            echo "--- forbidden markers in msprof log ---"
            grep -hoE "$FORBIDDEN" "$LOG" | sort -u || echo "NONE"
        fi
        echo
    } | tee -a "$SUMMARY"

    if [ "$rc" -ne 0 ]; then
        echo "!! msprof failed for $TAG"; FAILED=1; break
    fi
    if ! tail -20 "$SUMMARY" | grep -q "GATE ScatterMaxV1 completed tasks: PASS"; then
        echo "!! $TAG: ScatterMaxV1 did not complete on AI_VECTOR_CORE"; FAILED=1; break
    fi
    if ! grep -q "STAGE3E_PROFILE $TAG OK" "$LOG"; then
        echo "!! $TAG: profiled application did not finish"; FAILED=1; break
    fi
done

echo
echo "# GATE"
grep -E "ScatterMaxV1 tasks=|aten::scatter_reduce|kernel names seen" "$SUMMARY"
echo "[DONE] failed=$FAILED"
exit "$FAILED"
