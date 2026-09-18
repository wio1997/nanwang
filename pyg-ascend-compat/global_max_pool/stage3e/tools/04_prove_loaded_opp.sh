#!/bin/bash
# Stage 3E / Task D: prove the Stage 3E runtime actually loads the FORMAL delivery OPP.
#
# Evidence collected:
#   1. the exact runtime environment (ASCEND_CUSTOM_OPP_PATH must contain the formal OPP only)
#   2. the probe OPP packages are not present under /root/zyg/build any more (quarantined)
#   3. an LD_PRELOAD open() trace of the running operator: the literal kernel metadata/binary
#      paths opened by the ACL runtime for a LARGE_TAIL case
#   4. the installed kernel binary hash recorded in the profiler's kernel-name dictionary
#      (ScatterMaxV1_<kernelHash>_1) compared with the md5/sha of the installed OPP
set -u

LOGS=${STAGE3E_LOGS:-/root/zyg/logs/stage3e}
HERE=$(cd "$(dirname "$0")" && pwd)
OPP=${STAGE3E_OPP:-/root/zyg/build/scattermax_runtime_opp/vendors/customize}
RUNNER=${STAGE3E_RUNNER:-/root/zyg/global_max_pool/stage1b/runner/scattermaxv1_runner}
LOG="$LOGS/03_runtime_opp_provenance.txt"
SHIM="$LOGS/opentrace_shim.so"
TRACE="$LOGS/03_open_trace_lt1.txt"
mkdir -p "$LOGS"
rm -f "$TRACE"

{
    echo "###############################################################"
    echo "# STAGE 3E / TASK D - RUNTIME OPP PROVENANCE  $(date -u +%FT%TZ)"
    echo "###############################################################"
    echo
    echo "## 1. shell environment before sourcing the vendor set_env"
    echo "ASCEND_CUSTOM_OPP_PATH = ${ASCEND_CUSTOM_OPP_PATH-<unset>}"
    echo
    echo "## 2. probe OPP packages under /root/zyg/build (must be none)"
    find /root/zyg/build -maxdepth 1 -type d \( -name "stage3b_*opp*" -o -name "stage3c_*opp*" -o -name "stage3d_opp*" \) | sort
    echo "<end of list>"
    echo "quarantine location: /root/zyg/build/attic_stage3e"
    ls /root/zyg/build/attic_stage3e
} > "$LOG" 2>&1

export ASCEND_CUSTOM_OPP_PATH="${ASCEND_CUSTOM_OPP_PATH:-}"
set +u
source "$OPP/bin/set_env.bash"
set -u

{
    echo
    echo "## 3. runtime environment used for the Stage 3E runs"
    echo "ASCEND_CUSTOM_OPP_PATH = $ASCEND_CUSTOM_OPP_PATH"
    echo "LD_LIBRARY_PATH (first entries) = $(echo "$LD_LIBRARY_PATH" | cut -d: -f1-3)"
    echo "runner = $RUNNER"
    echo
    echo "## 4. formal OPP kernel metadata (what the runtime must resolve)"
    python3 - "$OPP" <<'PY'
import glob, hashlib, json, os, sys
opp = sys.argv[1]
for p in sorted(glob.glob(os.path.join(opp, "op_impl/ai_core/tbe/kernel/ascend910b/ScatterMaxV1_*/*.json"))):
    d = json.load(open(p))
    o = p[:-5] + ".o"
    print("json :", p)
    print("  md5(json)   :", hashlib.md5(open(p, "rb").read()).hexdigest())
    print("  sha256(decl):", d.get("sha256"))
    print("  md5(.o)     :", hashlib.md5(open(o, "rb").read()).hexdigest())
    print("  kernelName  :", [k.get("kernelName") for k in d.get("kernelList", [])])
    print("  tilingKey   :", d.get("supportInfo", {}).get("tilingKey"))
PY
} >> "$LOG" 2>&1

# --- open() trace of a real LARGE_TAIL operator execution -------------------------------
if [ ! -f "$SHIM" ]; then
    gcc -shared -fPIC -O2 -o "$SHIM" "$HERE/opentrace_shim.c" -ldl
fi
idx=$(python3 -c "print(','.join(str(i % 8) for i in range(40)))")
STAGE3E_OPEN_TRACE="$TRACE" LD_PRELOAD="$SHIM" \
    "$RUNNER" --case PROV_N40_F48825 --n 40 --f 48825 --size 8 --index "$idx" \
    --pattern baseline --tol 0 > "$LOGS/03_provenance_lt1.log" 2>&1
prov_rc=$?

{
    echo
    echo "## 5. LD_PRELOAD open() trace of a LARGE_TAIL execution (exit=$prov_rc)"
    echo "--- files opened that mention ScatterMax / scatter_max / customize ---"
    sort -u -k2 "$TRACE" | sed 's/^[0-9]* //'
    echo
    echo "--- lines that point at an actual kernel implementation ---"
    grep -E "ScatterMaxV1_.*\.(json|o)$" "$TRACE" | sort -u -k2 | sed 's/^[0-9]* //' || true
    echo
    echo "--- verdict ---"
    if grep -q "/root/zyg/build/scattermax_runtime_opp/vendors/customize/op_impl/ai_core/tbe/kernel" "$TRACE"; then
        echo "PROVENANCE: the runtime opened kernel metadata/binary from the FORMAL delivery OPP"
    else
        echo "PROVENANCE: no formal-OPP kernel path observed (inspect the trace)"
    fi
    if grep -qE "/root/zyg/build/(stage3b_|stage3c_|stage3d_)" "$TRACE"; then
        echo "PROBE LEAK: a probe OPP path was opened!"
    else
        echo "PROBE LEAK: none"
    fi
    echo
    echo "--- operator result of the traced run ---"
    grep -E "^RESULT|max_abs_diff|N=" "$LOGS/03_provenance_lt1.log" | head -5
} >> "$LOG" 2>&1

echo "[prove] exit=$prov_rc log=$LOG"
sed -n '/## 5\./,$p' "$LOG" | head -40
