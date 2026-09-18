#!/bin/bash
# Stage 3E / Task C: clean rebuild of the FORMAL delivery OPP from the promoted delivery source.
#
#   source : /root/zyg/build/scattermax_probe          (formal delivery build source)
#   command: bash build.sh                              (CMakePresets default -> package)
#   output : build_out/custom_opp_ubuntu_aarch64.run     (CPack custom OPP package)
#
# Everything (command line, stdout, stderr, timing, source hashes, kernel metadata) is persisted
# under /root/zyg/logs/stage3e/.
set -u

LOGS=${STAGE3E_LOGS:-/root/zyg/logs/stage3e}
SRC=${STAGE3E_SRC:-/root/zyg/build/scattermax_probe}
LOG="$LOGS/01_build_delivery_opp.log"
mkdir -p "$LOGS"

# The AscendC `opc` front-end (kernel compilation) imports `decorator`, which is NOT part of the
# python3.11.14 site-packages in this image.  The CANN-8.5.1 toolchain in this container is used
# with the vendored dependency directory that the earlier Stage 1B/3C/3D builds used as well.
STAGE3E_BUILD_DEPS="${STAGE3E_BUILD_DEPS:-/root/pyg_feasibility/R009-scattermax-raw-callability/deps}"
if [ -d "$STAGE3E_BUILD_DEPS" ]; then
    export PYTHONPATH="$STAGE3E_BUILD_DEPS:${PYTHONPATH:-}"
fi
if ! python3 -c "import decorator" 2>/dev/null; then
    echo "FATAL: python3 cannot import 'decorator' (needed by opc); set STAGE3E_BUILD_DEPS" >&2
    exit 3
fi

{
    echo "###############################################################"
    echo "# STAGE 3E FORMAL DELIVERY OPP CLEAN REBUILD  $(date -u +%FT%TZ)"
    echo "###############################################################"
    echo "source tree : $SRC"
    echo "build script: $SRC/build.sh"
    echo "command     : (cd $SRC && rm -rf build_out && bash build.sh)"
    echo "env         : ASCEND_HOME_PATH=${ASCEND_HOME_PATH:-unset} ASCEND_CUSTOM_OPP_PATH=${ASCEND_CUSTOM_OPP_PATH-<unset>}"
    echo "PYTHONPATH  : ${PYTHONPATH:-}"
    echo "decorator   : $(python3 -c 'import decorator;print(decorator.__version__)')"
    echo
    echo "## source hashes handed to the compiler"
    md5sum "$SRC"/op_kernel/* "$SRC"/op_host/scatter_max_v1.cpp
    echo
    echo "## promoted source diff vs the upstream DrivingSDK reference"
    diff -u /root/zyg/DrivingSDK/kernels/scatter_max/op_kernel/scatter_max_v1.cpp "$SRC/op_kernel/scatter_max_v1.cpp" || true
    diff -u /root/zyg/DrivingSDK/kernels/scatter_max/op_kernel/CMakeLists.txt "$SRC/op_kernel/CMakeLists.txt" || true
    diff -u /root/zyg/DrivingSDK/kernels/scatter_max/op_kernel/scatter_max_v1.h "$SRC/op_kernel/scatter_max_v1.h" || true
    echo
    echo "## build output"
} > "$LOG" 2>&1

start=$(date +%s)
(cd "$SRC" && rm -rf build_out && bash build.sh) >> "$LOG" 2>&1
rc=$?
end=$(date +%s)

{
    echo
    echo "## build exit code: $rc   elapsed: $((end - start))s"
} >> "$LOG"

if [ "$rc" -ne 0 ]; then
    echo "!! build failed (exit=$rc); see $LOG"
    tail -30 "$LOG"
    exit "$rc"
fi

PKG="$SRC/build_out/custom_opp_ubuntu_aarch64.run"
STAGE="$SRC/build_out/_CPack_Packages/Linux/External/custom_opp_ubuntu_aarch64.run/packages/vendors/customize"
{
    echo
    echo "## package"
    ls -l "$PKG" || echo "MISSING PACKAGE"
    echo
    echo "## kernel metadata in the rebuilt package"
    python3 - "$STAGE" <<'PY'
import glob, json, os, sys
root = sys.argv[1]
ok = False
for p in sorted(glob.glob(os.path.join(root, "op_impl/ai_core/tbe/kernel/ascend910b/*/ScatterMax*.json"))):
    d = json.load(open(p))
    names = [k.get("kernelName") for k in d.get("kernelList", [])]
    tkey = d.get("supportInfo", {}).get("tilingKey")
    print(os.path.relpath(p, root))
    print("   kernelList:", names)
    print("   tilingKey :", tkey)
    print("   coreType  :", d.get("coreType"), d.get("core_type"), d.get("magic"))
    print("   sha256    :", d.get("sha256"))
    if "ScatterMaxV1_" in os.path.basename(p) and not os.path.basename(p).startswith("ScatterMaxArgmax"):
        if len(names) == 2 and names[0].endswith("_0") and names[1].endswith("_1") and tkey == ["0", "1"]:
            ok = True
print()
print("GATE kernelList==[_0,_1] and tilingKey==['0','1']:", "PASS" if ok else "FAIL")
raise SystemExit(0 if ok else 1)
PY
} >> "$LOG" 2>&1
gate=$?

echo
echo "[build] exit=$rc elapsed=$((end - start))s log=$LOG"
echo "[build] package: $PKG"
grep -E "GATE kernelList|kernelList:|tilingKey :" "$LOG" | tail -12
if [ "$gate" -ne 0 ]; then
    echo "[build] !! kernel metadata gate FAILED"
    exit 1
fi
echo "[build] kernel metadata gate PASS"
