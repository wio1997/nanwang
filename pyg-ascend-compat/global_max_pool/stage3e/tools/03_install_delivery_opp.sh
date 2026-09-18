#!/bin/bash
# Stage 3E / Task C+D: install the rebuilt package as the FORMAL runtime OPP and quarantine the
# probe packages so no runtime path can silently fall back to a probe build.
#
#   * the pre-Stage3E runtime OPP is preserved (moved, not deleted) under $ATTIC
#   * probe OPPs (stage3b_*_opp, stage3c_*, stage3d_opp_*) are moved out of /root/zyg/build
#   * the rebuilt package is installed to the exact path the repo's stage6/env.sh uses
set -u

LOGS=${STAGE3E_LOGS:-/root/zyg/logs/stage3e}
SRC=${STAGE3E_SRC:-/root/zyg/build/scattermax_probe}
PKG="$SRC/build_out/custom_opp_ubuntu_aarch64.run"
RUNTIME_ROOT=/root/zyg/build/scattermax_runtime_opp
ATTIC=/root/zyg/build/attic_stage3e
LOG="$LOGS/02_install_delivery_opp.log"
mkdir -p "$LOGS" "$ATTIC"

{
    echo "###############################################################"
    echo "# STAGE 3E FORMAL DELIVERY OPP INSTALL  $(date -u +%FT%TZ)"
    echo "###############################################################"
    echo "package     : $PKG"
    echo "install path: $RUNTIME_ROOT"
    echo "attic       : $ATTIC"
    echo
    echo "## installer info"
    "$PKG" --info 2>&1 | head -20
    echo
    echo "## 1) preserve the pre-Stage3E runtime OPP"
} > "$LOG" 2>&1

if [ -d "$RUNTIME_ROOT" ]; then
    dst="$ATTIC/scattermax_runtime_opp_pre_stage3e"
    rm -rf "$dst"
    mv "$RUNTIME_ROOT" "$dst"
    {
        echo "moved: $RUNTIME_ROOT -> $dst"
        echo "pre-Stage3E runtime OPP kernel metadata:"
        python3 - "$dst" <<'PY'
import glob, json, os, sys
for p in sorted(glob.glob(os.path.join(sys.argv[1], "vendors/customize/op_impl/ai_core/tbe/kernel/ascend910b/*/ScatterMax*.json"))):
    d = json.load(open(p))
    print(" ", os.path.basename(p), [k.get("kernelName")[-2:] for k in d.get("kernelList", [])],
          d.get("supportInfo", {}).get("tilingKey", "<absent>"))
PY
    } >> "$LOG" 2>&1
else
    echo "WARNING: $RUNTIME_ROOT did not exist" >> "$LOG"
fi

{
    echo
    echo "## 2) quarantine probe OPPs out of /root/zyg/build"
} >> "$LOG" 2>&1
for probe in stage3b_tiling_probe_opp stage3b_tiling_probe2_opp stage3c_opp_fixed stage3c_opp_tk01 \
             stage3d_opp_fix1 stage3d_opp_fix2 stage3d_opp_fix3 stage3d_opp_fix4; do
    if [ -d "/root/zyg/build/$probe" ]; then
        rm -rf "$ATTIC/$probe"
        mv "/root/zyg/build/$probe" "$ATTIC/$probe"
        echo "quarantined: /root/zyg/build/$probe -> $ATTIC/$probe" >> "$LOG"
    fi
done

{
    echo
    echo "## 3) install the rebuilt package"
} >> "$LOG" 2>&1
"$PKG" --quiet --install-path="$RUNTIME_ROOT" >> "$LOG" 2>&1
rc=$?
{
    echo "installer exit: $rc"
    echo
    echo "## 4) installed layout"
    find "$RUNTIME_ROOT" -maxdepth 4 -type d | sort | head -20
    echo
    echo "## 5) installed kernel metadata (must be _0 + _1)"
    python3 - "$RUNTIME_ROOT" <<'PY'
import glob, json, os, sys
root = sys.argv[1]
ok = False
for p in sorted(glob.glob(os.path.join(root, "vendors/customize/op_impl/ai_core/tbe/kernel/ascend910b/*/ScatterMax*.json"))):
    d = json.load(open(p))
    names = [k.get("kernelName") for k in d.get("kernelList", [])]
    tkey = d.get("supportInfo", {}).get("tilingKey")
    print(os.path.relpath(p, root))
    print("   kernelList:", names)
    print("   tilingKey :", tkey)
    if os.path.basename(p).startswith("ScatterMaxV1") and len(names) == 2 \
            and names[0].endswith("_0") and names[1].endswith("_1") and tkey == ["0", "1"]:
        ok = True
print("GATE installed kernelList==[_0,_1] and tilingKey==['0','1']:", "PASS" if ok else "FAIL")
raise SystemExit(0 if ok else 1)
PY
} >> "$LOG" 2>&1
gate=$?

{
    echo
    echo "## 6) installed set_env.bash"
    cat "$RUNTIME_ROOT/vendors/customize/bin/set_env.bash"
    echo
    echo "## 7) /root/zyg/build OPP inventory after promotion"
    find /root/zyg/build -maxdepth 1 -type d -name "*opp*" | sort
    echo "--- attic ---"
    find "$ATTIC" -maxdepth 1 -mindepth 1 | sort
} >> "$LOG" 2>&1

echo "[install] exit=$rc gate=$gate log=$LOG"
grep -E "GATE installed|kernelList:|tilingKey :|quarantined|moved:" "$LOG"
if [ "$rc" -ne 0 ] || [ "$gate" -ne 0 ]; then
    echo "[install] !! FAILED"
    exit 1
fi
echo "[install] formal runtime OPP ready at $RUNTIME_ROOT/vendors/customize"
