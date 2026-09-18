#!/bin/bash
# Stage 3E / Task C: package-level verification of the promoted formal delivery OPP.
#
#   * source path / build command / package path / kernel json / binary metadata
#   * the packaged dynamic kernel source is byte-identical to the promoted build source
#   * the packaged source contains every repair
#   * old vs new kernel binary (proves the promoted package really was rebuilt)
#   * op_api / op_proto / tiling libs unchanged (bridge ABI untouched)
set -u

LOGS=${STAGE3E_LOGS:-/root/zyg/logs/stage3e}
SRC=${STAGE3E_SRC:-/root/zyg/build/scattermax_probe}
OPP=${STAGE3E_OPP:-/root/zyg/build/scattermax_runtime_opp/vendors/customize}
OLD=/root/zyg/build/attic_stage3e/scattermax_runtime_opp_pre_stage3e/vendors/customize
OUT="$LOGS/09_delivery_package_verification.txt"

exec > "$OUT" 2>&1
echo "###############################################################"
echo "# STAGE 3E DELIVERY PACKAGE VERIFICATION  $(date -u +%FT%TZ)"
echo "###############################################################"
echo
echo "formal delivery build source : $SRC"
echo "build command                : (cd $SRC && rm -rf build_out && bash build.sh)"
echo "package                      : $SRC/build_out/custom_opp_ubuntu_aarch64.run"
echo "runtime OPP (installed)      : $OPP"
echo "pre-Stage3E OPP (preserved)  : $OLD"
echo
echo "## 1. promoted delivery source"
md5sum "$SRC"/op_kernel/* "$SRC"/op_host/scatter_max_v1.cpp
echo
echo "## 2. packaged dynamic source == promoted build source"
md5sum "$OPP"/op_impl/ai_core/tbe/customize_impl/dynamic/scatter_max_v1.h "$SRC"/op_kernel/scatter_max_v1.h
md5sum "$OPP"/op_impl/ai_core/tbe/customize_impl/dynamic/scatter_max_v1.cpp "$SRC"/op_kernel/scatter_max_v1.cpp
echo
echo "## 3. every repair is present inside the PACKAGED source"
grep -n "TILING_KEY_IS(1)" "$OPP"/op_impl/ai_core/tbe/customize_impl/dynamic/scatter_max_v1.cpp
grep -n "_idxLocal.GetValue(k)\|n \* _srcBatchNum\], _srcLocal" "$OPP"/op_impl/ai_core/tbe/customize_impl/dynamic/scatter_max_v1.h
echo "byte-exact index loads (DataCopyPad(_idxLocal,...)): $(grep -c 'DataCopyPad(_idxLocal,' "$OPP"/op_impl/ai_core/tbe/customize_impl/dynamic/scatter_max_v1.h)"
echo "legacy 32B-granular index loads (DataCopy(_idxLocal,...)): $(grep -c 'DataCopy(_idxLocal,' "$OPP"/op_impl/ai_core/tbe/customize_impl/dynamic/scatter_max_v1.h || true)"
echo
echo "## 4. kernel json / binary metadata (installed OPP)"
KJSON=$(ls "$OPP"/op_impl/ai_core/tbe/kernel/ascend910b/scatter_max_v1/ScatterMaxV1_*.json)
cp -f "$KJSON" "$LOGS/09_kernel_json_scatter_max_v1.json"
cat "$KJSON"
echo
echo "## 5. kernel binary: pre-Stage3E vs promoted"
KREL=op_impl/ai_core/tbe/kernel/ascend910b/scatter_max_v1
md5sum "$OLD/$KREL"/ScatterMaxV1_*.o "$OPP/$KREL"/ScatterMaxV1_*.o
python3 - "$OLD/$KREL" "$OPP/$KREL" <<'PY'
import glob, json, hashlib, os, sys
for tag, d in (("pre-Stage3E", sys.argv[1]), ("promoted", sys.argv[2])):
    for p in sorted(glob.glob(os.path.join(d, "ScatterMaxV1_*.json"))):
        j = json.load(open(p))
        o = p[:-5] + ".o"
        print(f"{tag:12s} json_md5={hashlib.md5(open(p,'rb').read()).hexdigest()} "
              f"o_md5={hashlib.md5(open(o,'rb').read()).hexdigest()} "
              f"declared_sha256={j.get('sha256')[:16]}... "
              f"kernelList={[k['kernelName'][-2:] for k in j.get('kernelList',[])]} "
              f"tilingKey={j.get('supportInfo',{}).get('tilingKey')}")
PY
echo
echo "## 6. op_api / op_proto / tiling libs: unchanged by the promotion (bridge ABI)"
for f in op_api/lib/libcust_opapi.so op_api/include/aclnn_scatter_max_v1.h \
         op_proto/lib/linux/aarch64/libcust_opsproto_rt2.0.so \
         op_impl/ai_core/tbe/op_tiling/lib/linux/aarch64/libcust_opmaster_rt2.0.so; do
    printf "%-72s old=%s new=%s\n" "$f" \
        "$(md5sum "$OLD/$f" 2>/dev/null | cut -c1-32)" "$(md5sum "$OPP/$f" 2>/dev/null | cut -c1-32)"
done
echo
echo "## 7. runtime OPP file inventory (promoted)"
(cd "$OPP" && find . -type f | sort | xargs md5sum)
echo
echo "[done] $OUT"
