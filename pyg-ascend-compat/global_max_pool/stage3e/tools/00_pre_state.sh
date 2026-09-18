#!/bin/bash
# Stage 3E / Task A: freeze the pre-promotion state of the ScatterMaxV1 delivery chain.
#
# Records, before anything is modified:
#   * environment / versions
#   * nanwang repo branch + HEAD + status
#   * formal delivery build source hashes (and the upstream DrivingSDK reference)
#   * probe source hashes
#   * installed formal runtime OPP content + kernel metadata (_0 only, pre-fix)
#   * inventory of every OPP under /root/zyg/build (probe vs delivery)
#   * proof that the installed runtime OPP == the formal build's packaged staging tree
set -u

LOGS=${STAGE3E_LOGS:-/root/zyg/logs/stage3e}
OUT="$LOGS/00_pre_state.txt"
DELIV_SRC=/root/zyg/build/scattermax_probe
UPSTREAM_SRC=/root/zyg/DrivingSDK/kernels/scatter_max
RUNTIME_ROOT=/root/zyg/build/scattermax_runtime_opp
RUNTIME_OPP="$RUNTIME_ROOT/vendors/customize"
PACKAGE_STAGE="$DELIV_SRC/build_out/_CPack_Packages/Linux/External/custom_opp_ubuntu_aarch64.run/packages/vendors/customize"
REPO=/root/zyg/nanwang

mkdir -p "$LOGS"
exec > >(tee "$OUT") 2>&1

echo "###############################################################"
echo "# STAGE 3E PRE-PROMOTION STATE  $(date -u +%FT%TZ)"
echo "###############################################################"
echo
echo "## 1. environment"
echo "hostname      : $(hostname)"
echo "uname         : $(uname -srmo)"
echo "CANN          : $(cat /usr/local/Ascend/cann-8.5.1/version.cfg 2>/dev/null | tr '\n' ' ')"
echo "ASCEND_HOME   : ${ASCEND_HOME_PATH:-unset}"
echo "ASCEND_OPP    : ${ASCEND_OPP_PATH:-unset}"
echo "ASCEND_CUSTOM_OPP_PATH (this shell): ${ASCEND_CUSTOM_OPP_PATH-<unset>}"
echo "python        : $(python3 -V 2>&1)"
echo "torch         : $(python3 -c 'import torch;print(torch.__version__)' 2>&1)"
echo "torch_npu     : $(python3 -c 'import torch_npu;print(torch_npu.__version__)' 2>&1)"
echo "PyG           : $(python3 -c 'import torch_geometric;print(torch_geometric.__version__)' 2>&1)"
echo
echo "## 2. nanwang repo"
(cd "$REPO" && echo "branch : $(git rev-parse --abbrev-ref HEAD)" \
  && echo "HEAD   : $(git rev-parse HEAD)" \
  && echo "--- log ---" && git log --oneline -6 \
  && echo "--- status --porcelain ---" && git status --porcelain \
  && echo "--- main ---" && git rev-parse main \
  && echo "--- diff --check ---" && git diff --check && echo "diff --check: clean")
echo
echo "## 3. formal delivery build source  ($DELIV_SRC)"
md5sum "$DELIV_SRC"/op_kernel/* "$DELIV_SRC"/op_host/scatter_max_v1.cpp "$DELIV_SRC"/op_host/scatter_max_v1.h
echo "--- build script / presets present ---"
ls -l "$DELIV_SRC"/build.sh "$DELIV_SRC"/CMakeLists.txt "$DELIV_SRC"/CMakePresets.json
echo
echo "## 4. upstream DrivingSDK reference  ($UPSTREAM_SRC)"
md5sum "$UPSTREAM_SRC"/op_kernel/* "$UPSTREAM_SRC"/op_host/scatter_max_v1.cpp 2>/dev/null
(cd /root/zyg/DrivingSDK && echo "DrivingSDK HEAD: $(git rev-parse HEAD 2>/dev/null)" \
  && echo "DrivingSDK dirty files: $(git status --porcelain 2>/dev/null | wc -l)")
echo
echo "## 5. probe source copies (to be quarantined)"
for d in /root/zyg/build/scattermax_probe_fwdonly2 /root/zyg/build/stage3c_fix5 /root/zyg/build/stage3d_fix4; do
    echo "--- $d"
    md5sum "$d"/op_kernel/scatter_max_v1.cpp "$d"/op_kernel/scatter_max_v1.h 2>/dev/null
done
echo
echo "## 6. installed formal runtime OPP  ($RUNTIME_OPP)"
echo "--- file inventory + hashes ---"
(cd "$RUNTIME_OPP" && find . -type f | sort | xargs md5sum)
echo
echo "--- kernel metadata (pre-fix) ---"
python3 - "$RUNTIME_OPP" <<'PY'
import glob, json, os, sys
opp = sys.argv[1]
for p in sorted(glob.glob(os.path.join(opp, "op_impl/ai_core/tbe/kernel/ascend910b/*/ScatterMax*.json"))):
    d = json.load(open(p))
    print(os.path.relpath(p, opp))
    print("   kernelList :", [k.get("kernelName") for k in d.get("kernelList", [])])
    print("   tilingKey  :", d.get("supportInfo", {}).get("tilingKey", "<absent>"))
    print("   coreType   :", d.get("coreType"), d.get("magic"))
    print("   sha256     :", d.get("sha256"))
print("--- op-level kernel config json ---")
cfg = os.path.join(opp, "op_impl/ai_core/tbe/kernel/config/ascend910b/scatter_max_v1.json")
d = json.load(open(cfg))
print("binList entries:", len(d.get("binList", [])))
PY
echo
echo "## 7. runtime OPP == formal build package staging tree?"
python3 - "$RUNTIME_OPP" "$PACKAGE_STAGE" <<'PY'
import hashlib, os, sys
def snap(root):
    out = {}
    for dp, _, fn in os.walk(root):
        for f in fn:
            p = os.path.join(dp, f)
            out[os.path.relpath(p, root)] = hashlib.md5(open(p, "rb").read()).hexdigest()
    return out
a, b = snap(sys.argv[1]), snap(sys.argv[2])
same = [k for k in a if k in b and a[k] == b[k]]
only_installed = sorted(set(a) - set(b))
only_pkg = sorted(set(b) - set(a))
diff = sorted(k for k in a if k in b and a[k] != b[k])
print(f"installed={len(a)} packaged={len(b)} identical={len(same)} "
      f"differing={len(diff)} only_installed={len(only_installed)} only_packaged={len(only_pkg)}")
print("only_installed:", only_installed)
print("only_packaged :", only_pkg)
print("differing     :", diff)
print("VERDICT:", "installed runtime OPP is the formal build package staging tree (+installer files)"
      if not diff and not only_pkg else "MISMATCH")
PY
echo
echo "## 8. every OPP under /root/zyg/build (probe vs delivery)"
for opp in $(find /root/zyg/build -maxdepth 4 -type d -name customize 2>/dev/null | sort); do
    n=$(python3 - "$opp" <<'PY'
import glob, json, os, sys
opp = sys.argv[1]
ks = []
for p in sorted(glob.glob(os.path.join(opp, "op_impl/ai_core/tbe/kernel/ascend910b/*/ScatterMax*.json"))):
    d = json.load(open(p))
    ks.append((os.path.basename(p)[:24], [k.get("kernelName", "")[-2:] for k in d.get("kernelList", [])]))
print(" ".join(f"{n}{t}" for n, t in ks) or "<no scatter kernel>")
PY
)
    printf "%-72s %s\n" "$opp" "$n"
done
echo
echo "## 9. set_env.bash of the formal runtime OPP"
cat "$RUNTIME_OPP/bin/set_env.bash"
echo
echo "[done] pre-state captured to $OUT"
