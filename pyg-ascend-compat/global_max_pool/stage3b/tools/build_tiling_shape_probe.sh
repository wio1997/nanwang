#!/bin/bash
# Build the Stage 3B shape-only tiling probe (host tiling only, no kernel launch).
set -e
HERE=$(cd "$(dirname "$0")" && pwd)
OUT="${1:-/root/zyg/build/stage3b_ext/tiling_shape_probe}"
CANN="${ASCEND_HOME_PATH:-/usr/local/Ascend/cann-8.5.1}"
OPP="${STAGE3B_OPP:-/root/zyg/build/scattermax_runtime_opp/vendors/customize}"
mkdir -p "$(dirname "$OUT")"
g++ -std=c++17 -O2 -Wall -o "$OUT" "$HERE/tiling_shape_probe.cpp" \
    -I"$OPP/op_api/include" -I"$CANN/include" -I"$CANN/include/aclnn" \
    -L"$CANN/lib64" -lascendcl -lnnopbase \
    -L"$OPP/op_api/lib" -lcust_opapi \
    -Wl,-rpath,"$OPP/op_api/lib" -Wl,-rpath,"$CANN/lib64"
echo "[OK] built $OUT"
