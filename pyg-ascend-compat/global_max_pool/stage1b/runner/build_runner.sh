#!/bin/bash
# Build the Stage 1B ACLNN runner against the isolated (non-system) custom OPP.
set -e

HERE=$(cd "$(dirname "$0")" && pwd)
CANN="${ASCEND_HOME_PATH:-/usr/local/Ascend/cann-8.5.1}"
OPP="${STAGE1B_OPP:-/root/zyg/build/scattermax_runtime_opp/vendors/customize}"
OUT="${1:-$HERE/scattermaxv1_runner}"

if [ ! -d "$OPP/op_api/include" ]; then
    echo "[ERROR] custom OPP not found at $OPP (set STAGE1B_OPP)" >&2
    exit 1
fi

g++ -std=c++17 -O2 -Wall -o "$OUT" "$HERE/scattermaxv1_runner.cpp" \
    -I"$OPP/op_api/include" \
    -I"$CANN/include" \
    -I"$CANN/include/aclnn" \
    -L"$CANN/lib64" \
    -lascendcl -lnnopbase \
    -L"$OPP/op_api/lib" -lcust_opapi \
    -Wl,-rpath,"$OPP/op_api/lib" \
    -Wl,-rpath,"$CANN/lib64"

echo "[OK] built $OUT"
echo "[INFO] CANN=$CANN"
echo "[INFO] OPP=$OPP"
