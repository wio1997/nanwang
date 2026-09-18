#!/bin/bash
# Build the Stage 2 PyTorch <-> ACLNN bridge.
#
# Strategy (BRIDGE_STRATEGY = A, minimal PyTorch C++ extension):
#   * pybind11 comes from PyTorch's own bundled headers (torch/include/pybind11)
#   * compiled with plain g++ : `torch.utils.cpp_extension.load()` needs ninja, which is NOT
#     installed in this image and installing it is out of scope for Stage 2
#   * no new Python/system packages are required
#
# Usage: bash build_bridge.sh [output.so]
set -e

HERE=$(cd "$(dirname "$0")" && pwd)
OUT="${1:-/root/zyg/build/stage2_ext/scattermaxv1_bridge.so}"

PY_ROOT=$(python3 -c 'import sys; print(sys.base_prefix)')
TORCH=$(python3 -c 'import torch, os; print(os.path.dirname(torch.__file__))')
TORCH_NPU=$(python3 -c 'import torch_npu, os; print(os.path.dirname(torch_npu.__file__))')
CANN="${ASCEND_HOME_PATH:-/usr/local/Ascend/cann-8.5.1}"
OPP="${STAGE2_OPP:-/root/zyg/build/scattermax_runtime_opp/vendors/customize}"
ABI=$(python3 -c 'import torch; print(int(torch._C._GLIBCXX_USE_CXX11_ABI))')

mkdir -p "$(dirname "$OUT")"

g++ -shared -fPIC -std=c++17 -O2 -Wall \
    -D_GLIBCXX_USE_CXX11_ABI=$ABI \
    -DTORCH_EXTENSION_NAME=scattermaxv1_bridge \
    -DTORCH_API_INCLUDE_EXTENSION_H \
    -I"$TORCH/include" \
    -I"$TORCH/include/torch/csrc/api/include" \
    -I"$TORCH_NPU/include" \
    -I"$PY_ROOT/include/python3.11" \
    -I"$OPP/op_api/include" \
    -I"$CANN/include" \
    -o "$OUT" "$HERE/scattermaxv1_bridge.cpp" \
    -L"$TORCH/lib" -ltorch -ltorch_cpu -ltorch_python -lc10 \
    -L"$TORCH_NPU/lib" -ltorch_npu \
    -L"$OPP/op_api/lib" -lcust_opapi \
    -L"$CANN/lib64" -lascendcl \
    -Wl,-rpath,"$TORCH/lib" \
    -Wl,-rpath,"$TORCH_NPU/lib" \
    -Wl,-rpath,"$OPP/op_api/lib" \
    -Wl,-rpath,"$CANN/lib64"

echo "[OK] built $OUT"
echo "[INFO] torch=$TORCH (ABI=$ABI)"
echo "[INFO] torch_npu=$TORCH_NPU"
echo "[INFO] OPP=$OPP"
