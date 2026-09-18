#!/bin/bash
# Stage 6 environment for using the compat layer. Source it (do not execute):
#   source /root/zyg/global_max_pool/stage6/env.sh
#
# It only touches this shell: isolated custom OPP + the repo compat package.
# Nothing is written to /root/.bashrc, /etc/profile or site-packages.

STAGE6_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# isolated custom OPP (Stage 1 build, never the system CANN opp)
export STAGE6_OPP="${STAGE6_OPP:-/root/zyg/build/scattermax_runtime_opp/vendors/customize}"
if [ ! -d "$STAGE6_OPP" ]; then
    echo "[stage6] ERROR: custom OPP not found at $STAGE6_OPP" >&2
    return 1 2>/dev/null || exit 1
fi

# the vendor set_env.bash references ${ASCEND_CUSTOM_OPP_PATH} unconditionally
export ASCEND_CUSTOM_OPP_PATH="${ASCEND_CUSTOM_OPP_PATH:-}"
set +u
source "$STAGE6_OPP/bin/set_env.bash"
set -u

# repo compat package + adapter + bridge
export PYTHONPATH="$STAGE6_DIR:${PYTHONPATH:-}"
export PYG_ASCEND_ADAPTER_PATH="${PYG_ASCEND_ADAPTER_PATH:-/root/zyg/global_max_pool/stage2/python/global_max_pool_ascend.py}"
export SCATTERMAXV1_BRIDGE="${SCATTERMAXV1_BRIDGE:-/root/zyg/build/stage2_ext/scattermaxv1_bridge.so}"

echo "[stage6] ASCEND_CUSTOM_OPP_PATH=$ASCEND_CUSTOM_OPP_PATH"
echo "[stage6] PYTHONPATH includes $STAGE6_DIR"
echo "[stage6] adapter=$PYG_ASCEND_ADAPTER_PATH"
echo "[stage6] bridge=$SCATTERMAXV1_BRIDGE"
