#!/bin/bash
# Runtime environment for the PowerGraph global_max_pool validation.
#
# Paths are resolved from this script's own location, so the package works from
# any checkout:
#
#   <repo>/pyg-ascend-compat/global_max_pool/powergraph_validation/scripts/bench_env.sh
#
# Everything that cannot be derived from the checkout can be overridden by an
# environment variable before sourcing:
#
#   GLOBAL_MAX_POOL_OPP      custom ScatterMaxV1 OPP       (default: this server's build dir)
#   SCATTERMAXV1_BRIDGE      compiled ACLNN bridge .so     (default: this server's build dir)
#   POWERGRAPH_UPSTREAM_DIR  PowerGraph-Graph checkout      (default: <pkg>/upstream/PowerGraph-Graph)
#   POWERGRAPH_DATA_ROOT     raw/processed dataset root     (default: <pkg>/data)
#   POWERGRAPH_RESULTS_ROOT  benchmark output root          (default: <pkg>/results)
#   POWERGRAPH_PROFILE_ROOT  msprof run output root         (default: <pkg>/profiler_runs)
#                            note: the committed parsed summaries live in
#                            <pkg>/evidence/profiler and are never overwritten by
#                            a default run
#   ASCEND_RT_VISIBLE_DEVICES  NPU to pin                   (default: 0)
#
# This script only exports environment variables; it writes nothing.

_BENCH_ENV_SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VALIDATION_ROOT="$(cd "$_BENCH_ENV_SCRIPT_DIR/.." && pwd)"
GLOBAL_MAX_POOL_DIR="$(cd "$VALIDATION_ROOT/.." && pwd)"
REPO_ROOT="$(cd "$GLOBAL_MAX_POOL_DIR/../.." && pwd)"

# custom ScatterMaxV1 delivery OPP
export GLOBAL_MAX_POOL_OPP="${GLOBAL_MAX_POOL_OPP:-/root/zyg/build/scattermax_runtime_opp/vendors/customize}"
# compiled PyTorch <-> ACLNN bridge
export SCATTERMAXV1_BRIDGE="${SCATTERMAXV1_BRIDGE:-/root/zyg/build/stage2_ext/scattermaxv1_bridge.so}"

# data / output roots (all overridable, nothing is written by this script)
export POWERGRAPH_UPSTREAM_DIR="${POWERGRAPH_UPSTREAM_DIR:-$VALIDATION_ROOT/upstream/PowerGraph-Graph}"
export POWERGRAPH_DATA_ROOT="${POWERGRAPH_DATA_ROOT:-$VALIDATION_ROOT/data}"
export POWERGRAPH_RESULTS_ROOT="${POWERGRAPH_RESULTS_ROOT:-$VALIDATION_ROOT/results}"
export POWERGRAPH_PROFILE_ROOT="${POWERGRAPH_PROFILE_ROOT:-$VALIDATION_ROOT/profiler_runs}"

# frozen compat package + its adapter modules, taken from THIS checkout
export PYG_ASCEND_COMPAT_DIR="${PYG_ASCEND_COMPAT_DIR:-$GLOBAL_MAX_POOL_DIR/stage6}"
export PYG_ASCEND_ADAPTER_PATH="${PYG_ASCEND_ADAPTER_PATH:-$GLOBAL_MAX_POOL_DIR/stage2/python/global_max_pool_ascend.py}"
export PYG_ASCEND_AUTOGRAD_PATH="${PYG_ASCEND_AUTOGRAD_PATH:-$GLOBAL_MAX_POOL_DIR/stage4/python/global_max_pool_ascend_autograd.py}"
export PYG_ASCEND_STAGE5_PATH="${PYG_ASCEND_STAGE5_PATH:-$GLOBAL_MAX_POOL_DIR/stage5/python/global_max_pool_ascend_dtype.py}"

# repository identity, exported for report/provenance capture
export POWERGRAPH_VALIDATION_ROOT="$VALIDATION_ROOT"
export NANWANG_REPO_ROOT="$REPO_ROOT"

if [ ! -d "$GLOBAL_MAX_POOL_OPP" ]; then
    echo "[bench] ERROR: custom ScatterMaxV1 OPP not found at $GLOBAL_MAX_POOL_OPP" >&2
    echo "[bench]        set GLOBAL_MAX_POOL_OPP to the vendors/customize directory" >&2
    return 1 2>/dev/null || exit 1
fi
if [ ! -f "$PYG_ASCEND_COMPAT_DIR/pyg_ascend_compat/__init__.py" ]; then
    echo "[bench] ERROR: frozen compat package not found under $PYG_ASCEND_COMPAT_DIR" >&2
    return 1 2>/dev/null || exit 1
fi
if [ ! -f "$SCATTERMAXV1_BRIDGE" ]; then
    echo "[bench] ERROR: ScatterMaxV1 bridge not found at $SCATTERMAXV1_BRIDGE" >&2
    echo "[bench]        build it with pyg-ascend-compat/global_max_pool/stage2/extension/build_bridge.sh" >&2
    return 1 2>/dev/null || exit 1
fi

# CANN runtime
if [ -f /usr/local/Ascend/ascend-toolkit/set_env.sh ]; then
    # shellcheck disable=SC1091
    source /usr/local/Ascend/ascend-toolkit/set_env.sh
fi

# isolated custom OPP (frozen ScatterMaxV1 delivery)
export ASCEND_CUSTOM_OPP_PATH="$GLOBAL_MAX_POOL_OPP"
set +u
# shellcheck disable=SC1091
source "$GLOBAL_MAX_POOL_OPP/bin/set_env.bash"
set -u

# frozen compat package + this package's own modules on the import path
export PYTHONPATH="$PYG_ASCEND_COMPAT_DIR:$VALIDATION_ROOT/scripts:${PYTHONPATH:-}"

# Benchmark host may be shared; pin to one NPU for stable single-op numbers.
export ASCEND_RT_VISIBLE_DEVICES="${ASCEND_RT_VISIBLE_DEVICES:-0}"
