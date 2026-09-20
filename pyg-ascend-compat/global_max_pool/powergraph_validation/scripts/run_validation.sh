#!/bin/bash
# End-to-end driver for the PowerGraph global_max_pool validation.
#
#   source scripts/bench_env.sh
#   bash scripts/run_validation.sh [options]
#
# Options:
#   --tag NAME        output tag (default: repo_validation)
#   --smoke-only      syntax check + one IEEE24 FP32 batch, nothing else
#   --no-profiler     skip the msprof gate
#   --no-backward     skip the forward+backward matrix
#
# Outputs go to POWERGRAPH_RESULTS_ROOT (default <package>/results) using the
# chosen tag, so the committed evidence CSVs (forward_phaseB_ieee24.csv,
# forward_phaseC.csv, forward_backward_phaseC.csv) are never overwritten.
set -euo pipefail

# keep the working tree clean: never leave __pycache__ behind
export PYTHONDONTWRITEBYTECODE=1

_SCRIPTS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
_VALIDATION_ROOT="$(cd "$_SCRIPTS_DIR/.." && pwd)"

TAG="repo_validation"
SMOKE_ONLY=0
DO_PROFILER=1
DO_BACKWARD=1
while [ $# -gt 0 ]; do
    case "$1" in
        --tag) TAG="$2"; shift 2 ;;
        --smoke-only) SMOKE_ONLY=1; shift ;;
        --no-profiler) DO_PROFILER=0; shift ;;
        --no-backward) DO_BACKWARD=0; shift ;;
        -h|--help) sed -n '2,16p' "$0"; exit 0 ;;
        *) echo "unknown option: $1" >&2; exit 2 ;;
    esac
done

DATA_ROOT="${POWERGRAPH_DATA_ROOT:-$_VALIDATION_ROOT/data}"
RESULTS_ROOT="${POWERGRAPH_RESULTS_ROOT:-$_VALIDATION_ROOT/results}"
UPSTREAM="${POWERGRAPH_UPSTREAM_DIR:-$_VALIDATION_ROOT/upstream/PowerGraph-Graph}"
mkdir -p "$RESULTS_ROOT"

step() { echo; echo "=== [$(date -u +%FT%TZ)] $* ==="; }

# ---------------------------------------------------------------- 0. syntax
step "syntax check"
python3 - "$_SCRIPTS_DIR" <<'PY'
import pathlib
import sys

root = pathlib.Path(sys.argv[1])
files = sorted(set(root.glob("*.py")) | set(root.glob("analysis/*.py")))
bad = []
for f in files:
    try:
        compile(f.read_bytes(), str(f), "exec")
    except SyntaxError as exc:
        bad.append(f"{f}: {exc}")
if bad:
    print("\n".join(bad), file=sys.stderr)
    raise SystemExit(1)
print(f"python syntax OK: {len(files)} files (bytecode not written)")
PY
for f in "$_SCRIPTS_DIR"/*.sh "$0"; do
    bash -n "$f"
done
echo "bash syntax OK"

# ---------------------------------------------------------- 1. preconditions
step "preconditions"
if [ ! -f "${PYG_ASCEND_COMPAT_DIR:-}/pyg_ascend_compat/__init__.py" ]; then
    echo "ERROR: pyg_ascend_compat not found (PYG_ASCEND_COMPAT_DIR=${PYG_ASCEND_COMPAT_DIR:-unset})." >&2
    echo "       run 'source scripts/bench_env.sh' first." >&2
    exit 1
fi
if [ ! -d "$UPSTREAM/code/dataset" ]; then
    echo "ERROR: PowerGraph-Graph checkout not found at $UPSTREAM" >&2
    echo "       git clone https://github.com/PowerGraph-Datasets/PowerGraph-Graph.git $UPSTREAM" >&2
    exit 1
fi
missing=0
for ds in ieee24 ieee39 ieee118 uk; do
    if [ ! -d "$DATA_ROOT/$ds/$ds/raw" ]; then
        echo "ERROR: raw data missing for $ds under $DATA_ROOT/$ds/$ds/raw" >&2
        missing=1
    fi
done
if [ "$missing" != 0 ]; then
    echo "       run scripts/fetch_powergraph_data.sh then scripts/extract_powergraph_data.py" >&2
    exit 1
fi
echo "compat package : $PYG_ASCEND_COMPAT_DIR"
echo "custom OPP     : ${ASCEND_CUSTOM_OPP_PATH:-unset}"
echo "upstream repo  : $UPSTREAM"
echo "data root      : $DATA_ROOT"
echo "results root   : $RESULTS_ROOT"

# ------------------------------------------------------------ 2. smoke check
step "smoke: IEEE24 FP32 compat forward"
python3 "$_SCRIPTS_DIR/analysis/smoke_compat.py"
python3 "$_SCRIPTS_DIR/bench_forward.py" --datasets ieee24 --batch-sizes 8 \
    --dtypes fp32 --paths compat_ascend --warmup 5 --iters 20 \
    --data-root "$DATA_ROOT" --results-dir "$RESULTS_ROOT" --tag "${TAG}_smoke"
if [ "$SMOKE_ONLY" = 1 ]; then
    echo; echo "smoke-only run complete"; exit 0
fi

# ---------------------------------------------------------------- 3. forward
step "forward matrix (4 datasets x batch 1/8/32/128 x FP32/FP16/BF16 x compat/original)"
python3 "$_SCRIPTS_DIR/bench_forward.py" \
    --datasets ieee24,ieee39,ieee118,uk --batch-sizes 1,8,32,128 \
    --dtypes fp32,fp16,bf16 --paths compat_ascend,original_pyg \
    --warmup 30 --iters 200 \
    --data-root "$DATA_ROOT" --results-dir "$RESULTS_ROOT" --tag "$TAG"

# --------------------------------------------------------------- 4. backward
if [ "$DO_BACKWARD" = 1 ]; then
    step "forward + first-order backward (compat path)"
    python3 "$_SCRIPTS_DIR/bench_backward.py" \
        --datasets ieee24,ieee39,ieee118,uk --batch-sizes 1,8,32,128 \
        --dtypes fp32,fp16,bf16 --paths compat_ascend \
        --warmup 30 --iters 200 \
        --data-root "$DATA_ROOT" --results-dir "$RESULTS_ROOT" --tag "$TAG"
fi

# --------------------------------------------------------------- 5. profiler
if [ "$DO_PROFILER" = 1 ]; then
    step "profiler gate (msprof --ai-core=on)"
    bash "$_SCRIPTS_DIR/run_profiles.sh" \
        "ieee24_b128_fp32 ieee24 128 fp32 compat_ascend" \
        "ieee24_b128_fp16 ieee24 128 fp16 compat_ascend" \
        "ieee24_b128_bf16 ieee24 128 bf16 compat_ascend" \
        "ieee118_b128_fp32 ieee118 128 fp32 compat_ascend" \
        "uk_b128_fp32 uk 128 fp32 compat_ascend" \
        "ieee24_b128_fp32_origpyg ieee24 128 fp32 original_pyg"
    echo
    echo "profiler summary: ${POWERGRAPH_PROFILE_ROOT:-$_VALIDATION_ROOT/profiler_runs}/profiler_summary.txt"
fi

# -------------------------------------------------------------- 6. summaries
step "derived summaries"
python3 "$_SCRIPTS_DIR/analysis/make_summaries.py" --tag "$TAG"

step "done"
echo "results: $RESULTS_ROOT"
