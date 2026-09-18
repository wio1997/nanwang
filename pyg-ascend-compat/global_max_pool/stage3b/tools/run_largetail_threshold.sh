#!/bin/bash
# Stage 3B largeTail threshold evidence.
#
# 1) shape-only tiling probe against the INSTRUMENTED probe-only OPP -> ACTUAL tiling key for
#    arbitrary (N, F), including shapes whose real tensors would need tens of GB.
# 2) real-execution RAW runner for the affordable Group A threshold shapes, one process per case,
#    stopping at the first failure.
#
# The delivered custom OPP and the frozen kernel are never modified.
set -u

STAGE3B=/root/zyg/global_max_pool/stage3b
LOGS=/root/zyg/logs/stage3b
PROBE_OPP=${STAGE3B_PROBE_OPP:-/root/zyg/build/stage3b_tiling_probe_opp/vendors/customize}
RUNNER=${STAGE3B_RUNNER:-/root/zyg/global_max_pool/stage1b/runner/scattermaxv1_runner}
SHAPE_PROBE=${STAGE3B_SHAPE_PROBE:-/root/zyg/build/stage3b_ext/tiling_shape_probe}
mkdir -p "$LOGS"

export ASCEND_CUSTOM_OPP_PATH="${ASCEND_CUSTOM_OPP_PATH:-}"
set +u
source "$PROBE_OPP/bin/set_env.bash"
set -u

echo "# Stage 3B largeTail threshold evidence ($(date -u +%FT%TZ))" \
    > "$LOGS/largetail_tiling_keys.txt"
echo "# tiling source: INSTRUMENTED probe-only OPP at $PROBE_OPP (host tiling print)" \
    >> "$LOGS/largetail_tiling_keys.txt"

shape_probe() {  # N F S label
    local n="$1" f="$2" s="$3" label="$4"
    echo "--- shape-only: $label N=$n F=$f S=$s" >> "$LOGS/largetail_tiling_keys.txt"
    "$SHAPE_PROBE" --n "$n" --f "$f" --s "$s" >> "$LOGS/largetail_tiling_keys.txt" 2>&1
    echo "exit=$?" >> "$LOGS/largetail_tiling_keys.txt"
}

echo "=== 1) shape-only tiling probe ==="
shape_probe 40 48888 8 "groupA threshold-1 (expect SMALL_TAIL)"
shape_probe 40 48889 8 "groupA threshold   (expect LARGE_TAIL)"
shape_probe 40 48890 8 "groupA threshold+1 (expect LARGE_TAIL)"
shape_probe 80 48888 8 "groupA N=80 threshold-1"
shape_probe 80 48889 8 "groupA N=80 threshold"
shape_probe 163800 44800 8 "groupB threshold-1 (expect SMALL_TAIL)"
shape_probe 163800 44801 8 "groupB threshold   (expect LARGE_TAIL)"
shape_probe 163800 44802 8 "groupB threshold+1 (expect LARGE_TAIL)"
shape_probe 163799 44801 8 "groupB N-1"
shape_probe 163801 44801 8 "groupB N+1"
shape_probe 4097 48889 8 "leftSrc + largeTail"
grep -E "SHAPE_PROBE|stage3b-tiling|^---|^exit=" "$LOGS/largetail_tiling_keys.txt"

echo
echo "=== 2) real-execution RAW threshold cases (one process per case) ==="
FAILED=0
for spec in "40 48888 8" "40 48889 8" "40 48890 8" "80 48888 8" "80 48889 8" "80 48890 8"; do
    set -- $spec
    N=$1; F=$2; S=$3
    TAG="raw_N${N}_F${F}_S${S}"
    LOG="$LOGS/${TAG}.log"
    echo "[RUN ] $TAG"
    "$RUNNER" --case "$TAG" --n "$N" --f "$F" --size "$S" \
        --index "$(python3 -c "print(','.join(str(i % $S) for i in range($N)))")" \
        --pattern baseline --tol 0 > "$LOG" 2>&1
    rc=$?
    verdict=$(grep -h '^RESULT ' "$LOG" | awk '{print $2}')
    tkey=$(grep -h 'TILING_KEY=' "$LOG" | head -1 | sed 's/.*TILING_KEY=//')
    printf "%-24s exit=%s verdict=%s tiling=%s\n" "$TAG" "$rc" "${verdict:-none}" "${tkey:-unknown}"
    printf "%-24s exit=%s verdict=%s tiling=%s\n" "$TAG" "$rc" "${verdict:-none}" \
        "${tkey:-unknown}" >> "$LOGS/largetail_execution.txt"
    if [ "$rc" -ne 0 ]; then
        FAILED=1
        echo "!! STOPPING threshold execution series at $TAG (exit=$rc)"
        break
    fi
done
echo "[DONE] failed=$FAILED"
