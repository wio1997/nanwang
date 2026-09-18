# Stage 3B — largeTail / boundary / large-shape validation

Validates the frozen FP32 forward delivery (`5816ef6`) at the ScatterMaxV1 tiling boundaries:
core-count boundary (N = 39/40/41), leftSrc remainders, large N (`MAX_BATCH_NUM` cap), the
large-tail threshold, size/index limits and PyG end-to-end. No new features, no kernel rewrite.

**Result: PARTIAL** — everything validates except that the `LARGE_TAIL` kernel path cannot execute
at all in the delivered package (only tiling-key entry `_0` is registered, so any shape whose
`tailBatchNum` becomes 0 fails at launch with `aclnn` status 361001 **before** any device work).
See `../stage3b_large_tail_boundary.md` for the full evidence.

## Layout

```
stage3b/
├── tests/
│   ├── stage3b_common.py                 # patterns, CPU goldens, memory estimator
│   ├── run_stage3b_boundary_tests.py     # N/leftSrc/largeN/S/index matrix (45/45 PASS)
│   ├── run_stage3b_pyg_e2e.py            # real PyG global_max_pool E2E cases
│   └── profile_stage3b_cases.py          # msprof app (C1 smallTail, C2 leftSrc, C3 threshold-1)
└── tools/
    ├── tiling_calc.py                    # host tiling formula calculator (no device needed)
    ├── instrument_tiling_probe.py        # adds a tiling print to a PROBE-ONLY op_host copy
    ├── tiling_shape_probe.cpp/.sh        # host-only aclnnGetWorkspaceSize probe (no kernel launch)
    └── run_largetail_threshold.sh        # tiling-key evidence + real-execution threshold runs
```

The delivered OPP/kernel and the frozen Stage 6 artifacts are never modified; the instrumented
op_host lives in a throwaway copy under `/root/zyg/build/stage3b_tiling_probe*`.

## Key numbers

* runtime UB used by the tiling: **196352 B** (the ini says 196608; the tiling runtime value is used)
* `idxBatchNum` clamped at **4095** (N ≥ 163800 → `remainUbSize` = 178944)
* `N < 40` → `idxNumPerCore == 0` → SMALL_TAIL for every F
* large-tail threshold: **F = 48825** for N ≈ 40–320, **F = 44737** for N ≥ 163799
* safe envelope for any N: **F ≤ 44736** (per-N higher, e.g. F ≤ 48824 for N ≤ 320)

## Run

```bash
source /root/zyg/global_max_pool/stage6/env.sh
python3 tools/tiling_calc.py --table
python3 tests/run_stage3b_boundary_tests.py
python3 tests/run_stage3b_pyg_e2e.py
bash    tools/run_largetail_threshold.sh
```
