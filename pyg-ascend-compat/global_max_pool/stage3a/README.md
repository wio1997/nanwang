# Stage 3A — non-aligned feature dims for `global_max_pool_ascend`

The adapter now accepts any `F >= 0`:

* `F % 8 == 0` → unchanged Stage 2 raw path
* `F % 8 != 0` → pad the feature dim to `ceil(F/8)*8` with `-inf`, run the same ScatterMaxV1 kernel,
  crop back to `F` (`alignment_mode="auto"`, the default)
* `alignment_mode="raw"` forces the unpadded kernel call (Stage 3A proved it is safe too);
  `alignment_mode="pad"` forces padding
* `F == 0` → device-side fast path returning `[S, 0]`

## Layout

```
stage3a/
├── tests/
│   ├── run_stage3a_tests.py        # B1..B13, alignment transitions, F=0, raw/pad parity (34/34 PASS)
│   ├── probe_padding_ops.py        # F.pad / new_full+slice / cat fallback probe
│   ├── probe_special_values.py     # +inf / -inf / NaN / +-0 semantics vs CPU
│   └── profile_adapter_padded.py   # msprof app for the padded path (N=4096, F=33, S=64)
└── tools/
    ├── raw_non_aligned_probe.sh    # controlled RAW probe, one process per F, stops on first failure
    └── perf_observe.py             # F=32 vs F=33 latency/memory observation
```

## Usage

```bash
cd /root/zyg/global_max_pool/stage3a
export ASCEND_CUSTOM_OPP_PATH=/root/zyg/build/scattermax_runtime_opp/vendors/customize
export LD_LIBRARY_PATH=$ASCEND_CUSTOM_OPP_PATH/op_api/lib:$LD_LIBRARY_PATH
bash tools/raw_non_aligned_probe.sh        # source-backed probe, only run after the audit
python3 tests/run_stage3a_tests.py         # 34/34 PASS expected
python3 tests/probe_special_values.py      # NaN / inf / signed-zero parity with CPU
python3 tools/perf_observe.py              # observation only
```

Contract details, the tiling branch matrix, the 491520 / `N*(M+1)` evidence and the profiler
results: `../stage3a_non_aligned_feature.md`.
