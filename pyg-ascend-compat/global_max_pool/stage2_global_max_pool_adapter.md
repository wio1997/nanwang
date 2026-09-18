# Stage 2 — FP32 aligned `global_max_pool` adapter on Ascend (ScatterMaxV1)

Date: 2026-09-18
Container: `wio-pyg-cann851-pyg280` · CANN 8.5.1 · torch 2.9.0+cpu · torch_npu 2.9.0 · PyG 2.8.0.post1
Hardware: Ascend 910B3 (device 0 only), driver 26.0.rc1
Scope: standalone `global_max_pool_ascend(x, batch, size=None)` adapter calling the Stage 1
`ScatterMaxV1` custom op. **No PyG / torch_npu / aten patching (that is Stage 6).**

## 1. Bridge strategy (BRIDGE_STRATEGY = A)

Minimal PyTorch C++ extension (pybind11), built with plain `g++`:

* pybind11 comes from **PyTorch's own bundled headers** (`torch/include/pybind11`) — no `pybind11`
  pip package was installed
* `torch.utils.cpp_extension.load()` was avoided because it needs `ninja`, which is not installed
  in this image; plain `g++` keeps the build dependency-free and deterministic
* the op is launched on the **current torch_npu stream** (`c10_npu::getCurrentNPUStream()`), so it
  stays ordered with the surrounding torch ops (occupancy `scatter_`, `masked_fill_`, …)
* no new Python or system package was required

Files (`/root/zyg/global_max_pool/stage2/`):

| file | role |
|---|---|
| `extension/scattermaxv1_bridge.cpp` | `torch.Tensor` → `aclTensor` → `aclnnScatterMaxV1` |
| `extension/build_bridge.sh` | g++ build (includes/ABI/libs/rpath resolved from the live env) |
| `python/global_max_pool_ascend.py` | the adapter (validation, INT64→INT32, occupancy, empty→0) |
| `tests/run_stage2_tests.py` | correctness matrix A1..A16 + CPU golden |
| `tests/probe_occupancy.py` | occupancy/post-process op fallback probe |
| `tests/profile_adapter_path.py` | app used for the msprof capture of the full path |

C++ API: `scatter_max_v1_forward(src, index_int32, out, argmax_scratch) -> out` (in-place).
Python API: `global_max_pool_ascend(x, batch, size=None)` (+ `get_last_run_info()` diagnostics).
Build: `bash extension/build_bridge.sh` → `/root/zyg/build/stage2_ext/scattermaxv1_bridge.so`.

## 2. Adapter contract

| item | contract |
|---|---|
| x | NPU, `float32`, rank 2 `[N, F]`, contiguous or safely copied, `requires_grad=False` |
| batch | NPU (same device), `int64`, rank 1 `[N]` |
| size | `None` or python `int >= 0` |
| output | NPU `float32` `[S, F]`, `S = size` or `max(batch)+1`, `S = 0` for empty input without size |
| F | must satisfy `F % 8 == 0` (32 B aligned, FP32); otherwise rejected before the kernel |
| autograd | rejected (`x.requires_grad=True` → RuntimeError) — backward is Stage 4 |
| empty groups | `0` |
| non-empty groups | exact max, including a genuine `-inf` max |

## 3. INT64 → INT32

* validation before any launch: `batch >= 0`, `batch < S` (when `size` is explicit), `batch < 491520`
  (documented ScatterMaxV1 limit — enforced by the adapter because the kernel does not check it and
  an out-of-range index would write outside the output buffer), `batch.numel() == x.size(0)`
* conversion: `batch.to(torch.int32)` (NPU-native `Cast`, confirmed by profiler), then contiguous
* host sync: one `.cpu().tolist()` of `[min(batch), max(batch)]` → `HOST_SYNC_FOR_DIM_SIZE = YES`
  when `size is None`, `HOST_SYNC_FOR_RANGE_VALIDATION = YES` always. This is a scalar sync, not a
  CPU op fallback. It is the **only** sync on the normal path: diagnostic counters
  (`occupied_groups` / `empty_groups`) are computed only when `debug=True`, so they do not add
  syncs to regular calls.

## 4. ScatterMaxV1 invocation

* `out = torch.full((S,F), -inf, fp32, npu)` — caller-provided, `-inf` initialised (never 0)
* `argmax = torch.empty((S,F), int32, npu)` — the generated ACLNN schema has a mandatory second
  output; the forward kernel does not use it (`ARGMAX_SCRATCH_BYTES = S*F*4`)
* `workspaceSize == 0` for every case (tiling writes `workspace[0]=0`); the bridge still handles a
  non-zero workspace (allocate → launch → sync → free)
* explicit size is passed straight through as the caller-allocated output's dim0

## 5. Empty-group semantics (occupancy mask)

`out == -inf → 0` (the DrivingSDK wrapper rule) is **not** used: it would corrupt a non-empty group
whose true maximum is `-inf`. Instead:

```python
occupied = torch.zeros(S, dtype=torch.int32, device=...)   # 1 device op
occupied.scatter_(0, batch, 1)                             # ScatterElementsV2, AI_VECTOR_CORE
out.masked_fill_((occupied == 0).view(S, 1), 0.0)          # MaskedFill, AI_VECTOR_CORE
```

Everything stays on the NPU; no `.cpu()` on the mask, no host tensor processing.
Occupancy op selection was probed first (`tests/probe_occupancy.py`): `scatter_`, `masked_fill_`,
`torch.where`, `index_fill_`, `bincount` all showed **no** `npu_cpu_fallback` warning; `scatter_` +
`masked_fill_` were chosen and are confirmed device-native by msprof (§7).

## 6. Correctness matrix (all comparisons against an independent CPU golden, tol = 0)

| id | case | result |
|---|---|---|
| A1 | baseline N=8 F=8, size=None, mixed +/- | PASS (max_diff 0, exact) |
| A2 | explicit size=5 while max(batch)+1=3, empty rows → 0 | PASS |
| A3 | repeated index (16 rows → 1 group) | PASS |
| A4 | negative-only non-empty groups | PASS |
| A5 | **non-empty group with true max `-inf` vs empty group** | PASS — group0 `-inf`, group1 `0` |
| A6 | empty input, size=None → `[0, 8]` | PASS (fast path, no kernel) |
| A7 | empty input, size=4 → `[4, 8]` all zeros | PASS (fast path) |
| A8 | F=16 | PASS |
| A9 | F=7 rejected before kernel | PASS |
| A10 | index >= size rejected before kernel | PASS |
| A11 | negative index rejected | PASS |
| A12 | `requires_grad=True` rejected | PASS |
| A13 | batch not int64 rejected | PASS |
| A14 | x not fp32 rejected | PASS |
| A15 | x rank != 2 rejected | PASS |
| A16 | `x.size(0) != batch.numel()` rejected | PASS |

Rejection tests additionally assert that the adapter's run-info state is unchanged, i.e. no kernel
launch happened. Log: `/root/zyg/logs/stage2_tests.log` (16/16 PASS).

## 7. Runtime execution of the full adapter path (msprof, device 0, N=4096 F=32 size=64)

```
OP Type            Core Type        Count  Avg us
ScatterMaxV1       AI_VECTOR_CORE       8   18.5   <- the reduction
ScatterElementsV2  AI_VECTOR_CORE       8   61.6   <- occupancy scatter_
MaskedFill         AI_VECTOR_CORE       8   13.2   <- empty-group -> 0
Fill               AI_VECTOR_CORE       8    1.9   <- -inf init
Cast               AI_VECTOR_CORE      40    1.4   <- int64 -> int32 etc.
ReduceMin/Max/Sum  MIX_AIV              8/8/16     <- batch min/max (device side)
ZerosLike/Equal/NotEqual/LinearIndex/BroadcastTo/Pack -> AI_VECTOR_CORE
```

* `AI_CPU` tasks for the adapter path: **0**
* fallback warnings during adapter calls: **0**
* `HOST_CPU_OP_FALLBACK = NONE OBSERVED`
* `HOST_SYNC = 1x` scalar sync (`batch` min/max → python ints for size/range validation);
  `debug=True` adds 2 more (diagnostic counters only)
* raw profiler: `/root/zyg/profiler/stage2_adapter/PROF_000001_20260918023728673_*/`
  readable summary `/root/zyg/logs/stage2_profiler_summary.txt`

## 8. Temporary memory and latency (observation only, no optimisation in Stage 2)

| shape | output | argmax scratch | index int32 | occupancy | per-call latency (incl. sync) |
|---|---|---|---|---|---|
| N=8, F=8, S=4 | 128 B | 128 B | 32 B | 16 B | ~1.04 ms |
| N=4096, F=32, S=64 | 8 192 B | 8 192 B | 16 384 B | 256 B | ~1.32 ms |

First call in a fresh process was ~0.99 s (cold `aclnnScatterMaxV1GetWorkspaceSize`, incl.
`BinaryLoadFromData`) — recorded, not optimised.

## 9. Known limitations (real ones only)

* FP32 only (fp16/bf16 → Stage 5)
* aligned feature dim only, `F % 8 != 0` rejected (non-aligned → Stage 3)
* forward only; `requires_grad=True` rejected (backward/tie-gradient → Stage 4)
* ScatterMaxV1 index must be INT32; the adapter handles the INT64→INT32 conversion and the
  `batch < 491520` documented bound
* `argmax` scratch is a mandatory ACLNN output, so `S*F*4` extra bytes are allocated per call
  (accepted in Stage 2; forward-only packaging is optional later)
* the adapter works on `torch.Tensor` on NPU; it does not patch `aten::scatter_reduce` or PyG
  (Stage 6), so existing PyG code is unaffected

## 10. Verdict

**GLOBAL_MAX_POOL_ADAPTER = PASS**

P1–P11 of the Stage 2 acceptance list are all satisfied: the adapter is callable from Python/PyTorch
with NPU tensors, converts INT64 batch to the INT32 kernel index with explicit range checks,
honours explicit size, preserves negative-only and genuine `-inf` maxima, returns 0 for empty
groups, rejects non-aligned F / invalid indices / `requires_grad` before the kernel, and the whole
path executes on AIV/MIX cores without any host CPU op fallback.

## 11. Reproduction

```bash
cd /root/zyg/global_max_pool/stage2
export ASCEND_CUSTOM_OPP_PATH=/root/zyg/build/scattermax_runtime_opp/vendors/customize
export LD_LIBRARY_PATH=$ASCEND_CUSTOM_OPP_PATH/op_api/lib:$LD_LIBRARY_PATH
bash extension/build_bridge.sh                    # -> /root/zyg/build/stage2_ext/scattermaxv1_bridge.so
python3 tests/run_stage2_tests.py                 # A1..A16
python3 tests/probe_occupancy.py                  # occupancy op probe
msprof --application="python3 tests/profile_adapter_path.py" \
       --output=/root/zyg/profiler/stage2_adapter --ai-core=on --ascendcl=on --task-time=on
```

## 12. Next step

Stage 3 — boundary shapes and non-32B-aligned feature dims (`F % 8 != 0`), plus N/F/index/size
boundaries. PyG integration remains Stage 6.
