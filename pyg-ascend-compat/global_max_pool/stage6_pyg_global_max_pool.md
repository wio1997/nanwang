# Stage 6 — PyG `global_max_pool` integration on Ascend (ScatterMaxV1)

Date: 2026-09-18 · container `wio-pyg-cann851-pyg280` · CANN 8.5.1 · torch 2.9.0+cpu ·
torch_npu 2.9.0 · PyG 2.8.0.post1 · Ascend 910B3 (device 0) · driver 26.0.rc1
Scope: make the real `torch_geometric.nn.global_max_pool` use the validated ScatterMaxV1 adapter for
the FP32 / NPU / forward case, and prove correctness + device execution + removal of the host
fallback. No backward, no fp16/bf16, no general `aten::scatter_reduce` backend.

## 1. Current PyG call chain (read from the installed 2.8.0.post1)

`torch_geometric/nn/pool/glob.py::global_max_pool` (exported as `torch_geometric.nn.global_max_pool`,
also re-exported by `torch_geometric.nn.pool`):

```python
dim = -1 if isinstance(x, Tensor) and x.dim() == 1 else -2
if batch is None:
    return x.max(dim=dim, keepdim=x.dim() <= 2)[0]
return scatter(x, batch, dim=dim, dim_size=size, reduce='max')
```

`torch_geometric/utils/_scatter.py::scatter`:

* `dim = src.dim() + dim if dim < 0 else dim` → `dim = 0` for a 2-D `[N, F]` input
* `dim_size = int(index.max()) + 1 if index.numel() > 0 else 0` when `size is None`
* `size = src.size()[:dim] + (dim_size,) + src.size()[dim+1:]`
* for `reduce in ['min','max','amin','amax']`, because `WITH_TORCH_SCATTER` is False
  (`torch_scatter` is not installed) the branch taken is:

```python
index = broadcast(index, src, dim)
return src.new_zeros(size).scatter_reduce_(dim, index, src, reduce='amax', include_self=False)
```

⇒ `aten::scatter_reduce.two_out` → torch_npu has no NPU kernel → **host CPU fallback**:

```
Warning: CAUTION: The operator 'aten::scatter_reduce.two_out' is not currently supported on the
NPU backend and will fall back to run on the CPU. (function npu_cpu_fallback)
```

Note the semantics PyG relies on: output starts as zeros, `include_self=False` (zeros are not
candidates) → empty groups are 0, and a non-empty group keeps its true max even if it is negative or
`-inf`. That is exactly the adapter contract from Stage 2/3A.

## 2. Integration design (Option A — dispatch wrapper inside pyg-ascend-compat)

The repository is a validation-artifact repo (docs + screen scripts), so a small compat package was
added instead of a new framework:

```
pyg-ascend-compat/global_max_pool/stage6/
├── pyg_ascend_compat/__init__.py     # enable() / disable() / stats() / set_debug()
├── env.sh                            # shell-local env (isolated custom OPP + this package)
├── demo_global_max_pool_ascend.py
├── DELIVERY.md, README.md
└── tests/{run_stage6_tests.py, pyg_fallback_probe.py, profile_pyg_global_max_pool.py}
```

* `enable()` imports the adapter (`global_max_pool_ascend`) **by file path** (never re-implemented)
  and rebinds `global_max_pool` on `torch_geometric.nn`, `torch_geometric.nn.pool` and
  `torch_geometric.nn.pool.glob` — all three module attributes that hold the same function object.
* site-packages is **not** modified; `disable()` restores the originals; the wrapper keeps
  `__wrapped__`, `__doc__` and a `_pyg_ascend_compat` marker.
* dispatch conditions (everything else → original PyG):
  `x` is a Tensor on NPU, `float32`, `dim == 2`; `batch` on the same NPU, `int64`, `dim == 1`,
  `batch.numel() == x.size(0)`; `x.requires_grad == False`.
* NPU + fp32 + `requires_grad=True` → explicit `RuntimeError` (no silent detach, no CPU fallback).
* NPU + fp16/bf16 → passthrough to the original PyG path and counted as
  `non_fp32_passthrough` (documented, not reported as a success).
* `stats()` gives Python-level proof of the dispatch (`ascend_calls` / `original_calls`), and
  `set_debug(True)` prints one line per dispatched call (default off).

Documented usage (import the PyG symbol **after** `enable()`):

```python
import pyg_ascend_compat
pyg_ascend_compat.enable()
from torch_geometric.nn import global_max_pool
out = global_max_pool(x, batch, size)
```

## 3. Final call path

```text
torch_geometric.nn.global_max_pool
  -> pyg_ascend_compat wrapper            (fp32 + NPU + forward)
  -> global_max_pool_ascend               (Stage 2/3A adapter, imported from the same file)
       ├─ torch.nn.functional.pad         (only when F % 8 != 0)
       ├─ int64 -> int32 index cast + range validation
       ├─ scatter_max_v1_forward          (minimal PyTorch C++ bridge, torch_npu stream)
       ├─ occupancy scatter_ + masked_fill (empty group -> 0)
       └─ crop (+ contiguous) back to F
  -> aclnnScatterMaxV1 -> AI_VECTOR_CORE
```

## 4. Correctness matrix (all through the real PyG entry point)

`stage6/tests/run_stage6_tests.py` — 20/20 PASS, log `/root/zyg/logs/stage6_tests.log`

| id | case | result |
|---|---|---|
| – | BEFORE: plain PyG on NPU | `npu_cpu_fallback` on `aten::scatter_reduce.two_out` **observed (YES)** |
| – | AFTER: same API with compat | no fallback text; wrapper marker present |
| PYG1 | baseline `[8,8]`, size=None, mixed ± | PASS (tol 0, exact) |
| PYG2 | non-aligned `F=7` (padding path) | PASS |
| PYG3 | explicit `size=5` with `max(batch)+1=4` | PASS, empty row = 0 |
| PYG4 | repeated index (16 rows → 1 group) | PASS |
| PYG5 | negative-only groups | PASS |
| PYG6 | true `-inf` max in a non-empty group vs empty group | PASS (`g0 = -inf`, `g1 = 0`) |
| PYG7 | `F=0` | PASS (`[3, 0]`); PyG itself accepts this shape |
| PYG8 | empty input, `size=None` and `size=4` | PASS (`[0,8]` / `[4,8]`; PyG accepts both) |
| – | PyG CPU vs PyG NPU(compat) on 4 shapes | PASS (identical) |
| – | counters match the dispatch | PASS (`ascend_calls=13` = NPU calls, `original_calls=4` = the intentional CPU parity calls) |

Independent CPU golden (not PyG) is used for every PYG case, and PyG-on-CPU is used as an extra
reference.

## 5. End-to-end profiler (real PyG API, N=4096, F=33 → pad 40, S=64)

```
msprof --application="python3 tests/profile_pyg_global_max_pool.py" --output=/root/zyg/profiler/stage6_pyg \
       --ai-core=on --ascendcl=on --runtime-api=on --task-time=on
```

| OP Type | Core Type | Count | Avg us |
|---|---|---|---|
| `ScatterMaxV1` | **AI_VECTOR_CORE** | 8 | 25.9 |
| `ScatterElementsV2` (occupancy) | AI_VECTOR_CORE | 8 | 61.6 |
| `MaskedFill` (empty → 0) | AI_VECTOR_CORE | 8 | 14.7 |
| `PadV3` (Stage 3A padding) | MIX_AIV | 8 | 16.4 |
| `Slice` (crop) | AI_VECTOR_CORE | 8 | 3.8 |
| `Cast` (int64→int32), `Fill`, `MemSet`, `ZerosLike`, `Equal`, `BroadcastTo`, `LinearIndex`, `Pack` | AI_VECTOR_CORE | 8–40 | ≤ 10.2 |
| `ReduceMin`/`ReduceMax` (batch min/max for size + range validation) | MIX_AIV | 8 | ~41.9 |

* `PYG_ATEN_SCATTER_REDUCE_CALLED = NO` (0 occurrences of `scatter_reduce` in `api_statistic`)
* `AI_CPU_TASKS = 0` · `HOST_CPU_OP_FALLBACK = NONE` · fallback warnings during the run: 0
* raw: `/root/zyg/profiler/stage6_pyg/PROF_000001_20260918031009628_*/`, summary:
  `/root/zyg/logs/stage6_profiler_summary.txt`

## 6. Demo

```
$ source /root/zyg/global_max_pool/stage6/env.sh
$ python3 /root/zyg/global_max_pool/stage6/demo_global_max_pool_ascend.py
PyG version          : 2.8.0.post1
torch version        : 2.9.0+cpu
torch_npu version    : 2.9.0
device               : npu:0
input shape          : (8, 7)
batch dtype          : torch.int64
size (explicit)      : 5  (max(batch)+1 = 4 -> group 4 is empty)
output shape         : (5, 7)
output device/dtype  : npu:0 / torch.float32
CPU golden match     : True
empty group row all 0: True
compat path used     : ascend_calls=1 original_calls=0 (enabled=True)
Host CPU fallback    : NONE
RESULT: PASS
```

## 7. Reproduction from a clean checkout

```bash
# 1) isolated custom OPP (Stage 1) must exist and the bridge must be built once
cd <repo>/global_max_pool/stage2
bash extension/build_bridge.sh

# 2) shell-local environment: isolated OPP + stage6 package on PYTHONPATH
source <repo>/global_max_pool/stage6/env.sh

# 3) demo + full matrix
python3 <repo>/global_max_pool/stage6/demo_global_max_pool_ascend.py
python3 <repo>/global_max_pool/stage6/tests/run_stage6_tests.py
```

No manual `export` of internal paths, no copying `.so` files into site-packages, no edits to PyG
sources; `site-packages/torch_geometric/**` is untouched and `/usr/local/Ascend/cann-8.5.1` is
untouched (custom OPP stays in `/root/zyg/build/scattermax_runtime_opp`).

## 8. Known limitations (today's scope)

* forward **and FP32 first-order backward** are supported (`requires_grad=True` now routes to the
  Stage 4 autograd path — see `stage4_fp32_backward.md`); second-order autograd is out of scope.
  The Stage 2/3A adapter itself remains forward-only (it still rejects `requires_grad=True`); the
  differentiable entry is `global_max_pool/stage4/python/global_max_pool_ascend_autograd.py`.
* FP32 only; fp16/bf16 pass through to the original PyG path (no claim of support)
* only `global_max_pool` is routed; `aten::scatter_reduce` and other PyG scatter reductions are
  intentionally untouched
* documented index/shape limits of the custom op are enforced by the adapter:
  `index < size`, `index < 491520` (documented support limit), `N*(F+1) < 4,026,531,840`
* `F ≳ 4.48–4.89 万` (fp32, 910B3) switches the kernel to the large-tail path; that path was
  *not* covered by the Stage 3A/6 evidence at the time of this report. **Update (Stage 3E):** the
  Stage 3C/3D large-tail repairs are now promoted into the formal delivery OPP and the large-tail
  matrix + PyG E2E run green on it — see `stage3e_large_tail_delivery_promotion.md`. The extreme
  combined case `N ≥ 163800` **and** large-tail `F` remains shape-only (HBM budget).
* first call in a fresh process pays the ACLNN executor cold start (~1 s); not optimised

## 9. Delivery verdict

**TODAY DELIVERY = PASS**

S6-P1…S6-P9 are satisfied: the real PyG API is used, fp32 NPU correctness passes (including
non-aligned F, explicit size, empty/negative/`-inf` semantics), the profiler shows `ScatterMaxV1` on
AI_VECTOR_CORE with zero AI_CPU tasks and no `aten::scatter_reduce` call, and the mechanism lives in
the repository with a reproducible enable path that leaves site-packages untouched.
