# PyG Ascend global_max_pool — FP32 forward delivery

> **PyG Ascend `global_max_pool` FP32 forward 版本已完成** (this delivery covers the FP32 forward
> `global_max_pool` path only — it is not a claim that the whole operator family is finished).

## Problem

On Ascend 910B3 with CANN 8.5.1 / torch_npu 2.9.0, PyG's `global_max_pool` runs:

```
torch_geometric.nn.global_max_pool      (torch_geometric/nn/pool/glob.py)
  -> torch_geometric.utils.scatter      (torch_geometric/utils/_scatter.py, reduce="max")
  -> src.new_zeros(size).scatter_reduce_(dim=-2, index, src, reduce="amax", include_self=False)
  -> aten::scatter_reduce.two_out       -> Host CPU fallback
     (torch_npu: "not currently supported on the NPU backend and will fall back to run on the CPU")
```

## Solution

```
torch_geometric.nn.global_max_pool
  -> pyg_ascend_compat dispatch          (fp32 + NPU + forward only)
  -> global_max_pool_ascend              (Stage 2/3A adapter: INT64->INT32, -inf init,
                                          occupancy mask, padding for non-aligned F)
  -> aclnnScatterMaxV1                   (DrivingSDK ScatterMaxV1, custom OPP)
  -> Ascend AI_VECTOR_CORE
```

## Supported today

* Ascend 910B3, CANN 8.5.1, torch 2.9.0+cpu, torch_npu 2.9.0, PyG 2.8.0.post1
* FP32, forward only, arbitrary `F >= 0` (non-aligned dims are padded and cropped)
* `batch` int64, explicit `size` supported, empty group → 0, negative-only groups correct,
  genuine `-inf` max preserved
* single NPU (device 0 in the tests)

## Not yet supported

* backward / `requires_grad=True` (explicitly rejected — Stage 4)
* FP16 / BF16 (falls through to the original PyG path — Stage 5)
* Stage 3B largeTail / extreme boundary validation (`F ≳ 4.48–4.89 万` on 910B3, huge N/S)
* other PyG scatter reductions (`aten::scatter_reduce` is intentionally not patched)

## Run

```bash
cd /root/zyg/global_max_pool/stage2 && bash extension/build_bridge.sh   # once
source /root/zyg/global_max_pool/stage6/env.sh
python3 /root/zyg/global_max_pool/stage6/demo_global_max_pool_ascend.py
```

Application usage:

```python
import pyg_ascend_compat
pyg_ascend_compat.enable()                    # 1) enable first
from torch_geometric.nn import global_max_pool  # 2) then import the PyG symbol
out = global_max_pool(x, batch, size)           # 3) now it dispatches to ScatterMaxV1
```

**Import order matters:** `enable()` must run *before*
`from torch_geometric.nn import global_max_pool`. A Python name bound earlier keeps pointing at the
original function, so a later `enable()` cannot affect it. Import the symbol after `enable()` (or
call it as `torch_geometric.nn.global_max_pool(...)`), and re-import if you enabled afterwards.

## Evidence

| item | result |
|---|---|
| correctness (real PyG API, 20 checks incl. PYG1–PYG8) | **PASS** (tol 0, independent CPU golden + PyG CPU parity) |
| before compat | `npu_cpu_fallback` on `aten::scatter_reduce.two_out` (**YES**) |
| after compat | fallback text **NONE**; `aten::scatter_reduce` **NOT CALLED** (0 occurrences in the profiler API trace) |
| AI Core | `ScatterMaxV1` = **AI_VECTOR_CORE**, 8/8 tasks (msprof) |
| occupancy / padding / crop | `ScatterElementsV2`, `PadV3`, `MaskedFill`, `Slice` — all device |
| AI_CPU tasks in the profiler | **0** |
| Host CPU fallback on the compat path | **NONE** |
| system PyG / site-packages modified | **NO** (runtime wrapper + `disable()` restores) |
