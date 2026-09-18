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
* **large-tail feature dims** (`F` above the per-N small-tail threshold: `F ≥ 48825` for
  `N ≤ 320`, `F ≥ 44737` for `N ≥ 163800`) — the Stage 3C kernel-entry repair and the two Stage 3D
  kernel repairs are **promoted into the delivery OPP** (Stage 3E); validated at
  `N = 40…320`, `F = 48825…48960` including non-aligned `F`, `leftSrc`, negative-only data,
  repeated indices, true `-inf`, empty groups and explicit `size`
* single NPU (device 0 in the tests)

## Not yet supported

* backward / `requires_grad=True` (explicitly rejected — Stage 4)
* FP16 / BF16 (falls through to the original PyG path — Stage 5)
* the **extreme combined case** `N ≥ 163800` **and** large-tail `F`: shape-only tiling evidence
  exists (`F = 44736` → SMALL_TAIL, `F = 44737` → LARGE_TAIL at `N = 163800`), but the runtime is
  not exercised — the input alone would need ≈29 GB of HBM
* other PyG scatter reductions (`aten::scatter_reduce` is intentionally not patched)

## Guards (unchanged, still enforced)

```
index < 491520
N*(F+1) < 4,026,531,840
N >= 163800 combined with largeTail: shape-only evidence, runtime not exercised (HBM budget)
```

This delivery makes no "unlimited arbitrary shapes" claim.

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

### Stage 3E — large-tail promotion (2026-09-18)

| item | result |
|---|---|
| delivery OPP kernel package | `kernelList = [..._0, ..._1]`, `supportInfo.tilingKey = ["0","1"]` |
| runtime package provenance | `LD_PRELOAD` open() trace shows only the formal OPP's `ScatterMaxV1_*.o` being opened |
| large-tail runtime matrix (one shape per process) | LT0–LT6 + LTA1/LTA2 = **9/9 PASS**, `max_abs_diff = 0` |
| large-tail + `leftSrc` / + padding | **PASS** (LT3/LT5, LT2/LTA1/LTA2, P3, E2E_LT3) |
| tail attack (last row / final chunk / repeated index / negative-only / `-inf` / empty / explicit size) | **13/13 PASS**, tol 0 |
| PyG end-to-end large-tail | `E2E_LT1/2/3` **PASS**, `ascend_calls=3 original_calls=0`, fallback **NONE** |
| profiler | `ScatterMaxV1` = **AI_VECTOR_CORE** (8 tasks per case, entry `_1`), `AI_CPU = 0` |
| `507035` / `507011` / MTE OOB / AIV exception | **NONE** |
| index GM 32 B over-read | **REMOVED_BY_FIX** (byte-exact `DataCopyPad` index loads) |
| regression | Stage 3A **34/34** · Stage 3B **45/45** · Stage 3D **9/9** · Stage 6 demo **PASS** · Stage 6 **20/20** |

Repairs promoted (source diff and evidence):

```
op_kernel/scatter_max_v1.cpp : `else` -> `else if (TILING_KEY_IS(1))`          (Stage 3C entry)
op_kernel/CMakeLists.txt     : add_ops_compile_options(ScatterMaxV1 OPTIONS --tiling_key=0,1)
op_kernel/scatter_max_v1.h   : _idxLocal.GetValue(idxOffset + k) -> GetValue(k) (Stage 3D)
                               _resGM[idxVal*_tailElemNum] + n*_srcBatchNum     (Stage 3D)
                               index GM loads -> byte-exact DataCopyPad         (Stage 3E Task E)
```

Formal delivery build source `/root/zyg/build/scattermax_probe`; runtime OPP
`/root/zyg/build/scattermax_runtime_opp/vendors/customize` (both paths are the ones used by
`stage6/env.sh`). Full evidence: `global_max_pool/stage3e_large_tail_delivery_promotion.md`.
