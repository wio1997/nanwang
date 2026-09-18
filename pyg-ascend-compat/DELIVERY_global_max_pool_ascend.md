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
* **FP32 first-order backward** (`x.requires_grad=True`): exact PyG/PyTorch tie-gradient semantics
  (equal split per `(group, feature)`, the `include_self=False` zero-max denominator quirk, empty
  groups, `-inf`, NaN contagion) — Stage 4; see
  `global_max_pool/stage4_fp32_backward.md`
* **FP16 and BF16 forward + first-order backward** (`requires_grad` either way) — Stage 5; the
  delivered `ScatterMaxV1` is fp32-only (measured `aclnn` status 161002 for fp16/bf16), so the path
  is an audited device cast chain (`dtype → fp32 → ScatterMaxV1 → dtype`) with a dtype-exact
  tie-gradient backward; see `global_max_pool/stage5_fp16_bf16.md`

## Not yet supported

* second-order autograd / `create_graph=True` (gradgrad) — Stage 4 is first-order only
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

### Stage 4 — FP32 backward (2026-09-18)

| item | result |
|---|---|
| CPU gradient oracle (real PyG/PyTorch on CPU) | frozen contract, validated as an executable model on **300/300** random cases |
| tie semantics | equal split per `(group, feature)`; zero-valued maxima get `+1` in the denominator (PyTorch `include_self=False` backward quirk, proven by a controlled experiment) |
| empty groups / `-inf` / NaN / ±0 | matched exactly (NaN ⇒ every row of that cell gets `nan`) |
| backward matrix vs CPU oracle | **35/35 PASS** (31/32 gradient cases bit-exact, max **1 ULP** on tie-split cells) |
| real PyG API E2E backward | **11/11 PASS**, `ascend_calls=10 original_calls=0 autograd_calls=10` |
| profiler (forward+loss+backward) | `ScatterMaxV1` AI_VECTOR_CORE; `GatherV3/Equal/Cast/InplaceIndexAdd/RealDiv/Mul/MaskedFill` all device; **AI_CPU = 0**; `aten::scatter_reduce` **NOT CALLED** |
| host CPU fallback | **NONE** |
| forward regression | Stage 3A 34/34 · 3B 45/45 · 3D 9/9 · 3E 13/13 · Stage 6 demo PASS · 6 20/20 · Stage 2 16/16 |
| `batch=None` | PyG `x.max` path, native on NPU, unchanged (outside the Stage 4 custom-backward scope) |

Known numerical note: the Ascend vector unit has no correctly-rounded fp32 divide (and no fp64), so
tie-split cells can differ from the IEEE-rounded CPU result by **≤1 ULP**; winner masks, counts,
zeros and NaN structure are exact.

### Stage 5 — FP16 / BF16 (2026-09-18)

| item | fp16 | bf16 |
|---|---|---|
| capability | supported (device cast chain) | supported |
| CPU oracle | 26/26 cases, out/grad dtype = input dtype | same |
| count arithmetic (measured) | exact integer rounded to dtype (21/21) | (22/22) |
| semantics matrix (31 cases each) | **31/31 PASS, bit-exact (max ULP 0)** | **31/31 PASS, bit-exact** |
| real PyG API E2E | PASS (`ascend_calls=24 original_calls=0`, dtype16_autograd 22 / forward 2) | PASS |
| profiler (4 cases each, forward+loss+backward) | ScatterMaxV1 AI_VECTOR_CORE, AI_CPU 0, `scatter_reduce` NOT CALLED, fallback NONE | same |
| FP32 regression | 3A 34/34 · 3B 45/45 · 3D 9/9 · 3E 13/13 · Stage 6 20/20 · Stage 2 16/16 · Stage 4 35/35+11/11 (ULP unchanged) | unchanged |

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
