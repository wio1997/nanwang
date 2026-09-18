# Stage 3B — largeTail / boundary / large-shape validation

Date: 2026-09-18 · container `wio-pyg-cann851-pyg280` · CANN 8.5.1 · torch 2.9.0+cpu ·
torch_npu 2.9.0 · Ascend 910B3 · frozen baseline `5816ef6`
Scope: validate the frozen FP32 forward delivery at tiling boundaries, large F/N, leftSrc remainders
and near the documented limits. No new features, no kernel rewrite, no frozen-semantics change.

## 1. Exact tiling thresholds (source + runtime)

Formulas re-read from `kernels/scatter_max/op_host/scatter_max_v1.cpp` (`GetTilingData`):

```
ubSize(platform, runtime)  = 196352 B      # 192 KiB - 256 B, read at runtime via GetCoreMemSize
ubSizeAfterPreserved       = 196352 - 1024 = 195328
idxNumPerCore              = N / 40        (integer division)
idxBatchNum                = min(idxNumPerCore, 4095)
remainUbSize               = 195328 - CeilAlign(idxBatchNum, 8) * 4
tailSizeAlign              = CeilAlign(F, 8) * 4
tailBatchNum               = remainUbSize / tailSizeAlign
LARGE_TAIL (key 1)  <=>  idxNumPerCore != 0  AND  tailSizeAlign > remainUbSize
```

Note: the platform ini (`Ascend910B3.ini`) lists `ub_size=196608`, but the value the **tiling**
receives at runtime is `196352`; all thresholds below are derived from and verified against the
runtime value (see the instrumented tiling evidence in §2).

| N | idxNumPerCore | idxBatchNum | remainUbSize | tailBatchNum (F=8) | path | leftIdxNum | leftSrc used |
|---|---|---|---|---|---|---|---|
| 39 | 0 | 0 | 195328 | 6104 | SMALL_TAIL | 39 | yes (all rows) |
| 40 | 1 | 1 | 195296 | 6103 | SMALL_TAIL | 0 | no |
| 41 | 1 | 1 | 195296 | 6103 | SMALL_TAIL | 1 | yes |
| 79 | 1 | 1 | 195296 | 6103 | SMALL_TAIL | 39 | yes |
| 81 | 2 | 2 | 195296 | 6103 | SMALL_TAIL | 1 | yes |
| 127 | 3 | 3 | 195296 | 6103 | SMALL_TAIL | 7 | yes |
| 320 | 8 | 8 | 195296 | 6103 | SMALL_TAIL | 0 | no |
| 321 | 8 | 8 | 195296 | 6103 | SMALL_TAIL | 1 | yes |
| 4097 | 102 | 102 | 194912 | 6091 (F=8) / 1218 (F=33) | SMALL_TAIL | 17 | yes |
| 163799 | 4094 | 4094 | 178944 | 5592 | SMALL_TAIL | 39 | yes |
| 163800 | 4095 | 4095 | 178944 | 5592 | SMALL_TAIL | 0 | no |
| 163801 | 4095 | 4095 | 178944 | 5592 | SMALL_TAIL | 1 | yes |

Large-tail threshold `T(N)` (smallest F whose tiling key is 1), measured at the exact boundary:

| N | F = T-1 | F = T | F = T+1 |
|---|---|---|---|
| 40 | 48824 → SMALL_TAIL (tailBatchNum=1) | **48825 → LARGE_TAIL** | 48826 → LARGE_TAIL |
| 80 | 48824 → SMALL_TAIL | 48825 → LARGE_TAIL | – |
| 320 | 48824 → SMALL_TAIL | 48825 → LARGE_TAIL | – |
| 163800 | 44736 → SMALL_TAIL (tailBatchNum=1) | **44737 → LARGE_TAIL** | 44738 → LARGE_TAIL |
| 163801 | 44736 → SMALL_TAIL | 44737 → LARGE_TAIL | – |

* `N < 40` ⇒ `idxNumPerCore == 0` ⇒ **SMALL_TAIL for every F** (no large-tail threshold at all)
* for `N >= 40` the threshold moves from **48825** (N ≈ 40–320) down to **44737** (N ≥ 163799)
  because the index batch buffer consumes up to 16 384 B of UB
* therefore the **N-independent safe envelope** for any shape is `F <= 44736`; per-N the limit is
  higher (e.g. `F <= 48824` for N ≤ 320)

## 2. How the tiling key was measured

Stage 3A only had a computed estimate (and it was wrong: it used the ini's 196608 instead of the
runtime 196352, predicting 48889). Stage 3B measures the **actual** decision twice:

1. **Probe-only instrumented op_host** (`tools/instrument_tiling_probe.py` patches a throwaway copy
   of the Stage 1A probe project; the delivered OPP and kernel are untouched) prints
   `N/F/idxNumPerCore/idxBatchNum/tailBatchNum/srcBatchNum/leftIdxNum/coreNumPerTail/leftSrcBatchNum/`
   `ubSize/remainUbSize/tailSizeAlign/TILING_KEY=...` to stderr on every tiling call.
2. **Shape-only tiling probe** (`tools/tiling_shape_probe.cpp`) calls only
   `aclnnScatterMaxV1GetWorkspaceSize` (host tiling) with tiny device buffers, so shapes that would
   need tens of GB (N=163800 × F≈44737 ≈ 29 GB) can be measured **without allocating** and without
   ever launching a kernel.

Raw output: `/root/zyg/logs/stage3b/exact_tiling_runtime.txt`.

## 3. Core-count boundary and leftSrc (adapter level, tol 0)

All cases below ran the **delivered** adapter (padded path where `F % 8 != 0`) and were compared
against an independent CPU golden (`scatter_reduce(amax, include_self=False)` + occupancy fix,
cross-checked against an explicit python-loop golden on the small cases).

| case group | shapes | result |
|---|---|---|
| N core boundary | N = 39/40/41 × F = 8/33, index spread and all-same | PASS (12/12) |
| leftSrc | N = 41/79/81/127/4097 × F = 8/17/33 | PASS (15/15) |
| leftover-row focus | N=41 leftover = duplicate of group 0 / new group / true max | PASS |
| tail winner | last row + last feature holds the group max (N=4097 F=33) | PASS |

Tiling evidence shows `processLeftSrc` is exercised exactly when `N % 40 != 0`
(`leftIdxNum != 0`), and that for `N = 39` the entire batch loop is skipped
(`idxNumPerCore = 0`), i.e. all rows go through the leftover path.

## 4. Large N / MAX_BATCH_NUM

| N | shape | result |
|---|---|---|
| 163799 / 163800 / 163801 | F=8, S=8 | PASS (correctness, ~5.6 MiB/case) |
| 163800 | F=33, S=8 (padded → F_kernel=40) | PASS (~46 MiB) |
| 163801 | F=8, S=16, tail-winner pattern | PASS |

Runtime tiling confirms `idxBatchNum` is clamped to **4095** for N ≥ 163800 (`remainUbSize` drops to
178944), and the kernel-side `_idxLoop = ceilDiv(idxNumPerCore, idxBatchNum)` therefore processes the
batch loop in a single iteration per core.

## 5. largeTail: BLOCKER (precise, reproducible)

Real execution at the boundary with the **delivered** OPP (`scattermax_runtime_opp`):

| N=40, S=8 | F=40000 | F=48823 | F=48824 | **F=48825** | F=48888 |
|---|---|---|---|---|---|
| result | PASS | PASS | PASS | **FAIL: `aclnnScatterMaxV1` status 361001** | FAIL 361001 |

`361001 = ACLNN_ERR_RUNTIME_ERROR` (`include/aclnn/opdev/op_errno.h`). The verbose CANN log shows the
exact layer:

```
[ERROR] RUNTIME[...][api_impl.cc:969] BinaryGetFunctionByEntry:BinaryGetFunctionByEntry failed, funcEntry=1
[ERROR] RUNTIME[...][api_c_kernel.cc:408] rtsFuncGetByEntry:ErrCode=107000, desc=[invalid value]
[ERROR] OP[...][indv_bininfo.cpp:103][NNOP][GetFuncHandleByEntry] errno[361001] OpName:[ScatterMaxV1_0]
        Assert ((aclrtBinaryGetFunctionByEntry(binHandle, funcEntry, funcHandle)) == 0) failed
[ERROR] OP[...][indv_executor.cpp:999] Check GetFuncHandleByEntry(..., tilingKey, &funcHandle) failed
```

**Root cause**: the op-api resolves the kernel entry by **tiling key**
(`GetFuncHandleByEntry(..., tilingKey)`), but our compiled op registers only one function entry:

```json
"kernelList": [{"kernelName": "ScatterMaxV1_7d55161965c898907fdb3028d01c7c76_0"}]
```

The `_0` suffix is the tiling key. There is **no `_1` entry**, so any shape that selects
`LARGE_TAIL` fails at launch — on the host, before any device task is created.

Why the entry is missing: the AscendC custom-op build only emits one kernel entry unless the op
declares additional tiling keys (`ascendc_bin_param_build.py` passes `--tiling_key="<keys>"` to the
kernel compiler; `ascendc_gen_options.py` reads it from the op's compile options). Our minimal
Stage 1A port declared none (`custom_compile_options.ini` was empty), and **DrivingSDK's own tree
does not declare any either** (`grep -rn "tiling_key" DrivingSDK` → no hits, only
`add_ops_compile_options(ALL OPTIONS -g -O0)`), so this is very likely an upstream limitation of the
largeTail path on this toolchain rather than something our port introduced.

### Impact and safety reading

* **Fail-safe**: host-side rejection, deterministic, reproducible → **no device execution, no GM
  over-read/over-write, no 507011, no MTE/AIV exception, no silent wrong results**
* the failure happens *before* any data is written, so the frozen FP32 forward contract is intact
  for every shape that stays on `SMALL_TAIL`
* `F` values above the threshold are therefore **unsupported** (they raise `RuntimeError` through the
  bridge), not silently incorrect

### Minimal fix (direction; NOT applied, NOT verified in this stage)

Register the second tiling key in the op build so `kernelList` contains `_0` **and** `_1`
(`--tiling_key` compile option consumed by the AscendC kernel build). Stage 3B attempted three
syntax variants through the template CMake and all were mangled by CMake list/ini parsing
(`ScatterMaxV1,ascend910b,1\"`, `...,1`, empty), so the exact wiring is deferred to a dedicated
stage. Verification criterion for that stage: rebuild and confirm
`"kernelList": [{"kernelName": "..._0"}, {"kernelName": "..._1"}]`, then re-run F=48825 (N=40) and
F=44737 (N=163800). Until then the delivery envelope of §1 stands.

## 6. largeTail memory-safety analysis (source)

For completeness, the address model of the two paths (from `op_kernel/scatter_max_v1.h`):

```
SMALL_TAIL (key 0, all validated shapes):
  SRC   : DataCopyPad(srcGM[row*F], blockLen = F*4 bytes)      -> byte-exact read  [safe]
  INDEX : DataCopy(idxGM[base], AlignUp(cnt,8) elements)       -> read rounds up to 32 B (<=28 B extra)
  RES   : DataCopyPad(resGM[idx*F], blockLen = F*4 bytes)      -> byte-exact write [no over-write]

LARGE_TAIL (key 1, cannot launch in this package):
  SRC   : DataCopy(srcGM[row*F + n*srcBatchNum], AlignUp(srcLoadNum,8))  -> 32 B-rounded read
  RES   : DataCopyPad(resGM[idx*F + ...], srcLoadNum*4 bytes)            -> byte-exact write
```

Only the 32 B-rounded `DataCopy` reads can touch up to 28 B past a logical row (and, for the very
last row, past the tensor end); the writes are always exact. This rounding already happens in the
validated `processLeftSrc` path for every tested shape (e.g. the index read is always 8 int32) and
was exercised thousands of times in Stage 1B/2/3A/6 without a fault. For `LARGE_TAIL` the risk could
not be measured because the kernel never launches (see §5).

## 7. size / index boundaries

| case | result |
|---|---|
| `S=0` with `N=0` | PASS (fast path, `[0,8]`) |
| `S=1` with N=1/40/41/4097 (all index 0) | PASS |
| sparse `S=1024` (only groups 0/17/1023 used) | PASS, empty rows 0, highest group correct |
| `index = 491518` (F=8, S=491519) | accepted, matched golden (≈47 MiB) |
| `index = 491519` (F=8, S=491520) | accepted, matched golden |
| `index = 491520` | **rejected before the kernel** (ValueError; no launch) |
| `N*(F+1)` budget guard | real code path exercised with a temporarily lowered budget → rejected; the real constant 4 026 531 840 (= 0xF0000000 ≈ 16 GiB fp32) was never allocated |

## 8. Correctness matrix summary

`run_stage3b_boundary_tests.py`: **45/45 PASS, 0 FAIL, 0 SKIP** (max_abs_diff = 0 on every case,
`inf`/`-inf`/NaN positions compared exactly, `padded`/`F_kernel` recorded per case).
Data patterns covered: deterministic `(n,f)`, all-same group, repeated index, leftover-row variants
(duplicate / new group / true max), negative-only, controlled `-inf` rows and columns.
Log: `/root/zyg/logs/stage3b/boundary_tests.log`, JSON: `boundary_results.json`.

## 9. Runtime profiler (adapter level, device 0)

| case | shape | ScatterMaxV1 | other device ops | AI_CPU |
|---|---|---|---|---|
| C1 smallTail | N=4096 F=33 S=64 | AI_VECTOR_CORE ×8 (avg 28.5us) | PadV3(MIX_AIV), ScatterElementsV2, MaskedFill, Slice | 0 |
| C2 leftSrc | N=4097 F=33 S=64 | AI_VECTOR_CORE ×8 (avg 33.1us) | PadV3(MIX_AIV), ScatterElementsV2, MaskedFill, Slice | 0 |
| C3 threshold-1 | N=40 F=48824 S=8 | AI_VECTOR_CORE ×8 (avg 50.6us) | ScatterElementsV2, MaskedFill (F aligned → no pad/crop) | 0 |

`HOST_CPU_OP_FALLBACK = NONE` in all three; raw trees under
`/root/zyg/profiler/stage3b/{C1_smallTail,C2_leftSrc,C3_threshold_minus_1}/`.
The largeTail case could not be profiled because no device task is ever created (§5).

## 10. Performance observation (no tuning)

| shape | latency/call (adapter, incl. host sync) |
|---|---|
| N=40, F=48824 (threshold-1, SMALL_TAIL) | ~1.9 ms |
| N=4096, F=33 (reference) | ~1.0–1.3 ms (Stage 6) |

The threshold-1 case is slower mostly because F=48824 makes the `MaskedFill`/occupancy steps operate
on 8×48824 rows; no largeTail-vs-smallTail comparison is possible. Observation only.

## 11. PyG end-to-end (real `torch_geometric.nn.global_max_pool`)

| case | shape | result |
|---|---|---|
| E2E-1 | N=41 F=33 S=8 (leftSrc) | PASS, max_diff 0 |
| E2E-2 | N=4097 F=33 S=8 (leftSrc) | PASS, max_diff 0 |
| E2E-3 | N=40 F=48824 S=8 (threshold-1) | PASS, max_diff 0 |
| E2E-4 | N=40 F=40000 S=8 | PASS, max_diff 0 |

`compat path used` (ascend_calls=4, original_calls=0), `aten::scatter_reduce` not called,
host fallback text **NONE**. Log/JSON: `/root/zyg/logs/stage3b/pyg_e2e_results.json`.

## 12. Frozen-baseline regression

* Stage 6 demo: **PASS**
* Stage 6 20-case suite: **20/20 PASS** (two consecutive runs), `BEFORE_HOST_CPU_FALLBACK=YES`,
  `AFTER_HOST_CPU_FALLBACK=NO`
* one earlier attempt produced 18/20 with the two *subprocess-probe* checks failing while device 0
  had only ~120 MB free HBM (other tenants); the same probes pass when run directly and in the two
  re-runs, so it was an environment (external HBM pressure) flake, not a regression. Recorded
  honestly in `/root/zyg/logs/stage3b/stage6_tests_regression.log` (flake) vs
  `..._run1.log` / `..._run2.log` (20/20).

## 13. Verdict

**STAGE3B = PARTIAL** — one precise blocker, everything else PASS:

* thresholds, core boundary, leftSrc, large N, size/index boundaries, patterns, profiler and PyG
  end-to-end are all PASS
* the `LARGE_TAIL` kernel path cannot execute in the delivered package (missing tiling-key entry
  `_1`); it fails **before device execution** with a deterministic 361001 error, so there is no
  correctness or memory-safety exposure
  **(updated by Stage 3C:** the entry was repaired — `_1` now exists — and the path then faults on
  the device with `507035` / MTE DDR out-of-range; the earlier "adapter raises a clear error"
  wording below describes the *old* package's host-side failure, **not** an adapter guard, and
  large F is therefore still unsupported pending a kernel/tiling fix or a real adapter guard. See
  `stage3c_large_tail_entry_repair.md`. **Stage 3D then root-caused and repaired it**: the fault was
  a garbage index (local lookup off by the block offset) producing a wild GM write, plus a missing
  `+ n*_srcBatchNum` chunk offset on the write side; with those two lines fixed the large-tail path
  passes LT0–LT6 and the PyG end-to-end cases — see `stage3d_large_tail_mte_repair.md`.)**
* delivery envelope update (documentation only, frozen contract unchanged): FP32 forward is
  supported for `F <= 44736` regardless of N, and up to `F <= 48824` for N ≤ 320; above the
  per-N threshold the op cannot run — with the frozen package the launch fails host-side (361001,
  see Stage 3C for the repaired-entry behaviour: device exception 507035)

## 14. Reproduction

```bash
source /root/zyg/global_max_pool/stage6/env.sh
python3 /root/zyg/global_max_pool/stage3b/tools/tiling_calc.py --table   # formulas
bash    /root/zyg/global_max_pool/stage3b/tools/run_largetail_threshold.sh  # tiling keys + raw runs
python3 /root/zyg/global_max_pool/stage3b/tests/run_stage3b_boundary_tests.py
python3 /root/zyg/global_max_pool/stage3b/tests/run_stage3b_pyg_e2e.py
```

## 15. Status update — Stage 3C/3D repair + Stage 3E promotion (2026-09-18)

The single blocker recorded in this report (large-tail `F` shapes: no kernel entry `_1`, then a
wild index-driven GM write) is closed:

* **Stage 3C** added the explicit `TILING_KEY_IS(1)` kernel branch plus
  `--tiling_key=0,1`, so the package now registers `..._0` **and** `..._1`.
* **Stage 3D** root-caused and fixed the two kernel defects
  (`_idxLocal.GetValue(idxOffset + k)` → `GetValue(k)`, and the missing `+ n*_srcBatchNum`
  chunk offset on the result write).
* **Stage 3E** promoted both repairs into the **formal delivery OPP**
  (`/root/zyg/build/scattermax_runtime_opp/vendors/customize`, rebuilt from
  `/root/zyg/build/scattermax_probe`), proved by an `LD_PRELOAD` open() trace that the runtime loads
  that package's kernel binary, and re-ran everything on it: LT0–LT6 + LTA1/LTA2 **9/9 PASS**,
  tail-attack + PyG E2E **13/13 PASS**, this report's matrix **45/45 PASS**, Stage 3A 34/34,
  Stage 3D 9/9, Stage 6 demo PASS and 20/20.

The envelope numbers in §13 above remain the tiling *thresholds*; shapes above them now run
correctly instead of failing. Evidence: `stage3e_large_tail_delivery_promotion.md`.
