# Stage 3D — largeTail MTE root-cause + kernel repair (evidence)

Date: 2026-09-18 · container `wio-pyg-cann851-pyg280` · CANN 8.5.1 · torch 2.9.0+cpu ·
torch_npu 2.9.0 · Ascend 910B3 · history `5816ef6 → 1a8df65 → 382301e → this commit`

## 1. Branch / HEAD chain

```
5816ef6 (frozen) → 1a8df65 (Stage 3B) → 382301e (Stage 3C) → <this commit> (Stage 3D)
```

All repairs were made in **probe-only copies** of the kernel/tiling; the delivered
`/root/zyg/build/scattermax_runtime_opp` and the DrivingDSDK tree are untouched.

## 2. Fault PC / instruction attribution

Device reports (plog, `device_error_core_proc.cc:PrintCoreInfo`) for the pre-fix runs:

```
LT1  N=40 F=48825 : pc start 0x12c041e00c94  current 0x12c041e01908  core 14/34  blk 33/34
LTA1 N=40 F=48832 : same signature
LTA2 N=40 F=48960 : pc start 0x12c041e00c6c  current 0x12c041e01894  core 11  blk 8
                    errType 0x1 (task exception), errorStr:
                    "The DDR address of the MTE instruction is out of range"  (subErrType 4)
ACL  : 507035 = ACL_ERROR_RT_VECTOR_CORE_EXCEPTION
```

No PC→source-line mapping tool was available in this container (no objdump-like AscendC
disassembler with the kernel debug info, and the `.o` carries only
`.ascend.meta.ScatterMaxV1_…_0/_1` markers), so attribution was done by **single-variable A/B
experiments** on the kernel source instead of by disassembly. The result is unambiguous: the fault
is caused by the local **index lookup** producing a wild GM write address (see §4/§5).

## 3. Complete large-tail DataCopy / MTE address audit

Parameters used below: `F=48825`, `N=40`, `S=8`, `tailElemNum=48825`, `srcBatchNum=48824`,
`idxBatchNum=1` (idxNumPerCore=1), `elemNumPerBlock=8`, `_srcLoop=2`, `UB=196352 B`,
`_srcBuf = _srcBatchNumAlign*4 = 195,296 B`, `_idxBuf = 32 B`.

| operation | direction | base | offset formula | logical elems | physical transfer | allocated extent | legal max | verdict |
|---|---|---|---|---|---|---|---|---|
| index load (`batchProcess`) | GM→UB | `_idxGM` | `idxOffset = idxNumPerCore*blockIdx + i*idxBatchNum` | `idxLoadNum=1`, `AlignUp→8` | `DataCopy` len 8 (element units) | index tensor `40*4 = 160 B` | 160 B | reads ≤28 B past the tensor for the last cores; values beyond `idxLoadNum` are unused → **no fault** (also true in the shipped smallTail path) |
| index lookup (`elemWiseBatchProcess`) | **UB (local)** | `_idxLocal` | `GetValue(idxOffset + k)` ← **bug** | 1 | local read of element `blockIdx` from a **32 B (8 entry)** buffer | `_idxBuf = 32 B` | 32 B | for core i this reads `_idxLocal[i]` (i up to 39) → **out-of-UB read → garbage index** |
| src row load (`elemWiseBatchProcess`) | GM→UB | `_srcGM` | `(idxOffset+k)*F + n*srcBatchNum` | `srcLoadNum = min(48824, F-n*48824)` → 48824 (n=0), 1 (n=1); `AlignUp(...,8)` | `DataCopy` (element units), ≤195,296 B | src tensor `40*48825*4 = 7.81 MB` | 7.81 MB | in-range for n=0; n=1 reads 8 elems (≤28 B past the row, ≤28 B past the tensor on the last row) → **no fault** |
| result write (`elemWiseBatchProcess`) | UB→GM | `_resGM` | `idxVal*F (+ n*srcBatchNum)` | `srcLoadNum` | `DataCopyPad`, exact bytes | res tensor `8*48825*4 = 1.56 MB` | 1.56 MB | with the garbage index (`idxVal`) the address `idxVal*F` can be arbitrarily far outside the tensor → **the wild write that the MTE reports**; after the index fix this site also needed the missing `+ n*srcBatchNum` (§4.2) |
| idem, `processLeftSrc` | GM→UB | `_srcGM` | `_leftSrcBaseOffset + i*_leftSrcBatchNum` | ≤ `_leftSrcBatchNum` | `DataCopy` (element units) | 7.81 MB | 7.81 MB | in-range (leftSrc path passes with the fix: LT3/LT5) |
| result write, `processLeftSrc` | UB→GM | `_resGM` | `idxVal*F + srcOffset % F` | `srcLoadNum` | `DataCopyPad`, exact bytes | 1.56 MB | 1.56 MB | offset already includes the chunk remainder → correct |

Note: an earlier Stage 3C hypothesis ("`DataCopyParams.blockLen` is in 32-B units, so the src load
reads 32× too much") was **tested and refuted** - see §5, change #1.

## 4. Actual root cause

Two independent defects, both in the **large-tail** (`elemWiseBatchProcess`) code path:

1. **Local index lookup off by the block offset (fatal).**
   `batchProcess()` loads the index block that starts at GM offset `idxOffset` into `_idxLocal[0..]`,
   so entry *k* of that block is `_idxLocal.GetValue(k)`. The large-tail code reads
   `_idxLocal.GetValue(idxOffset + k)`. With `idxBatchNum = 1` and `N = 40` the buffer holds 8
   entries (32 B) while `idxOffset` runs 0…39, so for every core with `idxOffset ≥ 8` the read
   leaves the local buffer, returns an arbitrary value, and that value is used as the destination
   row of `_resGM[idxVal * _tailElemNum]` - an effectively arbitrary GM address. The device reports
   exactly that: `MTE instruction address out of range` (`507035`).
   (The small-tail path uses `tailWisebatchProcess()` with `GetValue(k*_tailBatchNum + n)` and is
   correct - which is why every SMALL_TAIL shape always passed.)
2. **Missing chunk offset on the write side (correctness).**
   The source row is read in chunks `n = 0 … _srcLoop-1` at
   `srcOffset = row*F + n*srcBatchNum`, but the write-back used `_resGM[idxVal*F]` without
   `+ n*srcBatchNum`. Whenever `F > srcBatchNum` (i.e. exactly the large-tail regime) chunk 1…n
   overwrites the row head and the row tail stays at the initial `-inf`
   (observed: `max_abs_diff=inf`, 10 mismatching elements for N=40 F=48825).

## 5. A/B isolation evidence (one change per build)

| build | change | LT1 N=40 F=48825 |
|---|---|---|
| `stage3c_opp_fixed` (Stage 3C) | entry `_1` only | 507035 (MTE DDR out of range) |
| `stage3d_fix1` | #1 src `DataCopy` → byte-exact `DataCopyPad` | **still 507035** → hypothesis refuted |
| `stage3d_fix2` | #2 index lookup `GetValue(idxOffset+k)` → `GetValue(k)` | **no fault**, but `max_abs_diff=inf` (10 elems) |
| `stage3d_fix3` | #1 + #2 | no fault, still mismatching |
| `stage3d_fix4` | #2 + #3 write offset `+ n*srcBatchNum` | **PASS, max_abs_diff = 0** |

## 6. Minimal kernel fix

Exactly two one-line semantic fixes in `op_kernel/scatter_max_v1.h` (large-tail paths only);
no tiling change, no algorithm change, key 0/1 semantics untouched:

```c
-    DTYPE_INDEX idxVal = _idxLocal.GetValue(idxOffset + k);
+    DTYPE_INDEX idxVal = _idxLocal.GetValue(k);

-    DataCopyPad(_resGM[idxVal * _tailElemNum], _srcLocal, {1, srcLoadNum*sizeof(DTYPE_RES), …});
+    DataCopyPad(_resGM[idxVal * _tailElemNum + n * _srcBatchNum], _srcLocal,
+                {1, srcLoadNum*sizeof(DTYPE_RES), …});
```

(tools: `tools/apply_largetail_index_fix.py`, `tools/apply_largetail_write_offset_fix.py`;
Stage 3C's entry repair `tools/apply_largetail_entry_fix.py` + `--tiling_key=0,1` remain required.)

## 7. Exact source diff (probe copy)

See §6 - the two lines above, applied to the copy that also carries the Stage 3C entry repair and the
`add_ops_compile_options(ScatterMaxV1 OPTIONS --tiling_key=0,1)` declaration. Kernel package under
test: `/root/zyg/build/stage3d_opp_fix4` (`kernelList = [_0, _1]`).

## 8. LT1 / LTA1 / LTA2 (one process each, tol 0)

| case | shape | tiling | result |
|---|---|---|---|
| LT1 | N=40 F=48825 S=8 | LARGE_TAIL(1) | **PASS** max_diff 0 |
| LTA1 | N=40 F=48832 S=8 (32 B aligned) | LARGE_TAIL(1) | **PASS** |
| LTA2 | N=40 F=48960 S=8 (32 B/512 B aligned) | LARGE_TAIL(1) | **PASS** |
| LT0 control | N=40 F=48824 S=8 | SMALL_TAIL(0) | **PASS** |

## 9. LT2–LT6 (one process each, tol 0)

| case | shape | tiling | result |
|---|---|---|---|
| LT2 non-aligned | N=40 F=48826 S=8 | LARGE_TAIL(1) | **PASS** |
| LT3 largeTail + leftSrc | N=41 F=48825 S=8 | LARGE_TAIL(1) | **PASS** |
| LT4 | N=80 F=48825 S=8 | LARGE_TAIL(1) | **PASS** |
| LT5 largeTail + leftSrc | N=81 F=48825 S=8 | LARGE_TAIL(1) | **PASS** |
| LT6 | N=320 F=48825 S=8 (negative data) | LARGE_TAIL(1) | **PASS** |

Summary: `/root/zyg/logs/stage3d/largetail_matrix_summary.txt` (LT0–LT6 all PASS, failed=0).

## 10. Tail-attack correctness (adapter + PyG, tol 0)

`tests/run_stage3d_largetail_tests.py` → **9/9 PASS** (`largetail_correctness.log`):

* A: N=40 F=48825, last physical row + **first feature** and **last logical feature** hold unique
  maxima → PASS, `last_row_group_max_ok=True`
* B: N=41 F=48825, repeated index (all rows → group 0) + tail winner → PASS (`row0_max=4321`)
* C: N=40 F=48826 (non-aligned → padded 48832) → PASS
* D: negative-only data → PASS
* E: `-inf` row vs empty group → PASS (`max(-inf, -1) = -1`, empty group 0)
* F: explicit `size` larger than `max(index)+1` → PASS
* PyG E2E: E2E_LT1 (N=40 F=48825), E2E_LT2 (N=41 F=48825), E2E_LT3 (N=40 F=48826) → all PASS,
  `ascend_calls=3, original_calls=0`

## 11. 507035 / MTE status

* before the fix: `507035` (vector core exception) + `MTE DDR address out of range`, reproducible on
  F=48825 / 48832 / 48960
* after the fix: **no 507035, no MTE OOB, no AIV exception, no hang** in LT0–LT6, LTA1/LTA2, the
  tail-attack suite, P1–P3 profiling and the PyG E2E
* no `507011` at any point in Stage 3D; no host CPU fallback

## 12. Profiler (repaired package)

| case | shape | ScatterMaxV1 | AI_CPU | app exit |
|---|---|---|---|---|
| P1 | N=40 F=48825 | AI_VECTOR_CORE ×8, avg 11.5 us | 0 | OK (completed) |
| P2 | N=41 F=48825 (leftSrc) | AI_VECTOR_CORE ×8, avg 13.9 us | 0 | OK |
| P3 | N=40 F=48826 (padded) | AI_VECTOR_CORE ×8, avg 11.4 us | 0 | OK |

Raw: `/root/zyg/profiler/stage3d/{P1_largeTail,P2_largeTail_leftsrc,P3_largeTail_nonaligned}/`,
logs `/root/zyg/logs/stage3d/msprof_P*.log`.

## 13. PyG end-to-end (repaired package)

`torch_geometric.nn.global_max_pool` with `pyg_ascend_compat.enable()`:

```
E2E_LT1 N=40 F=48825 : PASS (max_diff 0)
E2E_LT2 N=41 F=48825 : PASS (max_diff 0)
E2E_LT3 N=40 F=48826 : PASS (max_diff 0)
compat counters: ascend_calls=3, original_calls=0
```

`aten::scatter_reduce` is not called (the compat path never calls it); no fallback warnings.

## 14. Stage 3B regression

`run_stage3b_boundary_tests.py` → **45/45 PASS, 0 FAIL, 0 SKIP** + SMALL_TAIL threshold control
(N=40 F=48824, raw) **PASS** (`logs/stage3d/stage3b_regression.log`,
`smalltail_control_regression.log`).

## 15. Stage 6 demo

`demo_global_max_pool_ascend.py` → **PASS** (`stage6_demo_regression.log`).

## 16. Stage 6 20-case suite

**20/20 PASS**, `BEFORE_HOST_CPU_FALLBACK=YES`, `AFTER=NO` (`stage6_tests_regression.log`).

## 17. Git / bundle

* Stage 3D commit on top of `382301e` (see the final chat report for the SHA); history unchanged:
  `5816ef6 → 1a8df65 → 382301e → <Stage 3D>`
* files: `stage3d_large_tail_mte_repair.md`, `global_max_pool/stage3d/{README.md,
  tools/apply_largetail_index_fix.py, tools/apply_largetail_write_offset_fix.py,
  tools/apply_largetail_datacopy_fix.py, tests/run_stage3d_largetail_tests.py}` plus a correction
  paragraph in `stage3b_large_tail_boundary.md` / `stage3c_large_tail_entry_repair.md`
* no binaries / profiler raw / build trees committed; not pushed
* bundle `/data/wio/zyg-src/backup/nanwang-<HEAD>.bundle`, `git bundle verify` OK; older bundles kept

## 18. Remaining risks

1. The fix lives in a **probe copy**: the delivered OPP must be rebuilt with the three changes
   (entry repair + 2 kernel lines) before the shipped package supports large F.
2. `index < 491520` and `N*(F+1) < 4,026,531,840` remain documented-limit guards (unchanged).
3. The threshold is still N-dependent (`F ≥ 44737` for N ≥ 163800); shapes beyond ~29 GB of
   tensors are not exercised (shape-only tiling evidence only).
4. The fix was validated on N ≤ 320 / F ≤ 48960 at the large-tail side; the extreme
   N ≥ 163800 + large-tail combination remains analytically supported but not executed (memory).

## 19. Verdict

**STAGE3D: PASS**

root cause proven (A/B, single change per build) · minimal fix explained (2 lines) · largeTail
kernel completes (LT0–LT6, LTA1/LTA2) · correctness PASS (tol 0, incl. tail attacks) · largeTail +
leftSrc PASS (LT3/LT5) · largeTail + padding PASS (LT2/P3/E2E_LT3) · 507035 absent · MTE OOB absent ·
PyG E2E no fallback · Stage 3B 45/45 · Stage 6 20/20.

```
STAGE 3B LARGE-TAIL / BOUNDARY VALIDATION IS NOW CLOSED.
READY FOR HUMAN REVIEW BEFORE STAGE 4.
```

---

## 20. Status update — Stage 3E promotion (2026-09-18)

Risk #1 of §18 ("the fix lives in a probe copy") is closed. Both one-line kernel repairs — plus the
Stage 3C entry repair and a byte-exact index load (Stage 3E Task E, see below) — are now in the
**formal delivery source** `/root/zyg/build/scattermax_probe`, were clean-rebuilt into the formal
delivery OPP, and the whole Stage 3D evidence set was re-produced on that package:

* LT0–LT6 + LTA1/LTA2 = **9/9 PASS**, `max_abs_diff = 0`
* tail-attack suite + PyG E2E (`E2E_LT1/2/3`) = **13/13 PASS**, `ascend_calls=3 original_calls=0`
* profiler: `ScatterMaxV1` = **AI_VECTOR_CORE** (entry `_1`), `AI_CPU = 0`, `aten::scatter_reduce`
  not called, no `507035` / `507011` / MTE OOB / AIV exception / host fallback
* Stage 3A 34/34 · Stage 3B 45/45 · Stage 3D 9/9 · Stage 6 demo PASS · Stage 6 20/20

Additional repair promoted in Stage 3E (Task E): the index GM loads used the 32 B-block
`DataCopy(..., AlignUp(idxLoadNum, 8))` form, so the last core could read up to 28 B past the
logical index tensor. They are now byte-exact `DataCopyPad` loads; index values/semantics are
unchanged. Full evidence: `stage3e_large_tail_delivery_promotion.md`.
