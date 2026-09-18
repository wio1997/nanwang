# Stage 3E — delivery promotion + final freeze (evidence)

Date: 2026-09-18 · container `wio-pyg-cann851-pyg280` · CANN 8.5.1 · torch 2.9.0+cpu ·
torch_npu 2.9.0 · PyG 2.8.0.post1 · Ascend 910B3 (device 0) · driver 26.0.rc1

Branch `feat/global-max-pool-scattermax-zyg`, history
`5816ef6 → 1a8df65 → 382301e → 028112b → <Stage 3E commit>`.

**Verdict: STAGE3E = PASS** — the Stage 3C/3D largeTail repairs are promoted into the formal
delivery OPP, the runtime provably loads that package, and the whole largeTail matrix + full
regression is green on it.

---

## 1. The delivery chain (Task A)

| role | path |
|---|---|
| upstream kernel reference (pristine when promoted) | `/root/zyg/DrivingSDK/kernels/scatter_max/` (git `27375a9c…`, clean) |
| **formal delivery build source** | `/root/zyg/build/scattermax_probe/` |
| kernel source | `…/scattermax_probe/op_kernel/scatter_max_v1.{cpp,h}` |
| op_host / tiling source | `…/scattermax_probe/op_host/scatter_max_v1.cpp` |
| kernel CMakeLists | `…/scattermax_probe/op_kernel/CMakeLists.txt` |
| build script | `…/scattermax_probe/build.sh` (+ `CMakePresets.json`, CANN 8.5.1, `ascend910b`, vendor `customize`) |
| package | `…/scattermax_probe/build_out/custom_opp_ubuntu_aarch64.run` |
| **formal runtime OPP** | `/root/zyg/build/scattermax_runtime_opp/vendors/customize` |

How the formal chain was identified (not assumed): the CPack staging tree of
`build_out/custom_opp_ubuntu_aarch64.run` was compared file-by-file with the installed runtime OPP —
**22/22 files byte-identical**, the only extra file being the installer-generated
`bin/set_env.bash` (see `logs/stage3e/00_pre_state.txt` §7). The `scattermax_probe` sources are also
byte-identical to the DrivingSDK reference (`e01846f8…` cpp / `8bbf0ea3…` h pre-Stage3E).

Probe-only artifacts (explicitly **not** the delivery) were the `stage3b_fixtest*`,
`stage3b_tiling_probe*`, `stage3c_fix*`/`stage3c_opp_*` and `stage3d_fix*`/`stage3d_opp_fix*`
copies. They have been quarantined to `/root/zyg/build/attic_stage3e/` so no runtime path can
reach them any more.

## 2. Exact promoted source diff (Task B)

Applied by `stage3e/tools/01_apply_delivery_fixes.py` to `/root/zyg/build/scattermax_probe`
(strict anchors, count-checked, idempotent). Full diff: `logs/stage3e/01_apply_fixes.log` and
`logs/stage3e/01_build_delivery_opp.log` §"promoted source diff".

```diff
--- a/op_kernel/scatter_max_v1.cpp
+++ b/op_kernel/scatter_max_v1.cpp
@@ -18,7 +18,7 @@
     if (TILING_KEY_IS(0)) { // TILING_KEY_SMALL_TAIL
         KernelScatterMaxV1<true> op(src, idx, res, argmax, &tiling_data, &pipe);
         op.Process();
-    } else { // TILING_KEY_LARGE_TAIL
+    } else if (TILING_KEY_IS(1)) { // TILING_KEY_LARGE_TAIL
         KernelScatterMaxV1<false> op(src, idx, res, argmax, &tiling_data, &pipe);
         op.Process();
     }
--- a/op_kernel/CMakeLists.txt
+++ b/op_kernel/CMakeLists.txt
@@ -4,3 +4,5 @@
 endif()

 add_kernels_compile()
+
+add_ops_compile_options(ScatterMaxV1 OPTIONS --tiling_key=0,1)
--- a/op_kernel/scatter_max_v1.h
+++ b/op_kernel/scatter_max_v1.h
@@ largeTail local index lookup (x2: ScatterMaxV1 + ScatterMaxArgmaxV1)
-        DTYPE_INDEX idxVal = _idxLocal.GetValue(idxOffset + k);
+        DTYPE_INDEX idxVal = _idxLocal.GetValue(k);
@@ largeTail result write (x1)
-            DataCopyPad(_resGM[idxVal * _tailElemNum], _srcLocal,
+            DataCopyPad(_resGM[idxVal * _tailElemNum + n * _srcBatchNum], _srcLocal,
@@ index GM loads -> byte-exact DataCopyPad (x4: batch x2 + leftSrc x2)   [Task E]
-        DataCopy(_idxLocal, _idxGM[idxOffset], idxLoadNumAlgin);
+        DataCopyExtParams idxCopyParams = {1, static_cast<uint32_t>(idxLoadNum * sizeof(DTYPE_INDEX)), 0, 0, 0};
+        DataCopyPad(_idxLocal, _idxGM[idxOffset], idxCopyParams, {0, 0, 0, 0});
@@
-        DataCopy(_idxLocal, _idxGM[_leftSrcIdxPos], _elemNumPerBlock);
+        DataCopyExtParams idxCopyParams = {1, static_cast<uint32_t>(sizeof(DTYPE_INDEX)), 0, 0, 0};
+        DataCopyPad(_idxLocal, _idxGM[_leftSrcIdxPos], idxCopyParams, {0, 0, 0, 0});
```

Deliberately **not** promoted: the probe-only `[stage3b-tiling]` stderr print in the probe copy of
`op_host/scatter_max_v1.cpp`. The delivery op_host is therefore byte-identical to the DrivingSDK
reference, and the tiling decisions used below come from that identical source (see §9).
No algorithm, tiling threshold, PyG semantics or op interface was changed.

The fixed source is frozen in this commit under `global_max_pool/stage3e/delivery_source/` together
with a hash manifest.

## 3. Clean rebuild (Task C)

```bash
cd /root/zyg/build/scattermax_probe && rm -rf build_out && bash build.sh
```

* environment note (newly documented): the AscendC `opc` front-end needs the `decorator` module,
  which is not part of this image's python3.11.14 site-packages; the build must run with
  `PYTHONPATH=/root/pyg_feasibility/R009-scattermax-raw-callability/deps:$PYTHONPATH`
  (the same dependency directory the earlier Stage 1B/3C/3D builds used).
* result: exit 0, 39 s, package `custom_opp_ubuntu_aarch64.run` regenerated
  (`logs/stage3e/01_build_delivery_opp.log`, stdout+stderr in the same file).
* the generated `opc` command line contains `--tiling_key="0,1"` (Fix 1b is live).

Kernel metadata gate (packaged, and after install):

```
kernelList : [ScatterMaxV1_7d55161965c898907fdb3028d01c7c76_0,
              ScatterMaxV1_7d55161965c898907fdb3028d01c7c76_1]
supportInfo.tilingKey : ["0","1"]
coreType : VectorCore   magic : RT_DEV_BINARY_MAGIC_ELF_AIVEC
sha256   : aacd96e17ef1902b218bc3242027d5a3ed6c8ae1fd5cebe34a36dbfe57b22a12
```

| artifact | pre-Stage3E | promoted |
|---|---|---|
| kernel json md5 | `e0545e9ab70aca51b4d9982685ac1e33` | `5b708412f292fd789c44f91c5f02ba48` |
| kernel `.o` md5 | `2f33d478509f7c6b4c95dadeb78eb050` | `5f27b82e8e0828d52e063c9c4a556845` |
| declared sha256 | `95f878a7…f812998` | `aacd96e1…57b22a12` |
| `kernelList` | `[_0]` | `[_0,_1]` |
| `supportInfo.tilingKey` | absent | `["0","1"]` |

The packaged dynamic source is byte-identical to the promoted build source
(`scatter_max_v1.h` `dde990670c8a14969869669b4949308e`, `.cpp` `c4e087beed4f430d2bfe0eacca86803d`)
and contains all four repairs (`logs/stage3e/09_delivery_package_verification.txt`).
`op_api`, `op_proto` and the tiling library are unchanged by the promotion
(`libcust_opapi.so` `80ce5dc8…`), so the pre-built bridge extension stays valid.

Install/package paths:

```
package        /root/zyg/build/scattermax_probe/build_out/custom_opp_ubuntu_aarch64.run
install cmd    ./custom_opp_ubuntu_aarch64.run --quiet --install-path=/root/zyg/build/scattermax_runtime_opp
runtime OPP    /root/zyg/build/scattermax_runtime_opp/vendors/customize   (used by repo stage6/env.sh)
preserved old  /root/zyg/build/attic_stage3e/scattermax_runtime_opp_pre_stage3e
quarantined    /root/zyg/build/attic_stage3e/{stage3b_*_opp,stage3c_opp_*,stage3d_opp_fix*}
```

## 4. Runtime really uses the formal OPP (Task D)

1. `ASCEND_CUSTOM_OPP_PATH` is unset in the container image; every Stage 3E run sourced exactly
   `…/scattermax_runtime_opp/vendors/customize/bin/set_env.bash`.
2. All probe OPPs were moved out of `/root/zyg/build` before the runtime matrix.
3. **`LD_PRELOAD` open() trace of a real LARGE_TAIL execution** (the only ScatterMax path opened):

```
/root/zyg/build/scattermax_runtime_opp/vendors/customize/op_impl/ai_core/tbe/kernel/ascend910b/scatter_max_v1/ScatterMaxV1_7d55161965c898907fdb3028d01c7c76.o
PROVENANCE: the runtime opened kernel metadata/binary from the FORMAL delivery OPP
PROBE LEAK: none
```

   (`logs/stage3e/03_runtime_opp_provenance.txt`, shim `stage3e/tools/opentrace_shim.c`; the same
   traced run returned `RESULT PASS max_abs_diff=0`.)
4. The profiler's kernel-name dictionary of every Stage 3E run contains
   `ScatterMaxV1_7d55161965c898907fdb3028d01c7c76_1` — an entry that **did not exist** in the
   pre-Stage3E package (it had only `_0`; a key-1 launch then failed with aclnn 361001).
5. The installed `.o` md5 differs from the pre-Stage3E package (`5f27b82e…` vs `2f33d478…`).

## 5. Index GM 32B over-read closure (Task E)

**Finding: `INDEX_GM_OVERREAD = REMOVED_BY_FIX`.**

CANN 8.5.1 first-hand evidence (all under `/usr/local/Ascend/cann-8.5.1/aarch64-linux/asc/`):

* `impl/basic_api/kernel_operator_data_copy_intf_impl.h:659-694` — the Level-2
  `DataCopy(LocalTensor, GlobalTensor, uint32_t count)` overload asserts
  `count % GetC0Count(sizeof(T)) == 0` with the message *"count * sizeof(T) must be 32B align"* and
  sets `repeatParams.blockLen = count / GetC0Count(sizeof(T))`, i.e. **`DataCopyParams.blockLen`
  counts 32-byte blocks** (`GetC0Count = GetC0Size()/dtypeSize`, `kernel_utils_base.h:74-78`;
  for int32 that is 8 elements). The minimum transfer is therefore 32 B.
* `impl/basic_api/dav_c220/kernel_operator_data_copy_impl.h:404-416` (`DataCopySliceGm2UBImpl`)
  converts `blockLen * ONE_BLK_SIZE` — same 32 B unit.
* The framework itself treats a transfer that leaves the registered GM range as a fault:
  `kernel_utils_base.h:362-423` (`CheckGmMemOverflowNormal` → `GetGMLen` → `CheckGmMemOverflow`)
  computes the transferred span and `trap()`s (`0x5A5A0001`) when it leaves the recorded GM range
  (ASCENDC_OOM build option). There is **no clause** allowing a read up to the 32 B block boundary
  past the logical tensor end, so `SAFE_BY_SPEC` could not be claimed honestly.
* `DataCopyPad` (GM→UB, `DataCopyExtParams`) takes `blockLen` in **bytes**
  (`CheckDataCopyPadParams` only requires divisibility by `sizeof(T)`) and dispatches to
  `copy_gm_to_ubuf_align_b32` (`dav_c220/…:457-489`), i.e. arbitrary byte lengths.

The delivered kernel requested `AlignUp(idxLoadNum, 8)` = 32 B for a 1-element (4 B) logical read,
so the last core could read up to 28 B past the index tensor. Fix: the four index GM loads
(`batchProcess` and `processLeftSrc` in both kernel classes) now use byte-exact `DataCopyPad`
(4 B / `idxLoadNum*4 B`). Index *values and semantics* are untouched — only the transfer length
changed. Evidence: `logs/stage3e/09_delivery_package_verification.txt` §3 (4 byte-exact index loads,
0 legacy loads in the packaged source); the whole small-tail regression suite (Stage 3A/3B/6) and
the largeTail matrix were re-run after this change.

Related, pre-existing and **not** part of the index scope: the large-tail `elemWiseBatchProcess`
still loads source rows with `DataCopy(..., AlignUp(srcLoadNum,8))`, which can read up to 28 B past
the last row of the last chunk. It is a data read (never written back, values beyond `srcLoadNum`
are unused), it lies inside the runtime allocation, and it has never faulted; it is recorded in §11
as an open hardening item rather than silently declared safe.

## 6. largeTail runtime matrix (Task F) — formal OPP, one shape per process

| case | shape | tiling (from identical op_host) | result |
|---|---|---|---|
| LT0 | N=40 F=48824 S=8 | SMALL_TAIL(0) | **PASS** max_abs_diff=0 |
| LT1 | N=40 F=48825 S=8 | LARGE_TAIL(1) | **PASS** max_abs_diff=0 |
| LT2 | N=40 F=48826 S=8 (non-aligned) | LARGE_TAIL(1) | **PASS** max_abs_diff=0 |
| LT3 | N=41 F=48825 S=8 (leftSrc) | LARGE_TAIL(1) | **PASS** max_abs_diff=0 |
| LT4 | N=80 F=48825 S=8 | LARGE_TAIL(1) | **PASS** max_abs_diff=0 |
| LT5 | N=81 F=48825 S=8 (leftSrc) | LARGE_TAIL(1) | **PASS** max_abs_diff=0 |
| LT6 | N=320 F=48825 S=8 (negative data) | LARGE_TAIL(1) | **PASS** max_abs_diff=0 |
| LTA1 | N=40 F=48832 S=8 (32 B aligned) | LARGE_TAIL(1) | **PASS** max_abs_diff=0 |
| LTA2 | N=40 F=48960 S=8 (512 B aligned) | LARGE_TAIL(1) | **PASS** max_abs_diff=0 |

`logs/stage3e/04_largetail_matrix_summary.txt`, per-case logs `04_LT*.log` / `04_LTA*.log`.
Repeated indices are exercised by every case (index = row % S); no case triggered 507035/507011,
MTE OOB, AIV exception, hang or host fallback. (The delivery package carries no debug print, so the
tiling column comes from the identical host tiling source, measured in §9.)

## 7. Tail attack + PyG E2E (Task G/H) — 13/13 PASS

`logs/stage3e/05_largetail_tests.log` (+ `.json`), CPU golden, tol 0, full-tensor comparison:

```
A_tail_attack_N40_F48825              PASS max_diff=0  group_max=4321.0        (last row, first+last feature)
A2_final_chunk_vs_head_N40_F48825     PASS max_diff=0  row_max=4321.0          (final chunk must not overwrite head)
B_repeated_index_N41_F48825           PASS max_diff=0  row0_max=4321.0         (all rows -> group 0 + tail winner)
C_non_aligned_N40_F48826              PASS max_diff=0                          (padded 48826 -> 48832)
D_negative_only_N40_F48825            PASS max_diff=0                          (negative-only)
E_neg_inf_vs_empty_N3_F48825          PASS max_diff=0  g0_all_neg_inf=True g1_empty_zero=True g2_all_minus1=True
F_explicit_size_N40_F48825            PASS max_diff=0                          (explicit size > max(index)+1)
G_final_chunk_edges_N40_F48832        PASS max_diff=0  row_max=4321.0          (first/last column of the final chunk)
H_leftsrc_negative_repeat_N41_F48825  PASS max_diff=0                          (leftSrc + negative + repeated index)
E2E_LT1_N40_F48825                    PASS max_diff=0  fallback=NONE
E2E_LT2_N41_F48825                    PASS max_diff=0  fallback=NONE
E2E_LT3_N40_F48826                    PASS max_diff=0  fallback=NONE
E2E_dispatch_counters                 PASS ascend_calls=3 original_calls=0
TOTAL 13  PASS 13  FAIL 0
```

## 8. Profiler (Task I) — formal OPP

| case | shape | ScatterMaxV1 tasks | core type | AI_CPU | `aten::scatter_reduce` |
|---|---|---|---|---|---|
| P1 | N=40 F=48825 | 8 | **AI_VECTOR_CORE** (avg 12.34 µs) | 0 | 0 occurrences |
| P2 | N=41 F=48825 | 8 | **AI_VECTOR_CORE** (avg 13.57 µs) | 0 | 0 occurrences |
| P3 | N=40 F=48826 | 8 | **AI_VECTOR_CORE** (avg 11.59 µs) | 0 | 0 occurrences |

Every run's kernel-name dictionary contains `ScatterMaxV1_7d55161965c898907fdb3028d01c7c76_1`
(entry `_1`). `logs/stage3e/06_profiler_summary.txt`, raw data under
`/root/zyg/profiler/stage3e/{P1_largeTail,P2_largeTail_leftsrc,P3_largeTail_nonaligned}`,
msprof logs `logs/stage3e/06_msprof_P*.log`.

## 9. Extreme combined shape (shape-only)

`logs/stage3e/08_extreme_shape_tiling.txt` — host-tiling probe only (`aclnnScatterMaxV1GetWorkspaceSize`,
no kernel launch), instrumented probe OPP reached through a temporary symlink and removed again:

```
N=40     F=48824  tailBatchNum=1  TILING_KEY=SMALL_TAIL(0)
N=40     F=48825  tailBatchNum=0  TILING_KEY=LARGE_TAIL(1)
N=163800 F=44736  idxNumPerCore=4095 idxBatchNum=4095 srcBatchNum=44736 tailBatchNum=1 TILING_KEY=SMALL_TAIL(0)
N=163800 F=44737  idxNumPerCore=4095 idxBatchNum=4095 srcBatchNum=44736 tailBatchNum=0 TILING_KEY=LARGE_TAIL(1)
N=163800 F=44738  idxNumPerCore=4095 idxBatchNum=4095 srcBatchNum=44736 tailBatchNum=0 TILING_KEY=LARGE_TAIL(1)
```

`srcBatchNum=44736` ⇒ `remainUbSize = 178944 B`, matching the documented threshold arithmetic.
**The extreme combined `N ≥ 163800` + largeTail runtime was not exercised — the input alone would
need ≈29 GB of HBM. Shape-only tiling evidence only.**

## 10. Full regression on the formal OPP (Task J)

| suite | result | log |
|---|---|---|
| Stage 3A | **34/34 PASS** | `logs/stage3e/07_stage3a.log` |
| Stage 3B | **45/45 PASS, 0 SKIP** | `logs/stage3e/07_stage3b.log` |
| Stage 3D (original suite re-run) | **9/9 PASS** | `logs/stage3e/07_stage3d.log` |
| Stage 3E tail attack | **13/13 PASS** | `logs/stage3e/07_stage3e_largetail.log` |
| Stage 6 demo | **PASS** | `logs/stage3e/07_stage6_demo.log` |
| Stage 6 matrix | **20/20 PASS**, `BEFORE_HOST_CPU_FALLBACK=YES AFTER=NO` | `logs/stage3e/07_stage6_tests.log` |

## 11. Failure-marker summary

| marker | Stage 3E result |
|---|---|
| `507035` vector core exception | **NONE** in any Stage 3E runtime log |
| `507011` | **NONE** |
| MTE DDR address out of range | **NONE** |
| AIV exception / aclnn launch failure | **NONE** |
| AI_CPU tasks (profiler) | **0** |
| Host CPU fallback on the compat path | **NONE** (only the deliberate BEFORE phase of the Stage 6 suite shows it) |
| `aten::scatter_reduce` | **NOT CALLED** (0 api_statistic hits) |

## 12. Guards kept unchanged

```
index < 491520
N*(F+1) < 4,026,531,840
N >= 163800 combined with largeTail: shape-only evidence, runtime not exercised (HBM budget)
```

Also unchanged / out of scope: `ScatterMaxArgmaxV1` still registers a single kernel entry (`_0`); it
is shipped by the same package but is not part of this delivery and is not validated here.

## 13. Repository artifacts

```
pyg-ascend-compat/global_max_pool/stage3e/
  README.md
  tools/{00_pre_state.sh,01_apply_delivery_fixes.py,02_build_delivery_opp.sh,03_install_delivery_opp.sh,
         04_prove_loaded_opp.sh,09_verify_delivery_package.sh,10_final_gate.sh,opentrace_shim.c,
         run_stage3e_largetail_matrix.sh,run_stage3e_profiler.sh,run_stage3e_regressions.sh}
  tests/{run_stage3e_largetail_tests.py,profile_stage3e_case.py}
  delivery_source/{MANIFEST.md,op_kernel/scatter_max_v1.cpp,op_kernel/scatter_max_v1.h,
                   op_kernel/CMakeLists.txt,op_host/scatter_max_v1.cpp}
```

No binaries, profiler raw data, build trees or large logs are committed.
