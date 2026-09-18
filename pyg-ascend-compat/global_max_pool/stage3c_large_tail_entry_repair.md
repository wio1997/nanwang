# Stage 3C — largeTail kernel entry repair + runtime closure (evidence)

Date: 2026-09-18 · container `wio-pyg-cann851-pyg280` · CANN 8.5.1 · torch 2.9.0+cpu ·
torch_npu 2.9.0 · Ascend 910B3 · frozen baseline `5816ef6` → Stage 3B `1a8df65`

## 1. Workspace

* branch `feat/global-max-pool-scattermax-zyg`
* frozen baseline `5816ef6` (untouched), Stage 3B `1a8df65` (untouched)
* HEAD before this stage: `1a8df65`; HEAD after: see §15
* the repaired package was built in an **isolated** directory; the delivered
  `/root/zyg/build/scattermax_runtime_opp` and the DrivingSDK tree were never modified

## 2. Exact root cause (three layers, each with local evidence)

1. **Runtime lookup is by tiling key.** `indv_bininfo.cpp:GetFuncHandleByEntry` calls
   `aclrtBinaryGetFunctionByEntry(binHandle, tilingKey, &funcHandle)`. The delivered package's
   kernel binary contained only one entry (`..._0`), so every `LARGE_TAIL` shape failed with
   `aclnn status 361001` (`rtsFuncGetByEntry ErrCode=107000`).
2. **The entry list comes from the kernel source, not from the build flag.**
   `tbe/tikcpp/ascendc_compile_gen_json.py::_dynamic_kernel_list_to_json(kernel_name,
   tiling_key_list, …)` fills `kernelList`, and `tbe/tikcpp/ascendc_common_utility.py:879`
   derives `tiling_key_list` from `config.tiling_key_infos`, i.e. from the kernel entry's
   **explicit `TILING_KEY_IS(<key>)` patterns** (`kernel_info_infer.py`). `scatter_max_v1.cpp`
   wrote the large-tail branch as a bare `else` → only key 0 was registered.
3. **The key list must also be declared to the compiler**, through
   `add_ops_compile_options(<OP> OPTIONS --tiling_key=0,1)`:
   CMake helper → `ascendc_gen_options.py` (splits on `,`, writes
   `<OP>@<unit>@--tiling_key=0;1` into `custom_opc_options.ini`) →
   `ascendc_bin_param_build.py` (parses `--tiling_key`, appends `--tiling_key="0,1"` to the
   `opc` command). The flag alone is **not** sufficient: it only enriched
   `supportInfo.tilingKey` and still produced a single entry.
   Extra quirk found on the way: `ascendc_compile_kernel.py:90` passes `self.op_type` where
   `parse_op_debug_confg(opc_file, soc)` expects the SoC, and that function compares the full
   soc string with `trans_soc_verion(field)` (a *short* name, e.g. `ascend910b` → `ascend910`),
   so a declaration that includes `COMPUTE_UNIT ascend910b` is silently discarded. Declaring
   **without** the compute unit (the idiomatic `add_ops_compile_options(<OP> OPTIONS …)`) writes
   an empty unit and bypasses that filter.

## 3. CANN 8.5.1 tiling-key syntax evidence (first-hand)

```
$ opc --help | grep -A2 tiling_key
  --tiling_key        Set tiling key list for op, default is None.
                      For expl: --tiling_key=1,2,3,4

$ cat <probe>/build_out/autogen/custom_opc_options.ini
ScatterMaxV1@@--tiling_key=0;1

$ grep -n "tiling_key" <probe>/build_out/op_kernel/ScatterMaxV1_ascend910b/bin_param/*.sh   # regenerated
  res=$(opc $1 --main_func=scatter_max_v1 --input_param=… --soc_version=Ascend910B1 \
        --output=$2 --impl_mode=high_performance,optional --simplified_key_mode=0 \
        --op_mode=dynamic  --tiling_key="0,1")
```

## 4. Exact build/package diff (probe-only copies)

| file | change | comment |
|---|---|---|
| `op_kernel/scatter_max_v1.cpp` | `} else { // TILING_KEY_LARGE_TAIL` → `} else if (TILING_KEY_IS(1)) { // TILING_KEY_LARGE_TAIL` | behaviour for keys 0/1 is identical; undefined keys now do nothing instead of silently running the large-tail kernel. No algorithm, no tiling, no semantic merge. Tool: `tools/apply_largetail_entry_fix.py` |
| `op_kernel/CMakeLists.txt` | `add_ops_compile_options(ScatterMaxV1 OPTIONS --tiling_key=0,1)` | declares the key list to the AscendC build (no `COMPUTE_UNIT`, see §2.3) |

(A `KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY)` probe was also tried and is *not* required —
the explicit `TILING_KEY_IS(1)` branch is the deciding factor.)

## 5. Kernel json `_0` / `_1` evidence

```
before (delivered package):
  "kernelList": [{"kernelName": "ScatterMaxV1_7d55…_0"}]          supportInfo.tilingKey: <absent>
  .ascend.meta.ScatterMaxV1_7d55…_0

after (repaired package, /root/zyg/build/stage3c_opp_fixed):
  "kernelList": [{"kernelName": "ScatterMaxV1_7d55…_0"}, {"kernelName": "ScatterMaxV1_7d55…_1"}]
  "supportInfo": { …, "tilingKey": ["0","1"] }
  .ascend.meta.ScatterMaxV1_7d55…_0   and   .ascend.meta.ScatterMaxV1_7d55…_1
```

Package: `/root/zyg/build/stage3c_opp_fixed` · build log `/root/zyg/logs/stage3c/fix5_build.log`

## 6. LT0–LT6 results (one process per case, stop on first failure)

| id | shape | tiling | result |
|---|---|---|---|
| LT0 control | N=40 F=48824 S=8 | SMALL_TAIL(0) | **PASS** (correctness, tol 0) |
| LT1 first largeTail | N=40 F=48825 S=8 | LARGE_TAIL(1) | **FAIL — device exception 507035** |
| LT2..LT6 | (N=40 F=48826 / N=41 F=48825 / N=80 / N=81 / N=320) | — | **not attempted** (series stops at the first failure, per protocol) |
| LTA1 extra probe | N=40 F=48832 S=8 (32 B aligned) | LARGE_TAIL(1) | **FAIL — 507035** |
| LTA2 extra probe | N=40 F=48960 S=8 (32 B/512 B aligned) | LARGE_TAIL(1) | **FAIL — 507035** |

Logs: `/root/zyg/logs/stage3c/{LT0_sanity,LT1_first_largetail_N40_F48825_S8,LTA1_N40_F48832,LTA2_N40_F48960}.log`,
summary `/root/zyg/logs/stage3c/largetail_matrix_summary.txt`.

## 7. Actual runtime tiling evidence

```
LT1  N=40  F=48825  idxNumPerCore=1 idxBatchNum=1 tailBatchNum=0 srcBatchNum=48824
                    ubSize=195328 remainUbSize=195296 tailSizeAlign=195328  TILING_KEY=LARGE_TAIL(1)
LTA1 N=40  F=48832  … tailSizeAlign=195328  TILING_KEY=LARGE_TAIL(1)
LTA2 N=40  F=48960  … tailSizeAlign=195840  TILING_KEY=LARGE_TAIL(1)
LT0  N=40  F=48824  … tailBatchNum=1         TILING_KEY=SMALL_TAIL(0)
```

So the host tiling selects `LARGE_TAIL` exactly as predicted, and after the repair the device task
is actually created — but it faults.

## 8. largeTail correctness

**Not established.** The kernel faults before producing any output, so there is no correctness
claim for the large-tail path (neither positive nor "silently wrong"). All largeTail shapes tested
(3 different F, two of them 32 B-aligned) fault deterministically.

## 9. DataCopy / OOB evidence

```
ACL error      : 507035 = ACL_ERROR_RT_VECTOR_CORE_EXCEPTION   (acl/error_codes/rt_error_codes.h:117)
plog (RUNTIME) : ProcLogicCqReport:Task run failed, device_id=0, stream_id=47, task_id=0,
                 sqe_type=0(ffts), errType=0x1(task exception), sqSwStatus=0x20000
                 PrintCoreInfo: exception of aivec error, core id 0/8/11/34/37, error code = 0x800000
                 errorStr: The DDR address of the MTE instruction is out of range
                 (fixp_error0 0x600006a, subErrType 4)
```

* no `507011`, no hang, no silent data corruption; the device stayed healthy (an LT0 control run
  immediately after the first fault passed)
* the faulting code path is the large-tail `elemWiseBatchProcess`:
  `DataCopy(_srcLocal, _srcGM[srcOffset], AlignUp(srcLoadNum, 8))` with
  `srcBatchNum = 48824` elements (= 195 296 B) while the tiling's UB budget after
  `UB_PRESERVED` is 195 328 B (of a 196 352 B UB) — i.e. the copy consumes the whole budget with
  no margin; alignment of `F` is **not** the trigger (LTA1/LTA2 are aligned and still fault)
* a kernel/tiling-side fix (shrink `srcBatchNum`, e.g. reserve more UB margin) is required; this is
  a **kernel change** and was deliberately not attempted in Stage 3C

## 10. Profiler

* P1 (N=40 F=48825) with the repaired package: the pre-kernel device ops complete
  (`PadV3` MIX_AIV, `MemSet`/`Fill` AI_VECTOR_CORE, `ReduceMin` MIX_AIV), then the process dies with
  `507035`; **no completed `ScatterMaxV1` task row** — the task exists only as the failed task in
  the plog (`Task run failed … task exception`)
* the SMALL_TAIL control profile (P0 equivalent) is the Stage 3B C3 case: `ScatterMaxV1`
  AI_VECTOR_CORE, `HOST_CPU_OP_FALLBACK = NONE`, `AI_CPU = 0`
* raw: `/root/zyg/profiler/stage3c/P1_largeTail/PROF_000001_20260918061015968_*/`,
  log `/root/zyg/logs/stage3c/msprof_P1.log`

## 11. PyG largeTail end-to-end

Not runnable: the PyG path calls the same adapter, and the adapter→kernel call for a largeTail
shape triggers the same device exception. Direct adapter probe
(`/root/zyg/logs/stage3c/adapter_largetail_failure_mode.log`):

```
F=48824: OK padded=False F_kernel=48824 correctness=True
F=48825: RAISED RuntimeError: … AclrtSynchronizeStreamWithTimeout(copy_stream), error code is 507035
```

This asynchronous device-exception failure mode is exactly why a **real adapter guard** (not the
underlying error code) is needed before large F is exposed to users — a follow-up decision, not
implemented here.

## 12. Stage 3B regression

`run_stage3b_boundary_tests.py` → **45/45 PASS, 0 FAIL, 0 SKIP** (`stage3b_regression.log`).

## 13. Stage 6 regression

* demo → **PASS** (`stage6_demo_regression.log`)
* 20-case suite → **20/20 PASS**, `BEFORE_HOST_CPU_FALLBACK=YES`, `AFTER=NO`
  (`stage6_tests_regression.log`)

## 14. 507011 / MTE / AIV / fallback summary

| item | result |
|---|---|
| 507011 | **not observed** |
| ACL 507035 (vector core exception) | **observed** for all 3 largeTail probes |
| MTE DDR address out of range | **observed** (device-side extend info) |
| AI core hang / timeout | none |
| Host CPU op fallback | **NONE** (no fallback warnings during any Stage 3C run) |
| AI_CPU tasks | 0 (Stage 3B/6 profiles) |

## 15. Git

* commits: `5816ef6` (frozen) → `1a8df65` (Stage 3B) → Stage 3C commit (this report + tools)
* files: `pyg-ascend-compat/global_max_pool/stage3c_large_tail_entry_repair.md`,
  `global_max_pool/stage3c/{README.md,tools/apply_largetail_entry_fix.py,
  tools/patch_kernel_task_type.py,tools/run_stage3c_largetail_matrix.sh}` and a one-paragraph
  wording correction in `stage3b_large_tail_boundary.md`
* working tree clean, **not pushed** (awaiting human review); no binaries / profiler raw / build
  trees / big logs committed

## 16. Backup

`/data/wio/zyg-src/backup/nanwang-<NEW_HEAD_SHORT>.bundle` created with
`git bundle create … --all` and verified (`The bundle records a complete history`); the frozen
`nanwang-5816ef6.bundle` and `nanwang-1a8df65.bundle` are preserved untouched.

## 17. Remaining risks

1. **largeTail is not usable in this build**: even with a correct kernel entry the path faults
   (`507035` / MTE DDR out-of-range) for every tested F (aligned and non-aligned).
2. The failure surfaces **asynchronously** (at the next sync), so an adapter guard (threshold per N,
   from the Stage 3B formula) is required before exposing large F to users.
3. The threshold is N-dependent: `F ≤ 44736` (any N) / `F ≤ 48824` (N ≤ 320) stay on the validated
   SMALL_TAIL path.
4. A kernel/tiling-side UB-margin fix (e.g. smaller `srcBatchNum` in the large-tail branch) is the
   prerequisite for enabling largeTail at all; that is a kernel change and needs its own stage.

## 18. Verdict

**STAGE3C = PARTIAL**

* the packaging/entry repair is **verified**: `kernelList` now contains `_0` **and** `_1`
  (plus `supportInfo.tilingKey = ["0","1"]`), and the `361001 BinaryGetFunctionByEntry(funcEntry=1)`
  failure is gone
* the largeTail kernel path still does **not** execute correctly: real device task, immediate
  `507035` vector-core exception with an MTE DDR out-of-range report
* therefore the Stage 3C PASS gate is not met (needs: largeTail real device task executes **and**
  correctness passes **and** no 507011/MTE/AIV)
* the frozen FP32 forward delivery (SMALL_TAIL envelope) remains fully green: Stage 3B 45/45,
  Stage 6 demo PASS, Stage 6 20/20 PASS, no host fallback
