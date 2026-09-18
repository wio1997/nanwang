# Stage 1B — ScatterMaxV1 NPU runtime bring-up (Ascend 910B3, CANN 8.5.1)

Date: 2026-09-18
Container: `wio-pyg-cann851-pyg280` (image `local/wio-pyg-cann851:torch2.9-pyg2.8.0.post1`)
Hardware: 910B3 x8 (device 0 only used), driver 26.0.rc1, CANN 8.5.1, Python 3.11.14
Scope: run the custom op compiled in Stage 1A (`ScatterMaxV1`, FP32 data + INT32 index) on the
device through the generated ACLNN API and answer whether caller-provided output / explicit size
is supported. No PyG / torch_npu / mx_driving integration in this stage.

## 1. Forward-only closure check

| Question | Answer | Evidence |
|---|---|---|
| Does the `ScatterMaxV1` runtime depend on `ScatterMaxArgmaxV1`? | **No** | `KernelScatterMaxV1` never reads/writes `argmax` GM; the two ops are separate `OpDef`s with separate kernel entries and separate ACLNN APIs |
| Can `aclnnScatterMaxV1` be executed alone? | **Yes** | every Stage 1B test called only `aclnnScatterMaxV1GetWorkspaceSize` + `aclnnScatterMaxV1`; `aclnnScatterMaxArgmaxV1` was never called, and all 6 cases produced correct results |
| Why did the Stage 1A build also emit an Argmax kernel? | Because the op_host source declares **both** `OpDef`s (`OP_ADD(ScatterMaxV1)`, `OP_ADD(ScatterMaxArgmaxV1)`) and `op_kernel/` contained both kernel entry files. The toolkit builds one kernel target per declared op (`ScatterMaxV1_ascend910b`, `ScatterMaxArgmaxV1_ascend910b`) |
| If `scatter_max_argmax_v1.cpp` is dropped from the kernel source list (copy A) | `ScatterMaxV1` still compiles (`[100%] Built target ScatterMaxV1_ascend910b`), **but** the build then fails overall: `[ERROR]: operator: scatter_max_argmax_v1 source file: .../scatter_max_argmax_v1.cpp does not found` → `binary` target fails → `binary/config` missing → `CPack`/install fails. The blocker is the still-declared `ScatterMaxArgmaxV1` **OpDef**, not a dependency of `ScatterMaxV1` |
| Dropping the argmax kernel **and** the `ScatterMaxArgmaxV1` OpDef (copy B, probe only) | build exits 0, package `custom_opp_ubuntu_aarch64.run` (292 KB) contains **only** `scatter_max_v1` kernel + config; installed to a second isolated dir (`/root/zyg/build/scattermax_fwdonly_opp`) and `aclnnScatterMaxV1` ran correctly (T1 and T4 re-run against the forward-only package: both PASS, max_abs_diff 0) |

Logs:

* `/root/zyg/logs/stage1b_forward_only_build.log` (copy A, argmax kernel file removed only)
* `/root/zyg/logs/stage1b_forward_only_build2.log` (copy B, argmax kernel + OpDef removed) — exit 0
* `/root/zyg/logs/stage1b_forward_only_runtime.log` (T1/T4 against the forward-only package)

**FORWARD_ONLY_INDEPENDENT = YES**

Runtime-wise `ScatterMaxV1` is fully independent of `ScatterMaxArgmaxV1`. Packaging-wise the two ops
are currently declared in the same op_host file; a trimmed forward-only package therefore requires
removing the `ScatterMaxArgmaxV1` `OpDef` block as well (a 1393-byte deletion, applied only to a
build-probe copy via `stage1b/tools/strip_argmax_opdef.py`). That is a package-shaping option for
Stage 2, not a runtime dependency.

Note for Stage 2: the generated `aclnnScatterMaxV1` signature **requires an argmax output tensor**
(`resOut` *and* `argmaxOut`), because the OpDef declares two outputs. A forward-only caller must
still pass an argmax buffer (values unused).

## 2. Runtime deployment (isolated)

* Custom OPP installed with the Stage 1A package into our own directory, not the system CANN:

```
$ /root/zyg/build/scattermax_probe/build_out/custom_opp_ubuntu_aarch64.run \
      --quiet --install-path=/root/zyg/build/scattermax_runtime_opp
$ ls /usr/local/Ascend/cann-8.5.1/opp/vendors      # -> empty, system CANN untouched
```

* Layout: `/root/zyg/build/scattermax_runtime_opp/vendors/customize/{bin,op_api,op_proto,op_impl,framework}`
* Env used (test shell only, exactly what the vendor `bin/set_env.bash` sets):

```
ASCEND_CUSTOM_OPP_PATH=/root/zyg/build/scattermax_runtime_opp/vendors/customize
LD_LIBRARY_PATH=/root/zyg/build/scattermax_runtime_opp/vendors/customize/op_api/lib:$LD_LIBRARY_PATH
```

* `/root/.bashrc`, `/etc/profile` and the global `PYTHONPATH` were **not** modified.
* Recorded environment: `/root/zyg/logs/stage1b_runtime_env.txt`
* `mx_driving` / DrivingSDK were **not** installed or built; only the Stage 1A package was reused.

## 3. Runtime caller

Method: minimal C++ ACLNN runner (`stage1b/runner/scattermaxv1_runner.cpp`), linked against the
isolated `libcust_opapi.so`, single device (`aclrtSetDevice(0)`), no torch/torch_npu/PyG.

Generated API (used verbatim, taken from the generated header):

```c
aclnnStatus aclnnScatterMaxV1GetWorkspaceSize(
    const aclTensor *src, const aclTensor *index,
    const aclTensor *resOut, const aclTensor *argmaxOut,
    uint64_t *workspaceSize, aclOpExecutor **executor);
aclnnStatus aclnnScatterMaxV1(
    void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream);
```

* tensors: `src` FP32 `[N,F]`, `index` INT32 `[N]`, `resOut` FP32 `[S,F]`, `argmaxOut` INT32 `[S,F]`
* `resOut` is caller-provided and pre-filled with `-inf` on the host (raw-op semantics)
* workspace: **`workspaceSize == 0` for every case** (tiling sets `workspace[0]=0`), so no workspace
  allocation was needed — the runner still handles the non-zero case
* CPU golden implemented in the runner: `out[S][F] = -inf; for n: out[index[n]][f] = max(out[index[n]][f], src[n][f])`

## 4. Test results (all on device 0, tolerance 0.0)

| case | shape | index | notes | result |
|---|---|---|---|---|
| T1 aligned baseline | N=8, F=8, size=4 | `0,1,0,2,1,2,0,3` | duplicates + positives + negatives | **PASS** max_abs_diff=0 |
| T1b aligned baseline F=16 | N=32, F=16, size=8 | sparse 8-group pattern | larger baseline | **PASS** max_abs_diff=0 |
| T2 repeated index | N=16, F=8, size=1 | 16x `0` | atomic-max reduction path | **PASS** max_abs_diff=0 |
| T3 negative-only | N=8, F=8, size=2 | `0,0,0,0,1,1,1,1` | all values strictly negative | **PASS** (result stays negative, never 0) |
| T4 explicit size | N=4, F=8, **size=5** | `0,2,0,1` (max+1 = 3) | caller-provided larger output | **PASS**; groups 3,4 keep `-inf` |
| T5 sparse groups | N=5, F=8, size=8 | `1,1,5,2,5` | only groups 1,2,5 written | **PASS**; 5 untouched rows keep `-inf` |

Key numbers:

* T1: `expected_row0 = -2.5,10,13,-2,12.25,15.25,0.25,14.5` == `actual_row0`
* T2: max over 16 rows is exactly `15,15.125,...,15.875` (the per-feature maxima) — atomic max works
* T3: `actual_row0 = -1,-1.125,-1.25,-1.375,-1.5,-1.625,-1.75,-1.875` — negative-only group never
  became 0, so `-inf` initialisation + atomic max is honoured by the raw op
* T4: requested `size=5`, `max(index)+1=3`, actual output shape `[5,8]`, `empty_rows=2`,
  `untouched_row_violations=0`

Raw per-case dumps (src/index/expected/actual): `/root/zyg/global_max_pool/stage1b/results/*.txt`
Logs: `/root/zyg/logs/stage1b_t*.log`, summary `/root/zyg/logs/stage1b_test_summary.txt`

## 5. Runtime AI Core evidence (msprof, dev 0)

```
msprof --application="scattermaxv1_runner --case t1 ..." --output=/root/zyg/profiler/stage1b_scattermax \
       --ascendcl=on --runtime-api=on --task-time=on --ai-core=on --model-execution=on
```

| field | value |
|---|---|
| Op Name / OP Type | `ScatterMaxV1` / `ScatterMaxV1` |
| Task Type / Core Type | **`AI_VECTOR_CORE`** (AIV) |
| Device | 0 |
| Block Dim | 40 (AIV blocks; tiling used `PlatformAscendC::GetCoreNumAiv()`) |
| Task Duration | 13.380 us (aiv_time 9.574 us, aiv_mte2 1.503 us, aiv_mte3 0.329 us) |
| Input/Output | `8,8;8` FLOAT;INT32 → `4,8;4,8` FLOAT;INT32 |
| aclnn trace | `aclnnScatterMaxV1` x1, `aclnnScatterMaxV1GetWorkspaceSize` x1, `aclnnScatterMaxV1Tiling` x1 |

* Profiler raw: `/root/zyg/profiler/stage1b_scattermax/PROF_000001_20260918021408380_*/`
* Readable summary: `/root/zyg/logs/stage1b_profiler_summary.txt`
* Plog (device run logs, 2026-09-18): `kernel execute failed = 0`, no op fallback entries
* **HOST_FALLBACK_OBSERVED = NO** (the only device task of the run is an `AI_VECTOR_CORE` task with
  non-zero AIV pipe times; no CPU/AICPU task exists for this op). This is runtime evidence, in
  contrast to the Stage 1A compile-time kernel json.

## 6. Errors

NONE. No `507011`, no MTE OOB, no AICORE/AIV exception, no ACLNN/GE error, no "custom op not found",
no workspace error, no shape error. (The legacy `ge.ScatterMax` 507011 artifacts from 2026-09-16 are
unrelated and were not touched.)

## 7. Stage 1B verdict

| aspect | verdict |
|---|---|
| compile | PASS (Stage 1A, reused unchanged) |
| runtime | **PASS** — op loads through the isolated custom OPP and executes |
| correctness | **PASS** — T1/T1b/T2/T3/T4/T5, bit-exact vs CPU golden, tol 0.0 |
| AI Core | **PASS** — `AI_VECTOR_CORE` task on device 0 with AIV pipe activity |
| explicit size | **EXPLICIT_SIZE_SUPPORTED = YES** |

**SCATTERMAXV1_RUNTIME = PASS**

Explicit size detail: passing a caller-allocated `[S,F]` output tensor with `S > max(index)+1` works;
the op neither rejects the shape, nor overrides it, nor truncates it. Rows not referenced by any
index keep their caller-provided `-inf`. This directly enables `global_max_pool(x, batch, size)`.

## 8. Stage 2 inputs / open items (not blockers)

1. The generated ACLNN API needs an `argmaxOut` tensor even for forward-only use (OpDef has 2 outputs).
   For PyG this means an extra `[size, F]` int32 scratch buffer, unless the OpDef is trimmed later.
2. First-call overhead is significant (`aclnnScatterMaxV1GetWorkspaceSize` ~1.1 s cold, includes
   `BinaryLoadFromData`/`BinaryGetFunctionByEntry`); steady state kernel time is ~13 us. Not optimised
   here (out of scope), but relevant for benchmarking.
3. `max(index)+1` / explicit `size` must be computed by the caller (PyG adapter) — confirmed feasible.
4. Semantics not yet covered by Stage 1B (deliberately deferred): non-32B-aligned feature size,
   argmax values, backward, fp16/bf16, multi-NPU, large-N tiling edge cases.

## 9. Reproduction

```bash
# inside container wio-pyg-cann851-pyg280
bash /root/zyg/global_max_pool/stage1b/runner/build_runner.sh
bash /root/zyg/global_max_pool/stage1b/tests/run_stage1b_tests.sh   # T1..T5
# profiler
source /root/zyg/build/scattermax_runtime_opp/vendors/customize/bin/set_env.bash
msprof --application="/root/zyg/global_max_pool/stage1b/runner/scattermaxv1_runner \
       --case t1 --n 8 --f 8 --size 4 --index 0,1,0,2,1,2,0,3 --pattern baseline --tol 0" \
       --output=/root/zyg/profiler/stage1b_scattermax --ai-core=on --ascendcl=on --task-time=on
```

Not committed to git: DrivingSDK repo, `.o`, `.so`, `custom_opp_*.run`, profiler raw data,
dump files, build cache.
