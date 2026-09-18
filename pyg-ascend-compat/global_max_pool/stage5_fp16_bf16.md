# Stage 5 — FP16 / BF16 forward + first-order backward (evidence)

Date: 2026-09-18 · container `wio-pyg-cann851-pyg280` · CANN 8.5.1 · torch 2.9.0+cpu ·
torch_npu 2.9.0 · PyG 2.8.0.post1 · Ascend 910B3 · formal delivery OPP.

History: `… → d506208 → af17d6f → 692ccd2 → 2d8e2c1 → ede301f → a586e48 → <Stage 5 commits>`
(`a586e48` was pushed to `origin/feat/global-max-pool-scattermax-zyg` as the Stage 4 freeze before
any Stage 5 edit).

**Verdict: FP16 = PASS · BF16 = PASS · STAGE5 = PASS.**

---

## 1. Capability audit (no assumptions)

**CPU oracle (real PyG/PyTorch, CPU):** both dtypes fully supported, 26/26 cases each, forward and
backward; `out.dtype == x.dtype` and `x.grad.dtype == x.dtype`. Contract and arithmetic in
`stage5_cpu_dtype_oracle.md`.

**Delivered `ScatterMaxV1` OPP is FP32-only — hard evidence:**

* op definition (`op_host/scatter_max_v1.cpp`): `src`/`res` declared `DataType({ge::DT_FLOAT})`
  (index `DT_INT32`) — there is no fp16/bf16 kernel variant in the package;
* packaged op-info: `input0.dtype = "float32"`;
* direct op-API probe (`tools/dtype_probe.cpp`, host tiling only):

```
DTYPE_PROBE fp32(control) status=0
DTYPE_PROBE fp16         status=161002     (= ACLNN_ERR_PARAM_INVALID)
DTYPE_PROBE bf16         status=161002
```

* the bridge also rejects non-fp32 tensors (`TORCH_CHECK(src.scalar_type() == at::kFloat)`).

**Backward/forward primitives on NPU (fp16 and bf16):** cast up/down, `index_select`, `==`,
bool→dtype cast, `index_add`, dtype division, multiply, `masked_fill(nan)`, `zeros_like`,
`sum` — all device-native, dtype-preserving, **bit-exact vs CPU**; the dtype division matched CPU
over 4096 random pairs (max ULP 0). Zero host-fallback warnings, zero AI_CPU tasks
(`logs/stage5/npu_primitive_dtype_audit.log`).

## 2. Implementation (cast-based device path)

Because the OPP has no fp16/bf16 kernel, the Stage 5 path is the audited device cast chain:

```
x (fp16/bf16) --Cast--> fp32 --frozen ScatterMaxV1--> fp32 --Cast--> out (fp16/bf16)
```

* new module `global_max_pool/stage5/python/global_max_pool_ascend_dtype.py`:
  * `forward_dtype()` calls the **frozen Stage 2/3A adapter** (`global_max_pool_ascend`) — no forward
    re-implementation, all Stage 3 guards inherited;
  * `GlobalMaxPoolAscendDtypeFunction` (autograd) with the measured dtype contract backward;
  * `global_max_pool_ascend_dtype()` public entry (forward-only path when `requires_grad=False`).
* `pyg_ascend_compat` now dispatches fp16/bf16 (NPU, 2-D, valid int64 batch) to the Stage 5 entry
  (autograd when `requires_grad=True` and grad enabled, otherwise the plain forward); the fp32
  branch and the `batch=None` delegation are unchanged. New counters
  `dtype16_forward_calls`, `dtype16_autograd_calls`.
* FP32 forward/backward code is untouched (`git diff` shows only the dispatch additions).

Numerical justification for the cast path (required by the task): the max is exact under it —
fp16/bf16 values are exactly representable in fp32, and the reduced value cast back is the same
dtype value; backward counts are accumulated in fp32 (exact) and rounded once to the input dtype,
matching the measured CPU denominator rule.

## 3. Forward + backward semantics (62/62 PASS, 31/31 bit-exact per dtype)

`tests/run_stage5_dtype_tests.py` — each dtype compared element-wise against the same-dtype CPU
oracle (NaN-aware, ±0-aware), with ULP reporting and gradient invariants:

| group | fp16 | bf16 |
|---|---|---|
| unique max / 2-way tie / 3-way tie / per-feature ties | PASS (bit-exact) | PASS (bit-exact) |
| weighted upstream (+ / − / zero) | PASS | PASS |
| multiple groups, explicit size + empty groups | PASS | PASS |
| negative-only, true `-inf`, `+inf` ties, `±0` (+ zero-max quirk) | PASS | PASS |
| NaN (single, double, NaN+inf, zero upstream) | PASS | PASS |
| F = 1, 7, 8, 9, 17, 33 (pad/crop forward **and** backward) | PASS | PASS |
| N = 39, 40, 41 (leftSrc), 4097 | PASS | PASS |
| largeTail N = 40 F = 48825, N = 41 F = 48825 (backward included) | PASS | PASS |

```
fp16: TOTAL 31 PASS 31 FAIL 0 | gradient bit-exact 31/31 | max ULP 0
bf16: TOTAL 31 PASS 31 FAIL 0 | gradient bit-exact 31/31 | max ULP 0
TOTAL 62  PASS 62  FAIL 0
```

Invariants verified per case: non-winners exactly `0`; winner-sum equals `grad_out` (or
`grad_out·n/(n+1)` for zero-max cells) within one dtype epsilon; empty groups contribute nothing.

## 4. Real PyG API end-to-end (`tests/run_stage5_pyg_e2e.py`, 25/25 PASS)

Both dtypes × {unique, tie2, tie3, per-feature, multi-group, empty+explicit, ±0, `-inf`, NaN,
N=41 F=33 leftSrc, largeTail N=40 F=48825} with `requires_grad=True`, plus a `requires_grad=False`
check for each dtype: forward **and** `x.grad` equal the CPU oracle, `out.dtype`/`grad.dtype` match
the input dtype, fallback NONE.

```
compat counters: total=24 ascend_calls=24 original_calls=0 dtype16_autograd_calls=22
                 dtype16_forward_calls=2 non_fp32_passthrough=0
```

## 5. Profiler (forward + loss + backward, 8/8 cases)

| case | shape | ScatterMaxV1 | AI_CPU | scatter_reduce |
|---|---|---|---|---|
| fp16/bf16 unique aligned | N=64 F=8 S=4 | AI_VECTOR_CORE | 0 | not called |
| fp16/bf16 tie + N=41 | N=41 F=33 S=8 | AI_VECTOR_CORE | 0 | not called |
| fp16/bf16 non-aligned | N=64 F=17 S=4 | AI_VECTOR_CORE | 0 | not called |
| fp16/bf16 largeTail | N=40 F=48825 S=8 | AI_VECTOR_CORE | 0 | not called |

Backward primitives seen on device: `Cast`, `Equal`, `GatherV3`, `InplaceIndexAdd`, `RealDiv`,
`Mul`, `MaskedFill`, `Fill`/`ZerosLike`/`BroadcastTo`/`Add` (all `AI_VECTOR_CORE`). The app banner
records `out`/`grad` shapes and dtypes plus the gradient sum, proving the backward ran
(`logs/stage5/stage5_profiler_summary.txt`, raw under `/root/zyg/profiler/stage5/`).

## 6. FP32 regression (hard gate) — unchanged

| suite | result |
|---|---|
| Stage 3A | **34/34** |
| Stage 3B | **45/45** |
| Stage 3D | **9/9** |
| Stage 3E largeTail | **13/13** |
| Stage 6 demo | **PASS** |
| Stage 6 matrix | **20/20** (`BEFORE=YES AFTER=NO`) |
| Stage 2 adapter suite | **16/16** |
| Stage 4 backward matrix | **35/35**, still 31/32 bit-exact, max ULP **1** (no regression) |
| Stage 4 real PyG E2E | **11/11** |
| Stage 4 profiler sanity | 4/4 cases PASS |

## 7. `batch=None`

Unchanged and native for both dtypes: PyG uses `x.max`, the wrapper delegates, CPU == NPU for
forward and backward including ties/`-inf`/1-D, no fallback
(`logs/stage5/batch_none_dtype_probe.log`).

## 8. Numerical mismatches / ULP analysis

* FP16 and BF16 matrices: **no mismatches at all** — 31/31 gradient cases bit-exact per dtype,
  forward bit-exact everywhere, NaN/Inf patterns equal.
* The only precision-sensitive step (tie denominator) was pinned by measurement: the count is the
  exact integer rounded to the input dtype (fp16 21/21, bf16 22/22 against the alternative model),
  and the dtype division is bit-exact against CPU (4096 random pairs, max ULP 0).
* FP32 keeps its previously documented ≤1 ULP tie-divide property (unchanged, not widened).

## 9. Scope limits / remaining risks

* first-order only; `create_graph=True` / gradgrad unsupported (as in Stage 4);
* the fp16/bf16 path is a **device cast chain** around the fp32-only custom op; it is device-native
  and bit-exact against the dtype CPU oracle, but it is not a native fp16/bf16 kernel (no such
  kernel exists in the delivered OPP — hard evidence in §1);
* `batch=None` remains the original PyG `x.max` path;
* guards unchanged (`index < 491520`, `N*(F+1) < 4,026,531,840`); extreme `N ≥ 163800` + largeTail
  runtime still not exercised;
* the Stage 3E source-row 32 B read hardening item remains DEFER.

## 10. Files

```
global_max_pool/stage5/
  python/global_max_pool_ascend_dtype.py    NEW  cast-based forward + dtype backward + autograd
  tests/cpu_dtype_oracle.py                 FP16/BF16 CPU oracle (26 cases each)
  tests/cpu_dtype_arithmetic_probe.py       count/division discrimination
  tests/cpu_dtype_count_model.py            M_round vs M_inc (21/21, 22/22)
  tests/cpu_dtype_div_search.py             exhaustive discriminating-pair search
  tests/cpu_dtype_div_decide.py             decisive oracle measurement
  tests/batch_none_dtype_probe.py           batch=None CPU/NPU for fp16/bf16
  tests/npu_primitive_dtype_audit.py        primitive fallback/ULP audit
  tests/run_stage5_dtype_tests.py           62-case matrix (31 per dtype)
  tests/run_stage5_pyg_e2e.py               real PyG API E2E (25 cases)
  tests/profile_stage5_case.py              msprof app
  tools/dtype_probe.cpp                     OPP fp16/bf16 rejection probe
  tools/run_stage5_profiler.sh              P5 profiler gate (8 runs)
  tools/run_stage5_regressions.sh           full regression sweep
global_max_pool/stage5_cpu_dtype_oracle.md  frozen FP16/BF16 contract
global_max_pool/stage5_fp16_bf16.md         this report
global_max_pool/stage6/pyg_ascend_compat/__init__.py   dispatch update
```
