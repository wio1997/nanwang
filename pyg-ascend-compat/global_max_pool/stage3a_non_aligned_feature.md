# Stage 3A — ScatterMaxV1 non-aligned feature contract + safe forward support

Date: 2026-09-18 · container `wio-pyg-cann851-pyg280` · CANN 8.5.1 · torch 2.9.0+cpu ·
torch_npu 2.9.0 · Ascend 910B3 (device 0) · driver 26.0.rc1
Scope: derive the real F/alignment contract of ScatterMaxV1 from source, decide whether the raw
non-aligned path is safe, and make the Stage 2 adapter accept any `F >= 0` (padding path).
No backward, no fp16/bf16, no PyG integration, no kernel rewrite.

## 1. Raw kernel alignment contract (source-derived)

Source: `DrivingSDK/kernels/scatter_max/{op_host,op_kernel}/scatter_max_v1.*`

| quantity | value | source |
|---|---|---|
| logical feature count | `F = x.shape[1]`, `tailElemNum = srcElemNum / src.shape[0]` = F | `op_host/scatter_max_v1.cpp:75` |
| element size | 4 B (FP32 only) | `srcDSize = kDataSizeMap[DT_FLOAT]` |
| 32 B align quantity | `elemNumPerBlock = BLOCK_SIZE / srcDSize = 32/4 = 8` elements | `op_host:10,62` |
| `tailSize` | `F * 4` bytes (exact logical row) | `op_host:77` |
| `tailElemNumAlign` | `AlignUp(F, 8)` elements | `op_kernel:33` |
| `tailSizeAlign` | `AlignUp(F, 8) * 4` bytes | `op_host:78` |
| GM row stride | `_tailElemNum` = **F elements** (`_srcGM[tailOffset * _tailElemNum]`, `_resGM[idxVal * _tailElemNum]`) | `op_kernel:208,215,238,264` |
| UB row spacing | `_srcLocal[n * _tailElemNumAlign]` (one 32 B-rounded row per block) | `op_kernel:215` |
| copy length (tailWise read) | `DataCopyExtParams{blockCount=tailLoadNum, blockLen=_tailSize}` → exactly `F*4` bytes/row, rows contiguous in GM (`srcStride=0`) | `op_kernel:205,208` |
| copy length (tailWise write) | `DataCopyPad(_resGM[...], ..., {1, _tailSize})` → exactly `F*4` bytes | `op_kernel:215` |
| largeTail read | `DataCopy(_srcGM[srcOffset], AlignUp(srcLoadNum, 8))` → **32 B-rounded length** | `op_kernel:230,233` |
| largeTail write | `DataCopyPad(..., {1, srcLoadNum * sizeof(DTYPE_RES)})` → exact bytes | `op_kernel:238` |
| leftSrc path | aligned `DataCopy` for index (8 int32 = 32 B) and src rows, exact `DataCopyPad` write | `op_kernel:255,256,264` |

## 2. GM address model

```
SRC_ROW_STRIDE        = F * 4 bytes        (element index: row * F)
RES_ROW_STRIDE        = F * 4 bytes        (element index: idx[n] * F)
COPY_LENGTH (read)    = F * 4 bytes per row, rows contiguous  (tailWise path)
ALIGNED_COPY_LENGTH   = AlignUp(F,8) * 4 bytes  (UB row layout)
TAIL_WRITE_LENGTH     = F * 4 bytes exactly   (both tailWise and largeTail writes)
```

Answers to the specific Stage 3A questions:

* **Can it read the next row?** In the `tailWise` path, no: the read is byte-exact (`blockLen = F*4`),
  so a row never reaches into the next one. In `elemWiseBatchProcess` / `processLeftSrc` the *aligned*
  `DataCopy` rounds the length up to 32 B, so up to 7 elements (28 B) beyond the logical row are
  fetched — for a middle row that is the next row's data (harmless, it is ignored) and for the very
  last row it can read up to 28 B past the tensor. This behaviour is **independent of F alignment**
  (it also happens with F=8) and already occurred in every Stage 1B/2 run (e.g. the index read
  `DataCopy(_idxGM[_leftSrcIdxPos], 8)` with N=8 reads up to 7 int32 past the index tensor).
* **Can it write the next row?** No. Every GM write is `DataCopyPad` with an exact byte count
  (`F*4`, or `srcLoadNum*4` in the large-tail path).
* **Padding write beyond the logical output?** Only if an index is `>= size`; the adapter rejects that
  before launching (`batch < size`).
* **"For F=7 FP32, does the kernel perform a 32 B GM transaction and thereby access the 8th element?"**
  The hardware align-copy engine may touch the enclosing 32 B block internally, but semantically the
  transfer is 28 B: the missing 4 B are materialised in the **UB pad region** (`tailElemNumAlign`
  layout), not taken from the next GM row, and the write side writes exactly 28 B. Empirically, the
  F=7 / F=9 / F=17 runtime probes compared **all** rows and features against a CPU golden and found
  zero differences — no cross-row corruption.

Decisive CANN-side evidence (this is what makes the raw path safe, not the DrivingSDK doc):

```
compiler/tikcpp/tikcfw/impl/dav_c220/kernel_operator_data_copy_impl.h
  DataCopyPadGm2UBImpl : asserts only the ***UB (dst)*** address is 32B aligned; the GM src address has no constraint
  DataCopyPadUB2GMImpl : asserts only the ***UB (src)*** address is 32B aligned; the GM dst address has no constraint
  both call the hardware align-copy intrinsics (copy_gm_to_ubuf_align / copy_ubuf_to_gm_align)
  CheckDataCopyPadParams : asserts blockLen == multiple of sizeof(T) for GM→UB (debug builds only)
```

So GM addresses that are not 32 B aligned (which is exactly what `row * F * 4` is when `F % 8 != 0`)
are an explicitly supported DataCopyPad input.

## 3. Tiling branch matrix (computed for Ascend910B3, FP32)

Platform facts used: `ub_size = 196608` B, `vector_core_cnt = 40` (from
`aarch64-linux/data/platform_config/Ascend910B3.ini`), `UB_PRESERVED = 1024`,
`MAX_BATCH_NUM = 4095`, `elemNumPerBlock = 8`.

```
ubSize        = 196608 - 1024 = 195584
idxNumPerCore = N / 40                     (integer division; 0 when N < 40)
idxBatchNum   = min(idxNumPerCore, 4095)
remainUbSize  = 195584 - AlignUp(idxBatchNum, 8) * 4
tailBatchNum  = remainUbSize / (AlignUp(F, 8) * 4)
TILING_KEY 0 (SMALL_TAIL) iff NOT(idxNumPerCore != 0 AND tailBatchNum == 0)
TILING_KEY 1 (LARGE_TAIL) iff idxNumPerCore != 0 AND tailBatchNum == 0
```

| branch | condition | meaning | F threshold (fp32, 910B3) | N threshold |
|---|---|---|---|---|
| SMALL_TAIL (key 0) | `idxNumPerCore == 0` **or** `tailBatchNum >= 1` | at least one full row fits in UB → row-wise (`DataCopyPad`, byte-exact) | F ≤ 48 888 (N 40–320), F ≤ 44 800 (N ≥ 163 800) | N < 40 always small-tail |
| LARGE_TAIL (key 1) | `idxNumPerCore != 0` **and** `tailBatchNum == 0` | a single row does not fit in UB → element-wise (`DataCopy` with 32 B-rounded length) | F ≥ 48 889 (N 40…320) … F ≥ 44 801 (N ≥ 163 800) | N ≥ 40 required |

There is no UB-size branch on N other than the `idxBatchNum` term above (max 4095 → 16 384 B of UB).
Callback for Stage 3B: the large-tail boundary is `F ∈ [44 801, 48 889]` depending on N, which is far
outside normal pooling feature dims but is a real, computable threshold.

Additional N-driven path: `idxElemNum % coreNum != 0` (i.e. `N % 40 != 0`) activates
`processLeftSrc`, which handles the remaining index rows. All Stage 3A tests exercise it
(N = 1, 8, 16 < 40).

## 4. Raw non-aligned verdict

**RAW_NON_ALIGNED_F = SAFE** (tailWise path, i.e. `F <= ~48.9K` fp32)

* length safety: every GM read/write in the tailWise path is byte-exact (`F*4`); proven from
  `op_host:77,78` + `op_kernel:205,208,215,264`
* address safety: `DataCopyPad` constrains only the UB address; the GM address is handled by the
  align-copy engine (CANN impl quoted in §2)
* result-shape safety: the kernel writes only to `idx[n] * F` rows; the adapter guarantees
  `idx < size`, so no row beyond the caller's output is touched
* residual, **F-independent** caveat: the aligned `DataCopy` reads in `processLeftSrc` /
  `elemWiseBatchProcess` can over-read up to 28 B past the end of a tensor (already exercised in
  Stage 1B/2 with aligned F); the repository's own `g_gm_overflow_check` framework exists for this

**Raw runtime probe: PASS** (`/root/zyg/logs/stage3a_raw_probe_summary.txt`,
`stage3a_raw_f{1,7,9,17}.log`) — one process per F, N=2, size=2, tolerance 0, series would have
stopped at the first failure:

| F | exit | max_abs_diff | verdict |
|---|---|---|---|
| 1 | 0 | 0 | PASS |
| 7 | 0 | 0 | PASS |
| 9 | 0 | 0 | PASS |
| 17 | 0 | 0 | PASS |

No 507011, no MTE error, no AIV exception, no memory corruption. The legacy `ge.ScatterMax`
failure mode did **not** reproduce.

## 5. Padding design (implemented in the adapter)

```
F_pad  = ceil(F / 8) * 8                       (FP32, 32 B)
x_pad  = F.pad(x, (0, F_pad - F), value=-inf)  # NPU-native
out_pad= full((S, F_pad), -inf) → ScatterMaxV1 → occupancy → masked_fill_(empty → 0)
out    = out_pad[:, :F].contiguous()           # crop + contiguous
```

* padding value `-inf` cannot change the max over the original F columns (max with `-inf` is the
  identity for any value, including NaN on this HW — see §7)
* `alignment_mode` parameter: `"auto"` (default: pad iff `F % 8 != 0`), `"raw"` (never pad),
  `"pad"` (force padding when unaligned)
* padding op probe: `F.pad(constant)`, `new_full + slice assign` and `torch.cat` are all
  fallback-free and correct; **`F.pad`** was chosen (single op)
* crop: `Slice` on device + one contiguous copy (recorded as `crop_copy_bytes`)

## 6. Correctness matrix (`stage3a/tests/run_stage3a_tests.py`, 34/34 PASS, tol 0)

| case | F | N | result |
|---|---|---|---|
| B1 | 1 | 8 | PASS |
| B2 | 7 | 8 | PASS |
| B3 | 9 | 8 | PASS |
| B4 | 17 | 16 | PASS |
| B5 | 31 | 8 | PASS |
| B6 | 33 | 16 | PASS |
| B7 non-aligned + repeated index | 7 | 16 | PASS |
| B8 non-aligned + explicit size 5 | 9 | 4 | PASS |
| B9 non-aligned + negative-only | 17 | 8 | PASS |
| B10 non-aligned + true `-inf` vs empty | 7 | 3 | PASS (group0 `-inf`, group1 `0`) |
| B11 / B12 / B13 | 2 / 3 / 15 | 1 / 8 / 16 | PASS |
| transitions | 7/8/9, 15/16/17, 31/32/33 | 8 | PASS (`padded` flag flips, values identical) |
| raw vs padded parity | 1,7,9,17,31,33 | 16 | PASS (`torch.equal`) |
| invalid `alignment_mode` | – | – | PASS (ValueError) |

Stage 2 regression re-run with the new adapter: 16/16 PASS (A9 changed from "F=7 rejected" to "F=7
supported since Stage 3A").

## 7. F=0 contract

`F = 0` is handled by a device-side fast path (no ScatterMaxV1 launch, no padding):

| case | expected | result |
|---|---|---|
| N=0, F=0, size=None | `[0, 0]` | PASS |
| N=0, F=0, size=3 | `[3, 0]` | PASS |
| N=4, F=0, size=None | `[max(batch)+1, 0]` = `[3, 0]` | PASS |
| N=4, F=0, size=5 | `[5, 0]` | PASS |
| N=4, F=0, invalid index vs size | ValueError before launch | PASS |

Index validation still runs for F=0 because it determines `S` and the `idx < size` contract.

## 8. `index < 491520` contract

**INDEX_491520_CONTRACT = DOCUMENTED_SUPPORT_LIMIT**

| item | finding |
|---|---|
| source | `DrivingSDK/docs/zh/api/context/scatter_max.md:32` — the only occurrence in the repository (searched `kernels/`, `mx_driving/`, tests, docs, host + kernel sources) |
| formula | none derivable: the V1 kernel/host never mention it; `491520 = 0x78000 = 15 * 2^15` and `4_026_531_840 = 491520 * 8192` (the two documented bounds are the same "budget" family) |
| which variable | the *value* of `indices`, not the element count |
| relation to output size | none enforced; the operational requirement is `idx < size` (else the kernel writes outside the caller's output) |
| relation to F / N | none found |
| enforcement in source | **none** in op_host, op_kernel, or the C++/Python wrapper; CANN-side hits of `491520` are unrelated (matmul tuning bank entries, one `nd2nz` comment) |
| adapter guard | `KEEP` — an index at or above the documented limit is rejected before launch. The guard is free at PyG scale and avoids depending on an unverified envelope |

## 9. `N * (M + 1) < 4,026,531,840` constraint

| item | finding |
|---|---|
| DOC_N | `updates.shape[0]` — the number of scatter sources (rows), i.e. our `N` = `index.numel()` |
| DOC_M | product of `updates.shape[1:]` — the elements per row, i.e. our `F` |
| meaning | an upper bound of `0xF0000000` (~3.75 Gi elements) on the flattened element count of an `[N, M+1]` layout: approximately `N*F + N`, i.e. the largest row-relative offset the original (legacy) implementation was willing to address |
| reason | `4_026_531_840 = 0xF0000000` sits just below `2^32`; it is the classic 32-bit offset budget with headroom |
| kernel formula | **not applicable to ScatterMaxV1**: every tiling field is `uint64_t` (`op_host/scatter_max_v1.h`), every kernel offset/length is `uint64_t` (`op_kernel/scatter_max_v1.h`), and the GM pointers are 64-bit. No 32-bit narrowing was found in the op, the tiling function, or the generated ACLNN layer |
| overflow risk | none identified in V1 for realistic sizes; the documented bound corresponds to ≈16 GB of FP32 data |
| host enforcement | none |
| adapter handling | a conservative guard was added (`N*(F+1) < 4,026,531,840`, ValueError otherwise). It never triggers at PyG scale and documents the inherited limit |

## 10. FP32 special values (`stage3a/tests/probe_special_values.py`, F=7 padded and raw)

| case (feature 0 values) | CPU `torch.max` | CPU `scatter_reduce(amax)` | Ascend raw | Ascend padded |
|---|---|---|---|---|
| `+inf, -inf` | `+inf` | `+inf` | `+inf` | `+inf` |
| `+inf, NaN` | `NaN` | `NaN` | `NaN` | `NaN` |
| `NaN, -inf` | `NaN` | `NaN` | `NaN` | `NaN` |
| `NaN, 1` | `NaN` | `NaN` | `NaN` | `NaN` |
| `+0.0, -0.0` | `+0.0` | `+0.0` | `+0.0` | `+0.0` |
| `-0.0, 1` | `1` | `1` | `1` | `1` |
| `-inf, -inf` | `-inf` | `-inf` | `-inf` | `-inf` |

* `PYTORCH_CPU_NAN_SEMANTICS = NaN propagates (NaN wins over +inf)` for both `torch.max` and
  `scatter_reduce(reduce="amax")`
* `ASCEND_SCATTERMAX_NAN_SEMANTICS = same (NaN propagates)`; signed zero also matches bitwise
* `MATCH = YES` (raw and padded identical on every case)

## 11. Runtime execution of the padded path (msprof, device 0, N=4096, F=33 → pad 40, S=64)

| step | op | core type | avg us |
|---|---|---|---|
| padding | `PadV3` | MIX_AIV | 16.1 |
| reduction | `ScatterMaxV1` | AI_VECTOR_CORE | 26.6 |
| occupancy | `ScatterElementsV2` | AI_VECTOR_CORE | 61.6 |
| empty → 0 | `MaskedFill` | AI_VECTOR_CORE | 14.1 |
| crop | `Slice` (+ contiguous copy) | AI_VECTOR_CORE | 3.8 |
| int64→int32 / init / masks | `Cast`, `Fill`, `MemSet`, `ZerosLike`, `Equal`, `BroadcastTo`, `Pack`, `LinearIndex` | AI_VECTOR_CORE | ≤ 11 |
| batch min/max | `ReduceMin`/`ReduceMax` | MIX_AIV | ~42 |

`AI_CPU` tasks: **0** · fallback warnings: **0** ·
`HOST_CPU_OP_FALLBACK = NONE OBSERVED` · `HOST_SYNC = 1` (scalar `batch` min/max)
Raw: `/root/zyg/profiler/stage3a_padded/PROF_000001_20260918025734675_*/`, summary:
`/root/zyg/logs/stage3a_profiler_summary.txt`

## 12. Performance observation (no optimisation)

N=4096, S=64, latencies include the adapter's single host sync:

| F | F_kernel | padded | ms/call | padded input B | kernel output row B | argmax B | crop B |
|---|---|---|---|---|---|---|---|
| 32 | 32 | no | 0.625 | 0 | 8 192 | 8 192 | 0 |
| 33 | 40 | yes | 0.761 | 655 360 (full padded tensor; +114 688 B extra) | 10 240 | 10 240 | 8 448 |

Extra cost of padding F=33 → 40: +114 688 B input, +1 792 B output row width, +1 792 B argmax
scratch, 8 448 B contiguous crop copy, ≈ +0.14 ms/call. Recorded only, not tuned.

## 13. Verdict

**NON_ALIGNED_FORWARD = PASS**

P1–P9 satisfied: the F/tail/alignment contract is derived from source; raw non-aligned is judged
SAFE with a source proof and a passing controlled probe; the padding path supports F=1/7/9/17/31/33
and the transition pairs; negative / `-inf` / empty semantics do not regress; F=0 is defined and
tested; both documented limits are traced to their evidence level; and the padded adapter path has
no CPU op fallback.

## 14. Reproduction

```bash
cd /root/zyg/global_max_pool/stage3a
export ASCEND_CUSTOM_OPP_PATH=/root/zyg/build/scattermax_runtime_opp/vendors/customize
export LD_LIBRARY_PATH=$ASCEND_CUSTOM_OPP_PATH/op_api/lib:$LD_LIBRARY_PATH
bash tools/raw_non_aligned_probe.sh          # controlled raw probe, one process per F
python3 tests/run_stage3a_tests.py           # 34/34 PASS
python3 tests/probe_padding_ops.py           # padding op fallback probe
python3 tests/probe_special_values.py        # +inf/-inf/NaN/±0 semantics
python3 tools/perf_observe.py                # F=32 vs F=33 observation
msprof --application="python3 tests/profile_adapter_padded.py" \
       --output=/root/zyg/profiler/stage3a_padded --ai-core=on --ascendcl=on --task-time=on
```

Not committed: `.so`, `.o`, `custom_opp_*.run`, profiler raw trees, build outputs.
