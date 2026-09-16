# PyG Ascend operator feasibility attribution

## Purpose and decision rules

This document follows the initial seven-API compatibility screen with controlled root-cause experiments. It does not rank performance benefit and does not propose a custom kernel.

Final issue classes are restricted to:

- `TYPE_LIMITATION`: the supported execution path changes with a legal dtype while shape, values, attributes and call form remain fixed.
- `ADAPTER_GAP`: equivalent device capability is established, but PyTorch/torch_npu does not map the API to it correctly.
- `COMPUTE_CAPABILITY_GAP`: no reusable device-side route remains after supported capabilities have been checked.

Compatibility status `A/B/C1/C2/D/U` is separate from this root-cause classification. In particular, C1 means an AICPU path was observed and C2 means Host CPU tensor fallback was observed.

## Environment and scope

- Host: `root@61.241.77.34:60006`, aarch64 Ubuntu 22.04, kernel `5.15.0-25-generic`
- NPU: 8 × Ascend 910B3; experiments used `npu:1`
- CANN: 8.5.1
- Python: 3.11.14
- PyTorch: 2.9.0+cpu
- torch_npu: 2.9.0
- PyG: 2.8.0.post1
- Container: `wio-pyg-cann851-pyg280`
- Observed host driver: 26.0.rc1, not the originally requested 25.2.0

The follow-up work covers the integer Cumsum and stable Sort metadata paths used by global sort, TopK and SAG pooling, plus the batched scatter-max path used by `global_max_pool`.

## Current conclusions

| Problem | Controlled evidence | Current classification | Existing device capability | Decision |
|---|---|---|---|---|
| INT64 Cumsum AICPU | Same `[32]` values and `torch.cumsum(..., out=...)`: INT32 uses `aclnnCumsum_CumsumAiCore_Cumsum`; INT64 uses `aclnnCumsum_CumsumAiCpu_Cumsum`; both exact | `TYPE_LIMITATION` | Yes, conditional | Stop root-cause work. INT32 is reusable only with a proven full prefix-sum range and preserved downstream dtype semantics. |
| Stable integer Sort AICPU | Same `[1024]`, values, `stable=True`, ascending: INT32 and INT64 use `aclnnSort_SortAiCpu_Sort`; FP32 uses `aclnnSort_SortAiCore_Sort`; all values and permutations exact | `TYPE_LIMITATION` | Yes, conditional | Stop root-cause work. This is an integer-family limitation, not INT64-only. |
| `scatter_reduce.two_out` Host fallback | Both `include_self` values and both in-place/functional call forms fall back; all results remain CPU-exact | Not final; `ADAPTER_GAP` hypothesis | CANN candidate exists, reuse unproven | Continue only through a supported installed `ScatterMax` invocation/binding path. Do not infer `COMPUTE_CAPABILITY_GAP`. |

## Cumsum result

The Cumsum experiment changed only input/output dtype. Shape `[32]`, numeric values, dimension, call form, warmup and profiler setup were fixed. The INT32 arm used AI Core; the INT64 arm used AICPU. There was no Host CPU fallback and both outputs matched CPU exactly.

The result meets the `TYPE_LIMITATION` stop condition. An INT32 substitution is not universally legal: every prefix sum must fit INT32, and downstream consumers must preserve the public dtype contract.

This affects:

- the graph-count prefix sum in `global_sort_pool`/`SortAggregation`;
- graph offsets in the ratio paths of TopKPooling and SAGPooling.

Sort and Cumsum were evaluated independently; the Cumsum conclusion is not used as evidence for Sort.

## Sort result

The first Sort contrast changed only INT32 versus INT64. Both legal integer arms selected AICPU, so INT64 was not the deciding factor. A second contrast kept the same shape, values, ascending order and stable mode while changing INT32 to FP32; FP32 selected AI Core and preserved the exact stable permutation.

The tested stable integer Sort family is therefore `TYPE_LIMITATION`.

For the `SelectTopK` batch-regrouping sort, FP32 AI Core Sort is conditionally reusable under all of these constraints:

1. Batch IDs are integers in the inclusive range `0..16,777,216` (`2^24`).
2. Sort is ascending with `stable=True`.
3. Only sort keys are cast to FP32.
4. The returned permutation is applied to the original INT64 batch.
5. Inputs outside the contract are rejected or keep the original path.

Within that range, conversion to FP32 preserves integer values, ordering and equality, so stable ordering is unchanged. This was a semantic feasibility result, not an implemented PyG modification. End-to-end acceptance would still cover repeated/interleaved IDs, the `2^24` boundary, ties, empty inputs and gradients downstream of the selected indices.

## scatter-max result

### Confirmed framework behavior

PyG 2.8.0.post1 uses:

```python
src.new_zeros(size).scatter_reduce_(
    dim, broadcast(index), src,
    reduce="amax", include_self=False,
)
```

On the tested NPU stack, runtime reports Host CPU fallback for `aten::scatter_reduce.two_out`. The returned tensor is copied back to `npu:1`; its output device therefore does not disprove fallback.

`scatter_reduce.two_out` is an ATen overload, not a CANN kernel name. Controlled experiments found:

| Variable | Control | Treatment | Result |
|---|---|---|---|
| `include_self` | `False` | `True` | Both legal, exact, and Host CPU fallback |
| call form | `Tensor.scatter_reduce_` | `torch.scatter_reduce` | Both legal, exact, and Host CPU fallback |

Neither attribute explains the fallback.

### Existing device candidate

The installed CANN 8.5.1 package contains an Ascend 910B AI Core `ScatterMax` implementation. Its installed configuration includes float32 data with both INT32 and INT64 indices, and its declared shape contract supports:

- `var`: `[32,64]` float32;
- `indices`: `[4096]` INT32 or INT64;
- `updates`: `[4096,64]` float32, equal to `indices.shape + var.shape[1:]`.

TorchAir exposes a generated raw GE `ScatterMax` op. In contrast, the relevant `scatter_reduce.two`, `scatter_reduce.two_out` and `scatter_reduce_.two` converters raise `NotImplementedError`. This is concrete evidence of missing framework mapping, but it is not by itself proof that the candidate is a semantics-complete replacement.

### Reuse and validation boundary

Direct raw-op probes using the declared legal shapes reached `origin_op_name [ScatterMax]` on the device. Both INT32 and INT64 index variants then failed with runtime code `507011` and an AI Vector MTE address-out-of-range error. This excluded index dtype as the deciding factor for that raw-probe failure.

Because installed `ScatterMax` describes `var` as rewritten state, a second probe declared mutation/alias semantics in its temporary Torch schema. PyTorch AOTAutograd functionalization rejected custom non-ATen alias outputs before device execution. It therefore did not realize a different device binding and supplies no device-capability conclusion.

For forward semantics, a possible device composition is:

1. initialize the reduction buffer to negative infinity;
2. apply slice `ScatterMax` for repeated graph indices;
3. build an independent per-group occupancy mask on device;
4. select reduced values for occupied groups and zero for empty groups.

The mask is necessary: replacing every negative-infinity output with zero would confuse an empty group with a non-empty group whose maximum is negative infinity. Exact validation must also cover negative-only groups, repeated indices, NaN/Inf and signed zero. Training support additionally requires matching PyTorch tie-gradient, self-gradient and alias behavior.

The current evidence does not meet the final gate for either `ADAPTER_GAP` or `COMPUTE_CAPABILITY_GAP`:

- `ADAPTER_GAP` remains the leading hypothesis because a matching AI Core candidate exists and framework converters are missing.
- It is not final because a supported invocation/binding path has not yet produced correct device output.
- `COMPUTE_CAPABILITY_GAP` is not justified because the installed device primitive and binaries have not been ruled out.

The next minimal step is to identify an installed, supported CANN or framework entry point that realizes `ScatterMax` rewritten-variable binding, then run only the fixed legal INT32 testcase through that path. No custom kernel work is justified at this stage.

## Engineering consequence

- Do not develop a new Cumsum kernel: the existing INT32 AI Core route is conditionally reusable.
- Do not develop a new Sort kernel from the current evidence: the existing FP32 stable Sort route is conditionally reusable for bounded graph IDs.
- Do not develop a new scatter-max kernel yet: first close the supported `ScatterMax` invocation and semantic-reuse gate. Missing ATen mapping and missing validation are not proof that the device lacks the computation.

These statements are feasibility conclusions, not performance-priority rankings.
