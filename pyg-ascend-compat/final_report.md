# PyG 2.8.0.post1 on Ascend 910B3 — compatibility revalidation

## Conclusion

The isolated PyG 2.8.0.post1 process stack was built successfully and all seven requested API screens passed their CPU-reference checks. The recorded per-case correctness results are byte-for-byte identical at the JSON `cases` level to the earlier PyG 2.6.1 run. Execution-path classifications also remain unchanged:

**A / A / A / C2 / C1 / C1 / C1** for GraphNorm, global_mean_pool, global_add_pool, global_max_pool, global_sort_pool, TopKPooling and SAGPooling respectively.

This does not constitute a fully exact match to the requested environment table: the host driver is **26.0.rc1**, not 25.2.0. The process-level Python/CANN/PyTorch/torch_npu/PyG versions do match.

Accuracy thresholds are `rtol=1e-4, atol=1e-5` for fp32 and `rtol=atol=0.02` for fp16/bf16. Low-precision NPU results are compared against fp32 CPU references.

| API | Function and CPU-reference precision | Output device | Representative execution path | CPU → NPU median | Scope / limitations | Status |
|---|---|---|---|---:|---|---:|
| GraphNorm | fp32/fp16/bf16 eager forward/backward pass; max output abs error `2.38e-7 / 1.84e-3 / 1.45e-2`; input and parameter gradients pass | `npu:1` | fp32 forward: AI_VECTOR_CORE/MIX_AIV; no AICPU or Host fallback observed | `12.151 → 1.290 ms` (`9.42x`) | Native-path claim is limited to representative fp32 forward; compile, empty graphs, extreme values and NaN/Inf inputs untested | **A** |
| global_mean_pool | Three dtype forward/backward pass; max output abs error `0 / 5.86e-4 / 1.82e-3` | `npu:1` | fp32 forward: AI_VECTOR_CORE/MIX_AIV; no AICPU or Host fallback observed | `4.217 → 0.547 ms` (`7.70x`) | Explicit batch/size only; `batch=None`, empty graphs and compile untested | **A** |
| global_add_pool | Three dtype forward/backward pass; max output abs error `0 / 1.17e-3 / 1.25e-2` | `npu:1` | fp32 forward: AI_VECTOR_CORE/MIX_AIV; no AICPU or Host fallback observed | `1.448 → 0.294 ms` (`4.92x`) | Same bounded scope as mean pool | **A** |
| global_max_pool | Three dtype forward/backward pass; max output abs error `0 / 7.81e-4 / 6.25e-3` | Returned tensor `npu:1` | Explicit `npu_cpu_fallback` for unsupported `aten::scatter_reduce.two_out` | `3.336 → 5.949 ms` (`0.56x`) | C2 is for the tested batched scatter path; `batch=None` uses another path and is untested | **C2** |
| global_sort_pool | Deprecated wrapper remains callable; three dtype forward/backward pass; max output abs error `0 / 7.81e-4 / 6.25e-3` | `npu:1` | AI_CPU Cumsum `INT64[32]` plus AI_VECTOR/MIX_AIV; Host tensor fallback not confirmed | `33.344 → 2.534 ms` (`13.16x`) | `k=4` correctness, `k=64` benchmark; ties, empty input, compile and extreme values untested | **C1** |
| TopKPooling | Three dtype forward; all six outputs match CPU; discrete outputs exact; max floating abs error `2.98e-8 / 1.03e-3 / 7.00e-3` | All returned tensors `npu:1` | AI_CPU Sort `INT64[1024]` and Cumsum `INT64[16]` plus AI_VECTOR/MIX_AIV | `25.520 → 3.709 ms` (`6.88x`) | ratio=0.5, min_score=None, eager forward; backward, ties, empty graphs, compile and min-score mode untested | **C1** |
| SAGPooling | Three dtype forward; all six outputs match CPU; discrete outputs exact; max floating abs error `5.96e-8 / 4.42e-4 / 5.61e-3` | All returned tensors `npu:1` | AI_CORE/AI_VECTOR/MIX_AIV plus the same AI_CPU Sort/Cumsum metadata path | `11.901 → 4.202 ms` (`2.83x`) | Default GraphConv(add), ratio=0.5, eager forward; backward, alternate GNN/aggr, compile and min-score mode untested | **C1** |

## Environment

- NPU: 8 × Ascend 910B3; validation used `npu:1`
- Host: aarch64, Ubuntu 22.04 LTS, kernel `5.15.0-25-generic`
- Host driver: **26.0.rc1** — mismatch against the requested 25.2.0 row
- CANN: 8.5.1
- Python: 3.11.14
- PyTorch: `2.9.0+cpu`
- torch_npu: 2.9.0
- PyG: 2.8.0.post1
- Image: `local/wio-pyg-cann851:torch2.9-pyg2.8.0.post1`
- Container: `wio-pyg-cann851-pyg280`

Eight NPUs were visible, a synchronized fp32 matmul on `npu:1` passed, and all seven API symbols imported. Existing PyG 2.6.1 and service containers were not modified.

## Cross-version consistency

- For every API, the complete recorded `cases` array is identical between PyG 2.6.1 and 2.8.0.post1. This covers the tested shapes, dtypes, errors, gradients, devices and discrete-output comparisons only; it is not a claim about all PyG semantics.
- `global_sort_pool` is still present in PyG 2.8.0.post1, but remains deprecated in favor of the aggregation API.
- The previously identified execution mechanisms persist: global max Host fallback; sort's integer Cumsum AICPU; TopK/SAG integer Sort and Cumsum AICPU.
- Performance values come from separate wall-clock runs. CPU medians varied substantially, so differences from the PyG 2.6.1 run are not attributed to the PyG upgrade.

Timing covers synchronized fp32 forward calls with inputs already resident on the target device, 10 warmups and 50 measured iterations. It excludes input transfer and CPU-reference comparison.

## Follow-up feasibility attribution

The A/C1/C2 values above are compatibility-screening statuses, not root-cause categories. Controlled follow-up experiments produced these bounded conclusions:

- Cumsum is `TYPE_LIMITATION`: identical legal INT32 and INT64 calls select AI Core and AICPU respectively. The INT32 path is reusable only when the complete prefix-sum range fits INT32 and downstream dtype semantics are preserved.
- Stable Sort for the tested integer family is `TYPE_LIMITATION`: INT32 and INT64 both select AICPU, while an otherwise identical FP32 stable Sort selects AI Core. For non-negative graph IDs no greater than `2^24`, sorting FP32 keys and applying the returned permutation to the original INT64 batch is mathematically equivalent under the documented contract. This was not integrated into PyG.
- The batched max path remains an unresolved attribution with an `ADAPTER_GAP` hypothesis. Both `include_self` values and both PyTorch call forms use the same Host CPU fallback. CANN contains an Ascend 910B AI Core `ScatterMax` candidate for float32 data with INT32/INT64 indices, while the relevant torch_npu/TorchAir ATen converters are unimplemented. Direct raw-op probes reached `ScatterMax` device execution but failed with an MTE address error for both index dtypes; an alias-aware test was rejected by frontend functionalization before device execution. These probes do not justify either a final adapter-gap claim or a compute-capability-gap claim.

See `feasibility_attribution.md` for the experiment matrix, reuse constraints and stopping conditions. No custom operator or production adapter was implemented.

## Evidence

- `environment_gate.md`
- `native_path_evidence.md`
- `feasibility_attribution.md`
- `graphnorm_result.json`
- `global_mean_pool_result.json`
- `global_add_pool_result.json`
- `global_max_pool_result.json`
- `global_sort_pool_result.json`
- `topk_pooling_result.json`
- `sag_pooling_result.json`
- Raw profiler trees: `wio-pyg-cann851-pyg280:/root/pyg_validation/<API>/profile_complete`
