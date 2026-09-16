# PyG 2.8.0.post1 representative execution-path evidence

All profiler captures are representative fp32, no-grad forward paths. Correctness cases cover the additional dtypes and backward modes stated in the result JSON, but those modes were not separately profiled.

| API | Observed device core types | AICPU evidence | Host tensor fallback evidence | Suggested status |
|---|---|---|---|---|
| GraphNorm | AI_VECTOR_CORE, MIX_AIV | none | none observed | A |
| global_mean_pool | AI_VECTOR_CORE, MIX_AIV | none | none observed | A |
| global_add_pool | AI_VECTOR_CORE, MIX_AIV | none | none observed | A |
| global_max_pool | AI_VECTOR_CORE | none | explicit runtime `npu_cpu_fallback` for `aten::scatter_reduce.two_out` | C2 |
| global_sort_pool | AI_VECTOR_CORE, MIX_AIV, AI_CPU | `Cumsum`, INT64[32] | not confirmed | C1 |
| TopKPooling | AI_VECTOR_CORE, MIX_AIV, AI_CPU | `Sort` INT64[1024], `Cumsum` INT64[16] | not confirmed | C1 |
| SAGPooling | AI_CORE, AI_VECTOR_CORE, MIX_AIV, AI_CPU | `Sort` INT64[1024], `Cumsum` INT64[16] | not confirmed | C1 |

Raw profiler trees remain inside `wio-pyg-cann851-pyg280:/root/pyg_validation/<API>/profile_complete`.

