# Representative native-path evidence

Scope: the profiler capture for each API is a representative **fp32, no-grad/eval forward** at the performance shape. Functional/precision cases separately cover fp32/fp16/bf16 and backward where reported. Absence of fallback in this capture is not extrapolated to every dtype, backward, shape, or compile mode.

| API | Profile location in container | Core types in `op_statistic.csv` | `AICPU` row in `kernel_details.csv` | Fallback marker search in profile tree | Runtime warning during capture |
|---|---|---|---|---|---|
| GraphNorm | `/root/pyg_validation/graphnorm/profile_complete` | `AI_VECTOR_CORE`, `MIX_AIV` | none | none | none observed |
| global_mean_pool | `/root/pyg_validation/global_mean_pool_rerun/profile_complete` | `AI_VECTOR_CORE`, `MIX_AIV` | none | none | none observed |
| global_add_pool | `/root/pyg_validation/global_add_pool/profile_complete` | `AI_VECTOR_CORE`, `MIX_AIV` | none | none | none observed |

This evidence supports status A for the bounded representative path under the frozen rubric. It does not prove universal native execution.

