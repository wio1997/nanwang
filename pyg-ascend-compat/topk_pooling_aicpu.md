# TopKPooling profiler evidence

- Representative case: `x=[1024,32]`, 16 graphs, 64 nodes per graph, ratio `0.5`, device `npu:1`.
- Observed core types: `AI_CPU`, `AI_VECTOR_CORE`, `MIX_AIV`.
- AICPU kernels:
  - `aclnnSort_SortAiCpu_Sort`, input `INT64[1024]`.
  - `aclnnCumsum_CumsumAiCpu_Cumsum`, input `INT64[16]` (plus scalar axis).
- No runtime `npu_cpu_fallback` warning was observed; this establishes device-side AICPU, not Host CPU tensor fallback.
- Raw profiler remains in container `wio-pyg-cann851-pyg280:/root/pyg_validation/TopKPooling/profile_complete`.
