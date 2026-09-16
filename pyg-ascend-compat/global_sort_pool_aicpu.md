# global_sort_pool AICPU evidence

The legal fp32 profile completed with correct `npu:1` output. Its
`ASCEND_PROFILER_OUTPUT/kernel_details.csv` contains:

```text
aclnnCumsum_CumsumAiCpu_Cumsum,Cumsum,dynamic,AI_CPU,...
Input Data Types: INT64;INT64, Output Data Types: INT64
```

All other observed core types were `AI_VECTOR_CORE` or `MIX_AIV`. This is a
confirmed AICPU component, not a Host Tensor fallback. PyG 2.8.0.post1 also emits a
deprecation warning for `global_sort_pool`, recommending `SortAggregation`.
