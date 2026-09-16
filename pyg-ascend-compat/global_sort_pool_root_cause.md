# global_sort_pool root-cause note

- Status remains **C1**: correctness passes; profiler confirms device-side `aclnnCumsum_CumsumAiCpu_Cumsum` on `INT64[32]`. No Host CPU tensor fallback is established.
- Call chain: deprecated `global_sort_pool` wrapper → `SortAggregation` → `Aggregation.to_dense_batch` → `to_dense_batch` counts nodes per graph → PyG `cumsum(num_nodes)` → `torch.cumsum(..., out=...)`.
- The `INT64[32]` shape matches the 32 graph counts in the representative case, providing high-confidence source attribution. It does not mean feature sorting as a whole ran on AICPU.
- The wrapper is deprecated in PyG 2.6.1. Moving to public `SortAggregation` removes the wrapper warning but uses the same dense-batch/Cumsum path, so it does not by itself remove AICPU.
- Untested limits include tied sort keys, empty input, NaN/Inf, extreme fp16/bf16 values, compile mode, and alternative graph-size regimes.

