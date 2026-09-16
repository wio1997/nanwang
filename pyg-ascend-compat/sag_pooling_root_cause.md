# SAGPooling root-cause note

- Status remains **C1**: correctness passes; device-side AICPU Sort/Cumsum is confirmed; Host CPU tensor fallback is not confirmed.
- Call chain: `SAGPooling.forward` → default `GraphConv(add)` scoring → `SelectTopK` ratio path → `topk` → `FilterEdges`.
- `INT64[1024]` Sort is the stable per-batch regrouping of score-ordered nodes; `INT64[16]` Cumsum builds offsets from per-graph node counts. These are shared with TopKPooling's ratio selection and do not originate in the default GraphConv.
- Controlled attribution is the same as TopKPooling: stable INT32 and INT64 integer Sort use AICPU, FP32 stable Sort uses AI Core, and INT32 versus INT64 Cumsum selects AI Core versus AICPU. The two operations are independently classified as `TYPE_LIMITATION` for the tested modes.
- FP32 stable Sort is conditionally reusable only for exact non-negative graph IDs `<= 2^24`, with its permutation applied to the original INT64 batch. INT32 Cumsum is conditionally reusable only with a proven prefix-sum range and preserved downstream dtype contract. Neither change was integrated.
- Scope is default GraphConv, `min_score=None`, ratio mode, eager forward. Backward, compile, alternate GNNs, ties near the selection boundary, empty graphs, and `min_score` are unverified.
- `min_score` changes semantics and may introduce max-scatter risks; it is not an equivalent workaround. No new operator or alternative implementation was attempted.
