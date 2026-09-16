# SAGPooling root-cause note

- Status remains **C1**: correctness passes; device-side AICPU Sort/Cumsum is confirmed; Host CPU tensor fallback is not confirmed.
- Call chain: `SAGPooling.forward` → default `GraphConv(add)` scoring → `SelectTopK` ratio path → `topk` → `FilterEdges`.
- `INT64[1024]` Sort is the stable per-batch regrouping of score-ordered nodes; `INT64[16]` Cumsum builds offsets from per-graph node counts. These are shared with TopKPooling's ratio selection and do not originate in the default GraphConv.
- Scope is default GraphConv, `min_score=None`, ratio mode, eager forward. Backward, compile, alternate GNNs, ties near the selection boundary, empty graphs, and `min_score` are unverified.
- `min_score` changes semantics and may introduce max-scatter risks; it is not an equivalent workaround. No new operator or alternative implementation was attempted.

