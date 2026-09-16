# TopKPooling root-cause note

- Status remains **C1**: correctness passes; profiler confirms device-side AICPU Sort and Cumsum. No Host CPU tensor fallback is established.
- Ratio-path call chain: `TopKPooling.forward` → `SelectTopK.forward` → `topk(score, ratio, batch, min_score=None)`.
- `INT64[1024]` Sort is the second sort in `topk`: stable sorting the score-ordered `batch` vector so nodes are regrouped per graph while retaining within-graph score order.
- `INT64[16]` Cumsum is the prefix sum of per-graph node counts used to construct graph offsets.
- Changing feature dtype cannot remove these Long metadata paths. The `min_score` branch changes selection semantics and is not an equivalent workaround.
- Only ratio/eager/forward was tested; backward, compile, empty graphs, score ties, and `min_score` mode remain unverified.

