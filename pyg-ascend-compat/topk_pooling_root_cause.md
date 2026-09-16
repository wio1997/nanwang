# TopKPooling root-cause note

- Status remains **C1**: correctness passes; profiler confirms device-side AICPU Sort and Cumsum. No Host CPU tensor fallback is established.
- Ratio-path call chain: `TopKPooling.forward` → `SelectTopK.forward` → `topk(score, ratio, batch, min_score=None)`.
- `INT64[1024]` Sort is the second sort in `topk`: stable sorting the score-ordered `batch` vector so nodes are regrouped per graph while retaining within-graph score order.
- `INT64[16]` Cumsum is the prefix sum of per-graph node counts used to construct graph offsets.
- Controlled Sort tests show both INT32 and INT64 stable integer Sort use AICPU, while otherwise identical FP32 stable Sort uses AI Core and returns the exact CPU stable permutation. This is `TYPE_LIMITATION` for the tested integer Sort family, not an INT64-only effect.
- For non-negative graph IDs in the inclusive range `0..2^24`, casting only the batch sort keys to FP32 preserves ordering and equality exactly. The FP32 stable-sort permutation can therefore be applied to the original INT64 batch without changing `SelectTopK` semantics under that contract. Inputs outside the contract must retain the original path. This reuse was reviewed but not implemented end to end.
- Cumsum remains independently `TYPE_LIMITATION`: INT32 uses AI Core and INT64 uses AICPU in the controlled test. Changing feature dtype cannot remove either Long metadata path. The `min_score` branch changes selection semantics and is not an equivalent workaround.
- Only ratio/eager/forward was tested; backward, compile, empty graphs, score ties, and `min_score` mode remain unverified.
