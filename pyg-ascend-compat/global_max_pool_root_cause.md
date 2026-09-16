# global_max_pool root-cause note

- Status remains **C2**: correctness passes, but runtime explicitly reports Host CPU tensor fallback for `aten::scatter_reduce.two_out`.
- Call chain: `global_max_pool(x, batch)` → PyG `scatter(..., reduce='max')` → on NPU `src.is_cuda` is false, so PyG selects in-place `scatter_reduce_(amax, include_self=False)` → unsupported torch_npu ATen operator → `npu_cpu_fallback`.
- `batch=None` takes `x.max(dim=0)[0]` and bypasses this exact path; it is a distinct mode and was not part of the representative batched profiler case.
- Installing `torch_scatter` or setting `size` does not remove this NPU branch under the examined PyG logic.
- A per-graph slice-plus-`amax` formulation is only a candidate semantic workaround and was not implemented. Tie-gradient semantics and empty-graph behavior must be checked before claiming equivalence; `max(dim)` is not automatically equivalent to `amax` for tied gradients.

