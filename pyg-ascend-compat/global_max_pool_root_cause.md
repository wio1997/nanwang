# global_max_pool root-cause note

- Status remains **C2**: correctness passes, but runtime explicitly reports Host CPU tensor fallback for `aten::scatter_reduce.two_out`.
- Call chain: `global_max_pool(x, batch)` → PyG `scatter(..., reduce='max')` → on NPU `src.is_cuda` is false, so PyG selects in-place `scatter_reduce_(amax, include_self=False)` → unsupported torch_npu ATen operator → `npu_cpu_fallback`.
- `batch=None` takes `x.max(dim=0)[0]` and bypasses this exact path; it is a distinct mode and was not part of the representative batched profiler case.
- Installing `torch_scatter` or setting `size` does not remove this NPU branch under the examined PyG logic.
- Controlled tests show that neither `include_self=False/True` nor `Tensor.scatter_reduce_` versus `torch.scatter_reduce` changes the fallback. The fallback is therefore not attributed to either property.
- `scatter_reduce.two_out` is an ATen overload reached by the framework lowering; it is not the name of a CANN kernel. The fallback occurs because this ATen route has no supported NPU implementation in the tested stack.
- CANN 8.5.1 contains an Ascend 910B AI Core `ScatterMax` implementation and precompiled float32 variants for both INT32 and INT64 indices. TorchAir also generates a raw GE `ScatterMax` op, while its `scatter_reduce.two`, `scatter_reduce.two_out` and `scatter_reduce_.two` converters are unimplemented.
- This makes `ADAPTER_GAP` a plausible hypothesis, not a final classification. Raw INT32 and INT64 `ScatterMax` probes both reached device execution but failed with `507011` and an AI Vector MTE address-out-of-range error. A mutation/alias-aware temporary invocation was rejected by PyTorch functionalization before device execution, so it did not test the device path.
- A semantics-preserving forward composition would require a negative-infinity initialization, device-side `ScatterMax`, an independent occupancy mask, and zero restoration for empty groups. It must separately handle negative-only groups, empty groups, repeated indices, non-finite values and dimension mapping. Training support additionally requires tie-gradient validation.
- No workaround, adapter or new operator was implemented. The next gate is a supported installed invocation path that correctly realizes `ScatterMax` rewritten-variable binding; until then, neither `ADAPTER_GAP` nor `COMPUTE_CAPABILITY_GAP` is final.
