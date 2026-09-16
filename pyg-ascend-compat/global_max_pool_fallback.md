# global_max_pool fallback evidence

Environment: CANN 8.5.1, PyTorch 2.9.0+cpu, torch_npu 2.9.0,
PyG 2.6.1, Ascend 910B3, `npu:1`.

Observed during the legal unified screen and benchmark:

```text
Warning: CAUTION: The operator 'aten::scatter_reduce.two_out' is not currently
supported on the NPU backend and will fall back to run on the CPU. This may
have performance implications. (function npu_cpu_fallback)
```

The final output tensors remained on `npu:1` and fp32/fp16/bf16 forward and
backward comparisons passed. The explicit torch_npu warning nevertheless
confirms Host Tensor fallback; output device alone does not negate it.

