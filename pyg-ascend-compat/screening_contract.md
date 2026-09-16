# First-pass screening contract

## Status rubric

- **A**: legal cases pass function, accuracy and device contracts; complete trace
  observes AI Core/Vector device execution and no fallback in the covered range.
- **B**: A conditions hold for a clearly bounded supported subset, with a
  reproduced dtype, shape, argument or mode restriction.
- **C1**: function, accuracy and device contracts pass, but AICPU execution is
  observed; Host Tensor fallback is not confirmed.
- **C2**: function and accuracy pass, but Host Tensor fallback is confirmed.
- **D**: a legal testcase has a confirmed functional failure, excessive error or
  device-contract violation. Scope the result to the failing configuration.
- **U**: environment, legal reference or execution-path evidence is insufficient.

Output-on-NPU alone is not evidence against fallback. AICPU execution is recorded
separately from Host Tensor fallback. Normal Python/CPU scheduling and scalar
synchronization are not, by themselves, Host Tensor fallback.

## Common measurements

Each API row must record: exact environment, case id, mode, dtype, input/output
shapes, recursive output devices, success/error, CPU reference policy, allclose,
max absolute/relative error, exact comparison for integer outputs, profiler
coverage, AI Core/Vector observed, AICPU observed, Host Tensor fallback evidence,
warmup/iterations, CPU median/p90, NPU wall median/p90, and status.

Timing excludes input transfer and reference comparison. NPU timing synchronizes
before and after measurement. Profiler and benchmark runs are separate.

## API baselines

- GraphNorm: `x=[9,8]`, `batch=[0,0,0,1,1,1,1,1,1]`; independent formula uses
  per-graph mean, learnable `mean_scale`, variance of centered values, weight and
  bias. Train/eval both use current graph statistics.
- global_mean_pool/global_add_pool/global_max_pool: the same non-equal two-graph
  input, explicit `size=2`, plus `batch=None` single-graph mode.
- global_sort_pool: non-tied final feature, `k=4`; compare descending last-channel
  order and zero padding. If absent in the installed PyG version, record the API
  availability conflict rather than silently substituting `SortAggregation`.
- TopKPooling: two graphs, `N=[5,7]`, `F=8`, deterministic non-tied scores,
  bidirectional within-graph edges, `ratio=0.5`; copy one state dict to CPU/NPU and
  compare all six returned fields. Test `min_score` only as a separate mode.
- SAGPooling: same graph contract and return comparison, default `GraphConv`, one
  shared state dict. A GraphConv-path failure is first reported as SAG failure;
  internal attribution happens only in an escalation loop.

Baseline dtype is float32. Float16 and bfloat16 are separate probe rows. Untested
compile/export/distributed modes remain explicitly `untested`.

