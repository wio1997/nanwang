# PyTorch Geometric API compatibility screening

## Executive conclusion

Target: Ascend 910B3, CANN 8.5.1, PyTorch/torch_npu 2.9.0, PyG 2.6.1. All seven legal first-round cases completed successfully and matched their CPU references within the frozen dtype tolerances. No `D` result was found. Execution-path screening yields three `A`, three `C1`, and one `C2`:

Accuracy thresholds were `rtol=1e-4, atol=1e-5` for fp32 and `rtol=atol=0.02` for fp16/bf16. Low-precision NPU outputs were compared against fp32 CPU references; therefore passing means tolerance-based agreement, not bitwise or same-dtype identity.

| API | Function | CPU-reference precision | Output device | Execution-path evidence | Basic performance, CPU → NPU wall median | Limits of this conclusion | Status |
|---|---|---|---|---|---|---|---|
| `GraphNorm` | Pass; eager forward/backward, including input and parameter gradients | Max output abs error: fp32 `2.38e-7`, fp16 `1.84e-3`, bf16 `1.45e-2`; all within tolerance | `npu:1` | Representative fp32 eval/no-grad forward: AI Vector/MIX_AIV; no AICPU row or Host fallback marker/warning observed | `[4096,64]`, B=32: `11.925 → 1.338 ms` (`8.91x`) | fp32/fp16/bf16 training-path correctness tested; native-path claim is limited to representative fp32 eval forward. Compile, empty graphs, extreme/NaN/Inf untested | **A** |
| `global_mean_pool` | Pass; eager forward/backward | Max output abs error: `0`, `5.86e-4`, `1.82e-3` | `npu:1` | Representative fp32 no-grad forward: AI Vector/MIX_AIV; no AICPU row or Host fallback marker/warning observed | `[4096,64]`, B=32: `12.345 → 0.593 ms` (`20.81x`) | Explicit sorted batch and `size`; native-path claim is limited to representative fp32 forward. `batch=None`, empty graphs, compile untested | **A** |
| `global_add_pool` | Pass; eager forward/backward | Max output abs error: `0`, `1.17e-3`, `1.25e-2` | `npu:1` | Representative fp32 no-grad forward: AI Vector/MIX_AIV; no AICPU row or Host fallback marker/warning observed | `[4096,64]`, B=32: `1.523 → 0.291 ms` (`5.24x`) | Explicit sorted batch and `size`; native-path claim is limited to representative fp32 forward. `batch=None`, empty graphs, compile untested | **A** |
| `global_max_pool` | Pass; eager forward/backward | Max output abs error: `0`, `7.81e-4`, `6.25e-3` | Returned tensor is `npu:1` | Explicit `npu_cpu_fallback` warning for unsupported `aten::scatter_reduce.two_out`; returned NPU device does not negate this Host tensor fallback | `[4096,64]`, B=32: `26.093 → 23.829 ms` (`1.10x`) | C2 applies to tested batched path. `batch=None` uses a distinct direct max path and was not screened here | **C2** |
| `global_sort_pool` | Pass; eager forward/backward; `k=4` functional case | Max output abs error: `0`, `7.81e-4`, `6.25e-3` | `npu:1` | Device-side AICPU `Cumsum`, `INT64[32]`; AI Vector/MIX_AIV also observed; no Host tensor fallback evidence | `[4096,64]`, B=32, k=64: `11.228 → 2.167 ms` (`5.18x`) | Wrapper is deprecated; `SortAggregation` uses the same dense-batch Cumsum path. Ties, empty input, compile, NaN/Inf untested | **C1** |
| `TopKPooling` | Pass; all six forward outputs checked | Float max abs error across outputs: fp32 `2.98e-8`, fp16 `1.03e-3`, bf16 `7.00e-3`; discrete outputs exact | All returned tensors `npu:1` | Device-side AICPU Sort `INT64[1024]` and Cumsum `INT64[16]`; AI Vector/MIX_AIV also observed; no Host tensor fallback warning | `[1024,32]`, B=16, ratio=0.5: `27.607 → 3.201 ms` (`8.62x`) | Ratio, `min_score=None`, eager forward only; backward, ties, empty graphs, compile and min-score mode untested | **C1** |
| `SAGPooling` | Pass; all six forward outputs checked with default `GraphConv(add)` | Float max abs error across outputs: fp32 `5.96e-8`, fp16 `4.42e-4`, bf16 `5.61e-3`; discrete outputs exact | All returned tensors `npu:1` | AI Core/Vector/MIX_AIV plus device-side AICPU Sort `INT64[1024]` and Cumsum `INT64[16]`; no Host tensor fallback warning | `[1024,32]`, B=16, ratio=0.5: `41.453 → 3.293 ms` (`12.59x`) | Default GraphConv, ratio, eager forward only; alternate GNN/aggr, backward, ties, compile and min-score mode untested | **C1** |

The three `A` results mean no fallback was observed in the representative profiler capture; they are not universal proofs for every shape or execution mode. `C1` denotes confirmed device-side AICPU, while `C2` denotes confirmed Host CPU tensor fallback.

## Environment rebuilt

- New image: `local/wio-pyg-cann851:torch2.9-pyg2.6.1`
- New container: `wio-pyg-cann851-torch290`
- Runtime: Python 3.11.14, CANN 8.5.1 at `/usr/local/Ascend/cann-8.5.1`, torch `2.9.0+cpu`, torch_npu `2.9.0`, PyG `2.6.1`
- Eight NPUs are visible; validation used `npu:1`. An NPU matmul smoke test passed.
- The original `wio-cann-8card-service` container remains running and unchanged.
- PyG 2.6.1 was selected because it still exposes the requested deprecated `global_sort_pool` wrapper.

This is an empirical compatibility qualification. The referenced public torch_npu compatibility matrix pairs the 2.9.0 line with CANN 8.5.0, whereas this request deliberately tests CANN 8.5.1.

## Triggered root-cause analysis

`global_max_pool` reaches PyG scatter-max and then `aten::scatter_reduce.two_out` for an explicit NPU batch. torch_npu reports that operator unsupported and executes it through Host CPU fallback. This establishes C2. The measured CPU/NPU wall-time ratio is `1.10x`; this screen did not isolate or quantify fallback overhead. No replacement operator was developed.

`global_sort_pool` constructs dense batches using a prefix sum of per-graph Long node counts. The profiler's `INT64[32]` AICPU Cumsum matches B=32. The deprecation replacement `SortAggregation` retains this internal path.

TopKPooling and SAGPooling share `SelectTopK`'s ratio path. Their `INT64[1024]` AICPU Sort is the stable regrouping of score-ordered nodes by graph; `INT64[16]` Cumsum builds offsets from per-graph node counts. For SAGPooling these two kernels come from selection, not default GraphConv. In all three C1 cases, AICPU is device-side execution and is not classified as Host CPU tensor fallback.

## Evidence and reproducibility

- Functional/profiler harnesses: `run_graphnorm_screen.py`, `run_pool_screen.py`
- Per-API machine-readable results: `graphnorm_result.json`, `global_mean_pool_result.json`, `global_add_pool_result.json`, `global_max_pool_result.json`, `global_sort_pool_result.json`, `topk_pooling_result.json`, `sag_pooling_result.json`
- Root-cause notes: `global_max_pool_root_cause.md`, `global_sort_pool_root_cause.md`, `topk_pooling_root_cause.md`, `sag_pooling_root_cause.md`
- Bounded A-path trace summary: `native_path_evidence.md`
- Raw profiler trees remain inside `wio-pyg-cann851-torch290:/root/pyg_validation/<API>/profile_complete`.

Timing is synchronized fp32-forward API wall time, with inputs already resident on the target device, 10 warmups and 50 measured iterations. It excludes input transfer and CPU-reference comparison. It is a screening baseline, not application end-to-end latency, a throughput certification, or a decomposition of AICPU/fallback cost.
