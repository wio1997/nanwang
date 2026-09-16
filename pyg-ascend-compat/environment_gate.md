# PyG Ascend compatibility environment gate

Checked: 2026-09-16 (Asia/Shanghai)

## Requested target

- Hardware: Ascend 910B3
- CANN: 8.5.1
- PyTorch: 2.9.0
- torch_npu: 2.9.0
- APIs: GraphNorm, global_mean_pool, global_add_pool, global_max_pool,
  global_sort_pool, TopKPooling, SAGPooling

## Observed original environment

- Connection and host identifiers are omitted from the public artifact.
- Container: `wio-cann-8card-service`
- Image: `quay.io/ascend/cann:9.0.0-910b-ubuntu22.04-py3.12`
- `ASCEND_HOME_PATH=/usr/local/Ascend/cann-9.0.0`
- Python: `/usr/local/python3.12.13/bin/python3`
- `import torch`: `ModuleNotFoundError: No module named 'torch'`
- No conda/venv activation scripts or installed `torch` package were found in the
  inspected container paths.
- `npu-smi info` reports eight healthy 910B3 devices. Device 0 was already used by
  an unrelated `vllm` process; devices 1-7 showed no process at inspection time.

## Gate result

The original container remains `BLOCKED / U` for the requested target.

After explicit user authorization, an isolated replacement environment was
created without modifying or stopping the original service:

- Image: `local/wio-pyg-cann851:torch2.9-pyg2.6.1`
- Container: `wio-pyg-cann851-torch290`
- Base: `swr.cn-south-1.myhuaweicloud.com/ascendhub/cann:8.5.1-910b-ubuntu22.04-py3.11`
- Active CANN path: `/usr/local/Ascend/cann-8.5.1`
- PyTorch: `2.9.0+cpu`
- torch_npu: `2.9.0`
- PyG: `2.6.1`
- NPU availability/count: `True / 8`
- All seven requested API symbols import successfully.
- Runtime smoke test on `npu:1`: a 4x4 float32 matrix product synchronized,
  matched the expected CPU values, and returned a tensor on `npu:1`.

Replacement gate: `PASS`. API compatibility results may now be collected in
the isolated container. PyTorch's `+cpu` local suffix is expected for the CPU
base package used by torch_npu; device execution was independently proven.

The base CANN image has incomplete third-party dependency metadata for optional
compile/tuning tools (`pip check` reports entries such as decorator/scipy and
also standard-library names). This is retained as an environment risk, but did
not prevent torch_npu import or the NPU smoke test. No packages were added merely
to silence those warnings.

## Version evidence

The official Ascend PyTorch compatibility table maps torch_npu package 2.9.0 and
PyTorch 2.9.0 to the CANN 8.5.0 release family. The requested CANN 8.5.1 patch
level therefore needs an explicitly provisioned image/runtime and must be
captured from the actual process, not inferred from an image name.

Sources:

- https://github.com/Ascend/pytorch/blob/master/COMPATIBILITY.en.md
- https://github.com/Ascend/DrivingSDK/blob/master/docker/8.5.1-910b-ubuntu22.04/Dockerfile
