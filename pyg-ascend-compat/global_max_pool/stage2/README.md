# Stage 2 — FP32 aligned `global_max_pool` adapter (ScatterMaxV1)

Standalone Python API:

```python
global_max_pool_ascend(x, batch, size=None)   # x: [N,F] fp32 on NPU, batch: [N] int64 on NPU
```

The reduction is DrivingSDK's `ScatterMaxV1` custom op (from Stage 1) reached through a minimal
PyTorch C++ bridge. PyG / `torch_npu` / `aten::scatter_reduce` are **not** patched here.

## Layout

```
stage2/
├── extension/
│   ├── scattermaxv1_bridge.cpp   # torch.Tensor -> aclTensor -> aclnnScatterMaxV1
│   └── build_bridge.sh           # plain g++ build (torch-bundled pybind11, no ninja needed)
├── python/
│   └── global_max_pool_ascend.py # adapter: validation, INT64->INT32, occupancy, empty->0
└── tests/
    ├── run_stage2_tests.py       # A1..A16 correctness matrix (CPU golden, tol 0)
    ├── probe_occupancy.py        # occupancy/post-process op fallback probe
    └── profile_adapter_path.py   # app profiled by msprof for the full path
```

## Usage

```bash
cd /root/zyg/global_max_pool/stage2
export ASCEND_CUSTOM_OPP_PATH=/root/zyg/build/scattermax_runtime_opp/vendors/customize
export LD_LIBRARY_PATH=$ASCEND_CUSTOM_OPP_PATH/op_api/lib:$LD_LIBRARY_PATH

bash extension/build_bridge.sh          # -> /root/zyg/build/stage2_ext/scattermaxv1_bridge.so
python3 tests/run_stage2_tests.py       # 16/16 PASS expected
```

```python
import sys; sys.path.insert(0, "python")
from global_max_pool_ascend import global_max_pool_ascend

out = global_max_pool_ascend(x, batch, size)   # size=None -> max(batch)+1
```

Semantics: empty group → 0; non-empty group → exact max, **including a genuine `-inf` max**
(occupancy mask, not `out == -inf → 0`).

Limits (Stage 2): fp32 only, `F % 8 == 0`, forward only (`requires_grad=True` is rejected),
batch must be int64 with `0 <= idx < min(size, 491520)`.

Details, profiler evidence and the full matrix: `../stage2_global_max_pool_adapter.md`.
