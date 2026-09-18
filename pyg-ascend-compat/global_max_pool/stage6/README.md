# Stage 6 — PyG `global_max_pool` on Ascend via ScatterMaxV1

Routes the real `torch_geometric.nn.global_max_pool` to the validated Ascend adapter for the
FP32 / NPU / forward case, replacing the `aten::scatter_reduce` host CPU fallback.

```
torch_geometric.nn.global_max_pool
  -> pyg_ascend_compat wrapper            (dispatch, forward, fp32, NPU)
  -> global_max_pool_ascend               (Stage 2/3A adapter, reused as-is)
  -> aclnnScatterMaxV1                    (DrivingSDK custom op)
  -> AI_VECTOR_CORE
```

## Run it

```bash
# 1. one-time build of the bridge extension (Stage 2) - needs the isolated custom OPP from Stage 1
cd /root/zyg/global_max_pool/stage2 && bash extension/build_bridge.sh

# 2. shell-local environment (isolated custom OPP + this package on PYTHONPATH)
source /root/zyg/global_max_pool/stage6/env.sh

# 3. demo / tests
python3 /root/zyg/global_max_pool/stage6/demo_global_max_pool_ascend.py
python3 /root/zyg/global_max_pool/stage6/tests/run_stage6_tests.py
```

Usage in application code:

```python
import pyg_ascend_compat
pyg_ascend_compat.enable()             # 1) enable first
from torch_geometric.nn import global_max_pool   # 2) then import the PyG symbol
out = global_max_pool(x, batch, size)
```

**Import order constraint:** `enable()` must happen *before* `from torch_geometric.nn import
global_max_pool`. An earlier `from ... import ...` binds the original function object and later
`enable()` calls cannot rebind that local name (re-import after enabling, or call
`torch_geometric.nn.global_max_pool(...)` which is always rebound).

`disable()` restores the original PyG function; `stats()` reports the call counters
(`ascend_calls` vs `original_calls`); `set_debug(True)` prints one line per dispatched call.

## Files

| file | role |
|---|---|
| `pyg_ascend_compat/__init__.py` | enable/disable/stats wrapper around the PyG symbol |
| `env.sh` | shell-local env (custom OPP, adapter path, bridge path) |
| `tests/run_stage6_tests.py` | PYG1..PYG8 + before/after fallback evidence (20/20 PASS) |
| `tests/pyg_fallback_probe.py` | fresh-process probe used to capture the fallback notice |
| `tests/profile_pyg_global_max_pool.py` | msprof app calling the real PyG API (N=4096, F=33, S=64) |
| `demo_global_max_pool_ascend.py` | one-shot demo (N=8, F=7, size=5) |

Delivery summary: `../../DELIVERY_global_max_pool_ascend.md` ·
full report: `../stage6_pyg_global_max_pool.md`
