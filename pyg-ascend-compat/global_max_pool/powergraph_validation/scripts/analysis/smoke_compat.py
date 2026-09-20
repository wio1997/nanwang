"""Smoke test: prove the frozen compat layer intercepts global_max_pool on NPU."""

from __future__ import annotations

import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # scripts/
import pg_env  # noqa: E402

state = pg_env.bootstrap_compat(debug=True)
print("[smoke] wrapped modules:", sorted(state))

from torch_geometric.nn import global_max_pool  # noqa: E402

print("[smoke] global_max_pool:", global_max_pool, "compat=", getattr(global_max_pool, "_pyg_ascend_compat", False))

import torch_npu  # noqa: E402

dev = "npu:0"
for dtype in (torch.float32, torch.float16, torch.bfloat16):
    x = torch.randn(64, 8, device=dev, dtype=dtype)
    b = torch.randint(0, 16, (64,), device=dev, dtype=torch.int64)
    b[0] = 0
    out = global_max_pool(x, b)
    torch.npu.synchronize()
    print(f"[smoke] {str(dtype):>16}: out={tuple(out.shape)} dtype={out.dtype} finite={torch.isfinite(out).all().item()}")

import pyg_ascend_compat  # noqa: E402

print("[smoke] stats:", pyg_ascend_compat.stats())
print("[smoke] adapter:", pyg_ascend_compat.stats()["adapter_path"])
