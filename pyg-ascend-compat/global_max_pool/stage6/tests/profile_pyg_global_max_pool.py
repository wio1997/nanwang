#!/usr/bin/env python3
"""msprof application: REAL torch_geometric.nn.global_max_pool on NPU with compat enabled.

  msprof --application="python3 tests/profile_pyg_global_max_pool.py" --output=<dir> --ai-core=on ...

Shape: N=4096, F=33 (non-aligned -> padded path), S=64, FP32, forward.
"""

from __future__ import annotations

import os
import sys

import torch
import torch_npu  # noqa: F401
import torch_geometric

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import pyg_ascend_compat  # noqa: E402

pyg_ascend_compat.enable(debug=False)
from torch_geometric.nn import global_max_pool  # noqa: E402


def main():
    torch.npu.set_device(0)
    n, f, s = 4096, 33, 64
    torch.manual_seed(0)
    x = (torch.rand((n, f), dtype=torch.float32) - 0.5).to("npu:0")
    batch = torch.randint(0, s, (n,), dtype=torch.int64).to("npu:0")

    for _ in range(3):  # warm-up (first call also builds the ACLNN executor)
        out = global_max_pool(x, batch, s)
    torch.npu.synchronize()
    del out
    for _ in range(5):  # profiled region
        out = global_max_pool(x, batch, s)
    torch.npu.synchronize()
    print("PyG global_max_pool OK", tuple(out.shape), out.dtype, out.device,
          "| pyg", torch_geometric.__version__,
          "| compat", pyg_ascend_compat.stats()["ascend_calls"], "ascend calls")


if __name__ == "__main__":
    main()
