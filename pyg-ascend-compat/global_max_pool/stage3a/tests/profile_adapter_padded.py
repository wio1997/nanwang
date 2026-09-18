#!/usr/bin/env python3
"""msprof application: full adapter path for a NON-ALIGNED feature dim (N=4096, F=33, S=64).

msprof --application="python3 tests/profile_adapter_padded.py" --output=<dir> --ai-core=on ...
"""

from __future__ import annotations

import os
import sys

import torch
import torch_npu  # noqa: F401

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.normpath(os.path.join(HERE, "..", "..", "stage2", "python")))

from global_max_pool_ascend import global_max_pool_ascend  # noqa: E402


def main():
    torch.npu.set_device(0)
    n, f, s = 4096, 33, 64
    torch.manual_seed(0)
    x = (torch.rand((n, f), dtype=torch.float32) - 0.5).to("npu:0")
    batch = torch.randint(0, s, (n,), dtype=torch.int64).to("npu:0")

    for _ in range(3):
        out = global_max_pool_ascend(x, batch, s)
    torch.npu.synchronize()
    del out
    for _ in range(5):
        out = global_max_pool_ascend(x, batch, s)
    torch.npu.synchronize()
    print("padded adapter path OK", tuple(out.shape), out.dtype, out.device)


if __name__ == "__main__":
    main()
