#!/usr/bin/env python3
"""Run the full adapter path a few times so msprof can attribute every step to a device core.

msprof --application="python3 tests/profile_adapter_path.py" --output=<dir> --ai-core=on ...
"""

from __future__ import annotations

import os
import sys

import torch
import torch_npu  # noqa: F401

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "python"))

from global_max_pool_ascend import global_max_pool_ascend  # noqa: E402


def main():
    torch.npu.set_device(0)
    n, f, s = 4096, 32, 64

    # build the inputs on CPU first (a CPU generator cannot seed NPU tensor creation)
    torch.manual_seed(0)
    x = (torch.rand((n, f), dtype=torch.float32) - 0.5).to("npu:0")
    batch = torch.randint(0, s, (n,), dtype=torch.int64).to("npu:0")

    # warm-up: loads the extension, builds the op executor/tiling cache once
    for _ in range(3):
        out = global_max_pool_ascend(x, batch, s)
    torch.npu.synchronize()
    del out

    # profiled region
    for _ in range(5):
        out = global_max_pool_ascend(x, batch, s)
    torch.npu.synchronize()
    print("adapter path OK", tuple(out.shape), out.dtype, out.device)


if __name__ == "__main__":
    main()
