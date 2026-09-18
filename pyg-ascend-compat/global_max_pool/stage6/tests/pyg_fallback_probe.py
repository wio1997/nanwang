#!/usr/bin/env python3
"""Tiny PyG call used to capture the torch_npu host-fallback notice in a fresh process.

  python3 pyg_fallback_probe.py            # plain PyG  -> expect npu_cpu_fallback on stderr
  python3 pyg_fallback_probe.py --enable   # with compat -> expect no fallback text

torch_npu prints the notice from C++ to stderr, so it cannot be observed reliably from inside the
process that triggers it; this script is meant to be run by the Stage 6 test via subprocess.
"""

import sys

import torch
import torch_npu  # noqa: F401
import torch_geometric.nn as pyg_nn

if "--enable" in sys.argv:
    import pyg_ascend_compat

    pyg_ascend_compat.enable(debug=True)
    from torch_geometric.nn import global_max_pool as call_gmp
else:
    call_gmp = pyg_nn.global_max_pool

torch.npu.set_device(0)
x = torch.tensor([[1.0, -2.0], [3.0, -4.0], [5.0, -6.0]], dtype=torch.float32).to("npu:0")
b = torch.tensor([0, 1, 0], dtype=torch.int64).to("npu:0")
out = call_gmp(x, b)
torch.npu.synchronize()
print("PROBE_OUTPUT", out.detach().cpu().tolist())
