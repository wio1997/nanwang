#!/usr/bin/env python3
"""Minimal end-to-end demo: real PyG global_max_pool on Ascend via pyg-ascend-compat.

  source /root/zyg/global_max_pool/stage6/env.sh
  python3 /root/zyg/global_max_pool/stage6/demo_global_max_pool_ascend.py

Case: N=8, F=7 (non-aligned -> padded path), batch int64, size=5 (one empty group), negatives.
"""

from __future__ import annotations

import os
import subprocess
import sys

import torch
import torch_npu  # noqa: F401
import torch_geometric

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import pyg_ascend_compat  # noqa: E402

pyg_ascend_compat.enable(debug=True)
from torch_geometric.nn import global_max_pool  # noqa: E402  (import after enable)

N, F, SIZE = 8, 7, 5


def golden(x, batch, size):
    out = torch.zeros((size, x.shape[1]), dtype=torch.float32)
    for g in range(size):
        mask = batch == g
        if bool(mask.any()):
            out[g] = x[mask].max(dim=0).values
    return out


def main():
    torch.npu.set_device(0)
    torch.manual_seed(0)
    x_cpu = torch.round((torch.rand((N, F)) * 20 - 10) * 4) / 4   # mixed positive/negative
    batch_cpu = torch.tensor([0, 1, 0, 2, 1, 2, 0, 3], dtype=torch.int64)

    x = x_cpu.to("npu:0")
    batch = batch_cpu.to("npu:0")
    out = global_max_pool(x, batch, SIZE)          # <-- real PyG API
    torch.npu.synchronize()

    exp = golden(x_cpu, batch_cpu, SIZE)
    got = out.detach().cpu()
    match = torch.equal(exp, got)

    probe = subprocess.run(
        [sys.executable, os.path.join(HERE, "tests", "pyg_fallback_probe.py"), "--enable"],
        capture_output=True, text=True, env=dict(os.environ))
    fallback = ("npu_cpu_fallback" in probe.stderr
                or "fall back to run on the CPU" in probe.stderr)
    st = pyg_ascend_compat.stats()

    print("=" * 68)
    print(f"PyG version          : {torch_geometric.__version__}")
    print(f"torch version        : {torch.__version__}")
    print(f"torch_npu version    : {torch_npu.__version__}")
    print(f"device               : {x.device}")
    print(f"input shape          : {tuple(x.shape)}")
    print(f"batch dtype          : {batch.dtype}")
    print(f"size (explicit)      : {SIZE}  (max(batch)+1 = {int(batch_cpu.max()) + 1} "
          f"-> group {int(batch_cpu.max()) + 1} is empty)")
    print(f"output shape         : {tuple(out.shape)}")
    print(f"output device/dtype  : {out.device} / {out.dtype}")
    print(f"CPU golden match     : {match}")
    print(f"empty group row all 0: {bool((got[int(batch_cpu.max()) + 1] == 0).all())}")
    print(f"compat path used     : ascend_calls={st['ascend_calls']} "
          f"original_calls={st['original_calls']} "
          f"(enabled={st['enabled']})")
    print(f"Host CPU fallback    : {'YES' if fallback else 'NONE'}")
    print("=" * 68)
    print(f"RESULT: {'PASS' if (match and not fallback and st['ascend_calls'] >= 1) else 'FAIL'}")
    return 0 if (match and not fallback) else 1


if __name__ == "__main__":
    raise SystemExit(main())
