#!/usr/bin/env python3
"""msprof application for the Stage 4 profiler gate: forward + loss + backward.

  msprof --application="python3 profile_stage4_backward.py --n 64 --f 8 --s 4 --tag P-BWD1" \
         --output=<dir> --ai-core=on

Deterministic data with ties (so the backward really executes the tie path).
"""

from __future__ import annotations

import argparse
import os
import sys

import torch
import torch_npu  # noqa: F401

sys.path.insert(0, "/root/zyg/global_max_pool/stage4/python")
import global_max_pool_ascend_autograd as AG  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, required=True)
    ap.add_argument("--f", type=int, required=True)
    ap.add_argument("--s", type=int, required=True)
    ap.add_argument("--tag", default="case")
    args = ap.parse_args()

    torch.npu.set_device(0)
    n, f, s = args.n, args.f, args.s
    x = ((torch.arange(n, dtype=torch.float32).unsqueeze(1) % 5.0)
         .expand(n, f).contiguous())
    x[0] = 9.0                     # unique max at the head
    x[n - 1] = 9.0                 # tie at the tail
    if f > 1:
        x[:, -1] = 13.0            # tie in the last feature
    x = x.to("npu:0")
    batch = (torch.arange(n) % s).to("npu:0")
    up = torch.ones((s, f), dtype=torch.float32, device="npu:0")

    for _ in range(2):  # warm-up
        xr = x.clone().requires_grad_(True)
        out = AG.global_max_pool_ascend_autograd(xr, batch, s)
        (out * up).sum().backward()
        torch.npu.synchronize()
        del xr, out
    for _ in range(3):  # profiled region (forward + loss + backward)
        xr = x.clone().requires_grad_(True)
        out = AG.global_max_pool_ascend_autograd(xr, batch, s)
        loss = (out * up).sum()
        loss.backward()
        torch.npu.synchronize()
    print(f"STAGE4_PROFILE {args.tag} OK shape={tuple(out.shape)} grad_shape={tuple(xr.grad.shape)} "
          f"grad_sum={float(xr.grad.sum().item())}")


if __name__ == "__main__":
    main()
