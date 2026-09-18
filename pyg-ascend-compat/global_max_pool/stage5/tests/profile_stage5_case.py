#!/usr/bin/env python3
"""msprof application for Stage 5: forward + loss + backward for FP16 / BF16."""

from __future__ import annotations

import argparse
import sys

import torch
import torch_npu  # noqa: F401

sys.path.insert(0, "/root/zyg/global_max_pool/stage5/python")
import global_max_pool_ascend_dtype as S5  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, required=True)
    ap.add_argument("--f", type=int, required=True)
    ap.add_argument("--s", type=int, required=True)
    ap.add_argument("--dtype", choices=("fp16", "bf16"), required=True)
    ap.add_argument("--tag", default="case")
    args = ap.parse_args()

    dt = torch.float16 if args.dtype == "fp16" else torch.bfloat16
    torch.npu.set_device(0)
    n, f, s = args.n, args.f, args.s
    x = ((torch.arange(n, dtype=torch.float32).unsqueeze(1) % 5.0).expand(n, f).contiguous())
    x[0] = 9.0
    x[n - 1] = 9.0
    if f > 1:
        x[:, -1] = 13.0
    x = x.to(dt).to("npu:0")
    batch = (torch.arange(n) % s).to("npu:0")
    up = torch.ones((s, f), dtype=dt, device="npu:0")

    for _ in range(2):
        xr = x.clone().requires_grad_(True)
        out = S5.global_max_pool_ascend_dtype(xr, batch, s)
        (out * up).sum().backward()
        torch.npu.synchronize()
        del xr, out
    for _ in range(3):
        xr = x.clone().requires_grad_(True)
        out = S5.global_max_pool_ascend_dtype(xr, batch, s)
        loss = (out * up).sum()
        loss.backward()
        torch.npu.synchronize()
    print(f"STAGE5_PROFILE {args.tag} OK dtype={dt} out={tuple(out.shape)}/{out.dtype} "
          f"grad={tuple(xr.grad.shape)}/{xr.grad.dtype} grad_sum={float(xr.grad.float().sum().item())}")


if __name__ == "__main__":
    main()
