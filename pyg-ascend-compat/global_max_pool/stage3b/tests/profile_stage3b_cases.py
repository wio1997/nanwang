#!/usr/bin/env python3
"""msprof application for the Stage 3B representative cases (adapter level).

  python3 profile_stage3b_cases.py --n 4096 --f 33 --s 64 --tag C1_smallTail

Cases used by the report:
  C1 smallTail reference : N=4096  F=33   S=64
  C2 leftSrc             : N=4097  F=33   S=64
  C3 threshold-1 (max F that still uses the SMALL_TAIL kernel with N=40) : N=40 F=48824 S=8

NOTE: shapes whose tiling selects LARGE_TAIL cannot be profiled: the delivered package only
registers kernel function entry 0 (SMALL_TAIL), so the launch fails with aclnn status 361001
before any device task is created (see stage3b_large_tail_boundary.md).
"""

from __future__ import annotations

import argparse
import os
import sys

import torch
import torch_npu  # noqa: F401

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import stage3b_common as C  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, required=True)
    ap.add_argument("--f", type=int, required=True)
    ap.add_argument("--s", type=int, required=True)
    ap.add_argument("--tag", default="case")
    args = ap.parse_args()

    torch.npu.set_device(0)
    adapter = C.load_adapter()
    x = C.make_src(args.n, args.f, "det").to(C.DEV)
    batch = C.make_index(args.n, args.s, "spread").to(C.DEV)

    for _ in range(3):  # warm-up
        out = adapter.global_max_pool_ascend(x, batch, args.s)
    torch.npu.synchronize()
    del out
    for _ in range(5):  # profiled region
        out = adapter.global_max_pool_ascend(x, batch, args.s)
    torch.npu.synchronize()
    print(f"STAGE3B_PROFILE {args.tag} OK shape={tuple(out.shape)} dtype={out.dtype} "
          f"device={out.device}")


if __name__ == "__main__":
    main()
