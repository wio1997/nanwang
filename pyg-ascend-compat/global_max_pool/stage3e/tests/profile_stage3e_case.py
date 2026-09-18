#!/usr/bin/env python3
"""msprof application for the Stage 3E largeTail profiler runs (adapter level, formal OPP).

  msprof --application="python3 profile_stage3e_case.py --n 40 --f 48825 --s 8 --tag P1" \
         --output=<dir> --ai-core=on

3 warm-up calls + 5 profiled calls, exactly like the Stage 2/3B/3D profiler runs.
"""

from __future__ import annotations

import argparse
import os
import sys

import torch
import torch_npu  # noqa: F401

sys.path.insert(0, "/root/zyg/global_max_pool/stage3b/tests")
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
    print(f"STAGE3E_PROFILE {args.tag} OK shape={tuple(out.shape)} dtype={out.dtype} "
          f"device={out.device}")


if __name__ == "__main__":
    main()
