#!/usr/bin/env python3
"""msprof application: profile one real PowerGraph ``global_max_pool`` case.

Usage (driven by ``run_profiles.sh``):

    msprof --application="python3 profile_app.py --dataset ieee24 --batch 128 \
           --dtype fp32 --path compat_ascend --tag ieee24_b128_fp32" \
           --output=<dir> --ai-core=on

Warm-up calls followed by the profiled calls, same shape as the operator's own
Stage 3E profiler runs.  Nothing is written into any repository.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pg_env  # noqa: E402

import torch_geometric.nn as tgnn  # noqa: E402

ORIGINAL_GMP = tgnn.global_max_pool

import pyg_ascend_compat  # noqa: E402

pyg_ascend_compat.enable(debug=False)

from torch_geometric.nn import global_max_pool as COMPAT_GMP  # noqa: E402

import pg_dataset  # noqa: E402

DTYPES = {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}
DEV = "npu:0"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--dtype", default="fp32", choices=sorted(DTYPES))
    ap.add_argument("--path", default="compat_ascend",
                    choices=["compat_ascend", "original_pyg"])
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--tag", default="case")
    ap.add_argument("--data-root", default=pg_env.data_root())
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    torch.npu.set_device(0)
    dataset = pg_dataset.get_dataset(args.data_root, args.dataset, "Binary")
    batch, bs = pg_dataset.first_batch(dataset, args.batch)
    x = batch.x.float().to(DEV).to(DTYPES[args.dtype])
    bidx = batch.batch.to(torch.int64).to(DEV)
    num_graphs = int(bidx.max().item()) + 1

    fn = COMPAT_GMP if args.path == "compat_ascend" else ORIGINAL_GMP
    pyg_ascend_compat.reset_stats()

    with torch.no_grad():
        for _ in range(args.warmup):
            out = fn(x, bidx)
        torch.npu.synchronize()
        del out
        for _ in range(args.reps):
            out = fn(x, bidx)
        torch.npu.synchronize()

    record = {
        "tag": args.tag,
        "dataset": args.dataset,
        "batch": args.batch,
        "batch_effective": bs,
        "dtype": args.dtype,
        "path": args.path,
        "graphs": num_graphs,
        "nodes": int(x.size(0)),
        "F": int(x.size(1)),
        "reps": args.reps,
        "warmup": args.warmup,
        "out_shape": list(out.shape),
        "out_dtype": str(out.dtype),
        "compat_stats": pyg_ascend_compat.stats(),
    }
    print(f"BENCH_PROFILE {args.tag} OK path={args.path} shape={tuple(out.shape)} "
          f"dtype={out.dtype} device={out.device}")
    print("BENCH_PROFILE_STATS " + json.dumps(record["compat_stats"]))
    if args.out:
        os.makedirs(os.path.dirname(args.out), exist_ok=True)
        with open(args.out, "w") as fh:
            json.dump(record, fh, indent=2)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
