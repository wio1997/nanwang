#!/usr/bin/env python3
"""Repeatability check for the original-PyG (host CPU fallback) baseline.

The fallback path is the only comparison point we have, so its stability has to
be characterised before any speed-up number built on it can be trusted.
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # parent scripts/ dir (pg_env, pg_dataset)
from bench_forward import DTYPES, ORIGINAL_GMP, time_async  # noqa: E402
import pg_dataset  # noqa: E402

DEV = "npu:0"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cases", default="ieee24:128,ieee39:128,ieee118:32,ieee118:64,"
                                       "ieee118:128,uk:128")
    ap.add_argument("--dtype", default="fp32")
    ap.add_argument("--repeats", type=int, default=5)
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--iters", type=int, default=50)
    ap.add_argument("--data-root", default=pg_dataset.pg_env.data_root())
    ap.add_argument("--out", default=os.path.join(
        os.environ.get("POWERGRAPH_RESULTS_ROOT", pg_dataset.pg_env.results_root()),
        "baseline_stability.json"))
    args = ap.parse_args()

    torch.npu.set_device(0)
    cache = {}
    results = []
    for spec in args.cases.split(","):
        ds_name, bs = spec.split(":")
        bs = int(bs)
        if ds_name not in cache:
            cache[ds_name] = pg_dataset.get_dataset(args.data_root, ds_name, "Binary")
        dataset = cache[ds_name]
        batch, bs_eff = pg_dataset.first_batch(dataset, bs)
        x = batch.x.float().to(DEV).to(DTYPES[args.dtype])
        bidx = batch.batch.to(torch.int64).to(DEV)
        nodes = int(x.size(0))
        means = []
        for _ in range(args.repeats):
            with torch.no_grad():
                _, ev = time_async(lambda: ORIGINAL_GMP(x, bidx), args.warmup, args.iters)
            means.append(float(np.mean(ev)))
        rec = {
            "dataset": ds_name, "batch": bs, "dtype": args.dtype, "nodes": nodes,
            "repeats": args.repeats, "iters": args.iters,
            "repeat_mean_us": [round(m, 2) for m in means],
            "min_us": round(min(means), 2), "median_us": round(statistics.median(means), 2),
            "max_us": round(max(means), 2),
            "spread_x": round(max(means) / min(means), 2) if min(means) > 0 else None,
            "loadavg": list(os.getloadavg()),
        }
        results.append(rec)
        print(f"{ds_name:>8} b{bs:<4} nodes={nodes:<6} repeats={rec['repeat_mean_us']} "
              f"spread={rec['spread_x']}x loadavg={rec['loadavg']}", flush=True)

    with open(args.out, "w") as fh:
        json.dump({"cases": results}, fh, indent=2)
    print("wrote", args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
