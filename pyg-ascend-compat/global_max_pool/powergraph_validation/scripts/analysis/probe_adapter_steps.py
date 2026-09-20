#!/usr/bin/env python3
"""Per-step cost decomposition of the frozen adapter's per-call work.

Replays exactly the device work the frozen Stage-2 adapter performs for one
``global_max_pool_ascend`` call on a real PowerGraph batch, timing each step in
isolation with a full ``torch.npu.synchronize()`` around it.  This is a
diagnostic probe only - the operator implementation itself is untouched.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # parent scripts/ dir (pg_env, pg_dataset)
import pg_env  # noqa: E402

import pyg_ascend_compat  # noqa: E402

pyg_ascend_compat.enable(debug=False)

from torch_geometric.nn import global_max_pool as COMPAT_GMP  # noqa: E402

import pg_dataset  # noqa: E402

DEV = "npu:0"


def load_adapter():
    path = os.environ["PYG_ASCEND_ADAPTER_PATH"]
    spec = importlib.util.spec_from_file_location("_step_probe_adapter", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def timed_sync(fn, warmup, iters):
    for _ in range(warmup):
        fn()
    torch.npu.synchronize()
    vals = []
    for _ in range(iters):
        torch.npu.synchronize()
        t0 = time.perf_counter()
        fn()
        torch.npu.synchronize()
        vals.append((time.perf_counter() - t0) * 1e6)
    a = np.asarray(vals)
    return {"mean_us": round(float(a.mean()), 2), "p50_us": round(float(np.percentile(a, 50)), 2),
            "p95_us": round(float(np.percentile(a, 95)), 2)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="ieee24")
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--warmup", type=int, default=30)
    ap.add_argument("--iters", type=int, default=200)
    ap.add_argument("--data-root", default=pg_env.data_root())
    ap.add_argument("--out", default=os.path.join(
        os.environ.get("POWERGRAPH_RESULTS_ROOT", pg_env.results_root()),
        "adapter_step_breakdown.json"))
    args = ap.parse_args()

    torch.npu.set_device(0)
    ds = pg_dataset.get_dataset(args.data_root, args.dataset, "Binary")
    batch, bs = pg_dataset.first_batch(ds, args.batch)
    x = batch.x.float().to(DEV)
    bidx = batch.batch.to(torch.int64).to(DEV)
    n, f = x.shape
    graphs = int(bidx.max().item()) + 1

    adapter = load_adapter()
    bridge = adapter._load_bridge()
    f_pad = adapter._align_up(f, adapter.FEATURE_ALIGN_ELEMS)
    s = graphs

    steps = {}
    steps["00_full_op_compat_call"] = timed_sync(
        lambda: COMPAT_GMP(x, bidx), args.warmup, args.iters)
    steps["01_batch_minmax_to_host"] = timed_sync(
        lambda: torch.stack((bidx.min(), bidx.max())).cpu().tolist(), args.warmup, args.iters)
    steps["02_index_to_int32"] = timed_sync(
        lambda: bidx.to(torch.int32).contiguous(), args.warmup, args.iters)
    steps["03_pad_input_features"] = timed_sync(
        lambda: torch.nn.functional.pad(x, (0, f_pad - f), mode="constant", value=float("-inf")),
        args.warmup, args.iters)
    steps["04_torch_full_inf"] = timed_sync(
        lambda: torch.full((s, f_pad), float("-inf"), dtype=torch.float32, device=DEV),
        args.warmup, args.iters)
    steps["05_torch_empty_argmax"] = timed_sync(
        lambda: torch.empty((s, f_pad), dtype=torch.int32, device=DEV), args.warmup, args.iters)

    x_kernel = (torch.nn.functional.pad(x, (0, f_pad - f), mode="constant", value=float("-inf"))
                if f_pad != f else x)
    idx32 = bidx.to(torch.int32).contiguous()
    out_kernel = torch.full((s, f_pad), float("-inf"), dtype=torch.float32, device=DEV)
    argmax = torch.empty((s, f_pad), dtype=torch.int32, device=DEV)
    steps["06_scattermaxv1_kernel_only"] = timed_sync(
        lambda: bridge.scatter_max_v1_forward(x_kernel, idx32, out_kernel, argmax),
        args.warmup, args.iters)

    def occ():
        o = torch.zeros(s, dtype=torch.int32, device=DEV)
        o.scatter_(0, bidx, 1)

    steps["07_occupancy_zeros_plus_scatter"] = timed_sync(occ, args.warmup, args.iters)
    occupied = torch.zeros(s, dtype=torch.int32, device=DEV)
    occupied.scatter_(0, bidx, 1)
    steps["08_masked_fill_empty_groups"] = timed_sync(
        lambda: out_kernel.masked_fill_((occupied == 0).view(s, 1), 0.0),
        args.warmup, args.iters)
    steps["09_crop_contiguous"] = timed_sync(
        lambda: out_kernel[:, :f].contiguous(), args.warmup, args.iters)

    record = {
        "dataset": args.dataset, "batch": bs, "graphs": graphs, "nodes": int(n), "F": int(f),
        "F_kernel": f_pad, "padded": f_pad != f,
        "warmup": args.warmup, "iters": args.iters,
        "methods": "torch.npu.synchronize() immediately before and after each timed call",
        "steps": steps,
        "sum_of_device_steps_us": round(sum(
            v["mean_us"] for k, v in steps.items() if k != "00_full_op_compat_call"), 2),
    }
    for k, v in steps.items():
        print(f"{k:<34} mean={v['mean_us']:8.2f}us p50={v['p50_us']:8.2f} p95={v['p95_us']:8.2f}")
    print(f"sum(device steps) = {record['sum_of_device_steps_us']} us  vs full op "
          f"{steps['00_full_op_compat_call']['mean_us']} us")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as fh:
        json.dump(record, fh, indent=2)
    print("wrote", args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
