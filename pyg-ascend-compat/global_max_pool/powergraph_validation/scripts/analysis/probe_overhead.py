#!/usr/bin/env python3
"""Break down where the measured ``global_max_pool`` latency comes from.

Measures, on the same real PowerGraph batch:

  A. ``torch.npu.synchronize()`` on an empty queue          (sync floor)
  B. a trivial device op, synchronised every iteration      (launch+sync floor)
  C. the frozen adapter call (compat layer)                 (full path)
  D. the raw ScatterMaxV1 bridge kernel with all buffers pre-materialised
                                                            (pure device kernel)

All numbers use ``torch.npu.Event(enable_timing=True)`` pairs plus an explicit
``torch.npu.synchronize()`` before and after each measurement window.
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

import torch_geometric.nn as tgnn  # noqa: E402

ORIGINAL_GMP = tgnn.global_max_pool

import pyg_ascend_compat  # noqa: E402

pyg_ascend_compat.enable(debug=False)

from torch_geometric.nn import global_max_pool as COMPAT_GMP  # noqa: E402

import pg_dataset  # noqa: E402

DEV = "npu:0"


def load_adapter():
    path = os.environ["PYG_ASCEND_ADAPTER_PATH"]
    spec = importlib.util.spec_from_file_location("_probe_adapter", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def timed_async(fn, warmup, iters):
    for _ in range(warmup):
        fn()
    torch.npu.synchronize()
    s = [torch.npu.Event(enable_timing=True) for _ in range(iters)]
    e = [torch.npu.Event(enable_timing=True) for _ in range(iters)]
    torch.npu.synchronize()
    t0 = time.perf_counter()
    for i in range(iters):
        s[i].record()
        fn()
        e[i].record()
    torch.npu.synchronize()
    t1 = time.perf_counter()
    return ((t1 - t0) / iters * 1e6,
            np.array([a.elapsed_time(b) * 1000.0 for a, b in zip(s, e)]))


def timed_sync(fn, warmup, iters):
    for _ in range(warmup):
        fn()
    torch.npu.synchronize()
    out = []
    for _ in range(iters):
        torch.npu.synchronize()
        t0 = time.perf_counter()
        fn()
        torch.npu.synchronize()
        out.append((time.perf_counter() - t0) * 1e6)
    return np.array(out)


def summarize(tag, host_mean, arr):
    return {
        "tag": tag,
        "host_mean_us": round(float(host_mean), 2),
        "mean_us": round(float(arr.mean()), 2),
        "p50_us": round(float(np.percentile(arr, 50)), 2),
        "p95_us": round(float(np.percentile(arr, 95)), 2),
        "p99_us": round(float(np.percentile(arr, 99)), 2),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="ieee24")
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--warmup", type=int, default=30)
    ap.add_argument("--iters", type=int, default=200)
    ap.add_argument("--data-root", default=pg_env.data_root())
    ap.add_argument("--out", default=os.path.join(
        os.environ.get("POWERGRAPH_RESULTS_ROOT", pg_env.results_root()),
        "overhead_breakdown.json"))
    args = ap.parse_args()

    torch.npu.set_device(0)
    ds = pg_dataset.get_dataset(args.data_root, args.dataset, "Binary")
    batch, bs = pg_dataset.first_batch(ds, args.batch)
    x_cpu = batch.x.float().contiguous()
    bidx_cpu = batch.batch.to(torch.int64).contiguous()
    graphs = int(bidx_cpu.max().item()) + 1
    n, f = x_cpu.shape
    x = x_cpu.to(DEV)
    bidx = bidx_cpu.to(DEV)
    y = torch.zeros(8, device=DEV)

    results = {"dataset": args.dataset, "batch": bs, "graphs": graphs,
               "nodes": n, "F": f, "warmup": args.warmup, "iters": args.iters,
               "measurements": []}

    # A. plain sync floor
    for _ in range(args.warmup):
        torch.npu.synchronize()
    out = []
    for _ in range(args.iters):
        t0 = time.perf_counter()
        torch.npu.synchronize()
        out.append((time.perf_counter() - t0) * 1e6)
    out = np.array(out)
    results["measurements"].append(summarize("A_sync_empty_queue", float(out.mean()), out))

    # B. trivial device op with sync per iteration
    a = timed_sync(lambda: torch.add(y, 1.0), args.warmup, args.iters)
    results["measurements"].append(summarize("B_trivial_op_sync_per_iter", float(a.mean()), a))

    # C. frozen compat path
    h, ev = timed_async(lambda: COMPAT_GMP(x, bidx), args.warmup, args.iters)
    results["measurements"].append(summarize("C_compat_global_max_pool", h, ev))
    sy = timed_sync(lambda: COMPAT_GMP(x, bidx), args.warmup, args.iters)
    results["measurements"].append(summarize("C2_compat_sync_per_iter", float(sy.mean()), sy))

    # D. raw bridge kernel, all buffers pre-materialised (pure device work)
    adapter = load_adapter()
    bridge = adapter._load_bridge()
    f_pad = adapter._align_up(f, adapter.FEATURE_ALIGN_ELEMS)
    x_kernel = (torch.nn.functional.pad(x, (0, f_pad - f), mode="constant",
                                        value=float("-inf")) if f_pad != f else x)
    idx32 = bidx.to(torch.int32).contiguous()
    out_kernel = torch.full((graphs, f_pad), float("-inf"), dtype=torch.float32, device=DEV)
    argmax = torch.empty((graphs, f_pad), dtype=torch.int32, device=DEV)
    h, ev = timed_async(
        lambda: bridge.scatter_max_v1_forward(x_kernel, idx32, out_kernel, argmax),
        args.warmup, args.iters)
    results["measurements"].append(summarize("D_raw_bridge_kernel_only", h, ev))

    # E. original PyG fallback (host CPU per torch_npu warning)
    h, ev = timed_async(lambda: ORIGINAL_GMP(x, bidx), args.warmup, args.iters)
    results["measurements"].append(summarize("E_original_pyg_fallback", h, ev))

    results["adapter_notes"] = {
        "F_kernel_after_padding": f_pad,
        "padded": f_pad != f,
        "host_sync_per_call": "adapter validates batch range via "
                              "torch.stack((batch.min(), batch.max())).cpu().tolist()",
    }

    for m in results["measurements"]:
        print(f"{m['tag']:<32} mean={m['mean_us']:9.2f}us p50={m['p50_us']:9.2f} "
              f"p95={m['p95_us']:9.2f} host_mean={m['host_mean_us']:9.2f}")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as fh:
        json.dump(results, fh, indent=2)
    print("wrote", args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
