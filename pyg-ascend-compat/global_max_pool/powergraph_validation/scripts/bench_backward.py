#!/usr/bin/env python3
"""Optional table: forward + first-order backward for the frozen Ascend path.

Kept strictly separate from the forward results (separate CSV / separate Markdown
table) as required.  Each iteration is::

    x.grad = None                    # no gradient accumulation
    out  = global_max_pool(x, batch) # x.requires_grad_(True)
    loss = out.sum()
    loss.backward()
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
import traceback

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pg_env  # noqa: E402
from bench_forward import (COMPAT_GMP, DTYPES, ORIGINAL_GMP, _stats,  # noqa: E402
                           os_env_record, time_async, time_sync)
import pg_dataset  # noqa: E402

DEV = "npu:0"


def make_fwdbwd(path: str, x, bidx):
    gmp = COMPAT_GMP if path == "compat_ascend" else ORIGINAL_GMP

    def run():
        x.grad = None
        out = gmp(x, bidx)
        out.sum().backward()

    return run


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", default="ieee24")
    ap.add_argument("--batch-sizes", default="1,8,32,128")
    ap.add_argument("--dtypes", default="fp32,fp16,bf16")
    ap.add_argument("--paths", default="compat_ascend")
    ap.add_argument("--warmup", type=int, default=30)
    ap.add_argument("--iters", type=int, default=200)
    ap.add_argument("--skip-sync-loop", action="store_true")
    ap.add_argument("--data-root", default=pg_env.data_root())
    ap.add_argument("--tag", default="phaseB")
    ap.add_argument("--results-dir",
                    default=os.environ.get("POWERGRAPH_RESULTS_ROOT", pg_env.results_root()))
    args = ap.parse_args()

    out_csv = os.path.join(args.results_dir, f"forward_backward_{args.tag}.csv")
    per_iter_csv = os.path.join(args.results_dir, f"forward_backward_per_iter_{args.tag}.csv")
    os.makedirs(args.results_dir, exist_ok=True)

    datasets = [d for d in args.datasets.split(",") if d]
    batch_sizes = [int(b) for b in args.batch_sizes.split(",") if b]
    dtypes = [d for d in args.dtypes.split(",") if d]
    paths = [p for p in args.paths.split(",") if p]

    env = os_env_record()
    env.update({"warmup": args.warmup, "iters": args.iters, "mode": "forward+first-order backward",
                "batch_sizes": batch_sizes, "dtypes": dtypes, "datasets": datasets, "paths": paths})
    with open(os.path.join(args.results_dir, f"forward_backward_{args.tag}_env.json"), "w") as fh:
        json.dump(env, fh, indent=2)

    rows, per_iter_rows = [], []
    for ds_name in datasets:
        print(f"\n[bench-fb] === dataset {ds_name} ===", flush=True)
        dataset = pg_dataset.get_dataset(args.data_root, ds_name, "Binary")
        for bs in batch_sizes:
            batch, bs_eff = pg_dataset.first_batch(dataset, bs)
            x_cpu = batch.x.float().contiguous()
            bidx_cpu = batch.batch.to(torch.int64).contiguous()
            num_graphs = int(bidx_cpu.max().item()) + 1
            total_nodes, feat = x_cpu.shape
            bidx = bidx_cpu.to(DEV)
            print(f"[bench-fb] batch={bs} graphs={num_graphs} nodes={total_nodes} F={feat}",
                  flush=True)

            for dtype_name in dtypes:
                x = x_cpu.to(DEV).to(DTYPES[dtype_name]).detach().requires_grad_(True)
                for path in paths:
                    fn = make_fwdbwd(path, x, bidx)
                    row = {"dataset": ds_name, "batch": bs, "graphs": num_graphs,
                           "nodes": total_nodes, "F": feat, "dtype": dtype_name,
                           "path": path, "warmup": args.warmup, "iters": args.iters}
                    try:
                        fn()
                        torch.npu.synchronize()
                        row["grad_finite_ok"] = bool(torch.isfinite(x.grad).all().item())
                        row["grad_nonzero_count"] = int((x.grad != 0).sum().item())
                        host_total, ev = time_async(fn, args.warmup, args.iters)
                        st = _stats(ev)
                        row.update({
                            "host_total_us": host_total,
                            "mean_us": st["mean"], "p50_us": st["p50"], "p95_us": st["p95"],
                            "p99_us": st["p99"], "min_us": st["min"], "max_us": st["max"],
                            "std_us": st["std"],
                            "graphs_per_sec": num_graphs / (st["mean"] * 1e-6),
                            "nodes_per_sec": total_nodes / (st["mean"] * 1e-6),
                            "status": "ok",
                        })
                        for i, v in enumerate(ev):
                            per_iter_rows.append([ds_name, bs, dtype_name, path,
                                                  "async_event", i, v])
                        if not args.skip_sync_loop:
                            sy = time_sync(fn, args.warmup, args.iters)
                            sst = _stats(sy)
                            row.update({"sync_mean_us": sst["mean"], "sync_p50_us": sst["p50"],
                                        "sync_p95_us": sst["p95"], "sync_p99_us": sst["p99"]})
                            for i, v in enumerate(sy):
                                per_iter_rows.append([ds_name, bs, dtype_name, path, "sync", i, v])
                        print(f"[bench-fb]   {dtype_name:>4} {path:<15} mean={st['mean']:9.2f}us "
                              f"p50={st['p50']:9.2f} p95={st['p95']:9.2f} p99={st['p99']:9.2f}",
                              flush=True)
                    except Exception as exc:
                        row.update({"status": f"FAILED: {type(exc).__name__}: {exc}",
                                    "traceback": traceback.format_exc()[-2000:]})
                        print(f"[bench-fb]   {dtype_name:>4} {path:<15} FAILED: {exc}", flush=True)
                    rows.append(row)
                del x
            del batch, x_cpu, bidx_cpu, bidx
            torch.npu.empty_cache()
        del dataset

    fields = ["dataset", "batch", "graphs", "nodes", "F", "dtype", "path", "warmup", "iters",
              "status", "mean_us", "p50_us", "p95_us", "p99_us", "min_us", "max_us", "std_us",
              "host_total_us", "graphs_per_sec", "nodes_per_sec",
              "sync_mean_us", "sync_p50_us", "sync_p95_us", "sync_p99_us",
              "grad_finite_ok", "grad_nonzero_count", "traceback"]
    with open(out_csv, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in fields})
    with open(per_iter_csv, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["dataset", "batch", "dtype", "path", "loop", "iter", "latency_us"])
        w.writerows(per_iter_rows)
    print(f"\n[bench-fb] wrote {out_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
