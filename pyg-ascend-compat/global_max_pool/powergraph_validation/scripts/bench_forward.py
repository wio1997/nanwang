#!/usr/bin/env python3
"""PowerGraph real-batch forward benchmark of the frozen Ascend ``global_max_pool``.

For every ``dataset x batch_size x dtype`` this measures, on real PyG DataLoader
batches built from the PowerGraph graph-level datasets:

  * ``compat_ascend``  — ``torch_geometric.nn.global_max_pool`` after
    ``pyg_ascend_compat.enable()`` (frozen ScatterMaxV1 path)
  * ``original_pyg``   — the untouched upstream PyG implementation captured
    *before* the compat layer was enabled

Timing follows the NPU asynchronous-execution rule:

  * ``warmup``    calls before any measurement
  * ``torch.npu.synchronize()`` before ``t0`` and after the last launch
  * per-iteration ``torch.npu.Event(enable_timing=True)`` pairs, so the reported
    percentiles are device-side per-op durations
  * an additional fully-synchronised loop (sync before and after every single
    call) reported separately as ``sync_*`` — it additionally contains the host
    launch and synchronisation overhead
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import platform
import socket
import sys
import time
import traceback

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pg_env  # noqa: E402

# --- capture the upstream PyG implementation BEFORE enabling the compat layer ---
import torch_geometric.nn as tgnn  # noqa: E402

ORIGINAL_GMP = tgnn.global_max_pool

import pyg_ascend_compat  # noqa: E402

pyg_ascend_compat.enable(debug=False)

from torch_geometric.nn import global_max_pool as COMPAT_GMP  # noqa: E402

import pg_dataset  # noqa: E402

DEV = "npu:0"
DTYPES = {
    "fp32": torch.float32,
    "fp16": torch.float16,
    "bf16": torch.bfloat16,
}


# --------------------------------------------------------------------------- timing
def _stats(samples_us):
    a = np.asarray(samples_us, dtype=np.float64)
    return {
        "mean": float(a.mean()),
        "p50": float(np.percentile(a, 50)),
        "p95": float(np.percentile(a, 95)),
        "p99": float(np.percentile(a, 99)),
        "min": float(a.min()),
        "max": float(a.max()),
        "std": float(a.std(ddof=1)) if a.size > 1 else 0.0,
    }


def time_async(fn, warmup: int, iters: int):
    """Back-to-back launches with device event timing around each call."""
    for _ in range(warmup):
        fn()
    torch.npu.synchronize()
    starts = [torch.npu.Event(enable_timing=True) for _ in range(iters)]
    ends = [torch.npu.Event(enable_timing=True) for _ in range(iters)]
    torch.npu.synchronize()
    t0 = time.perf_counter()
    for i in range(iters):
        starts[i].record()
        fn()
        ends[i].record()
    torch.npu.synchronize()
    t1 = time.perf_counter()
    per_iter = [s.elapsed_time(e) * 1000.0 for s, e in zip(starts, ends)]  # ms -> us
    return (t1 - t0) / iters * 1e6, per_iter


def time_sync(fn, warmup: int, iters: int):
    """Synchronise around every single call (includes host launch overhead)."""
    for _ in range(warmup):
        fn()
    torch.npu.synchronize()
    per_iter = []
    for _ in range(iters):
        torch.npu.synchronize()
        t0 = time.perf_counter()
        fn()
        torch.npu.synchronize()
        per_iter.append((time.perf_counter() - t0) * 1e6)
    return per_iter


# --------------------------------------------------------------------------- helpers
def make_fn(path: str, x, bidx):
    if path == "compat_ascend":
        return lambda: COMPAT_GMP(x, bidx)
    if path == "original_pyg":
        return lambda: ORIGINAL_GMP(x, bidx)
    raise ValueError(path)


def sanity(out, x_cpu, bidx_cpu, num_graphs, feat, dtype_name, oracle=None):
    rec = {
        "out_shape": str(tuple(out.shape)),
        "shape_ok": bool(tuple(out.shape) == (num_graphs, feat)),
        "nan_count": int(torch.isnan(out).sum().item()),
        "inf_count": int(torch.isinf(out).sum().item()),
    }
    rec["finite_ok"] = (rec["nan_count"] == 0 and rec["inf_count"] == 0)
    if oracle is not None:
        got = out.detach().float().cpu()
        diff = (got - oracle).abs()
        rec["oracle_max_abs_diff"] = float(diff.max().item())
        rec["oracle_exact_mismatch"] = int((diff > 0).sum().item())
        denom = oracle.abs().clamp_min(1e-12)
        rec["oracle_max_rel_diff"] = float((diff / denom).max().item())
        rtol = {"fp32": 1e-5, "fp16": 2e-3, "bf16": 2e-2}[dtype_name]
        atol = {"fp32": 1e-6, "fp16": 1e-2, "bf16": 1e-1}[dtype_name]
        ok = torch.isclose(got, oracle, rtol=rtol, atol=atol)
        rec["oracle_rtol"] = rtol
        rec["oracle_atol"] = atol
        rec["oracle_tol_mismatch"] = int((~ok).sum().item())
        rec["oracle_ok"] = rec["oracle_tol_mismatch"] == 0
    return rec


def os_env_record():
    return {
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "python": sys.version.split()[0],
        "torch": torch.__version__,
        "torch_npu": __import__("torch_npu").__version__,
        "pyg": __import__("torch_geometric").__version__,
        "numpy": np.__version__,
        "device": torch.npu.get_device_name(0),
        "ascend_custom_opp_path": os.environ.get("ASCEND_CUSTOM_OPP_PATH"),
        "scattermaxv1_bridge": os.environ.get("SCATTERMAXV1_BRIDGE"),
        "adapter_path": pyg_ascend_compat.stats().get("adapter_path"),
        "compat_version": pyg_ascend_compat.__version__,
    }


# --------------------------------------------------------------------------- main
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", default="ieee24")
    ap.add_argument("--batch-sizes", default="1,8,32,128")
    ap.add_argument("--dtypes", default="fp32")
    ap.add_argument("--paths", default="compat_ascend,original_pyg")
    ap.add_argument("--warmup", type=int, default=30)
    ap.add_argument("--iters", type=int, default=200)
    ap.add_argument("--skip-sync-loop", action="store_true")
    ap.add_argument("--skip-baseline", action="store_true")
    ap.add_argument("--data-root", default=pg_env.data_root())
    ap.add_argument("--tag", default="phaseB")
    ap.add_argument("--results-dir",
                    default=os.environ.get("POWERGRAPH_RESULTS_ROOT", pg_env.results_root()))
    args = ap.parse_args()

    out_csv = os.path.join(args.results_dir, f"forward_{args.tag}.csv")
    per_iter_csv = os.path.join(args.results_dir, f"forward_per_iter_{args.tag}.csv")
    meta_json = os.path.join(args.results_dir, f"forward_{args.tag}_env.json")
    os.makedirs(args.results_dir, exist_ok=True)

    datasets = [d for d in args.datasets.split(",") if d]
    batch_sizes = [int(b) for b in args.batch_sizes.split(",") if b]
    dtype_names = [d for d in args.dtypes.split(",") if d]
    paths = [p for p in args.paths.split(",") if p]
    if args.skip_baseline:
        paths = [p for p in paths if p != "original_pyg"]

    env = os_env_record()
    env.update({"warmup": args.warmup, "iters": args.iters, "batch_sizes": batch_sizes,
                "dtypes": dtype_names, "datasets": datasets, "paths": paths,
                "timing": "torch.npu.Event(enable_timing=True) per iteration; "
                          "torch.npu.synchronize() before t0 and after last launch"})
    with open(meta_json, "w") as fh:
        json.dump(env, fh, indent=2)
    print("[bench] env:", json.dumps(env, indent=2), flush=True)

    rows, per_iter_rows = [], []
    workload = {}

    for ds_name in datasets:
        print(f"\n[bench] === dataset {ds_name} ===", flush=True)
        t_ds = time.time()
        dataset = pg_dataset.get_dataset(args.data_root, ds_name, "Binary")
        print(f"[bench] loaded/processed {ds_name} in {time.time()-t_ds:.1f}s "
              f"(graphs={dataset.len()})", flush=True)
        workload[ds_name] = pg_dataset.dataset_workload_stats(dataset)
        print("[bench] workload:", json.dumps(workload[ds_name]), flush=True)

        for bs in batch_sizes:
            batch, bs_eff = pg_dataset.first_batch(dataset, bs)
            x_cpu = batch.x.float().contiguous()
            bidx_cpu = batch.batch.to(torch.int64).contiguous()
            num_graphs = int(bidx_cpu.max().item()) + 1
            total_nodes, feat = x_cpu.shape
            x_npu32 = x_cpu.to(DEV)
            bidx_npu = bidx_cpu.to(DEV)
            print(f"[bench] batch={bs} (effective {bs_eff}) graphs={num_graphs} "
                  f"nodes={total_nodes} F={feat}", flush=True)

            # CPU oracle, computed once per (dataset, batch)
            oracle = ORIGINAL_GMP(x_cpu, bidx_cpu)

            for dtype_name in dtype_names:
                tdt = DTYPES[dtype_name]
                x_npu = x_npu32.to(tdt)

                for path in paths:
                    fn = make_fn(path, x_npu, bidx_npu)
                    row = {
                        "dataset": ds_name,
                        "batch": bs,
                        "graphs": num_graphs,
                        "nodes": total_nodes,
                        "F": feat,
                        "dtype": dtype_name,
                        "path": path,
                        "warmup": args.warmup,
                        "iters": args.iters,
                    }
                    try:
                        with torch.no_grad():
                            out = fn()
                            torch.npu.synchronize()
                        row.update(sanity(out, x_cpu, bidx_cpu, num_graphs, feat,
                                          dtype_name, oracle))
                        del out

                        host_total, ev = time_async(fn, args.warmup, args.iters)
                        st = _stats(ev)
                        row.update({
                            "host_total_us": host_total,
                            "mean_us": st["mean"], "p50_us": st["p50"],
                            "p95_us": st["p95"], "p99_us": st["p99"],
                            "min_us": st["min"], "max_us": st["max"], "std_us": st["std"],
                            "graphs_per_sec": num_graphs / (st["mean"] * 1e-6),
                            "nodes_per_sec": total_nodes / (st["mean"] * 1e-6),
                            "status": "ok",
                        })
                        for i, v in enumerate(ev):
                            per_iter_rows.append([ds_name, bs, dtype_name, path, "async_event", i, v])

                        if not args.skip_sync_loop:
                            sy = time_sync(fn, args.warmup, args.iters)
                            sst = _stats(sy)
                            row.update({
                                "sync_mean_us": sst["mean"], "sync_p50_us": sst["p50"],
                                "sync_p95_us": sst["p95"], "sync_p99_us": sst["p99"],
                                "sync_min_us": sst["min"], "sync_max_us": sst["max"],
                            })
                            for i, v in enumerate(sy):
                                per_iter_rows.append([ds_name, bs, dtype_name, path, "sync", i, v])
                        print(f"[bench]   {dtype_name:>4} {path:<15} mean={st['mean']:9.2f}us "
                              f"p50={st['p50']:9.2f} p95={st['p95']:9.2f} p99={st['p99']:9.2f} "
                              f"host={host_total:9.2f}us", flush=True)
                    except Exception as exc:  # keep going, record the failure honestly
                        row.update({"status": f"FAILED: {type(exc).__name__}: {exc}",
                                    "traceback": traceback.format_exc()[-2000:]})
                        print(f"[bench]   {dtype_name:>4} {path:<15} FAILED: {exc}", flush=True)
                    rows.append(row)

            del batch, x_cpu, bidx_cpu, x_npu32, bidx_npu, oracle
            torch.npu.empty_cache()

        del dataset

    fields = ["dataset", "batch", "graphs", "nodes", "F", "dtype", "path", "warmup", "iters",
              "status", "mean_us", "p50_us", "p95_us", "p99_us", "min_us", "max_us", "std_us",
              "host_total_us", "graphs_per_sec", "nodes_per_sec",
              "sync_mean_us", "sync_p50_us", "sync_p95_us", "sync_p99_us",
              "sync_min_us", "sync_max_us",
              "out_shape", "shape_ok", "nan_count", "inf_count", "finite_ok",
              "oracle_max_abs_diff", "oracle_exact_mismatch", "oracle_tol",
              "oracle_max_rel_diff", "oracle_rtol", "oracle_atol",
              "oracle_tol_mismatch", "oracle_ok", "traceback"]

    with open(out_csv, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in fields})
    with open(per_iter_csv, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["dataset", "batch", "dtype", "path", "loop", "iter", "latency_us"])
        w.writerows(per_iter_rows)

    with open(os.path.join(args.results_dir, f"forward_{args.tag}_workload.json"), "w") as fh:
        json.dump(workload, fh, indent=2)

    print(f"\n[bench] wrote {out_csv}")
    print(f"[bench] wrote {per_iter_csv}")
    print("[bench] compat stats:", pyg_ascend_compat.stats())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
