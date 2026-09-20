#!/usr/bin/env python3
"""Derive the compact summary CSVs from the raw benchmark evidence.

Outputs (into POWERGRAPH_RESULTS_ROOT, default <package>/results):

  performance_summary.csv   one row per dataset x batch x dtype, compat vs original PyG
  profiler_summary.csv      one row per profiled case (parsed from the gate JSONs)

Two input modes:

  archived (default)  the committed evidence files
                      forward_phaseB_ieee24.csv + forward_phaseC.csv +
                      forward_backward_phaseC.csv, profiler gates from
                      evidence/profiler/*.gate.json
  fresh (--tag NAME)  forward_NAME.csv / forward_backward_NAME.csv and gates from
                      POWERGRAPH_PROFILE_ROOT; outputs are suffixed with _NAME so
                      the archived summaries are never overwritten.
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import os

ARCHIVED_FORWARD = ("forward_phaseB_ieee24.csv", "forward_phaseC.csv")
ARCHIVED_BACKWARD = ("forward_backward_phaseC.csv",)
DS_ORDER = ["ieee24", "ieee39", "ieee118", "uk"]
DT_ORDER = ["fp32", "fp16", "bf16"]


def jload(path, default=None):
    try:
        with open(path) as fh:
            return json.load(fh)
    except Exception:
        return default


def read_csv(path):
    if not os.path.exists(path):
        return []
    with open(path, newline="") as fh:
        return list(csv.DictReader(fh))


def sort_key(row):
    return (DS_ORDER.index(row["dataset"]) if row["dataset"] in DS_ORDER else 99,
            int(row["batch"]),
            DT_ORDER.index(row["dtype"]) if row["dtype"] in DT_ORDER else 99)


def write_performance(rows, out_path):
    compat = {(r["dataset"], r["batch"], r["dtype"]): r for r in rows
              if r.get("path") == "compat_ascend"}
    orig = {(r["dataset"], r["batch"], r["dtype"]): r for r in rows
            if r.get("path") == "original_pyg"}
    order = sorted({k for k in compat}, key=lambda k: (DS_ORDER.index(k[0]), int(k[1]),
                                                       DT_ORDER.index(k[2])))
    fields = ["dataset", "batch", "graphs", "nodes", "F", "dtype",
              "compat_status", "compat_mean_us", "compat_p50_us", "compat_p95_us",
              "compat_p99_us", "graphs_per_sec", "nodes_per_sec",
              "oracle_tol_mismatch", "oracle_ok",
              "original_pyg_status", "original_pyg_mean_us", "speedup",
              "baseline_execution_location"]
    with open(out_path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        for k in order:
            c = compat[k]
            o = orig.get(k)
            row = {
                "dataset": c["dataset"], "batch": c["batch"], "graphs": c["graphs"],
                "nodes": c["nodes"], "F": c["F"], "dtype": c["dtype"],
                "compat_status": c.get("status"),
                "compat_mean_us": c.get("mean_us"), "compat_p50_us": c.get("p50_us"),
                "compat_p95_us": c.get("p95_us"), "compat_p99_us": c.get("p99_us"),
                "graphs_per_sec": c.get("graphs_per_sec"),
                "nodes_per_sec": c.get("nodes_per_sec"),
                "oracle_tol_mismatch": c.get("oracle_tol_mismatch"),
                "oracle_ok": c.get("oracle_ok"),
                "original_pyg_status": o.get("status") if o else "not_run",
                "original_pyg_mean_us": o.get("mean_us") if o else "",
                "speedup": (round(float(o["mean_us"]) / float(c["mean_us"]), 4)
                            if o and o.get("status") == "ok" and c.get("status") == "ok"
                            and float(c["mean_us"]) > 0 else ""),
                "baseline_execution_location": "host CPU (torch_npu fallback)" if o else "",
            }
            w.writerow(row)
    return len(order)


def write_profiler(profile_dir, out_path):
    fields = ["case", "path", "scattermaxv1_tasks", "scattermaxv1_core_types",
              "scattermaxv1_dur_avg_us", "ai_cpu_task_count", "ai_cpu_task_types",
              "scatter_reduce_api_occurrences", "device_kernel_names", "gate_pass",
              "compat_total_calls", "compat_ascend_calls", "compat_original_calls",
              "compat_dtype16_forward_calls", "compat_non_fp32_passthrough"]
    n = 0
    with open(out_path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        for gpath in sorted(glob.glob(os.path.join(profile_dir, "*.gate.json"))):
            g = jload(gpath, {}) or {}
            case = g.get("tag") or os.path.basename(gpath)[:-len(".gate.json")]
            app = jload(os.path.join(profile_dir, f"{case}.json"), {}) or {}
            cs = app.get("compat_stats") or {}
            w.writerow({
                "case": case,
                "path": g.get("path"),
                "scattermaxv1_tasks": g.get("scattermaxv1_tasks"),
                "scattermaxv1_core_types": ";".join(g.get("scattermaxv1_core_types") or []),
                "scattermaxv1_dur_avg_us": g.get("scattermaxv1_dur_avg_us"),
                "ai_cpu_task_count": g.get("ai_cpu_task_count"),
                "ai_cpu_task_types": ";".join(g.get("ai_cpu_task_types") or []),
                "scatter_reduce_api_occurrences": g.get("scatter_reduce_api_occurrences"),
                "device_kernel_names": ";".join(g.get("device_kernel_names") or []),
                "gate_pass": g.get("gate_scattermaxv1_on_ai_vector_core"),
                "compat_total_calls": cs.get("total_calls", ""),
                "compat_ascend_calls": cs.get("ascend_calls", ""),
                "compat_original_calls": cs.get("original_calls", ""),
                "compat_dtype16_forward_calls": cs.get("dtype16_forward_calls", ""),
                "compat_non_fp32_passthrough": cs.get("non_fp32_passthrough", ""),
            })
            n += 1
    return n


def main() -> int:
    pkg = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir",
                    default=os.environ.get("POWERGRAPH_RESULTS_ROOT",
                                           os.path.join(pkg, "results")))
    ap.add_argument("--profile-dir",
                    default=os.environ.get("POWERGRAPH_REPORT_PROFILE_ROOT",
                                           os.path.join(pkg, "evidence", "profiler")))
    ap.add_argument("--tag", default=None,
                    help="summarise a fresh run named forward_<tag>.csv instead of the archive")
    ap.add_argument("--out-dir", default=None)
    args = ap.parse_args()

    if args.tag:
        forward_files = [os.path.join(args.results_dir, f"forward_{args.tag}.csv")]
        profile_dir = os.environ.get("POWERGRAPH_PROFILE_ROOT", args.profile_dir)
        suffix = f"_{args.tag}"
    else:
        forward_files = [os.path.join(args.results_dir, n) for n in ARCHIVED_FORWARD]
        profile_dir = args.profile_dir
        suffix = ""

    rows = []
    for f in forward_files:
        rows += read_csv(f)
    out_dir = args.out_dir or args.results_dir
    os.makedirs(out_dir, exist_ok=True)

    perf_path = os.path.join(out_dir, f"performance_summary{suffix}.csv")
    prof_path = os.path.join(out_dir, f"profiler_summary{suffix}.csv")

    n_perf = write_performance(rows, perf_path) if rows else 0
    n_prof = write_profiler(profile_dir, prof_path)
    print(f"forward rows read : {len(rows)}")
    print(f"wrote {perf_path} ({n_perf} rows)")
    print(f"wrote {prof_path} ({n_prof} cases) from {profile_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
