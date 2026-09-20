#!/usr/bin/env python3
"""Parse an msprof PROF_* directory into a gate record for one case."""

from __future__ import annotations

import csv
import glob
import json
import os
import re
import sys


def main() -> int:
    prof, tag, path_kind = sys.argv[1], sys.argv[2], sys.argv[3]
    out_json = sys.argv[4]

    rows = []
    for p in glob.glob(os.path.join(prof, "mindstudio_profiler_output", "op_summary_*.csv")):
        with open(p, newline="") as fh:
            rows += list(csv.DictReader(fh))

    ops: dict = {}
    for r in rows:
        name = (r.get("OP Type") or "").strip()
        if not name:
            continue
        rec = ops.setdefault(name, {"n": 0, "types": set(), "dur": []})
        rec["n"] += 1
        rec["types"].add((r.get("Task Type") or "").strip())
        d = r.get("Task Duration(us)")
        if d:
            try:
                rec["dur"].append(float(d))
            except ValueError:
                pass

    sc = ops.get("ScatterMaxV1")
    core_types = sorted(sc["types"]) if sc else []
    ai_cpu_types = sorted({t for o in ops.values() for t in o["types"] if "CPU" in t.upper()})
    ai_cpu_count = sum(o["n"] for o in ops.values() for t in o["types"] if "CPU" in t.upper())

    apicnt = 0
    apis = set()
    for p in glob.glob(os.path.join(prof, "mindstudio_profiler_output", "api_statistic_*.csv")):
        with open(p, newline="") as fh:
            for r in csv.DictReader(fh):
                api = str(r.get("API Name") or r.get("Name") or "")
                if "scatter_reduce" in api:
                    apicnt += 1
                    apis.add(api)

    kernel_names = set()
    for p in glob.glob(os.path.join(prof, "host", "data", "*")):
        try:
            blob = open(p, "rb").read().decode("utf-8", "ignore")
        except Exception:
            continue
        kernel_names |= set(re.findall(r"ScatterMaxV1_[0-9a-f]{32}_[0-9]+", blob))

    record = {
        "tag": tag,
        "path": path_kind,
        "prof_dir": prof,
        "op_types": {
            k: {
                "count": v["n"],
                "task_types": sorted(v["types"]),
                "dur_avg_us": round(sum(v["dur"]) / len(v["dur"]), 2) if v["dur"] else None,
            }
            for k, v in sorted(ops.items())
        },
        "scattermaxv1_tasks": sc["n"] if sc else 0,
        "scattermaxv1_core_types": core_types,
        "scattermaxv1_dur_avg_us": round(sum(sc["dur"]) / len(sc["dur"]), 2)
        if sc and sc["dur"] else None,
        "scattermaxv1_dur_min_us": min(sc["dur"]) if sc and sc["dur"] else None,
        "scattermaxv1_dur_max_us": max(sc["dur"]) if sc and sc["dur"] else None,
        "ai_cpu_task_types": ai_cpu_types,
        "ai_cpu_task_count": ai_cpu_count,
        "scatter_reduce_api_occurrences": apicnt,
        "scatter_reduce_api_names": sorted(apis),
        "device_kernel_names": sorted(kernel_names),
        "gate_scattermaxv1_on_ai_vector_core": bool(sc) and core_types == ["AI_VECTOR_CORE"],
    }
    with open(out_json, "w") as fh:
        json.dump(record, fh, indent=2)

    print("GATE ScatterMaxV1 -> AI_VECTOR_CORE:",
          "PASS" if record["gate_scattermaxv1_on_ai_vector_core"] else "N/A")
    print(f"  scattermaxv1_tasks={record['scattermaxv1_tasks']} core_types={core_types} "
          f"avg={record['scattermaxv1_dur_avg_us']}us")
    print(f"  AI_CPU_task_types={ai_cpu_types} count={ai_cpu_count}")
    print(f"  aten::scatter_reduce occurrences = {apicnt} {sorted(apis)}")
    print(f"  device kernel names = {sorted(kernel_names)[:4]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
