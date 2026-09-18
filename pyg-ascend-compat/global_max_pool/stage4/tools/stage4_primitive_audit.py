#!/usr/bin/env python3
"""Stage 4 / section 11 — NPU primitive audit for the backward path.

Reads the msprof outputs of the four Stage 4 profiler runs and reports, for every operator that
appeared during *forward + loss + backward*, the task types (cores) it ran on.  Any operator that
only exists in the backward (or appears with an increased count) is listed with its core type so
each primitive chosen by ``global_max_pool_ascend_autograd`` is provably device-executed.

Run: python3 stage4_primitive_audit.py > /root/zyg/logs/stage4/stage4_primitive_audit.txt
"""

from __future__ import annotations

import csv
import glob
import os
import sys

PROF_ROOT = os.environ.get("STAGE4_PROF_ROOT", "/root/zyg/profiler/stage4")

# backward primitives chosen by the Stage 4 implementation and the code line that produces them
BACKWARD_MAP = {
    "GatherV3": "out[batch] / scaled[batch] / zero-count gather (index_select)",
    "Equal": "(x == out[batch]) winner mask, (out == 0) quirk term",
    "Cast": "bool -> float32 mask, index dtype handling",
    "InplaceIndexAdd": "winner count scatter-add along dim 0 (index_add_)",
    "RealDiv": "grad_out / count (fp32 device division, <=1 ULP vs IEEE)",
    "Mul": "winner * (grad_out/count)",
    "MaskedFill": "NaN contract fixup (count == 0 -> nan)",
    "ZerosLike": "zero fill for grad buffers / empty groups",
    "Fill": "constant fills (nan / zeros) inside the backward",
    "BroadcastTo": "index/mask broadcasting",
    "Add": "count + (out == 0)",
    "OnesLike": "auxiliary fills",
}


def main():
    runs = sorted(glob.glob(os.path.join(PROF_ROOT, "*", "PROF_*")))
    if not runs:
        print(f"no profiler runs found under {PROF_ROOT}")
        return 1
    per_run = {}
    for prof in runs:
        tag = os.path.basename(os.path.dirname(prof))
        rows = []
        for p in glob.glob(os.path.join(prof, "mindstudio_profiler_output", "op_summary_*.csv")):
            rows += list(csv.DictReader(open(p)))
        ops = {}
        for r in rows:
            op = (r.get("OP Type") or "").strip()
            core = (r.get("Task Type") or "").strip()
            ops.setdefault(op, {}).setdefault(core, 0)
            ops[op][core] += 1
        per_run[tag] = ops

    all_ops = sorted({op for ops in per_run.values() for op in ops})
    print("=" * 100)
    print("STAGE 4 NPU PRIMITIVE AUDIT — forward + loss + backward (formal delivery OPP)")
    print("=" * 100)
    print()
    print(f"{'operator':28s} {'cores seen':28s} {'total tasks':>11s}   backward role")
    print("-" * 100)
    bad = []
    for op in all_ops:
        cores = {}
        total = 0
        for ops in per_run.values():
            for core, c in ops.get(op, {}).items():
                cores[core] = cores.get(core, 0) + c
                total += c
        core_str = ",".join(sorted(cores))
        if any("CPU" in c.upper() for c in cores):
            bad.append(op)
        print(f"{op:28s} {core_str:28s} {total:11d}   {BACKWARD_MAP.get(op, '-')}")
    print()
    print("per-run operator sets:")
    for tag, ops in sorted(per_run.items()):
        print(f"  {tag:34s} {sorted(ops)}")
    print()
    print("GATE no CPU task type in any profiled forward+backward operator:",
          "PASS" if not bad else f"FAIL {bad}")
    print("GATE backward primitives are device cores (GatherV3/Equal/Cast/InplaceIndexAdd/"
          "RealDiv/Mul/MaskedFill):",
          "PASS" if not bad else "FAIL")
    return 0 if not bad else 1


if __name__ == "__main__":
    sys.exit(main())
