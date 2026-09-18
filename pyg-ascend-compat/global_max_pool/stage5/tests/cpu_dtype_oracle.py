#!/usr/bin/env python3
"""Stage 5A — real CPU oracle for FP16 and BF16 ``global_max_pool`` (forward + backward).

Installed stack only: python 3.11.14, torch 2.9.0+cpu, PyG 2.8.0.post1, public
``torch_geometric.nn.global_max_pool``.  Nothing is inferred from the FP32 oracle: every number
below is measured for the dtype itself.

Run: python3 cpu_dtype_oracle.py [--json out.json]
"""

from __future__ import annotations

import argparse
import json
import math
import os
import platform

import torch
import torch_geometric
from torch_geometric.nn import global_max_pool

DTYPES = {"fp16": torch.float16, "bf16": torch.bfloat16}
RESULTS = []


def f(v):
    if isinstance(v, float):
        if math.isnan(v):
            return "nan"
        if v == float("inf"):
            return "+inf"
        if v == float("-inf"):
            return "-inf"
        return f"{v!r}"
    return str(v)


def fmt(t):
    if t is None:
        return "None"
    d = t.tolist()
    if not isinstance(d[0], list):
        return "[" + ", ".join(f(v) for v in d) + "]"
    return "[" + ", ".join("[" + ", ".join(f(v) for v in row) + "]" for row in d) + "]"


def case(dt_name, name, xv, batch=None, size=None, upstream=None, oned=False, note=""):
    dt = DTYPES[dt_name]
    x = torch.tensor(xv, dtype=dt)
    if oned:
        x = x.reshape(-1)
    b = None if batch is None else torch.tensor(batch, dtype=torch.int64)
    up = None if upstream is None else torch.tensor(upstream, dtype=dt)
    xr = x.clone().requires_grad_(True)
    rec = {"dtype": dt_name, "case": name, "note": note, "x": x.tolist(),
           "batch": None if b is None else b.tolist(), "size": size,
           "upstream": None if up is None else up.tolist()}
    try:
        out = global_max_pool(xr, b, size)
        loss = out.sum() if up is None else (out * up).sum()
        loss.backward()
        rec.update({"ok": True, "out": out.detach().tolist(), "out_dtype": str(out.dtype),
                    "grad": xr.grad.tolist(), "grad_dtype": str(xr.grad.dtype)})
    except Exception as exc:  # noqa: BLE001
        rec.update({"ok": False, "error": f"{type(exc).__name__}: {exc}"})
    RESULTS.append(rec)
    print(f"[{dt_name}] {name:34s} " + (f"out={fmt(out.detach())} grad={fmt(xr.grad)} "
                                        f"dtypes={out.dtype}/{xr.grad.dtype}"
                                        if rec["ok"] else f"ERROR {rec.get('error')}"))
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", default="/root/zyg/logs/stage5/cpu_dtype_oracle.json")
    args = ap.parse_args()

    print("=" * 100)
    print("STAGE 5 CPU DTYPE ORACLE — real PyG API on CPU (no assumptions from FP32)")
    print(f"  python {platform.python_version()} | torch {torch.__version__} | "
          f"PyG {torch_geometric.__version__}")
    print("=" * 100)

    for dt in DTYPES:
        print(f"\n########## {dt} ##########")
        case(dt, "unique_max", [[1.0, 10.0], [3.0, 20.0], [2.0, 30.0]], [0, 0, 0], 1)
        case(dt, "two_way_tie", [[3.0, 1.0], [3.0, 2.0], [1.0, 3.0]], [0, 0, 0], 1)
        case(dt, "three_way_tie", [[5.0, 5.0], [5.0, 5.0], [5.0, 5.0]], [0, 0, 0], 1)
        case(dt, "per_feature_ties", [[5.0, 1.0, 7.0], [5.0, 3.0, 7.0], [2.0, 3.0, 0.0]],
             [0, 0, 0], 1)
        case(dt, "weighted_pos_upstream", [[3.0, 1.0], [3.0, 2.0], [1.0, 3.0]], [0, 0, 0], 1,
             upstream=[[2.0, 4.0]])
        case(dt, "weighted_neg_upstream", [[3.0, 1.0], [3.0, 2.0], [1.0, 3.0]], [0, 0, 0], 1,
             upstream=[[-2.0, -4.0]])
        case(dt, "zero_upstream", [[3.0, 1.0], [3.0, 2.0], [1.0, 3.0]], [0, 0, 0], 1,
             upstream=[[0.0, 0.0]])
        case(dt, "multiple_groups",
             [[4.0, 1.0], [4.0, 2.0], [0.0, 9.0], [7.0, 7.0], [7.0, 7.0], [1.0, 0.0]],
             [0, 0, 0, 1, 1, 1], 2, upstream=[[1.0, 2.0], [3.0, 4.0]])
        case(dt, "explicit_size_empty", [[4.0, 1.0], [4.0, 2.0], [0.0, 9.0]], [0, 0, 3], 5,
             upstream=[[1.0, 1.0], [5.0, 5.0], [7.0, 7.0], [11.0, 11.0], [13.0, 13.0]])
        case(dt, "negative_only", [[-5.0], [-2.0], [-2.0]], [0, 0, 0], 1)
        case(dt, "all_neg_inf", [[float("-inf")], [float("-inf")]], [0, 0], 1)
        case(dt, "neg_inf_mixed", [[float("-inf")], [float("-inf")], [-1.0]], [0, 0, 0], 1)
        case(dt, "pos_inf_tie", [[float("inf")], [float("inf")], [1.0]], [0, 0, 0], 1)
        case(dt, "pm_zero", [[0.0], [-0.0]], [0, 0], 1)
        case(dt, "pm_zero_mixed", [[0.0], [-0.0], [-1.0]], [0, 0, 0], 1)
        case(dt, "single_zero", [[0.0]], [0], 1)
        case(dt, "zero_max_group", [[4.0], [4.0], [0.0]], [0, 0, 3], 5,
             upstream=[[1.0], [5.0], [7.0], [11.0], [13.0]])
        case(dt, "one_nan", [[float("nan")], [1.0], [2.0]], [0, 0, 0], 1)
        case(dt, "two_nan", [[float("nan")], [float("nan")], [2.0]], [0, 0, 0], 1)
        case(dt, "nan_and_inf", [[float("nan")], [float("inf")], [1.0]], [0, 0, 0], 1)
        case(dt, "batch_none_unique", [[1.0, 10.0], [3.0, 20.0], [2.0, 30.0]], None, None)
        case(dt, "batch_none_tie", [[3.0, 1.0], [3.0, 2.0], [1.0, 3.0]], None, None)
        case(dt, "batch_none_1d", [1.0, 3.0, 2.0], None, None, oned=True)
        # precision-sensitive tie denominators (do they divide in the dtype or in fp32?)
        big = [[1.0], ] * 122
        case(dt, "tie_122", big, [0] * 122, 1, note="1/122: fp32-divide-then-cast vs dtype-divide")
        big3 = [[1.0], ] * 3
        case(dt, "tie_3", big3, [0, 0, 0], 1)
        big7 = [[1.0], ] * 7
        case(dt, "tie_7", big7, [0] * 7, 1)

    os.makedirs(os.path.dirname(args.json), exist_ok=True)
    with open(args.json, "w") as fh:
        json.dump(RESULTS, fh, indent=1)
    print(f"\n[json] {args.json}")

    print("\n=== summary: output/grad dtypes ===")
    for dt in DTYPES:
        ok = [r for r in RESULTS if r["dtype"] == dt and r.get("ok")]
        print(f"  {dt}: {len(ok)} cases OK, out_dtype={sorted({r['out_dtype'] for r in ok})}, "
              f"grad_dtype={sorted({r['grad_dtype'] for r in ok})}")


if __name__ == "__main__":
    main()
