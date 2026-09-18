#!/usr/bin/env python3
"""Stage 4 / G0-G10 + batch=None: the REAL CPU gradient oracle.

Uses the installed PyTorch 2.9.0 and PyG 2.8.0.post1 on CPU and the public API
``torch_geometric.nn.global_max_pool`` only.  No assumptions, no documentation: whatever this
script prints is the contract the Ascend backward has to match.

Run:  python3 cpu_gradient_oracle.py [--json out.json]
"""

from __future__ import annotations

import argparse
import json
import math
import platform

import torch
import torch_geometric
from torch_geometric.nn import global_max_pool

RESULTS = []


def fmt(t):
    if t is None:
        return "None"
    data = t.tolist()
    if not isinstance(data[0], list):
        return "[" + ", ".join(f(v) for v in data) + "]"
    return "[" + ", ".join("[" + ", ".join(f(v) for v in row) + "]" for row in data) + "]"


def fmt_exact(t):
    """Full-precision values (repr) so no contract detail is lost to formatting."""
    if t is None:
        return "None"
    data = t.tolist()
    if not isinstance(data[0], list):
        return "[" + ", ".join(repr(v) for v in data) + "]"
    return "[" + ", ".join("[" + ", ".join(repr(v) for v in row) + "]" for row in data) + "]"


def f(v):
    if isinstance(v, float):
        if math.isnan(v):
            return "nan"
        if v == float("inf"):
            return "+inf"
        if v == float("-inf"):
            return "-inf"
        return f"{v:g}"
    return str(v)


def case(name, x, batch=None, size=None, upstream=None, note="", oned=False):
    """Run one CPU oracle case: real PyG forward + backward, record out/grad exactly."""
    x = torch.tensor(x, dtype=torch.float32)
    if oned:
        x = x.reshape(-1)
    b = None if batch is None else torch.tensor(batch, dtype=torch.int64)
    up = None if upstream is None else torch.tensor(upstream, dtype=torch.float32)
    xr = x.clone().requires_grad_(True)
    warn = ""
    try:
        out = global_max_pool(xr, b, size)
        loss = out.sum() if up is None else (out * up).sum()
        loss.backward()
        grad = xr.grad
    except Exception as exc:  # noqa: BLE001
        out, grad, warn = None, None, f"{type(exc).__name__}: {exc}"
    rec = {
        "case": name,
        "note": note,
        "x_shape": list(x.shape),
        "batch": None if b is None else b.tolist(),
        "size": size,
        "upstream": None if up is None else up.tolist(),
        "out": None if out is None else out.detach().tolist(),
        "grad": None if grad is None else grad.tolist(),
        "error": warn,
    }
    RESULTS.append(rec)
    print(f"--- {name}")
    print(f"    x        : {fmt(x)}")
    if b is not None:
        print(f"    batch    : {b.tolist()}   size={size}")
    else:
        print("    batch    : None")
    if up is not None:
        print(f"    upstream : {fmt(up)}")
    if warn:
        print(f"    ERROR    : {warn}")
    else:
        print(f"    forward  : {fmt(out.detach())}")
        print(f"    x.grad   : {fmt(grad)}")
        print(f"    x.grad~  : {fmt_exact(grad)}")
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", default="/root/zyg/logs/stage4/cpu_gradient_oracle.json")
    args = ap.parse_args()

    print("=" * 78)
    print("STAGE 4 CPU GRADIENT ORACLE — real PyG API on CPU")
    print(f"  python     : {platform.python_version()}")
    print(f"  torch      : {torch.__version__}")
    print(f"  torch_geometric : {torch_geometric.__version__}")
    print(f"  global_max_pool : {global_max_pool}")
    print("=" * 78)

    # ---- G0 unique maximum -------------------------------------------------------------
    case("G0_unique_max", [[1.0, 10.0], [3.0, 20.0], [2.0, 30.0]], [0, 0, 0], 1,
         note="one strict maximum per feature -> gradient only to the argmax")

    # ---- G1 2-way tie ------------------------------------------------------------------
    case("G1_two_way_tie", [[3.0, 1.0], [3.0, 2.0], [1.0, 3.0]], [0, 0, 0], 1,
         note="feature 0 has a 2-way tie (rows 0,1), feature 1 a unique max (row 2)")

    # ---- G2 3-way tie ------------------------------------------------------------------
    case("G2_three_way_tie", [[5.0, 5.0], [5.0, 5.0], [5.0, 5.0]], [0, 0, 0], 1,
         note="all rows equal -> full 3-way tie in both features")

    # ---- G3 different tie sets per feature ---------------------------------------------
    case("G3_per_feature_ties", [[5.0, 1.0, 7.0], [5.0, 3.0, 7.0], [2.0, 3.0, 0.0]], [0, 0, 0], 1,
         note="f0 tie rows{0,1}, f1 tie rows{1,2}, f2 tie rows{0,1}")

    # ---- G4 weighted upstream gradient --------------------------------------------------
    case("G4_weighted_upstream", [[3.0, 1.0], [3.0, 2.0], [1.0, 3.0]], [0, 0, 0], 1,
         upstream=[[2.0, 4.0]], note="loss = (out * upstream).sum(); grad must scale by upstream")

    # ---- G5 multiple groups / repeated index -------------------------------------------
    case("G5_multi_group", [[4.0, 1.0], [4.0, 2.0], [0.0, 9.0], [7.0, 7.0], [7.0, 7.0], [1.0, 0.0]],
         [0, 0, 0, 1, 1, 1], 2,
         note="group 0: f0 tie(0,1)/f1 unique(2); group 1: f0 tie(3,4)/f1 tie(3,4)+f? ")

    # ---- G6 explicit size + empty group --------------------------------------------------
    case("G6_explicit_size_empty_groups", [[4.0, 1.0], [4.0, 2.0], [0.0, 9.0]],
         [0, 0, 3], 5, upstream=[[1.0, 1.0], [5.0, 5.0], [7.0, 7.0], [11.0, 11.0], [13.0, 13.0]],
         note="size=5, groups 1/2/4 empty; upstream on empty groups must not create x.grad")

    # ---- G7 negative-only ---------------------------------------------------------------
    case("G7_negative_only", [[-5.0], [-2.0], [-2.0]], [0, 0, 0], 1,
         note="max is -2 with a 2-way tie")

    # ---- G8 true -inf -------------------------------------------------------------------
    case("G8_all_neg_inf", [[float("-inf")], [float("-inf")]], [0, 0], 1,
         note="non-empty group whose maximum is -inf")
    case("G8b_neg_inf_mixed", [[float("-inf")], [float("-inf")], [-1.0]], [0, 0, 0], 1,
         note="-inf rows plus a finite value")
    case("G8c_neg_inf_vs_empty", [[float("-inf")], [float("-inf")], [5.0]], [0, 0, 3], 5,
         note="group 0 = all -inf, groups 1/2/4 empty")

    # ---- G9 signed zeros ----------------------------------------------------------------
    case("G9_pm_zero", [[0.0], [-0.0]], [0, 0], 1, note="max of +0.0 and -0.0")
    case("G9b_pm_zero_mixed", [[0.0], [-0.0], [-1.0]], [0, 0, 0], 1,
         note="+0/-0 tie with a smaller finite value")

    # ---- G10 NaN -------------------------------------------------------------------------
    case("G10a_one_nan", [[float("nan")], [1.0], [2.0]], [0, 0, 0], 1, note="one NaN in the group")
    case("G10b_two_nan", [[float("nan")], [float("nan")], [2.0]], [0, 0, 0], 1,
         note="two NaNs in the group")
    case("G10c_nan_and_inf", [[float("nan")], [float("inf")], [1.0]], [0, 0, 0], 1,
         note="NaN + +inf")

    # ---- batch=None (PyG uses x.max(dim=-2) in that case) --------------------------------
    case("N1_batch_none_unique", [[1.0, 10.0], [3.0, 20.0], [2.0, 30.0]], None, None,
         note="batch=None -> PyG x.max(dim=-2)")
    case("N2_batch_none_tie", [[3.0, 1.0], [3.0, 2.0], [1.0, 3.0]], None, None,
         note="batch=None with ties")
    case("N3_batch_none_1d", [1.0, 3.0, 2.0], None, None, oned=True,
         note="1-D input, batch=None -> x.max(dim=-1, keepdim=True)")

    # ---- cross-check: the raw scatter_reduce PyG itself calls ---------------------------
    print("\n### cross-check: src.new_zeros(size).scatter_reduce_(-2, idx, src, amax, include_self=False)")
    x = torch.tensor([[3.0, 1.0], [3.0, 2.0], [1.0, 3.0]], requires_grad=True)
    idx = torch.tensor([0, 0, 0])
    out = x.new_zeros((1, 2)).scatter_reduce_(-2, idx.unsqueeze(-1).expand(-1, 2).clone(), x,
                                              reduce="amax", include_self=False)
    out.sum().backward()
    print(f"    scatter_reduce amax out : {fmt(out.detach())}")
    print(f"    scatter_reduce amax grad: {fmt(x.grad)}")
    RESULTS.append({"case": "XC_scatter_reduce_amax", "out": out.detach().tolist(),
                    "grad": x.grad.tolist()})

    # ---- empty-group upstream sensitivity ------------------------------------------------
    print("\n### empty-group upstream sensitivity (does upstream on an empty group matter?)")
    x = torch.tensor([[4.0], [4.0]], requires_grad=True)
    b = torch.tensor([0, 0])
    out = global_max_pool(x, b, 3)
    out.sum().backward()
    print(f"    size=3, groups 0(occupied),1,2(empty): out={fmt(out.detach())} grad={fmt(x.grad)}")
    RESULTS.append({"case": "XC_empty_group_out", "out": out.detach().tolist(),
                    "grad": x.grad.tolist()})

    import os
    os.makedirs(os.path.dirname(args.json), exist_ok=True)
    with open(args.json, "w") as fh:
        json.dump(RESULTS, fh, indent=1)
    print(f"\n[json] {args.json}")


if __name__ == "__main__":
    main()
