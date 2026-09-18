#!/usr/bin/env python3
"""Stage 2 correctness matrix A1..A12 for global_max_pool_ascend.

Golden is an independent CPU reference (NOT torch_geometric.global_max_pool):
  out = zeros([S, F]); for each group g: out[g] = max(x[n] for batch[n] == g)  (0 if empty)
"""

from __future__ import annotations

import os
import sys
import time
from typing import Optional

import torch
import torch_npu  # noqa: F401

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "python"))

from global_max_pool_ascend import (  # noqa: E402
    get_last_run_info,
    global_max_pool_ascend,
)

DEV = "npu:0"
NEG_INF = float("-inf")
RESULTS = []


def reference(x: torch.Tensor, batch: torch.Tensor, size: Optional[int] = None) -> torch.Tensor:
    """Independent CPU golden: empty group -> 0, otherwise exact max along dim 0."""
    x = x.detach().cpu().to(torch.float32)
    batch = batch.detach().cpu().to(torch.int64)
    n, f = x.shape
    if size is None:
        s = 0 if n == 0 else int(batch.max().item()) + 1
    else:
        s = int(size)
    out = torch.zeros((s, f), dtype=torch.float32)
    for g in range(s):
        mask = batch == g
        if bool(mask.any()):
            out[g] = x[mask].max(dim=0).values
    return out


def check(name, x, batch, size=None, expect_error=None, tol=0.0):
    x_dev = x.to(DEV) if x.device.type != "npu" else x
    b_dev = batch.to(DEV) if batch.device.type != "npu" else batch

    info_before = dict(get_last_run_info())
    t0 = time.perf_counter()
    try:
        out = global_max_pool_ascend(x_dev, b_dev, size)
        torch.npu.synchronize()
        err = None
    except Exception as exc:  # noqa: BLE001
        out = None
        err = exc
    elapsed_ms = (time.perf_counter() - t0) * 1000.0

    if expect_error is not None:
        ok = isinstance(err, expect_error)
        # a rejected call must not have launched the kernel
        no_kernel = dict(get_last_run_info()) == info_before
        msg = f"{type(err).__name__}: {err}" if err else "NO ERROR RAISED"
        status = "PASS" if (ok and no_kernel) else "FAIL"
        RESULTS.append((name, status, msg))
        print(f"{name:28s} {status}  expected {expect_error.__name__}; got {msg}")
        if not no_kernel:
            print(f"{'':28s}        WARNING: adapter state changed -> kernel may have been launched")
        return status == "PASS"

    if err is not None:
        RESULTS.append((name, "FAIL", f"unexpected {type(err).__name__}: {err}"))
        print(f"{name:28s} FAIL  unexpected {type(err).__name__}: {err}")
        return False

    exp = reference(x, batch, size)
    got = out.detach().cpu()
    if exp.shape != got.shape:
        RESULTS.append((name, "FAIL", f"shape {tuple(got.shape)} != expected {tuple(exp.shape)}"))
        print(f"{name:28s} FAIL  shape {tuple(got.shape)} != {tuple(exp.shape)}")
        return False

    same = torch.equal(exp, got)
    finite = torch.isfinite(exp) & torch.isfinite(got)
    max_diff = float((exp[finite] - got[finite]).abs().max()) if bool(finite.any()) else 0.0
    status = "PASS" if (same or max_diff <= tol) else "FAIL"
    RESULTS.append((name, status, f"max_diff={max_diff} exact={same} t={elapsed_ms:.1f}ms"))
    print(
        f"{name:28s} {status}  shape={tuple(got.shape)} max_diff={max_diff} exact={same} "
        f"t={elapsed_ms:.1f}ms"
    )
    if status == "FAIL":
        print("   expected:\n", exp)
        print("   actual:\n", got)
    return status == "PASS"


def build(n, f, pattern, seed=0):
    """Deterministic fp32 source with positives, negatives and duplicates."""
    g = torch.Generator().manual_seed(seed)
    if pattern == "neg_only":
        return -(torch.rand((n, f), generator=g) * 10.0 + 1.0)
    if pattern == "neg_inf":
        return torch.full((n, f), NEG_INF, dtype=torch.float32)
    return torch.round((torch.rand((n, f), generator=g) * 20.0 - 10.0) * 4.0) / 4.0


def main():
    torch.npu.set_device(0)
    print(f"device={DEV} torch={torch.__version__} torch_npu={torch_npu.__version__}\n")

    # ---- A1 baseline, size=None ----
    x = build(8, 8, "mixed", 1)
    b = torch.tensor([0, 1, 0, 2, 1, 2, 0, 3], dtype=torch.int64)
    check("A1_baseline_size_None", x, b)

    # ---- A2 explicit size with empty rows ----
    x = build(4, 8, "mixed", 2)
    b = torch.tensor([0, 2, 0, 1], dtype=torch.int64)
    check("A2_explicit_size_5", x, b, size=5)

    # ---- A3 repeated index ----
    x = build(16, 8, "mixed", 3)
    b = torch.zeros(16, dtype=torch.int64)
    check("A3_repeated_index", x, b)

    # ---- A4 negative-only groups ----
    x = build(8, 8, "neg_only", 4)
    b = torch.tensor([0, 0, 1, 1, 2, 2, 2, 2], dtype=torch.int64)
    check("A4_negative_only", x, b, size=4)

    # ---- A5 true -inf vs empty group ----
    x = torch.full((3, 8), NEG_INF, dtype=torch.float32)
    x[2] = torch.tensor([-3.0, -1.5, -2.25, -4.0, -0.5, -7.0, -6.5, -8.0])
    b = torch.tensor([0, 0, 2], dtype=torch.int64)
    check("A5_neg_inf_vs_empty", x, b, size=3)
    out = global_max_pool_ascend(x.to(DEV), b.to(DEV), 3).detach().cpu()
    ok = bool((out[0] == NEG_INF).all()) and bool((out[1] == 0).all())
    print(
        f"{'':28s} check: group0 all -inf -> {bool((out[0] == NEG_INF).all())}; "
        f"group1 (empty) all 0 -> {bool((out[1] == 0).all())}"
    )
    if not ok:
        RESULTS.append(("A5_semantics_detail", "FAIL", "occupancy mask semantics wrong"))

    # ---- A6/A7 empty input ----
    check("A6_empty_N_size_None", torch.zeros((0, 8)), torch.zeros((0,), dtype=torch.int64))
    check("A7_empty_N_size_4", torch.zeros((0, 8)), torch.zeros((0,), dtype=torch.int64), size=4)

    # ---- A8 F=16 ----
    x = build(12, 16, "mixed", 8)
    b = torch.tensor([0, 5, 3, 3, 7, 1, 0, 2, 5, 7, 7, 4], dtype=torch.int64)
    check("A8_F16", x, b)

    # ---- A9 feature dim: F=7 was rejected in Stage 2, is supported from Stage 3A on
    #      (padded path). Kept here as a positive regression case. ----
    check("A9_F7_supported_since_3A", build(4, 7, "mixed", 9),
          torch.tensor([0, 1, 0, 1], dtype=torch.int64), size=2)

    # ---- A10 index >= size (must reject before kernel) ----
    check("A10_index_ge_size", build(4, 8, "mixed", 10), torch.tensor([0, 3, 0, 1], dtype=torch.int64),
          size=2, expect_error=ValueError)

    # ---- A11 negative index ----
    check("A11_negative_index", build(4, 8, "mixed", 11), torch.tensor([0, -1, 0, 1], dtype=torch.int64),
          size=4, expect_error=ValueError)

    # ---- A12 requires_grad ----
    x = build(4, 8, "mixed", 12).requires_grad_(True)
    check("A12_requires_grad", x, torch.tensor([0, 1, 0, 1], dtype=torch.int64), size=2,
          expect_error=RuntimeError)

    # ---- extra contract checks (dtype / rank / device) ----
    check("A13_batch_not_int64", build(4, 8, "mixed", 13),
          torch.tensor([0, 1, 0, 1], dtype=torch.int32), size=2, expect_error=ValueError)
    check("A14_x_not_fp32", build(4, 8, "mixed", 14).to(torch.float16),
          torch.tensor([0, 1, 0, 1], dtype=torch.int64), size=2, expect_error=ValueError)
    check("A15_x_wrong_rank", build(4, 8, "mixed", 15).reshape(4, 2, 4),
          torch.tensor([0, 1, 0, 1], dtype=torch.int64), size=2, expect_error=ValueError)
    check("A16_N_mismatch", build(4, 8, "mixed", 16),
          torch.tensor([0, 1, 0], dtype=torch.int64), size=2, expect_error=ValueError)

    print("\n=== adapter diagnostics of the last kernel run ===")
    for k, v in sorted(get_last_run_info().items()):
        print(f"  {k}: {v}")

    print("\n=== summary ===")
    failed = [r for r in RESULTS if r[1] != "PASS"]
    for name, status, msg in RESULTS:
        print(f"{status:4s} {name:28s} {msg}")
    print(f"\nTOTAL {len(RESULTS)}  PASS {len(RESULTS) - len(failed)}  FAIL {len(failed)}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
