#!/usr/bin/env python3
"""Stage 3A correctness matrix: non-aligned feature dims (padding path) + F=0 + transitions.

Golden is the same independent CPU reference used in Stage 2 (NOT torch_geometric).
"""

from __future__ import annotations

import os
import sys
import time

import torch
import torch_npu  # noqa: F401

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.normpath(os.path.join(HERE, "..", "..", "stage2", "python")))

from global_max_pool_ascend import (  # noqa: E402
    get_last_run_info,
    global_max_pool_ascend,
)

DEV = "npu:0"
NEG_INF = float("-inf")
RESULTS = []


def reference(x, batch, size=None):
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


def compare(name, x, batch, size=None, mode="auto", expect_error=None):
    x_dev = x.to(DEV) if x.device.type != "npu" else x
    b_dev = batch.to(DEV) if batch.device.type != "npu" else batch
    info_before = dict(get_last_run_info())
    t0 = time.perf_counter()
    try:
        out, info = global_max_pool_ascend(x_dev, b_dev, size, alignment_mode=mode, debug=True)
        torch.npu.synchronize()
        err = None
    except Exception as exc:  # noqa: BLE001
        out, info, err = None, {}, exc
    dt = (time.perf_counter() - t0) * 1000.0

    if expect_error is not None:
        ok = isinstance(err, expect_error) and dict(get_last_run_info()) == info_before
        RESULTS.append((name, "PASS" if ok else "FAIL", str(err)))
        print(f"{name:34s} {'PASS' if ok else 'FAIL'}  expected {expect_error.__name__}, got {err}")
        return ok
    if err is not None:
        RESULTS.append((name, "FAIL", f"{type(err).__name__}: {err}"))
        print(f"{name:34s} FAIL  unexpected {type(err).__name__}: {err}")
        return False

    exp = reference(x, batch, size)
    got = out.detach().cpu()
    if exp.shape != got.shape:
        RESULTS.append((name, "FAIL", f"shape {tuple(got.shape)} != {tuple(exp.shape)}"))
        print(f"{name:34s} FAIL  shape {tuple(got.shape)} != {tuple(exp.shape)}")
        return False
    exact = torch.equal(exp, got)
    finite = torch.isfinite(exp) & torch.isfinite(got)
    max_diff = float((exp[finite] - got[finite]).abs().max()) if bool(finite.any()) else 0.0
    # -inf positions must match exactly (empty groups are 0 by contract, handled by reference)
    inf_ok = bool(torch.equal(torch.isinf(exp), torch.isinf(got)))
    status = "PASS" if (exact and max_diff == 0.0 and inf_ok) else "FAIL"
    RESULTS.append((name, status, f"max_diff={max_diff} exact={exact} padded={info.get('padded')}"))
    print(
        f"{name:34s} {status}  shape={tuple(got.shape)} max_diff={max_diff} exact={exact} "
        f"padded={info.get('padded')} t={dt:.1f}ms"
    )
    if status == "FAIL":
        print("   expected:\n", exp)
        print("   actual:\n", got)
    return status == "PASS"


def build(n, f, pattern, seed=0):
    g = torch.Generator().manual_seed(seed + f * 31 + n)
    if pattern == "neg_only":
        return -(torch.rand((n, f), generator=g) * 10.0 + 1.0)
    if pattern == "neg_inf":
        return torch.full((n, f), NEG_INF, dtype=torch.float32)
    return torch.round((torch.rand((n, f), generator=g) * 20.0 - 10.0) * 4.0) / 4.0


def main():
    torch.npu.set_device(0)
    print(f"device={DEV} torch={torch.__version__} torch_npu={torch_npu.__version__}\n")
    print("--- B matrix: non-aligned F via padding (alignment_mode=auto) ---")
    b8 = torch.tensor([0, 1, 0, 2, 1, 2, 0, 3], dtype=torch.int64)
    compare("B1_F1", build(8, 1, "mixed", 1), b8)
    compare("B2_F7", build(8, 7, "mixed", 2), b8)
    compare("B3_F9", build(8, 9, "mixed", 3), b8)
    compare("B4_F17", build(16, 17, "mixed", 4),
            torch.tensor([0, 5, 3, 3, 7, 1, 0, 2, 5, 7, 7, 4, 6, 1, 2, 0], dtype=torch.int64))
    compare("B5_F31", build(8, 31, "mixed", 5), b8)
    compare("B6_F33", build(16, 33, "mixed", 6),
            torch.tensor([0, 3, 7, 1, 5, 2, 0, 6, 4, 7, 3, 1, 5, 2, 6, 4], dtype=torch.int64))
    compare("B7_F7_repeated", build(16, 7, "mixed", 7), torch.zeros(16, dtype=torch.int64))
    compare("B8_F9_explicit_size", build(4, 9, "mixed", 8),
            torch.tensor([0, 2, 0, 1], dtype=torch.int64), size=5)
    compare("B9_F17_negative_only", build(8, 17, "neg_only", 9),
            torch.tensor([0, 0, 1, 1, 2, 2, 2, 2], dtype=torch.int64), size=4)

    x = torch.full((3, 7), NEG_INF)
    x[2] = torch.tensor([-3.0, -1.5, -2.25, -4.0, -0.5, -7.0, -6.5])
    compare("B10_F7_neg_inf_vs_empty", x, torch.tensor([0, 0, 2], dtype=torch.int64), size=3)

    print("\n--- additional non-aligned F / N coverage ---")
    compare("B11_F2_N1", build(1, 2, "mixed", 11), torch.tensor([0], dtype=torch.int64))
    compare("B12_F3_N8", build(8, 3, "mixed", 12), b8)
    compare("B13_F15_N16", build(16, 15, "mixed", 13),
            torch.tensor([0, 3, 7, 1, 5, 2, 0, 6, 4, 7, 3, 1, 5, 2, 6, 4], dtype=torch.int64))

    print("\n--- alignment transition pairs (branch switch must not change results) ---")
    for f in (7, 8, 9, 15, 16, 17, 31, 32, 33):
        compare(f"T_F{f}", build(8, f, "mixed", 100 + f), b8)

    print("\n--- F=0 contract ---")
    compare("Z1_N0_F0_size_None", torch.zeros((0, 0)), torch.zeros((0,), dtype=torch.int64))
    compare("Z2_N0_F0_size_3", torch.zeros((0, 0)), torch.zeros((0,), dtype=torch.int64), size=3)
    compare("Z3_N4_F0_size_None", torch.zeros((4, 0)),
            torch.tensor([0, 2, 0, 1], dtype=torch.int64))
    compare("Z4_N4_F0_size_5", torch.zeros((4, 0)),
            torch.tensor([0, 2, 0, 1], dtype=torch.int64), size=5)
    compare("Z5_N4_F0_bad_index", torch.zeros((4, 0)),
            torch.tensor([0, 5, 0, 1], dtype=torch.int64), size=5, expect_error=ValueError)

    print("\n--- raw path parity (alignment_mode=raw vs padded auto) ---")
    for f in (1, 7, 9, 17, 31, 33):
        x = build(16, f, "mixed", 200 + f)
        batch = torch.tensor([0, 5, 3, 3, 7, 1, 0, 2, 5, 7, 7, 4, 6, 1, 2, 0], dtype=torch.int64)
        a = global_max_pool_ascend(x.to(DEV), batch.to(DEV), 8, alignment_mode="auto")
        r = global_max_pool_ascend(x.to(DEV), batch.to(DEV), 8, alignment_mode="raw")
        torch.npu.synchronize()
        same = torch.equal(a.detach().cpu(), r.detach().cpu())
        ref = reference(x, batch, 8)
        both_ok = same and torch.equal(a.detach().cpu(), ref)
        RESULTS.append((f"P_F{f}_raw_vs_pad", "PASS" if both_ok else "FAIL", f"identical={same}"))
        print(f"P_F{f}_raw_vs_pad{'':14s} {'PASS' if both_ok else 'FAIL'}  identical={same}")

    print("\n--- invalid alignment_mode ---")
    compare("V1_bad_mode", build(4, 8, "mixed", 300),
            torch.tensor([0, 1, 0, 1], dtype=torch.int64), mode="bogus", expect_error=ValueError)

    print("\n=== summary ===")
    failed = [r for r in RESULTS if r[1] != "PASS"]
    for name, status, msg in RESULTS:
        print(f"{status:4s} {name:28s} {msg}")
    print(f"\nTOTAL {len(RESULTS)}  PASS {len(RESULTS) - len(failed)}  FAIL {len(failed)}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
