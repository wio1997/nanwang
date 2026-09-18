#!/usr/bin/env python3
"""Stage 3B boundary matrix (adapter level): N/core boundary, leftSrc, large N, S/index bounds.

Every case: independent CPU golden, tol 0, per-case memory estimate against the 1 GiB budget.
"""

from __future__ import annotations

import json
import os
import sys
import time

import torch
import torch_npu  # noqa: F401

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import stage3b_common as C  # noqa: E402

RESULTS = []


def run_case(name, n, f, s, idx_mode="spread", src_pattern="det", seed=0, adapter=None,
             alignment_mode="auto", expect_error=None, budget=C.BUDGET_BYTES):
    est = C.estimate_bytes(n, f, s)
    if expect_error is None and est["total"] > budget:
        msg = f"SKIPPED (estimate {est['total'] / 2**20:.1f} MiB > budget {budget / 2**20:.0f} MiB)"
        RESULTS.append((name, "SKIP", msg))
        print(f"{name:34s} SKIP  {msg}")
        return False
    x = C.make_src(n, f, src_pattern)
    batch = C.make_index(n, s, idx_mode, seed)
    t0 = time.perf_counter()
    if expect_error is not None:
        try:
            adapter.global_max_pool_ascend(x.to(C.DEV), batch.to(C.DEV), s)
            ok, detail = False, "NO ERROR RAISED"
        except Exception as exc:  # noqa: BLE001
            ok, detail = isinstance(exc, expect_error), f"{type(exc).__name__}: {exc}"
    else:
        ok, detail = C.compare(name, x, batch, s, adapter, alignment_mode=alignment_mode)
    dt = (time.perf_counter() - t0) * 1000.0
    est_s = f"~{est['total'] / 2**20:.1f}MiB"
    RESULTS.append((name, "PASS" if ok else "FAIL", detail))
    print(f"{name:34s} {'PASS' if ok else 'FAIL'}  {detail} t={dt:.0f}ms mem={est_s}")
    return ok


def main():
    torch.npu.set_device(0)
    adapter = C.load_adapter()
    print(f"Stage 3B boundary matrix | torch={torch.__version__} | device={C.DEV}\n")

    print("--- §3 N = 39/40/41 core boundary (F=8 aligned, F=33 padded) ---")
    for n in (39, 40, 41):
        for f in (8, 33):
            run_case(f"N{n}_F{f}_spread", n, f, 8, "spread", "det", adapter=adapter)
            run_case(f"N{n}_F{f}_repeat", n, f, 1, "repeat", "det", adapter=adapter)

    print("\n--- §3/§4 leftSrc (N % 40 != 0) ---")
    for n in (41, 79, 81, 127, 4097):
        for f in (8, 17, 33):
            run_case(f"left_N{n}_F{f}", n, f, 8, "leftover_max", "det", adapter=adapter, seed=n)
    run_case("left_N41_F33_leftover_repeat", 41, 33, 8, "leftover_repeat", "det", adapter=adapter)
    run_case("left_N41_F33_leftover_new", 41, 33, 8, "leftover_new", "det", adapter=adapter)
    run_case("left_N4097_F33_tail_winner", 4097, 33, 16, "spread", "tail_winner",
             adapter=adapter)

    print("\n--- §8 large N / MAX_BATCH_NUM boundary ---")
    for n in (163799, 163800, 163801):
        run_case(f"largeN{n}_F8", n, 8, 8, "spread", "det", adapter=adapter, seed=1)
    run_case("largeN163800_F33", 163800, 33, 8, "spread", "det", adapter=adapter, seed=2)
    run_case("largeN163801_F8_tail_winner", 163801, 8, 16, "spread", "tail_winner",
             adapter=adapter)

    print("\n--- §9 S / size boundaries ---")
    run_case("S1_N1", 1, 8, 1, "repeat", "det", adapter=adapter)
    run_case("S1_N40", 40, 8, 1, "repeat", "det", adapter=adapter)
    run_case("S1_N41", 41, 8, 1, "repeat", "det", adapter=adapter)
    run_case("S1_N4097", 4097, 33, 1, "repeat", "det", adapter=adapter)
    run_case("S1024_sparse_N64", 64, 8, 1024, "sparse_0_17_1023", "det", adapter=adapter)
    run_case("S0_N0", 0, 8, 0, "spread", "det", adapter=adapter)

    print("\n--- §10/§11 index + element-budget guards (validation only) ---")
    guard_ok = run_guard_tests(adapter)

    print("\n=== summary ===")
    failed = [r for r in RESULTS if r[1] == "FAIL"]
    skipped = [r for r in RESULTS if r[1] == "SKIP"]
    for name, status, msg in RESULTS:
        print(f"{status:4s} {name:34s} {msg}")
    print(f"\nTOTAL {len(RESULTS)}  PASS {len(RESULTS) - len(failed) - len(skipped)}  "
          f"FAIL {len(failed)}  SKIP {len(skipped)}")
    with open("/root/zyg/logs/stage3b/boundary_results.json", "w") as fh:
        json.dump([{"case": n, "status": s, "detail": d} for n, s, d in RESULTS], fh, indent=1)
    return 1 if failed or not guard_ok else 0


def run_guard_tests(adapter) -> bool:
    """Index upper-bound guard and the documented N*(F+1) budget guard (no huge allocation)."""
    ok = True

    # index 491518 / 491519 allowed, 491520 rejected before the kernel
    n, f = 2, 8
    for idx, s, expect_reject in ((491518, 491519, False), (491519, 491520, False),
                                  (491520, 491520, True)):
        x = C.make_src(n, f, "det")
        batch = torch.tensor([0, idx], dtype=torch.int64)
        est = C.estimate_bytes(n, f, s)
        try:
            out, info = adapter.global_max_pool_ascend(x.to(C.DEV), batch.to(C.DEV), s, debug=True)
            torch.npu.synchronize()
            err = None
            got = out.detach().cpu()
        except Exception as exc:  # noqa: BLE001
            err, got = exc, None
        if expect_reject:
            good = isinstance(err, ValueError)
            detail = f"rejected={type(err).__name__ if err else 'NO'} (must reject before kernel)"
        else:
            ref = C.reference_vectorized(x, batch, s)
            good = err is None and got is not None and torch.equal(ref[:, f - 1], got[:, f - 1])
            detail = f"accepted, last-feature column matches golden (est {est['total'] / 2**20:.1f}MiB)"
        ok = ok and good
        RESULTS.append((f"index_guard_{idx}", "PASS" if good else "FAIL", detail))
        print(f"{'index_guard_' + str(idx):34s} {'PASS' if good else 'FAIL'}  {detail}")

    # documented N*(M+1) budget: exercise the real code path with a temporarily lowered budget
    import global_max_pool_ascend as mod  # noqa: PLC0415
    original = mod.DOC_ELEMENT_BUDGET
    try:
        mod.DOC_ELEMENT_BUDGET = 1000
        x = C.make_src(50, 33, "det")
        batch = C.make_index(50, 8, "spread")
        rejected = False
        try:
            adapter.global_max_pool_ascend(x.to(C.DEV), batch.to(C.DEV), 8)
        except ValueError as exc:
            rejected = "budget" in str(exc)
        finally:
            mod.DOC_ELEMENT_BUDGET = original
        detail = (f"with budget=1000, N*(F+1)={50 * 34} rejected={rejected}; real budget="
                  f"{original} (= 0xF0000000, ~16 GiB fp32 -> not allocated in tests)")
        RESULTS.append(("budget_guard", "PASS" if rejected else "FAIL", detail))
        print(f"{'budget_guard':34s} {'PASS' if rejected else 'FAIL'}  {detail}")
        ok = ok and rejected
    finally:
        mod.DOC_ELEMENT_BUDGET = original
    return ok


if __name__ == "__main__":
    raise SystemExit(main())
