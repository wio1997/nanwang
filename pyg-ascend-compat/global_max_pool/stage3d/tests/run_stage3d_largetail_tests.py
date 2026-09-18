#!/usr/bin/env python3
"""Stage 3D large-tail correctness (tail-attack patterns) + PyG end-to-end, against the REPAIRED
package (kernel entries _0/_1 + index-lookup fix + write-offset fix).

Run:
  ASCEND_CUSTOM_OPP_PATH=<fix4>/vendors/customize \
  LD_LIBRARY_PATH=<fix4>/vendors/customize/op_api/lib:$LD_LIBRARY_PATH \
  SCATTERMAXV1_BRIDGE=<bridge built against fix4> \
  PYG_ASCEND_ADAPTER_PATH=/root/zyg/global_max_pool/stage2/python/global_max_pool_ascend.py \
  python3 run_stage3d_largetail_tests.py
"""

from __future__ import annotations

import json
import os
import sys

import torch
import torch_npu  # noqa: F401

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.normpath(os.path.join(HERE, "..", "..", "stage3b", "tests")))
sys.path.insert(0, "/root/zyg/global_max_pool/stage6")

import stage3b_common as C  # noqa: E402

RESULTS = []


def check(name, x, batch, size, expect=None):
    adapter = C.load_adapter()
    x_dev, b_dev = x.to(C.DEV), batch.to(C.DEV)
    try:
        out = adapter.global_max_pool_ascend(x_dev, b_dev, size)
        torch.npu.synchronize()
        err = None
    except Exception as exc:  # noqa: BLE001
        out, err = None, exc
    if err is not None:
        RESULTS.append((name, "FAIL", f"{type(err).__name__}: {str(err)[:120]}"))
        print(f"{name:34s} FAIL  {type(err).__name__}: {str(err)[:120]}")
        return False
    ref = C.reference_vectorized(x, batch, size)
    got = out.detach().cpu()
    finite = torch.isfinite(ref) & torch.isfinite(got)
    diff = float((ref[finite] - got[finite]).abs().max()) if bool(finite.any()) else 0.0
    inf_ok = torch.equal(torch.isinf(ref), torch.isinf(got))
    shape_ok = tuple(got.shape) == tuple(ref.shape)
    ok = bool(shape_ok and diff == 0.0 and inf_ok)
    detail = f"shape={tuple(got.shape)} max_diff={diff} inf_ok={inf_ok}"
    if expect:
        detail += " " + expect(got)
    RESULTS.append((name, "PASS" if ok else "FAIL", detail))
    print(f"{name:34s} {'PASS' if ok else 'FAIL'}  {detail}")
    return ok


def tail_attack_src(n, f):
    """Deterministic values with unique maxima in the last row and at the row edges."""
    x = C.make_src(n, f, "det")
    x[n - 1, :] = x[n - 1, :] + 100.0      # last physical row dominates its group
    x[n - 1, 0] = 1234.0                   # last row / first feature unique max
    x[n - 1, f - 1] = 4321.0               # last row / last logical feature unique max
    return x


def main():
    torch.npu.set_device(0)
    print(f"stage3d large-tail tests | F-based tail attack | torch={torch.__version__}\n")

    F = 48825
    N = 40
    S = 8
    x = tail_attack_src(N, F)
    b = C.make_index(N, S, "spread", seed=1)
    check("A_tail_attack_N40_F48825", x, b, S,
          expect=lambda g: f"last_row_group_max_ok={g.max().item() == 4321.0}")

    # repeated index + tail winner at a large-tail shape
    x2 = tail_attack_src(41, F)
    b2 = torch.zeros(41, dtype=torch.int64)
    check("B_repeat_tail_N41_F48825", x2, b2, 1,
          expect=lambda g: f"row0_max={g[0].max().item()}")

    # non-aligned F (adapter pads to 48832)
    x3 = tail_attack_src(N, 48826)
    b3 = C.make_index(N, S, "spread", seed=2)
    check("C_non_aligned_N40_F48826", x3, b3, S)

    # negative-only data at a large-tail shape
    x4 = -torch.abs(C.make_src(N, F, "det")) - 1.0
    b4 = C.make_index(N, S, "spread", seed=3)
    check("D_negative_only_N40_F48825", x4, b4, S,
          expect=lambda g: f"all_negative_or_zero={bool((g <= 0).all())}")

    # true -inf group (non-empty) vs empty group
    x5 = torch.full((3, F), -1.0)
    x5[0] = float("-inf")
    b5 = torch.tensor([0, 0, 2], dtype=torch.int64)
    check("E_neg_inf_vs_empty_N3_F48825", x5, b5, 3,
          expect=lambda g: f"g0_all_neg_inf={bool((g[0] == float('-inf')).all())} "
                           f"g1_zero={bool((g[1] == 0).all())}")

    # explicit size larger than max(index)+1
    check("F_explicit_size_N40_F48825", C.make_src(N, F, "det"),
          C.make_index(N, 4, "spread", seed=4), 8)

    # ---- PyG end-to-end through torch_geometric.nn.global_max_pool ----
    import pyg_ascend_compat
    pyg_ascend_compat.enable(debug=True)
    from torch_geometric.nn import global_max_pool  # noqa: PLC0415

    def e2e(name, n, f, s, seed=7):
        x = tail_attack_src(n, f)
        b = C.make_index(n, s, "spread", seed=seed)
        out = global_max_pool(x.to(C.DEV), b.to(C.DEV), s)
        torch.npu.synchronize()
        ref = C.reference_vectorized(x, b, s)
        got = out.detach().cpu()
        finite = torch.isfinite(ref) & torch.isfinite(got)
        diff = float((ref[finite] - got[finite]).abs().max()) if bool(finite.any()) else 0.0
        ok = bool(diff == 0.0 and tuple(got.shape) == tuple(ref.shape)
                  and torch.equal(torch.isinf(ref), torch.isinf(got)))
        RESULTS.append((name, "PASS" if ok else "FAIL", f"shape={tuple(got.shape)} max_diff={diff}"))
        print(f"{name:34s} {'PASS' if ok else 'FAIL'}  shape={tuple(got.shape)} max_diff={diff}")
        return ok

    e2e("E2E_LT1_N40_F48825", 40, 48825, 8)
    e2e("E2E_LT2_N41_F48825", 41, 48825, 8, seed=8)
    e2e("E2E_LT3_N40_F48826", 40, 48826, 8, seed=9)
    st = pyg_ascend_compat.stats()
    print(f"\ncompat counters: {st}")

    failed = [r for r in RESULTS if r[1] != "PASS"]
    print(f"\nTOTAL {len(RESULTS)}  PASS {len(RESULTS) - len(failed)}  FAIL {len(failed)}")
    with open("/root/zyg/logs/stage3d/largetail_correctness.json", "w") as fh:
        json.dump({"results": RESULTS, "counters": st}, fh, indent=1)
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
