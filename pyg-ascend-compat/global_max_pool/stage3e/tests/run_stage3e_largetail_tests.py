#!/usr/bin/env python3
"""Stage 3E / Task G+H: largeTail tail-attack correctness + PyG end-to-end, on the FORMAL
delivery OPP (kernel entries _0/_1 + index lookup fix + write-offset fix + byte-exact index loads).

Every case: independent CPU golden, tol 0, full-tensor comparison (so a wrong write offset cannot
hide), plus targeted assertions on the tail/head columns that the Stage 3D defects corrupted.

Run:
  source <formal_opp>/bin/set_env.bash
  PYG_ASCEND_ADAPTER_PATH=/root/zyg/global_max_pool/stage2/python/global_max_pool_ascend.py \
  SCATTERMAXV1_BRIDGE=/root/zyg/build/stage2_ext/scattermaxv1_bridge.so \
  python3 run_stage3e_largetail_tests.py
"""

from __future__ import annotations

import json
import os
import sys
import warnings

import torch
import torch_npu  # noqa: F401

HERE = os.path.dirname(os.path.abspath(__file__))
STAGE6_DIR = os.environ.get("STAGE3E_STAGE6_DIR", "/root/zyg/global_max_pool/stage6")
sys.path.insert(0, os.path.normpath(os.path.join(HERE, "..", "..", "stage3b", "tests")))
sys.path.insert(0, "/root/zyg/global_max_pool/stage3b/tests")
sys.path.insert(0, STAGE6_DIR)

import stage3b_common as C  # noqa: E402

RESULTS = []
LOGDIR = os.environ.get("STAGE3E_LOGS", "/root/zyg/logs/stage3e")


def record(name, ok, detail):
    RESULTS.append({"case": name, "result": "PASS" if ok else "FAIL", "detail": detail})
    print(f"{name:38s} {'PASS' if ok else 'FAIL'}  {detail}")


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
        record(name, False, f"{type(err).__name__}: {str(err)[:140]}")
        return False
    ref = C.reference_vectorized(x, batch, size)
    got = out.detach().cpu()
    finite = torch.isfinite(ref) & torch.isfinite(got)
    diff = float((ref[finite] - got[finite]).abs().max()) if bool(finite.any()) else 0.0
    inf_ok = torch.equal(torch.isinf(ref), torch.isinf(got))
    shape_ok = tuple(got.shape) == tuple(ref.shape)
    extra = expect(got) if expect else ""
    ok = bool(shape_ok and diff == 0.0 and inf_ok)
    record(name, ok, f"shape={tuple(got.shape)} max_diff={diff} inf_ok={inf_ok} {extra}")
    return ok


def tail_attack_src(n, f):
    """Deterministic values with unique maxima in the last row and at the row edges."""
    x = C.make_src(n, f, "det")
    x[n - 1, :] = x[n - 1, :] + 100.0
    x[n - 1, 0] = 1234.0
    x[n - 1, f - 1] = 4321.0
    return x


def main():
    torch.npu.set_device(0)
    print(f"stage3e large-tail tests | FORMAL delivery OPP | torch={torch.__version__}\n")

    F = 48825
    N = 40
    S = 8

    # A: last row unique max at the first and at the last logical feature
    x = tail_attack_src(N, F)
    b = C.make_index(N, S, "spread", seed=1)
    check("A_tail_attack_N40_F48825", x, b, S,
          expect=lambda g: f"group_max={g.max().item()}")

    # A2: final feature chunk only - head columns must NOT be overwritten by the chunk-1 write
    x2 = C.make_src(N, F, "det") - 100.0
    x2[N - 1, :] = -500.0
    x2[N - 1, 0] = 1234.0              # head column of the last row
    x2[N - 1, F - 1] = 4321.0          # last column (final chunk, n = 1)
    g = C.make_index(N, S, "spread", seed=11)
    check("A2_final_chunk_vs_head_N40_F48825", x2, g, S,
          expect=lambda g_: f"row_max={g_.max().item()}")

    # B: repeated index (all rows -> group 0) + tail winner
    x3 = tail_attack_src(41, F)
    b3 = torch.zeros(41, dtype=torch.int64)
    check("B_repeated_index_N41_F48825", x3, b3, 1,
          expect=lambda g_: f"row0_max={g_[0].max().item()}")

    # C: non-aligned F (adapter pads 48826 -> 48832)
    x4 = tail_attack_src(N, 48826)
    b4 = C.make_index(N, S, "spread", seed=2)
    check("C_non_aligned_N40_F48826", x4, b4, S)

    # D: negative-only data at a large-tail shape
    x5 = -torch.abs(C.make_src(N, F, "det")) - 1.0
    b5 = C.make_index(N, S, "spread", seed=3)
    check("D_negative_only_N40_F48825", x5, b5, S,
          expect=lambda g_: f"all_negative={bool((g_ <= 0).all())}")

    # E: group of all -inf rows must stay -inf (not 0), empty group must be 0
    x6 = torch.full((3, F), -1.0)
    x6[0] = float("-inf")
    x6[1] = float("-inf")                 # group 0 = rows {0,1} -> genuinely -inf
    b6 = torch.tensor([0, 0, 2], dtype=torch.int64)
    check("E_neg_inf_vs_empty_N3_F48825", x6, b6, 3,
          expect=lambda g_: f"g0_all_neg_inf={bool((g_[0] == float('-inf')).all())} "
                            f"g1_empty_zero={bool((g_[1] == 0).all())} "
                            f"g2_all_minus1={bool((g_[2] == -1.0).all())}")

    # F: explicit size larger than max(index)+1
    check("F_explicit_size_N40_F48825", C.make_src(N, F, "det"),
          C.make_index(N, 4, "spread", seed=4), 8)

    # G: first/last column of the final chunk, F 32 B aligned (48832)
    FG = 48832
    x7 = C.make_src(N, FG, "det") - 100.0
    x7[N - 1, :] = -500.0
    x7[N - 1, FG - 8] = 1234.0        # first column of the final chunk (48824)
    x7[N - 1, FG - 1] = 4321.0        # last column of the final chunk
    b7 = C.make_index(N, S, "spread", seed=5)
    check("G_final_chunk_edges_N40_F48832", x7, b7, S,
          expect=lambda g_: f"row_max={g_.max().item()}")

    # H: leftSrc + negative-only + repeated index (N = 41 -> 1 remainder row)
    x8 = -torch.abs(C.make_src(41, F, "det")) - 1.0
    b8 = torch.zeros(41, dtype=torch.int64)
    check("H_leftsrc_negative_repeat_N41_F48825", x8, b8, 1,
          expect=lambda g_: f"row0_min={g_[0].min().item()}")

    # ---- PyG end-to-end through torch_geometric.nn.global_max_pool ----
    import pyg_ascend_compat
    pyg_ascend_compat.enable(debug=True)
    from torch_geometric.nn import global_max_pool  # noqa: PLC0415

    def e2e(name, n, f, s, seed=7):
        x = tail_attack_src(n, f)
        batch = C.make_index(n, s, "spread", seed=seed)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            out = global_max_pool(x.to(C.DEV), batch.to(C.DEV), s)
            torch.npu.synchronize()
        fb = [str(w.message) for w in caught if "fall back" in str(w.message)]
        ref = C.reference_vectorized(x, batch, s)
        got = out.detach().cpu()
        finite = torch.isfinite(ref) & torch.isfinite(got)
        diff = float((ref[finite] - got[finite]).abs().max()) if bool(finite.any()) else 0.0
        ok = bool(diff == 0.0 and tuple(got.shape) == tuple(ref.shape)
                  and torch.equal(torch.isinf(ref), torch.isinf(got)) and not fb)
        record(name, ok, f"shape={tuple(got.shape)} max_diff={diff} fallback={'YES' if fb else 'NONE'}")
        return ok

    e2e("E2E_LT1_N40_F48825", 40, 48825, 8)
    e2e("E2E_LT2_N41_F48825", 41, 48825, 8, seed=8)
    e2e("E2E_LT3_N40_F48826", 40, 48826, 8, seed=9)
    st = pyg_ascend_compat.stats()
    print(f"\ncompat counters: {st}")
    if st.get("original_calls", 1) != 0 or st.get("ascend_calls", 0) <= 0:
        record("E2E_dispatch_counters", False, str(st))
    else:
        record("E2E_dispatch_counters", True, f"ascend_calls={st['ascend_calls']} original_calls=0")

    failed = [r for r in RESULTS if r["result"] != "PASS"]
    print(f"\nTOTAL {len(RESULTS)}  PASS {len(RESULTS) - len(failed)}  FAIL {len(failed)}")
    os.makedirs(LOGDIR, exist_ok=True)
    with open(os.path.join(LOGDIR, "05_largetail_correctness.json"), "w") as fh:
        json.dump({"results": RESULTS, "counters": st}, fh, indent=1)
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
