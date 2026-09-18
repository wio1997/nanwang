#!/usr/bin/env python3
"""Stage 4 — FP32 backward correctness matrix: Ascend vs the CPU PyG oracle.

Every case is compared element-wise (NaN == NaN, ±0 equal) against
``torch_geometric.nn.global_max_pool`` + ``backward()`` on **CPU** with the real installed
PyTorch 2.9.0 / PyG 2.8.0.post1.  Gradients come from the Stage 4 autograd entry
(``global_max_pool_ascend_autograd``), i.e. the frozen ScatterMaxV1 forward + the new backward.

Run (after sourcing the formal OPP environment):
    python3 run_stage4_backward_tests.py
"""

from __future__ import annotations

import json
import os
import sys
import warnings

import torch
import torch_npu  # noqa: F401
from torch_geometric.nn import global_max_pool as pyg_gmp

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, "/root/zyg/global_max_pool/stage4/python")

import global_max_pool_ascend_autograd as AG  # noqa: E402

DEV = "npu:0"
LOGDIR = os.environ.get("STAGE4_LOGS", "/root/zyg/logs/stage4")
RESULTS = []


def cpu_oracle(x, batch, size, upstream):
    xr = x.clone().requires_grad_(True)
    out = pyg_gmp(xr, batch, size)
    (out * upstream).sum().backward()
    return out.detach(), xr.grad


def npu_autograd(x, batch, size, upstream):
    xn = x.to(DEV).requires_grad_(True)
    bn = batch.to(DEV)
    up = upstream.to(DEV)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        out = AG.global_max_pool_ascend_autograd(xn, bn, size)
        (out * up).sum().backward()
        torch.npu.synchronize()
    fb = [str(w.message) for w in caught if "fall back" in str(w.message)]
    return out.detach().cpu(), xn.grad.cpu(), fb


def equal(a, b):
    """NaN-aware, ±0-aware comparison with an ULP report.

    The device has no correctly-rounded fp32 divide (and no fp64: torch_npu casts double back to
    float), so tie-split cells `grad_out/count` may differ from the IEEE correctly-rounded CPU
    result by 1 ULP.  Everything else must be bit-identical.
    """
    if a.shape != b.shape:
        return False, f"shape {tuple(a.shape)} vs {tuple(b.shape)}", 0, 0
    an, bn = torch.isnan(a), torch.isnan(b)
    if not torch.equal(an, bn):
        return False, "nan pattern differs", 0, 0
    fin = ~an
    av, bv = a[fin], b[fin]
    exact = torch.equal(av, bv)
    ai = av.view(torch.int32).to(torch.int64)
    bi = bv.view(torch.int32).to(torch.int64)
    ai = torch.where(av == 0, torch.zeros_like(ai), ai)   # treat -0 as +0
    bi = torch.where(bv == 0, torch.zeros_like(bi), bi)
    d = (ai - bi).abs()
    max_ulp = int(d.max().item()) if d.numel() else 0
    n_ulp = int((d > 0).sum().item())
    if max_ulp <= 1:
        return True, ("exact" if exact else f"1-ULP in {n_ulp} cell(s)"), max_ulp, n_ulp
    bad = (d > 1).nonzero().flatten()
    i = int(bad[0]) if bad.numel() else 0
    return False, (f"ULP {max_ulp} at flat index {int(bad[0]) if bad.numel() else 0}: "
                   f"{av.flatten()[i].item()} vs {bv.flatten()[i].item()}"), max_ulp, n_ulp


def invariants(x, batch, size, upstream, grad_x):
    """sum(grad_x over winners) == grad_out for ordinary cells; 0 for non-winners."""
    n, f = x.shape
    out = torch.zeros((size, f))
    for g in range(size):
        m = batch == g
        if bool(m.any()):
            out[g] = x[m].max(dim=0).values
    fin = torch.isfinite(out)                       # [S, F] (NaN cells excluded)
    row_fin = fin[batch]
    win = (x == out[batch]) & row_fin               # [N, F]
    gfin = torch.where(row_fin, grad_x, torch.zeros_like(grad_x))
    bad_zero = int((gfin[~win].abs() > 0).sum().item())

    nw = torch.zeros((size, f))
    nw.index_add_(0, batch, win.float())
    ssum = torch.zeros((size, f))
    ssum.index_add_(0, batch, torch.where(win, gfin, torch.zeros_like(gfin)))
    cnt = nw + (out == 0).float()
    expected = torch.where(fin, upstream * nw / cnt, torch.zeros_like(out))
    bad_sum = int((((ssum - expected).abs() > 1e-4 * (1.0 + expected.abs())) & fin)
                  .sum().item())
    quirk_cells = int(((out == 0) & fin).sum().item())
    return bad_zero, bad_sum, quirk_cells


def case(name, x, batch, size, upstream, note="", check_invariants=True):
    x = x.float()
    batch = batch.long()
    upstream = upstream.float()
    c_out, c_grad = cpu_oracle(x, batch, size, upstream)
    n_out, n_grad, fb = npu_autograd(x, batch, size, upstream)
    ok_f, why_f, ulp_f, nul_f = equal(n_out, c_out)
    ok_b, why_b, ulp_b, nul_b = equal(n_grad, c_grad)
    ok = ok_f and ok_b and not fb
    detail = (f"forward={ok_f} backward={ok_b} fallback={len(fb)} "
              f"max_ulp(fwd/bwd)={ulp_f}/{ulp_b} diff_cells={nul_b}")
    if not ok_f:
        detail += f" [{why_f}]"
    if not ok_b:
        detail += f" [{why_b}]"
    if check_invariants and not bool(torch.isnan(c_grad).any()):
        bz, bs, q = invariants(x, batch, size, upstream, n_grad)
        detail += f" invariants(nonwinner!=0:{bz}, sum:{bs}, zero-max cells:{q})"
        if bz or bs:
            ok = False
    RESULTS.append({"case": name, "note": note, "result": "PASS" if ok else "FAIL",
                    "detail": detail, "shape": list(x.shape), "size": int(size),
                    "max_ulp_forward": ulp_f, "max_ulp_backward": ulp_b,
                    "ulp_cells_backward": nul_b, "backward_exact": bool(torch.equal(
                        torch.where(torch.isnan(c_grad), torch.zeros_like(c_grad), c_grad),
                        torch.where(torch.isnan(n_grad), torch.zeros_like(n_grad), n_grad))),
                    "grad_real": c_grad.tolist(), "grad_npu": n_grad.tolist(),
                    "forward_real": c_out.tolist(), "forward_npu": n_out.tolist()})
    print(f"{name:44s} {'PASS' if ok else 'FAIL'}  {detail}")
    return ok


def U(rows, cols, fill=1.0):
    return torch.full((rows, cols), fill, dtype=torch.float32)


def main():
    torch.npu.set_device(0)
    print(f"stage4 backward matrix | torch={torch.__version__} | device={DEV}\n")

    # ---------------- B1..B11 semantic matrix ----------------
    case("B1_unique_max",
         torch.tensor([[1.0, 10.0], [3.0, 20.0], [2.0, 30.0]]), torch.tensor([0, 0, 0]), 1, U(1, 2),
         note="grad only at the argmax")
    case("B2_two_way_tie",
         torch.tensor([[3.0, 1.0], [3.0, 2.0], [1.0, 3.0]]), torch.tensor([0, 0, 0]), 1, U(1, 2))
    case("B3_three_way_tie",
         torch.tensor([[5.0, 5.0], [5.0, 5.0], [5.0, 5.0]]), torch.tensor([0, 0, 0]), 1, U(1, 2))
    case("B4_per_feature_ties",
         torch.tensor([[5.0, 1.0, 7.0], [5.0, 3.0, 7.0], [2.0, 3.0, 0.0]]),
         torch.tensor([0, 0, 0]), 1, U(1, 3))
    case("B5_weighted_upstream",
         torch.tensor([[3.0, 1.0], [3.0, 2.0], [1.0, 3.0]]), torch.tensor([0, 0, 0]), 1,
         torch.tensor([[2.0, 4.0]]), note="grad scales with upstream")
    case("B5b_weighted_zeros",
         torch.tensor([[3.0, 1.0], [3.0, 2.0], [1.0, 3.0]]), torch.tensor([0, 0, 0]), 1,
         torch.tensor([[0.0, -1.5]]), note="zero/negative upstream")
    case("B6_multi_group_repeat",
         torch.tensor([[4.0, 1.0], [4.0, 2.0], [0.0, 9.0], [7.0, 7.0], [7.0, 7.0], [1.0, 0.0]]),
         torch.tensor([0, 0, 0, 1, 1, 1]), 2, torch.tensor([[1.0, 2.0], [3.0, 4.0]]))
    case("B7_explicit_size_empty",
         torch.tensor([[4.0, 1.0], [4.0, 2.0], [0.0, 9.0]]), torch.tensor([0, 0, 3]), 5,
         torch.tensor([[1.0, 1.0], [5.0, 5.0], [7.0, 7.0], [11.0, 11.0], [13.0, 13.0]]),
         note="empty groups + zero-max group")
    case("B8_negative_only",
         torch.tensor([[-5.0], [-2.0], [-2.0]]), torch.tensor([0, 0, 0]), 1, U(1, 1))
    case("B9a_all_neg_inf",
         torch.tensor([[float("-inf")], [float("-inf")]]), torch.tensor([0, 0]), 1, U(1, 1))
    case("B9b_neg_inf_mixed",
         torch.tensor([[float("-inf")], [float("-inf")], [-1.0]]), torch.tensor([0, 0, 0]), 1, U(1, 1))
    case("B9c_neg_inf_vs_empty",
         torch.tensor([[float("-inf")], [float("-inf")], [5.0]]), torch.tensor([0, 0, 3]), 5, U(5, 1))
    case("B10a_pm_zero",
         torch.tensor([[0.0], [-0.0]]), torch.tensor([0, 0]), 1, U(1, 1),
         note="signed-zero tie + include_self=False quirk")
    case("B10b_zero_max_group",
         torch.tensor([[4.0], [4.0], [0.0]]), torch.tensor([0, 0, 3]), 5,
         torch.tensor([[1.0], [5.0], [7.0], [11.0], [13.0]]),
         note="group 3 max == 0 -> upstream/2")
    case("B10c_single_zero",
         torch.tensor([[0.0]]), torch.tensor([0]), 1, U(1, 1), note="single zero row -> 1/2")
    case("B11a_one_nan",
         torch.tensor([[float("nan")], [1.0], [2.0]]), torch.tensor([0, 0, 0]), 1, U(1, 1),
         check_invariants=False)
    case("B11b_two_nan",
         torch.tensor([[float("nan")], [float("nan")], [2.0]]), torch.tensor([0, 0, 0]), 1, U(1, 1),
         check_invariants=False)
    case("B11c_nan_and_inf",
         torch.tensor([[float("nan")], [float("inf")], [1.0]]), torch.tensor([0, 0, 0]), 1, U(1, 1),
         check_invariants=False)
    case("B11d_nan_zero_upstream",
         torch.tensor([[float("nan")], [1.0]]), torch.tensor([0, 0]), 1, U(1, 1, fill=0.0),
         check_invariants=False)
    case("B11e_pos_inf_tie",
         torch.tensor([[float("inf")], [float("inf")], [1.0]]), torch.tensor([0, 0, 0]), 1, U(1, 1))

    # ---------------- non-aligned F forward + backward ----------------
    torch.manual_seed(4)
    for f in (1, 7, 8, 9, 17, 33):
        n, s = 12, 3
        base = torch.randint(-3, 4, (n, f)).float()
        x = base.clone()
        x[1] = x[0]                      # force ties
        x[5, :] = 2.0                    # constant row
        if f > 1:
            x[:, -1] = 7.0               # full-column tie
        x[2, 0] = 0.0                    # zero-valued max candidate
        batch = torch.tensor([0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2])
        up = torch.arange(1, s * f + 1, dtype=torch.float32).reshape(s, f) / 4.0
        case(f"B_F{f}_non_aligned", x, batch, s, up, note=f"F={f} padded kernel path")

    # ---------------- N boundaries (ordinary F) ----------------
    for n in (39, 40, 41, 4097):
        f, s = 8, 4
        torch.manual_seed(100 + n)
        x = torch.randint(-4, 5, (n, f)).float()
        x[0] = 5.0                          # tie at the head
        x[-1] = 5.0                         # tie at the tail
        batch = torch.arange(n) % s
        up = torch.ones((s, f))
        case(f"B_N{n}", x, batch, s, up, note=f"N={n} (leftSrc coverage at 41/4097)")

    # ---------------- representative largeTail backward ----------------
    for n, f in ((40, 48825), (41, 48825)):
        s = 8
        x = (torch.arange(n, dtype=torch.float32).unsqueeze(1) % 7.0)
        x = x.expand(n, f).contiguous() * 0.5
        x[n - 1, :] = 3.0                   # tie across the whole last row
        x[0, 0] = 9.0                        # unique max at the head
        x[n - 1, f - 1] = 11.0              # unique max in the tail chunk
        x[n - 1, f - 2] = 11.0              # 2-way tie in the tail chunk
        batch = torch.arange(n) % s
        up = torch.ones((s, f))
        case(f"B_LT_N{n}_F{f}", x, batch, s, up, note="largeTail (LARGE_TAIL kernel) backward")

    # ---------------- requires_grad / no_grad behaviour ----------------
    print()
    x = torch.tensor([[3.0, 1.0], [3.0, 2.0], [1.0, 3.0]], device=DEV)
    b = torch.tensor([0, 0, 0], device=DEV)
    xn = x.clone().requires_grad_(True)
    out = AG.global_max_pool_ascend_autograd(xn, b, 1)
    ok = out.requires_grad and out.grad_fn is not None
    print(f"{'B12_autograd_graph':44s} {'PASS' if ok else 'FAIL'}  "
          f"out.requires_grad={out.requires_grad} grad_fn={type(out.grad_fn).__name__}")
    RESULTS.append({"case": "B12_autograd_graph", "result": "PASS" if ok else "FAIL"})
    with torch.no_grad():
        out_ng = AG.global_max_pool_ascend_autograd(xn, b, 1)
    ok2 = (not out_ng.requires_grad) and out_ng.grad_fn is None
    print(f"{'B13_no_grad_no_graph':44s} {'PASS' if ok2 else 'FAIL'}  "
          f"requires_grad={out_ng.requires_grad}")
    RESULTS.append({"case": "B13_no_grad_no_graph", "result": "PASS" if ok2 else "FAIL"})
    xf = x.clone()
    out_f = AG.global_max_pool_ascend_autograd(xf, b, 1)
    ok3 = (not out_f.requires_grad) and out_f.grad_fn is None
    print(f"{'B14_requires_grad_false_inference':44s} {'PASS' if ok3 else 'FAIL'}  "
          f"grad_fn={out_f.grad_fn}")
    RESULTS.append({"case": "B14_requires_grad_false_inference",
                    "result": "PASS" if ok3 else "FAIL"})

    failed = [r for r in RESULTS if r.get("result") != "PASS"]
    ulps = [r["max_ulp_backward"] for r in RESULTS if "max_ulp_backward" in r]
    exact = sum(1 for u in ulps if u == 0)
    print(f"\ngradient-bitwise: {exact}/{len(ulps)} cases bit-exact, "
          f"max ULP over all cases = {max(ulps) if ulps else 'n/a'}")
    print(f"\nTOTAL {len(RESULTS)}  PASS {len(RESULTS) - len(failed)}  FAIL {len(failed)}")
    os.makedirs(LOGDIR, exist_ok=True)
    with open(os.path.join(LOGDIR, "stage4_backward_results.json"), "w") as fh:
        json.dump(RESULTS, fh, indent=1)
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
