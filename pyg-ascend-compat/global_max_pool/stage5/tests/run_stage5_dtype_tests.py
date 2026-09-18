#!/usr/bin/env python3
"""Stage 5 — FP16 / BF16 forward+backward correctness matrix vs the same-dtype CPU oracle.

Comparison discipline (section 6): exact equality (NaN-aware) is required; every mismatch is
characterised by max ULP / max abs diff — no `allclose` hiding.

Run (formal OPP sourced):  python3 run_stage5_dtype_tests.py
"""

from __future__ import annotations

import json
import os
import sys
import warnings

import torch
import torch_npu  # noqa: F401
from torch_geometric.nn import global_max_pool as pyg_gmp

sys.path.insert(0, "/root/zyg/global_max_pool/stage5/python")
import global_max_pool_ascend_dtype as S5  # noqa: E402

DEV = "npu:0"
LOGDIR = os.environ.get("STAGE5_LOGS", "/root/zyg/logs/stage5")
RESULTS = []


def ulp(a: torch.Tensor, b: torch.Tensor, dt):
    if dt == torch.float16:
        ai, bi = a.view(torch.int16).to(torch.int32), b.view(torch.int16).to(torch.int32)
    else:
        ai = a.float().view(torch.int32).to(torch.int64)
        bi = b.float().view(torch.int32).to(torch.int64)
    return int((ai - bi).abs().max().item()) if a.numel() else 0


def cmp_tensor(got: torch.Tensor, ref: torch.Tensor, dt):
    """(equal, max_ulp, max_abs, note) with NaN == NaN and ±0 equal."""
    if got.shape != ref.shape:
        return False, -1, float("inf"), f"shape {tuple(got.shape)} vs {tuple(ref.shape)}"
    gn, rn = torch.isnan(got), torch.isnan(ref)
    if not torch.equal(gn, rn):
        return False, -1, float("inf"), "nan pattern differs"
    fin = ~rn
    if not bool(fin.any()):
        return True, 0, 0.0, "all nan"
    g, r = got[fin], ref[fin]
    eq = bool(torch.equal(g, r))
    max_abs = float((g.float() - r.float()).abs().max().item())
    u = ulp(g, r, dt)
    return eq, u, max_abs, ""


def invariants(x, batch, size, upstream, grad_x, dt):
    n, f = x.shape
    out = torch.zeros((size, f), dtype=dt)
    for g in range(size):
        m = batch == g
        if bool(m.any()):
            out[g] = x[m].max(dim=0).values
    fin = torch.isfinite(out)
    row_fin = fin[batch]
    win = (x == out[batch]) & row_fin
    gfin = torch.where(row_fin, grad_x, torch.zeros_like(grad_x))
    bad_zero = int((gfin[~win].abs() > 0).sum().item())
    nw = torch.zeros((size, f))
    nw.index_add_(0, batch, win.float())
    ssum = torch.zeros((size, f))
    ssum.index_add_(0, batch, torch.where(win, gfin.float(), torch.zeros_like(gfin.float())))
    cnt = nw + (out == 0).float()
    expected = torch.where(fin, upstream.float() * nw / cnt, torch.zeros_like(out.float()))
    tol = 2.0 * float(torch.finfo(dt).eps) * expected.abs().clamp(min=1.0)
    bad_sum = int((((ssum - expected).abs() > tol) & fin).sum().item())
    return bad_zero, bad_sum


def case(name, x, batch, size, upstream, dt_tag, dt, note="", check_inv=True):
    x = x.to(dt)
    batch = batch.long()
    upstream = upstream.to(dt)
    xr = x.clone().requires_grad_(True)
    out_c = pyg_gmp(xr, batch, size)
    (out_c * upstream).sum().backward()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        xn = x.clone().to(DEV).requires_grad_(True)
        out_n = S5.global_max_pool_ascend_dtype(xn, batch.to(DEV), size)
        (out_n * upstream.to(DEV)).sum().backward()
        torch.npu.synchronize()
    fb = [str(w.message) for w in caught if "fall back" in str(w.message)]
    f_eq, f_ulp, f_abs, f_note = cmp_tensor(out_n.detach().cpu(), out_c.detach(), dt)
    g_eq, g_ulp, g_abs, g_note = cmp_tensor(xn.grad.cpu(), xr.grad, dt)
    dtype_ok = (out_n.dtype == dt and xn.grad.dtype == dt)
    ok = f_eq and g_eq and dtype_ok and not fb
    detail = (f"fwd_eq={f_eq}(ulp {f_ulp}) bwd_eq={g_eq}(ulp {g_ulp}) dtype_ok={dtype_ok} "
              f"fallback={len(fb)}")
    if not f_eq:
        detail += f" [{f_note}]"
    if not g_eq:
        detail += f" [{g_note}]"
    if check_inv and not bool(torch.isnan(xr.grad).any()):
        bz, bs = invariants(x, batch, size, upstream, xn.grad.cpu(), dt)
        detail += f" invariants(nonwinner:{bz}, sum:{bs})"
        if bz or bs:
            ok = False
    RESULTS.append({"dtype": dt_tag, "case": name, "note": note, "result": "PASS" if ok else "FAIL",
                    "detail": detail, "max_ulp_forward": f_ulp, "max_ulp_backward": g_ulp,
                    "max_abs_forward": f_abs, "max_abs_backward": g_abs})
    print(f"[{dt_tag}] {name:38s} {'PASS' if ok else 'FAIL'}  {detail}")
    return ok


def U(r, c, fill=1.0):
    return torch.full((r, c), fill, dtype=torch.float32)


def main():
    torch.npu.set_device(0)
    print(f"stage5 dtype matrix | torch={torch.__version__} | device={DEV}\n")

    for dt_tag, dt in (("fp16", torch.float16), ("bf16", torch.bfloat16)):
        print(f"########## {dt_tag} ##########")
        case("unique_max", torch.tensor([[1.0, 10.0], [3.0, 20.0], [2.0, 30.0]]),
             torch.tensor([0, 0, 0]), 1, U(1, 2), dt_tag, dt)
        case("two_way_tie", torch.tensor([[3.0, 1.0], [3.0, 2.0], [1.0, 3.0]]),
             torch.tensor([0, 0, 0]), 1, U(1, 2), dt_tag, dt)
        case("three_way_tie", torch.tensor([[5.0, 5.0], [5.0, 5.0], [5.0, 5.0]]),
             torch.tensor([0, 0, 0]), 1, U(1, 2), dt_tag, dt)
        case("per_feature_ties",
             torch.tensor([[5.0, 1.0, 7.0], [5.0, 3.0, 7.0], [2.0, 3.0, 0.0]]),
             torch.tensor([0, 0, 0]), 1, U(1, 3), dt_tag, dt)
        case("weighted_pos", torch.tensor([[3.0, 1.0], [3.0, 2.0], [1.0, 3.0]]),
             torch.tensor([0, 0, 0]), 1, torch.tensor([[2.0, 4.0]]), dt_tag, dt)
        case("weighted_neg", torch.tensor([[3.0, 1.0], [3.0, 2.0], [1.0, 3.0]]),
             torch.tensor([0, 0, 0]), 1, torch.tensor([[-2.0, -4.0]]), dt_tag, dt)
        case("zero_upstream", torch.tensor([[3.0, 1.0], [3.0, 2.0], [1.0, 3.0]]),
             torch.tensor([0, 0, 0]), 1, torch.tensor([[0.0, 0.0]]), dt_tag, dt)
        case("multiple_groups",
             torch.tensor([[4.0, 1.0], [4.0, 2.0], [0.0, 9.0], [7.0, 7.0], [7.0, 7.0], [1.0, 0.0]]),
             torch.tensor([0, 0, 0, 1, 1, 1]), 2, torch.tensor([[1.0, 2.0], [3.0, 4.0]]), dt_tag, dt)
        case("explicit_size_empty", torch.tensor([[4.0, 1.0], [4.0, 2.0], [0.0, 9.0]]),
             torch.tensor([0, 0, 3]), 5,
             torch.tensor([[1.0, 1.0], [5.0, 5.0], [7.0, 7.0], [11.0, 11.0], [13.0, 13.0]]),
             dt_tag, dt)
        case("negative_only", torch.tensor([[-5.0], [-2.0], [-2.0]]), torch.tensor([0, 0, 0]), 1,
             U(1, 1), dt_tag, dt)
        case("all_neg_inf", torch.tensor([[float("-inf")], [float("-inf")]]),
             torch.tensor([0, 0]), 1, U(1, 1), dt_tag, dt)
        case("neg_inf_mixed", torch.tensor([[float("-inf")], [float("-inf")], [-1.0]]),
             torch.tensor([0, 0, 0]), 1, U(1, 1), dt_tag, dt)
        case("pos_inf_tie", torch.tensor([[float("inf")], [float("inf")], [1.0]]),
             torch.tensor([0, 0, 0]), 1, U(1, 1), dt_tag, dt)
        case("pm_zero", torch.tensor([[0.0], [-0.0]]), torch.tensor([0, 0]), 1, U(1, 1), dt_tag, dt)
        case("zero_max_group", torch.tensor([[4.0], [4.0], [0.0]]), torch.tensor([0, 0, 3]), 5,
             torch.tensor([[1.0], [5.0], [7.0], [11.0], [13.0]]), dt_tag, dt)
        case("one_nan", torch.tensor([[float("nan")], [1.0], [2.0]]), torch.tensor([0, 0, 0]), 1,
             U(1, 1), dt_tag, dt, check_inv=False)
        case("two_nan", torch.tensor([[float("nan")], [float("nan")], [2.0]]),
             torch.tensor([0, 0, 0]), 1, U(1, 1), dt_tag, dt, check_inv=False)
        case("nan_and_inf", torch.tensor([[float("nan")], [float("inf")], [1.0]]),
             torch.tensor([0, 0, 0]), 1, U(1, 1), dt_tag, dt, check_inv=False)
        case("nan_zero_upstream", torch.tensor([[float("nan")], [1.0]]), torch.tensor([0, 0]), 1,
             U(1, 1, fill=0.0), dt_tag, dt, check_inv=False)

        # non-aligned F
        torch.manual_seed(11)
        for f in (1, 7, 8, 9, 17, 33):
            n, s = 12, 3
            x = torch.randint(-3, 4, (n, f)).float()
            x[1] = x[0]
            if f > 1:
                x[:, -1] = 7.0
            x[2, 0] = 0.0
            batch = torch.tensor([0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2])
            up = (torch.arange(1, s * f + 1, dtype=torch.float32).reshape(s, f) / 4.0)
            case(f"F{f}_non_aligned", x, batch, s, up, dt_tag, dt)

        # N boundaries
        for n in (39, 40, 41, 4097):
            f, s = 8, 4
            torch.manual_seed(100 + n)
            x = torch.randint(-4, 5, (n, f)).float()
            x[0] = 5.0
            x[-1] = 5.0
            batch = torch.arange(n) % s
            case(f"N{n}", x, batch, s, U(s, f), dt_tag, dt)

        # largeTail (forward + backward)
        for n, f in ((40, 48825), (41, 48825)):
            s = 8
            x = ((torch.arange(n, dtype=torch.float32).unsqueeze(1) % 7.0).expand(n, f)
                 .contiguous() * 0.5)
            x[n - 1, :] = 3.0
            x[0, 0] = 9.0
            x[n - 1, f - 1] = 11.0
            x[n - 1, f - 2] = 11.0
            case(f"largeTail_N{n}_F{f}", x, torch.arange(n) % s, s, U(s, f), dt_tag, dt)
        print()

    failed = [r for r in RESULTS if r["result"] != "PASS"]
    for tag in ("fp16", "bf16"):
        rs = [r for r in RESULTS if r["dtype"] == tag]
        f = [r for r in rs if r["result"] != "PASS"]
        ulps = [r["max_ulp_backward"] for r in rs]
        exact = sum(1 for u in ulps if u == 0)
        print(f"{tag}: TOTAL {len(rs)} PASS {len(rs) - len(f)} FAIL {len(f)} | "
              f"gradient bit-exact {exact}/{len(rs)} | max ULP {max(ulps) if ulps else 'n/a'}")
    print(f"\nTOTAL {len(RESULTS)}  PASS {len(RESULTS) - len(failed)}  FAIL {len(failed)}")
    os.makedirs(LOGDIR, exist_ok=True)
    with open(os.path.join(LOGDIR, "stage5_dtype_results.json"), "w") as fh:
        json.dump(RESULTS, fh, indent=1)
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
