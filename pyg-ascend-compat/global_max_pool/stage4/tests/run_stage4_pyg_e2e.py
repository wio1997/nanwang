#!/usr/bin/env python3
"""Stage 4 — real PyG API end-to-end backward (``requires_grad=True``).

Uses the public API exactly as an application would:

    import pyg_ascend_compat; pyg_ascend_compat.enable()
    from torch_geometric.nn import global_max_pool
    out = global_max_pool(x, batch, size); loss.backward()

and compares forward output *and* ``x.grad`` element-wise against the real PyG API on CPU.
"""

from __future__ import annotations

import json
import os
import sys
import warnings

import torch
import torch_npu  # noqa: F401

sys.path.insert(0, "/root/zyg/global_max_pool/stage6")

import pyg_ascend_compat  # noqa: E402

DEV = "npu:0"
LOGDIR = os.environ.get("STAGE4_LOGS", "/root/zyg/logs/stage4")
RESULTS = []


def equal(a, b):
    if a.shape != b.shape:
        return False
    an, bn = torch.isnan(a), torch.isnan(b)
    return bool(torch.equal(an, bn) and torch.equal(a[~an], b[~bn]))


def main():
    torch.npu.set_device(0)
    pyg_ascend_compat.enable(debug=True)
    from torch_geometric.nn import global_max_pool  # noqa: PLC0415  (import AFTER enable)
    # the CPU oracle must NOT go through the compat wrapper (it would count as original_calls)
    cpu_global_max_pool = getattr(global_max_pool, "__wrapped__", global_max_pool)

    torch.manual_seed(7)
    cases = []
    cases.append(("E1_unique", torch.tensor([[1.0, 10.0], [3.0, 20.0], [2.0, 30.0]]),
                  torch.tensor([0, 0, 0]), 1, torch.ones(1, 2)))
    cases.append(("E2_two_way_tie", torch.tensor([[3.0, 1.0], [3.0, 2.0], [1.0, 3.0]]),
                  torch.tensor([0, 0, 0]), 1, torch.ones(1, 2)))
    cases.append(("E3_per_feature_ties",
                  torch.tensor([[5.0, 1.0, 7.0], [5.0, 3.0, 7.0], [2.0, 3.0, 0.0]]),
                  torch.tensor([0, 0, 0]), 1, torch.tensor([[2.0, 3.0, 4.0]])))
    cases.append(("E4_multi_group",
                  torch.tensor([[4.0, 1.0], [4.0, 2.0], [0.0, 9.0], [7.0, 7.0], [7.0, 7.0], [1.0, 0.0]]),
                  torch.tensor([0, 0, 0, 1, 1, 1]), 2, torch.tensor([[1.0, 2.0], [3.0, 4.0]])))
    cases.append(("E5_explicit_size_empty",
                  torch.tensor([[4.0, 1.0], [4.0, 2.0], [0.0, 9.0]]), torch.tensor([0, 0, 3]), 5,
                  torch.tensor([[1.0, 1.0], [5.0, 5.0], [7.0, 7.0], [11.0, 11.0], [13.0, 13.0]])))
    cases.append(("E6_pm_zero", torch.tensor([[0.0], [-0.0]]), torch.tensor([0, 0]), 1,
                  torch.ones(1, 1)))
    cases.append(("E7_neg_inf", torch.tensor([[float("-inf")], [float("-inf")]]),
                  torch.tensor([0, 0]), 1, torch.ones(1, 1)))
    cases.append(("E8_nan", torch.tensor([[float("nan")], [1.0], [2.0]]),
                  torch.tensor([0, 0, 0]), 1, torch.ones(1, 1)))

    # non-aligned F, N=41 (leftSrc) and a representative largeTail shape
    x = (torch.arange(41, dtype=torch.float32).unsqueeze(1) % 5.0).expand(41, 33).contiguous()
    x[0] = 9.0
    cases.append(("E9_N41_F33_leftsrc", x, torch.arange(41) % 8, 8, torch.ones(8, 33)))

    lt = (torch.arange(40, dtype=torch.float32).unsqueeze(1) % 7.0).expand(40, 48825).contiguous()
    lt[0, 0] = 9.0
    lt[39, 48824] = 11.0
    lt[39, 48823] = 11.0
    cases.append(("E10_largeTail_N40_F48825", lt, torch.arange(40) % 8, 8, torch.ones(8, 48825)))

    ok_all = True
    for name, x, batch, size, up in cases:
        xr = x.clone().requires_grad_(True)
        out_cpu = cpu_global_max_pool(xr, batch, size)
        (out_cpu * up).sum().backward()

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            xn = x.clone().to(DEV).requires_grad_(True)
            bn = batch.to(DEV)
            out = global_max_pool(xn, bn, size)
            (out * up.to(DEV)).sum().backward()
            torch.npu.synchronize()
        fb = [str(w.message) for w in caught if "fall back" in str(w.message)]

        f_ok = equal(out_cpu.detach(), out.detach().cpu())
        b_ok = equal(xr.grad, xn.grad.cpu())
        ok = f_ok and b_ok and not fb
        ok_all &= ok
        print(f"{name:28s} {'PASS' if ok else 'FAIL'}  forward={f_ok} backward={b_ok} "
              f"fallback={len(fb)}")
        RESULTS.append({"case": name, "result": "PASS" if ok else "FAIL", "forward": f_ok,
                        "backward": b_ok, "fallback": len(fb),
                        "grad_npu": xn.grad.detach().cpu().tolist(),
                        "grad_cpu": xr.grad.tolist()})

    st = pyg_ascend_compat.stats()
    print(f"\ncompat counters: {st}")
    counters_ok = (st["ascend_calls"] == len(cases) and st["original_calls"] == 0
                   and st["autograd_calls"] == len(cases))
    print(f"counters: ascend_calls={st['ascend_calls']} original_calls={st['original_calls']} "
          f"autograd_calls={st['autograd_calls']} -> {'PASS' if counters_ok else 'FAIL'}")
    RESULTS.append({"case": "E_counters", "result": "PASS" if counters_ok else "FAIL"})
    ok_all &= counters_ok

    failed = [r for r in RESULTS if r["result"] != "PASS"]
    print(f"\nTOTAL {len(RESULTS)}  PASS {len(RESULTS) - len(failed)}  FAIL {len(failed)}")
    os.makedirs(LOGDIR, exist_ok=True)
    with open(os.path.join(LOGDIR, "stage4_pyg_e2e.json"), "w") as fh:
        json.dump({"results": RESULTS, "counters": st}, fh, indent=1)
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
