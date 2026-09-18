#!/usr/bin/env python3
"""Stage 5 — real PyG API end-to-end for FP16 / BF16 (forward, requires_grad=False and True).

    import pyg_ascend_compat; pyg_ascend_compat.enable()
    from torch_geometric.nn import global_max_pool
    out = global_max_pool(x, batch, size)      # x fp16/bf16 on NPU
    out.sum().backward()
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
LOGDIR = os.environ.get("STAGE5_LOGS", "/root/zyg/logs/stage5")
RESULTS = []


def eq_dtype(a, b):
    an, bn = torch.isnan(a), torch.isnan(b)
    if not torch.equal(an, bn):
        return False
    fin = ~an
    return bool(torch.equal(a[fin], b[fin]))


def main():
    torch.npu.set_device(0)
    pyg_ascend_compat.enable(debug=True)
    from torch_geometric.nn import global_max_pool  # noqa: PLC0415
    cpu_gmp = getattr(global_max_pool, "__wrapped__", global_max_pool)

    cases = []
    for dt in (torch.float16, torch.bfloat16):
        tag = str(dt).replace("torch.", "")
        cases += [
            (f"{tag}_unique", torch.tensor([[1.0, 10.0], [3.0, 20.0], [2.0, 30.0]]), torch.tensor([0, 0, 0]), 1, torch.ones(1, 2), dt),
            (f"{tag}_tie2", torch.tensor([[3.0, 1.0], [3.0, 2.0], [1.0, 3.0]]), torch.tensor([0, 0, 0]), 1, torch.ones(1, 2), dt),
            (f"{tag}_tie3", torch.tensor([[5.0], [5.0], [5.0]]), torch.tensor([0, 0, 0]), 1, torch.ones(1, 1), dt),
            (f"{tag}_per_feature", torch.tensor([[5.0, 1.0, 7.0], [5.0, 3.0, 7.0], [2.0, 3.0, 0.0]]), torch.tensor([0, 0, 0]), 1, torch.tensor([[2.0, 3.0, 4.0]]), dt),
            (f"{tag}_multi_group", torch.tensor([[4.0, 1.0], [4.0, 2.0], [0.0, 9.0], [7.0, 7.0], [7.0, 7.0], [1.0, 0.0]]), torch.tensor([0, 0, 0, 1, 1, 1]), 2, torch.tensor([[1.0, 2.0], [3.0, 4.0]]), dt),
            (f"{tag}_empty_explicit", torch.tensor([[4.0, 1.0], [4.0, 2.0], [0.0, 9.0]]), torch.tensor([0, 0, 3]), 5, torch.full((5, 2), 2.0), dt),
            (f"{tag}_pm_zero", torch.tensor([[0.0], [-0.0]]), torch.tensor([0, 0]), 1, torch.ones(1, 1), dt),
            (f"{tag}_neg_inf", torch.tensor([[float("-inf")], [float("-inf")]]), torch.tensor([0, 0]), 1, torch.ones(1, 1), dt),
            (f"{tag}_nan", torch.tensor([[float("nan")], [1.0], [2.0]]), torch.tensor([0, 0, 0]), 1, torch.ones(1, 1), dt),
        ]
        x = ((torch.arange(41, dtype=torch.float32).unsqueeze(1) % 5.0).expand(41, 33).contiguous())
        x[0] = 9.0
        cases.append((f"{tag}_N41_F33_leftsrc", x, torch.arange(41) % 8, 8, torch.ones(8, 33), dt))
        lt = ((torch.arange(40, dtype=torch.float32).unsqueeze(1) % 7.0).expand(40, 48825).contiguous())
        lt[0, 0] = 9.0
        lt[39, 48824] = 11.0
        lt[39, 48823] = 11.0
        cases.append((f"{tag}_largeTail_N40_F48825", lt, torch.arange(40) % 8, 8, torch.ones(8, 48825), dt))

    ok_all = True
    for name, x, batch, size, up, dt in cases:
        x = x.to(dt)
        up = up.to(dt)
        xr = x.clone().requires_grad_(True)
        out_c = cpu_gmp(xr, batch, size)
        (out_c * up).sum().backward()
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            xn = x.clone().to(DEV).requires_grad_(True)
            out = global_max_pool(xn, batch.to(DEV), size)
            (out * up.to(DEV)).sum().backward()
            torch.npu.synchronize()
        fb = [str(w.message) for w in caught if "fall back" in str(w.message)]
        f_ok = eq_dtype(out.detach().cpu(), out_c.detach())
        g_ok = eq_dtype(xn.grad.cpu(), xr.grad)
        dtype_ok = out.dtype == dt and xn.grad.dtype == dt
        ok = f_ok and g_ok and dtype_ok and not fb
        ok_all &= ok
        print(f"{name:32s} {'PASS' if ok else 'FAIL'}  fwd={f_ok} bwd={g_ok} "
              f"out_dtype={out.dtype} grad_dtype={xn.grad.dtype} fallback={len(fb)}")
        RESULTS.append({"case": name, "result": "PASS" if ok else "FAIL", "forward": f_ok,
                        "backward": g_ok, "dtype_ok": dtype_ok, "fallback": len(fb)})

    # requires_grad=False through the same public API
    for dt in (torch.float16, torch.bfloat16):
        tag = str(dt).replace("torch.", "")
        x = torch.tensor([[3.0, 1.0], [3.0, 2.0], [1.0, 3.0]], dtype=dt)
        b = torch.tensor([0, 0, 0])
        ref = cpu_gmp(x, b, 1)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            got = global_max_pool(x.to(DEV), b.to(DEV), 1)
            torch.npu.synchronize()
        fb = [str(w.message) for w in caught if "fall back" in str(w.message)]
        ok = eq_dtype(got.detach().cpu(), ref.detach()) and got.dtype == dt and got.grad_fn is None
        ok_all &= ok
        print(f"{tag}_requires_grad_false        {'PASS' if ok else 'FAIL'}  dtype={got.dtype} "
              f"grad_fn={got.grad_fn} fallback={len(fb)}")
        RESULTS.append({"case": f"{tag}_no_grad_forward", "result": "PASS" if ok else "FAIL"})

    st = pyg_ascend_compat.stats()
    n_auto = len([c for c in cases])
    counters_ok = (st["original_calls"] == 0 and st["dtype16_autograd_calls"] == n_auto
                   and st["dtype16_forward_calls"] == 2 and st["ascend_calls"] == n_auto + 2)
    print(f"\ncompat counters: {st}")
    print(f"counters -> {'PASS' if counters_ok else 'FAIL'} "
          f"(autograd={st['dtype16_autograd_calls']}/{n_auto}, forward={st['dtype16_forward_calls']}/2, "
          f"original={st['original_calls']})")
    RESULTS.append({"case": "E_counters", "result": "PASS" if counters_ok else "FAIL"})
    ok_all &= counters_ok

    failed = [r for r in RESULTS if r["result"] != "PASS"]
    print(f"\nTOTAL {len(RESULTS)}  PASS {len(RESULTS) - len(failed)}  FAIL {len(failed)}")
    os.makedirs(LOGDIR, exist_ok=True)
    with open(os.path.join(LOGDIR, "stage5_pyg_e2e.json"), "w") as fh:
        json.dump({"results": RESULTS, "counters": st}, fh, indent=1)
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
