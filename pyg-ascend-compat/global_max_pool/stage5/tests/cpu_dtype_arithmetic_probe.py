#!/usr/bin/env python3
"""Stage 5A — which internal arithmetic reproduces the CPU tie-gradient for FP16 / BF16?

The observable difference: `grad = grad_out / count`.  If the count is accumulated/rounded in the
input dtype, large counts round (fp16: integers > 2048, bf16: integers > 256), which changes the
result; if everything is computed in fp32 and only the final value is cast, the result is the
correctly-rounded dtype value of the fp32 quotient.

All-equal rows with a single group make the count exactly N, so N controls the denominator.
"""

from __future__ import annotations

import torch
from torch_geometric.nn import global_max_pool

DTYPES = {"fp16": torch.float16, "bf16": torch.bfloat16}
COUNTS = (2, 3, 7, 122, 256, 257, 2048, 2049, 4096, 4097, 8193)


def predict(dt, n, mode):
    """Predict grad for N identical rows in one group, grad_out = 1, count = N (+0 quirk)."""
    t = torch.tensor([1.0], dtype=dt)
    if mode == "dtype_all":
        c = torch.tensor([float(n)], dtype=dt)
        return float((t / c)[0])
    if mode == "fp32_then_cast":
        c32 = torch.tensor([float(n)], dtype=torch.float32)
        q = torch.tensor([1.0], dtype=torch.float32) / c32        # [1]
        return float(q.to(dt)[0])
    if mode == "dtype_count_fp32_div":
        c_dt = torch.tensor([float(n)], dtype=dt)
        c32 = c_dt.to(torch.float32)
        return float((torch.tensor([1.0], dtype=torch.float32) / c32).to(dt)[0])
    raise ValueError(mode)


def main():
    for dt_name, dt in DTYPES.items():
        print("=" * 96)
        print(f"### {dt_name}")
        print(f"{'count':>6} {'measured grad':>22} {'dtype_all':>22} {'fp32_then_cast':>22} "
              f"{'dtype_count_fp32_div':>22}   match")
        for n in COUNTS:
            x = torch.ones((n, 1), dtype=dt, requires_grad=True)
            b = torch.zeros(n, dtype=torch.int64)
            out = global_max_pool(x, b, 1)
            out.sum().backward()
            got = float(x.grad[0, 0])
            p1 = predict(dt, n, "dtype_all")
            p2 = predict(dt, n, "fp32_then_cast")
            p3 = predict(dt, n, "dtype_count_fp32_div")
            matches = [name for name, p in (("dtype_all", p1), ("fp32_then_cast", p2),
                                            ("dtype_count_fp32_div", p3)) if p == got]
            print(f"{n:>6} {got:>22.17g} {p1:>22.17g} {p2:>22.17g} {p3:>22.17g}   "
                  f"{','.join(matches) if matches else 'NONE'}")

        # does the *count* saturate/flush for very large N? (also probes the +0 quirk path)
        n = 4097
        x = torch.zeros((n, 1), dtype=dt, requires_grad=True)   # zero max -> +1 quirk
        b = torch.zeros(n, dtype=torch.int64)
        out = global_max_pool(x, b, 1)
        out.sum().backward()
        got = float(x.grad[0, 0])
        preds = {}
        for tag, cnt in (("count=n+1 fp32", torch.tensor([float(n + 1)], dtype=torch.float32)),
                         ("count=n+1 dtype", torch.tensor([float(n + 1)], dtype=dt))):
            preds[tag] = float((torch.tensor([1.0], dtype=torch.float32) / cnt.to(torch.float32))
                               .to(dt)[0])
        print(f"zero-max quirk, N={n}: measured={got:.17g}  fp32count_pred={preds['count=n+1 fp32']:.17g}"
              f"  dtypecount_pred={preds['count=n+1 dtype']:.17g}")
        print()


if __name__ == "__main__":
    main()
