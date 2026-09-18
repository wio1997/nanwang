#!/usr/bin/env python3
"""Stage 5A — decisive measurement: is the FP16/BF16 tie division done in the dtype or in fp32?

Uses *exact* fp16/bf16-representable upstream gradients and efficient search over tie counts to
find (grad_out, count) pairs where the two candidates differ, then measures the real CPU oracle
for those shapes.
"""

from __future__ import annotations

from fractions import Fraction

import numpy as np
import torch
from torch_geometric.nn import global_max_pool


def exact_repr(dtype_np, value: float) -> Fraction:
    """Exact rational value of the dtype-rounded number."""
    return Fraction(float(dtype_np(value)))


def round_np(dtype_np, q: Fraction):
    return dtype_np(float(q)).item()


def round_bf16(q: Fraction):
    return torch.tensor(float(q)).to(torch.bfloat16).item()


def find_pairs(grads_exact, counts, rounds, limit=3):
    hits = []
    for g in grads_exact:
        for c in counts:
            q = g / Fraction(c)
            direct = rounds(q)
            via32 = rounds(Fraction(np.float32(float(q)).item()))
            if float(direct) != float(via32):
                hits.append((g, c, direct, via32))
                if len(hits) >= limit:
                    return hits
    return hits


def oracle(dt, g, c):
    x = torch.ones((c, 1), dtype=dt, requires_grad=True)
    b = torch.zeros(c, dtype=torch.int64)
    out = global_max_pool(x, b, 1)
    up = torch.tensor([[g]], dtype=dt)
    (out * up).sum().backward()
    return float(x.grad[0, 0])


def main():
    fp16_grads = [exact_repr(np.float16, v) for v in
                  (1.7, 1.3, 0.3, 2.3, 3.1, 5.7, 7.3, 11.1, 13.7, 23.5, 1.1, 0.7, 0.9, 1.9)]
    fp16_counts = list(range(1, 6000))
    hits = find_pairs(fp16_grads, fp16_counts, lambda q: round_np(np.float16, q))
    print("## fp16 discriminating pairs")
    for g, c, d, v in hits:
        got = oracle(torch.float16, float(g), c)
        match = ("dtype_divide" if got == d else "") + (",fp32_then_cast" if got == v else "")
        print(f"   grad={float(g)!r} count={c}: dtype_div={d!r} fp32_then_cast={v!r} "
              f"CPU_measured={got!r} -> {match or 'NEITHER'}")

    torch.manual_seed(0)
    bf16_grads = []
    for v in (1.7, 1.3, 0.3, 2.3, 3.1, 5.7, 7.3, 11.1, 13.7, 23.5, 1.1, 0.7, 0.9, 1.9):
        bf16_grads.append(Fraction(torch.tensor(v).to(torch.bfloat16).item()))
    bf16_counts = list(range(1, 20000))
    hits = find_pairs(bf16_grads, bf16_counts, round_bf16)
    print("## bf16 discriminating pairs")
    if not hits:
        print("   none found in 1..19999 x 14 grads")
    for g, c, d, v in hits:
        got = oracle(torch.bfloat16, float(g), c)
        match = ("dtype_divide" if got == d else "") + (",fp32_then_cast" if got == v else "")
        print(f"   grad={float(g)!r} count={c}: dtype_div={d!r} fp32_then_cast={v!r} "
              f"CPU_measured={got!r} -> {match or 'NEITHER'}")


if __name__ == "__main__":
    main()
