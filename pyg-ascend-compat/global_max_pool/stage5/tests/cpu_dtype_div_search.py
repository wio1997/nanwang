#!/usr/bin/env python3
"""Stage 5A — find (grad_out, count) pairs where dtype-division and fp32-division-then-cast differ.

If such a pair exists and the CPU oracle picks one side, the division precision is pinned;
otherwise the two are observationally equivalent for representable inputs.

Exact rounding is done with Fraction (no double-rounding artefacts in the search itself).
"""

from __future__ import annotations

from fractions import Fraction

import numpy as np


def round_to(dtype, q: Fraction):
    """Round a rational to the given numpy float type via float64 (exact enough for these ranges)."""
    return dtype(float(q)).item()


def search(dtype, counts, grads, limit=5):
    hits = []
    for c in counts:
        cf = Fraction(c)
        for g in grads:
            q = Fraction(g) / cf
            direct = round_to(dtype, q)                       # dtype-rounded quotient
            via32 = round_to(dtype, Fraction(round_to(np.float32, q)))   # fp32 then dtype
            if np.float64(direct) != np.float64(via32):
                hits.append((g, c, direct, via32))
                if len(hits) >= limit:
                    return hits
    return hits


def main():
    fp16_grads = [0.3, 0.7, 1.0, 1.3, 1.7, 2.3, 3.1, 5.7, 7.3, 11.1, 13.7, 23.5, 1e-2, 5e-2]
    fp16_counts = list(range(1, 3000)) + [4096, 4097, 5000, 8192, 8193]
    print("## fp16: dtype-divide vs fp32-divide-then-cast")
    hits = search(np.float16, fp16_counts, fp16_grads)
    for g, c, d, v in hits:
        print(f"   grad={g} count={c}: dtype_div={d!r}  fp32_then_cast={v!r}")
    if not hits:
        print("   no discriminating pair found in the searched domain")

    bf16_grads = fp16_grads
    bf16_counts = list(range(1, 1200)) + [4096, 4097, 8192, 10000]
    print("## bf16: dtype-divide vs fp32-divide-then-cast")
    hits = search(np.float32, [1], [1], limit=0)  # noop to keep bfloat16 import path simple
    import torch
    def round_bf16(q: Fraction):
        return torch.tensor(float(q)).to(torch.bfloat16).item()
    found = []
    for c in bf16_counts:
        cf = Fraction(c)
        for g in bf16_grads:
            q = Fraction(g) / cf
            direct = round_bf16(q)
            via32 = round_bf16(Fraction(np.float32(float(q)).item()))
            if direct != via32:
                found.append((g, c, direct, via32))
                if len(found) >= 5:
                    break
        if len(found) >= 5:
            break
    for g, c, d, v in found:
        print(f"   grad={g} count={c}: dtype_div={d!r}  fp32_then_cast={v!r}")
    if not found:
        print("   no discriminating pair found in the searched domain")


if __name__ == "__main__":
    main()
