#!/usr/bin/env python3
"""Stage 4: executable spec of the PyG/PyTorch tie-gradient contract + corpus validation.

Model (derived from cpu_gradient_oracle.py / cpu_tie_quirk_probe.py):

    out[g,f]   = max over { x[i,f] : batch[i] == g }        (empty group -> 0)
    winner     = (x == out[batch])
    count[g,f] = sum_i winner[i,f] over the group + (1 if out[g,f] == 0 else 0)
    grad_x     = winner * (grad_out[batch] / count[batch])

The "+ (out == 0)" term is the PyTorch scatter_reduce(..., include_self=False) backward quirk: the
excluded zero self slot is still counted whenever it equals the reduced value.  NaN groups
naturally become 0 * (grad/0) = nan, which is what the CPU oracle produces.

This script validates the model against the real PyG API on a randomized corpus (ties, zeros,
negatives, -inf, NaN, multiple groups, explicit size, weighted upstream).

Run: python3 cpu_contract_model.py
"""

from __future__ import annotations

import json
import random

import torch
from torch_geometric.nn import global_max_pool


def model_grad(x: torch.Tensor, batch: torch.Tensor, size: int, grad_out: torch.Tensor):
    """Reference implementation of the frozen contract (CPU, fp32)."""
    n, f = x.shape
    out = torch.zeros((size, f), dtype=torch.float32)
    occupied = torch.zeros(size, dtype=torch.bool)
    for g in range(size):
        m = batch == g
        if bool(m.any()):
            out[g] = x[m].max(dim=0).values
            occupied[g] = True
    out = torch.where(occupied.view(size, 1), out, torch.zeros_like(out))
    winner = (x == out[batch])                                    # [N, F]
    count = torch.zeros((size, f), dtype=torch.float32)
    count.index_add_(0, batch, winner.to(torch.float32))
    count = count + (out == 0).to(torch.float32)                  # include_self=False quirk
    scaled = grad_out / count                                     # 0/0 -> nan, x/0 -> inf
    return winner.to(torch.float32) * scaled[batch], out


def real_grad(x: torch.Tensor, batch: torch.Tensor, size, grad_out: torch.Tensor):
    xr = x.clone().requires_grad_(True)
    out = global_max_pool(xr, batch, size)
    (out * grad_out).sum().backward()
    return xr.grad, out.detach()


def same(a, b):
    """Equality where NaN == NaN counts as equal (semantic comparison)."""
    if a.shape != b.shape:
        return False
    an, bn = torch.isnan(a), torch.isnan(b)
    if not torch.equal(an, bn):
        return False
    fin = ~an
    return bool(torch.equal(a[fin], b[fin]))


def random_case(rng, kind):
    n = rng.randint(1, 12)
    f = rng.choice([1, 2, 3, 7, 8, 9])
    s = rng.randint(1, 4)
    pool = {
        "small": [0.0, 1.0, 2.0, 3.0, -1.0, -2.0],
        "with_zero": [0.0, -0.0, 0.0, 1.0, -1.0],
        "tie": [5.0, 5.0, 5.0, 1.0],
        "neg_inf": [float("-inf"), float("-inf"), -1.0, 0.0],
        "nan": [float("nan"), float("nan"), 0.0, 1.0, -1.0],
    }[kind]
    x = torch.tensor([[rng.choice(pool) for _ in range(f)] for _ in range(n)], dtype=torch.float32)
    batch = torch.tensor([rng.randrange(s) for _ in range(n)], dtype=torch.int64)
    grad_out = torch.tensor([[rng.choice([0.0, 1.0, -1.0, 2.5, 0.0]) for _ in range(f)]
                             for _ in range(s)], dtype=torch.float32)
    return x, batch, s, grad_out


def main():
    rng = random.Random(20260918)
    torch.manual_seed(20260918)
    mismatches = []
    total = 0
    for kind in ("small", "with_zero", "tie", "neg_inf", "nan"):
        n_ok = n_bad = 0
        for _ in range(60):
            total += 1
            x, batch, s, grad_out = random_case(rng, kind)
            g_model, out_model = model_grad(x, batch, s, grad_out)
            g_real, out_real = real_grad(x, batch, s, grad_out)
            if same(g_model, g_real) and same(out_model, out_real):
                n_ok += 1
            else:
                n_bad += 1
                mismatches.append({
                    "kind": kind,
                    "x": x.tolist(), "batch": batch.tolist(), "size": s,
                    "grad_out": grad_out.tolist(),
                    "forward_real": out_real.tolist(), "forward_model": out_model.tolist(),
                    "grad_real": g_real.tolist(), "grad_model": g_model.tolist(),
                })
        print(f"{kind:10s} ok={n_ok:3d} bad={n_bad:3d}")

    print()
    if not mismatches:
        print(f"CONTRACT MODEL: exact match with the real PyG API on all {total} random cases")
    else:
        print(f"CONTRACT MODEL: {len(mismatches)} mismatch(es) out of {total}")
        print(json.dumps(mismatches[0], indent=1)[:2000])

    print("\nNaN group with zero upstream:")
    xx = torch.tensor([[float("nan")], [1.0]], requires_grad=True)
    b = torch.tensor([0, 0])
    out = global_max_pool(xx, b, 1)
    (out * torch.tensor([[0.0]])).sum().backward()
    gm, _ = model_grad(torch.tensor([[float("nan")], [1.0]]), b, 1, torch.tensor([[0.0]]))
    print(f"  real ={xx.grad.tolist()}")
    print(f"  model={gm.tolist()}")

    print("\n+inf group (ties among +inf):")
    xi = torch.tensor([[float("inf")], [float("inf")], [1.0]], requires_grad=True)
    b = torch.tensor([0, 0, 0])
    global_max_pool(xi, b, 1).sum().backward()
    gm, _ = model_grad(torch.tensor([[float("inf")], [float("inf")], [1.0]]), b, 1,
                       torch.tensor([[1.0]]))
    print(f"  real ={xi.grad.tolist()}")
    print(f"  model={gm.tolist()}")


if __name__ == "__main__":
    main()
