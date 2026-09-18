#!/usr/bin/env python3
"""Stage 5A — pin the *count* arithmetic of the FP16/BF16 CPU tie-gradient.

Two candidate models for the denominator of `grad = grad_out / count`:

  M_round : count is formed as an integer and converted (round-to-nearest-even) to the input dtype
  M_inc   : count is accumulated by repeated `+1` in the input dtype (what `zeros_like(src)` +
            `scatter_add_` would do), so it can saturate differently

and two candidates for the division: performed in the input dtype, or in fp32 then cast.

For every N the measured CPU gradient is compared against all four combinations.
"""

from __future__ import annotations

import torch
from torch_geometric.nn import global_max_pool

DTYPES = {"fp16": torch.float16, "bf16": torch.bfloat16}
RANGES = {"fp16": list(range(4090, 4104)) + [2046, 2047, 2048, 2049, 2050, 8193, 16385],
          "bf16": list(range(250, 266)) + [510, 511, 512, 513, 514, 1025]}


def inc_count(dt, n):
    """Simulate repeated fp16/bf16 `+1` accumulation (scatter_add_ of ones)."""
    c = torch.tensor(0.0, dtype=dt)
    one = torch.tensor(1.0, dtype=dt)
    for _ in range(n):
        c = c + one
    return c


def models(dt, n):
    out = {}
    c_round = torch.tensor(float(n), dtype=torch.float32).to(dt)
    c_inc = inc_count(dt, n)
    for ctag, c in (("round", c_round), ("inc", c_inc)):
        # division in dtype
        out[f"M_{ctag}/div_dtype"] = float((torch.tensor(1.0, dtype=dt) /
                                            torch.tensor(float(c), dtype=dt)))
        # division in fp32 with the dtype-count, then cast
        q32 = torch.tensor(1.0, dtype=torch.float32) / torch.tensor(float(c), dtype=torch.float32)
        out[f"M_{ctag}/div_fp32"] = float(q32.to(dt))
    return out, float(c_round), float(c_inc)


def main():
    for dt_name, dt in DTYPES.items():
        print("=" * 110)
        print(f"### {dt_name} — measured vs count-model predictions (grad_out = 1, all rows tie)")
        print(f"{'N':>6} {'measured':>22} {'c_round':>10} {'c_inc':>10}  matching models")
        summary = {}
        for n in RANGES[dt_name]:
            x = torch.ones((n, 1), dtype=dt, requires_grad=True)
            b = torch.zeros(n, dtype=torch.int64)
            out = global_max_pool(x, b, 1)
            out.sum().backward()
            got = float(x.grad[0, 0])
            preds, cr, ci = models(dt, n)
            m = [k for k, v in preds.items() if v == got]
            print(f"{n:>6} {got:>22.17g} {cr:>10.1f} {ci:>10.1f}  "
                  f"{','.join(m) if m else 'NONE: ' + str({k: v for k, v in preds.items()})}")
            for k in preds:
                summary.setdefault(k, 0)
                if preds[k] == got:
                    summary[k] += 1
        print(f"\n  model hit-rate over {len(RANGES[dt_name])} cases: "
              f"{ {k: f'{v}/{len(RANGES[dt_name])}' for k, v in summary.items()} }")
        print()


if __name__ == "__main__":
    main()
