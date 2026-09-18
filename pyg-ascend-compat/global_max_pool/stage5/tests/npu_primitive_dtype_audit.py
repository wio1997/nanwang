#!/usr/bin/env python3
"""Stage 5A / section 4.7-4.8 — NPU primitive support for FP16 and BF16.

For every primitive the Stage 4 backward design uses (and the cast-based forward needs), run it on
NPU in fp16 and bf16 and record:
  * host-CPU-fallback warnings (torch_npu `npu_cpu_fallback`)
  * device-vs-CPU value agreement (exact / max ULP / max abs diff)
  * dtype preservation

Run: python3 npu_primitive_dtype_audit.py
"""

from __future__ import annotations

import json
import os
import warnings

import torch
import torch_npu  # noqa: F401

DEV = "npu:0"
DTYPES = {"fp16": torch.float16, "bf16": torch.bfloat16}
RESULTS = []


def ulp_diff(a_cpu: torch.Tensor, b_cpu: torch.Tensor, dt):
    """Max ULP distance between two same-dtype CPU tensors (NaN pattern must match)."""
    an, bn = torch.isnan(a_cpu), torch.isnan(b_cpu)
    if not torch.equal(an, bn):
        return None, "nan pattern differs"
    fin = ~an
    a, b = a_cpu[fin], b_cpu[fin]
    if a.numel() == 0:
        return 0, "all nan"
    ai = a.view(torch.int16).to(torch.int32) if dt == torch.float16 else None
    if dt == torch.float16:
        bi = b.view(torch.int16).to(torch.int32)
    else:  # bfloat16 -> widen to fp32 for a monotone ULP-ish measure
        ai = a.float().view(torch.int32).to(torch.int64)
        bi = b.float().view(torch.int32).to(torch.int64)
    d = (ai - bi).abs()
    return int(d.max().item()), ""


def check(name, dt, fn_cpu, fn_npu, tag: str):
    out_cpu = fn_cpu()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        out_npu = fn_npu()
        torch.npu.synchronize()
    fb = [str(w.message) for w in caught if "fall back" in str(w.message)]
    same_dtype = torch.is_tensor(out_npu) and out_npu.dtype == out_cpu.dtype
    o_cpu = out_cpu.detach().cpu() if torch.is_tensor(out_cpu) else out_cpu
    o_npu = out_npu.detach().cpu() if torch.is_tensor(out_npu) else out_npu
    if torch.is_tensor(o_cpu):
        exact = bool(torch.equal(o_cpu, o_npu))
        ulp, note = ulp_diff(o_cpu, o_npu, dt)
        maxabs = float((o_cpu.float() - o_npu.float()).abs().max().item())
    else:
        exact, ulp, note, maxabs = True, 0, "", 0.0
    rec = {"dtype": tag, "op": name, "fallback": len(fb), "dtype_ok": same_dtype,
           "exact": exact, "max_ulp": ulp, "max_abs_diff": maxabs, "note": note}
    RESULTS.append(rec)
    print(f"[{tag}] {name:34s} fallback={len(fb)} dtype_ok={same_dtype} exact={exact} "
          f"max_ulp={ulp} max_abs={maxabs:.3g} {note}")
    return rec


def main():
    torch.npu.set_device(0)
    torch.manual_seed(5)
    n, f, s = 64, 8, 4
    for tag, dt in DTYPES.items():
        print(f"\n########## {tag} ##########")
        xc = (torch.randn(n, f) * 2).to(dt)
        xc[0, 0] = 3.5
        xc[1, 0] = 3.5
        xn = xc.to(DEV)
        bc = (torch.arange(n) % s)
        bn = bc.to(DEV)
        oc = (torch.randn(s, f) * 2).to(dt)
        on = oc.to(DEV)

        # forward-path primitive: cast up / down
        check("cast_up_to_fp32", dt, lambda: xc.float(), lambda: xn.float(), tag)
        check("cast_down_to_dtype", dt, lambda: xc.float().to(dt), lambda: xn.float().to(dt), tag)

        # backward primitives
        check("gather(out[batch])", dt, lambda: oc.index_select(0, bc),
              lambda: on.index_select(0, bn), tag)
        check("equal(x == out_rows)", dt, lambda: (xc == oc.index_select(0, bc)),
              lambda: (xn == on.index_select(0, bn)), tag)
        check("cast(bool)->dtype", dt, lambda: (xc == 0).to(dt),
              lambda: (xn == 0).to(dt), tag)
        check("index_add(ones)", dt,
              lambda: torch.zeros(s, f, dtype=dt).index_add_(0, bc, torch.ones(n, f, dtype=dt)),
              lambda: torch.zeros(s, f, dtype=dt, device=DEV).index_add_(0, bn,
                                                                        torch.ones(n, f, dtype=dt,
                                                                                   device=DEV)),
              tag)
        check("divide(dtype)", dt, lambda: xc / (0.5 + torch.arange(f, dtype=dt)),
              lambda: xn / (0.5 + torch.arange(f, dtype=dt, device=DEV)), tag)
        check("multiply(dtype)", dt, lambda: xc * xc, lambda: xn * xn, tag)
        check("masked_fill(nan)", dt,
              lambda: xc.masked_fill((xc == 0), float("nan")),
              lambda: xn.masked_fill((xn == 0), float("nan")), tag)
        check("zeros_like(dtype)", dt, lambda: torch.zeros_like(xc), lambda: torch.zeros_like(xn), tag)
        check("sum/ReduceSum(dtype)", dt, lambda: xc.sum(0), lambda: xn.sum(0), tag)

        # division precision of the dtype (vs CPU dtype division)
        num_cpu = torch.rand(4096, dtype=dt) + 0.5
        den_cpu = torch.randint(1, 5000, (4096,), dtype=torch.float32).to(dt)
        den_cpu = torch.clamp(den_cpu, min=1)
        check("divide vs CPU (ULP)", dt, lambda: num_cpu / den_cpu,
              lambda: (num_cpu.to(DEV) / den_cpu.to(DEV)), tag)

    os.makedirs("/root/zyg/logs/stage5", exist_ok=True)
    with open("/root/zyg/logs/stage5/npu_primitive_dtype_audit.json", "w") as fh:
        json.dump(RESULTS, fh, indent=1)
    bad = [r for r in RESULTS if r["fallback"] or not r["dtype_ok"]]
    print(f"\nGATE no fallback and dtype preserved for all audited primitives: "
          f"{'PASS' if not bad else 'FAIL ' + str(bad)}")


if __name__ == "__main__":
    main()
