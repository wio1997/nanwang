#!/usr/bin/env python3
"""Stage 4 / section 7: what does ``global_max_pool(x, batch=None)`` do — and is the NPU path native?

PyG 2.8.0.post1 source (torch_geometric/nn/pool/glob.py):

    dim = -1 if x.dim() == 1 else -2
    if batch is None:
        return x.max(dim=dim, keepdim=x.dim() <= 2)[0]
    return scatter(x, batch, dim=dim, dim_size=size, reduce='max')

So batch=None is NOT the scatter path at all; it is a plain ``x.max``.  This probe records:
  * CPU forward/backward (the oracle),
  * NPU forward/backward through the ORIGINAL PyG path (compat disabled),
  * whether the compat wrapper intercepts it (compat enabled: it must delegate),
  * host-CPU-fallback warnings on the NPU path.
"""

from __future__ import annotations

import os
import sys
import warnings

import torch
import torch_npu  # noqa: F401

sys.path.insert(0, "/root/zyg/global_max_pool/stage6")

DEV = "npu:0"


def run(x, name):
    print(f"--- {name}")
    xc = x.clone().requires_grad_(True)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        from torch_geometric.nn import global_max_pool
        outc = global_max_pool(xc, None)
        outc.sum().backward()
        cpu_out, cpu_grad = outc.detach(), xc.grad
    cpu_fb = [str(w.message) for w in caught if "fall back" in str(w.message)]

    xn = x.clone().to(DEV).requires_grad_(True)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        outn = global_max_pool(xn, None)
        torch.npu.synchronize()
        outn.sum().backward()
        torch.npu.synchronize()
    npu_fb = [str(w.message) for w in caught if "fall back" in str(w.message)]
    n_out = outn.detach().cpu()
    n_grad = xn.grad.detach().cpu()

    f_ok = torch.equal(cpu_out, n_out)
    g_ok = torch.equal(cpu_grad, n_grad)
    print(f"    CPU out {tuple(cpu_out.shape)} {cpu_out.tolist()}")
    print(f"    NPU out {tuple(n_out.shape)} {n_out.tolist()}   forward_equal={f_ok}")
    print(f"    CPU grad {cpu_grad.tolist()}")
    print(f"    NPU grad {n_grad.tolist()}   backward_equal={g_ok}")
    print(f"    CPU fallback warnings: cpu={len(cpu_fb)} npu={len(npu_fb)}")
    for m in npu_fb:
        print(f"      warn: {m.strip()[:140]}")
    return f_ok, g_ok, len(npu_fb)


def main():
    torch.npu.set_device(0)
    torch.manual_seed(0)
    ok = True
    ok &= all(run(torch.tensor([[1.0, 10.0], [3.0, 20.0], [2.0, 30.0]]), "unique max (2-D)")[:2])
    ok &= all(run(torch.tensor([[3.0, 1.0], [3.0, 2.0], [1.0, 3.0]]), "tie (2-D)")[:2])
    ok &= all(run(torch.tensor([1.0, 3.0, 2.0]), "1-D input")[:2])
    ok &= all(run(torch.tensor([[float("-inf")], [float("-inf")], [5.0]]), "-inf 2-D")[:2])

    print("\n### compat enabled: does the wrapper intercept batch=None?")
    import pyg_ascend_compat
    pyg_ascend_compat.enable(debug=True)
    pyg_ascend_compat.reset_stats()
    from torch_geometric.nn import global_max_pool  # noqa: PLC0415
    x = torch.tensor([[1.0], [3.0], [2.0]], device=DEV, requires_grad=True)
    out = global_max_pool(x, None)
    out.sum().backward()
    torch.npu.synchronize()
    st = pyg_ascend_compat.stats()
    print(f"    compat stats: total={st['total_calls']} ascend={st['ascend_calls']} "
          f"original={st['original_calls']} rejected={st['requires_grad_rejected']}")
    print(f"    out={out.detach().cpu().tolist()} grad={x.grad.detach().cpu().tolist()}")
    print(f"    delegated to original PyG for batch=None: "
          f"{st['original_calls'] == 1 and st['ascend_calls'] == 0}")
    print(f"\nRESULT {'PASS' if ok else 'FAIL'} (batch=None parity CPU vs NPU)")


if __name__ == "__main__":
    main()
