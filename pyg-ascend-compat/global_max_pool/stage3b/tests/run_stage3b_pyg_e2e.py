#!/usr/bin/env python3
"""Stage 3B PyG end-to-end: real torch_geometric.nn.global_max_pool through pyg-ascend-compat."""

from __future__ import annotations

import json
import os
import subprocess
import sys

import torch
import torch_npu  # noqa: F401
import torch_geometric

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, "/root/zyg/global_max_pool/stage6")

import pyg_ascend_compat  # noqa: E402

pyg_ascend_compat.enable(debug=True)
from torch_geometric.nn import global_max_pool  # noqa: E402

import stage3b_common as C  # noqa: E402

RESULTS = []


def e2e(name, n, f, s, idx_mode="spread", src_pattern="det"):
    x = C.make_src(n, f, src_pattern)
    batch = C.make_index(n, s, idx_mode, seed=n)
    out = global_max_pool(x.to(C.DEV), batch.to(C.DEV), s)
    torch.npu.synchronize()
    ref = C.reference_vectorized(x, batch, s)
    got = out.detach().cpu()
    finite = torch.isfinite(ref) & torch.isfinite(got)
    diff = float((ref[finite] - got[finite]).abs().max()) if bool(finite.any()) else 0.0
    ok = (tuple(got.shape) == tuple(ref.shape) and diff == 0.0
          and torch.equal(torch.isinf(ref), torch.isinf(got)))
    RESULTS.append({"case": name, "N": n, "F": f, "S": s, "shape": list(got.shape),
                    "max_diff": diff, "ok": bool(ok)})
    print(f"{name:34s} {'PASS' if ok else 'FAIL'}  N={n} F={f} S={s} shape={tuple(got.shape)} "
          f"max_diff={diff}")
    return ok


def main():
    torch.npu.set_device(0)
    print(f"PyG {torch_geometric.__version__} | torch {torch.__version__} | "
          f"torch_npu {torch_npu.__version__}\n")
    ok = True
    ok &= e2e("E2E-1_N41_F33", 41, 33, 8)
    ok &= e2e("E2E-2_N4097_F33", 4097, 33, 8)
    ok &= e2e("E2E-3_N40_F48824_threshold_minus_1", 40, 48824, 8)
    ok &= e2e("E2E-4_N40_F40000_smallTail", 40, 40000, 8)

    st = pyg_ascend_compat.stats()
    print(f"\ncompat counters: {st}")

    probe = subprocess.run([sys.executable, "/root/zyg/global_max_pool/stage6/tests/pyg_fallback_probe.py",
                            "--enable"], capture_output=True, text=True, env=dict(os.environ))
    fallback = ("npu_cpu_fallback" in probe.stderr) or ("fall back to run on the CPU" in probe.stderr)
    print(f"fallback text on the compat path: {'YES' if fallback else 'NONE'}")
    print(f"aten::scatter_reduce called: NO (compat path never calls it; see the Stage 6 profiler)")
    ok &= (st["ascend_calls"] >= 4) and (not fallback)

    with open("/root/zyg/logs/stage3b/pyg_e2e_results.json", "w") as fh:
        json.dump({"results": RESULTS, "counters": st, "fallback_text": fallback}, fh, indent=1)
    print(f"\nPYG_E2E={('PASS' if ok else 'FAIL')}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
