#!/usr/bin/env python3
"""Stage 3A: FP32 special-value semantics of the adapter vs PyTorch CPU.

We do NOT define NaN semantics ourselves - we record both sides:
  * CPU reference: per-bin torch.max over the scattered rows (mathematical max)
  * CPU reference: torch.scatter_reduce(..., reduce="amax", include_self=False) (PyTorch's own rule)
  * Ascend: global_max_pool_ascend (raw path and padded path)
"""

import os
import struct
import sys

import torch
import torch_npu  # noqa: F401

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.normpath(os.path.join(HERE, "..", "..", "stage2", "python")))

from global_max_pool_ascend import global_max_pool_ascend  # noqa: E402

INF = float("inf")
NAN = float("nan")
DEV = "npu:0"


def fmt(v):
    if v != v:
        return "nan"
    if v == INF:
        return "+inf"
    if v == -INF:
        return "-inf"
    if v == 0.0:
        return "-0.0" if str(v)[0] == "-" else "+0.0"
    return f"{v:g}"


def same(a, b):
    """Bitwise comparison, with NaN == NaN treated as equal (semantic equality)."""
    if a != a and b != b:
        return True
    try:
        return struct.pack("<f", a) == struct.pack("<f", b)
    except struct.error:
        return a == b


def main():
    torch.npu.set_device(0)
    pairs = [
        ("inf_-inf", INF, -INF),
        ("inf_nan", INF, NAN),
        ("nan_-inf", NAN, -INF),
        ("nan_1", NAN, 1.0),
        ("+0_-0", 0.0, -0.0),
        ("-0_1", -0.0, 1.0),
        ("-inf_-inf", -INF, -INF),
        ("finite_finite", 3.0, -2.0),
    ]
    f = 7  # non-aligned: also proves the -inf padding columns do not mask NaN/-inf results
    batch = torch.tensor([0, 0], dtype=torch.int64)
    print(f"F={f} (padded to 8), batch=[0,0], size=1, feature 0 = a, feature 1 = b\n")
    print(f"{'case':14s} {'a':>7s} {'b':>7s} | {'cpu max':>8s} {'cpu amax':>9s} | "
          f"{'ascend raw':>11s} {'ascend pad':>11s} | match")
    all_match = True
    for name, a, b in pairs:
        x = torch.zeros((2, f), dtype=torch.float32)
        x[0, 0] = a
        x[1, 0] = b
        x[:, 1] = torch.tensor([-1.0, -2.0])  # a finite feature so padding is distinguishable

        cpu_max = float(torch.max(x[:, 0]))
        out_cpu = torch.full((1, f), -INF)
        index2d = batch.view(-1, 1).expand(-1, f)
        out_cpu.scatter_reduce_(0, index2d, x, reduce="amax", include_self=False)
        cpu_amax = float(out_cpu[0, 0])

        x_dev = x.to(DEV)
        b_dev = batch.to(DEV)
        raw = global_max_pool_ascend(x_dev, b_dev, 1, alignment_mode="raw").detach().cpu()
        pad = global_max_pool_ascend(x_dev, b_dev, 1, alignment_mode="pad").detach().cpu()
        torch.npu.synchronize()
        raw_v = float(raw[0, 0])
        pad_v = float(pad[0, 0])

        match = same(raw_v, pad_v) and same(raw_v, cpu_max)
        all_match = all_match and match
        print(f"{name:14s} {fmt(a):>7s} {fmt(b):>7s} | {fmt(cpu_max):>8s} {fmt(cpu_amax):>9s} | "
              f"{fmt(raw_v):>11s} {fmt(pad_v):>11s} | {'yes' if match else 'NO'}")

    print("\nraw == padded on every case:", "yes" if all_match else "NO")
    print(f"ascend == cpu torch.max semantics (incl. NaN propagation, signed zero): "
          f"{'yes' if all_match else 'NO'}")
    print("note: torch.scatter_reduce(include_self=False) column is reported for reference only; "
          "PyG's global_max_pool semantics for empty bins are handled by the adapter's occupancy "
          "mask (0), not by this op.")


if __name__ == "__main__":
    main()
