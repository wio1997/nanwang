#!/usr/bin/env python3
"""Stage 3A performance OBSERVATION only (no tuning): aligned F=32 vs padded F=33."""

import os
import sys
import time

import torch
import torch_npu  # noqa: F401

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.normpath(os.path.join(HERE, "..", "..", "stage2", "python")))

from global_max_pool_ascend import global_max_pool_ascend  # noqa: E402


def measure(n, f, s, iters=20):
    torch.manual_seed(0)
    x = (torch.rand((n, f), dtype=torch.float32) - 0.5).to("npu:0")
    batch = torch.randint(0, s, (n,), dtype=torch.int64).to("npu:0")
    out, info = global_max_pool_ascend(x, batch, s, debug=True)
    torch.npu.synchronize()
    for _ in range(3):
        global_max_pool_ascend(x, batch, s)
    torch.npu.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        global_max_pool_ascend(x, batch, s)
    torch.npu.synchronize()
    ms = (time.perf_counter() - t0) / iters * 1000.0
    return ms, info


def main():
    torch.npu.set_device(0)
    n, s = 4096, 64
    print(f"N={n} S={s}, latencies include the host sync of the adapter\n")
    header = f"{'F':>4s} {'F_kernel':>9s} {'padded':>7s} {'ms/call':>8s} {'pad_in_B':>9s} " \
             f"{'out_kernel_B':>13s} {'argmax_B':>9s} {'crop_B':>8s}"
    print(header)
    for f in (32, 33):
        ms, info = measure(n, f, s)
        print(f"{f:>4d} {info['F_kernel']:>9d} {str(info['padded']):>7s} {ms:>8.3f} "
              f"{info['pad_input_bytes']:>9d} {info['output_kernel_bytes']:>13d} "
              f"{info['argmax_scratch_bytes']:>9d} {info['crop_copy_bytes']:>8d}")

    f_aligned, f_unaligned = 32, 33
    f_pad = ((f_unaligned + 7) // 8) * 8
    extra_in = n * (f_pad - f_unaligned) * 4
    extra_out = s * (f_pad - f_unaligned) * 4
    print(f"\npadding overhead for F={f_unaligned} -> {f_pad}:")
    print(f"  extra input bytes      : {extra_in}")
    print(f"  extra output bytes     : {extra_out} (kernel output row width)")
    print(f"  extra argmax scratch   : {extra_out} (int32, same row width)")
    print(f"  crop copy              : {s * f_unaligned * 4} bytes (contiguous crop back to F={f_unaligned})")


if __name__ == "__main__":
    main()
