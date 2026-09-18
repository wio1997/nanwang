#!/usr/bin/env python3
"""Exact ScatterMaxV1 tiling calculator (host tiling formulas, Ascend910B3 / FP32).

Mirrors `DrivingSDK/kernels/scatter_max/op_host/scatter_max_v1.cpp` (GetTilingData) and the kernel
constructor in `op_kernel/scatter_max_v1.h`. Pure arithmetic - no device access.

Usage:
    python3 tiling_calc.py --table            # the N boundary table used by Stage 3B
    python3 tiling_calc.py --n 40 --f 48888
    python3 tiling_calc.py --threshold --n 40 # first F that takes the LARGE_TAIL path
"""

from __future__ import annotations

import argparse
import json

BLOCK_SIZE = 32
UB_PRESERVED = 1024
MAX_BATCH_NUM = 4095
DTYPE_SIZE = 4          # FP32
IDX_DTYPE_SIZE = 4      # INT32
UB_SIZE_910B3 = 196608  # from aarch64-linux/data/platform_config/Ascend910B3.ini (ub_size)
CORE_NUM_910B3 = 40     # vector_core_cnt
TILING_KEY_SMALL_TAIL = 0
TILING_KEY_LARGE_TAIL = 1


def align_up(value: int, align: int) -> int:
    """CeilAlign / AlignUp: DivCeil(a, b) * b."""
    if align == 0:
        return 0
    return ((value + align - 1) // align) * align


def compute(n: int, f: int, ub_size: int = UB_SIZE_910B3, core_num: int = CORE_NUM_910B3) -> dict:
    elem_num_per_block = BLOCK_SIZE // DTYPE_SIZE          # 8 fp32 elements
    ub = ub_size - UB_PRESERVED

    idx_elem_num = n
    tail_elem_num = f                                       # 2-D [N, F]
    tail_size = tail_elem_num * DTYPE_SIZE
    tail_size_align = align_up(tail_elem_num, elem_num_per_block) * DTYPE_SIZE

    idx_num_per_core = idx_elem_num // core_num
    idx_batch_num = min(idx_num_per_core, MAX_BATCH_NUM)
    remain_ub_size = ub - align_up(idx_batch_num, elem_num_per_block) * IDX_DTYPE_SIZE
    tail_batch_num = remain_ub_size // tail_size_align if tail_size_align else 0
    src_batch_num = remain_ub_size // BLOCK_SIZE * elem_num_per_block

    left_idx_num = idx_elem_num % core_num
    if left_idx_num == 0:
        core_num_per_tail = left_src_num_big_core = left_src_big_core_num = left_src_batch_num = 0
    else:
        core_num_per_tail = min(core_num // left_idx_num, tail_elem_num)
        left_src_num_big_core = -(-tail_elem_num // core_num_per_tail)      # DivCeil
        left_src_big_core_num = tail_elem_num - core_num_per_tail * (left_src_num_big_core - 1)
        left_src_batch_num = ub // BLOCK_SIZE * elem_num_per_block

    large_tail = (idx_num_per_core != 0) and (tail_batch_num == 0)

    # kernel-side derived values (KernelScatterMaxBase constructor)
    tail_elem_num_align = align_up(tail_elem_num, elem_num_per_block)
    idx_loop = -(-idx_num_per_core // idx_batch_num) if idx_num_per_core else 0
    left_src_num_cur_core = (
        min(left_src_num_big_core, tail_elem_num) if left_idx_num else 0
    )
    left_src_loop = (-(-left_src_num_cur_core // left_src_batch_num)
                     if left_src_batch_num else 0)

    return {
        "N": n,
        "F": f,
        "elemNumPerBlock": elem_num_per_block,
        "ubSize": ub_size,
        "ubSizeAfterPreserved": ub,
        "idxNumPerCore": idx_num_per_core,
        "idxBatchNum": idx_batch_num,
        "remainUbSize": remain_ub_size,
        "tailElemNum": tail_elem_num,
        "tailSize": tail_size,
        "tailSizeAlign": tail_size_align,
        "tailBatchNum": tail_batch_num,
        "srcBatchNum": src_batch_num,
        "leftIdxNum": left_idx_num,
        "coreNumPerTail": core_num_per_tail,
        "leftSrcNumBigCore": left_src_num_big_core,
        "leftSrcBigCoreNum": left_src_big_core_num,
        "leftSrcBatchNum": left_src_batch_num,
        "leftSrcNumCurCore": left_src_num_cur_core,
        "kernel_tailElemNumAlign": tail_elem_num_align,
        "kernel_idxLoop": idx_loop,
        "kernel_leftSrcLoop": left_src_loop,
        "uses_leftSrc_path": bool(n % core_num),
        "tilingKey": TILING_KEY_LARGE_TAIL if large_tail else TILING_KEY_SMALL_TAIL,
        "tilingKeyName": "LARGE_TAIL" if large_tail else "SMALL_TAIL",
    }


def first_large_tail_f(n: int, ub_size: int = UB_SIZE_910B3,
                       core_num: int = CORE_NUM_910B3) -> int | None:
    """Smallest F whose tiling key is LARGE_TAIL for this N (None if N too small)."""
    if n // core_num == 0:
        return None
    for f in range(1, 60000):
        if compute(n, f, ub_size, core_num)["tilingKey"] == TILING_KEY_LARGE_TAIL:
            return f
    return None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int)
    ap.add_argument("--f", type=int)
    ap.add_argument("--threshold", action="store_true")
    ap.add_argument("--table", action="store_true")
    args = ap.parse_args()

    if args.table:
        ns = [39, 40, 41, 79, 81, 127, 320, 321, 4097, 163799, 163800, 163801]
        print(f"{'N':>8} {'idxNumPerCore':>13} {'idxBatchNum':>11} {'remainUbSize':>12} "
              f"{'tailBatchNum(F=8)':>17} {'key(F=8)':>12} {'key(F=33)':>12} "
              f"{'leftIdxNum':>10} {'uses_leftSrc':>12}")
        for n in ns:
            r8 = compute(n, 8)
            r33 = compute(n, 33)
            print(f"{n:>8} {r8['idxNumPerCore']:>13} {r8['idxBatchNum']:>11} "
                  f"{r8['remainUbSize']:>12} {r8['tailBatchNum']:>17} "
                  f"{r8['tilingKeyName']:>12} {r33['tilingKeyName']:>12} "
                  f"{r8['leftIdxNum']:>10} {str(r8['uses_leftSrc_path']):>12}")
        print()
        for n in (40, 80, 320, 321, 163800):
            th = first_large_tail_f(n)
            print(f"threshold: N={n:<7} first F with LARGE_TAIL = {th}")
        return 0

    if args.threshold:
        if args.n is None:
            raise SystemExit("--threshold needs --n")
        print(first_large_tail_f(args.n))
        return 0

    if args.n is None or args.f is None:
        raise SystemExit("need --n and --f (or --table)")
    print(json.dumps(compute(args.n, args.f), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
