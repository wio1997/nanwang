#!/usr/bin/env python3
"""Shared helpers for Stage 3B: deterministic data patterns, CPU goldens, memory budget."""

from __future__ import annotations

import os
import sys

import torch

DEV = "npu:0"
NEG_INF = float("-inf")
BUDGET_BYTES = 1 << 30  # 1 GiB per-case cap (Stage 3B safety rule)

ADAPTER_DIR_DEFAULT = "/root/zyg/global_max_pool/stage2/python"


def load_adapter():
    path = os.environ.get("PYG_ASCEND_ADAPTER_PATH", ADAPTER_DIR_DEFAULT)
    sys.path.insert(0, os.path.dirname(path))
    import global_max_pool_ascend as adapter  # noqa: PLC0415

    return adapter


def make_src(n: int, f: int, pattern: str = "det") -> torch.Tensor:
    """Deterministic CPU source tensor. `tail_winner`/`neg_only`/`neg_inf_rows` variants."""
    if f == 0 or n == 0:
        return torch.zeros((n, f), dtype=torch.float32)
    idx_n = torch.arange(n, dtype=torch.float32).unsqueeze(1)
    idx_f = torch.arange(f, dtype=torch.float32).unsqueeze(0)
    # deterministic, covers negatives and positives, unique per (n, f) column-wise
    x = ((idx_n * 13.0 + idx_f * 7.0) % 41.0) - 20.0 + 0.25 * ((idx_n + idx_f) % 4.0)
    if pattern == "tail_winner":
        x[n - 1, :] = x[n - 1, :] + 1000.0          # last row wins everywhere
        x[:, f - 1] = x[:, f - 1] + 0.5             # last feature stays distinguishable
        x[n - 1, f - 1] = 12345.0                   # last row + last feature = global max
    elif pattern == "neg_only":
        x = -x.abs() - 1.0                          # strictly negative
    elif pattern == "neg_inf_rows":
        x[:, 0] = NEG_INF                           # one feature is all -inf
        x[0, :] = NEG_INF                           # first row fully -inf
    return x


def make_index(n: int, s: int, mode: str, seed: int = 0) -> torch.Tensor:
    """int64 index patterns; `mode` targets the leftover/last-row behaviour."""
    if n == 0:
        return torch.zeros((0,), dtype=torch.int64)
    g = torch.Generator().manual_seed(seed + n + s)
    if mode == "spread":
        base = torch.randint(0, s, (n,), generator=g)
        if n >= 1:
            base[0] = 0
            base[-1] = s - 1
        return base
    if mode == "repeat":
        return torch.zeros((n,), dtype=torch.int64)
    if mode == "leftover_repeat":      # last row duplicates the first row's group
        base = torch.randint(0, s, (n,), generator=g)
        base[0] = 0
        base[-1] = base[0]
        return base
    if mode == "leftover_new":         # last row goes to a brand-new group
        base = torch.randint(0, max(1, s - 1), (n,), generator=g)
        base[-1] = s - 1
        return base
    if mode == "leftover_max":         # last row owns the true max of its group
        base = torch.randint(0, s, (n,), generator=g)
        base[-1] = s - 1
        return base
    if mode == "sparse_0_17_1023":     # only three groups used out of a large S
        allowed = torch.tensor([0, 17, min(1023, s - 1)], dtype=torch.int64)
        base = allowed[torch.randint(0, allowed.numel(), (n,), generator=g)]
        return base
    raise ValueError(f"unknown index mode {mode}")


def reference_loop(x: torch.Tensor, batch: torch.Tensor, size: int) -> torch.Tensor:
    """Independent CPU golden, explicit python loops (small shapes)."""
    x = x.detach().cpu().to(torch.float32)
    batch = batch.detach().cpu().to(torch.int64)
    f = x.shape[1]
    out = torch.zeros((size, f), dtype=torch.float32)
    for g in range(size):
        mask = batch == g
        if bool(mask.any()):
            out[g] = x[mask].max(dim=0).values
    return out


def reference_vectorized(x: torch.Tensor, batch: torch.Tensor, size: int) -> torch.Tensor:
    """CPU golden for large shapes: scatter_reduce(amax, include_self=False) + occupancy fix."""
    x = x.detach().cpu().to(torch.float32)
    batch = batch.detach().cpu().to(torch.int64)
    f = x.shape[1]
    if f == 0:
        return torch.zeros((size, 0), dtype=torch.float32)
    out = torch.full((size, f), NEG_INF, dtype=torch.float32)
    if x.shape[0] > 0:
        out.scatter_reduce_(0, batch.view(-1, 1).expand(-1, f), x, reduce="amax",
                            include_self=False)
    occupied = torch.zeros(size, dtype=torch.int32)
    if x.shape[0] > 0:
        occupied.scatter_(0, batch, 1)
    out[occupied == 0] = 0.0
    return out


def estimate_bytes(n: int, f: int, s: int, padded: bool = True) -> dict:
    """Rough device-memory estimate of one adapter call (bytes)."""
    f_kernel = ((f + 7) // 8) * 8 if padded else f
    d = {
        "input": n * f * 4,
        "padded_input": n * f_kernel * 4 if f_kernel != f else 0,
        "index": n * 4,
        "output": s * f * 4,
        "output_kernel": s * f_kernel * 4,
        "argmax": s * f_kernel * 4,
        "occupancy": s * 4,
        "crop_copy": s * f * 4 if f_kernel != f else 0,
    }
    d["total"] = sum(d.values())
    return d


def compare(name, x, batch, size, adapter, alignment_mode="auto", use_loop_ref=False):
    """Run the adapter and compare with the CPU golden. Returns (ok, detail)."""
    x_dev = x.to(DEV) if x.device.type != "npu" else x
    b_dev = batch.to(DEV) if batch.device.type != "npu" else batch
    out, info = adapter.global_max_pool_ascend(x_dev, b_dev, size, alignment_mode=alignment_mode,
                                              debug=True)
    torch.npu.synchronize()
    ref = (reference_loop if use_loop_ref else reference_vectorized)(x, batch, size)
    got = out.detach().cpu()
    shape_ok = tuple(got.shape) == tuple(ref.shape)
    if not shape_ok:
        return False, f"shape {tuple(got.shape)} != {tuple(ref.shape)}"
    finite = torch.isfinite(ref) & torch.isfinite(got)
    diff = float((ref[finite] - got[finite]).abs().max()) if bool(finite.any()) else 0.0
    inf_ok = torch.equal(torch.isinf(ref), torch.isinf(got))
    sign_ok = torch.equal(torch.isinf(ref) & (ref < 0), torch.isinf(got) & (got < 0))
    nan_ok = torch.equal(torch.isnan(ref), torch.isnan(got))
    ok = bool(diff == 0.0 and inf_ok and sign_ok and nan_ok)
    detail = (f"shape={tuple(got.shape)} max_diff={diff} inf_ok={inf_ok} neginf_ok={sign_ok} "
              f"nan_ok={nan_ok} padded={info.get('padded')} F_kernel={info.get('F_kernel')}")
    return ok, detail
