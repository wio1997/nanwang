"""Stage 4 — FP32 backward (tie-gradient) for the Ascend ``global_max_pool`` path.

The forward is NOT re-implemented here: this module calls the frozen Stage 2/3A adapter
(``global_max_pool_ascend``) unchanged, which in turn calls the formal delivery OPP's
``aclnnScatterMaxV1``.  Only the backward is new.

Frozen gradient contract (CPU oracle — see
``pyg-ascend-compat/global_max_pool/stage4_cpu_gradient_oracle.md``)::

    out[g,f]   = max over { x[i,f] : batch[i] == g }        # empty group -> 0
    winner     = (x == out[batch])
    count[g,f] = sum_i winner[i,f] over the group + (1 if out[g,f] == 0 else 0)
    grad_x     = winner * (grad_out[batch] / count[batch])

The ``+ (out == 0)`` term reproduces the PyTorch ``scatter_reduce(..., include_self=False)``
backward quirk: the zero self slot that the *forward* excludes is still counted whenever it equals
the reduced value (a group whose maximum is exactly ``0`` therefore gets ``grad/(n+1)`` per
winner).  NaN groups follow from the same formula: every comparison is false, ``count == 0`` and
``0 * (grad/0) = nan`` — exactly what PyG/PyTorch produce on CPU.

Every operation is elementwise / gather / scatter-add on the NPU; the module never calls
``aten::scatter_reduce`` and never falls back to the host CPU (audited in the Stage 4 report).
"""

from __future__ import annotations

import importlib.util
import os
from typing import Any, Dict, Optional

import torch
import torch_npu  # noqa: F401  (registers the NPU backend)

_ADAPTER_ENV = "PYG_ASCEND_ADAPTER_PATH"
_ADAPTER_DEFAULT = "/root/zyg/global_max_pool/stage2/python/global_max_pool_ascend.py"

_adapter = None
_last_backward: Dict[str, Any] = {}
_DEBUG = bool(int(os.environ.get("STAGE4_BACKWARD_DEBUG", "0")))


def _load_adapter():
    """Import the frozen Stage 2/3A adapter (the forward implementation)."""
    global _adapter
    if _adapter is None:
        path = os.environ.get(_ADAPTER_ENV, _ADAPTER_DEFAULT)
        if not os.path.isfile(path):
            raise RuntimeError(
                f"Stage 4 backward needs the Stage 2/3A adapter at '{path}'; set {_ADAPTER_ENV}."
            )
        spec = importlib.util.spec_from_file_location("pyg_ascend_compat._stage2_adapter", path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        _adapter = module
    return _adapter


def get_last_backward_info() -> Dict[str, Any]:
    """Diagnostics of the most recent backward call (sizes, primitive choices)."""
    return dict(_last_backward)


def _scatter_add_rows(dst: torch.Tensor, index: torch.Tensor, src: torch.Tensor):
    """dst[index[i]] += src[i] along dim 0, using the cheapest native NPU primitive.

    ``index_add_`` is the documented path; ``scatter_add_`` with a broadcast index is the
    fallback.  Both are device ops (verified in the Stage 4 primitive audit).
    """
    try:
        return dst.index_add_(0, index, src), "index_add_"
    except (RuntimeError, NotImplementedError):
        return dst.scatter_add_(0, index.view(-1, 1).expand(-1, src.size(1)), src), "scatter_add_"


def _gather_rows(src: torch.Tensor, index: torch.Tensor):
    """src[index] along dim 0 (NPU gather; ``index_select`` with an ``indices`` fallback)."""
    try:
        return src.index_select(0, index), "index_select"
    except (RuntimeError, NotImplementedError):
        return src.gather(0, index.view(-1, 1).expand(-1, src.size(1))), "gather"


def backward_grad_x(
    x: torch.Tensor,
    batch: torch.Tensor,
    out: torch.Tensor,
    grad_out: torch.Tensor,
    *,
    nan_fixup: bool = True,
) -> torch.Tensor:
    """Exact contract gradient (see the module docstring).

    Args:
        x: ``[N, F]`` fp32 input of the forward pass.
        batch: ``[N]`` int64 group index.
        out: ``[S, F]`` forward output (empty groups are 0).
        grad_out: ``[S, F]`` upstream gradient.
        nan_fixup: force ``nan`` for (group, feature) cells whose winner count is zero.  Those are
            exactly the NaN groups of the forward; the fixup keeps the contract independent of the
            device's division/flush behaviour.
    """
    n, f = x.shape
    s = out.size(0)
    device = x.device

    if n == 0 or f == 0 or s == 0:
        _last_backward.clear()
        _last_backward.update({"path": "empty_fast_path", "N": n, "F": f, "S": s})
        return torch.zeros_like(x)

    rows = batch
    out_rows, gather_op = _gather_rows(out, rows)            # [N, F]
    winner = (x == out_rows)                                 # bool [N, F]
    w = winner.to(torch.float32)

    count = torch.zeros((s, f), dtype=torch.float32, device=device)
    count, scatter_op = _scatter_add_rows(count, rows, w)     # winners per (group, feature)
    count = count + (out == 0).to(torch.float32)             # include_self=False quirk

    scaled = grad_out / count                                # 0/0 -> nan, x/0 -> +-inf
    scaled_rows, _ = _gather_rows(scaled, rows)
    grad_x = w * scaled_rows

    if nan_fixup:
        # count == 0 <=> the group/feature maximum is NaN (a real maximum always has >= 1 winner),
        # so the contract value is nan for every row of that group/feature.
        zero_count = (count == 0)                            # [S, F]
        zero_rows, _ = _gather_rows(zero_count, rows)
        grad_x = grad_x.masked_fill(zero_rows, float("nan"))

    if _DEBUG:
        _last_backward.clear()
        _last_backward.update({
            "path": "kernel_backward",
            "N": n,
            "F": f,
            "S": s,
            "gather_op": gather_op,
            "scatter_op": scatter_op,
            "winner_cells": int(w.sum().item()),
            "zero_count_cells": int((count == 0).sum().item()),
            "nan_fixup": bool(nan_fixup),
        })
    return grad_x


class GlobalMaxPoolAscendFunction(torch.autograd.Function):
    """Autograd wrapper around the frozen forward + the Stage 4 tie-gradient backward."""

    @staticmethod
    def forward(ctx, x, batch, size, alignment_mode):  # type: ignore[override]
        adapter = _load_adapter()
        with torch.no_grad():
            out = adapter.global_max_pool_ascend(x.detach(), batch, size,
                                                 alignment_mode=alignment_mode)
        ctx.save_for_backward(x, batch, out)
        ctx.alignment_mode = alignment_mode
        return out

    @staticmethod
    def backward(ctx, grad_out):  # type: ignore[override]
        x, batch, out = ctx.saved_tensors
        grad_x = backward_grad_x(x, batch, out, grad_out.contiguous())
        # x, batch, size, alignment_mode
        return grad_x, None, None, None


def global_max_pool_ascend_autograd(
    x: torch.Tensor,
    batch: torch.Tensor,
    size: Optional[int] = None,
    *,
    alignment_mode: str = "auto",
) -> torch.Tensor:
    """Differentiable ``[N, F] -> [S, F]`` global max pooling on Ascend (FP32, forward + backward).

    The forward/guard semantics are exactly those of the frozen Stage 3 adapter; this entry only
    adds the autograd edge.  Use it when ``x.requires_grad`` is True and grad mode is enabled.
    """
    if not isinstance(x, torch.Tensor) or x.device.type != "npu":
        raise ValueError("global_max_pool_ascend_autograd expects an NPU tensor")
    if x.dtype != torch.float32:
        raise ValueError(f"Stage 4 backward supports float32 only, got {x.dtype}")
    if x.dim() != 2:
        raise ValueError(f"x must be 2D [N, F], got {tuple(x.shape)}")
    if not x.requires_grad:
        # nothing to differentiate: identical to the frozen forward path
        return _load_adapter().global_max_pool_ascend(x, batch, size,
                                                      alignment_mode=alignment_mode)
    return GlobalMaxPoolAscendFunction.apply(x, batch, size, alignment_mode)


__all__ = [
    "GlobalMaxPoolAscendFunction",
    "global_max_pool_ascend_autograd",
    "backward_grad_x",
    "get_last_backward_info",
]
