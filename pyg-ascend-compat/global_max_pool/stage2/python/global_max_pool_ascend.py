"""Stage 2 standalone Ascend adapter for ``global_max_pool`` (forward only).

    global_max_pool_ascend(x, batch, size=None) -> Tensor

The reduction itself is DrivingSDK's ``ScatterMaxV1`` custom op (FP32 data + INT32 index),
executed on Ascend 910B3 through its generated ACLNN API. PyG / ``aten::scatter_reduce`` /
``torch_npu`` are intentionally NOT patched here; this module is a standalone adapter
(PyG integration is Stage 6).

Semantics implemented (matches ``torch_geometric.nn.global_max_pool`` for the forward pass):

* ``out[g] = max over { x[n] : batch[n] == g }`` for every non-empty group
* groups with no node are **0**
* a non-empty group whose true maximum is ``-inf`` stays ``-inf`` (occupancy mask, see below)

Design notes
------------
* The raw op is called with a caller-provided output pre-filled with ``-inf`` so that negative-only
  groups stay correct.
* ``out.masked_fill_(~occupied, 0)`` is applied with an **occupancy mask** instead of the
  DrivingSDK-wrapper style ``out == -inf -> 0``: that naive rule would wrongly turn a genuine
  ``-inf`` maximum of a *non-empty* group into 0.
* Forward-only: ``x.requires_grad`` is rejected instead of silently producing a tensor with a
  wrong/absent backward.
* Any feature dim ``F >= 0`` is accepted. When ``F % 8 != 0`` (i.e. a row is not 32-byte aligned for
  FP32) the adapter pads the feature dim to ``ceil(F/8)*8`` with ``-inf`` columns, runs the same
  ScatterMaxV1 kernel and crops back (Stage 3A). ``alignment_mode="raw"`` forces the raw
  (unpadded) path, which Stage 3A proved is also safe; see
  ``pyg-ascend-compat/global_max_pool/stage3a_non_aligned_feature.md``.
"""

from __future__ import annotations

import importlib.util
import os
from typing import Any, Dict, Optional

import torch
import torch_npu  # noqa: F401  (registers the PrivateUse1/NPU device backend)

FEATURE_ALIGN_ELEMS = 8  # 32 B / sizeof(float32)
# ScatterMaxV1 documents `max(indices) < 491520`; the kernel does not hard-check it, so the
# adapter enforces it before launching anything.
INDEX_UPPER_BOUND = 491520
# Documented `N * (M + 1) < 4,026,531,840` element budget (see the Stage 3A report); the V1 kernel
# uses 64-bit offsets, so this is kept as a conservative documented guard only.
DOC_ELEMENT_BUDGET = 4026531840
_ALIGNMENT_MODES = ("auto", "raw", "pad")
_BRIDGE_ENV = "SCATTERMAXV1_BRIDGE"
_BRIDGE_DEFAULT = "/root/zyg/build/stage2_ext/scattermaxv1_bridge.so"

_bridge_module = None
_last_run: Dict[str, Any] = {}


def _load_bridge():
    """Import the minimal PyTorch<->ACLNN bridge extension lazily."""
    global _bridge_module
    if _bridge_module is None:
        path = os.environ.get(_BRIDGE_ENV, _BRIDGE_DEFAULT)
        if not os.path.isfile(path):
            raise RuntimeError(
                f"ScatterMaxV1 bridge extension not found at '{path}'. "
                f"Build it with extension/build_bridge.sh or set {_BRIDGE_ENV}."
            )
        spec = importlib.util.spec_from_file_location("scattermaxv1_bridge", path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        _bridge_module = module
    return _bridge_module


def get_last_run_info() -> Dict[str, Any]:
    """Diagnostics of the most recent adapter call (memory, syncs, copies)."""
    return dict(_last_run)


def _align_up(value: int, align: int) -> int:
    return ((value + align - 1) // align) * align


def _validate_inputs(x: torch.Tensor, batch: torch.Tensor, size: Optional[int]):
    if not isinstance(x, torch.Tensor):
        raise TypeError(f"x must be a torch.Tensor, got {type(x).__name__}")
    if not isinstance(batch, torch.Tensor):
        raise TypeError(f"batch must be a torch.Tensor, got {type(batch).__name__}")
    if x.device.type != "npu":
        raise ValueError(f"x must be on an Ascend NPU device, got {x.device}")
    if batch.device.type != "npu":
        raise ValueError(f"batch must be on an Ascend NPU device, got {batch.device}")
    if x.device != batch.device:
        raise ValueError(f"x and batch must be on the same device ({x.device} vs {batch.device})")
    if x.dim() != 2:
        raise ValueError(f"x must be 2D [N, F] (Stage 2 adapter), got {tuple(x.shape)}")
    if batch.dim() != 1:
        raise ValueError(f"batch must be 1D [N] (Stage 2 adapter), got {tuple(batch.shape)}")
    if x.dtype != torch.float32:
        raise ValueError(
            f"Stage 2 adapter supports float32 x only, got {x.dtype} "
            "(fp16/bf16 are Stage 5)"
        )
    if batch.dtype != torch.int64:
        raise ValueError(f"batch must be torch.int64 (PyG convention), got {batch.dtype}")
    if x.requires_grad:
        raise RuntimeError(
            "global_max_pool_ascend Stage 2 adapter is forward-only and does not support "
            "autograd yet; call it under torch.no_grad() or pass x.detach() "
            "(backward/tie-gradient is Stage 4)"
        )
    n, f = x.shape
    if batch.numel() != n:
        raise ValueError(f"x.size(0)={n} must equal batch.numel()={batch.numel()}")
    if size is not None:
        if isinstance(size, bool) or not isinstance(size, int):
            raise TypeError(f"size must be None or a python int, got {type(size).__name__}")
        if size < 0:
            raise ValueError(f"size must be >= 0, got {size}")
    if n == 0 and size is None:
        return n, f, 0, False
    if n > 0 and size == 0:
        raise ValueError("size=0 is invalid when N > 0 (no group could hold any node)")
    return n, f, size, True


def global_max_pool_ascend(
    x: torch.Tensor,
    batch: torch.Tensor,
    size: Optional[int] = None,
    *,
    alignment_mode: str = "auto",
    debug: bool = False,
):
    """Global max pooling over ``batch`` for a 2D FP32 NPU tensor.

    Args:
        x: ``[N, F]`` float32 tensor on an Ascend NPU, ``F >= 0``, ``requires_grad=False``.
        batch: ``[N]`` int64 tensor on the same NPU with values in ``[0, S)``.
        size: optional explicit number of output groups ``S``. If ``None``, ``S = max(batch)+1``.
        alignment_mode: ``"auto"`` (default) pads non-32B-aligned feature dims with ``-inf`` and
            crops back; ``"raw"`` always calls the kernel with the original feature dim;
            ``"pad"`` forces padding when the dim is not already aligned.
        debug: when True returns ``(out, info)`` where ``info`` carries adapter diagnostics.

    Returns:
        ``[S, F]`` float32 NPU tensor; empty groups are 0, non-empty groups are the exact max
        (including ``-inf`` maxima of non-empty groups).
    """
    if alignment_mode not in _ALIGNMENT_MODES:
        raise ValueError(
            f"alignment_mode must be one of {_ALIGNMENT_MODES}, got {alignment_mode!r}"
        )
    n, f, size, needs_kernel = _validate_inputs(x, batch, size)

    # ---- N == 0 fast paths (no kernel launch at all) ----
    if n == 0:
        s = 0 if size is None else size
        out = torch.zeros((s, f), dtype=torch.float32, device=x.device)
        info = {
            "path": "empty_fast_path",
            "N": 0,
            "F": f,
            "size": s,
            "host_sync": False,
            "x_copied": False,
            "argmax_scratch_bytes": 0,
            "output_bytes": s * f * 4,
        }
        _last_run.clear()
        _last_run.update(info)
        return (out, info) if debug else out

    # ---- F == 0 fast path: nothing to reduce along the feature dim ----
    if f == 0:
        lo, hi = torch.stack((batch.min(), batch.max())).cpu().tolist()
        lo, hi = int(lo), int(hi)
        if lo < 0:
            raise ValueError(f"batch contains negative group index ({lo}); indices must be >= 0")
        s = int(size) if size is not None else hi + 1
        if size is not None and hi >= s:
            raise ValueError(
                f"batch contains index {hi} but size={s}; every index must satisfy batch < size"
            )
        out = torch.zeros((s, 0), dtype=torch.float32, device=x.device)
        info = {
            "path": "zero_feature_fast_path",
            "N": n,
            "F": 0,
            "size": s,
            "host_sync_for_size": size is None,
            "host_sync_for_range_validation": True,
            "x_copied": False,
            "argmax_scratch_bytes": 0,
            "output_bytes": 0,
        }
        _last_run.clear()
        _last_run.update(info)
        return (out, info) if debug else out

    # ---- index range validation (single D2H sync for both bounds) ----
    lo, hi = torch.stack((batch.min(), batch.max())).cpu().tolist()
    lo, hi = int(lo), int(hi)
    if lo < 0:
        raise ValueError(f"batch contains negative group index ({lo}); indices must be >= 0")
    if hi >= INDEX_UPPER_BOUND:
        raise ValueError(
            f"batch contains index {hi} >= {INDEX_UPPER_BOUND}, which is outside the documented "
            "ScatterMaxV1 support range; refusing to launch the kernel"
        )
    s = int(size) if size is not None else hi + 1
    if size is not None and hi >= s:
        raise ValueError(
            f"batch contains index {hi} but size={s}; every index must satisfy batch < size "
            "(otherwise ScatterMaxV1 would write outside the output buffer)"
        )
    if n * (f + 1) >= DOC_ELEMENT_BUDGET:
        raise ValueError(
            f"N*(F+1) = {n * (f + 1)} reaches the documented ScatterMaxV1 element budget "
            f"({DOC_ELEMENT_BUDGET}); refusing to launch"
        )

    # ---- device-side preparation ----
    x_c = x if x.is_contiguous() else x.contiguous()
    index32 = batch.to(torch.int32)
    if not index32.is_contiguous():
        index32 = index32.contiguous()

    # ---- 32 B feature alignment handling (Stage 3A) ----
    f_pad = _align_up(f, FEATURE_ALIGN_ELEMS)
    if alignment_mode == "raw":
        use_pad = False
    else:  # "auto" and "pad"
        use_pad = f_pad != f
    f_kernel = f_pad if use_pad else f
    if use_pad:
        # -inf padding columns cannot change max over the original F columns
        x_kernel = torch.nn.functional.pad(x_c, (0, f_pad - f), mode="constant",
                                           value=float("-inf"))
    else:
        x_kernel = x_c

    # raw op semantics: caller-provided output, initialised with -inf
    out_kernel = torch.full((s, f_kernel), float("-inf"), dtype=torch.float32, device=x.device)
    # argmax is a mandatory output of the generated ACLNN schema; the forward kernel does not
    # use it, so the scratch is left uninitialised
    argmax_scratch = torch.empty((s, f_kernel), dtype=torch.int32, device=x.device)

    _load_bridge().scatter_max_v1_forward(x_kernel, index32, out_kernel, argmax_scratch)

    # ---- occupancy: empty group -> 0, non-empty group keeps its exact max ----
    occupied = torch.zeros(s, dtype=torch.int32, device=x.device)
    occupied.scatter_(0, batch, 1)
    out_kernel.masked_fill_((occupied == 0).view(s, 1), 0.0)

    if f_kernel == f:
        out = out_kernel
    else:
        # crop the padding columns; contiguous() keeps the documented [S, F] layout guarantee
        out = out_kernel[:, :f].contiguous()

    info = {
        "path": "kernel",
        "N": n,
        "F": f,
        "F_kernel": f_kernel,
        "padded": bool(f_kernel != f),
        "alignment_mode": alignment_mode,
        "size": s,
        "size_from_host_sync": size is None,
        "host_sync_for_size": size is None,
        "host_sync_for_range_validation": True,
        "x_copied": x_c is not x,
        "index_dtype_before": str(batch.dtype),
        "index_dtype_kernel": "torch.int32",
        "argmax_scratch_bytes": s * f_kernel * 4,
        "output_bytes": s * f * 4,
        "output_kernel_bytes": s * f_kernel * 4,
        "index_bytes": n * 4,
        "occupancy_bytes": s * 4,
        "pad_input_bytes": (n * f_kernel * 4) if use_pad else 0,
        "crop_copy_bytes": (s * f * 4) if f_kernel != f else 0,
    }
    if debug:
        # extra host syncs are only paid when diagnostics are explicitly requested
        info["occupied_groups"] = int((occupied != 0).sum().item())
        info["empty_groups"] = int((occupied == 0).sum().item())
    _last_run.clear()
    _last_run.update(info)
    return (out, info) if debug else out


__all__ = [
    "global_max_pool_ascend",
    "get_last_run_info",
    "FEATURE_ALIGN_ELEMS",
    "INDEX_UPPER_BOUND",
    "DOC_ELEMENT_BUDGET",
]
