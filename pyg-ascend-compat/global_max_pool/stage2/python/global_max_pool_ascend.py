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
* Stage 2 is forward-only: ``x.requires_grad`` is rejected instead of silently producing a tensor
  with a wrong/absent backward.
* Stage 2 only handles the 32-byte aligned feature range (``F % 8 == 0`` for FP32). Non-aligned
  features are Stage 3.
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
    if f % FEATURE_ALIGN_ELEMS != 0:
        raise ValueError(
            f"Stage 2 forward adapter only supports 32-byte aligned FP32 feature dims "
            f"(F % {FEATURE_ALIGN_ELEMS} == 0), got F={f}. "
            "Non-aligned feature support is Stage 3."
        )
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
    debug: bool = False,
):
    """Global max pooling over ``batch`` for a 2D FP32 NPU tensor.

    Args:
        x: ``[N, F]`` float32 tensor on an Ascend NPU, ``F % 8 == 0``, ``requires_grad=False``.
        batch: ``[N]`` int64 tensor on the same NPU with values in ``[0, S)``.
        size: optional explicit number of output groups ``S``. If ``None``, ``S = max(batch)+1``.
        debug: when True returns ``(out, info)`` where ``info`` carries adapter diagnostics.

    Returns:
        ``[S, F]`` float32 NPU tensor; empty groups are 0, non-empty groups are the exact max
        (including ``-inf`` maxima of non-empty groups).
    """
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

    # ---- device-side preparation ----
    x_c = x if x.is_contiguous() else x.contiguous()
    index32 = batch.to(torch.int32)
    if not index32.is_contiguous():
        index32 = index32.contiguous()

    # raw op semantics: caller-provided output, initialised with -inf
    out = torch.full((s, f), float("-inf"), dtype=torch.float32, device=x.device)
    # argmax is a mandatory output of the generated ACLNN schema; the forward kernel does not
    # use it, so the scratch is left uninitialised
    argmax_scratch = torch.empty((s, f), dtype=torch.int32, device=x.device)

    _load_bridge().scatter_max_v1_forward(x_c, index32, out, argmax_scratch)

    # ---- occupancy: empty group -> 0, non-empty group keeps its exact max ----
    occupied = torch.zeros(s, dtype=torch.int32, device=x.device)
    occupied.scatter_(0, batch, 1)
    out.masked_fill_((occupied == 0).view(s, 1), 0.0)

    info = {
        "path": "kernel",
        "N": n,
        "F": f,
        "size": s,
        "size_from_host_sync": size is None,
        "host_sync_for_size": size is None,
        "host_sync_for_range_validation": True,
        "x_copied": x_c is not x,
        "index_dtype_before": str(batch.dtype),
        "index_dtype_kernel": "torch.int32",
        "argmax_scratch_bytes": s * f * 4,
        "output_bytes": s * f * 4,
        "index_bytes": n * 4,
        "occupancy_bytes": s * 4,
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
]
