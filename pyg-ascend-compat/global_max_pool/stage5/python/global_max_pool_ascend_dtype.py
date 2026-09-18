"""Stage 5 — FP16 / BF16 forward + first-order backward for the Ascend ``global_max_pool`` path.

The delivered ScatterMaxV1 OPP is FP32-only (op definition ``DataType({ge::DT_FLOAT})`` and
``aclnnScatterMaxV1GetWorkspaceSize`` returns 161002 for fp16/bf16 tensors — see the Stage 5
report).  The FP16/BF16 path therefore runs on the device as:

    x (fp16/bf16) --Cast--> fp32 --frozen ScatterMaxV1--> fp32 out --Cast--> fp16/bf16 out

Both casts and every backward primitive were audited on the NPU: no host fallback, no AI_CPU,
dtype preserved, and bit-exact against CPU (``logs/stage5/npu_primitive_dtype_audit.log``).
The max itself is exact under this cast chain: fp16/bf16 values are exactly representable in fp32,
so the reduced value cast back is bit-identical to the dtype computation.

Frozen backward contract (measured from the real CPU oracle, see
``global_max_pool/stage5_cpu_dtype_oracle.md``)::

    out[g,f]    = max over { x[i,f] : batch[i] == g }            # empty group -> 0
    winner      = (x == out[batch])                              # in the input dtype
    count[g,f]  = (# winners) + (1 if out[g,f] == 0 else 0)      # exact integer
    count_dtype = cast_to_input_dtype(count)                     # round-to-nearest (measured!)
    grad_x      = winner * (grad_out[batch] / count_dtype[batch])

The count is accumulated in FP32 (so large tie counts stay exact) and only *then* rounded to the
input dtype — CPU measurements show the denominator is the dtype-rounded count, not a dtype
accumulation (fp16 N=4100 -> 1/cast_fp16(4100), not the saturating incremental sum; 21/21 fp16 and
22/22 bf16 cases).  The division itself is performed in the input dtype, which the audit shows to
be bit-exact against CPU half/bfloat16 division.
"""

from __future__ import annotations

import importlib.util
import os
from typing import Any, Dict, Optional

import torch
import torch_npu  # noqa: F401  (registers the NPU backend)

SUPPORTED_DTYPES = (torch.float16, torch.bfloat16)

_ADAPTER_ENV = "PYG_ASCEND_ADAPTER_PATH"
_ADAPTER_DEFAULT = "/root/zyg/global_max_pool/stage2/python/global_max_pool_ascend.py"
_DEBUG = bool(int(os.environ.get("STAGE5_BACKWARD_DEBUG", "0")))

_adapter = None
_last_backward: Dict[str, Any] = {}


def _load_adapter():
    """Import the frozen Stage 2/3A FP32 adapter (the actual ScatterMaxV1 forward)."""
    global _adapter
    if _adapter is None:
        path = os.environ.get(_ADAPTER_ENV, _ADAPTER_DEFAULT)
        if not os.path.isfile(path):
            raise RuntimeError(
                f"Stage 5 needs the Stage 2/3A adapter at '{path}'; set {_ADAPTER_ENV}."
            )
        spec = importlib.util.spec_from_file_location("pyg_ascend_compat._stage2_adapter", path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        _adapter = module
    return _adapter


def get_last_backward_info() -> Dict[str, Any]:
    return dict(_last_backward)


def forward_dtype(x: torch.Tensor, batch: torch.Tensor, size: Optional[int],
                  alignment_mode: str = "auto") -> torch.Tensor:
    """Device cast chain: dtype -> fp32 -> frozen ScatterMaxV1 -> dtype."""
    out32 = _load_adapter().global_max_pool_ascend(x.float(), batch, size,
                                                   alignment_mode=alignment_mode)
    return out32.to(x.dtype)


def _gather_rows(src: torch.Tensor, index: torch.Tensor):
    try:
        return src.index_select(0, index), "index_select"
    except (RuntimeError, NotImplementedError):
        return src.gather(0, index.view(-1, 1).expand(-1, src.size(1))), "gather"


def backward_grad_x(x: torch.Tensor, batch: torch.Tensor, out: torch.Tensor,
                    grad_out: torch.Tensor) -> torch.Tensor:
    """Dtype-exact Stage 5 gradient (see the module docstring)."""
    n, f = x.shape
    s = out.size(0)
    dt = x.dtype
    device = x.device

    if n == 0 or f == 0 or s == 0:
        _last_backward.clear()
        _last_backward.update({"path": "empty_fast_path", "N": n, "F": f, "S": s,
                               "dtype": str(dt)})
        return torch.zeros_like(x)

    rows = batch
    out_rows, gather_op = _gather_rows(out, rows)          # [N, F] input dtype
    winner = (x == out_rows)                               # bool
    w = winner.to(dt)                                      # 1/0 in the input dtype

    # exact integer count in fp32, then rounded to the input dtype (measured CPU contract)
    count32 = torch.zeros((s, f), dtype=torch.float32, device=device)
    count32 = count32.index_add_(0, rows, winner.to(torch.float32))
    count32 = count32 + (out == 0).to(torch.float32)       # include_self=False quirk
    count_dt = count32.to(dt)

    scaled = _gather_rows(grad_out, rows)[0] / _gather_rows(count_dt, rows)[0]   # dtype divide
    grad_x = w * scaled

    # NaN contract: a NaN group has no winners -> count 0 -> nan for every row of that cell
    zero_rows = _gather_rows((count32 == 0), rows)[0]
    grad_x = grad_x.masked_fill(zero_rows, float("nan"))

    if _DEBUG:
        _last_backward.clear()
        _last_backward.update({
            "path": "dtype_kernel_backward", "dtype": str(dt), "N": n, "F": f, "S": s,
            "gather_op": gather_op,
            "winner_cells": int(w.sum().item()),
            "zero_count_cells": int((count32 == 0).sum().item()),
        })
    return grad_x


class GlobalMaxPoolAscendDtypeFunction(torch.autograd.Function):
    """Autograd wrapper: device cast forward + Stage 5 dtype backward."""

    @staticmethod
    def forward(ctx, x, batch, size, alignment_mode):  # type: ignore[override]
        with torch.no_grad():
            out = forward_dtype(x.detach(), batch, size, alignment_mode)
        ctx.save_for_backward(x, batch, out)
        ctx.alignment_mode = alignment_mode
        return out

    @staticmethod
    def backward(ctx, grad_out):  # type: ignore[override]
        x, batch, out = ctx.saved_tensors
        grad_x = backward_grad_x(x, batch, out, grad_out.contiguous())
        return grad_x, None, None, None


def global_max_pool_ascend_dtype(
    x: torch.Tensor,
    batch: torch.Tensor,
    size: Optional[int] = None,
    *,
    alignment_mode: str = "auto",
) -> torch.Tensor:
    """FP16/BF16 ``[N, F] -> [S, F]`` global max pooling on Ascend (forward + first-order backward)."""
    if not isinstance(x, torch.Tensor):
        raise TypeError(f"x must be a torch.Tensor, got {type(x).__name__}")
    if x.dtype not in SUPPORTED_DTYPES:
        raise ValueError(f"Stage 5 supports {SUPPORTED_DTYPES}, got {x.dtype}")
    if x.device.type != "npu":
        raise ValueError(f"Stage 5 requires an NPU tensor, got {x.device}")
    if x.dim() != 2:
        raise ValueError(f"x must be 2D [N, F], got {tuple(x.shape)}")
    if not x.requires_grad:
        return forward_dtype(x, batch, size, alignment_mode)
    return GlobalMaxPoolAscendDtypeFunction.apply(x, batch, size, alignment_mode)


__all__ = [
    "SUPPORTED_DTYPES",
    "GlobalMaxPoolAscendDtypeFunction",
    "global_max_pool_ascend_dtype",
    "forward_dtype",
    "backward_grad_x",
    "get_last_backward_info",
]
