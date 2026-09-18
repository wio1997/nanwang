"""pyg-ascend-compat — route ``torch_geometric.nn.global_max_pool`` to the Ascend adapter.

Usage::

    import pyg_ascend_compat
    pyg_ascend_compat.enable(debug=False)      # one call, then use PyG normally
    from torch_geometric.nn import global_max_pool
    out = global_max_pool(x, batch, size)      # -> ScatterMaxV1 on Ascend for fp32/NPU

The wrapper is installed at runtime on the ``torch_geometric`` module objects; site-packages is
never modified and the original function is restored by :func:`disable`.

Dispatch (Stage 5 scope: FP32 + FP16 + BF16 forward and first-order backward):

* x on NPU, 2-D, dtype float32 / float16 / bfloat16
* batch on the same NPU, int64, 1-D, ``batch.numel() == x.size(0)``
* anything else → the original PyG implementation (e.g. CPU, 1-D x, other dtypes)
* float32: Stage 4 autograd ``Function`` (frozen ScatterMaxV1 forward + tie-gradient backward)
* float16/bfloat16: Stage 5 entry — device ``Cast`` to fp32, frozen ScatterMaxV1, cast back
  (forward) and the dtype-exact tie-gradient backward; the delivered OPP itself is fp32-only.
* ``requires_grad=True`` with grad mode enabled uses the autograd entry; under ``torch.no_grad()``
  the plain forward path is used, exactly like any other PyTorch op.
* ``batch=None`` → the original PyG implementation (``x.max``); it is native on NPU and is *not*
  part of the Stage 4 custom backward scope.
"""

from __future__ import annotations

import importlib.util
import os
import sys
from typing import Any, Dict, Optional

import torch

__all__ = [
    "enable",
    "disable",
    "is_enabled",
    "stats",
    "reset_stats",
    "set_debug",
    "adapter_version",
]

__version__ = "0.1.0-stage6"

_STATE: Dict[str, Any] = {
    "enabled": False,
    "debug": False,
    "originals": {},          # module name -> (module object, attribute, original function)
    "adapter": None,
    "autograd": None,
    "stage5": None,
    "stats": {
        "total_calls": 0,
        "ascend_calls": 0,
        "original_calls": 0,
        "requires_grad_rejected": 0,
        "autograd_calls": 0,
        "dtype16_forward_calls": 0,
        "dtype16_autograd_calls": 0,
        "non_fp32_passthrough": 0,
    },
}

_ADAPTER_ENV = "PYG_ASCEND_ADAPTER_PATH"
_ADAPTER_REL = os.path.join("global_max_pool", "stage2", "python", "global_max_pool_ascend.py")
_ADAPTER_FALLBACKS = (
    "/root/zyg/global_max_pool/stage2/python/global_max_pool_ascend.py",
)

_AUTOGRAD_ENV = "PYG_ASCEND_AUTOGRAD_PATH"
_AUTOGRAD_REL = os.path.join("global_max_pool", "stage4", "python",
                             "global_max_pool_ascend_autograd.py")
_AUTOGRAD_FALLBACKS = (
    "/root/zyg/global_max_pool/stage4/python/global_max_pool_ascend_autograd.py",
)

_STAGE5_ENV = "PYG_ASCEND_STAGE5_PATH"
_STAGE5_REL = os.path.join("global_max_pool", "stage5", "python",
                           "global_max_pool_ascend_dtype.py")
_STAGE5_FALLBACKS = (
    "/root/zyg/global_max_pool/stage5/python/global_max_pool_ascend_dtype.py",
)


def _candidate_adapter_paths():
    env = os.environ.get(_ADAPTER_ENV)
    if env:
        yield env
    here = os.path.dirname(os.path.abspath(__file__))
    # walk up looking for a checkout that contains global_max_pool/stage2/python/...
    cur = here
    for _ in range(6):
        yield os.path.join(cur, _ADAPTER_REL)
        cur = os.path.dirname(cur)
    yield from _ADAPTER_FALLBACKS


def _candidate_autograd_paths():
    env = os.environ.get(_AUTOGRAD_ENV)
    if env:
        yield env
    here = os.path.dirname(os.path.abspath(__file__))
    cur = here
    for _ in range(6):
        yield os.path.join(cur, _AUTOGRAD_REL)
        cur = os.path.dirname(cur)
    yield from _AUTOGRAD_FALLBACKS


def _load_autograd():
    """Import the Stage 4 autograd module (forward + tie-gradient backward)."""
    if _STATE["autograd"] is not None:
        return _STATE["autograd"]
    tried = []
    for path in _candidate_autograd_paths():
        if path and os.path.isfile(path):
            spec = importlib.util.spec_from_file_location("pyg_ascend_compat._autograd_impl", path)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            _STATE["autograd"] = module
            _STATE["autograd_path"] = path
            return module
        tried.append(path)
    raise RuntimeError(
        "pyg-ascend-compat could not find global_max_pool_ascend_autograd.py. Set "
        f"{_AUTOGRAD_ENV} to its absolute path. Tried: {tried}"
    )


def _candidate_stage5_paths():
    env = os.environ.get(_STAGE5_ENV)
    if env:
        yield env
    here = os.path.dirname(os.path.abspath(__file__))
    cur = here
    for _ in range(6):
        yield os.path.join(cur, _STAGE5_REL)
        cur = os.path.dirname(cur)
    yield from _STAGE5_FALLBACKS


def _load_stage5():
    """Import the Stage 5 FP16/BF16 module (cast-based forward + dtype backward)."""
    if _STATE["stage5"] is not None:
        return _STATE["stage5"]
    tried = []
    for path in _candidate_stage5_paths():
        if path and os.path.isfile(path):
            spec = importlib.util.spec_from_file_location("pyg_ascend_compat._stage5_impl", path)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            _STATE["stage5"] = module
            _STATE["stage5_path"] = path
            return module
        tried.append(path)
    raise RuntimeError(
        "pyg-ascend-compat could not find global_max_pool_ascend_dtype.py. Set "
        f"{_STAGE5_ENV} to its absolute path. Tried: {tried}"
    )


def _load_adapter():
    """Import the validated Stage 2/3A adapter module (never re-implemented here)."""
    if _STATE["adapter"] is not None:
        return _STATE["adapter"]
    tried = []
    for path in _candidate_adapter_paths():
        if path and os.path.isfile(path):
            spec = importlib.util.spec_from_file_location("pyg_ascend_compat._adapter_impl", path)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            _STATE["adapter"] = module
            _STATE["adapter_path"] = path
            return module
        tried.append(path)
    raise RuntimeError(
        "pyg-ascend-compat could not find global_max_pool_ascend.py. Set "
        f"{_ADAPTER_ENV} to its absolute path. Tried: {tried}"
    )


def adapter_version() -> Optional[str]:
    module = _STATE["adapter"]
    return None if module is None else getattr(module, "__file__", None)


def _looks_like_pyg_global_max_pool(fn) -> bool:
    return (
        callable(fn)
        and getattr(fn, "__name__", "") == "global_max_pool"
        and getattr(fn, "__module__", "").startswith("torch_geometric")
    )


def _target_modules():
    """Modules that re-export ``global_max_pool`` and therefore need the wrapper."""
    names = ["torch_geometric.nn", "torch_geometric.nn.pool"]
    modules = []
    for name in names:
        module = sys.modules.get(name)
        if module is None:
            module = importlib.import_module(name)
        if _looks_like_pyg_global_max_pool(getattr(module, "global_max_pool", None)):
            modules.append(module)
    # canonical definition site
    module = importlib.import_module("torch_geometric.nn.pool.glob")
    if _looks_like_pyg_global_max_pool(getattr(module, "global_max_pool", None)):
        modules.append(module)
    return modules


def _dispatch(x, batch, size):
    """Return the adapter result, or None when the original PyG path must be used."""
    if not isinstance(x, torch.Tensor):
        return None
    dtype16 = x.dtype in (torch.float16, torch.bfloat16)
    if x.device.type != "npu" or (x.dtype != torch.float32 and not dtype16) or x.dim() != 2:
        if x.device.type == "npu" and x.dtype != torch.float32 and not dtype16:
            _STATE["stats"]["non_fp32_passthrough"] += 1
            if _STATE["debug"]:
                print(f"[pyg-ascend-compat] passthrough (dtype={x.dtype}) -> original PyG path")
        return None
    if batch is None or not isinstance(batch, torch.Tensor):
        return None
    if (batch.device.type != "npu" or batch.device != x.device or batch.dtype != torch.int64
            or batch.dim() != 1 or batch.numel() != x.size(0)):
        return None
    if dtype16:
        # Stage 5: device cast chain (dtype -> fp32 -> ScatterMaxV1 -> dtype) + dtype backward
        stage5 = _load_stage5()
        _STATE["stats"]["ascend_calls"] += 1
        if x.requires_grad and torch.is_grad_enabled():
            _STATE["stats"]["dtype16_autograd_calls"] += 1
            if _STATE["debug"]:
                print(
                    "[pyg-ascend-compat] global_max_pool -> Stage5 dtype autograd "
                    f"(dtype={x.dtype}, x={tuple(x.shape)}, size={size})"
                )
            return stage5.global_max_pool_ascend_dtype(x, batch, size)
        _STATE["stats"]["dtype16_forward_calls"] += 1
        if _STATE["debug"]:
            print(
                "[pyg-ascend-compat] global_max_pool -> Stage5 dtype forward "
                f"(dtype={x.dtype}, x={tuple(x.shape)}, size={size})"
            )
        return stage5.global_max_pool_ascend_dtype(x, batch, size)
    if x.requires_grad and torch.is_grad_enabled():
        # Stage 4: differentiable Ascend path (frozen forward + tie-gradient backward)
        autograd = _load_autograd()
        _STATE["stats"]["autograd_calls"] += 1
        _STATE["stats"]["ascend_calls"] += 1
        if _STATE["debug"]:
            print(
                "[pyg-ascend-compat] global_max_pool -> ScatterMaxV1 autograd (forward+backward) "
                f"(x={tuple(x.shape)}, batch={batch.dtype}, size={size})"
            )
        return autograd.global_max_pool_ascend_autograd(x, batch, size)
    adapter = _load_adapter()
    _STATE["stats"]["ascend_calls"] += 1
    if _STATE["debug"]:
        print(
            "[pyg-ascend-compat] global_max_pool -> ScatterMaxV1 adapter "
            f"(x={tuple(x.shape)}, batch={batch.dtype}, size={size})"
        )
    return adapter.global_max_pool_ascend(x, batch, size)


def _make_wrapper(original):
    def global_max_pool(x, batch=None, size=None):  # noqa: D401 - mirrors PyG signature
        _STATE["stats"]["total_calls"] += 1
        result = _dispatch(x, batch, size)
        if result is not None:
            return result
        _STATE["stats"]["original_calls"] += 1
        return original(x, batch, size)

    global_max_pool.__name__ = getattr(original, "__name__", "global_max_pool")
    global_max_pool.__qualname__ = getattr(original, "__qualname__", "global_max_pool")
    global_max_pool.__doc__ = original.__doc__
    global_max_pool.__module__ = getattr(original, "__module__", "torch_geometric.nn")
    global_max_pool.__wrapped__ = original
    global_max_pool._pyg_ascend_compat = True  # marker for tests
    return global_max_pool


def enable(debug: bool = False):
    """Install the Ascend dispatch for ``torch_geometric.nn.global_max_pool``."""
    if _STATE["enabled"]:
        _STATE["debug"] = debug
        return _STATE["originals"]
    _load_adapter()  # fail early if the adapter/extension is missing
    originals = {}
    for module in _target_modules():
        original = module.global_max_pool
        originals[module.__name__] = (module, "global_max_pool", original)
        module.global_max_pool = _make_wrapper(original)
    if not originals:
        raise RuntimeError(
            "pyg-ascend-compat could not find torch_geometric.nn.global_max_pool to wrap"
        )
    _STATE["originals"] = originals
    _STATE["enabled"] = True
    _STATE["debug"] = debug
    if debug:
        print(f"[pyg-ascend-compat] enabled on {sorted(originals)} "
              f"(adapter={_STATE.get('adapter_path')})")
    return originals


def disable():
    """Restore the original PyG implementations."""
    for module, attr, original in _STATE["originals"].values():
        setattr(module, attr, original)
    _STATE["originals"] = {}
    _STATE["enabled"] = False


def is_enabled() -> bool:
    return bool(_STATE["enabled"])


def set_debug(debug: bool) -> None:
    _STATE["debug"] = bool(debug)


def stats() -> Dict[str, Any]:
    """Call counters of the compat layer (Python-level evidence for the dispatch)."""
    out = dict(_STATE["stats"])
    out["enabled"] = _STATE["enabled"]
    out["debug"] = _STATE["debug"]
    out["adapter_path"] = _STATE.get("adapter_path")
    out["autograd_path"] = _STATE.get("autograd_path")
    out["stage5_path"] = _STATE.get("stage5_path")
    return out


def reset_stats() -> None:
    for key in _STATE["stats"]:
        _STATE["stats"][key] = 0
