"""pyg-ascend-compat — route ``torch_geometric.nn.global_max_pool`` to the Ascend adapter.

Usage::

    import pyg_ascend_compat
    pyg_ascend_compat.enable(debug=False)      # one call, then use PyG normally
    from torch_geometric.nn import global_max_pool
    out = global_max_pool(x, batch, size)      # -> ScatterMaxV1 on Ascend for fp32/NPU

The wrapper is installed at runtime on the ``torch_geometric`` module objects; site-packages is
never modified and the original function is restored by :func:`disable`.

Dispatch (Stage 6 scope, forward only):

* x on NPU, float32, 2-D, ``requires_grad=False``
* batch on the same NPU, int64, 1-D, ``batch.numel() == x.size(0)``
* anything else → the original PyG implementation (e.g. CPU, fp16/bf16, 1-D x)
* NPU + float32 + ``requires_grad=True`` → explicit RuntimeError (Stage 6 has no backward);
  the call is never silently detached and never sent to the CPU fallback.
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
    "stats": {
        "total_calls": 0,
        "ascend_calls": 0,
        "original_calls": 0,
        "requires_grad_rejected": 0,
        "non_fp32_passthrough": 0,
    },
}

_ADAPTER_ENV = "PYG_ASCEND_ADAPTER_PATH"
_ADAPTER_REL = os.path.join("global_max_pool", "stage2", "python", "global_max_pool_ascend.py")
_ADAPTER_FALLBACKS = (
    "/root/zyg/global_max_pool/stage2/python/global_max_pool_ascend.py",
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
    if x.device.type != "npu" or x.dtype != torch.float32 or x.dim() != 2:
        if isinstance(x, torch.Tensor) and x.device.type == "npu" and x.dtype != torch.float32:
            _STATE["stats"]["non_fp32_passthrough"] += 1
            if _STATE["debug"]:
                print(f"[pyg-ascend-compat] passthrough (dtype={x.dtype}) -> original PyG path")
        return None
    if batch is None or not isinstance(batch, torch.Tensor):
        return None
    if (batch.device.type != "npu" or batch.device != x.device or batch.dtype != torch.int64
            or batch.dim() != 1 or batch.numel() != x.size(0)):
        return None
    if x.requires_grad:
        _STATE["stats"]["requires_grad_rejected"] += 1
        raise RuntimeError(
            "pyg-ascend-compat (Stage 6) supports the Ascend global_max_pool forward pass only; "
            "x.requires_grad=True is not supported yet (backward/tie-gradient is Stage 4). "
            "Detach the input or run under torch.no_grad(). The call is not silently sent to the "
            "CPU fallback."
        )
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
    return out


def reset_stats() -> None:
    for key in _STATE["stats"]:
        _STATE["stats"][key] = 0
