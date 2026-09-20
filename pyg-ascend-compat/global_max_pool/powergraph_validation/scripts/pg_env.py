"""Shared environment bootstrap for the PowerGraph global_max_pool validation.

Everything is resolved relative to this file, so the package is portable across
checkouts.  No repository is modified; only ``sys.path`` and two import shims
are prepared.

Resolved locations (each overridable by an environment variable):

    VALIDATION_ROOT        <repo>/pyg-ascend-compat/global_max_pool/powergraph_validation
    REPO_ROOT              the nanwang checkout containing this package
    GLOBAL_MAX_POOL_DIR    <repo>/pyg-ascend-compat/global_max_pool

Environment overrides:

    POWERGRAPH_UPSTREAM_DIR   PowerGraph-Graph checkout (read only)
    POWERGRAPH_DATA_ROOT      raw/processed PowerGraph data root
    POWERGRAPH_RESULTS_ROOT   benchmark output root
    PYG_ASCEND_COMPAT_DIR     directory containing the ``pyg_ascend_compat`` package
"""

from __future__ import annotations

import os
import sys
import types

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
VALIDATION_ROOT = os.path.dirname(_SCRIPT_DIR)
GLOBAL_MAX_POOL_DIR = os.path.dirname(VALIDATION_ROOT)
REPO_ROOT = os.path.dirname(os.path.dirname(GLOBAL_MAX_POOL_DIR))

# PowerGraph-Graph upstream checkout (never modified). Cloned by the reviewer,
# or pointed at an existing checkout with POWERGRAPH_UPSTREAM_DIR.
PG_REPO = os.environ.get(
    "POWERGRAPH_UPSTREAM_DIR",
    os.path.join(VALIDATION_ROOT, "upstream", "PowerGraph-Graph"),
)
PG_CODE = os.path.join(PG_REPO, "code")

# Frozen compat package lives in the "stage6" delivery directory of this repo.
COMPAT_DIR = os.environ.get(
    "PYG_ASCEND_COMPAT_DIR",
    os.path.join(GLOBAL_MAX_POOL_DIR, "stage6"),
)


def _add_path(p: str) -> None:
    if p not in sys.path:
        sys.path.insert(0, p)


def install_sklearn_shim() -> None:
    """Provide ``sklearn.model_selection.train_test_split`` only if absent.

    ``PowerGraph-Graph/code/dataset/powergrid.py`` imports ``train_test_split``
    at module import time but the ``PowerGrid`` class never calls it.  Rather
    than installing scikit-learn (which would pull scipy and risks perturbing
    the frozen Ascend environment), we register a stub that raises loudly if it
    is ever actually invoked.
    """
    try:  # pragma: no cover - environment dependent
        import sklearn.model_selection  # noqa: F401

        return
    except Exception:
        pass

    sklearn = types.ModuleType("sklearn")
    model_selection = types.ModuleType("sklearn.model_selection")

    def _unused(*_args, **_kwargs):
        raise RuntimeError(
            "sklearn.model_selection.train_test_split shim was called; "
            "this benchmark never intentionally uses it."
        )

    model_selection.train_test_split = _unused
    sklearn.model_selection = model_selection
    sys.modules.setdefault("sklearn", sklearn)
    sys.modules.setdefault("sklearn.model_selection", model_selection)


def install_powergraph_utils_shim() -> None:
    """Neutralise the transitive ``utils.gen_utils`` import of the loader.

    ``PowerGraph-Graph/code/dataset/powergrid.py`` imports two helpers from
    ``utils.gen_utils``.  That module additionally imports ``pandas`` and
    ``scipy`` at module scope, but neither helper is used anywhere in the
    ``PowerGrid`` code path (they belong to the synthetic/padded-graph code).

    Installing pandas+scipy into the frozen Ascend image is a ~200 MB
    environment change for imports that are never executed, so instead we
    register a stub for ``utils.gen_utils`` only.  The real ``utils`` package
    from the checkout is still used for everything else, and any accidental
    call of the two stubbed helpers raises immediately.
    """
    if "utils.gen_utils" in sys.modules:
        return

    stub = types.ModuleType("utils.gen_utils")

    def _unused(*_args, **_kwargs):
        raise RuntimeError(
            "PowerGraph utils.gen_utils helper invoked at runtime; the benchmark's "
            "stub only exists to avoid importing pandas/scipy. Re-run with "
            "pandas+scipy installed if this helper is genuinely needed."
        )

    stub.from_edge_index_to_adj = _unused
    stub.padded_datalist = _unused
    sys.modules["utils.gen_utils"] = stub


def bootstrap_powergraph() -> None:
    """Make the (unmodified) PowerGraph dataset loader importable."""
    install_sklearn_shim()
    install_powergraph_utils_shim()
    _add_path(PG_CODE)


def bootstrap_compat(debug: bool = False):
    """Enable the frozen PyG Ascend compat layer *before* importing PyG.

    Returns the dict of wrapped modules (the return value of ``enable``).
    """
    _add_path(COMPAT_DIR)
    import pyg_ascend_compat  # noqa: E402

    return pyg_ascend_compat.enable(debug=debug)


def data_root() -> str:
    return os.environ.get("POWERGRAPH_DATA_ROOT", os.path.join(VALIDATION_ROOT, "data"))


def results_root() -> str:
    return os.environ.get("POWERGRAPH_RESULTS_ROOT", os.path.join(VALIDATION_ROOT, "results"))


def profile_root() -> str:
    return os.environ.get(
        "POWERGRAPH_PROFILE_ROOT", os.path.join(VALIDATION_ROOT, "profiler_runs")
    )
