"""Helpers for loading the (unmodified) PowerGraph graph-level datasets.

Nothing here patches a repository: the loader module is imported straight from
the read-only PowerGraph-Graph checkout and only two *runtime* accommodation
shims are applied in this process:

1. ``sklearn.model_selection.train_test_split`` — imported by the loader but
   never called.  We register a stub instead of installing scikit-learn.
2. ``torch.load(..., weights_only=False)`` — PyTorch >= 2.6 defaults to
   ``weights_only=True``, which cannot unpickle a PyG ``Data`` object.  The
   frozen loader calls ``torch.load`` without arguments, so we temporarily
   restore the pre-2.6 default for the duration of dataset construction.
"""

from __future__ import annotations

import contextlib
import importlib.util
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pg_env  # noqa: E402

pg_env.bootstrap_powergraph()


@contextlib.contextmanager
def _torch_load_weights_only_false():
    original = torch.load

    def patched(*args, **kwargs):
        kwargs.setdefault("weights_only", False)
        return original(*args, **kwargs)

    torch.load = patched
    try:
        yield
    finally:
        torch.load = original


def load_powergrid_module():
    path = os.path.join(pg_env.PG_CODE, "dataset", "powergrid.py")
    spec = importlib.util.spec_from_file_location("pg_powergrid", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_PG_MODULE = None


def get_dataset(data_root: str, name: str, datatype: str = "Binary", force_reprocess: bool = False):
    """Return a ``PowerGrid`` dataset; process the raw data on first use."""
    global _PG_MODULE
    if _PG_MODULE is None:
        _PG_MODULE = load_powergrid_module()
    root = os.path.join(data_root, name)
    with _torch_load_weights_only_false():
        return _PG_MODULE.PowerGrid(root=root, name=name, datatype=datatype)


def dataset_workload_stats(dataset) -> dict:
    """Exact per-dataset graph workload statistics taken from the processed data."""
    n_graphs = int(dataset.len())
    slices = dataset.slices
    n_per_graph = (slices["x"][1:] - slices["x"][:-1]).to(torch.int64)
    e_per_graph = (slices["edge_index"][1:] - slices["edge_index"][:-1]).to(torch.int64)
    x = dataset._data.x
    uniq_n = int(torch.unique(n_per_graph).numel())
    uniq_e = int(torch.unique(e_per_graph).numel())
    return {
        "num_graphs": n_graphs,
        "feature_dim": int(x.size(1)),
        "x_dtype": str(x.dtype),
        "total_nodes_all_graphs": int(x.size(0)),
        "total_directed_edges_all_graphs": int(dataset._data.edge_index.size(1)),
        "nodes_per_graph_min": int(n_per_graph.min()),
        "nodes_per_graph_max": int(n_per_graph.max()),
        "nodes_per_graph_mean": float(n_per_graph.double().mean()),
        "nodes_per_graph_num_unique": uniq_n,
        "nodes_per_graph_constant": uniq_n == 1,
        "edges_per_graph_min": int(e_per_graph.min()),
        "edges_per_graph_max": int(e_per_graph.max()),
        "edges_per_graph_mean": float(e_per_graph.double().mean()),
        "edges_per_graph_num_unique": uniq_e,
        "edges_per_graph_constant": uniq_e == 1,
    }


def first_batch(dataset, batch_size: int):
    """First PyG DataLoader batch (shuffle=False) of the requested size."""
    from torch_geometric.loader import DataLoader

    bs = min(batch_size, dataset.len())
    loader = DataLoader(dataset, batch_size=bs, shuffle=False)
    batch = next(iter(loader))
    return batch, bs
