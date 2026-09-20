"""Phase A — verify the unmodified PowerGraph loader under PyG 2.8.0.post1.

Builds the ``PowerGrid`` InMemoryDataset for each requested dataset, reporting
graph counts, per-graph node/edge counts, ``x`` shape/dtype and the on-disk
processed artefact.  No benchmarking happens here.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # parent scripts/ dir (pg_env, pg_dataset)
import pg_dataset  # noqa: E402
import pg_env  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", default="ieee24")
    ap.add_argument("--data-root", default=pg_env.data_root())
    ap.add_argument("--out", default=os.path.join(
        os.environ.get("POWERGRAPH_RESULTS_ROOT", pg_env.results_root()),
        "phase_a_loader.json"))
    args = ap.parse_args()

    results = []
    for name in [d for d in args.datasets.split(",") if d]:
        print(f"[loader] {name} ...", flush=True)
        t0 = time.time()
        ds = pg_dataset.get_dataset(args.data_root, name, "Binary")
        build_s = time.time() - t0
        stats = pg_dataset.dataset_workload_stats(ds)
        proc = os.path.join(args.data_root, name, name, "processed_b", "data.pt")
        stats.update({
            "dataset": name,
            "build_seconds": round(build_s, 2),
            "processed_path": proc,
            "processed_bytes": os.path.getsize(proc) if os.path.exists(proc) else None,
            "num_classes_binary": int(ds._data.y.max().item()) + 1
            if ds._data.y is not None else None,
        })
        batch, bs = pg_dataset.first_batch(ds, 4)
        stats["first_batch_smoke"] = {
            "batch_size": bs,
            "batch_x_shape": list(batch.x.shape),
            "batch_x_dtype": str(batch.x.dtype),
            "batch_index_dtype": str(batch.batch.dtype),
            "num_graphs_in_batch": int(batch.batch.max().item()) + 1,
            "num_features": int(batch.x.size(1)),
        }
        results.append(stats)
        print(f"[loader] {name}: graphs={stats['num_graphs']} F={stats['feature_dim']} "
              f"x={stats['x_dtype']} nodes/graph={stats['nodes_per_graph_min']}.."
              f"{stats['nodes_per_graph_max']} edges/graph={stats['edges_per_graph_min']}.."
              f"{stats['edges_per_graph_max']} ({build_s:.1f}s) -> {stats['processed_bytes']} bytes",
              flush=True)
        del ds, batch

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as fh:
        json.dump({"datasets": results}, fh, indent=2)
    print(f"[loader] wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
