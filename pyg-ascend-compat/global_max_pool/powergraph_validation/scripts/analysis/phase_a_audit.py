"""Phase A — PowerGraph raw-data / loader audit (no benchmarking).

Reads the raw ``.mat`` files that the unmodified PowerGraph ``PowerGrid`` loader
consumes and reports, per dataset:

  * number of graphs
  * nodes per graph, edges per graph (raw and post-contingency, i.e. after the
    loader deletes zeroed-out (tripped) branch rows)
  * ``x`` shape / dtype produced by the loader
  * label / explanation array shapes

Nothing is written into either repository.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import sys
import time

os.environ.setdefault("OMP_NUM_THREADS", "8")

import numpy as np  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # parent scripts/ dir (pg_env, pg_dataset)
import pg_env  # noqa: E402

pg_env.bootstrap_powergraph()
import mat73  # noqa: E402

DATASETS = ["ieee24", "ieee39", "ieee118", "uk"]


def load(path: str):
    t0 = time.time()
    d = mat73.loadmat(path)
    return d, time.time() - t0


def audit_dataset(data_root: str, name: str) -> dict:
    raw = os.path.join(data_root, name, name, "raw")
    out: dict = {"dataset": name, "raw_dir": raw, "files": {}}
    for fn in sorted(os.listdir(raw)):
        p = os.path.join(raw, fn)
        out["files"][fn] = {"size_bytes": os.path.getsize(p)}

    t_all = time.time()

    bf, t_bf = load(os.path.join(raw, "Bf.mat"))
    bf = bf["B_f_tot"]
    n_graphs = len(bf)
    out["B_f_tot"] = {
        "container": f"{type(bf).__name__}[{n_graphs}]",
        "element_shape": list(bf[0][0].shape),
        "element_dtype": str(bf[0][0].dtype),
        "load_s": round(t_bf, 2),
    }
    nodes = np.fromiter((bf[i][0].reshape(-1, 3).shape[0] for i in range(n_graphs)),
                        dtype=np.int64, count=n_graphs)
    out["nodes_per_graph"] = int(nodes[0])
    out["nodes_per_graph_constant"] = bool((nodes == nodes[0]).all())
    out["nodes_per_graph_stats"] = {"min": int(nodes.min()), "max": int(nodes.max()),
                                    "sum": int(nodes.sum())}
    del bf
    gc.collect()

    ef, t_ef = load(os.path.join(raw, "Ef.mat"))
    ef = ef["E_f_post"]
    ne = ef[0][0].reshape(-1, 4).shape[0]
    out["E_f_post"] = {
        "container": f"{type(ef).__name__}[{len(ef)}]",
        "element_shape": list(ef[0][0].shape),
        "element_dtype": str(ef[0][0].dtype),
        "load_s": round(t_ef, 2),
    }
    out["edges_defined_per_graph"] = int(ne)

    # contingency detection == loader's `cont = [j for j ... if np.all(f[j]) == 0]`
    n_cont = np.fromiter(
        (int(np.all(ef[i][0].reshape(-1, 4) == 0, axis=1).sum()) for i in range(n_graphs)),
        dtype=np.int64, count=n_graphs)
    used_edges = (ne - n_cont) * 2  # loader concatenates forward + reversed edges
    uniq = np.unique(n_cont)
    out["tripped_branches_per_graph"] = {
        "min": int(n_cont.min()),
        "max": int(n_cont.max()),
        "mean": float(n_cont.mean()),
        "num_unique": int(len(uniq)),
        "value_counts": {int(k): int(v) for k, v in zip(*np.unique(n_cont, return_counts=True))}
        if len(uniq) <= 20 else "many",
    }
    out["edge_index_edges_per_graph(used)"] = {
        "min": int(used_edges.min()),
        "max": int(used_edges.max()),
        "mean": float(used_edges.mean()),
        "num_unique": int(len(np.unique(used_edges))),
    }
    del ef, n_cont, used_edges
    gc.collect()

    out["num_graphs"] = n_graphs
    out["x_shape_per_graph"] = [int(nodes[0]), 3]
    out["x_dtype_produced_by_loader"] = "torch.float32"

    for key, field in (("of_bi.mat", "output_features"),
                       ("of_mc.mat", "category"),
                       ("of_reg.mat", "dns_MW"),
                       ("exp.mat", "explainations")):
        d, t = load(os.path.join(raw, key))
        arr = d[field]
        info = {"field": field, "load_s": round(t, 2)}
        try:
            info["shape"] = list(arr.shape)
            info["dtype"] = str(arr.dtype)
        except AttributeError:
            info["type"] = str(type(arr))
            info["len"] = len(arr)
        out[key] = info
        del d, arr
        gc.collect()

    blist, t = load(os.path.join(raw, "blist.mat"))
    b = blist["bList"]
    out["blist.mat"] = {
        "field": "bList",
        "shape": list(b.shape),
        "dtype": str(b.dtype),
        "min": float(np.min(b)),
        "max": float(np.max(b)),
        "load_s": round(t, 2),
    }
    del blist, b
    gc.collect()

    out["audit_seconds"] = round(time.time() - t_all, 2)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", default=pg_env.data_root())
    ap.add_argument("--out", default=os.path.join(
        os.environ.get("POWERGRAPH_RESULTS_ROOT", pg_env.results_root()),
        "phase_a_raw_audit.json"))
    ap.add_argument("--datasets", default=",".join(DATASETS))
    args = ap.parse_args()

    results = []
    for name in [d for d in args.datasets.split(",") if d]:
        print(f"[audit] {name} ...", flush=True)
        r = audit_dataset(args.data_root, name)
        results.append(r)
        print(f"[audit] {name}: graphs={r['num_graphs']} nodes/graph={r['nodes_per_graph']} "
              f"edges_defined={r['edges_defined_per_graph']} "
              f"x={r['x_shape_per_graph']} ({r['audit_seconds']}s)", flush=True)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as fh:
        json.dump({"datasets": results}, fh, indent=2)
    print(f"[audit] wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
