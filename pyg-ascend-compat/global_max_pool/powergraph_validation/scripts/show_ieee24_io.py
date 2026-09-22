#!/usr/bin/env python3
"""Show the real PowerGraph input/output of the frozen Ascend ``global_max_pool``.

This is the smallest end-to-end demonstration for a customer: it loads one real
PyG batch from the *unmodified* PowerGraph ``PowerGrid`` dataset, runs the frozen
Ascend ``global_max_pool`` on the NPU, and compares the result against a plain
CPU reference produced by an explicit per-graph ``max`` loop.

Nothing here re-implements the PowerGraph loader: it reuses ``pg_env`` and
``pg_dataset`` exactly like the benchmark scripts, preserving the required
import / enable order (compat layer enabled *before* importing PyG).

Run::

    cd pyg-ascend-compat/global_max_pool/powergraph_validation
    source scripts/bench_env.sh
    python3 scripts/show_ieee24_io.py

Options::

    --dataset     graph-level dataset name   (default: ieee24)
    --batch-size  number of graphs per batch (default: 2)
    --dtype       fp32 | fp16 | bf16         (default: fp32)

If the PowerGraph data or the upstream loader checkout cannot be found the
script prints actionable instructions instead of a raw traceback.
"""

from __future__ import annotations

import argparse
import os
import sys

# --- required order: enable the frozen compat layer BEFORE importing PyG -----
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pg_env  # noqa: E402

pg_env.bootstrap_compat(debug=False)

import torch  # noqa: E402
from torch_geometric.nn import global_max_pool  # noqa: E402

import pg_dataset  # noqa: E402

DEV = "npu:0"
DTYPES = {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}

# CPU-reference tolerances, identical to the benchmark oracle thresholds.
TOL = {
    "fp32": {"rtol": 1e-5, "atol": 1e-6, "require_exact": True},
    "fp16": {"rtol": 2e-3, "atol": 1e-2, "require_exact": False},
    "bf16": {"rtol": 2e-2, "atol": 1e-1, "require_exact": False},
}


def precheck(data_root: str, dataset: str) -> None:
    """Fail early with a readable message when data / upstream are missing."""
    upstream = pg_env.PG_REPO
    loader = os.path.join(upstream, "code", "dataset", "powergrid.py")
    raw_mat = os.path.join(data_root, dataset, dataset, "raw", "Bf.mat")
    processed = os.path.join(data_root, dataset, dataset, "processed_b", "data.pt")

    if not os.path.isfile(loader):
        print("PowerGraph upstream checkout 未找到。", file=sys.stderr)
        print("", file=sys.stderr)
        print(f"当前解析到的路径: {upstream}", file=sys.stderr)
        print("", file=sys.stderr)
        print("请先阅读:", file=sys.stderr)
        print("powergraph_validation/DATASET.md", file=sys.stderr)
        print("", file=sys.stderr)
        print("或设置:", file=sys.stderr)
        print("POWERGRAPH_UPSTREAM_DIR=/path/to/PowerGraph-Graph", file=sys.stderr)
        raise SystemExit(2)

    if not os.path.isfile(raw_mat) and not os.path.isfile(processed):
        print("PowerGraph 数据未找到。", file=sys.stderr)
        print("", file=sys.stderr)
        print(f"数据集 '{dataset}' 期望的原始数据: {raw_mat}", file=sys.stderr)
        print(f"或已处理的缓存: {processed}", file=sys.stderr)
        print("", file=sys.stderr)
        print("请先阅读:", file=sys.stderr)
        print("powergraph_validation/DATASET.md", file=sys.stderr)
        print("", file=sys.stderr)
        print("或设置:", file=sys.stderr)
        print("POWERGRAPH_DATA_ROOT=/path/to/powergraph/data", file=sys.stderr)
        print("POWERGRAPH_UPSTREAM_DIR=/path/to/PowerGraph-Graph", file=sys.stderr)
        raise SystemExit(2)


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Show real PowerGraph input/output of Ascend global_max_pool.",
    )
    ap.add_argument("--dataset", default="ieee24",
                    help="graph-level dataset name (default: ieee24)")
    ap.add_argument("--batch-size", type=int, default=2,
                    help="number of graphs per batch (default: 2)")
    ap.add_argument("--dtype", default="fp32", choices=sorted(DTYPES),
                    help="execution dtype: fp32 | fp16 | bf16 (default: fp32)")
    args = ap.parse_args()

    dtype_name = args.dtype
    tdt = DTYPES[dtype_name]
    tol = TOL[dtype_name]

    data_root = pg_env.data_root()
    precheck(data_root, args.dataset)

    dataset = pg_dataset.get_dataset(data_root, args.dataset, "Binary")
    dataset_size = dataset.len()
    batch, bs_eff = pg_dataset.first_batch(dataset, args.batch_size)

    # Real input, kept in the dataset's native precision.
    x_cpu = batch.x.detach().contiguous()
    batch_cpu = batch.batch.to(torch.int64).contiguous()
    num_graphs = int(batch_cpu.max().item()) + 1
    total_nodes, feat = x_cpu.shape
    nodes_per_graph = torch.bincount(batch_cpu, minlength=num_graphs)

    # The op executes (and the CPU reference is computed) in the requested dtype.
    x_ref = x_cpu.to(tdt)

    print("========== POWERGRAPH DATASET ==========")
    print(f"dataset      : {args.dataset}")
    print(f"dataset size : {dataset_size}")
    print(f"batch graphs : {num_graphs}")
    print(f"x.shape      : {tuple(x_cpu.shape)}")
    print(f"batch.shape  : {tuple(batch_cpu.shape)}")
    print(f"x.dtype      : {x_cpu.dtype}")
    print(f"nodes/graph  : {nodes_per_graph}")

    print("\n========== REAL INPUT ==========")
    print("x 前 10 个节点:")
    print(x_cpu[:10])
    print("batch 前 30 个元素:")
    print(batch_cpu[:30])

    x_npu = x_ref.to(DEV)
    batch_npu = batch_cpu.to(DEV)

    print("\n========== NPU INPUT ==========")
    print(f"x.device     : {x_npu.device}")
    print(f"batch.device : {batch_npu.device}")
    print(f"exec dtype   : {dtype_name} ({x_npu.dtype})")

    with torch.no_grad():
        out_npu = global_max_pool(x_npu, batch_npu)
    torch.npu.synchronize()

    print("\n========== NPU OUTPUT ==========")
    print(out_npu)
    print(f"out.shape  : {tuple(out_npu.shape)}")
    print(f"out.device : {out_npu.device}")
    print(f"out.dtype  : {out_npu.dtype}")

    expected = torch.stack([
        x_ref[batch_cpu == g].max(dim=0).values
        for g in range(num_graphs)
    ])

    print("\n========== CPU EXPECTED ==========")
    print(expected)

    out_cpu = out_npu.detach().to("cpu").to(expected.dtype)
    exact = bool(torch.equal(out_cpu, expected))
    allclose = bool(torch.allclose(out_cpu.float(), expected.float(),
                                   rtol=tol["rtol"], atol=tol["atol"]))

    print("\n========== CHECK ==========")
    print(f"exact equal : {exact}")
    print(f"allclose    : {allclose}")
    print(f"tolerance   : rtol={tol['rtol']}, atol={tol['atol']}")

    if tol["require_exact"] and not exact:
        print(f"\n[FAIL] {dtype_name} requires exact equality but the outputs differ.")
        return 1
    if not allclose:
        print(f"\n[FAIL] {dtype_name} outputs are outside tolerance.")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
