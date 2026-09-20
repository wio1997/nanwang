# POWERGRAPH GLOBAL_MAX_POOL PERFORMANCE — EVIDENCE

Independent single-operator performance benchmark of the **frozen** PyG Ascend
`global_max_pool` implementation on **real PowerGraph power-grid graph data**.
No operator development, no GNN training, no repository modification.

## 1. Environment

| item | value |
|---|---|
| server | `S900K3-47` |
| container | `wio-pyg-cann851-pyg280` (container hostname `caef109ccb29`) |
| platform | `Linux-5.15.0-25-generic-aarch64-with-glibc2.35` |
| accelerators | 8 x Ascend 910B3, benchmark pinned via `ASCEND_RT_VISIBLE_DEVICES=0` |
| python | 3.11.14 |
| torch | 2.9.0+cpu |
| torch_npu | 2.9.0 |
| PyG | 2.8.0.post1 |
| numpy | 2.4.6 |
| device | Ascend910B3 |
| CANN | 8.5.1 (`/usr/local/Ascend/cann-8.5.1`) |

Environment changes made for this benchmark (recorded before/after in
`logs/pip_freeze_before.txt` and `logs/pip_freeze_after.txt`):

```
h5py==3.16.0      (new)
mat73==0.65       (new)
```

`torch 2.9.0+cpu`, `torch_npu 2.9.0`, `torch_geometric 2.8.0.post1` and
`numpy 2.4.6` were **not touched**. The PowerGraph `requirements.txt` was **not**
installed. `sklearn.model_selection.train_test_split` and `utils.gen_utils`
(unused by the `PowerGrid` code path but imported at module scope) are satisfied
by in-process stubs so that neither scikit-learn/scipy nor pandas had to be
installed; both stubs raise loudly if actually called.

## 2. Frozen operator provenance

```
repo=https://github.com/wio1997/nanwang
branch=feat/global-max-pool-scattermax-zyg
head=d1dc616b809c32819f3f1fab9a8c43528f824f05
frozen_operator_sha=c15423e7b303d2b1597621c64585252472301537
final_docs_head=d1dc616b809c32819f3f1fab9a8c43528f824f05
git_status_porcelain_lines=0
diff_frozen_to_head_files:
pyg-ascend-compat/global_max_pool/README_DELIVERY.md

--- sha256 frozen sources ---
417871482f1c7aa4fcf1f4cab16c3581958de5162d1499a1cc4db1220efbcd13  pyg-ascend-compat/global_max_pool/stage6/pyg_ascend_compat/__init__.py
60e037f3104dc8674c9f1a1a98d646f31d037f09764465e57036bdf9a1df0119  pyg-ascend-compat/global_max_pool/stage2/python/global_max_pool_ascend.py
6647ee038a30529a71f05745427300c0258eded0182d172ad8c5105dd4d7666c  pyg-ascend-compat/global_max_pool/stage4/python/global_max_pool_ascend_autograd.py
cac35c11c2e31ca9872ad70389dc1d85948b53cb1f0f51b7978684624299cf92  pyg-ascend-compat/global_max_pool/stage5/python/global_max_pool_ascend_dtype.py
773198c3b1c77e7d755b0ac3f4cb91f794aed6718de839c230d5e2cdb488a016  pyg-ascend-compat/global_max_pool/stage3e/delivery_source/op_kernel/scatter_max_v1.cpp
ab7fd9d2d18796e35a66df07dcaac82e625e07eed76341e217487acf4e83f654  pyg-ascend-compat/global_max_pool/stage3e/delivery_source/op_host/scatter_max_v1.cpp

--- sha256 installed OPP ---
47f31ff0fb949485e7b7dedc25e09963ae5442515a9d30b31f5010729b5a27bc  /root/zyg/build/scattermax_runtime_opp/vendors/customize/framework/plugin/npu_supported_ops.json
318a4b2c8904679d30bb268ef64fe0fa7f8f74b59f526cb94e7196c34bb4e012  /root/zyg/build/scattermax_runtime_opp/vendors/customize/op_api/lib/libcust_opapi.so
c513014df6503cacd7ca56970eb9c6201fac2a8111d09bb3f5320f93b2d08294  /root/zyg/build/scattermax_runtime_opp/vendors/customize/op_impl/ai_core/tbe/config/ascend910b/aic-ascend910b-ops-info.json
4fd053d121722475e7db47cf6173e43ae24804623b93b8cf0aa1fbf25f5cd070  /root/zyg/build/scattermax_runtime_opp/vendors/customize/op_impl/ai_core/tbe/kernel/ascend910b/scatter_max_argmax_v1/ScatterMaxArgmaxV1_798fddcf56f6774ec7fa251da5d51746.json
755a59daeba86e1388c860d633f3dfc94468975832bf05abc5695bb4802130a8  /root/zyg/build/scattermax_runtime_opp/vendors/customize/op_impl/ai_core/tbe/kernel/ascend910b/scatter_max_v1/ScatterMaxV1_7d55161965c898907fdb3028d01c7c76.json
489cc936a8fe5820f7547f3dac99c4c7503421519302806865b46fa564c67e6e  /root/zyg/build/scattermax_runtime_opp/vendors/customize/op_impl/ai_core/tbe/kernel/config/ascend910b/binary_info_config.json
b0685e8ec523103ea1820c01f35e371eae637b951daf3f6f437ee6f20c86cfa9  /root/zyg/build/scattermax_runtime_opp/vendors/customize/op_impl/ai_core/tbe/kernel/config/ascend910b/scatter_max_argmax_v1.json
cc5a4d60f04142a01abadaf73648b2de040c59985bf7d502d41e41fd185c35e4  /root/zyg/build/scattermax_runtime_opp/vendors/customize/op_impl/ai_core/tbe/kernel/config/ascend910b/scatter_max_v1.json
efad8b7293e7e1ae701474eb0984733a2624eeaed4ceee97fc4bc615c8c5e9d1  /root/zyg/build/scattermax_runtime_opp/vendors/customize/op_impl/ai_core/tbe/op_tiling/lib/linux/aarch64/libcust_opmaster_rt2.0.so
850fcbdb266e9ea139d9e8b784879ca2f04c03ee8e6f05351eae3f0c5cd53776  /root/zyg/build/scattermax_runtime_opp/vendors/customize/op_proto/lib/linux/aarch64/libcust_opsproto_rt2.0.so

--- sha256 bridge ---
f83769d63eb111cf8c7b331f2aae3e80e0dd1279a4583bfbf9cb2c51eaf5c519  /root/zyg/build/stage2_ext/scattermaxv1_bridge.so
```

The only difference between the frozen operator commit and the final docs HEAD
is `pyg-ascend-compat/global_max_pool/README_DELIVERY.md`; the operator, adapter,
autograd and dtype sources are byte-identical, and the checked-out worktree is
clean.

The benchmark always imports in this order:

```python
import pyg_ascend_compat
pyg_ascend_compat.enable()
from torch_geometric.nn import global_max_pool
```

## 3. PowerGraph data provenance

| item | value |
|---|---|
| repository | `https://github.com/PowerGraph-Datasets/PowerGraph-Graph` |
| commit used (read-only clone) | `eb100a2fd836bb8b6bd2d0b799af9c615eac8cb6` |
| file used | fetches exactly what README links: figshare article `22820534`, file id `46619158` (`dataset_cascades.zip`, v3) |
| download bytes / md5 | 61,628,977 / `70b677416d2f377ccfee9f51d8369867` |
| uncompressed | 2,958,249,040 bytes (2.75 GiB) |
| also fetched | file id `50083479` (v5 `dataset_cascades.zip`, md5 verified `d4d144b9e720a760e1e077a31f34802d`) — same files, same sizes, only an extra top-level directory |

In the **original validation environment** figshare.com returned HTTP 403 for
every path (article page, API and downloader alike, IPv4 and IPv6). The data was
therefore obtained by retrieving figshare's presigned S3 redirect through a
public HTTP proxy and then downloading the payload **directly from
`s3-eu-west-1.amazonaws.com/pfigshare-u-files/...`**; no proxy was used for the
payload transfer. See `DATASET.md` for the portable download procedure used by
the packaged scripts (`scripts/fetch_powergraph_data.sh`), which prefers the
official figshare URL and verifies the checksum. The dataset is stored under
`POWERGRAPH_DATA_ROOT` (default `<package>/data`) and is never committed to git.

All four datasets are present and were processed by the **unmodified**
`PowerGrid` `InMemoryDataset` loader under PyG 2.8.0.post1.

## 4. Dataset availability and workload statistics

Raw `.mat` audit (before PyG processing):

| dataset | graphs | nodes/graph (raw) | branches defined/graph | loader x shape | loader x dtype | edge_index edges/graph min | edge_index edges/graph max | edge_index edges/graph mean | tripped branches/graph min | tripped branches/graph max |
|---|---|---|---|---|---|---|---|---|---|---|
| ieee24 | 21500 | 24 | 38 | [24, 3] | torch.float32 | 68 | 74 | 73.76 | 1 | 4 |
| ieee39 | 28000 | 39 | 46 | [39, 3] | torch.float32 | 86 | 90 | 89.64 | 1 | 3 |
| ieee118 | 122500 | 118 | 186 | [118, 3] | torch.float32 | 362 | 370 | 369.44 | 1 | 5 |
| uk | 64000 | 29 | 99 | [29, 3] | torch.float32 | 190 | 196 | 195.53 | 1 | 4 |

Processed-dataset workload (exact, from `dataset.slices`):

| dataset | graphs | F | nodes/graph min | nodes/graph max | nodes/graph mean | edges/graph min | edges/graph max | edges/graph mean | nodes/graph constant | distinct edge counts |
|---|---|---|---|---|---|---|---|---|---|---|
| ieee24 | 21500 | 3 | 24 | 24 | 24.00 | 68 | 74 | 73.76 | True | 4 |
| ieee39 | 28000 | 3 | 39 | 39 | 39.00 | 86 | 90 | 89.64 | True | 3 |
| ieee118 | 122500 | 3 | 118 | 118 | 118.00 | 362 | 370 | 369.44 | True | 5 |
| uk | 64000 | 3 | 29 | 29 | 29.00 | 190 | 196 | 195.53 | True | 4 |

Nodes per graph are **fixed per dataset** (24 / 39 / 118 / 29); only the number
of *tripped branches* varies per graph, which changes `edge_index` /
`edge_attr` size between 68-74, 86-90, 362-370 and 190-196 directed edges
respectively. Node features are always `x: float32 [N, 3]` (net active power,
net apparent power, voltage magnitude), so `F = 3` for every dataset — the
column count is much smaller than the node axis, which matters for interpreting
the numbers below.

## 5. Benchmark methodology

* Input is a **real PyG DataLoader batch** (`torch_geometric.loader.DataLoader`,
  `shuffle=False`, first batch) of the requested size; only `batch.x` and
  `batch.batch` are moved to the NPU. No GNN convolution, no Linear, no
  optimizer, no training.
* `out = global_max_pool(x, batch_index)` inside `torch.no_grad()`.
* warmup = 30, measurement iterations = 200.
* `torch.npu.synchronize()` is called after warmup and again after the last
  launch; **no timer is taken around unsynchronised device work**.
* Primary metric: per-iteration `torch.npu.Event(enable_timing=True)` pairs,
  giving device-side per-op durations; mean/P50/P95/P99 are computed from those
  per-iteration samples (kept in `*_per_iter_*.csv`).
* Secondary metric: a fully synchronised loop (`sync_*` columns) that calls
  `torch.npu.synchronize()` before and after every single call, i.e. it also
  contains host launch overhead.
* `host_total_us` is the batch-timer cross-check (synchronise, t0, N launches,
  synchronise, t1) divided by N.

### 5.1 Important measurement caveat

The frozen Stage-2 adapter performs a **mandatory device-to-host synchronisation
on every call** (`torch.stack((batch.min(), batch.max())).cpu().tolist()` for
index-range validation, plus `size` inference) and issues roughly nine auxiliary
NPU operations (`to(int32)`, `F.pad`, `torch.full`, `torch.empty`, the
ScatterMaxV1 launch, `zeros`+`scatter_` occupancy, `masked_fill_`, cropped
`contiguous`). Consequently the packed-launch loop cannot actually queue work:
the measured latency **is** the true per-call latency of the frozen operator,
not a pipelined throughput figure. Both loops are reported so this is visible.

## 6. FP32 performance

| dataset | batch | graphs | nodes | F | dtype | mean_us | p50_us | p95_us | p99_us | graphs_per_sec | nodes_per_sec |
|---|---|---|---|---|---|---|---|---|---|---|---|
| ieee24 | 1 | 1 | 24 | 3 | fp32 | 914.24 | 925.71 | 970.82 | 995.26 | 1,093.8 | 26,251.3 |
| ieee24 | 8 | 8 | 192 | 3 | fp32 | 889.07 | 888.51 | 939.03 | 950.91 | 8,998.2 | 215,955.7 |
| ieee24 | 32 | 32 | 768 | 3 | fp32 | 894.23 | 894.20 | 942.93 | 961.53 | 35,784.9 | 858,836.6 |
| ieee24 | 128 | 128 | 3072 | 3 | fp32 | 921.36 | 897.73 | 951.75 | 1,008.3 | 138,925.1 | 3,334,202 |
| ieee39 | 1 | 1 | 39 | 3 | fp32 | 971.30 | 970.29 | 1,026.8 | 1,075.8 | 1,029.5 | 40,152.4 |
| ieee39 | 8 | 8 | 312 | 3 | fp32 | 917.79 | 933.51 | 995.73 | 1,011.3 | 8,716.6 | 339,946.6 |
| ieee39 | 32 | 32 | 1248 | 3 | fp32 | 877.62 | 866.83 | 936.08 | 1,114.2 | 36,462.1 | 1,422,020 |
| ieee39 | 128 | 128 | 4992 | 3 | fp32 | 937.63 | 930.94 | 1,013.5 | 1,076.5 | 136,514.5 | 5,324,066 |
| ieee118 | 1 | 1 | 118 | 3 | fp32 | 856.06 | 822.64 | 874.01 | 979.78 | 1,168.1 | 137,840.5 |
| ieee118 | 8 | 8 | 944 | 3 | fp32 | 875.66 | 872.08 | 910.18 | 932.92 | 9,136.0 | 1,078,049 |
| ieee118 | 32 | 32 | 3776 | 3 | fp32 | 979.09 | 974.39 | 1,040.3 | 1,053.5 | 32,683.3 | 3,856,629 |
| ieee118 | 128 | 128 | 15104 | 3 | fp32 | 1,132.5 | 1,132.4 | 1,179.6 | 1,232.0 | 113,021.1 | 13,336,486 |
| uk | 1 | 1 | 29 | 3 | fp32 | 995.14 | 990.23 | 1,111.3 | 1,188.3 | 1,004.9 | 29,141.6 |
| uk | 8 | 8 | 232 | 3 | fp32 | 841.61 | 839.63 | 870.69 | 902.99 | 9,505.6 | 275,661.3 |
| uk | 32 | 32 | 928 | 3 | fp32 | 865.91 | 866.69 | 903.54 | 932.28 | 36,955.4 | 1,071,706 |
| uk | 128 | 128 | 3712 | 3 | fp32 | 858.09 | 854.94 | 897.61 | 918.97 | 149,169.2 | 4,325,906 |

## 7. FP16 performance

FP16 is **not** a native kernel: it is a device cast chain
`fp16 -> fp32 -> ScatterMaxV1 -> fp16`.

| dataset | batch | graphs | nodes | F | dtype | mean_us | p50_us | p95_us | p99_us | graphs_per_sec | nodes_per_sec |
|---|---|---|---|---|---|---|---|---|---|---|---|
| ieee24 | 1 | 1 | 24 | 3 | fp16 | 916.17 | 913.29 | 950.54 | 974.05 | 1,091.5 | 26,196.0 |
| ieee24 | 8 | 8 | 192 | 3 | fp16 | 946.50 | 943.50 | 985.25 | 1,051.2 | 8,452.2 | 202,851.6 |
| ieee24 | 32 | 32 | 768 | 3 | fp16 | 956.68 | 950.42 | 1,025.0 | 1,067.7 | 33,449.0 | 802,776.3 |
| ieee24 | 128 | 128 | 3072 | 3 | fp16 | 945.58 | 936.62 | 993.35 | 1,045.0 | 135,366.2 | 3,248,790 |
| ieee39 | 1 | 1 | 39 | 3 | fp16 | 943.61 | 937.25 | 1,003.5 | 1,016.1 | 1,059.8 | 41,330.8 |
| ieee39 | 8 | 8 | 312 | 3 | fp16 | 985.93 | 980.76 | 1,070.3 | 1,107.3 | 8,114.2 | 316,452.3 |
| ieee39 | 32 | 32 | 1248 | 3 | fp16 | 1,026.8 | 1,012.2 | 1,123.5 | 1,555.9 | 31,165.3 | 1,215,446 |
| ieee39 | 128 | 128 | 4992 | 3 | fp16 | 950.71 | 943.35 | 1,001.0 | 1,050.6 | 134,635.7 | 5,250,793 |
| ieee118 | 1 | 1 | 118 | 3 | fp16 | 902.02 | 896.02 | 936.82 | 1,063.6 | 1,108.6 | 130,817.4 |
| ieee118 | 8 | 8 | 944 | 3 | fp16 | 1,111.7 | 1,109.9 | 1,158.4 | 1,171.1 | 7,196.0 | 849,129.2 |
| ieee118 | 32 | 32 | 3776 | 3 | fp16 | 1,108.9 | 1,105.0 | 1,154.9 | 1,169.6 | 28,858.3 | 3,405,279 |
| ieee118 | 128 | 128 | 15104 | 3 | fp16 | 1,117.7 | 1,105.0 | 1,215.2 | 1,284.4 | 114,524.0 | 13,513,830 |
| uk | 1 | 1 | 29 | 3 | fp16 | 1,106.8 | 1,091.3 | 1,233.6 | 1,303.5 | 903.5 | 26,201.3 |
| uk | 8 | 8 | 232 | 3 | fp16 | 989.53 | 991.06 | 1,062.2 | 1,106.2 | 8,084.6 | 234,454.6 |
| uk | 32 | 32 | 928 | 3 | fp16 | 924.76 | 923.60 | 960.02 | 989.04 | 34,603.4 | 1,003,499 |
| uk | 128 | 128 | 3712 | 3 | fp16 | 912.01 | 910.41 | 949.17 | 994.67 | 140,348.9 | 4,070,118 |

## 8. BF16 performance

BF16 is likewise `bf16 -> fp32 -> ScatterMaxV1 -> bf16`.

| dataset | batch | graphs | nodes | F | dtype | mean_us | p50_us | p95_us | p99_us | graphs_per_sec | nodes_per_sec |
|---|---|---|---|---|---|---|---|---|---|---|---|
| ieee24 | 1 | 1 | 24 | 3 | bf16 | 955.64 | 957.52 | 1,008.3 | 1,048.3 | 1,046.4 | 25,114.2 |
| ieee24 | 8 | 8 | 192 | 3 | bf16 | 956.27 | 948.65 | 1,029.2 | 1,081.5 | 8,365.9 | 200,780.8 |
| ieee24 | 32 | 32 | 768 | 3 | bf16 | 900.21 | 899.44 | 947.61 | 964.64 | 35,547.3 | 853,135.7 |
| ieee24 | 128 | 128 | 3072 | 3 | bf16 | 937.33 | 931.65 | 1,010.5 | 1,051.8 | 136,558.8 | 3,277,410 |
| ieee39 | 1 | 1 | 39 | 3 | bf16 | 983.12 | 992.82 | 1,043.3 | 1,073.0 | 1,017.2 | 39,669.5 |
| ieee39 | 8 | 8 | 312 | 3 | bf16 | 940.10 | 928.65 | 1,018.3 | 1,029.6 | 8,509.8 | 331,881.0 |
| ieee39 | 32 | 32 | 1248 | 3 | bf16 | 1,009.8 | 1,007.1 | 1,076.2 | 1,183.9 | 31,690.2 | 1,235,916 |
| ieee39 | 128 | 128 | 4992 | 3 | bf16 | 948.87 | 945.04 | 989.80 | 1,005.6 | 134,898.0 | 5,261,022 |
| ieee118 | 1 | 1 | 118 | 3 | bf16 | 963.04 | 934.30 | 1,004.7 | 1,096.2 | 1,038.4 | 122,528.9 |
| ieee118 | 8 | 8 | 944 | 3 | bf16 | 1,110.7 | 1,108.1 | 1,154.8 | 1,216.7 | 7,202.9 | 849,944.5 |
| ieee118 | 32 | 32 | 3776 | 3 | bf16 | 1,137.4 | 1,141.2 | 1,228.4 | 1,290.8 | 28,134.7 | 3,319,890 |
| ieee118 | 128 | 128 | 15104 | 3 | bf16 | 1,164.9 | 1,162.4 | 1,197.9 | 1,236.3 | 109,879.8 | 12,965,819 |
| uk | 1 | 1 | 29 | 3 | bf16 | 895.00 | 877.60 | 916.87 | 941.04 | 1,117.3 | 32,402.3 |
| uk | 8 | 8 | 232 | 3 | bf16 | 928.16 | 923.66 | 971.66 | 1,015.9 | 8,619.2 | 249,957.0 |
| uk | 32 | 32 | 928 | 3 | bf16 | 932.75 | 927.23 | 990.26 | 1,002.5 | 34,307.0 | 994,904.1 |
| uk | 128 | 128 | 3712 | 3 | bf16 | 906.05 | 901.52 | 944.82 | 963.88 | 141,272.0 | 4,096,888 |

## 9. Forward + first-order backward (independent table)

`x.requires_grad_(True); out = global_max_pool(x, batch); out.sum().backward()`,
with `x.grad = None` cleared every iteration to avoid accumulation. The frozen
implementation routes FP32 through the Stage-4 tie-gradient `Function` and
FP16/BF16 through the Stage-5 dtype path. Not mixed with the forward table.

| dataset | batch | dtype | mean_us | p50_us | p95_us | p99_us | grad finite | nonzero grads | status |
|---|---|---|---|---|---|---|---|---|---|
| ieee24 | 1 | bf16 | 2,786.2 | 2,985.7 | 3,316.8 | 3,438.5 | True | 6 | ok |
| ieee24 | 1 | fp16 | 2,881.1 | 2,955.6 | 3,344.2 | 5,725.6 | True | 6 | ok |
| ieee24 | 1 | fp32 | 2,627.3 | 2,710.7 | 3,052.1 | 3,214.9 | True | 3 | ok |
| ieee24 | 8 | bf16 | 2,804.1 | 2,916.2 | 3,267.9 | 3,386.8 | True | 48 | ok |
| ieee24 | 8 | fp16 | 2,814.8 | 2,946.8 | 3,256.3 | 3,410.7 | True | 48 | ok |
| ieee24 | 8 | fp32 | 2,582.3 | 2,691.5 | 2,984.8 | 3,184.8 | True | 24 | ok |
| ieee24 | 32 | bf16 | 2,804.9 | 2,916.5 | 3,267.9 | 3,449.2 | True | 192 | ok |
| ieee24 | 32 | fp16 | 2,078.1 | 2,065.5 | 2,181.2 | 2,280.3 | True | 192 | ok |
| ieee24 | 32 | fp32 | 2,603.1 | 2,718.2 | 2,960.8 | 3,044.0 | True | 96 | ok |
| ieee24 | 128 | bf16 | 2,823.2 | 2,971.5 | 3,222.4 | 3,393.2 | True | 642 | ok |
| ieee24 | 128 | fp16 | 2,853.6 | 2,979.7 | 3,301.8 | 3,545.2 | True | 642 | ok |
| ieee24 | 128 | fp32 | 2,591.0 | 2,697.1 | 2,997.4 | 3,090.8 | True | 384 | ok |
| ieee39 | 1 | bf16 | 1,979.3 | 1,970.0 | 2,062.9 | 2,137.7 | True | 3 | ok |
| ieee39 | 1 | fp16 | 2,671.7 | 2,766.3 | 3,214.8 | 3,391.4 | True | 3 | ok |
| ieee39 | 1 | fp32 | 2,471.1 | 2,638.8 | 2,851.9 | 2,918.5 | True | 3 | ok |
| ieee39 | 8 | bf16 | 2,769.2 | 2,926.2 | 3,295.5 | 3,399.4 | True | 24 | ok |
| ieee39 | 8 | fp16 | 2,449.5 | 2,218.7 | 3,058.0 | 3,281.5 | True | 24 | ok |
| ieee39 | 8 | fp32 | 1,932.5 | 1,863.9 | 2,541.9 | 2,885.3 | True | 24 | ok |
| ieee39 | 32 | bf16 | 2,253.9 | 2,068.5 | 3,114.9 | 3,516.0 | True | 96 | ok |
| ieee39 | 32 | fp16 | 2,871.7 | 3,022.2 | 3,371.6 | 3,500.6 | True | 96 | ok |
| ieee39 | 32 | fp32 | 2,579.2 | 2,692.3 | 2,929.1 | 3,530.4 | True | 96 | ok |
| ieee39 | 128 | bf16 | 2,110.3 | 2,105.2 | 2,163.5 | 2,244.3 | True | 384 | ok |
| ieee39 | 128 | fp16 | 2,119.1 | 2,107.1 | 2,201.9 | 2,276.8 | True | 384 | ok |
| ieee39 | 128 | fp32 | 1,915.7 | 1,903.8 | 1,975.2 | 2,022.9 | True | 384 | ok |
| ieee118 | 1 | bf16 | 1,961.3 | 1,956.9 | 2,018.1 | 2,043.7 | True | 7 | ok |
| ieee118 | 1 | fp16 | 2,016.8 | 2,007.4 | 2,070.3 | 2,200.7 | True | 7 | ok |
| ieee118 | 1 | fp32 | 1,932.8 | 1,928.9 | 2,058.3 | 2,230.6 | True | 3 | ok |
| ieee118 | 8 | bf16 | 2,007.0 | 2,012.1 | 2,068.7 | 2,130.0 | True | 56 | ok |
| ieee118 | 8 | fp16 | 2,217.0 | 2,025.0 | 3,125.2 | 3,360.6 | True | 56 | ok |
| ieee118 | 8 | fp32 | 1,864.3 | 1,864.8 | 1,916.6 | 1,951.4 | True | 24 | ok |
| ieee118 | 32 | bf16 | 2,087.1 | 2,026.4 | 2,549.5 | 2,935.6 | True | 224 | ok |
| ieee118 | 32 | fp16 | 2,502.2 | 2,554.3 | 3,071.7 | 3,168.5 | True | 224 | ok |
| ieee118 | 32 | fp32 | 1,884.1 | 1,862.6 | 1,955.9 | 2,112.4 | True | 96 | ok |
| ieee118 | 128 | bf16 | 2,612.4 | 2,703.5 | 3,111.3 | 3,881.2 | True | 896 | ok |
| ieee118 | 128 | fp16 | 2,635.6 | 2,725.5 | 3,087.6 | 3,168.1 | True | 896 | ok |
| ieee118 | 128 | fp32 | 2,288.1 | 2,037.5 | 2,864.6 | 2,905.4 | True | 384 | ok |
| uk | 1 | bf16 | 2,340.9 | 2,366.8 | 2,928.5 | 3,046.5 | True | 14 | ok |
| uk | 1 | fp16 | 2,570.1 | 2,699.9 | 2,997.7 | 3,233.7 | True | 13 | ok |
| uk | 1 | fp32 | 2,547.8 | 2,499.5 | 3,350.6 | 3,540.5 | True | 3 | ok |
| uk | 8 | bf16 | 2,623.3 | 2,745.1 | 3,049.5 | 3,375.3 | True | 112 | ok |
| uk | 8 | fp16 | 2,387.2 | 2,440.8 | 2,975.4 | 3,101.0 | True | 104 | ok |
| uk | 8 | fp32 | 2,150.4 | 2,111.6 | 2,672.8 | 2,721.9 | True | 24 | ok |
| uk | 32 | bf16 | 2,341.1 | 2,069.4 | 3,109.4 | 3,309.8 | True | 448 | ok |
| uk | 32 | fp16 | 2,292.4 | 1,976.5 | 3,026.1 | 3,304.9 | True | 416 | ok |
| uk | 32 | fp32 | 2,248.5 | 2,359.5 | 2,733.5 | 2,808.4 | True | 96 | ok |
| uk | 128 | bf16 | 1,969.6 | 1,955.9 | 2,052.7 | 2,366.8 | True | 1792 | ok |
| uk | 128 | fp16 | 2,340.5 | 2,015.4 | 3,051.4 | 3,202.3 | True | 1664 | ok |
| uk | 128 | fp32 | 2,127.3 | 1,857.9 | 2,765.6 | 2,849.0 | True | 384 | ok |

## 10. Original PyG fallback comparison

The upstream implementation is captured *before* `pyg_ascend_compat.enable()` and
called with the identical tensor/batch. torch_npu reports:

```
CAUTION: The operator 'aten::scatter_reduce.two_out' is not currently supported
on the NPU backend and will fall back to run on the CPU.
```

so the baseline executes on the **host CPU** with device copies, not on the
Ascend vector core. `speedup = original_mean_us / compat_mean_us`.

### 10.1 FP32

| dataset | batch | compat_mean_us | original_pyg_mean_us | speedup | baseline execution location |
|---|---|---|---|---|---|
| ieee24 | 1 | 914.24 | 779.20 | 0.85x | host CPU (torch_npu fallback) |
| ieee24 | 8 | 889.07 | 801.80 | 0.90x | host CPU (torch_npu fallback) |
| ieee24 | 32 | 894.23 | 801.61 | 0.90x | host CPU (torch_npu fallback) |
| ieee24 | 128 | 921.36 | 972.64 | 1.06x | host CPU (torch_npu fallback) |
| ieee39 | 1 | 971.30 | 773.16 | 0.80x | host CPU (torch_npu fallback) |
| ieee39 | 8 | 917.79 | 733.87 | 0.80x | host CPU (torch_npu fallback) |
| ieee39 | 32 | 877.62 | 837.41 | 0.95x | host CPU (torch_npu fallback) |
| ieee39 | 128 | 937.63 | 968.71 | 1.03x | host CPU (torch_npu fallback) |
| ieee118 | 1 | 856.06 | 744.00 | 0.87x | host CPU (torch_npu fallback) |
| ieee118 | 8 | 875.66 | 804.03 | 0.92x | host CPU (torch_npu fallback) |
| ieee118 | 32 | 979.09 | 932.74 | 0.95x | host CPU (torch_npu fallback) |
| ieee118 | 128 | 1,132.5 | 35,284.1 | 31.16x | host CPU (torch_npu fallback) |
| uk | 1 | 995.14 | 913.01 | 0.92x | host CPU (torch_npu fallback) |
| uk | 8 | 841.61 | 742.44 | 0.88x | host CPU (torch_npu fallback) |
| uk | 32 | 865.91 | 762.50 | 0.88x | host CPU (torch_npu fallback) |
| uk | 128 | 858.09 | 881.61 | 1.03x | host CPU (torch_npu fallback) |

### 10.2 FP16

| dataset | batch | compat_mean_us | original_pyg_mean_us | speedup | baseline execution location |
|---|---|---|---|---|---|
| ieee24 | 1 | 916.17 | 779.28 | 0.85x | host CPU (torch_npu fallback) |
| ieee24 | 8 | 946.50 | 795.34 | 0.84x | host CPU (torch_npu fallback) |
| ieee24 | 32 | 956.68 | 818.10 | 0.86x | host CPU (torch_npu fallback) |
| ieee24 | 128 | 945.58 | 913.60 | 0.97x | host CPU (torch_npu fallback) |
| ieee39 | 1 | 943.61 | 763.35 | 0.81x | host CPU (torch_npu fallback) |
| ieee39 | 8 | 985.93 | 762.91 | 0.77x | host CPU (torch_npu fallback) |
| ieee39 | 32 | 1,026.8 | 866.99 | 0.84x | host CPU (torch_npu fallback) |
| ieee39 | 128 | 950.71 | 977.18 | 1.03x | host CPU (torch_npu fallback) |
| ieee118 | 1 | 902.02 | 795.79 | 0.88x | host CPU (torch_npu fallback) |
| ieee118 | 8 | 1,111.7 | 810.62 | 0.73x | host CPU (torch_npu fallback) |
| ieee118 | 32 | 1,108.9 | 921.48 | 0.83x | host CPU (torch_npu fallback) |
| ieee118 | 128 | 1,117.7 | 32,070.6 | 28.69x | host CPU (torch_npu fallback) |
| uk | 1 | 1,106.8 | 700.74 | 0.63x | host CPU (torch_npu fallback) |
| uk | 8 | 989.53 | 735.42 | 0.74x | host CPU (torch_npu fallback) |
| uk | 32 | 924.76 | 770.19 | 0.83x | host CPU (torch_npu fallback) |
| uk | 128 | 912.01 | 900.24 | 0.99x | host CPU (torch_npu fallback) |

### 10.3 BF16

| dataset | batch | compat_mean_us | original_pyg_mean_us | speedup | baseline execution location |
|---|---|---|---|---|---|
| ieee24 | 1 | 955.64 | 791.25 | 0.83x | host CPU (torch_npu fallback) |
| ieee24 | 8 | 956.27 | 809.94 | 0.85x | host CPU (torch_npu fallback) |
| ieee24 | 32 | 900.21 | 799.49 | 0.89x | host CPU (torch_npu fallback) |
| ieee24 | 128 | 937.33 | 903.67 | 0.96x | host CPU (torch_npu fallback) |
| ieee39 | 1 | 983.12 | 755.05 | 0.77x | host CPU (torch_npu fallback) |
| ieee39 | 8 | 940.10 | 757.58 | 0.81x | host CPU (torch_npu fallback) |
| ieee39 | 32 | 1,009.8 | 843.48 | 0.84x | host CPU (torch_npu fallback) |
| ieee39 | 128 | 948.87 | 952.15 | 1.00x | host CPU (torch_npu fallback) |
| ieee118 | 1 | 963.04 | 748.38 | 0.78x | host CPU (torch_npu fallback) |
| ieee118 | 8 | 1,110.7 | 811.33 | 0.73x | host CPU (torch_npu fallback) |
| ieee118 | 32 | 1,137.4 | 937.77 | 0.82x | host CPU (torch_npu fallback) |
| ieee118 | 128 | 1,164.9 | 41,330.7 | 35.48x | host CPU (torch_npu fallback) |
| uk | 1 | 895.00 | 719.52 | 0.80x | host CPU (torch_npu fallback) |
| uk | 8 | 928.16 | 744.85 | 0.80x | host CPU (torch_npu fallback) |
| uk | 32 | 932.75 | 797.12 | 0.85x | host CPU (torch_npu fallback) |
| uk | 128 | 906.05 | 874.54 | 0.97x | host CPU (torch_npu fallback) |

### 10.4 Baseline stability

Repeatability probe of the original-PyG host-CPU fallback (5 independent measurement windows of 10 warmup + 50 iterations each, same process):

| case | nodes | repeat means (us) | min_us | median_us | max_us | within-process spread | host loadavg 1/5/15m |
|---|---|---|---|---|---|---|---|
| ieee24 b128 | 3072 | 968, 946, 963, 953, 971 | 946.49 | 962.87 | 971.13 | 1.03x | 26.2/29.9/28.2 |
| ieee39 b128 | 4992 | 1,123, 1,045, 1,055, 1,079, 1,103 | 1,044.9 | 1,079.1 | 1,123.0 | 1.07x | 26.2/29.9/28.2 |
| ieee118 b32 | 3776 | 1,048, 979, 1,049, 1,024, 966 | 966.41 | 1,023.7 | 1,049.1 | 1.09x | 25.7/29.8/28.1 |
| ieee118 b64 | 7552 | 1,231, 1,230, 1,236, 1,226, 1,230 | 1,226.4 | 1,230.2 | 1,236.5 | 1.01x | 25.7/29.8/28.1 |
| ieee118 b128 | 15104 | 30,264, 29,578, 29,415, 30,385, 28,900 | 28,899.7 | 29,578.4 | 30,385.2 | 1.05x | 57.6/36.6/30.4 |
| uk b128 | 3712 | 973, 967, 972, 1,145, 982 | 966.80 | 973.49 | 1,144.7 | 1.18x | 57.6/36.6/30.4 |

Two behaviours are visible:

* most cells are repeatable within ~1.0-1.2x, so their ratios are meaningful;
* `ieee118` batch=128 (15,104 nodes) is a genuine threshold effect rather than
  sampling noise: the host-CPU fallback costs ~29-30 ms there versus 1.23 ms at
  batch=64 (7,552 nodes), i.e. ~24x more for 2x the nodes. Across separate
  processes the very same cell measured 4.7 ms, 27.2 ms and 35.3 ms, so its
  absolute value is **not** a stable number; only the qualitative conclusion
  (the upstream fallback degrades catastrophically on this cell while the
  Ascend path stays ~1.1-1.3 ms) is trustworthy.

## 11. Where the latency actually goes

Measured on real `ieee24` batch=128 (128 graphs, 3072 nodes, F=3), warmup=30, iters=200.

| measurement | device-event mean_us | p50_us | p95_us | host mean_us |
|---|---|---|---|---|
| A_sync_empty_queue | 33.07 | 32.73 | 35.76 | 33.07 |
| B_trivial_op_sync_per_iter | 101.62 | 96.76 | 123.16 | 101.62 |
| C_compat_global_max_pool | 869.95 | 865.32 | 908.56 | 925.35 |
| C2_compat_sync_per_iter | 795.56 | 785.95 | 846.00 | 795.56 |
| D_raw_bridge_kernel_only | 111.79 | 110.01 | 133.81 | 157.30 |
| E_original_pyg_fallback | 853.28 | 849.57 | 905.13 | 904.38 |

Per-step decomposition of the frozen adapter's per-call device work (each step timed with a full synchronise immediately before and after, so every row also carries the ~100 us launch+sync floor and the rows do not sum to the full-op latency):

| step | mean_us | p50_us | p95_us |
|---|---|---|---|
| 00_full_op_compat_call | 800.09 | 801.93 | 842.16 |
| 01_batch_minmax_to_host | 237.95 | 236.83 | 251.64 |
| 02_index_to_int32 | 76.86 | 73.71 | 92.68 |
| 03_pad_input_features | 92.67 | 90.00 | 109.67 |
| 04_torch_full_inf | 91.56 | 89.05 | 108.58 |
| 05_torch_empty_argmax | 53.22 | 52.95 | 57.16 |
| 06_scattermaxv1_kernel_only | 74.99 | 72.78 | 85.09 |
| 07_occupancy_zeros_plus_scatter | 287.11 | 284.54 | 305.22 |
| 08_masked_fill_empty_groups | 116.91 | 116.04 | 127.05 |
| 09_crop_contiguous | 103.98 | 103.18 | 109.87 |


## 12. Profiler evidence (msprof --ai-core=on)

| case | path | ScatterMaxV1 tasks | core type | kernel avg_us | AI_CPU tasks | aten::scatter_reduce occurrences | gate |
|---|---|---|---|---|---|---|---|
| ieee118_b128_fp32 | compat_ascend | 8 | AI_VECTOR_CORE | 119.88 | 0 | 0 | PASS |
| ieee24_b128_bf16 | compat_ascend | 8 | AI_VECTOR_CORE | 25.46 | 0 | 0 | PASS |
| ieee24_b128_fp16 | compat_ascend | 8 | AI_VECTOR_CORE | 25.62 | 0 | 0 | PASS |
| ieee24_b128_fp32 | compat_ascend | 8 | AI_VECTOR_CORE | 27.43 | 0 | 0 | PASS |
| ieee24_b128_fp32_origpyg | original_pyg | 0 | - |  | 0 | 0 | - |
| uk_b128_fp32 | compat_ascend | 8 | AI_VECTOR_CORE | 32.82 | 0 | 0 | PASS |

Compat dispatch counters (Python-level proof the Ascend path was entered):

* `ieee118_b128_fp32` (compat_ascend, ieee118 batch=128 dtype=fp32): compat counters total_calls=8 ascend_calls=8 original_calls=0 dtype16_forward_calls=0 non_fp32_passthrough=0; device kernels=['ScatterMaxV1_7d55161965c898907fdb3028d01c7c76_0']
* `ieee24_b128_bf16` (compat_ascend, ieee24 batch=128 dtype=bf16): compat counters total_calls=8 ascend_calls=8 original_calls=0 dtype16_forward_calls=8 non_fp32_passthrough=0; device kernels=['ScatterMaxV1_7d55161965c898907fdb3028d01c7c76_0']
* `ieee24_b128_fp16` (compat_ascend, ieee24 batch=128 dtype=fp16): compat counters total_calls=8 ascend_calls=8 original_calls=0 dtype16_forward_calls=8 non_fp32_passthrough=0; device kernels=['ScatterMaxV1_7d55161965c898907fdb3028d01c7c76_0']
* `ieee24_b128_fp32` (compat_ascend, ieee24 batch=128 dtype=fp32): compat counters total_calls=8 ascend_calls=8 original_calls=0 dtype16_forward_calls=0 non_fp32_passthrough=0; device kernels=['ScatterMaxV1_7d55161965c898907fdb3028d01c7c76_0']
* `ieee24_b128_fp32_origpyg` (original_pyg, ieee24 batch=128 dtype=fp32): compat counters total_calls=0 ascend_calls=0 original_calls=0 dtype16_forward_calls=0 non_fp32_passthrough=0; device kernels=[]
* `uk_b128_fp32` (compat_ascend, uk batch=128 dtype=fp32): compat counters total_calls=8 ascend_calls=8 original_calls=0 dtype16_forward_calls=0 non_fp32_passthrough=0; device kernels=['ScatterMaxV1_7d55161965c898907fdb3028d01c7c76_0']

## 13. Host fallback / AI_CPU / scatter_reduce audit

For every `compat_ascend` profiled case the raw msprof log contained **none** of
`npu_cpu_fallback`, `fall back to run on the CPU`, `507035`, `507011`,
`out of range`, `vector core exception`, `aicore exception`, `AIV exception`;
`op_summary_*.csv` contains no task whose `Task Type` mentions CPU; and
`api_statistic_*.csv` contains zero `aten::scatter_reduce` entries. The installed
delivery OPP's kernel hash
`ScatterMaxV1_7d55161965c898907fdb3028d01c7c76_0` is the one that executes.

## 14. Correctness sanity

| dataset | batch | dtype | out shape | shape ok | NaN | Inf | max abs diff vs CPU oracle | max rel diff | rtol/atol | tolerance mismatches | oracle ok |
|---|---|---|---|---|---|---|---|---|---|---|---|
| ieee24 | 1 | fp32 | (1, 3) | True | 0 | 0 | 0.000000 | 0.000000 | 1e-05/1e-06 | 0 | True |
| ieee24 | 1 | fp16 | (1, 3) | True | 0 | 0 | 0.000183 | 0.000282 | 0.002/0.01 | 0 | True |
| ieee24 | 1 | bf16 | (1, 3) | True | 0 | 0 | 0.001953 | 0.002985 | 0.02/0.1 | 0 | True |
| ieee24 | 8 | fp32 | (8, 3) | True | 0 | 0 | 0.000000 | 0.000000 | 1e-05/1e-06 | 0 | True |
| ieee24 | 8 | fp16 | (8, 3) | True | 0 | 0 | 0.000183 | 0.000282 | 0.002/0.01 | 0 | True |
| ieee24 | 8 | bf16 | (8, 3) | True | 0 | 0 | 0.001953 | 0.002985 | 0.02/0.1 | 0 | True |
| ieee24 | 32 | fp32 | (32, 3) | True | 0 | 0 | 0.000000 | 0.000000 | 1e-05/1e-06 | 0 | True |
| ieee24 | 32 | fp16 | (32, 3) | True | 0 | 0 | 0.000183 | 0.000282 | 0.002/0.01 | 0 | True |
| ieee24 | 32 | bf16 | (32, 3) | True | 0 | 0 | 0.001953 | 0.002985 | 0.02/0.1 | 0 | True |
| ieee24 | 128 | fp32 | (128, 3) | True | 0 | 0 | 0.000000 | 0.000000 | 1e-05/1e-06 | 0 | True |
| ieee24 | 128 | fp16 | (128, 3) | True | 0 | 0 | 0.000225 | 0.000310 | 0.002/0.01 | 0 | True |
| ieee24 | 128 | bf16 | (128, 3) | True | 0 | 0 | 0.001953 | 0.002985 | 0.02/0.1 | 0 | True |
| ieee39 | 1 | fp32 | (1, 3) | True | 0 | 0 | 0.000000 | 0.000000 | 1e-05/1e-06 | 0 | True |
| ieee39 | 1 | fp16 | (1, 3) | True | 0 | 0 | 0.000082 | 0.000225 | 0.002/0.01 | 0 | True |
| ieee39 | 1 | bf16 | (1, 3) | True | 0 | 0 | 0.001047 | 0.001593 | 0.02/0.1 | 0 | True |
| ieee39 | 8 | fp32 | (8, 3) | True | 0 | 0 | 0.000000 | 0.000000 | 1e-05/1e-06 | 0 | True |
| ieee39 | 8 | fp16 | (8, 3) | True | 0 | 0 | 0.000082 | 0.000225 | 0.002/0.01 | 0 | True |
| ieee39 | 8 | bf16 | (8, 3) | True | 0 | 0 | 0.001047 | 0.001593 | 0.02/0.1 | 0 | True |
| ieee39 | 32 | fp32 | (32, 3) | True | 0 | 0 | 0.000000 | 0.000000 | 1e-05/1e-06 | 0 | True |
| ieee39 | 32 | fp16 | (32, 3) | True | 0 | 0 | 0.000082 | 0.000225 | 0.002/0.01 | 0 | True |
| ieee39 | 32 | bf16 | (32, 3) | True | 0 | 0 | 0.001047 | 0.001593 | 0.02/0.1 | 0 | True |
| ieee39 | 128 | fp32 | (128, 3) | True | 0 | 0 | 0.000000 | 0.000000 | 1e-05/1e-06 | 0 | True |
| ieee39 | 128 | fp16 | (128, 3) | True | 0 | 0 | 0.000235 | 0.000406 | 0.002/0.01 | 0 | True |
| ieee39 | 128 | bf16 | (128, 3) | True | 0 | 0 | 0.001328 | 0.001630 | 0.02/0.1 | 0 | True |
| ieee118 | 1 | fp32 | (1, 3) | True | 0 | 0 | 0.000000 | 0.000000 | 1e-05/1e-06 | 0 | True |
| ieee118 | 1 | fp16 | (1, 3) | True | 0 | 0 | 0.000167 | 0.000221 | 0.002/0.01 | 0 | True |
| ieee118 | 1 | bf16 | (1, 3) | True | 0 | 0 | 0.000810 | 0.001069 | 0.02/0.1 | 0 | True |
| ieee118 | 8 | fp32 | (8, 3) | True | 0 | 0 | 0.000000 | 0.000000 | 1e-05/1e-06 | 0 | True |
| ieee118 | 8 | fp16 | (8, 3) | True | 0 | 0 | 0.000167 | 0.000221 | 0.002/0.01 | 0 | True |
| ieee118 | 8 | bf16 | (8, 3) | True | 0 | 0 | 0.000810 | 0.001069 | 0.02/0.1 | 0 | True |
| ieee118 | 32 | fp32 | (32, 3) | True | 0 | 0 | 0.000000 | 0.000000 | 1e-05/1e-06 | 0 | True |
| ieee118 | 32 | fp16 | (32, 3) | True | 0 | 0 | 0.000167 | 0.000221 | 0.002/0.01 | 0 | True |
| ieee118 | 32 | bf16 | (32, 3) | True | 0 | 0 | 0.000810 | 0.001069 | 0.02/0.1 | 0 | True |
| ieee118 | 128 | fp32 | (128, 3) | True | 0 | 0 | 0.000000 | 0.000000 | 1e-05/1e-06 | 0 | True |
| ieee118 | 128 | fp16 | (128, 3) | True | 0 | 0 | 0.000167 | 0.000221 | 0.002/0.01 | 0 | True |
| ieee118 | 128 | bf16 | (128, 3) | True | 0 | 0 | 0.000810 | 0.001069 | 0.02/0.1 | 0 | True |
| uk | 1 | fp32 | (1, 3) | True | 0 | 0 | 0.000000 | 0.000000 | 1e-05/1e-06 | 0 | True |
| uk | 1 | fp16 | (1, 3) | True | 0 | 0 | 0.000067 | 0.000112 | 0.002/0.01 | 0 | True |
| uk | 1 | bf16 | (1, 3) | True | 0 | 0 | 0.001531 | 0.002103 | 0.02/0.1 | 0 | True |
| uk | 8 | fp32 | (8, 3) | True | 0 | 0 | 0.000000 | 0.000000 | 1e-05/1e-06 | 0 | True |
| uk | 8 | fp16 | (8, 3) | True | 0 | 0 | 0.000067 | 0.000112 | 0.002/0.01 | 0 | True |
| uk | 8 | bf16 | (8, 3) | True | 0 | 0 | 0.001531 | 0.002103 | 0.02/0.1 | 0 | True |
| uk | 32 | fp32 | (32, 3) | True | 0 | 0 | 0.000000 | 0.000000 | 1e-05/1e-06 | 0 | True |
| uk | 32 | fp16 | (32, 3) | True | 0 | 0 | 0.000067 | 0.000112 | 0.002/0.01 | 0 | True |
| uk | 32 | bf16 | (32, 3) | True | 0 | 0 | 0.001531 | 0.002103 | 0.02/0.1 | 0 | True |
| uk | 128 | fp32 | (128, 3) | True | 0 | 0 | 0.000000 | 0.000000 | 1e-05/1e-06 | 0 | True |
| uk | 128 | fp16 | (128, 3) | True | 0 | 0 | 0.000067 | 0.000112 | 0.002/0.01 | 0 | True |
| uk | 128 | bf16 | (128, 3) | True | 0 | 0 | 0.001531 | 0.002103 | 0.02/0.1 | 0 | True |

## 15. Benchmark artefacts

Scripts live in this package; see `powergraph_validation/README.md` for the
authoritative file list.

| script | purpose |
|---|---|
| `bench_env.sh` | runtime env: custom OPP, frozen adapter/autograd/stage5 paths, bridge, `ASCEND_RT_VISIBLE_DEVICES` |
| `pg_env.py` | compat bootstrap + sklearn/`utils.gen_utils` shims |
| `pg_dataset.py` | unmodified `PowerGrid` loader wrapper + `torch.load(weights_only=False)` context |
| `analysis/phase_a_audit.py` | raw `.mat` audit (graph/node/edge/dtype) |
| `analysis/phase_a_loader_check.py` | processed-dataset + PyG 2.8 loader verification |
| `bench_forward.py` | forward benchmark (compat + original PyG) |
| `bench_backward.py` | forward + first-order backward benchmark |
| `analysis/probe_overhead.py` | breakdown vs sync floor / trivial op / raw kernel |
| `analysis/probe_adapter_steps.py` | per-step cost of the frozen adapter's per-call device work |
| `analysis/probe_baseline_stability.py` | repeatability probe of the original-PyG host-CPU fallback |
| `analysis/smoke_compat.py` | 3-line import-order + dtype smoke test of the frozen compat path |
| `profile_app.py` | msprof application for one representative case |
| `parse_profile.py` | msprof PROF_* parser -> gate JSON |
| `run_profiles.sh` | profiler driver for the representative cases |
| `run_validation.sh` | end-to-end driver (syntax check, forward, fwd+bwd, profiler gate) |
| `fetch_powergraph_data.sh` / `extract_powergraph_data.py` | dataset download + extraction |
| `analysis/make_report.py` | renders this report |
| `analysis/make_summaries.py` | derives `performance_summary.csv` / `profiler_summary.csv` |

## 16. Raw paths

The **original validation environment** used the absolute paths recorded below.
The packaged scripts use repository-relative defaults instead; each root is
overridable (`POWERGRAPH_DATA_ROOT`, `POWERGRAPH_RESULTS_ROOT`,
`POWERGRAPH_PROFILE_ROOT`, `POWERGRAPH_UPSTREAM_DIR`, `GLOBAL_MAX_POOL_OPP`).

```
original validation env: /root/zyg/powergraph-global-max-pool-bench/   (container)
original validation env: /data/zyg/powergraph-global-max-pool-bench/   (host mirror)
package default data    : <package>/data
package default results : <package>/results
package default profile : <package>/evidence/profiler
raw data                : $POWERGRAPH_DATA_ROOT/<ds>/<ds>/raw/
processed data          : $POWERGRAPH_DATA_ROOT/<ds>/<ds>/processed_b/data.pt
forward CSV           : results/forward_phaseB_ieee24.csv, results/forward_phaseC.csv
per-iteration CSV     : results/forward_per_iter_*.csv
fwd+bwd CSV           : results/forward_backward_*.csv
loader check JSON     : results/phase_a_loader.json, results/phase_a_loader_rest.json
workload JSON         : results/forward_*_workload.json, results/phase_a_raw_audit.json
overhead JSON         : results/overhead_breakdown.json, results/adapter_step_breakdown.json
baseline stability    : results/baseline_stability.json
provenance            : results/frozen_provenance.txt
download checksums    : data_download/MD5SUMS.txt
logs                  : logs/*.log, logs/pip_freeze_before.txt, logs/pip_freeze_after.txt
profiler              : profiler/<case>/PROF_*/mindstudio_profiler_output/, results/profiler_summary.txt
```

## 17. Limitations

* `F = 3` for every PowerGraph graph-level dataset, and 24-118 nodes per graph;
  the workload is very small per batch, so fixed per-call cost dominates and the
  numbers must not be extrapolated to large-`F` or large-`N` cases.
* The original PyG baseline runs on the host CPU (torch_npu fallback). It is a
  *reference implementation* comparison, explicitly **not** an NPU-vs-NPU one.
* The NPU is shared with other tenants on this host; runs were pinned to NPU 0
  and `npu-smi` reported AICore 0% during the runs, but the machine is not a
  dedicated benchmark box.
* Percentiles come from 200 per-iteration samples; P99 therefore rests on two
  samples.
* `dataset_cascades.zip` v3 (README link) and v5 (current article version)
  contain byte-identical raw files; v3 was used.
* pandas/scipy/scikit-learn were deliberately not installed; the two unused
  imports are stubbed in-process.
* Operator rows whose `status` is not `ok`: 0 of 96.
  None.

## 18. Conclusion

On real PowerGraph batches the frozen Ascend path is **functionally correct and
fully device-resident**: every profiled case runs `ScatterMaxV1` on
`AI_VECTOR_CORE` (kernel hash `...7d55161965c898907fdb3028d01c7c76_0`), with
zero AI_CPU tasks, zero `aten::scatter_reduce` calls and no host-CPU fallback
marker, and all 96 dataset x batch x dtype forward cases plus all 48
forward+backward cases match the CPU PyG oracle within tolerance.

Performance-wise the picture is much less favourable, and the reason is
structural rather than kernel-related:

* the ScatterMaxV1 kernel itself takes **27-33 us** for ieee24/uk and **120 us**
  for ieee118 at batch=128;
* the full `global_max_pool` call takes **840-1,170 us** across all 96 forward
  cases, because the frozen adapter pays one mandatory device->host
  synchronisation plus ~9 auxiliary NPU operations on every call;
* consequently the measured latency is essentially independent of batch size
  for ieee24/ieee39/uk (fixed overhead dominates) and rises only modestly for
  ieee118 (856 us at batch=1 to 1,133 us at batch=128 as the kernel becomes
  visible);
* against the upstream PyG implementation — which torch_npu executes on the
  **host CPU** at ~700-1,120 us for all cells except one — the Ascend path is
  therefore **0.6x-1.06x**, i.e. no speed-up at PowerGraph's `F = 3`,
  24-118 nodes scale. The single exception is `ieee118` batch=128, where the
  CPU fallback degrades to ~29 ms and the Ascend path is ~26x faster, but that
  cell is unstable across processes and must not be generalised.

Bottom line for this dataset: the frozen operator is a correct, host-fallback-free
Ascend implementation, and it wins decisively as soon as the reduction becomes
large enough to matter; for PowerGraph's very small `F = 3` graphs the fixed
per-call overhead dominates and the CPU fallback is competitive.
