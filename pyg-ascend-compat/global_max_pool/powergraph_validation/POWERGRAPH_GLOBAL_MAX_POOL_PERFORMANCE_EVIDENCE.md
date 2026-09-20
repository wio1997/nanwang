# PowerGraph `global_max_pool` 性能证据

> **说明**：本文档是机器生成报告
> （`scripts/analysis/make_report.py` 的英文原版输出）的**中文版本**；
> 所有数据来自原始 CSV / JSON / Profiler 证据，数值未做任何改动。

针对**已冻结（frozen）**的 PyG Ascend `global_max_pool` 实现，在**真实 PowerGraph
电力图数据**上做的单算子独立性能测试。不做算子开发、不训练 GNN、不修改仓库。

## 1. 测试环境

| 项目 | 值 |
|---|---|
| 服务器 | `S900K3-47` |
| 容器 | `wio-pyg-cann851-pyg280`（容器 hostname `caef109ccb29`） |
| 平台 | `Linux-5.15.0-25-generic-aarch64-with-glibc2.35` |
| 加速卡 | 8 × Ascend 910B3，测试通过 `ASCEND_RT_VISIBLE_DEVICES=0` 固定到单卡 |
| python | 3.11.14 |
| torch | 2.9.0+cpu |
| torch_npu | 2.9.0 |
| PyG | 2.8.0.post1 |
| numpy | 2.4.6 |
| device | Ascend910B3 |
| CANN | 8.5.1 (`/usr/local/Ascend/cann-8.5.1`) |

本次性能测试对环境做的改动（前后完整 package 列表记录在
`logs/pip_freeze_before.txt` 与 `logs/pip_freeze_after.txt`）：

```
h5py==3.16.0      (new)
mat73==0.65       (new)
```

`torch 2.9.0+cpu`、`torch_npu 2.9.0`、`torch_geometric 2.8.0.post1` 与
`numpy 2.4.6` **均未改动**。PowerGraph 的 `requirements.txt` **没有安装**。
`sklearn.model_selection.train_test_split` 与 `utils.gen_utils` 这两个在
`PowerGrid` 代码路径中并未使用、但在模块 import 阶段被引用的符号，用进程内 stub 满足，
因此无需安装 scikit-learn / scipy / pandas；两个 stub 被真正调用时会立即报错。

## 2. 被测算子来源（frozen）

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

算子冻结 commit 与最终文档 HEAD 的唯一差异是
`pyg-ascend-compat/global_max_pool/README_DELIVERY.md`；operator、adapter、autograd
与 dtype 源码逐字节一致，实测 checkout 的工作区为 clean。

性能测试脚本始终保持以下 import 顺序：

```python
import pyg_ascend_compat
pyg_ascend_compat.enable()
from torch_geometric.nn import global_max_pool
```

## 3. PowerGraph 数据来源

| 项目 | 值 |
|---|---|
| 仓库 | `https://github.com/PowerGraph-Datasets/PowerGraph-Graph` |
| 使用的 commit（只读 clone） | `eb100a2fd836bb8b6bd2d0b799af9c615eac8cb6` |
| 使用的文件 | 恰好是 README 链接的对象：figshare article `22820534`，file id `46619158`（`dataset_cascades.zip`，v3） |
| 下载大小 / md5 | 61,628,977 / `70b677416d2f377ccfee9f51d8369867` |
| 解压后大小 | 2,958,249,040 bytes（2.75 GiB） |
| 另做事后核对 | file id `50083479`（v5 `dataset_cascades.zip`，md5 校验值 `d4d144b9e720a760e1e077a31f34802d`）—— 文件内容与大小一致，仅多一层顶层目录 |

在**原始验证环境**中，figshare.com 对所有路径都返回 HTTP 403（article 页面、API 与
downloader 均是，IPv4 与 IPv6 相同）。因此数据获取方式是：通过公共 HTTP proxy 取得
figshare 的 presigned S3 重定向，再从
`s3-eu-west-1.amazonaws.com/pfigshare-u-files/...` **直接下载 payload** —— payload
传输本身不经过 proxy。打包脚本使用的可移植下载流程见 `DATASET.md`
（`scripts/fetch_powergraph_data.sh`：优先官方 figshare 地址，并校验 checksum）。
数据集存放于 `POWERGRAPH_DATA_ROOT`（默认 `<package>/data`），不随 git 提交。

四个数据集全部可用，并且均由**未修改的** `PowerGrid` `InMemoryDataset` loader 在
PyG 2.8.0.post1 下处理。

## 4. 数据集可用性与 workload 统计

原始 `.mat` 审计（PyG 处理之前）：

| 数据集 | graph 数 | 每图节点数（raw） | 每图定义的支路数 | loader x shape | loader x dtype | edge_index 每图边数 min | edge_index 每图边数 max | edge_index 每图边数 mean | 每图被切除支路 min | 每图被切除支路 max |
|---|---|---|---|---|---|---|---|---|---|---|
| ieee24 | 21500 | 24 | 38 | [24, 3] | torch.float32 | 68 | 74 | 73.76 | 1 | 4 |
| ieee39 | 28000 | 39 | 46 | [39, 3] | torch.float32 | 86 | 90 | 89.64 | 1 | 3 |
| ieee118 | 122500 | 118 | 186 | [118, 3] | torch.float32 | 362 | 370 | 369.44 | 1 | 5 |
| uk | 64000 | 29 | 99 | [29, 3] | torch.float32 | 190 | 196 | 195.53 | 1 | 4 |

处理后的数据集 workload（精确值，取自 `dataset.slices`）：

| 数据集 | graph 数 | F | 每图节点数 min | 每图节点数 max | 每图节点数 mean | 每图边数 min | 每图边数 max | 每图边数 mean | 每图节点数是否恒定 | 不同边数的取值个数 |
|---|---|---|---|---|---|---|---|---|---|---|
| ieee24 | 21500 | 3 | 24 | 24 | 24.00 | 68 | 74 | 73.76 | True | 4 |
| ieee39 | 28000 | 3 | 39 | 39 | 39.00 | 86 | 90 | 89.64 | True | 3 |
| ieee118 | 122500 | 3 | 118 | 118 | 118.00 | 362 | 370 | 369.44 | True | 5 |
| uk | 64000 | 3 | 29 | 29 | 29.00 | 190 | 196 | 195.53 | True | 4 |

每个数据集的每图节点数是**固定的**（24 / 39 / 118 / 29）；变化的只是每图**被切除支路
（tripped branches）** 的数量，它使 `edge_index` / `edge_attr` 的规模分别落在
68–74、86–90、362–370、190–196 条有向边之间。节点特征恒为 `x: float32 [N, 3]`
（net active power、net apparent power、voltage magnitude），因此所有数据集都是
`F = 3` —— feature 维度远小于节点维度，这一点对理解下面的数据很关键。

## 5. 性能测试方法

* 输入是**真实 PyG DataLoader batch**（`torch_geometric.loader.DataLoader`，
  `shuffle=False`，取第一个 batch），只把 `batch.x` 与
  `batch.batch` 搬到 NPU。不做 GNN 卷积、不做 Linear、不做优化器、不做训练。
* `out = global_max_pool(x, batch_index)` inside `torch.no_grad()`.
* warmup（预热）= 30 次，measurement iterations（测量迭代）= 200 次。
* warmup 之后以及最后一次 launch 之后都会调用 `torch.npu.synchronize()`；
  **不会在未同步的 device 工作周围取时间**。
* 主指标：每次迭代一对 `torch.npu.Event(enable_timing=True)`，得到 device 侧单次
  执行时长；mean / P50 / P95 / P99 由这些逐次迭代样本计算
  （保存在 `*_per_iter_*.csv`）。
* 次指标：完全同步的循环（`sync_*` 列），每次调用前后都调用
  `torch.npu.synchronize()`，因此还包含 host 侧 launch 开销。
* `host_total_us` 是批量计时器的交叉校验（synchronise、t0、N 次 launch、
  synchronise、t1）再除以 N。

### 5.1 重要测量说明

冻结的 Stage-2 adapter 在**每次调用中都会做一次强制的 device→host 同步**
（用于索引范围校验的 `torch.stack((batch.min(), batch.max())).cpu().tolist()`，以及
`size` 推导），并额外发起约 9 个辅助 NPU 操作（`to(int32)`、`F.pad`、`torch.full`、
`torch.empty`、ScatterMaxV1 launch、`zeros`+`scatter_` 占位、`masked_fill_`、
裁剪用的 `contiguous`）。因此 back-to-back launch 循环实际上无法把调用排队：
量到的**就是冻结算子的真实单次调用延迟**，而不是流水线吞吐。两种循环都报告出来，
以便读者看清这一点。

## 6. FP32 性能

| 数据集 | batch | graphs | nodes | F | dtype | mean_us | p50_us | p95_us | p99_us | graphs_per_sec | nodes_per_sec |
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

## 7. FP16 性能

FP16 **不是** native kernel：它是 device 上的 cast 链
`fp16 -> fp32 -> ScatterMaxV1 -> fp16`。

| 数据集 | batch | graphs | nodes | F | dtype | mean_us | p50_us | p95_us | p99_us | graphs_per_sec | nodes_per_sec |
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

## 8. BF16 性能

BF16 同样是 `bf16 -> fp32 -> ScatterMaxV1 -> bf16` 的 cast 链。

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

## 9. Forward + 一阶 backward（独立表格）

每轮执行 `x.requires_grad_(True); out = global_max_pool(x, batch);
out.sum().backward()`，并在每次迭代开始时用 `x.grad = None` 清零以避免梯度累积。
冻结实现中 FP32 走 Stage-4 的 tie-gradient `Function`，FP16/BF16 走 Stage-5 的 dtype
路径。本表与 forward 表分开统计，不混合。

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

## 10. 与原始 PyG fallback（主机侧回退）路径的对照

上游实现是在 `pyg_ascend_compat.enable()` **之前**捕获的，用完全相同的 tensor / batch
调用。torch_npu 会打印：

```
CAUTION: The operator 'aten::scatter_reduce.two_out' is not currently supported
on the NPU backend and will fall back to run on the CPU.
```

也就是说，对照组实际执行在 **host CPU** 上（伴随 device 拷贝），而不是在 Ascend
vector core 上。`speedup = original_mean_us / compat_mean_us`。

### 10.1 FP32

| 数据集 | batch | compat_mean_us | original_pyg_mean_us | speedup | 对照路径实际执行位置 |
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

| 数据集 | batch | compat_mean_us | original_pyg_mean_us | speedup | 对照路径实际执行位置 |
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

| 数据集 | batch | compat_mean_us | original_pyg_mean_us | speedup | 对照路径实际执行位置 |
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

### 10.4 对照路径的稳定性

对原始 PyG host CPU fallback 路径做重复性探测（同一进程内 5 个独立测量窗口，
每个窗口 10 次 warmup + 50 次迭代）：

| case | nodes | 各次重复的 mean (us) | min_us | median_us | max_us | 进程内波动倍数 | host loadavg 1/5/15m |
|---|---|---|---|---|---|---|---|
| ieee24 b128 | 3072 | 968, 946, 963, 953, 971 | 946.49 | 962.87 | 971.13 | 1.03x | 26.2/29.9/28.2 |
| ieee39 b128 | 4992 | 1,123, 1,045, 1,055, 1,079, 1,103 | 1,044.9 | 1,079.1 | 1,123.0 | 1.07x | 26.2/29.9/28.2 |
| ieee118 b32 | 3776 | 1,048, 979, 1,049, 1,024, 966 | 966.41 | 1,023.7 | 1,049.1 | 1.09x | 25.7/29.8/28.1 |
| ieee118 b64 | 7552 | 1,231, 1,230, 1,236, 1,226, 1,230 | 1,226.4 | 1,230.2 | 1,236.5 | 1.01x | 25.7/29.8/28.1 |
| ieee118 b128 | 15104 | 30,264, 29,578, 29,415, 30,385, 28,900 | 28,899.7 | 29,578.4 | 30,385.2 | 1.05x | 57.6/36.6/30.4 |
| uk b128 | 3712 | 973, 967, 972, 1,145, 982 | 966.80 | 973.49 | 1,144.7 | 1.18x | 57.6/36.6/30.4 |

可以看到两种现象：

* 大多数组合在 ~1.0–1.2x 之内可复现，因此它们的比值是有意义的；
* `ieee118` batch=128（15,104 nodes）是真实的**阈值效应**，而不是采样噪声：
  host CPU fallback 在该点约 29–30 ms，而 batch=64（7,552 nodes）只有 1.23 ms，
  即节点数增加 2 倍、耗时增加约 24 倍。跨进程来看，同一个组合分别测到
  4.7 ms、27.2 ms、35.3 ms，因此它的绝对值**不是一个稳定数字**；可以采信的只有定性
  结论：上游 fallback 在这个组合上出现严重退化，而 Ascend 路径稳定在 ~1.1–1.3 ms。

## 11. latency 的实际构成

在真实 `ieee24` batch=128（128 graphs、3072 nodes、F=3）上测得，warmup=30、iters=200。

| 测量项 | device event mean_us | p50_us | p95_us | host mean_us |
|---|---|---|---|---|
| A_sync_empty_queue | 33.07 | 32.73 | 35.76 | 33.07 |
| B_trivial_op_sync_per_iter | 101.62 | 96.76 | 123.16 | 101.62 |
| C_compat_global_max_pool | 869.95 | 865.32 | 908.56 | 925.35 |
| C2_compat_sync_per_iter | 795.56 | 785.95 | 846.00 | 795.56 |
| D_raw_bridge_kernel_only | 111.79 | 110.01 | 133.81 | 157.30 |
| E_original_pyg_fallback | 853.28 | 849.57 | 905.13 | 904.38 |

冻结 adapter 单次调用中各步骤的分解（每一步都在前后做完整 synchronise，因此每一行都
还包含约 100 us 的 launch+sync 下限，各行之和并不等于完整算子的 latency）：

| 步骤 | mean_us | p50_us | p95_us |
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


## 12. Profiler 验证证据（msprof --ai-core=on）

| case | path | ScatterMaxV1 任务数 | 执行核心类型 | kernel 平均耗时 us | AI_CPU 任务数 | aten::scatter_reduce 出现次数 | gate |
|---|---|---|---|---|---|---|---|
| ieee118_b128_fp32 | compat_ascend | 8 | AI_VECTOR_CORE | 119.88 | 0 | 0 | PASS |
| ieee24_b128_bf16 | compat_ascend | 8 | AI_VECTOR_CORE | 25.46 | 0 | 0 | PASS |
| ieee24_b128_fp16 | compat_ascend | 8 | AI_VECTOR_CORE | 25.62 | 0 | 0 | PASS |
| ieee24_b128_fp32 | compat_ascend | 8 | AI_VECTOR_CORE | 27.43 | 0 | 0 | PASS |
| ieee24_b128_fp32_origpyg | original_pyg | 0 | - |  | 0 | 0 | - |
| uk_b128_fp32 | compat_ascend | 8 | AI_VECTOR_CORE | 32.82 | 0 | 0 | PASS |

compat dispatch counter（Python 层证据，证明进入的是 Ascend 路径）：

* `ieee118_b128_fp32` (compat_ascend, ieee118 batch=128 dtype=fp32): compat counters total_calls=8 ascend_calls=8 original_calls=0 dtype16_forward_calls=0 non_fp32_passthrough=0; device kernels=['ScatterMaxV1_7d55161965c898907fdb3028d01c7c76_0']
* `ieee24_b128_bf16` (compat_ascend, ieee24 batch=128 dtype=bf16): compat counters total_calls=8 ascend_calls=8 original_calls=0 dtype16_forward_calls=8 non_fp32_passthrough=0; device kernels=['ScatterMaxV1_7d55161965c898907fdb3028d01c7c76_0']
* `ieee24_b128_fp16` (compat_ascend, ieee24 batch=128 dtype=fp16): compat counters total_calls=8 ascend_calls=8 original_calls=0 dtype16_forward_calls=8 non_fp32_passthrough=0; device kernels=['ScatterMaxV1_7d55161965c898907fdb3028d01c7c76_0']
* `ieee24_b128_fp32` (compat_ascend, ieee24 batch=128 dtype=fp32): compat counters total_calls=8 ascend_calls=8 original_calls=0 dtype16_forward_calls=0 non_fp32_passthrough=0; device kernels=['ScatterMaxV1_7d55161965c898907fdb3028d01c7c76_0']
* `ieee24_b128_fp32_origpyg` (original_pyg, ieee24 batch=128 dtype=fp32): compat counters total_calls=0 ascend_calls=0 original_calls=0 dtype16_forward_calls=0 non_fp32_passthrough=0; device kernels=[]
* `uk_b128_fp32` (compat_ascend, uk batch=128 dtype=fp32): compat counters total_calls=8 ascend_calls=8 original_calls=0 dtype16_forward_calls=0 non_fp32_passthrough=0; device kernels=['ScatterMaxV1_7d55161965c898907fdb3028d01c7c76_0']

## 13. Host fallback / AI_CPU / scatter_reduce 审计

对每一个 `compat_ascend` profiled case，原始 msprof log 中**都不包含**
`npu_cpu_fallback`、`fall back to run on the CPU`、`507035`、`507011`、
`out of range`、`vector core exception`、`aicore exception`、`AIV exception` 中的任何
一项；`op_summary_*.csv` 中没有任何 `Task Type` 含 CPU 的任务；
`api_statistic_*.csv` 中 `aten::scatter_reduce` 条目为 0。实际执行的是已安装正式
OPP 的 kernel hash `ScatterMaxV1_7d55161965c898907fdb3028d01c7c76_0`。

## 14. 正确性检查

| 数据集 | batch | dtype | 输出 shape | shape 正确 | NaN | Inf | 与 CPU oracle 的最大绝对偏差 | 最大相对偏差 | rtol/atol | 容差外元素数 | oracle 通过 |
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

## 15. 测试产物

脚本随本包一同提供，权威清单见 `powergraph_validation/README.md`。

| 脚本 | 用途 |
|---|---|
| `bench_env.sh` | 运行环境：custom OPP、frozen adapter/autograd/stage5 路径、bridge、`ASCEND_RT_VISIBLE_DEVICES` |
| `pg_env.py` | compat bootstrap + sklearn / `utils.gen_utils` stub |
| `pg_dataset.py` | 未修改的 `PowerGrid` loader 封装 + `torch.load(weights_only=False)` 兼容 context |
| `analysis/phase_a_audit.py` | 原始 `.mat` 审计（graph / node / edge / dtype） |
| `analysis/phase_a_loader_check.py` | processed 数据集 + PyG 2.8 loader 校验 |
| `bench_forward.py` | forward 性能测试（compat + 原始 PyG） |
| `bench_backward.py` | forward + 一阶 backward 性能测试 |
| `analysis/probe_overhead.py` | latency 分解：同步下限 / trivial op / 裸 kernel |
| `analysis/probe_adapter_steps.py` | 冻结 adapter 单次调用各步骤的开销 |
| `analysis/probe_baseline_stability.py` | 原始 PyG host CPU fallback 路径的重复性探测 |
| `analysis/smoke_compat.py` | 冻结算子 compat 路径的 3 行 import 顺序 + dtype 冒烟测试 |
| `profile_app.py` | 单个代表 case 的 msprof 应用 |
| `parse_profile.py` | msprof `PROF_*` 解析 → gate JSON |
| `run_profiles.sh` | 代表 case 的 Profiler 驱动 |
| `run_validation.sh` | 端到端驱动（syntax 检查、forward、fwd+bwd、Profiler gate） |
| `fetch_powergraph_data.sh` / `extract_powergraph_data.py` | 数据集下载 + 解压 |
| `analysis/make_report.py` | 渲染本报告 |
| `analysis/make_summaries.py` | 生成 `performance_summary.csv` / `profiler_summary.csv` |

## 16. 原始路径

**原始验证环境**使用下面记录的绝对路径。打包后的脚本改用仓库内相对路径默认值，
每个根目录都可以覆盖（`POWERGRAPH_DATA_ROOT`、`POWERGRAPH_RESULTS_ROOT`、
`POWERGRAPH_PROFILE_ROOT`、`POWERGRAPH_UPSTREAM_DIR`、`GLOBAL_MAX_POOL_OPP`）。

```
原始验证环境（容器） : /root/zyg/powergraph-global-max-pool-bench/
原始验证环境（host）  : /data/zyg/powergraph-global-max-pool-bench/
包内默认 data 根       : <package>/data
包内默认 results 根    : <package>/results
包内默认 Profiler 根   : <package>/profiler_runs
原始数据               : $POWERGRAPH_DATA_ROOT/<ds>/<ds>/raw/
processed 数据         : $POWERGRAPH_DATA_ROOT/<ds>/<ds>/processed_b/data.pt
forward CSV            : results/forward_phaseB_ieee24.csv, results/forward_phaseC.csv
逐次迭代 CSV           : results/forward_per_iter_*.csv
forward+backward CSV   : results/forward_backward_*.csv
loader 校验 JSON       : results/phase_a_loader.json, results/phase_a_loader_rest.json
workload JSON          : results/forward_*_workload.json, results/phase_a_raw_audit.json
开销分解 JSON          : results/overhead_breakdown.json, results/adapter_step_breakdown.json
对照路径稳定性         : results/baseline_stability.json
来源与 sha256          : results/frozen_provenance.txt
下载校验和             : data_download/MD5SUMS.txt
日志                   : logs/*.log, logs/pip_freeze_before.txt, logs/pip_freeze_after.txt
Profiler               : profiler/<case>/PROF_*/mindstudio_profiler_output/, results/profiler_summary.txt
```

## 17. 已知限制

* 所有 PowerGraph graph-level 数据集的 `F = 3`，每图节点数 24–118；单 batch 的
  workload 非常小，因此固定单次调用开销占主导，这些数字**不能**外推到大 `F` 或大 `N`
  的场景。
* 原始 PyG 对照路径运行在 host CPU（torch_npu fallback）上，它是与*参考实现*的对照，
  **明确不是**同设备（NPU vs NPU）的比较。
* 本机 NPU 与其他租户共享；测试固定使用 NPU 0，运行期间 `npu-smi` 显示 AICore 0%，
  但该机器并非专用性能测试机。
* 百分位数来自 200 个逐次迭代样本，因此 P99 实际只依赖约 2 个样本。
* `dataset_cascades.zip` v3（README 链接版本）与 v5（当前 article 版本）包含逐字节
  一致的 raw 文件；本次使用 v3。
* pandas / scipy / scikit-learn 有意未安装；两个未被调用的 import 用进程内 stub 满足。
* `status` 不为 `ok` 的算子记录：96 条中 0 条（无）。

## 18. 测试结论

在真实 PowerGraph batch 上，冻结的 Ascend 路径**功能正确且完全 device-resident**：
每个 profiled case 的 `ScatterMaxV1` 都运行在 `AI_VECTOR_CORE`
（kernel hash `...7d55161965c898907fdb3028d01c7c76_0`），AI_CPU 任务为 0、
`aten::scatter_reduce` 调用为 0、无 host CPU fallback 标记；96 个
dataset × batch × dtype 的 forward case 与 48 个 forward+backward case 全部在容差内
与 CPU PyG oracle 一致。

在性能方面，本数据集上的表现不那么有利，而且原因是结构性的，而非 kernel 本身：

* `ScatterMaxV1` kernel 本身在 ieee24 / uk 上约 **27–33 us**，在 ieee118 batch=128
  上约 **120 us**；
* 完整 `global_max_pool` 调用在全部 96 个 forward case 上为 **840–1,170 us**，
  因为冻结 adapter 每次调用都要付出一次强制 device→host 同步加约 9 个辅助 NPU 操作；
* 因此 ieee24 / ieee39 / uk 的实测 latency 基本与 batch 大小无关（固定开销占主导），
  ieee118 只是温和上升（batch=1 的 856 us 到 batch=128 的 1,133 us，kernel 开始显现）；
* 相对于上游 PyG 实现（torch_npu 将其执行在 **host CPU** 上，除一个组合外均为
  ~700–1,120 us），Ascend 路径为 **0.6x–1.06x**，即在 PowerGraph 的 `F = 3`、
  24–118 nodes 规模下没有加速。唯一例外是 `ieee118` batch=128：CPU fallback 退化到
  ~29 ms，Ascend 路径约快 26 倍，但该组合跨进程不稳定，不能推广。

对本数据集的总体判断：冻结算子是正确、无 host fallback 的 Ascend 实现；一旦 reduction
的规模足够大，它就会明显胜出；而在 PowerGraph 这种极小的 `F = 3` graph 上，固定单次
调用开销占主导，CPU fallback 具有竞争力。
