# PowerGraph 真实电力图 `global_max_pool` 验证

> **入库说明（仓库版本）。** 本文档是经人工评审通过的验证文档，随
> `pyg-ascend-compat/global_max_pool/powergraph_validation/` 一并入库。
> 所有实测数字、表格与证据与评审版本完全一致，仅调整了路径表述。
> 文中标注为 *original validation environment*（原始验证环境）的绝对路径
> （`/root/zyg/...`、`/data/zyg/...`）作为历史记录保留；所有可执行步骤已改为
> 仓库内相对路径，并可用环境变量覆盖（见 `README.md` §4）。

## 使用 PowerGraph 真实电力图数据，对 Ascend PyG `global_max_pool` 算子进行真实 workload 验证和性能测试

本文档是**测试交付文档**（test delivery / validation record），不是性能优化报告，
也不是算子开发记录。

| 项目 | 值 |
|---|---|
| 被测对象 | 已冻结的 PyG Ascend `global_max_pool` 实现 |
| 测试数据 | `PowerGraph-Datasets/PowerGraph-Graph` 真实电网 graph-level 数据 |
| 测试方式 | 真实 PyG `DataLoader` batch，单算子调用，不含 GNN / 训练 |
| 测试轮次 | forward + forward&backward + msprof Profiler gate |
| 本轮验收标准 | 真实数据可运行、真实 PyG batch 可运行、FP32/FP16/BF16 可运行、结果正确、真正跑在 NPU 上、benchmark 方法正确 |
| 本轮**不**作为验收标准 | 相对原始 PyG fallback 的 speedup |

**结论摘要**：本轮所有验收目标均已达成 —— 真实数据、真实 PyG batch、FP32/FP16/BF16、
forward 与 forward+backward 全部通过，结果与 CPU oracle 一致，且 Profiler 证明
`ScatterMaxV1` 实际运行在 `AI_VECTOR_CORE`，无 Host fallback（主机侧回退）、无 AI_CPU、
无 `aten::scatter_reduce`。性能数据作为 workload 画像保留，**不作为本轮 PASS/FAIL 判据**。

---

## 1. 为什么用 PowerGraph 测试？

选择 PowerGraph 的原因：

1. **它是真实数据，不是随机 tensor。** PowerGraph 是公开的电力系统级图神经网络
   benchmark 数据集（NeurIPS 2024 Datasets & Benchmarks），数据来自
   physics-based cascading-failure 仿真（Cascades 模型）产生的真实电网运行状态：
   每个 graph 是一次 outage 前的电网状态，节点是 bus（负荷 / 发电机），边是输电线路
   与变压器。
2. **它的 graph readout 支持 `max`，直接对应被测算子。** PowerGraph 的 graph-level
   readout 提供 `mean / sum / max`，其中 `max` 正是
   `torch_geometric.nn.global_max_pool`。因此这是该算子的**真实 upstream 使用场景**，
   而不是为 benchmark 构造出来的调用。
3. **它是 graph-level 数据集，天然产生多 graph batch。** 每个 dataset 有数万个 graph，
   PyG `DataLoader` 会把多个 graph 拼成一个 batch 并生成真实的 `batch.batch` 分组
   索引向量，这正是 `global_max_pool` 需要的输入形态（`x: [N, F]` +
   `batch_index: [N]`，group count = batch 内 graph 数）。
4. **覆盖了不同规模的电网拓扑。** IEEE24 / IEEE39 / IEEE118 / UK 四种电网，每图节点
   从 24 到 118，graph 数从 2.15 万到 12.25 万。

本轮**不做**的事情：不训练 GNN、不跑 conv / Linear / optimizer、不算
accuracy / AUC / R²、不复现论文指标。只把 PowerGraph 当作**真实图数据来源**，
对 `global_max_pool` 单算子做验证与性能测量。

---

## 2. 被测算子（frozen）

| 项目 | 值 |
|---|---|
| 仓库 | `https://github.com/wio1997/nanwang` |
| 分支 | `feat/global-max-pool-scattermax-zyg` |
| 算子实现冻结 SHA | `c15423e7b303d2b1597621c64585252472301537` |
| 交付 / 文档 HEAD | `d1dc616b809c32819f3f1fab9a8c43528f824f05` |

说明：

* `c15423e` 是**算子实现冻结版本**（compat package / adapter / autograd / dtype /
  交付 kernel 源码在该 commit 上冻结）。
* `d1dc616` 是之后的**仅文档提交**（docs-only delivery HEAD）。
* 实测 checkout 为 `d1dc616`，`git status --porcelain` 为**空**（工作区 clean）。
  `git diff c15423e..d1dc616` 仅涉及一个文件：
  `pyg-ascend-compat/global_max_pool/README_DELIVERY.md`（+1243 行，纯文档）。
  因此本轮实测使用的算子实现与 frozen SHA 完全一致。

被加载的 frozen source 及 sha256（完整清单见 `results/frozen_provenance.txt`）：

```text
417871482f1c7aa4fcf1f4cab16c3581958de5162d1499a1cc4db1220efbcd13  pyg-ascend-compat/global_max_pool/stage6/pyg_ascend_compat/__init__.py
60e037f3104dc8674c9f1a1a98d646f31d037f09764465e57036bdf9a1df0119  pyg-ascend-compat/global_max_pool/stage2/python/global_max_pool_ascend.py
6647ee038a30529a71f05745427300c0258eded0182d172ad8c5105dd4d7666c  pyg-ascend-compat/global_max_pool/stage4/python/global_max_pool_ascend_autograd.py
cac35c11c2e31ca9872ad70389dc1d85948b53cb1f0f51b7978684624299cf92  pyg-ascend-compat/global_max_pool/stage5/python/global_max_pool_ascend_dtype.py
773198c3b1c77e7d755b0ac3f4cb91f794aed6718de839c230d5e2cdb488a016  pyg-ascend-compat/global_max_pool/stage3e/delivery_source/op_kernel/scatter_max_v1.cpp
ab7fd9d2d18796e35a66df07dcaac82e625e07eed76341e217487acf4e83f654  pyg-ascend-compat/global_max_pool/stage3e/delivery_source/op_host/scatter_max_v1.cpp
```

---

## 3. 测试环境

| 项目 | 值 |
|---|---|
| 服务器 | `S900K3-47` |
| 加速卡 | 8 × Ascend 910B3（测试固定使用 `ASCEND_RT_VISIBLE_DEVICES=0`） |
| 操作系统 | Ubuntu 22.04 aarch64（`Linux 5.15.0-25-generic`） |
| 容器 | `wio-pyg-cann851-pyg280` |
| 镜像 | `local/wio-pyg-cann851:torch2.9-pyg2.8.0.post1` |
| CANN | 8.5.1（`/usr/local/Ascend/cann-8.5.1`） |
| Python | 3.11.14 |
| PyTorch | 2.9.0+cpu |
| torch_npu | 2.9.0 |
| PyG | 2.8.0.post1 |
| numpy | 2.4.6 |

正式 ScatterMax OPP（**original validation environment** 中的实际路径）：

```text
/root/zyg/build/scattermax_runtime_opp/vendors/customize
```

通过 `ASCEND_CUSTOM_OPP_PATH` 指向该目录，并由 `scripts/bench_env.sh` 统一设置
（同时设置 `PYG_ASCEND_ADAPTER_PATH` / `PYG_ASCEND_AUTOGRAD_PATH` /
`PYG_ASCEND_STAGE5_PATH` / `SCATTERMAXV1_BRIDGE`）。

仓库内脚本不再硬编码该路径：`scripts/bench_env.sh` 从自身位置推导 repo / package
根目录，OPP 与 bridge 可用 `GLOBAL_MAX_POOL_OPP` / `SCATTERMAXV1_BRIDGE` 覆盖，
上面是默认值（即 original validation environment 使用的构建产物位置）。
所有可覆盖变量见 `README.md` §4。

### 3.1 环境变更记录

为读取 PowerGraph 的 MATLAB v7.3 数据，额外安装了 2 个包：

```text
h5py==3.16.0
mat73==0.65
```

前后完整 package 列表见 `logs/pip_freeze_before.txt` 与 `logs/pip_freeze_after.txt`，
两者 diff **仅**包含上面两行。

明确未改动：

```text
torch      2.9.0+cpu    （未修改）
torch_npu  2.9.0        （未修改）
PyG        2.8.0.post1  （未修改）
numpy      2.4.6        （未修改）
```

PowerGraph 原仓库 `requirements.txt`（含 torch 1.9.1+cu111 / PyG 2.2.0 等旧版本）
**没有执行**。`sklearn.model_selection.train_test_split` 与 `utils.gen_utils` 这两个
loader 在 import 阶段引用、但在 `PowerGrid` 代码路径中**从未被调用**的符号，以进程内
stub 满足（stub 被真正调用时会立即抛错），从而避免再安装 pandas / scipy /
scikit-learn。详见 `scripts/pg_env.py` 与本文 §17 限制说明。

---

## 4. 用了哪些真实数据

数据来源仓库：

```text
https://github.com/PowerGraph-Datasets/PowerGraph-Graph
```

实测使用的 PowerGraph checkout SHA（完整 SHA）：

```text
eb100a2fd836bb8b6bd2d0b799af9c615eac8cb6
```

容器内路径 `vendor/PowerGraph-Graph`，`git status --porcelain` 为空，该目录只被读取，
未修改。

四个真实电网 graph-level dataset 全部可用并被使用：

```text
ieee24 / ieee39 / ieee118 / uk
```

实测 workload（数据取自 `results/phase_a_raw_audit.json`、
`results/forward_*_workload.json`、`results/phase_a_loader*.json`）：

| 数据集 | graph 数 | 每图节点数 | F | 每图边数（edge_index，有向） |
|---|---|---|---|---|
| ieee24 | 21,500 | 24 | 3 | 68–74 |
| ieee39 | 28,000 | 39 | 3 | 86–90 |
| ieee118 | 122,500 | 118 | 3 | 362–370 |
| uk | 64,000 | 29 | 3 | 190–196 |

补充事实（均来自真实 audit）：

* 每个 dataset 的**每图节点数是固定的**（24 / 39 / 118 / 29）。
* 每图边数**不固定**：每个 graph 有 1–5 条被切除（tripped）的支路，loader 会把对应行
  删除后再把前向边复制成双向，因此 edge_index 规模在 68–74 / 86–90 / 362–370 /
  190–196 之间变化。
* 节点特征 `x` 恒为 `torch.float32 [N, 3]`（net active power、net apparent power、
  voltage magnitude），因此**所有 dataset 的 feature dimension 都是 F = 3**。
* raw `.mat` 中 `B_f_tot` / `E_f_post` 为 float64，loader 显式转成 float32。

原始数据文件（README 链接的 PowerGraph 压缩包）：

| 项目 | 值 |
|---|---|
| figshare article / file | `22820534` / `46619158`（`dataset_cascades.zip`, v3） |
| 下载大小 | 61,628,977 bytes |
| md5 | `70b677416d2f377ccfee9f51d8369867` |
| 解压后大小 | 2,958,249,040 bytes（2.75 GiB） |
| 交叉核对 | 同一 article 的 v5 文件 `50083479` 也已下载，md5 与 figshare 公布值 `d4d144b9e720a760e1e077a31f34802d` 一致；两者 raw 文件逐字节同尺寸，仅多一层顶层目录 |

---

## 5. 测试输入是怎么来的

PowerGraph 使用 PyTorch Geometric `InMemoryDataset`。**没有把随机 tensor 作为主结果**，
输入全部来自真实数据，链路如下：

```text
PowerGraph raw .mat  (Bf.mat / blist.mat / Ef.mat / exp.mat / of_*.mat)
        |
        v
PowerGrid(InMemoryDataset)   <-- 仓库原始 loader，未修改
        |
        v
torch_geometric.loader.DataLoader(dataset, batch_size=B, shuffle=False)
        |
        v
Batch
        |
        v
batch.x       [total_nodes, 3]  float32
batch.batch   [total_nodes]     int64   分组索引
```

对算子只取这两个张量，然后调用：

```python
x = batch.x
batch_index = batch.batch
out = global_max_pool(x, batch_index)
```

明确**不执行**：

```text
GNN convolution / Linear / optimizer / training / accuracy evaluation
```

因此本测试测量的是 `global_max_pool` 自身，而不是整网性能。

### 5.1 数据获取方式（figshare 在本网络被阻断）

从本机网络访问 figshare.com 会得到 HTTP 403（article 页面、`api.figshare.com`、
`ndownloader.figshare.com` 全部 403，IPv4 / IPv6 相同），因此 README 中给出的
`wget -O data.tar.gz "https://figshare.com/ndownloader/files/46619158"` 无法直接执行。

实际采用的获取方式：

1. 通过 HTTP proxy 向 figshare downloader 请求 302 重定向，拿到 **presigned S3 URL**；
2. **payload 直接从 `s3-eu-west-1.amazonaws.com/pfigshare-u-files/...` 下载**，不走 proxy。

该流程已固化为可复现脚本 `scripts/fetch_powergraph_data.sh`（优先直连官方
figshare URL，直连失败时才使用 `POWERGRAPH_FIGSHARE_PROXY`，并且每次都重新获取
redirect，不依赖会过期的 presigned URL），解压脚本为
`scripts/extract_powergraph_data.py`；两者都带 checksum 校验。

数据放置位置：`POWERGRAPH_DATA_ARCHIVE`（默认 `<package>/data_download/`）保存原始
zip，解压后的 raw 与 PyG processed 数据位于 `POWERGRAPH_DATA_ROOT`
（默认 `<package>/data`）。**这些目录均在 `.gitignore` 中，原始数据不进入 git，
不 commit。** 完整的数据来源、checksum、目录结构与 license 说明见
[`DATASET.md`](DATASET.md)。

---

## 6. 我们的 global_max_pool 是怎么接进去的

实际测试脚本中的 import 顺序（`scripts/bench_forward.py`、
`scripts/bench_backward.py`、`scripts/profile_app.py` 一致）：

```python
# 1) 先 import 上游 PyG，并把原始实现保存下来做 baseline / CPU oracle
import torch
import torch_geometric.nn as tgnn
ORIGINAL_GMP = tgnn.global_max_pool

# 2) 再启用 compat —— 必须在 import global_max_pool 之前
import pyg_ascend_compat
pyg_ascend_compat.enable()

# 3) 此后从 torch_geometric.nn 取到的就是 Ascend dispatch 版本
from torch_geometric.nn import global_max_pool as COMPAT_GMP
```

`enable()` **早于** `global_max_pool` import 是硬性要求：compat 层在 enable 时替换
`torch_geometric.nn`、`torch_geometric.nn.pool`、`torch_geometric.nn.pool.glob`
三个模块上的 `global_max_pool` 属性；site-packages 不被修改，`disable()` 可还原。

实际调用链：

```text
PyG global_max_pool
    |
    v
pyg_ascend_compat            (按 device/dtype dispatch + 统计 counter)
    |
    v
global_max_pool_ascend       (frozen Stage 2 adapter，FP32 path)
    |
    v
aclnnScatterMaxV1            (frozen delivery OPP)
    |
    v
AI_VECTOR_CORE
```

FP16 / BF16 走 Stage 5 cast 链路；`requires_grad=True` 且 grad enabled 时走 Stage 4
autograd（FP32）或 Stage 5 dtype autograd（FP16/BF16）。

---

## 7. 测了哪些场景

### 7.1 Forward 主矩阵

```text
4 datasets        (ieee24 / ieee39 / ieee118 / uk)
  x 4 batch sizes (1 / 8 / 32 / 128)
  x 3 dtypes      (FP32 / FP16 / BF16)
  x 2 paths
      -> compat_ascend : pyg_ascend_compat 启用后的 Ascend 路径
      -> original_pyg  : enable() 之前捕获的上游 PyG 实现（baseline）
= 96 cases
```

**结果：96 / 96 forward case 全部完成，0 失败。**

实测证据：`results/forward_phaseB_ieee24.csv`（24 行，ieee24）+
`results/forward_phaseC.csv`（72 行，ieee39 / ieee118 / uk）= **96 行 CSV**。
每个 dataset 共 **24 个 forward cases**（4 batch × 3 dtype × 2 paths），
每个 case 在 CSV 中**恰好 1 行**；96 行的 `status` 全部为 `ok`。

### 7.2 Forward + 一阶 backward

```text
4 datasets
  x 4 batch sizes (1 / 8 / 32 / 128)
  x 3 dtypes      (FP32 / FP16 / BF16)
  x compat_ascend
= 48 cases
```

每次迭代执行，并在每轮开始时清零梯度以避免累积：

```python
x.requires_grad_(True)
x.grad = None
out = global_max_pool(x, batch_index)
loss = out.sum()
loss.backward()
```

**结果：48 / 48 cases completed，0 failures，全部 `grad_finite_ok = True`。**

backward 结果放在**独立 CSV / 独立表格**中（`results/forward_backward_phaseC.csv`），
不与 forward latency 混在一起。

### 7.3 batch 取法

每个 `(dataset, batch_size)` 组合使用
`DataLoader(dataset, batch_size=B, shuffle=False)` 的**第一个真实 batch**
（即 dataset 前 B 个 graph），只把 `batch.x` 与 `batch.batch` 搬到 NPU。
四个 dataset 的 graph 数均 ≥ 21,500，因此 batch 1 / 8 / 32 / 128 全部是完整 batch，
本轮没有出现因 memory 或规模无法运行的组合。

---

## 8. 数据类型（dtype）

测试了三种 dtype：`FP32` / `FP16` / `BF16`。

| dtype | 实际执行路径 |
|---|---|
| FP32 | `ScatterMaxV1` **native** path（交付 OPP 本身即 FP32 kernel） |
| FP16 | device **cast 链**：`FP16 -> FP32 -> ScatterMaxV1 -> FP32 -> FP16` |
| BF16 | device **cast 链**：`BF16 -> FP32 -> ScatterMaxV1 -> FP32 -> BF16` |

明确说明：**FP16 / BF16 不是 native half kernel**，交付的 OPP 本身是 FP32 的；
FP16 / BF16 通过 device 上的 Cast 前后转换实现。该 cast 链在 profiler 中可直接观察
（`ieee24 batch=128 fp16/bf16` 的 op_summary 中 `Cast` 出现 25 次，
`Task Type = AI_VECTOR_CORE`）。

---

## 9. 性能测试方法

以下参数直接取自实际执行的脚本，不是估计值。

### 9.1 实际使用的配置

来自 `scripts/run_final.sh` 中的实际调用：

| 项目 | 值 |
|---|---|
| warmup（预热）迭代次数 | **30** |
| measurement iterations | **200** |
| host timer | `time.perf_counter()`（`host_total_us` 交叉校验） |
| 设备计时器 | 每次迭代一对 `torch.npu.Event(enable_timing=True)` |
| sync 方式 | `torch.npu.synchronize()`：warmup 后、t0 前、最后一次 launch 后 |
| 逐次迭代 latency | **保存**（`results/forward_per_iter_*.csv`） |
| 设备 | `ASCEND_RT_VISIBLE_DEVICES=0`，`npu:0` |

实际命令行（`scripts/run_final.sh` 原文）：

```bash
# forward, ieee24
python3 scripts/bench_forward.py --datasets ieee24 --batch-sizes 1,8,32,128 \
    --dtypes fp32,fp16,bf16 --paths compat_ascend,original_pyg \
    --warmup 30 --iters 200 --tag phaseB_ieee24

# forward, ieee39 / ieee118 / uk
python3 scripts/bench_forward.py --datasets ieee39,ieee118,uk --batch-sizes 1,8,32,128 \
    --dtypes fp32,fp16,bf16 --paths compat_ascend,original_pyg \
    --warmup 30 --iters 200 --tag phaseC

# forward + backward
python3 scripts/bench_backward.py --datasets ieee24,ieee39,ieee118,uk \
    --batch-sizes 1,8,32,128 --dtypes fp32,fp16,bf16 --paths compat_ascend --tag phaseC
```

### 9.2 为什么不能直接 `t0 = time.time(); op(); t1 = time.time()`

NPU 是**异步执行**的：host 上的调用只把 kernel 投入队列就返回，`t1 - t0` 量到的基本是
host 端 launch 时间而不是 device 执行时间；不 synchronize 会严重低估延迟。

因此本测试采用两层措施。

**（1）设备侧逐次迭代 event 计时（主指标）**

```python
for _ in range(warmup):
    fn()
torch.npu.synchronize()                       # 清空队列
starts = [torch.npu.Event(enable_timing=True) for _ in range(iters)]
ends   = [torch.npu.Event(enable_timing=True) for _ in range(iters)]
t0 = time.perf_counter()
for i in range(iters):
    starts[i].record()
    fn()
    ends[i].record()
torch.npu.synchronize()                       # 确保 device 工作全部完成
t1 = time.perf_counter()
```

每轮 latency = `start.elapsed_time(end) * 1000`（ms → us），P50 / P95 / P99 直接由这些
per-iteration 样本计算。

**（2）完全同步的对照循环（`sync_*` 列）**

```python
for _ in range(iters):
    torch.npu.synchronize()                   # t0 之前
    t0 = time.perf_counter()
    fn()
    torch.npu.synchronize()                   # t1 之前
    per_iter.append((time.perf_counter() - t0) * 1e6)
```

该列额外包含 host launch / sync 开销，明确与主指标分开报告。

### 9.3 关于「无法真正 pipeline」的说明

被测的 frozen adapter 在**每次调用中都会做一次 device→host 同步**（用于 `batch`
索引范围校验与 `size` 推导：`torch.stack((batch.min(), batch.max())).cpu().tolist()`），
并额外发起约 9 个辅助 NPU op。因此 back-to-back launch 循环实际上无法把调用
排队 —— 量到的**就是 frozen 算子的真实单次调用延迟**，不是流水线吞吐。
这一点在 §12 用独立 probe 量化说明。

### 9.4 输出指标

每个 `dataset x batch_size x dtype x path` 至少输出：

```text
num_graphs, total_nodes, F, dtype
mean latency (us), P50, P95, P99, min, max, std
graphs/sec, nodes/sec
host_total_us（batch timer 交叉校验）
sync_mean / sync_p50 / sync_p95 / sync_p99（同步循环对照）
```

每次迭代的 latency 全部保留在 `results/forward_per_iter_*.csv`
（列：`dataset, batch, dtype, path, loop, iter, latency_us`，
`loop` 取 `async_event` 或 `sync`）。

---

## 10. 如何验证结果正确

虽然本轮主要目标是性能验证，benchmark 前仍做了 correctness sanity。实际检查项：

1. **output shape**：`out.shape == (num_graphs_in_batch, F)`
2. **NaN**：`torch.isnan(out).sum() == 0`
3. **Inf**：`torch.isinf(out).sum() == 0`
4. **CPU PyG oracle**：用 `enable()` **之前捕获的上游 PyG 实现**在 CPU 上对同一个真实
   batch 计算参考值，与 NPU 结果逐元素比较，记录
   `oracle_max_abs_diff`、`oracle_max_rel_diff`、tolerance 下的 mismatch 数量
   （rtol/atol：FP32 `1e-5/1e-6`、FP16 `2e-3/1e-2`、BF16 `2e-2/1e-1`）。

实测结果：

| 检查项 | 结果 |
|---|---|
| 输出 shape 检查 | **96 / 96 PASS** |
| NaN / Inf 检查 | **96 / 96 PASS**（NaN = 0，Inf = 0） |
| 与 oracle 的容差比对 | **0 处超差**（全部 96 个 case） |
| FP32 与 CPU oracle | **逐比特一致（bit-exact）**（`oracle_exact_mismatch = 0`，`max_abs_diff = 0`） |
| FP16 与 CPU oracle | 最大绝对偏差 `2.35e-4`，最大相对偏差 `4.06e-4`，容差外元素 = **0**（在 1,944 个非零偏差元素中） |
| BF16 与 CPU oracle | 最大绝对偏差 `1.95e-3`，最大相对偏差 `2.99e-3`，容差外元素 = **0**（在 2,028 个非零偏差元素中） |

补充说明（真实数据）：FP32 的 `oracle_exact_mismatch = 0`，即与 CPU PyG oracle
逐比特一致；FP16 / BF16 由于 cast 舍入存在非零偏差元素
（FP16 1,944 个、BF16 2,028 个，分布在 16 个 case × 每 case 最多 384 个输出元素上），
但**没有一个超出容差**。

forward + backward 侧：

```text
48 / 48 cases：grad finite = True
grad 非零元素数符合「每个 group 只有 argmax 位置拿到 1」的预期
（例：ieee24 batch=1 -> 3；batch=8 -> 24；batch=32 -> 96；
  batch=128 fp32 -> 384，即 128 x F(3)）
```

完整数值见同目录 `POWERGRAPH_GLOBAL_MAX_POOL_PERFORMANCE_EVIDENCE.md` §14。

> 说明：这些只是 benchmark 前的 sanity，不替代已冻结算子的既有 correctness 测试
> （Stage 3/4/5）。本轮**没有**重跑那套测试。

---

## 11. 如何证明真正跑在 NPU 上

这是本次交付最关键的 evidence。方法是用 CANN 自带的 `msprof` 对**实际 benchmark
调用**做 profiling，并检查 torch_npu 的 fallback 日志。

### 11.1 profiler 方法

* 工具：`msprof --application="python3 scripts/profile_app.py ..." --output=<dir> --ai-core=on`
* 应用：`scripts/profile_app.py`，在**真实 PowerGraph batch** 上执行
  `3 次 warmup + 5 次 profiled calls`（与算子自身交付时的 profiler 约定一致）
* 解析：`scripts/parse_profile.py` 解析 `mindstudio_profiler_output/op_summary_*.csv`
  与 `api_statistic_*.csv`，并扫描 msprof log 中的 fallback 标记
* 驱动脚本：`scripts/run_profiles.sh`

实际执行的 profiler case（`scripts/run_final.sh` 原文）：

```text
ieee24_b128_fp32           ieee24  128 fp32 compat_ascend
ieee24_b128_fp16           ieee24  128 fp16 compat_ascend
ieee24_b128_bf16           ieee24  128 bf16 compat_ascend
ieee118_b128_fp32          ieee118 128 fp32 compat_ascend
uk_b128_fp32               uk      128 fp32 compat_ascend
ieee24_b128_fp32_origpyg   ieee24  128 fp32 original_pyg   （baseline 对照）
```

### 11.2 profiler 结果

| case | path | ScatterMaxV1 任务数 | 执行核心类型 | kernel 平均耗时 (us) | AI_CPU 任务数 | aten::scatter_reduce | Host fallback 标记 | gate |
|---|---|---|---|---|---|---|---|---|
| ieee24_b128_fp32 | compat_ascend | 8 | `AI_VECTOR_CORE` | 27.43 | 0 | 0 | NONE | PASS |
| ieee24_b128_fp16 | compat_ascend | 8 | `AI_VECTOR_CORE` | 25.62 | 0 | 0 | NONE | PASS |
| ieee24_b128_bf16 | compat_ascend | 8 | `AI_VECTOR_CORE` | 25.46 | 0 | 0 | NONE | PASS |
| ieee118_b128_fp32 | compat_ascend | 8 | `AI_VECTOR_CORE` | 119.88 | 0 | 0 | NONE | PASS |
| uk_b128_fp32 | compat_ascend | 8 | `AI_VECTOR_CORE` | 32.82 | 0 | 0 | NONE | PASS |
| ieee24_b128_fp32_origpyg | original_pyg | **0** | – | – | 0 | 0 | **存在（PRESENT）** | N/A（预期） |

每个 case 为 3 次 warmup + 5 次 profiled call，任务数 8 是因为 warmup 期间的 device
任务同样被 profiler 捕获。

### 11.3 ScatterMaxV1 kernel 标识

所有 compat case 中实际执行的 device kernel 名（完整名称）：

```text
ScatterMaxV1_7d55161965c898907fdb3028d01c7c76_0
```

该 hash 与已安装的正式 OPP 中的 kernel 描述文件一致（路径为 **original
validation environment** 的实际安装位置）：

```text
/root/zyg/build/scattermax_runtime_opp/vendors/customize/op_impl/ai_core/tbe/kernel/
    ascend910b/scatter_max_v1/ScatterMaxV1_7d55161965c898907fdb3028d01c7c76.json
sha256 = 755a59daeba86e1388c860d633f3dfc94468975832bf05abc5695bb4802130a8
```

### 11.4 Host fallback / AI_CPU / scatter_reduce 审计

对所有 `compat_ascend` profiler case：

* msprof log 中**不存在**以下任一标记：`npu_cpu_fallback`、
  `fall back to run on the CPU`、`507035`、`507011`、`out of range`、
  `vector core exception`、`aicore exception`、`AIV exception`
* `op_summary_*.csv` 中**没有任何** `Task Type` 含 CPU 的任务（`AI_CPU = 0`）
* `api_statistic_*.csv` 中 `aten::scatter_reduce` 出现次数 = **0**
* `ScatterMaxV1` 出现在 `AI_VECTOR_CORE` 上

### 11.5 Python 层 counter 证据

`pyg_ascend_compat` 自带 dispatch counter，`profile_app.py` 在每个 case 前后读取
`pyg_ascend_compat.stats()`：

```text
ieee24_b128_fp32  : total_calls=8  ascend_calls=8  original_calls=0  dtype16_forward_calls=0
ieee24_b128_fp16  : total_calls=8  ascend_calls=8  original_calls=0  dtype16_forward_calls=8
ieee24_b128_bf16  : total_calls=8  ascend_calls=8  original_calls=0  dtype16_forward_calls=8
ieee118_b128_fp32 : total_calls=8  ascend_calls=8  original_calls=0  dtype16_forward_calls=0
uk_b128_fp32      : total_calls=8  ascend_calls=8  original_calls=0  dtype16_forward_calls=0
```

`ascend_calls == total_calls` 且 `original_calls == 0`，说明调用全部进入 Ascend path，
没有落到原始 PyG 实现。

整轮 forward benchmark 的累计 counter 同样是全量 Ascend：

```text
ieee24 运行 : total_calls=5532    ascend_calls=5532    original_calls=0
phaseC 运行 : total_calls=16596   ascend_calls=16596   original_calls=0
```

> 结论：以上证据证明测试过程中真正调用的是我们已经冻结的 Ascend `global_max_pool`
> path（`ScatterMaxV1` on `AI_VECTOR_CORE`），而不是原 PyG CPU fallback。

---

## 12. 最终测试结果是什么（性能如何解读）

完整表格见同目录 `POWERGRAPH_GLOBAL_MAX_POOL_PERFORMANCE_EVIDENCE.md`（含
mean / P50 / P95 / P99 / graphs/sec / nodes/sec 与逐迭代数据）。本节只保留解读所需内容。

### 12.1 FP32 / compat（平均 latency，单位 us）

| 数据集 | b=1 | b=8 | b=32 | b=128 |
|---|---|---|---|---|
| ieee24 | 914.2 | 889.1 | 894.2 | 921.4 |
| ieee39 | 971.3 | 917.8 | 877.6 | 937.6 |
| ieee118 | 856.1 | 875.7 | 979.1 | 1132.5 |
| uk | 995.1 | 841.6 | 865.9 | 858.1 |

### 12.2 FP16（cast 链，平均 latency，单位 us）

| 数据集 | b=1 | b=8 | b=32 | b=128 |
|---|---|---|---|---|
| ieee24 | 916.2 | 946.5 | 956.7 | 945.6 |
| ieee39 | 943.6 | 985.9 | 1026.8 | 950.7 |
| ieee118 | 902.0 | 1111.7 | 1108.9 | 1117.7 |
| uk | 1106.8 | 989.5 | 924.8 | 912.0 |

### 12.3 BF16（cast 链，平均 latency，单位 us）

| 数据集 | b=1 | b=8 | b=32 | b=128 |
|---|---|---|---|---|
| ieee24 | 955.6 | 956.3 | 900.2 | 937.3 |
| ieee39 | 983.1 | 940.1 | 1009.8 | 948.9 |
| ieee118 | 963.0 | 1110.7 | 1137.4 | 1164.9 |
| uk | 895.0 | 928.2 | 932.8 | 906.1 |

### 12.4 完整调用 vs kernel

| 指标 | 实测 |
|---|---|
| `ScatterMaxV1` kernel 本身（profiler 单次 task duration） | ieee24 / uk：**约 25–33 us**；ieee118 batch=128：**约 120 us** |
| 完整 `global_max_pool` 调用（96 个 forward case 的 mean） | **841.6 – 1164.9 us** |

### 12.5 固定开销分解（实测 probe）

在被测 batch（ieee24, batch=128, 3072 nodes, F=3）上做的分解测量
（`results/overhead_breakdown.json`）：

```text
A  空队列 torch.npu.synchronize()                        33.1 us
B  一个 trivial device op + 每轮同步                     101.6 us
C  完整 compat global_max_pool（主指标）                 870.0 us
C2 完整 compat global_max_pool（每轮同步）               795.6 us
D  仅 raw bridge kernel（buffer 预先备好，无 host 同步） 111.8 us
E  original PyG fallback（host CPU）                     853.3 us
```

逐步骤测量（`results/adapter_step_breakdown.json`：每一步单独同步，因此每行都含约
100 us 的 launch+sync floor，不能直接求和）显示主要固定开销来源为：

```text
batch min/max -> host 的 D2H 同步      约 238 us
occupancy 的 zeros + scatter_          约 287 us
（其余 pad / full / empty / masked_fill / crop / int32 cast 每项 53–117 us）
```

工程解读：

> **在 PowerGraph 的 F = 3 极小 feature workload 下，完整 API latency 主要受适配层
> 固定开销影响（一次强制 device→host 同步 + 约 9 个 auxiliary NPU op），而不是
> ScatterMaxV1 kernel 计算本身。**
>
> 该 workload 下的端到端 latency **不代表** `ScatterMaxV1` kernel 在更大 feature
> dimension 下的性能能力。F 越小固定开销占比越高；F 越大 kernel 占比越高
> （ieee118 batch=128 已可看到 kernel 从约 27 us 增长到约 120 us，完整调用也从
> 856 us 上升到 1133 us）。

这不是「算子失败」，也不是「性能不合格」；本轮验收目标不包含 speedup。

---

## 13. 原始 PyG 对照路径（工程观察，非验收条件）

本轮同时测量了 compat path 与 original PyG path 作为工程对照。

关键事实：**原始 PyG 实现在当前 torch_npu 环境中的实际执行位置是 host CPU
fallback**。torch_npu 明确打印：

```text
CAUTION: The operator 'aten::scatter_reduce.two_out' is not currently supported
on the NPU backend and will fall back to run on the CPU.
```

profiler 侧一致：`original_pyg` case 中 `ScatterMaxV1` 任务数 = 0，且 msprof log
中**出现** `npu_cpu_fallback` / `fall back to run on the CPU` 标记
（与 compat case 的 NONE 形成对照）。

因此这是：

```text
Ascend-native implementation   vs   host fallback implementation
```

的工程观察，**不是**纯粹的同设备 kernel-to-kernel 比较。

### 13.1 实测范围

```text
speedup = original_pyg_mean_us / compat_mean_us
```

| 范围 | 值 | 说明 |
|---|---|---|
| 除 ieee118 batch=128 外的 45 个组合 | **0.63x – 1.06x** | PowerGraph F=3 下的完整 API 观测值 |
| ieee118 batch=128（fp32 / fp16 / bf16） | 31.2x / 28.7x / 35.5x | **不作为正式稳定 speedup 结论** |

### 13.2 为什么 ieee118 batch=128 的单点 speedup 不采信

该 cell 的 baseline 在同一进程内稳定（5 次重复：30.3 / 29.6 / 29.4 / 30.4 / 28.9 ms，
进程内 spread 1.05x），但**不同进程之间**观察到：

```text
4.7 ms      27.2 ms      35.3 ms
```

即明显的 fallback cliff + process variance（同尺寸下 batch=64 只有 1.23 ms，
batch=128 跳到约 29 ms；节点数只增加 2 倍而时间增加约 24 倍）。
因此该 cell 只作为「host fallback 在该尺寸下会退化」的定性观察，
不写入正式 speedup 结论。完整数据见 `results/baseline_stability.json`。

> 重申：本轮交付**不**以 speedup 作为 PASS/FAIL 条件。上表只回答
> 「Ascend path 与当前可用的 PyG fallback 相比是什么量级」。

---

## 14. 测试文件与证据

本文件与其全部脚本、结果均已随本包进入算子仓库，包内路径：

```text
<repo>/pyg-ascend-compat/global_max_pool/powergraph_validation/
```

**original validation environment**（历史记录，评审人不需要拥有相同目录）：

```text
container : /root/zyg/powergraph-global-max-pool-bench/
host      : /data/zyg/powergraph-global-max-pool-bench/
```

自本包合入仓库起，脚本不再依赖上述绝对路径：所有输入/输出根目录都从脚本自身位置
推导，并可用环境变量覆盖（`POWERGRAPH_DATA_ROOT`、`POWERGRAPH_RESULTS_ROOT`、
`POWERGRAPH_PROFILE_ROOT`、`POWERGRAPH_UPSTREAM_DIR`、`GLOBAL_MAX_POOL_OPP`、
`SCATTERMAXV1_BRIDGE`）。仓库内路径与各变量默认值见 `README.md`。

### 14.1 性能测试 / 验证脚本

```text
scripts/bench_env.sh                          运行环境（OPP、frozen adapter 路径、bridge、各 root、NPU 绑定）
scripts/pg_env.py                             compat bootstrap + loader 所需的两个进程内 stub
scripts/pg_dataset.py                         未修改的 PowerGrid loader 封装 + torch.load 兼容 context
scripts/bench_forward.py                      forward benchmark（compat + original PyG，主脚本）
scripts/bench_backward.py                     forward + first-order backward benchmark
scripts/profile_app.py                        msprof 应用（单 case，3 warmup + 5 profiled）
scripts/parse_profile.py                      msprof PROF_* 解析 -> gate JSON
scripts/run_profiles.sh                       profiler 驱动 + gate 检查
scripts/run_validation.sh                     端到端驱动（syntax -> smoke -> forward -> fwd+bwd -> profiler -> summary）
scripts/fetch_powergraph_data.sh              PowerGraph 数据获取（官方 URL 优先，带 checksum）
scripts/extract_powergraph_data.py            dataset_cascades.zip 解压到 loader 期望的目录布局
scripts/analysis/phase_a_audit.py             raw .mat 审计（graph / node / edge / dtype）
scripts/analysis/phase_a_loader_check.py      processed dataset + PyG 2.8 loader 验证
scripts/analysis/smoke_compat.py              import 顺序 + 3 种 dtype 的冒烟测试
scripts/analysis/probe_overhead.py            sync floor / trivial op / raw kernel 分解
scripts/analysis/probe_adapter_steps.py       adapter 每步开销分解
scripts/analysis/probe_baseline_stability.py  original PyG fallback 重复性探测
scripts/analysis/make_summaries.py            生成 performance_summary.csv / profiler_summary.csv
scripts/analysis/make_report.py               渲染详细 evidence 报告
```

### 14.2 结果与证据

```text
README.md                                    Reviewer 入口
DATASET.md                                   数据来源 / checksum / 目录结构 / license attribution
POWERGRAPH_GLOBAL_MAX_POOL_VALIDATION.md     <- 本文档（完整技术报告）
POWERGRAPH_GLOBAL_MAX_POOL_PERFORMANCE_EVIDENCE.md   逐项性能证据（全部原始表格）
requirements-validation.txt                  除 frozen Ascend 环境外额外需要的依赖

results/forward_phaseB_ieee24.csv             forward, ieee24（24 行）
results/forward_phaseC.csv                    forward, ieee39/ieee118/uk（72 行）
results/forward_per_iter_phaseB_ieee24.csv    逐迭代 latency
results/forward_per_iter_phaseC.csv           逐迭代 latency
results/forward_phaseB_ieee24_workload.json   实测 workload 统计
results/forward_phaseC_workload.json
results/forward_backward_phaseC.csv            48 行 forward+backward
results/forward_backward_per_iter_phaseC.csv
results/phase_a_raw_audit.json                raw .mat / 每图节点边数 / tripped 支路
results/phase_a_loader.json                   processed dataset 验证
results/phase_a_loader_rest.json
results/overhead_breakdown.json               latency 分解
results/adapter_step_breakdown.json
results/baseline_stability.json               baseline 重复性
results/performance_summary.csv               compat vs original PyG 汇总（含 speedup 列）
results/profiler_summary.csv                  profiler case 汇总
results/profiler_summary.txt                  profiler gate 文本汇总

evidence/frozen_provenance.txt                冻结 SHA / sha256 清单
evidence/environment/pip_freeze_before.txt    环境变更前完整 package 列表
evidence/environment/pip_freeze_after.txt     环境变更后完整 package 列表
evidence/profiler/<case>.gate.json            解析后的 gate 记录
evidence/profiler/<case>.json                 profile_app 记录 + compat counters
evidence/profiler/<case>/mindstudio_profiler_output/*.csv
                                              op_summary / op_statistic / api_statistic / task_time
```

原始 msprof 输出目录（含 sqlite 与 per-op JSON dump）**不进入仓库**；可用
`scripts/run_profiles.sh` 重新生成，新生成的 gate JSON 与仓库中保存的解析结果
格式一致。PowerGraph 原始数据同样只提供获取脚本，见 `DATASET.md`。

---

## 15. 别人怎么复现

以下命令均取自仓库内脚本本体（`scripts/`），不依赖 original validation environment
的绝对路径。所有 root 都可由环境变量覆盖。

### 15.1 准备 frozen nanwang implementation（只读）

```bash
git clone https://github.com/wio1997/nanwang.git
cd nanwang
git checkout feat/global-max-pool-scattermax-zyg
git rev-parse HEAD          # 期望包含本 validation 包的 commit（见仓库 log）
git log --oneline -3
```

算子实现**不需要**复制到 site-packages：`pyg_ascend_compat` 在运行时按
`PYG_ASCEND_ADAPTER_PATH` / `PYG_ASCEND_AUTOGRAD_PATH` / `PYG_ASCEND_STAGE5_PATH`
直接加载本仓库内的 frozen 源码（`bench_env.sh` 会自动指向本 checkout 的
`pyg-ascend-compat/global_max_pool/{stage2,stage4,stage5}`）。

冻结实现本身的来源记录在 `evidence/frozen_provenance.txt`：

```text
operator implementation frozen SHA : c15423e7b303d2b1597621c64585252472301537
delivery / docs HEAD               : d1dc616b809c32819f3f1fab9a8c43528f824f05
```

本 validation 包只新增文档/脚本/结果，未改动上述 frozen 实现。

### 15.2 设置环境

```bash
cd pyg-ascend-compat/global_max_pool/powergraph_validation
source scripts/bench_env.sh
```

`bench_env.sh` 从自身位置推导 `<repo>` / `<package>` 并设置：

```text
PYG_ASCEND_COMPAT_DIR     = <repo>/.../global_max_pool/stage6     # pyg_ascend_compat 所在目录
PYG_ASCEND_ADAPTER_PATH   = <repo>/.../stage2/python/global_max_pool_ascend.py
PYG_ASCEND_AUTOGRAD_PATH  = <repo>/.../stage4/python/global_max_pool_ascend_autograd.py
PYG_ASCEND_STAGE5_PATH    = <repo>/.../stage5/python/global_max_pool_ascend_dtype.py
ASCEND_CUSTOM_OPP_PATH    = ${GLOBAL_MAX_POOL_OPP}
SCATTERMAXV1_BRIDGE       = ${SCATTERMAXV1_BRIDGE}
POWERGRAPH_DATA_ROOT      = ${POWERGRAPH_DATA_ROOT:-<package>/data}
POWERGRAPH_RESULTS_ROOT   = ${POWERGRAPH_RESULTS_ROOT:-<package>/results}
POWERGRAPH_PROFILE_ROOT   = ${POWERGRAPH_PROFILE_ROOT:-<package>/profiler_runs}
POWERGRAPH_UPSTREAM_DIR   = ${POWERGRAPH_UPSTREAM_DIR:-<package>/upstream/PowerGraph-Graph}
ASCEND_RT_VISIBLE_DEVICES = ${ASCEND_RT_VISIBLE_DEVICES:-0}
```

`GLOBAL_MAX_POOL_OPP` / `SCATTERMAXV1_BRIDGE` 的默认值即 original validation
environment 使用的构建产物路径（`/root/zyg/build/...`）；换机器时必须覆盖这两个
变量，指向本地构建的正式 OPP 与 bridge。若 bridge 尚未构建，可用
`pyg-ascend-compat/global_max_pool/stage2/extension/build_bridge.sh` 编译。

### 15.3 准备 PowerGraph dataset

```bash
# 1) 上游 loader checkout（只读）
git clone https://github.com/PowerGraph-Datasets/PowerGraph-Graph.git \
    "$POWERGRAPH_UPSTREAM_DIR"

# 2) 数据集归档（官方 figshare URL，自动校验 md5）
bash scripts/fetch_powergraph_data.sh

# 3) 解压到 loader 期望的目录布局
python3 scripts/extract_powergraph_data.py
```

若所在网络无法直连 figshare，可设置
`POWERGRAPH_FIGSHARE_PROXY=http://host:port`（脚本会先尝试直连，失败后才使用
proxy）。完整的数据来源、file id、checksum 与 license 说明见 [`DATASET.md`](DATASET.md)。

首次运行时未修改的 `PowerGrid` loader 会生成
`$POWERGRAPH_DATA_ROOT/<name>/<name>/processed_b/data.pt`
（ieee24 约 65 MB、ieee39 约 105 MB、ieee118 约 1.81 GB、uk 约 0.5 GB）；
后续运行直接复用。

### 15.4 运行性能测试

```bash
source scripts/bench_env.sh

# 冒烟：验证 import 顺序 / 三种 dtype / 环境正确
python3 scripts/analysis/smoke_compat.py

# forward 主矩阵
python3 scripts/bench_forward.py --datasets ieee24,ieee39,ieee118,uk \
    --batch-sizes 1,8,32,128 --dtypes fp32,fp16,bf16 \
    --paths compat_ascend,original_pyg --warmup 30 --iters 200 --tag reproduce

# forward + backward
python3 scripts/bench_backward.py --datasets ieee24,ieee39,ieee118,uk \
    --batch-sizes 1,8,32,128 --dtypes fp32,fp16,bf16 \
    --paths compat_ascend --warmup 30 --iters 200 --tag reproduce

# 生成汇总 CSV
python3 scripts/analysis/make_summaries.py --tag reproduce

# 或者一次跑完全部（含 syntax check / smoke / profiler gate）
bash scripts/run_validation.sh --tag reproduce
bash scripts/run_validation.sh --smoke-only     # 仅做最小迁移冒烟
```

### 15.5 查看结果

```bash
ls "$POWERGRAPH_RESULTS_ROOT"/forward_*.csv "$POWERGRAPH_RESULTS_ROOT"/forward_per_iter_*.csv
column -s, -t "$POWERGRAPH_RESULTS_ROOT"/forward_phaseC.csv | less -S   # 归档证据
column -s, -t "$POWERGRAPH_RESULTS_ROOT"/forward_reproduce.csv | less -S # 本次运行

# 由原始 CSV 重新导出汇总表
python3 scripts/analysis/make_summaries.py            # 归档证据 -> performance_summary.csv
python3 scripts/analysis/make_report.py               # 重新渲染详细 evidence 报告
```

### 15.6 运行代表 case 的 Profiler

```bash
source scripts/bench_env.sh

bash scripts/run_profiles.sh \
  "ieee24_b128_fp32 ieee24 128 fp32 compat_ascend" \
  "ieee24_b128_fp16 ieee24 128 fp16 compat_ascend" \
  "ieee24_b128_bf16 ieee24 128 bf16 compat_ascend" \
  "ieee118_b128_fp32 ieee118 128 fp32 compat_ascend" \
  "uk_b128_fp32 uk 128 fp32 compat_ascend" \
  "ieee24_b128_fp32_origpyg ieee24 128 fp32 original_pyg"

cat "${POWERGRAPH_PROFILE_ROOT}/profiler_summary.txt"
```

gate 检查点（脚本自动输出）：`ScatterMaxV1 -> AI_VECTOR_CORE: PASS`、
`AI_CPU_task_types=[]`、`aten::scatter_reduce occurrences = 0`、
forbidden markers 为空、compat counters `ascend_calls == total_calls`。

---

## 16. 最终结论

```text
POWERGRAPH REAL-WORKLOAD VALIDATION

Datasets:
IEEE24 / IEEE39 / IEEE118 / UK

Real PyG Batch:
PASS

FP32:
PASS

FP16:
PASS

BF16:
PASS

Forward:
96 / 96 PASS

Forward + Backward:
48 / 48 PASS

CPU Oracle:
PASS

ScatterMaxV1:
AI_VECTOR_CORE

AI_CPU:
0

Host fallback:
NONE on compat path

aten::scatter_reduce:
NOT CALLED on compat path

Repository modifications:
NONE
```

> 本次 PowerGraph 测试证明 frozen `global_max_pool` 实现能够在真实电力图 PyG
> workload 下完成 forward / backward、支持 FP32 / FP16 / BF16，并保持 NPU
> device-resident execution（`ScatterMaxV1` on `AI_VECTOR_CORE`，无 host fallback、
> 无 AI_CPU、无 `aten::scatter_reduce`），且结果与 CPU PyG oracle 一致。

性能结果作为 workload characterization 保留：在 PowerGraph 的 `F = 3` 极小 feature
场景下，完整 API latency 由适配层固定开销主导，因此**不作为本轮交付的 PASS/FAIL
条件**，也不代表算子在更大 feature dimension 下的性能能力。

本轮**未**修改 `wio1997/nanwang` 与 `PowerGraph-Datasets/PowerGraph-Graph` 的任何文件，
未创建分支，未 commit，未 push。

---

## 17. 限制与说明

1. **F = 3 是极小 feature workload。** PowerGraph 所有 graph-level dataset 的节点
   特征维度都是 3，单 batch 节点数 24–15,104。固定开销主导是这一 workload 的特性，
   不能外推到 `F` 很大或 `N` 很大的场景。
2. **baseline 运行在 host CPU。** 原始 PyG 路径在当前 torch_npu 上走 CPU fallback，
   因此 §13 是「Ascend-native vs host fallback」的工程观察，不是同设备对比。
3. **机器为共享机器。** 测试固定 NPU 0，运行期间 `npu-smi` 显示 AICore 0%，
   但同一 host 上有其他租户负载（host loadavg 约 26–58）。
4. **P99 基于 200 个样本**（P99 实际依赖约 2 个样本），尾部指标置信度有限；
   逐迭代原始数据已保留以便复查。
5. **数据版本**：使用 README 链接的 v3 `dataset_cascades.zip`（file 46619158）；
   同一 article 的 v5（file 50083479）已核对，raw 文件同尺寸一致。
6. **两个 stub**：`sklearn.model_selection.train_test_split` 与 `utils.gen_utils` 在
   loader import 阶段被引用，但 `PowerGrid` 代码路径从不调用它们；为避免安装
   pandas / scipy / scikit-learn，本 benchmark 在进程内注册 stub（真正被调用会立即
   报错）。这不是修改仓库，但属于对运行环境的已知偏离，已在此明确记录。
7. **未重跑 frozen 算子的 Stage 3/4/5 correctness 测试套件。** §10 的 sanity 只用于
   保证 benchmark 本身的输入输出合理。
