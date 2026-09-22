# PowerGraph 真实电力图 `global_max_pool` 验证

本文档是 PowerGraph 真实 workload 验证包的**入口页**。本验证针对已冻结（frozen）的
PyG Ascend `global_max_pool` 实现，在
[`PowerGraph-Datasets/PowerGraph-Graph`](https://github.com/PowerGraph-Datasets/PowerGraph-Graph)
提供的真实电力系统图数据上，做单算子的可运行性验证与性能测试。

> **本验证不训练 GNN。**
> PowerGraph 在这里**仅作为真实 PyG 图数据源**：图数据通过上游未修改的 `PowerGrid`
> `InMemoryDataset` 加载，用 `torch_geometric.loader.DataLoader` 组 batch，只把
> `batch.x` 与 `batch.batch` 交给 `global_max_pool`。不做卷积、不做 `Linear`、
> 不做优化器、不做精度评估。

完整技术报告：[`POWERGRAPH_GLOBAL_MAX_POOL_VALIDATION.md`](POWERGRAPH_GLOBAL_MAX_POOL_VALIDATION.md)

---

## 1. 本验证做什么

回答三个问题：冻结后的 Ascend `global_max_pool` 在**真实 PyG 图 workload** 下能否正常
运行？结果是否正确？是否**真的运行在 NPU** 上？

本轮的验收判据是功能性与证据性的，**不以相对原始 PyG fallback（主机侧回退）的加速比
作为判据**：

| 验收项 | 结果 |
|---|---|
| 真实数据 / 真实 PyG batch | PASS |
| FP32 / FP16 / BF16 | PASS |
| Forward | **96 / 96 PASS** |
| Forward + first-order backward | **48 / 48 PASS** |
| 与 CPU PyG oracle 一致 | PASS（FP32 逐比特一致） |
| `ScatterMaxV1` 执行核心类型 | `AI_VECTOR_CORE` |
| AI_CPU 任务数 | 0 |
| compat 路径 Host fallback（主机侧回退） | NONE |
| compat 路径 `aten::scatter_reduce` | 0（未被调用） |

---

## 交付材料导航

下表列出本验证交付的全部材料。**内容**一列可直接点击跳转（GitHub 页面支持）。

| 内容 | 位置 | 用途 |
|---|---|---|
| [本页：PowerGraph 测试入口](README.md) | `powergraph_validation/README.md` | 快速了解测试目的、结论和复现方式 |
| [完整验证报告](POWERGRAPH_GLOBAL_MAX_POOL_VALIDATION.md) | `powergraph_validation/POWERGRAPH_GLOBAL_MAX_POOL_VALIDATION.md` | 查看测试环境、测试矩阵、正确性、Profiler、复现过程和完整结论 |
| [性能证据报告](POWERGRAPH_GLOBAL_MAX_POOL_PERFORMANCE_EVIDENCE.md) | `powergraph_validation/POWERGRAPH_GLOBAL_MAX_POOL_PERFORMANCE_EVIDENCE.md` | 查看各数据集、batch、dtype 的性能数据和性能分析 |
| [数据集说明](DATASET.md) | `powergraph_validation/DATASET.md` | 查看 PowerGraph 数据来源、版本、MD5、下载方式、解压方式和目录结构 |
| [测试脚本](scripts/) | `powergraph_validation/scripts/` | 查看 forward、backward、Profiler 和一键验证脚本 |
| [测试结果](results/) | `powergraph_validation/results/` | 查看实际生成的 CSV / JSON / per-iteration 数据 |
| [真实输入输出演示脚本](scripts/show_ieee24_io.py) | `powergraph_validation/scripts/show_ieee24_io.py` | 加载真实 PowerGraph ieee24 batch，并打印输入、NPU 输出、CPU reference 和一致性检查 |
| [真实输入输出运行记录](results/ieee24_io_example.txt) | `powergraph_validation/results/ieee24_io_example.txt` | 查看 `python3 scripts/show_ieee24_io.py` 的实际运行输出 |
| [Profiler 证据](evidence/profiler/) | `powergraph_validation/evidence/profiler/` | 查看 `ScatterMaxV1 -> AI_VECTOR_CORE`、AI_CPU=0、无 fallback 的设备侧证据 |
| [环境记录](evidence/environment/) | `powergraph_validation/evidence/environment/` | 查看测试前后的 Python package 环境记录 |
| [算子总交付说明](../README_DELIVERY.md) | `../README_DELIVERY.md` | 查看 `global_max_pool` 算子的整体开发、功能、版本和交付说明 |

如果只想快速确认测试是否通过，阅读本 README 即可；如果需要审核测试方法和完整证据，
请继续查看《完整验证报告》；如果需要复现实测，请从「数据集说明」和「测试脚本」开始。

### 推荐阅读顺序

```text
1. README.md                                  本页：结论与导航
2. POWERGRAPH_GLOBAL_MAX_POOL_VALIDATION.md   完整验证报告（方法与证据）
3. DATASET.md                                  数据集来源与获取方式
4. scripts/                                    测试脚本与一键验证
5. results/ 与 evidence/                        原始测试结果与 Profiler 证据
```

## 真实 PowerGraph 输入输出示例

本节用最直接的方式回答客户最关心的问题：算子真实输入是什么、真实输出是什么、
shape / dtype / device 是什么、NPU 结果是否与 CPU 参考一致，以及如何用一条命令自己复现。

### 1. 输入输出分别是什么

`global_max_pool(x, batch)` 接收两个张量：

```text
x.shape     = [N, F]
batch.shape = [N]
```

- `N` = 当前 batch 中**所有图的总节点数**（各图节点拼接后的总数）
- `F` = 每个节点的特征维度
- `batch` 是 **“节点 → 图编号”的映射向量**：每一个元素告诉算子 `x` 对应那一行节点属于哪一张图

本 case（`ieee24`、`batch_size = 2`）：

```text
x.shape     = [48, 3]
batch.shape = [48]

前 24 个 batch 值为 0
后 24 个 batch 值为 1
```

意味着：

```text
2 张图
×
每张 24 个节点
=
48 个输入节点
```

输出：

```text
out.shape = [2, 3]
```

即：

```text
2 张图
每张图输出 1 个 3 维 global-max 特征向量
```

### 2. 数据流

```text
PowerGraph ieee24
      ↓
PyG DataLoader
      ↓
2 张图 × 24 节点 × 3 特征
      ↓
x.shape = [48, 3]
batch.shape = [48]
      ↓
Ascend NPU global_max_pool
      ↓
out.shape = [2, 3]
      ↓
每张图得到一个 3 维 global-max 图级特征
```

### 3. 真实运行结果（ieee24 / batch_size = 2 / FP32 / npu:0）

以下结果来自真实运行，与 [`results/ieee24_io_example.txt`](results/ieee24_io_example.txt) 一致：

```text
dataset      : ieee24
dataset size : 21500
x.shape      : (48, 3)
batch.shape  : (48,)
x.dtype      : torch.float32
nodes/graph  : tensor([24, 24])
```

真实输入前 10 个节点：

```text
tensor([[-0.0315, -0.1009, -0.2132],
        [-0.0170,  0.0018, -0.2245],
        [-0.2399, -0.0083, -0.1479],
        [-0.1005, -0.2604, -0.3756],
        [-0.0966, -0.2673, -0.0672],
        [-0.1820, -0.1139,  0.1419],
        [-0.0309, -0.2285, -0.5248],
        [-0.2280, -0.0300, -0.5927],
        [-0.2333, -0.0204, -0.0961],
        [-0.2596,  0.0277,  0.4073]])
```

batch 分组（无需在 README 打印完整 48 个元素）：

```text
前 24 个节点 -> graph 0
后 24 个节点 -> graph 1
```

NPU 输出：

```text
tensor([[0.7404, 0.6543, 0.4073],
        [0.7404, 0.6543, 0.4073]], device='npu:0')
```

```text
out.shape  = (2, 3)
out.device = npu:0
out.dtype  = torch.float32
```

CPU expected（对每张图**分别独立**执行 `x_cpu[batch_cpu == g].max(dim=0).values`）：

```text
tensor([[0.7404, 0.6543, 0.4073],
        [0.7404, 0.6543, 0.4073]])
```

验证：

```text
exact equal = True
allclose    = True
```

> 两张图输出刚好相同，是这批真实数据的数值结果，**不代表两张图被合并**：
> 前 24 个节点（graph 0）与后 24 个节点（graph 1）是各自**独立**池化的。

### 4. 如何自己复现

```bash
cd pyg-ascend-compat/global_max_pool/powergraph_validation

source scripts/bench_env.sh

python3 scripts/show_ieee24_io.py
```

如果数据不在默认目录，通过环境变量指定（脚本本身**不硬编码任何绝对路径**）：

```bash
export POWERGRAPH_UPSTREAM_DIR=/path/to/PowerGraph-Graph
export POWERGRAPH_DATA_ROOT=/path/to/powergraph/data

python3 scripts/show_ieee24_io.py
```

脚本支持命令行参数：

| 参数 | 默认值 | 说明 |
|---|---|---|
| `--dataset` | `ieee24` | graph-level 数据集名 |
| `--batch-size` | `2` | batch 中的图数量 |
| `--dtype` | `fp32` | `fp32` / `fp16` / `bf16` |

`FP32` 要求逐比特（exact）一致；`FP16` / `BF16` 使用与 benchmark 相同的容差
（`FP16`: rtol `2e-3` / atol `1e-2`，`BF16`: rtol `2e-2` / atol `1e-1`）。

若数据或上游 loader 缺失，脚本会给出中文提示而不是 Python traceback：

```text
PowerGraph 数据未找到。

请先阅读：
powergraph_validation/DATASET.md

或设置：
POWERGRAPH_DATA_ROOT=/path/to/powergraph/data
POWERGRAPH_UPSTREAM_DIR=/path/to/PowerGraph-Graph
```

## 算子性能情况

> 本节是**客户级摘要**，全部数字来自下方已提交的 CSV / 报告，**本轮未重新运行性能矩阵**。

### 测试覆盖

```text
dataset : IEEE24 / IEEE39 / IEEE118 / UK
batch   : 1 / 8 / 32 / 128
dtype   : FP32 / FP16 / BF16
warmup  : 30
measure : 200 (per-iteration)
```

指标：`mean latency`、`P50`、`P95`、`P99`、`graphs/s`、`nodes/s`。

### 代表 case（batch = 128）

| dataset | dtype | mean (us) | P50 (us) | P95 (us) | P99 (us) | graphs/s | nodes/s | 数据来源 |
|---|---|---|---|---|---|---|---|---|
| ieee24 | fp32 | 921.36 | 897.73 | 951.75 | 1,008.29 | 138,925.1 | 3,334,201.6 | `results/forward_phaseB_ieee24.csv`（ieee24 / batch 128 / fp32） |
| ieee24 | fp16 | 945.58 | 936.62 | 993.35 | 1,045.00 | 135,366.2 | 3,248,789.7 | `results/forward_phaseB_ieee24.csv`（ieee24 / batch 128 / fp16） |
| ieee24 | bf16 | 937.33 | 931.65 | 1,010.51 | 1,051.77 | 136,558.8 | 3,277,410.4 | `results/forward_phaseB_ieee24.csv`（ieee24 / batch 128 / bf16） |
| ieee118 | fp32 | 1,132.53 | 1,132.43 | 1,179.64 | 1,232.02 | 113,021.1 | 13,336,486.1 | `results/forward_phaseC.csv`（ieee118 / batch 128 / fp32） |
| uk | fp32 | 858.09 | 854.94 | 897.61 | 918.97 | 149,169.2 | 4,325,906.2 | `results/forward_phaseC.csv`（uk / batch 128 / fp32） |

> 以上行同时汇总在 [`results/performance_summary.csv`](results/performance_summary.csv) 中。

### 如何理解这些数字

- PowerGraph 的节点特征维度只有 **`F = 3`**，是很小的 feature workload。
- 算子 `ScatterMaxV1` kernel 本身的耗时（来源 [`results/profiler_summary.csv`](results/profiler_summary.csv)，
  列 `scattermaxv1_dur_avg_us`）：
  - `ieee24_b128_fp32`：**27.43 us**；`ieee24_b128_fp16`：25.62 us；`ieee24_b128_bf16`：25.46 us
  - `uk_b128_fp32`：**32.82 us**
  - `ieee118_b128_fp32`：**119.88 us**
- 而完整 `global_max_pool` **API** 调用约 **840–1,170 us**（96 个 forward case，见性能证据报告）。
  主要开销来自：
  - adapter 固定开销
  - device → host synchronization
  - 约 9 个辅助 NPU op
- 因此：**在 PowerGraph `F = 3` 场景下，完整 API latency 主要由适配层固定开销主导**，
  不适合用来代表 `ScatterMaxV1` 在更大 feature dimension 下的峰值计算能力。当 reduction
  规模变大时（如 `ieee118` batch=128），kernel 耗时开始显现（119.88 us，对应 API 约 1,133 us）。

### 与原始 PyG（主机侧回退）的对照

- 仓库同时测了 `compat_ascend` 与 `original_pyg` 两条路径。
- 在本 `torch_npu` 环境中，`original_pyg` 实际运行在 **Host CPU fallback** 上。
- 因此这是 **Ascend-native 实现 与 Host CPU fallback 的工程对照**，**不是**同设备
  （NPU ↔ NPU）的 kernel-to-kernel benchmark。多数组合下两条路径相当（约 **0.6x–1.06x**）。
- 唯一明显偏离的是 `ieee118` batch=128：CSV 中 Ascend 路径比 CPU fallback 快约 **31x**
  （来源 [`results/performance_summary.csv`](results/performance_summary.csv)，ieee118 / batch 128 / fp32，
  `speedup` 数值 `31.155`）。但该组合的 CPU fallback 延迟**跨进程波动很大**：
  [性能证据报告](POWERGRAPH_GLOBAL_MAX_POOL_PERFORMANCE_EVIDENCE.md) 记录到 `4.7 ms / 27.2 ms / 35.3 ms`
  三种结果。**因此不能把它当作稳定的加速结论。**

### 完整性能材料

| 内容 | 位置 |
|---|---|
| 性能证据报告（全部表格与分析） | [`POWERGRAPH_GLOBAL_MAX_POOL_PERFORMANCE_EVIDENCE.md`](POWERGRAPH_GLOBAL_MAX_POOL_PERFORMANCE_EVIDENCE.md) |
| 汇总表（compat 与原始 PyG 对照，含 speedup 列） | [`results/performance_summary.csv`](results/performance_summary.csv) |
| forward 逐项结果（ieee24） | [`results/forward_phaseB_ieee24.csv`](results/forward_phaseB_ieee24.csv) |
| forward 逐项结果（ieee39 / ieee118 / uk） | [`results/forward_phaseC.csv`](results/forward_phaseC.csv) |
| Profiler 汇总（kernel 耗时 / gate） | [`results/profiler_summary.csv`](results/profiler_summary.csv) |

## 2. 测试了哪些数据集

| 数据集 | graph 数 | 每图节点数 | F | 每图边数（有向） |
|---|---|---|---|---|
| `ieee24` | 21,500 | 24 | 3 | 68–74 |
| `ieee39` | 28,000 | 39 | 3 | 86–90 |
| `ieee118` | 122,500 | 118 | 3 | 362–370 |
| `uk` | 64,000 | 29 | 3 | 190–196 |

每个数据集的每图节点数是固定的；每图边数会变化，因为每个 graph 有 1–5 条被切除的支路
会在 loader 中删除，之后前向边再复制成双向。

PowerGraph 的节点特征维度只有 **F = 3**，属于非常小的 feature workload。在该规模下，
完整 API latency 中固定的 adapter、同步与辅助算子开销占比较高，因此**不适合用来代表
`ScatterMaxV1` 在更大 feature dimension 下的峰值性能能力**。详见完整验证报告。

## 3. 如何获取数据

原始数据**不随仓库提交**（解压后 2.75 GiB）。来源、校验值与许可证说明见
[`DATASET.md`](DATASET.md)。

```bash
# 上游 loader checkout（只读）+ 数据归档 + 解压
cd pyg-ascend-compat/global_max_pool/powergraph_validation

git clone https://github.com/PowerGraph-Datasets/PowerGraph-Graph.git \
    upstream/PowerGraph-Graph

# 官方 figshare 地址；在原始验证网络中 figshare 返回 HTTP 403，
# 因此脚本同时支持 proxy 回退（见 DATASET.md）
bash scripts/fetch_powergraph_data.sh
python3 scripts/extract_powergraph_data.py
```

解压后的目录结构：

```text
$POWERGRAPH_DATA_ROOT/<name>/<name>/raw/{Bf,blist,Ef,exp,of_bi,of_mc,of_reg}.mat
# 例如 data/ieee24/ieee24/raw/Bf.mat
```

首次运行时，未修改的 `PowerGrid` loader 会生成
`$POWERGRAPH_DATA_ROOT/<name>/<name>/processed_b/data.pt`。

## 4. 如何准备环境

```bash
# 在 CANN 容器内
source scripts/bench_env.sh
```

`bench_env.sh` 从脚本自身位置推导仓库根目录，并导出 frozen compat 包、adapter /
autograd / stage5 模块路径、custom OPP 以及各 benchmark 根目录。所有路径都可以覆盖：

| 环境变量 | 默认值 |
|---|---|
| `GLOBAL_MAX_POOL_OPP` | 本服务器的 `scattermax_runtime_opp/vendors/customize` |
| `SCATTERMAXV1_BRIDGE` | 本服务器的 `stage2_ext/scattermaxv1_bridge.so` |
| `POWERGRAPH_UPSTREAM_DIR` | `<package>/upstream/PowerGraph-Graph` |
| `POWERGRAPH_DATA_ROOT` | `<package>/data` |
| `POWERGRAPH_RESULTS_ROOT` | `<package>/results` |
| `POWERGRAPH_PROFILE_ROOT` | `<package>/profiler_runs` |
| `ASCEND_RT_VISIBLE_DEVICES` | `0` |

除冻结的 Ascend 环境之外，额外需要的 Python 包列在
[`requirements-validation.txt`](requirements-validation.txt) ——
**不要安装 PowerGraph 上游的 `requirements.txt`**（其中固定了旧的 CUDA 版
PyTorch / PyG）。

## 5. 如何运行 forward 性能测试

```bash
source scripts/bench_env.sh

python3 scripts/bench_forward.py \
    --datasets ieee24,ieee39,ieee118,uk \
    --batch-sizes 1,8,32,128 \
    --dtypes fp32,fp16,bf16 \
    --paths compat_ascend,original_pyg \
    --warmup 30 --iters 200 --tag repo_validation
```

`--paths compat_ascend` 测量冻结的 Ascend 路径；`original_pyg` 测量在
`pyg_ascend_compat.enable()` **之前**捕获的上游 PyG 实现（在本 torch_npu 环境中它运行在
**主机 CPU** 上）。

## 6. 如何运行 backward 性能测试

```bash
python3 scripts/bench_backward.py \
    --datasets ieee24,ieee39,ieee118,uk --batch-sizes 1,8,32,128 \
    --dtypes fp32,fp16,bf16 --paths compat_ascend \
    --warmup 30 --iters 200 --tag repo_validation
```

每次迭代执行 `x.grad = None; out = global_max_pool(x, batch); out.sum().backward()`。

## 7. 如何运行 Profiler

```bash
source scripts/bench_env.sh
bash scripts/run_profiles.sh \
  "ieee24_b128_fp32 ieee24 128 fp32 compat_ascend" \
  "ieee24_b128_fp16 ieee24 128 fp16 compat_ascend" \
  "ieee24_b128_bf16 ieee24 128 bf16 compat_ascend" \
  "ieee118_b128_fp32 ieee118 128 fp32 compat_ascend" \
  "uk_b128_fp32 uk 128 fp32 compat_ascend" \
  "ieee24_b128_fp32_origpyg ieee24 128 fp32 original_pyg"
cat "${POWERGRAPH_PROFILE_ROOT:-profiler_runs}/profiler_summary.txt"
```

每个 case 的 gate 要求：`ScatterMaxV1 -> AI_VECTOR_CORE: PASS`、`AI_CPU_task_types=[]`、
`aten::scatter_reduce occurrences = 0`、无 fallback marker，且 compat counter 满足
`ascend_calls == total_calls`、`original_calls == 0`。

也可以一次跑完全流程（syntax 检查 → smoke → forward → forward+backward → Profiler →
汇总）：

```bash
bash scripts/run_validation.sh              # 完整运行
bash scripts/run_validation.sh --smoke-only # 迁移后的快速检查
```

## 8. 结果文件在哪里

```text
results/
  forward_phaseB_ieee24.csv                forward，ieee24               （24 行）
  forward_phaseC.csv                       forward，ieee39/ieee118/uk    （72 行）
  forward_backward_phaseC.csv              forward + first-order backward（48 行）
  forward_per_iter_*.csv                   逐次迭代 latency（P50/P95/P99 的来源）
  forward_backward_per_iter_phaseC.csv
  performance_summary.csv                  汇总：compat 与原始 PyG 对照（含 speedup 列）
  profiler_summary.csv / profiler_summary.txt
  phase_a_raw_audit.json                   原始 .mat / 每图节点与边数审计
  phase_a_loader.json, phase_a_loader_rest.json
  overhead_breakdown.json                  同步下限 / trivial op / 裸 kernel 分解
  adapter_step_breakdown.json              adapter 单次调用各步骤开销
  baseline_stability.json                  主机 CPU fallback 路径的重复性
evidence/
  frozen_provenance.txt                    冻结 SHA 与各源码文件 sha256
  environment/pip_freeze_{before,after}.txt
  profiler/<case>.gate.json                解析后的 Profiler gate 记录
  profiler/<case>/mindstudio_profiler_output/{op_summary,op_statistic,api_statistic,task_time}.csv
```

原始 msprof 输出目录（数百 MB 的 sqlite / JSON）**不随仓库提交**；可用
`scripts/run_profiles.sh` 重新生成。

## 9. 完整技术报告在哪里

[`POWERGRAPH_GLOBAL_MAX_POOL_VALIDATION.md`](POWERGRAPH_GLOBAL_MAX_POOL_VALIDATION.md)
—— 包含测试定位、数据来源、完整性能测试方法、正确性检查、Profiler 验证证据、
性能数据解读、与原始 PyG 路径的对照、已知限制，以及 Reviewer 可直接执行的复现步骤。

逐项性能数据（全部表格）另见
[`POWERGRAPH_GLOBAL_MAX_POOL_PERFORMANCE_EVIDENCE.md`](POWERGRAPH_GLOBAL_MAX_POOL_PERFORMANCE_EVIDENCE.md)。

## 10. 脚本清单

| 脚本 | 用途 |
|---|---|
| `scripts/bench_env.sh` | 环境引导（OPP、adapter 路径、各根目录、NPU 绑定） |
| `scripts/pg_env.py` | 在 import `torch_geometric` 之前启用 compat，并提供 loader 所需 stub |
| `scripts/pg_dataset.py` | 未修改的 `PowerGrid` loader 封装 |
| `scripts/bench_forward.py` | forward 性能测试（compat + 原始 PyG） |
| `scripts/bench_backward.py` | forward + first-order backward 性能测试 |
| `scripts/profile_app.py` | 单个代表 case 的 msprof 应用 |
| `scripts/parse_profile.py` | 解析 msprof `PROF_*` 输出为 gate JSON |
| `scripts/run_profiles.sh` | Profiler 驱动与 gate 检查 |
| `scripts/run_validation.sh` | 端到端驱动脚本 |
| `scripts/fetch_powergraph_data.sh` | 数据集下载（官方来源 + checksum 校验） |
| `scripts/extract_powergraph_data.py` | 解压为 loader 期望的目录结构 |
| `scripts/analysis/phase_a_audit.py` | 原始 `.mat` 数据审计 |
| `scripts/analysis/phase_a_loader_check.py` | processed 数据集 / loader 校验 |
| `scripts/analysis/smoke_compat.py` | import 顺序 + 3 种 dtype 冒烟测试 |
| `scripts/analysis/probe_overhead.py` | latency 分解（同步下限 / 裸 kernel） |
| `scripts/analysis/probe_adapter_steps.py` | adapter 单步开销分解 |
| `scripts/analysis/probe_baseline_stability.py` | 主机 CPU fallback 路径重复性探测 |
| `scripts/analysis/make_summaries.py` | 生成 `performance_summary.csv`、`profiler_summary.csv` |
| `scripts/analysis/make_report.py` | 重新渲染性能证据报告 |

所有性能测试脚本都保持强制的 import 顺序：

```python
import pyg_ascend_compat
pyg_ascend_compat.enable()
from torch_geometric.nn import global_max_pool
```
