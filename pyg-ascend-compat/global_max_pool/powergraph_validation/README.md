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
5. results/ 与 evidence/                        原始数据与 Profiler 证据
```

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
