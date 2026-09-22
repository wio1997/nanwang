# Ascend PyG `global_max_pool`

## 状态

- 状态：已完成 / 已冻结
- 平台：Ascend 910B3
- CANN：8.5.1
- Python：3.11.14
- PyTorch：2.9.0+cpu
- torch_npu：2.9.0
- PyG：2.8.0.post1

```text
算子实现冻结 SHA (operator implementation frozen):
c15423e7b303d2b1597621c64585252472301537

当前客户交付分支:
feat/global-max-pool-scattermax-zyg

本轮整理前分支 HEAD:
fbd49b261045da71c6d80e6e7c538db8382bf9cf
```

> `c15423e...` 是**算子实现**的冻结点，**不是**分支 HEAD。
> 分支 HEAD 可能包含之后的文档 / 证据提交；这些后续提交不修改算子实现：
>
> ```text
> operator implementation frozen SHA = c15423e...
> current branch HEAD may contain later docs/test evidence only
> ```

## 我应该看什么？

1. **算子完整交付报告** — [`README_DELIVERY.md`](README_DELIVERY.md)
   查看算子实现、功能范围、版本、测试、限制和交付状态。
2. **PowerGraph 真实输入输出 + 性能** — [`powergraph_validation/README.md`](powergraph_validation/README.md)
   **最推荐客户阅读**：包含真实 ieee24 输入输出、性能摘要、设备侧执行证明和完整材料导航。
3. **PowerGraph 完整验证报告** — [`powergraph_validation/POWERGRAPH_GLOBAL_MAX_POOL_VALIDATION.md`](powergraph_validation/POWERGRAPH_GLOBAL_MAX_POOL_VALIDATION.md)
4. **PowerGraph 性能证据** — [`powergraph_validation/POWERGRAPH_GLOBAL_MAX_POOL_PERFORMANCE_EVIDENCE.md`](powergraph_validation/POWERGRAPH_GLOBAL_MAX_POOL_PERFORMANCE_EVIDENCE.md)
5. **真实输入输出运行记录** — [`powergraph_validation/results/ieee24_io_example.txt`](powergraph_validation/results/ieee24_io_example.txt)
6. **输入输出演示脚本** — [`powergraph_validation/scripts/show_ieee24_io.py`](powergraph_validation/scripts/show_ieee24_io.py)

## 这个算子做什么

`global_max_pool` 把**节点级特征**转换成**图级特征**：

```text
输入:
x     : [N, F]    节点特征
batch : [N]       每个节点属于哪张图

输出:
out   : [num_graphs, F]
```

`batch` 是「节点 -> 图编号」的映射：它的第 i 个元素告诉算子 `x` 的第 i 行属于哪张图。
输出对每张图、每个特征维度取该图内所有节点的最大值。

完整的语义细节（backward、tie gradient、zero-max、empty group、dtype 等）见
[`README_DELIVERY.md`](README_DELIVERY.md)。

## 极简真实 PowerGraph 示例

```text
ieee24
batch_size = 2

x.shape     = [48, 3]
batch.shape = [48]

out.shape   = [2, 3]

NPU output:
[[0.7404, 0.6543, 0.4073],
 [0.7404, 0.6543, 0.4073]]

CPU expected:
same

exact equal = True
```

完整输入、dtype / device、逐节点数据与复现命令见
[`powergraph_validation/README.md`](powergraph_validation/README.md)。

## 极简性能摘要

```text
真实 PowerGraph（forward, batch = 128, fp32）：

完整 Ascend compat API（mean）:
ieee24   ≈ 921.36 us
ieee118  ≈ 1132.53 us
uk       ≈ 858.09 us

ScatterMaxV1 kernel（mean）:
ieee24   ≈ 27.43 us
ieee118  ≈ 119.88 us
```

说明：

```text
PowerGraph F = 3，属于很小的 feature workload；
完整 API latency 主要受 adapter / synchronization 等固定开销影响；
完整性能解读见性能报告。
```

数字来源：[`powergraph_validation/results/performance_summary.csv`](powergraph_validation/results/performance_summary.csv)
与 [`powergraph_validation/results/profiler_summary.csv`](powergraph_validation/results/profiler_summary.csv)。
完整表格与解读见
[`powergraph_validation/POWERGRAPH_GLOBAL_MAX_POOL_PERFORMANCE_EVIDENCE.md`](powergraph_validation/POWERGRAPH_GLOBAL_MAX_POOL_PERFORMANCE_EVIDENCE.md)。

## 最终状态摘要

| 项目 | 结果 |
|---|---|
| FP32 forward | PASS |
| FP16 forward | PASS |
| BF16 forward | PASS |
| FP32 first-order backward | PASS |
| FP16/BF16 first-order backward | PASS |
| PyG public API integration | PASS |
| PowerGraph real workload | PASS |
| ScatterMaxV1 AI_VECTOR_CORE | PASS |
| compat Host CPU fallback | NONE |
| aten::scatter_reduce（compat 路径） | 0（未被调用） |

结论来源：[`README_DELIVERY.md`](README_DELIVERY.md) 与
[`powergraph_validation/README.md`](powergraph_validation/README.md)。

## 开发历史目录说明

```text
stage1 ~ stage6
```

是算子开发、修复、边界验证、dtype / backward / PyG integration 的历史阶段：

- `stage1a` / `stage1b`：ScatterMaxV1 audit 与 runtime 接入
- `stage2`：`global_max_pool` Ascend adapter
- `stage3a` ~ `stage3e`：非对齐 feature、largeTail 边界与 kernel-entry / MTE 修复、交付提升
- `stage4`：CPU gradient oracle 与 FP32 backward
- `stage5`：dtype oracle 与 FP16 / BF16
- `stage6`：PyG `global_max_pool` 集成与回归

同名的 `stage*.md` 文件是各阶段的阶段记录。

这些目录和文件保留用于：

```text
研发追溯
证据审查
问题定位
```

一般客户**无需逐个阅读**。推荐客户只看：

```text
README.md                        本页
README_DELIVERY.md               算子完整交付报告
powergraph_validation/README.md  PowerGraph 真实输入输出 + 性能
```

> 本页面只做导航：没有移动、重命名或删除任何 stage 目录或 stage 文件。

## 复现入口

真实 PowerGraph ieee24 输入输出（一条命令）：

```bash
cd pyg-ascend-compat/global_max_pool/powergraph_validation

source scripts/bench_env.sh

python3 scripts/show_ieee24_io.py
```

算子 smoke test、dtype 矩阵、FP32 regression 与 profiler 验证的完整复现步骤见
[`README_DELIVERY.md`](README_DELIVERY.md)。
