# nanwang

Ascend compatibility validation artifacts.

## 当前主要交付

**Ascend PyG `global_max_pool`**

- 状态：已完成并冻结
- 平台：Ascend 910B3
- CANN：8.5.1
- PyTorch / torch_npu：2.9.0
- PyG：2.8.0.post1

正式交付页面：

→ [`pyg-ascend-compat/global_max_pool/README.md`](./pyg-ascend-compat/global_max_pool/README.md)

该页面汇总算子实现、真实 PowerGraph 输入输出、性能报告和复现方法。

```text
Repository
   ↓
当前主要交付：Ascend PyG global_max_pool
   ↓
pyg-ascend-compat/global_max_pool/README.md
```

## PyTorch Geometric on Ascend 910B3

The [`pyg-ascend-compat`](./pyg-ascend-compat/) directory contains the first-round compatibility screening for:

- Ascend 910B3
- CANN 8.5.1
- PyTorch/torch_npu 2.9.0
- PyTorch Geometric 2.8.0.post1

Start with [`pyg-ascend-compat/final_report.md`](./pyg-ascend-compat/final_report.md). For the follow-up Sort/Cumsum/scatter-max feasibility attribution, see [`pyg-ascend-compat/feasibility_attribution.md`](./pyg-ascend-compat/feasibility_attribution.md).

The directory also includes the container recipe, test harnesses, machine-readable results, profiler summaries, and bounded root-cause notes. PyG 2.8.0.post1 retains the deprecated `global_sort_pool` wrapper, so all seven requested APIs are covered. Raw profiler databases are intentionally excluded because of their size and environment-specific metadata.
