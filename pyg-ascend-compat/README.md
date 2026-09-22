# PyG Ascend 项目导航

本目录包含 Ascend 910B3（CANN 8.5.1 / torch_npu 2.9.0 / PyG 2.8.0.post1）上的 PyG 相关工作。
内容分为两部分：**当前正式算子交付** 与 **早期 compatibility screening / historical artifacts**。
请按目标选择入口。

## A. 当前正式算子交付：`global_max_pool/`

入口：[`global_max_pool/README.md`](global_max_pool/README.md)

本目录中**最重要、且已经完成并冻结**的成果是 PyG `global_max_pool` 的 Ascend 原生实现：

- PyG `global_max_pool` Ascend native implementation
- FP32 / FP16 / BF16
- forward
- first-order backward
- 真实 PowerGraph workload
- 执行核心 AI_VECTOR_CORE
- 无 compat Host CPU fallback

### 如果你要看 global_max_pool

推荐阅读顺序：

```text
1. global_max_pool/README.md                       正式 landing page（结论 + 导航）
2. global_max_pool/README_DELIVERY.md              算子完整交付报告
3. global_max_pool/powergraph_validation/README.md  PowerGraph 真实输入输出 + 性能
```

## B. 早期 compatibility screening / historical artifacts

本目录下另有一批第一轮 PyG compatibility screening、归因分析和原始结果文件，例如：

```text
final_report.md
feasibility_attribution.md
global_max_pool_result.json
global_sort_pool_*
graphnorm_*
topk_pooling_*
sag_pooling_*
```

以及同期的 `screening_contract.md`、`environment_gate.md`、`native_path_evidence.md`、
各 API 的 `*_root_cause.md` / `*_result.json` 等。

这些材料都**保留在仓库中**：它们记录了早期 screening 的结论、归因与机器可读原始结果。

> 这些属于早期 PyG compatibility screening、归因和原始结果。
> 如果目标是查看当前已经完成的 `global_max_pool` 算子，请直接进入：
>
> [`global_max_pool/README.md`](global_max_pool/README.md)

其中 `DELIVERY_global_max_pool_ascend.md` 是早期 **FP32 forward 阶段**的交付说明（只覆盖
FP32 forward 路径）；完整的算子交付请看
[`global_max_pool/README_DELIVERY.md`](global_max_pool/README_DELIVERY.md)。

## 其它

`Dockerfile.cann851`、`HANDOFF.md` 等分别为容器配方与交接记录。
