# PyG Ascend 兼容性验证交接文档

更新时间：2026-09-17

## 1. 当前任务

当前工作是验证并归因 PyTorch Geometric（PyG）相关 API 在 Ascend 910B3 上的执行路径，重点区分：

- `A`：AI Core/AI Vector 原生设备执行；
- `C1`：功能正确，但存在 AICPU；
- `C2`：功能正确，但存在 Host CPU fallback；
- 根因归类进一步区分为 `TYPE_LIMITATION`、`ADAPTER_GAP`、`COMPUTE_CAPABILITY_GAP`。

当前不按性能收益排序开发优先级，也不直接进入自研算子。优先确认能否复用现有 CANN 设备能力。

## 2. 服务器与登录

当前验证服务器：

```bash
ssh -p 60006 root@61.241.77.34
```

主机环境：

- CPU 架构：aarch64
- OS：Ubuntu 22.04
- Kernel：`5.15.0-25-generic`
- NPU：8 × Ascend 910B3
- 当前验证主要使用：`npu:1`
- 实际 Host Driver：`26.0.rc1`

注意：客户原始表格中 Driver 为 `25.2.0`，当前主机实际 Driver 为 `26.0.rc1`，这是目前唯一明确的宿主版本不一致项。不要在没有明确授权的情况下替换 Host Driver，驱动变更是主机级、可能需要重启的高风险操作。

## 3. 当前 Docker / 镜像

当前正式验证容器：

```text
wio-pyg-cann851-pyg280
```

对应镜像：

```text
local/wio-pyg-cann851:torch2.9-pyg2.8.0.post1
```

进入容器示例：

```bash
docker exec -it wio-pyg-cann851-pyg280 bash
```

容器内目标软件栈：

- CANN：`8.5.1`
- CANN Path：`/usr/local/Ascend/cann-8.5.1`
- Python：`3.11.14`
- PyTorch：`2.9.0+cpu`
- torch_npu：`2.9.0`
- PyG：`2.8.0.post1`

说明：PyTorch 的 `+cpu` local suffix 是 torch_npu 场景下 CPU base package 的版本标识，不代表计算只能在 CPU 上执行；当前已经通过 NPU matmul smoke test 和 profiler 证明设备执行可用。

不要修改或停止其他已有业务/历史容器。现有 PyG 2.6.1 验证环境和原 service 容器此前均保持未修改状态。

## 4. GitHub 仓库

仓库：

```text
https://github.com/wio1997/nanwang
```

主目录：

```text
pyg-ascend-compat/
```

当前重要资料：

```text
pyg-ascend-compat/
├── final_report.md
├── feasibility_attribution.md
├── environment_gate.md
├── screening_contract.md
├── run_graphnorm_screen.py
├── run_pool_screen.py
├── graphnorm_result.json
├── global_mean_pool_result.json
├── global_add_pool_result.json
├── global_max_pool_result.json
├── global_sort_pool_result.json
├── topk_pooling_result.json
├── sag_pooling_result.json
├── global_max_pool_root_cause.md
├── global_sort_pool_root_cause.md
├── topk_pooling_root_cause.md
└── sag_pooling_root_cause.md
```

关键提交：

- `8936bbf`：将验证环境重新对齐到 PyG `2.8.0.post1`；
- `e947f50`：新增 Sort / Cumsum / scatter-max 可行性归因。

## 5. 七个 API 当前验证结果

PyG 2.8.0.post1 复测后，与之前 PyG 2.6.1 结果一致：

| API | 状态 | 当前结论 |
|---|---|---|
| `GraphNorm` | A | AI Core/AI Vector 路径，无已观察到的 AICPU/Host fallback |
| `global_mean_pool` | A | AI Core/AI Vector |
| `global_add_pool` | A | AI Core/AI Vector |
| `global_max_pool` | C2 | `scatter_reduce.two_out` 触发 Host CPU fallback |
| `global_sort_pool` | C1 | `INT64 Cumsum` 走 AICPU |
| `TopKPooling` | C1 | stable integer Sort + INT64 Cumsum 走 AICPU |
| `SAGPooling` | C1 | 与 TopKPooling 共用的选择路径中 stable integer Sort + INT64 Cumsum 走 AICPU |

所有七个测试项在当前冻结 testcase 范围内均通过 CPU reference 精度检查，没有 `D`。

## 6. Cumsum / Sort 可行性归因

### 6.1 Cumsum

控制变量实验结果：

```text
INT32 Cumsum -> aclnnCumsum_CumsumAiCore_Cumsum -> AI Core
INT64 Cumsum -> aclnnCumsum_CumsumAiCpu_Cumsum -> AICPU
```

shape、数值、dim、调用形式保持一致，两边结果均与 CPU 精确对齐。

当前归类：

```text
TYPE_LIMITATION
```

工程结论：

- 不开发新的 Cumsum kernel；
- 现有 INT32 AI Core 路径可条件复用；
- 只有在完整 prefix-sum 范围可证明不溢出 INT32，且下游 dtype 语义可保持时，才考虑内部转 INT32。

### 6.2 Stable Sort

控制变量实验结果：

```text
INT32 stable Sort -> AICPU
INT64 stable Sort -> AICPU
FP32 stable Sort  -> AI Core
```

因此不是 INT64 单独导致，而是当前测试的整数 stable Sort family 限制。

当前归类：

```text
TYPE_LIMITATION
```

对 `SelectTopK` batch regrouping 的候选复用方案：

- 仅将 batch sort key 转 FP32；
- stable ascending sort；
- 返回 permutation 后作用于原始 INT64 batch；
- batch ID 必须在 `0..2^24` 范围内，保证 FP32 对整数精确表示；
- 超出范围保留原路径。

该方案目前只是语义可行性结论，尚未集成进 PyG。

## 7. 当前最重要的未闭环问题：global_max_pool / scatter-max

### 7.1 当前失败路径

PyG 2.8.0.post1 当前路径：

```text
global_max_pool
    -> scatter(..., reduce="max")
    -> Tensor.scatter_reduce_(..., reduce="amax", include_self=False)
    -> aten::scatter_reduce.two_out
    -> Host CPU fallback
```

已验证：

- `include_self=False` / `True` 都回 Host CPU；
- `Tensor.scatter_reduce_` / `torch.scatter_reduce` 两种合法调用形式都回 Host CPU；
- 因此当前 fallback 不能归因于这两个属性。

### 7.2 已发现的 CANN 设备能力

当前 CANN 8.5.1 安装中存在 Ascend 910B 的 `ScatterMax` AI Core 实现候选，并包含：

- float32 data；
- INT32 index；
- INT64 index。

这说明目前不能直接判断为 `COMPUTE_CAPABILITY_GAP`。

当前最强假设：

```text
ADAPTER_GAP
```

但尚未最终确认。

### 7.3 为什么还不能直接判定 ADAPTER_GAP

已经做过 raw `ScatterMax` 探测：

- INT32 index 和 INT64 index 都能进入设备端 `ScatterMax`；
- 但均出现 runtime `507011` / AI Vector MTE address-out-of-range；
- 因此当前 raw binding 方式尚未证明可以正确复用该设备算子；
- index dtype 不是此次 raw probe 失败的决定因素。

同时，PyTorch alias/mutation 语义尝试在前端 functionalization 阶段被拒绝，没有形成新的设备能力结论。

### 7.4 下一步最小任务

不要开发新 ScatterMax kernel。

下一步只做：

```text
找到当前安装环境中受支持的 CANN / framework ScatterMax 调用入口
        ↓
用一个固定合法 INT32 testcase 正确调用
        ↓
确认 AI Core 执行
        ↓
与 CPU scatter_reduce(amax) reference 对齐
```

只有这一步成功后，才可以最终判断是否为 `ADAPTER_GAP` 并讨论薄适配层。

## 8. 如果 ScatterMax 可以正确复用，目标方案

理想链路：

```text
PyG global_max_pool
        ↓
PyTorch / torch_npu 薄适配
        ↓
复用 CANN ScatterMax
        ↓
AI Core
```

不要自行重新实现 max reduction kernel，除非最终证明没有可复用的设备端路径。

需要特别验证语义：

- repeated indices；
- 全负数 group；
- empty group；
- `-inf`；
- NaN / Inf；
- signed zero；
- output shape / dtype；
- 如果覆盖训练：tie-gradient、self-gradient、alias/in-place 语义。

一个候选 forward 组合方案是：

```text
reduction buffer 初始化为 -inf
        ↓
CANN ScatterMax
        ↓
独立 occupancy mask
        ↓
有节点 group 保留 reduced value
空 group 恢复为 0
```

不能简单用 0 初始化 reduction buffer，因为全负数 group 会得到错误的 0。

## 9. 当前停止条件

### Cumsum

已满足停止条件：`TYPE_LIMITATION`。当前阶段停止继续深挖。

### Sort

已满足停止条件：integer stable Sort family `TYPE_LIMITATION`。当前阶段停止自研讨论。

### scatter-max

尚未满足停止条件。

继续条件只有一个：

```text
找到受支持的 ScatterMax invocation / binding，并完成固定 testcase 的设备端正确输出。
```

在此之前：

- 不判定 `COMPUTE_CAPABILITY_GAP`；
- 不开发新 kernel；
- 不把 raw binding 失败等价为底层设备不支持。

## 10. 推荐交接后的执行顺序

1. 登录 `61.241.77.34:60006`；
2. 进入 `wio-pyg-cann851-pyg280`；
3. 确认 `npu:1` 可用，避免影响其他 NPU 上现有进程；
4. 阅读 `final_report.md` 和 `feasibility_attribution.md`；
5. 不重复跑已经结案的 Cumsum/Sort 根因实验；
6. 只继续 `ScatterMax` 正确受支持调用入口的最小验证；
7. 每一步先保存 log / profiler / testcase，再更新结论；
8. 所有新结论继续提交到 `wio1997/nanwang` 的 `pyg-ascend-compat/` 下。

## 11. 当前一句话状态

> PyG 2.8.0.post1 的七个目标 API 已完成兼容性复测；Cumsum 与 integer stable Sort 已归因为类型限制并停止自研讨论；当前唯一尚未闭环的是 `global_max_pool` 的 C2 Host fallback，CANN 已发现 `ScatterMax` AI Core 候选能力，下一步只需验证其正确、受支持的调用/binding 路径，再决定是否属于适配缺口。
