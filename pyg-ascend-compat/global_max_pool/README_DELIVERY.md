# PyG `global_max_pool` Ascend 适配说明

## 1. 背景

本项目针对 PyTorch Geometric（PyG）的：

```python
torch_geometric.nn.global_max_pool
```

在 Ascend NPU 上进行适配。

开发环境：

```text
Hardware : Ascend 910B3
OS       : Ubuntu 22.04 aarch64
CANN     : 8.5.1
Python   : 3.11.14
PyTorch  : 2.9.0
torch_npu: 2.9.0
PyG      : 2.8.0.post1
```

最终开发分支：

```text
feat/global-max-pool-scattermax-zyg
```

最终算子实现冻结（operator implementation freeze）：

```text
Operator implementation frozen at:
c15423e7b303d2b1597621c64585252472301537
```

说明：

```text
c15423e = final operator implementation freeze

README_DELIVERY.md is maintained by later docs-only commits.
Those commits do not change the frozen operator implementation.
```

也就是说，本文档所在的 `feat/global-max-pool-scattermax-zyg` 分支 HEAD 可能比
`c15423e` 更新（后续只可能是 docs-only commit）；**算子实现本身仍然冻结在 `c15423e`**，
不要把后续 docs-only SHA 当成新的「算子最终 HEAD」。

---

## 2. 原始问题

PyG 的 `global_max_pool` 在传入 `batch` 时，本质上通过 scatter max 完成分组最大值归约。

原始调用路径为：

```text
torch_geometric.nn.global_max_pool
        ↓
torch_geometric.utils.scatter(reduce="max")
        ↓
scatter_reduce_
        ↓
aten::scatter_reduce.two_out
```

在当前 Ascend 软件栈下，该路径虽然能够得到正确结果，但：

```text
aten::scatter_reduce.two_out
```

会进入 Host CPU fallback。

这意味着：

* 算子无法完整运行在 NPU 上；
* 存在 NPU / Host 之间的同步与数据迁移；
* profiler 中能够看到 CPU fallback；
* 无法作为真正的 Ascend-native PyG 执行路径使用。

因此本项目的核心目标是：

> 在不修改 PyG 用户调用方式的前提下，将 `global_max_pool` 的 batch scatter-max 路径替换为 Ascend NPU 原生实现，并补齐 forward、backward、FP16 和 BF16 支持。

---

## 3. 最终实现方案

最终调用方式仍保持 PyG 原生 API：

```python
import pyg_ascend_compat

pyg_ascend_compat.enable()

from torch_geometric.nn import global_max_pool

out = global_max_pool(x, batch, size)
```

需要注意：

```python
pyg_ascend_compat.enable()
```

必须在：

```python
from torch_geometric.nn import global_max_pool
```

之前调用。

启用兼容层之后，带 `batch` 的 Ascend 路径变为：

```text
torch_geometric.nn.global_max_pool
        ↓
pyg_ascend_compat
        ↓
global_max_pool Ascend adapter
        ↓
aclnnScatterMaxV1
        ↓
AI_VECTOR_CORE
```

因此不再经过：

```text
aten::scatter_reduce
```

---

## 4. Forward 适配过程中解决的问题

### 4.1 `ScatterMaxV1` 接入

首先使用 Ascend `aclnnScatterMaxV1` 替代 PyTorch 的 `scatter_reduce(max)`。

最终 FP32 forward 使用：

```text
aclnnScatterMaxV1
```

在真实 Ascend 910B3 上运行，并由 profiler 确认为：

```text
AI_VECTOR_CORE
```

而不是 Host CPU 或 AI_CPU。

---

### 4.2 非对齐 Feature 维度

底层 ScatterMax kernel 对 feature 数据存在对齐要求。

例如：

```text
F = 7
F = 9
F = 17
F = 33
```

均属于非对齐场景。

适配层采用：

```text
原始 F
  ↓
pad 到满足 kernel 要求的维度
  ↓
执行 ScatterMaxV1
  ↓
crop 回原始 F
```

因此用户无需手动调整 tensor shape。

当前已验证：

```text
F = 1
F = 7
F = 8
F = 9
F = 17
F = 33
```

forward 和 backward 均正确。

---

### 4.3 largeTail kernel entry 问题

在较大的 feature 维度下，tiling 会进入 largeTail 路径。

早期实现存在：

```text
TILING_KEY = 1
```

但正式生成的 OPP 中只有：

```text
kernel _0
```

没有对应的 `_1` kernel entry。

最终修改为同时生成：

```text
_0
_1
```

并声明：

```text
supportInfo.tilingKey = ["0", "1"]
```

从而使 normal path 和 largeTail path 都能够被 runtime 正确加载。

---

### 4.4 largeTail MTE 越界

largeTail 首次真正执行时曾出现：

```text
507035
ACL_ERROR_RT_VECTOR_CORE_EXCEPTION
MTE DDR address out of range
```

最终定位到两个 kernel 地址计算问题。

#### 问题一：LocalTensor 索引错误

错误逻辑：

```cpp
_idxLocal.GetValue(idxOffset + k)
```

但 GM 中从 `idxOffset` 开始的数据已经拷贝到：

```text
_idxLocal[0...]
```

因此 local tensor 再次加 `idxOffset` 会造成越界读取。

修复为：

```cpp
_idxLocal.GetValue(k)
```

#### 问题二：largeTail feature offset 缺失

原 result 地址：

```cpp
_resGM[idxVal * _tailElemNum]
```

没有考虑当前 feature chunk。

修复为：

```cpp
_resGM[
    idxVal * _tailElemNum
    + n * _srcBatchNum
]
```

修复之后 largeTail 可以稳定执行。

代表场景：

```text
N = 40, F = 48825
N = 41, F = 48825
N = 40, F = 48826
N = 80, F = 48825
N = 81, F = 48825
N = 320, F = 48825
```

均已通过验证。

---

### 4.5 index GM 尾部读取

早期 index copy 最小只需要：

```text
1 × int32 = 4 Bytes
```

但普通 block copy 可能按 32 Bytes 粒度读取，因此 tensor 尾部理论上可能多读取最多 28 Bytes。

最终没有继续依赖该行为，而是改用 byte-exact copy：

```text
DataCopyPad
```

从实现上消除了该 index GM over-read。

---

## 5. Backward / Tie Gradient

仅支持 forward 还不足以满足真实 PyG 训练场景，因此项目进一步实现了 first-order backward。

实现采用自定义：

```python
torch.autograd.Function
```

forward 仍复用已经冻结的 `ScatterMaxV1` forward，不重新实现归约逻辑。

---

### 5.1 为什么不能简单只给 argmax 传梯度

CPU PyTorch / PyG 实测表明，scatter max 的 backward 并不总是简单的：

```text
一个最大值 -> 梯度全部给该元素
```

当一个 `(group, feature)` 中有多个相同最大值时，需要遵循当前 PyTorch 2.9.0 / PyG 2.8.0.post1 的真实 gradient contract。

经过 CPU oracle 实测：

```text
winner = x == out[batch]
```

多个相同最大值会分摊 upstream gradient。

例如：

```text
2-way tie -> 1/2
3-way tie -> 1/3
```

---

### 5.2 zero-max 特殊语义

CPU oracle 还发现一个容易忽略的行为。

当 reduction result 为：

```text
0
```

时，scatter reduction 的 destination self-slot 也会参与 backward denominator。

例如：

```text
x = [0, -0]
```

两个 input 都是最大值，但 gradient 并不是：

```text
1/2, 1/2
```

而是：

```text
1/3, 1/3
```

因此最终 backward contract 为：

```text
winner = (x == out[batch])

count =
    group 中 winner 数量
    + 1 if out[group, feature] == 0

grad_x =
    winner * grad_out[batch] / count[batch]
```

该行为不是根据文档猜测，而是通过当前 PyTorch / PyG CPU 实测确定。

---

### 5.3 特殊值

以下情况均已单独验证：

```text
negative-only
true -inf
+inf
+0
-0
NaN
empty group
weighted upstream gradient
```

其中：

#### occupied `-inf`

非空 group 中如果所有值都是：

```text
-inf
```

则输出仍为：

```text
-inf
```

并按 tie 规则计算梯度。

#### empty group

如果显式 `size` 中存在空 group：

```text
output = 0
```

但没有任何 input 可以接收该 group 的 gradient。

因此：

```text
occupied -inf group
```

和：

```text
empty group
```

是两种不同语义。

#### NaN

NaN backward 也按照当前 CPU PyTorch / PyG 行为进行了匹配，没有简单忽略或声明为 unsupported。

---

## 6. `batch=None`

当：

```python
batch is None
```

时，PyG 本身不会进入 scatter 路径，而是使用：

```python
x.max(...)
```

因此兼容层不会强行将其改写为 ScatterMaxV1。

该场景继续调用原始 PyG / PyTorch 路径。

CPU 和 NPU 已验证：

```text
forward
tie
-inf
NaN
1-D
backward
```

语义保持一致。

---

## 7. FP16 / BF16

当前 CANN 8.5.1 环境中的自定义 `ScatterMaxV1` OPP 只提供：

```text
FP32
```

实际 capability probe：

```text
FP32 -> success
FP16 -> ACLNN_ERR_PARAM_INVALID
BF16 -> ACLNN_ERR_PARAM_INVALID
```

因此当前实现没有伪造一个不存在的原生 FP16 / BF16 kernel。

最终采用：

```text
FP16/BF16 input
      ↓
device Cast -> FP32
      ↓
frozen FP32 ScatterMaxV1
      ↓
device Cast -> original dtype
```

整个 cast chain 均在 NPU 上执行。

没有：

```text
Host CPU fallback
AI_CPU fallback
```

FP16 / BF16 backward 同样按照各自 dtype 的真实 CPU oracle 实现。

最终验证结果：

```text
FP16 forward   PASS
FP16 backward  PASS
BF16 forward   PASS
BF16 backward  PASS
```

FP16 / BF16 的输出和梯度 dtype 均保持输入 dtype。

需要说明：

> 当前 FP16/BF16 支持属于 device-native cast bridge，并不是独立的原生 FP16/BF16 ScatterMax kernel。

---

## 8. 当前支持范围

当前支持：

```text
FP32 forward
FP32 first-order backward

FP16 forward
FP16 first-order backward

BF16 forward
BF16 first-order backward

batch INT64
explicit size
empty group
negative-only
-inf / +inf
NaN
+0 / -0
tie gradient
weighted upstream gradient
non-aligned feature
largeTail
real PyG global_max_pool API
```

`requires_grad`：

```text
requires_grad=False -> inference path

requires_grad=True
    -> custom first-order autograd path
```

当前不包含：

```text
second-order backward / gradgrad
generic aten::scatter_reduce replacement
其他 PyG 算子
原生 FP16 ScatterMax kernel
原生 BF16 ScatterMax kernel
```

---

## 9. 已验证的边界

Feature：

```text
F = 1
F = 7
F = 8
F = 9
F = 17
F = 33
```

N boundary：

```text
N = 39
N = 40
N = 41
N = 4097
```

largeTail：

```text
N = 40, F = 48825
N = 41, F = 48825
```

其中：

```text
N = 41
```

覆盖 forward 的 leftSrc 边界。

---

## 10. 如何测试

### 10.1 Checkout：两种用途

**（一）阅读交付文档 / 使用当前交付分支**

```bash
git checkout feat/global-max-pool-scattermax-zyg
```

当前 feature branch 同时包含：

```text
frozen operator implementation
+
later docs-only delivery documentation
```

**（二）精确复现冻结算子版本**

如果 Reviewer 需要严格复现最终 operator implementation：

```bash
git checkout c15423e7b303d2b1597621c64585252472301537
```

需要注意：

```text
This commit is the frozen operator implementation.
Later commits only add/update delivery documentation.
```

该 checkout 会进入 detached HEAD（这是预期行为）；同时该 commit 不包含之后新增的
`README_DELIVERY.md`。两种方式下都请确认：

```bash
git status
```

应为 clean。

---

### 10.2 运行前提（本容器 / 本仓库）

本文档中所有命令都在以下环境中实际执行并通过（exit code = 0）：

```text
container : wio-pyg-cann851-pyg280
workdir   : <repo>/pyg-ascend-compat
formal OPP: /root/zyg/build/scattermax_runtime_opp/vendors/customize
```

先准备环境（`stage6/env.sh` 会设置 `ASCEND_CUSTOM_OPP_PATH`、`LD_LIBRARY_PATH`、
`PYG_ASCEND_ADAPTER_PATH`、`SCATTERMAXV1_BRIDGE` 与兼容包 `PYTHONPATH`）：

```bash
cd <repo>/pyg-ascend-compat

export STAGE6_OPP=/root/zyg/build/scattermax_runtime_opp/vendors/customize
source /root/zyg/global_max_pool/stage6/env.sh

# 该容器中的 CANN/msprof 工具链需要额外的 python 依赖目录（例如 decorator）
export PYTHONPATH=/root/pyg_feasibility/R009-scattermax-raw-callability/deps:$PYTHONPATH

# 可选：把测试输出重定向到独立目录，避免覆盖已归档的 evidence 日志
export STAGE5_LOGS=/root/zyg/logs/stage5_review
mkdir -p "$STAGE5_LOGS"
```

说明：Stage 5 的脚本默认使用容器内工作副本路径（`/root/zyg/global_max_pool/...`、
`/root/zyg/logs/stage5`、`/root/zyg/profiler/stage5`），可用 `STAGE5_LOGS`、
`STAGE5_OPP`、`STAGE5_PROFILE_APP`、`STAGE5_PROF_ROOT` 覆盖。

---

### A. Quick API smoke test

最小的 PyG forward/backward 示例。

#### A.1 基本 PyG API 测试

需要先加载项目提供的 Ascend compatibility package。

调用顺序：

```python
import pyg_ascend_compat
pyg_ascend_compat.enable()

from torch_geometric.nn import global_max_pool
```

示例：

```python
import torch
import torch_npu

import pyg_ascend_compat
pyg_ascend_compat.enable()

from torch_geometric.nn import global_max_pool

x = torch.tensor(
    [
        [1.0, 5.0],
        [3.0, 2.0],
        [4.0, 1.0],
        [2.0, 6.0],
    ],
    device="npu",
    dtype=torch.float32,
    requires_grad=True,
)

batch = torch.tensor(
    [0, 0, 1, 1],
    device="npu",
    dtype=torch.int64,
)

out = global_max_pool(x, batch)

print("out:")
print(out)

loss = out.sum()
loss.backward()

print("x.grad:")
print(x.grad)
```

预期 forward：

```text
[[3, 5],
 [4, 6]]
```

且：

```text
x.grad
```

能够正常生成。

---

#### A.2 FP16 / BF16

将：

```python
dtype=torch.float32
```

分别替换为：

```python
dtype=torch.float16
```

和：

```python
dtype=torch.bfloat16
```

即可测试对应 dtype。

需要确认：

```python
print(out.dtype)
print(x.grad.dtype)
```

都与输入 dtype 一致。

---

#### A.3 Tie gradient

例如：

```python
x = torch.tensor(
    [[3.0], [3.0]],
    device="npu",
    requires_grad=True,
)

batch = torch.tensor(
    [0, 0],
    device="npu",
    dtype=torch.int64,
)

out = global_max_pool(x, batch)
out.sum().backward()

print(out)
print(x.grad)
```

非 zero-max 的普通 2-way tie 应表现为两个最大值共同分摊 gradient。

---

#### A.4 Zero-max 特殊测试

建议额外测试：

```python
x = torch.tensor(
    [[0.0], [-0.0]],
    device="npu",
    requires_grad=True,
)

batch = torch.tensor(
    [0, 0],
    device="npu",
    dtype=torch.int64,
)

out = global_max_pool(x, batch)
out.sum().backward()

print(out)
print(x.grad)
```

该 case 用于验证 PyTorch scatter-max 的 zero self-slot backward contract。

不要简单使用普通 2-way tie 的 `1/2 + 1/2` 作为预期。

---

#### A.5 示例实测结果（本文档核验）

```text
A.1 smoke      : out = [[3.0, 5.0], [4.0, 6.0]]，x.grad 正常生成
A.2 dtype      : float32 / float16 / bfloat16 的 out.dtype == x.grad.dtype == 输入 dtype
A.3 tie        : out = [[3.0]]，grad = 0.5 / 0.5
A.4 zero-max   : out = [[0.0]]，grad = 1/3, 1/3（fp32 表现为 0.3333333432674408）
```

---

### B. Correctness matrix（FP16 / BF16）

```bash
python3 global_max_pool/stage5/tests/run_stage5_dtype_tests.py
```

结果（实测 exit code = 0）：

```text
fp16: TOTAL 31 PASS 31 FAIL 0 | gradient bit-exact 31/31 | max ULP 0
bf16: TOTAL 31 PASS 31 FAIL 0 | gradient bit-exact 31/31 | max ULP 0
TOTAL 62  PASS 62  FAIL 0
```

---

### C. Real PyG E2E

```bash
python3 global_max_pool/stage5/tests/run_stage5_pyg_e2e.py
```

结果（实测 exit code = 0）：

```text
TOTAL 25  PASS 25  FAIL 0
counters -> PASS (autograd=22/22, forward=2/2, original=0)
```

---

### D. FP32 regression（确认 frozen FP32 path 未回退）

```bash
bash global_max_pool/stage5/tools/run_stage5_regressions.sh
```

该脚本自带环境（会自行设置正式 OPP / adapter / bridge 并 source `stage6/env.sh`），
完整回归耗时约 6–8 分钟，并会同时重跑 Stage 5 correctness/E2E 与 Stage 4 profiler sanity。

结果（实测 exit code = 0）：

```text
PASS  FP32 Stage 3A 34/34      PASS  FP32 Stage 3B 45/45
PASS  FP32 Stage 3D 9/9        PASS  FP32 Stage 3E 13/13
PASS  Stage 6 demo PASS        PASS  Stage 6 20/20
PASS  Stage 2 16/16            PASS  FP32 Stage 4 bwd 35/35
PASS  FP32 Stage 4 E2E 11/11   PASS  Stage 5 dtype 62/62
PASS  Stage 5 E2E 25/25        PASS  Stage 4 profiler sanity
PASS  FP32 no new ULP regression (max ULP <= 1)
[DONE] script_failed=0 gate_failed=0
```

---

### E. Profiler verification（optional，约 3–5 分钟）

确认 forward / loss / backward 都运行在 device 上（验收标准见 §12）：

```bash
# 可选：把 profiler 原始输出也重定向，避免覆盖已归档的 profiler evidence
export STAGE5_PROF_ROOT=/root/zyg/profiler/stage5_review

bash global_max_pool/stage5/tools/run_stage5_profiler.sh
```

结果（实测 exit code = 0）：

```text
forward-on-device gates: 8
no-AI_CPU gates: 8
no-scatter_reduce gates: 8
[DONE] failed=0
```

---

### F. Historical final gate note（`stage5_final_gate.sh`）

`global_max_pool/stage5/tools/stage5_final_gate.sh` 是 **historical Stage 5 development/freeze
gate**，它用于 Stage 5 开发期间（以及 Stage 5 远程 freeze 之前）确认前置 Stage 4 已冻结。

其第一段检查是一条**历史性的 remote-head 断言**：

```text
origin/feat/global-max-pool-scattermax-zyg == a586e48    (Stage 4 freeze)
```

Stage 5 正式完成并 push 到 `c15423e` 之后，该历史断言按设计不再成立：

```text
post-freeze execution may report PARTIAL/FAIL only because its historical
remote-head assertion expects the pre-Stage-5 value a586e48.
This does not indicate a correctness regression.
```

因此：

* 该脚本属于 Stage 5 冻结工具，**不做修改**；
* **不要**把它的 post-freeze 运行结果当作「一键全 PASS」判据；
* Reviewer 当前应使用上面的 A–E 命令进行验证（这些命令在 post-freeze 状态下实测 exit code = 0）；
* 该脚本其余 correctness / profiler / FP32 regression / git 检查在 post-freeze 状态下仍然 PASS。

---

## 11. 测试资产清单

Stage 5 测试代码位于：

```text
global_max_pool/stage5/tests/
```

主要包括：

```text
cpu_dtype_oracle.py
cpu_dtype_count_model.py
cpu_dtype_div_decide.py
batch_none_dtype_probe.py
npu_primitive_dtype_audit.py
run_stage5_dtype_tests.py
run_stage5_pyg_e2e.py
profile_stage5_case.py
```

---

## 12. Profiler 验证

除了 correctness，还必须确认没有重新进入 CPU fallback。

需要 profile：

```text
forward
loss
backward
```

而不能只 profile forward。

最终 profiler 中应看到：

```text
ScatterMaxV1
    -> AI_VECTOR_CORE
```

backward 主要 device op 包括：

```text
Cast
Equal
GatherV3
InplaceIndexAdd
RealDiv
Mul
MaskedFill
```

验收要求：

```text
Host CPU fallback = NONE
AI_CPU = 0
aten::scatter_reduce = NOT CALLED
```

Stage 5 profiler 脚本：

```bash
bash global_max_pool/stage5/tools/run_stage5_profiler.sh
```

（运行前提与实测结果见 §10 的 E 小节。）

覆盖：

```text
aligned feature
N=41 boundary + tie
non-aligned feature
largeTail
```

FP16 和 BF16 均会执行。

---

## 13. 已完成回归

最终冻结前重新执行了完整 FP32 regression：

```text
Stage 3A : 34/34 PASS
Stage 3B : 45/45 PASS
Stage 3D : 9/9 PASS
Stage 3E : 13/13 PASS

Stage 2 adapter : 16/16 PASS

Stage 4 FP32 backward : 35/35 PASS
Stage 4 real PyG E2E : 11/11 PASS
Stage 4 profiler : 4/4 PASS

Stage 6 demo : PASS
Stage 6 regression : 20/20 PASS
```

Stage 5：

```text
FP16 semantic matrix : 31/31 PASS
BF16 semantic matrix : 31/31 PASS

FP16/BF16 gradients:
bit-exact against corresponding CPU oracle

real PyG E2E:
25/25 PASS

profiler:
8/8 PASS
```

---

## 14. 最终状态

算子实现最终冻结版本（operator implementation freeze）：

```text
branch:
feat/global-max-pool-scattermax-zyg

Operator implementation frozen at:
c15423e7b303d2b1597621c64585252472301537
```

```text
README_DELIVERY.md is maintained by later docs-only commits.
Those commits do not change the frozen operator implementation.
```

当前分支 HEAD 可能比 `c15423e` 更新（例如本文档所在的 docs-only commit）；这些后续 commit
**只新增/更新交付文档**，不改变算子实现，也不构成新的「算子最终 HEAD」。

main 保持未修改：

```text
a3c9ed18cba65cb713e26b334f94d70184960de4
```

最终状态：

```text
FP32: PASS
FP16: PASS
BF16: PASS

forward: PASS
first-order backward: PASS
tie-gradient: PASS
non-aligned feature: PASS
largeTail: PASS
real PyG API: PASS

Host CPU fallback: NONE
AI_CPU: 0
aten::scatter_reduce: NOT CALLED
```

当前 `global_max_pool` Ascend 适配已经完成并冻结。

---

## 15. PowerGraph real-workload validation

在冻结实现之上，使用 **PowerGraph 真实电力图数据**（真实 PyG graph-level dataset，
不训练 GNN、不做整网测试）完成了 `global_max_pool` 单算子的真实 workload 验证与性能测量。

```text
PowerGraph real-workload validation

Datasets:  IEEE24 / IEEE39 / IEEE118 / UK
Real PyG batch: PASS
FP32 / FP16 / BF16: PASS
Forward:            96 / 96 PASS
Forward + Backward: 48 / 48 PASS
CPU PyG oracle:     PASS（FP32 bit-exact）
ScatterMaxV1:       AI_VECTOR_CORE
AI_CPU:             0
Host fallback:      NONE on compat path
aten::scatter_reduce: NOT CALLED on compat path
```

详细内容：

* 入口 / 复现说明：[`powergraph_validation/README.md`](powergraph_validation/README.md)
* 完整技术报告：[`powergraph_validation/POWERGRAPH_GLOBAL_MAX_POOL_VALIDATION.md`](powergraph_validation/POWERGRAPH_GLOBAL_MAX_POOL_VALIDATION.md)
* 数据来源与 checksum：[`powergraph_validation/DATASET.md`](powergraph_validation/DATASET.md)

该验证只新增文档 / 脚本 / 结果，未改动本仓库任何 frozen operator implementation
（adapter / autograd / compat / Stage 1-6 源码均未变更）。

> 说明：PowerGraph 的节点特征维度为 `F = 3`，属于极小 feature workload，端到端
> API latency 主要受适配层固定开销影响，因此**相对原始 PyG fallback 的 speedup
> 不作为该验证的 PASS/FAIL 条件**。

---

## 附录：文档核验说明（docs-only）

本附录只记录落库时对正文中**命令与路径**的核验结果，不改动任何技术结论。

1. 正文引用的脚本与测试文件已逐一核对，全部存在于当前冻结仓库 `pyg-ascend-compat/global_max_pool/` 下：
   `stage5/tools/stage5_final_gate.sh`、`stage5/tools/run_stage5_profiler.sh`、
   `stage5/tests/{cpu_dtype_oracle,cpu_dtype_count_model,cpu_dtype_div_decide,batch_none_dtype_probe,
   npu_primitive_dtype_audit,run_stage5_dtype_tests,run_stage5_pyg_e2e,profile_stage5_case}.py`
   （同目录还包含 `cpu_dtype_arithmetic_probe.py`、`cpu_dtype_div_search.py` 与
   `stage5/tools/{dtype_probe.cpp,run_stage5_regressions.sh}`）。

2. Checkout 说明已按冻结语义修正：`feat/global-max-pool-scattermax-zyg` 用于「阅读文档 / 使用当前
   交付分支」（包含 frozen operator implementation + later docs-only documentation）；
   `c15423e7b303d2b1597621c64585252472301537` 用于「精确复现冻结算子版本」（detached HEAD，
   且该 commit 不包含之后新增的 `README_DELIVERY.md`）。

3. 运行前提：Stage 5 脚本默认读取容器内工作副本路径（`STAGE5_LOGS=/root/zyg/logs/stage5`、
   正式 OPP `/root/zyg/build/scattermax_runtime_opp/vendors/customize`、profiler app
   `/root/zyg/stage5/tests/profile_stage5_case.py`、profiler 输出
   `/root/zyg/profiler/stage5`；`stage5_final_gate.sh` 的 git 检查固定使用
   `/root/zyg/nanwang`）。这些默认值可用 `STAGE5_LOGS` / `STAGE5_OPP` / `STAGE5_PROFILE_APP` /
   `STAGE5_PROF_ROOT` 覆盖；在其它环境中执行前需要先设置对应变量。

4. `stage5_final_gate.sh` 是 **historical Stage 5 development/freeze gate**（用于在 Stage 5 开发开始
   时确认 Stage 4 已冻结）。它的第一段检查是一条历史性的 remote-head 断言：

   ```text
   origin/feat/global-max-pool-scattermax-zyg == a586e48    (Stage 4 freeze)
   ```

   Stage 5 正式完成并 push 到 `c15423e` 之后，该断言按设计不再成立，因此：

   ```text
   post-freeze execution may report PARTIAL/FAIL only because its historical
   remote-head assertion expects the pre-Stage-5 value a586e48.
   This does not indicate a correctness regression.
   ```

   该脚本属于 Stage 5 冻结工具，本轮未做修改；Reviewer 应使用 §10 的 A–E 命令进行验证。

5. §10 的示例与推荐命令均已在真实环境执行验证：

   ```text
   A.1 smoke (fp32)        : out = [[3.0, 5.0], [4.0, 6.0]]，x.grad 正常生成
   A.2 dtype               : float32 / float16 / bfloat16 -> out.dtype == x.grad.dtype == 输入 dtype
   A.3 tie                 : out = [[3.0]]，grad = 0.5 / 0.5
   A.4 zero-max            : out = [[0.0]]，grad = 1/3, 1/3（fp32: 0.3333333432674408）
   B correctness matrix    : exit 0，TOTAL 62 PASS 62 FAIL 0
   C real PyG E2E          : exit 0，TOTAL 25 PASS 25 FAIL 0，original_calls=0
   D FP32 regression       : exit 0，13/13 gates PASS（script_failed=0 gate_failed=0）
   E profiler verification : exit 0，forward-on-device 8、no-AI_CPU 8、no-scatter_reduce 8
   ```
