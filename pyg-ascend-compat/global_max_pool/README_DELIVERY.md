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

最终冻结提交：

```text
c15423e7b303d2b1597621c64585252472301537
```

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

### 10.1 Checkout

建议直接使用冻结 commit：

```bash
git checkout feat/global-max-pool-scattermax-zyg
git checkout c15423e7b303d2b1597621c64585252472301537
```

确认：

```bash
git status
```

应为 clean。

---

### 10.2 基本 PyG API 测试

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

### 10.3 FP16 / BF16

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

### 10.4 Tie gradient

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

### 10.5 Zero-max 特殊测试

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

## 11. 自动化测试

Stage 5 提供 consolidated final gate：

```bash
cd pyg-ascend-compat

bash global_max_pool/stage5/tools/stage5_final_gate.sh
```

该 gate 会检查 Stage 5 的关键 correctness evidence，包括：

```text
FP16 semantic matrix
BF16 semantic matrix
real PyG E2E
dtype counters
batch=None delegation
fallback checks
```

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

最终冻结版本：

```text
branch:
feat/global-max-pool-scattermax-zyg

HEAD:
c15423e7b303d2b1597621c64585252472301537
```

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

## 附录：文档核验说明（docs-only）

本附录只记录落库时对正文中**命令与路径**的核验结果，不改动任何技术结论。

1. 正文引用的脚本与测试文件已逐一核对，全部存在于当前冻结仓库 `pyg-ascend-compat/global_max_pool/` 下：
   `stage5/tools/stage5_final_gate.sh`、`stage5/tools/run_stage5_profiler.sh`、
   `stage5/tests/{cpu_dtype_oracle,cpu_dtype_count_model,cpu_dtype_div_decide,batch_none_dtype_probe,
   npu_primitive_dtype_audit,run_stage5_dtype_tests,run_stage5_pyg_e2e,profile_stage5_case}.py`
   （同目录还包含 `cpu_dtype_arithmetic_probe.py`、`cpu_dtype_div_search.py` 与
   `stage5/tools/{dtype_probe.cpp,run_stage5_regressions.sh}`）。

2. 运行前提：`stage5_final_gate.sh` 与 `run_stage5_profiler.sh` 默认读取容器内的工作路径
   （`STAGE5_LOGS=/root/zyg/logs/stage5`、正式 OPP
   `/root/zyg/build/scattermax_runtime_opp/vendors/customize`，gate 的 git 检查固定使用
   `/root/zyg/nanwang`）。这些默认值可用 `STAGE5_LOGS` / `STAGE5_OPP` / `STAGE5_PROFILE_APP`
   覆盖；在其它环境中执行前需要先设置对应变量。

3. `stage5_final_gate.sh` 的第一段检查断言的是 Stage 5 远程 freeze **之前**的历史状态
   （`origin/feat/global-max-pool-scattermax-zyg == a586e48`，即 Stage 4 freeze）。本阶段经人工
   review 后已完成最终远程 freeze，现在 `origin/feat/global-max-pool-scattermax-zyg == c15423e`，
   因此该脚本会把这一条**历史性检查**报为 `FAIL`，其余 correctness / profiler / FP32 regression /
   git 检查仍全部 `PASS`，脚本结尾相应显示 `STAGE5: PARTIAL / FAIL`。该脚本属于 Stage 5 冻结
   工具，本轮未做修改。

4. 正文 §10 的示例已在真实环境逐字执行验证：§10.2 的 forward 输出为 `[[3, 5], [4, 6]]` 且
   `x.grad` 正常生成；§10.4 的 2-way tie 得到 `0.5 / 0.5`；§10.5 的 zero-max case 得到
   `1/3, 1/3`（与正文警告一致）；§10.3 的 `float32 / float16 / bfloat16` 输出与梯度 dtype
   均与输入一致。
