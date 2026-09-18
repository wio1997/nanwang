# STAGE 1A — DRIVINGSDK ScatterMaxV1 源码 + 可移植性审计

日期: 2026-09-18
环境: S900K3-47 / 8x Ascend 910B3 / Ubuntu 22.04 aarch64 / driver 26.0.rc1
容器: `wio-pyg-cann851-pyg280` (image `local/wio-pyg-cann851:torch2.9-pyg2.8.0.post1`)
CANN 8.5.1 / Python 3.11.14 / torch 2.9.0+cpu / torch_npu 2.9.0 / PyG 2.8.0.post1

本阶段结论(一句话):
**ScatterMaxV1 FP32 + INT32 可以最小化提取并在 CANN 8.5.1 / ascend910b 上独立编译成功(已实测); 运行待 Stage 1B 验证。**

---

## 0. 工作区与源码获取

| 项 | 值 |
|---|---|
| 工作区 | `/root/zyg/` (容器内), 子目录 `nanwang/ DrivingSDK/ global_max_pool/ build/ logs/ profiler/` |
| nanwang | remote `https://github.com/wio1997/nanwang`, branch `feat/global-max-pool-scattermax-zyg` (从 main 新建), HEAD `a3c9ed1`, working tree clean, main 未改动 |
| DrivingSDK | remote `https://github.com/Ascend/DrivingSDK`, branch `master`, commit `27375a9`, 19MB, 容器内 `/root/zyg/DrivingSDK` |
| 获取方式 | 容器内直连 GitHub 出现 GnuTLS -110 中断, 改为宿主机 clone 后 `docker cp` 进容器 (宿主机副本: `/data/wio/zyg-src/DrivingSDK-master`) |

为什么用 master: 仓库自带 `docker/8.5.1-910b-ubuntu22.04/Dockerfile` (ARG CANN_VERSION=8.5.1,
NPU_TYPE=910b), 即 master 官方支持 CANN 8.5.1 + 910B。另有 `branch_v26.0.0 / v7.3.0 / v7.2.RC1 ...`
分支与同名 tag, 本阶段不需要。

搜索确认: 容器与宿主机此前**都没有** DrivingSDK / ScatterMaxV1 源码, `/usr/local/Ascend` 内也没有;
现有 `R005/R006/R009/R011/R013` 等只是上一阶段 legacy `ge.ScatterMax` 探针产物, 本阶段未使用未修改。

---

## 1. SCATTERMAXV1 SOURCE MANIFEST

| FILE | ROLE | REQUIRED_FOR_MINIMAL_PORT | DRIVINGSDK_PRIVATE_DEPENDENCY | NOTES |
|---|---|---|---|---|
| `kernels/scatter_max/op_kernel/scatter_max_v1.h` (19.6KB) | Ascend C 内核实现: `KernelScatterMaxBase` / `KernelScatterMaxV1<bool smallTail>` / `KernelScatterMaxArgmaxV1<bool>` | **是** | 否 (纯 AscendC API) | 唯一真正需要的内核文件; 用 `DTYPE_SRC/DTYPE_INDEX/DTYPE_RES/DTYPE_ARGMAX` 宏 |
| `kernels/scatter_max/op_kernel/scatter_max_v1.cpp` (690B) | 内核入口 `extern "C" __global__ __aicore__ scatter_max_v1(...)` | **是** | 否 | 只做 `GET_TILING_DATA` + `TILING_KEY_IS(0/1)` 分派 |
| `kernels/scatter_max/op_kernel/scatter_max_argmax_v1.cpp` (708B) | 内核入口 `scatter_max_argmax_v1` | 否 (仅 argmax 需要) | 否 | 与上面同构; global_max_pool 不需要 argmax |
| `kernels/scatter_max/op_host/scatter_max_v1.h` (1.2KB) | tiling data 结构 `ScatterMaxTilingDataV1` + `REGISTER_TILING_DATA_CLASS(ScatterMaxV1/ScatterMaxArgmaxV1)` | **是** | 否 (CANN `register/tilingdata_base.h`) | 14 个 uint64 字段 |
| `kernels/scatter_max/op_host/scatter_max_v1.cpp` (10.6KB) | OpDef(`ScatterMaxV1`, `ScatterMaxArgmaxV1`) + InferShape/InferDtype + `ScatterMaxV1Tiling` + TilingFunc + AddConfig | **是** | **是**: `#include "common/op_host/common.h"` | 实际内容仅需 `kDataSizeMap/DivCeil/CeilAlign` 三个工具 |
| `kernels/common/op_host/common.h` (1012B) | 上面文件的私有 helper (`kDataSizeMap`, `DivCeil`, `CeilAlign`, `DivFloor`, `FloorAlign`) | **是** (需 vendor 或内联) | 是 (DrivingSDK 私有) | 依赖 `register/op_def_registry.h`, `tiling/platform/platform_ascendc.h`, `tiling/tiling_api.h` |
| `kernels/common/op_kernel/common.h` | kernel 侧公共头 | 否 | 是 | scatter_max 内核未引用 |
| `mx_driving/csrc/ScatterMax.cpp` (3.5KB) | C++ wrapper: `scatter_max()`/`scatter_max_validate()`/`scatter_max_backward()`; 调用 `EXEC_NPU_CMD(aclnnScatterMaxV1/ArgmaxV1, ...)` | 否 (Stage 2) | 是 (torch_npu + OpApiCommon) | 含 out 初始化 `-inf`、`masked_fill_`、argmax 后处理 |
| `include/csrc/OpApiCommon.h` (+ `.cpp`) | `EXEC_NPU_CMD` / `EXEC_NPU_CMD_SYNC` / `GET_OP_API_FUNC` 宏与 acl 动态加载 | 否 (Stage 2) | 是 (依赖 torch_npu 头) | 依赖 `torch_npu/csrc/**`, `acl_rt.h` |
| `mx_driving/ops/scatter_max.py` (1.5KB) | Python autograd `ScatterMaxFunction` + `scatter_max` | 否 (Stage 2) | 是 (`import mx_driving._C`) | forward 调 `_C.scatter_max`, backward 调 `_C.scatter_max_backward` (内部 `UnsortedSegmentSum`) |
| `tests/torch/test_scatter_max.py` (9.7KB) | 13 个用例 (dim3/5, bigtail, unaligned, with_out, grad) | 否 (参考) | 是 (`torch_scatter` 做 golden, `mx_driving.common`) | 现有环境**没有** `torch_scatter`, 不能直接跑; 可作为用例形状来源 |
| `docs/zh/api/context/scatter_max.md` (2KB) | API 文档与约束 | 否 (参考) | 否 | 约束以此为准, 见 §4 |
| `model_examples/PointTransformerV3/Ptv3.patch` | 下游调用示例 | 否 | 是 | 只说明调用形态 |
| `CMakeLists.txt` `CMakePresets.json` `cmake/**` `scripts/build_kernel.sh` | DrivingSDK 整套构建 | 否 | 是 (fork 自 CANN 模板但已分叉) | 与 CANN 8.5.1 模板差异: `func.cmake` 726 行, `ascendc_ops_config.py` 317 行, `opdesc_parser.py` 95 行 |
| `mx_driving/csrc/*.cpp` 其它算子, `onnx_plugin/`, `model_examples/` | 无关 | 否 | — | 最小移植不需要 |

最小闭包 = 5 个文件: 2 个 op_kernel (`scatter_max_v1.cpp/.h`) + 2 个 op_host (`scatter_max_v1.cpp/.h`)
+ 1 个 helper (`common.h`, 可内联)。

---

## 2. 真实调用链 (依据源码, 非文件名推测)

推理/框架侧 (Stage 2 才需要):

1. `mx_driving/ops/scatter_max.py` → `ScatterMaxFunction.forward(ctx, updates, indices, out)`
   → `mx_driving._C.scatter_max(updates, indices, out)`
2. `mx_driving/csrc/ScatterMax.cpp` → `scatter_max(src, index, out)`:
   - `index.max()` 求 `idxMaxVal`, 定 `sizes[0] = idxMaxVal + 1`
   - `res = out.value_or(at::empty(sizes).fill_(-inf))`; `argmax = empty(...).fill_(-1)`
   - `scatter_max_validate()` 校验 dim / index 形状
   - `EXEC_NPU_CMD(aclnnScatterMaxV1, src, index, res, argmax)`
   - `res.masked_fill_(res == -inf, 0.0f)`
   - `EXEC_NPU_CMD(aclnnScatterMaxArgmaxV1, src, index, res, argmax)` (argmax 用 `res` 做比较基准)
   - `argmax.masked_fill_(argmax == -1, src.size(0))`
3. `include/csrc/OpApiCommon.h`: `EXEC_NPU_CMD` → 取 `aclnnXxxGetWorkspaceSize` / `aclnnXxx` 函数指针 → aclnn → GE/aclrt → 设备

算子与内核侧 (本阶段重点):

4. op_host `scatter_max_v1.cpp`: `OP_ADD(ScatterMaxV1)` 注册
   - `OpDef::Input("src").DataType({DT_FLOAT})`, `Input("index").DataType({DT_INT32})`,
     `Output("res").DataType({DT_FLOAT})`, `Output("argmax").DataType({DT_INT32})`
   - `SetInferShape(ge::ScatterMaxV1InferShape)` / `SetInferDataType(...)`
   - `AICore().SetTiling(optiling::ScatterMaxV1TilingFunc<false>)`, `AddConfig("ascend910b")`, `AddConfig("ascend910_93")`
5. `ScatterMaxV1TilingFunc<false>` → `ScatterMaxV1Tiling::GetTilingData(ctx)`
   → `init()` (读 shape/dtype, `coreNum = PlatformAscendC::GetCoreNumAiv()`, `ubSize`) →
   `idxNumPerCore/idxBatchNum/tailBatchNum/srcBatchNum/coreNumPerTail/leftSrc*` 计算 →
   `ctx->SetBlockDim(coreNum)`, `ctx->SetTilingKey(0=SMALL_TAIL | 1=LARGE_TAIL)` →
   `tilingData->SaveToBuffer(ctx->GetRawTilingData())`, workspace = 0
   (argmax 版本走 `GetArgmaxTilingData`, 额外减 `ARGAMX_PARALLEL_DEGREE-1` 的 tailBatchNum)
6. 内核入口 `scatter_max_v1(src, idx, res, argmax, workspace, tiling)`:
   `GET_TILING_DATA(tiling_data, tiling)` → `TILING_KEY_IS(0)` →
   `KernelScatterMaxV1<true|false> op(...)` → `op.Process()`
7. `KernelScatterMaxV1::Process()`:
   - `initBatchProcessBuffer()` (UB: `_idxBuf`, `_srcBuf`)
   - `batchProcess(i)` → `DataCopy` 索引 → 分两条路:
     * smallTail: `tailWisebatchProcess()` → `DataCopyPad` 按 tail 读 src →
       `SetAtomicMax<DTYPE_RES>()` + 逐行 `DataCopyPad(_resGM[idxVal*tailElemNum], ...)` → `SetAtomicNone()`
     * largeTail: `elemWiseBatchProcess()` → 逐元素 `DataCopy` + `SetAtomicMax` + `DataCopyPad`
   - `_pipe->Reset()` → `initLeftSrcBuffer()` → `processLeftSrc()` (处理 `idxElemNum % coreNum` 余数)
   (没有 CopyIn/Compute/CopyOut 三段式: 该 kernel 走"MTE3 原子写"路径, 不做向量计算)
8. argmax 路径 `KernelScatterMaxArgmaxV1`: 读 res/argmax → `Compare(mask, src, res, CMPMODE::NE)`
   → `Cast`(int32→float) → `Select(..., VSEL_TENSOR_SCALAR_MODE)` → `Cast`(float→int32, CAST_RINT)
   → `SetAtomicMax<DTYPE_ARGMAX>()` + `DataCopyPad(argmaxGM)`

---

## 3. DEPENDENCY MATRIX (全部在 `/usr/local/Ascend/cann-8.5.1` 实际验证)

内核侧 (tikcpp: `compiler/tikcpp/tikcfw/**`, 符号用 `grep -R` 跟随符号链接确认):

| SYMBOL / HEADER | SOURCE | 标准 CANN 8.5.1? | FOUND LOCALLY? | NEED PORT? | RISK |
|---|---|---|---|---|---|
| `SetAtomicMax<T>()` / `SetAtomicNone()` | `interface/kernel_operator_set_atomic_intf.h` | 是 (**模板形式**, 与源码调用一致) | 是 (impl: `impl/dav_c220/kernel_operator_set_atomic_impl.h`) | 否 | 低 (实测编译通过) |
| `AscendCUtils::GetBitSize` | `impl/kernel_utils_base.h` | 是 | 是 | 否 | 低 |
| `DataCopyPad` + `DataCopyExtParams` (GM 写回重载) | `interface/kernel_struct_data_copy.h`, `kernel_operator_data_copy_intf.h` | 是 | 是 | 否 | 低 |
| `Compare` / `Select` (`CMPMODE`, `SELMODE::VSEL_TENSOR_SCALAR_MODE`) | `interface/kernel_operator_vec_cmpsel_intf.h` | 是 | 是 | 否 | 中 (argmax 语义需 Stage 1B 校验) |
| `Cast` (`RoundMode::CAST_RINT`) | `impl/kernel_scalar.h` | 是 | 是 | 否 | 低 |
| `HardEvent::MTE2_V/V_MTE3/MTE3_MTE2/MTE2_MTE3` + `SetFlag/WaitFlag` | `interface/kernel_operator_block_sync_intf.h` | 是 | 是 | 否 | 低 |
| `TPipe/TBuf/InitBuffer/Reset` | `interface/kernel_tpipe.h` | 是 | 是 | 否 | 低 |
| `ReinterpretCast` / `SetGlobalBuffer` | `interface/kernel_tensor.h` | 是 | 是 | 否 | 低 |
| `TILING_KEY_IS` | `impl/utils/kernel_utils_macros.h` | 是 | 是 | 否 | 低 |
| `GET_TILING_DATA` | **不在静态头文件里** — 由构建工具生成 (`cmake/util/tiling_data_def_build.py` 写入 autogen tiling 头) | 是 (生成器存在) | 是 | 否 | 低 (实测生成成功) |
| `DTYPE_SRC/DTYPE_INDEX/DTYPE_RES/DTYPE_ARGMAX` | 构建期由 `ascendc_impl_build.py` 按 op info 的 param_name 注入 `-DDTYPE_xxx=` | 是 | 是 (命名规则 `param_name[:-5].upper()` 在 DrivingSDK 与 CANN 8.5.1 脚本中**完全一致**) | 否 | 低 |

Host 侧 (`aarch64-linux/include`, `include` 是指向它的符号链接):

| SYMBOL / HEADER | SOURCE | 标准 CANN 8.5.1? | FOUND LOCALLY? | NEED PORT? | RISK |
|---|---|---|---|---|---|
| `register/op_def_registry.h` / `OpDef` / `OP_ADD` / `AutoContiguous` | op_host | 是 | 是 | 否 | 低 |
| `register/tilingdata_base.h` / `BEGIN_TILING_DATA_DEF` / `REGISTER_TILING_DATA_CLASS` | op_host header | 是 | 是 | 否 | 低 |
| `tiling/platform/platform_ascendc.h` (`PlatformAscendC`, `GetCoreNumAiv`, `GetCoreMemSize`, `CoreMemType::UB`) | tiling | 是 | 是 | 否 | 低 |
| `tiling/tiling_api.h` | tiling | 是 | 是 | 否 | 低 |
| `AICore().AddConfig("ascend910b" / "ascend910_93")` | op_host | 是 | 是 | 否 | 低 |
| `ge::DT_FLOAT/DT_INT32`, `FORMAT_ND` | op_host | 是 | 是 | 否 | 低 |

DrivingSDK 私有 / 其他依赖:

| 项 | SOURCE | 标准 CANN? | FOUND LOCALLY? | NEED PORT? | RISK |
|---|---|---|---|---|---|
| `common/op_host/common.h` (`kDataSizeMap/DivCeil/CeilAlign`) | DrivingSDK 私有 | 否 | 是 (仓库内) | 是 (内联或 vendor, 13 行) | 低 |
| `__DRIVING_HOST_AICORE__` 宏 | 构建期宏, 仅用于 `#if == 310` 追加 ascend950 | 未定义时为 0, 自动跳过 | — | 否 | 低 |
| DrivingSDK `cmake/**`, `cmake/util/**` | fork 自 CANN 模板 | 否 | 是 | **不需要** (改用 CANN 8.5.1 自带模板) | 中→已规避 |
| `torch_npu` 头 (`torch_npu/csrc/**`, `OpApiCommon.h`) | wrapper (Stage 2) | 否 | 是 (torch_npu 2.9.0) | Stage 2 | 中 |
| `decorator` (Python) | CANN TBE 内核编译期需要 (`tbe/tvm/contrib/ccec.py`) | CANN 依赖但镜像未装 | 是 (仅 `/root/pyg_feasibility/R009-.../deps` vendor) | 是 (构建期 PYTHONPATH) | **中** — 见 §5 |
| `protobuf` / `scipy` / `torch_scatter` | 仅 wrapper(protobuf 属 TorchAir) / 测试 golden(torch_scatter) | 否 | 否 (deps 里有 protobuf/scipy) | Stage 2/测试时再定 | 中 |

---

## 4. dtype / shape 契约 (源码证据)

| 项 | 结论 | 证据 |
|---|---|---|
| src data dtype | **仅 FP32** | `OpDef: Input("src").DataType({ge::DT_FLOAT})`; tiling 注释 `// now only support float32`; ops-info.json `input0.dtype=float32` |
| index dtype | **仅 INT32** | `OpDef: Input("index").DataType({ge::DT_INT32})`; ops-info.json `input1.dtype=int32` |
| res / argmax dtype | FP32 / INT32 | `Output("res"){DT_FLOAT}`, `Output("argmax"){DT_INT32}` |
| FP16 / BF16 / INT64 | **不支持** | 无任何其它 DataType 枚举 |
| index 形状 | 可为多维, 但除第 0 维外必须全为 1; `index.size(0) == src.size(0)` | `scatter_max_validate()`: `indexLength == 1`, `index.sizes()[0] == src.sizes()[0]` |
| updates/src 形状 | 第 0 维为 N, 其余轴合轴为 M (`tailElemNum = srcElemNum / src.dim0`) | tiling `init()` |
| output 形状 | dim0 = `max(index)+1` (wrapper 计算), 其余轴与 src 相同 | `ScatterMax.cpp` + `ScatterMaxV1InferShape` 返回 `*src_shape` 后再由 wrapper 覆写 dim0 |
| axis | **仅 dim 0** | 全链路无 axis 参数 |
| 重复 index | 原子最大值累加 (`SetAtomicMax`), 结果与顺序无关 | kernel `SetAtomicMax<DTYPE_RES>()` + `DataCopyPad(resGM)` |
| 空 bin | 输出 0 (不是 -inf) | wrapper: `fill_(-inf)` → `masked_fill_(res == ninf, 0.0f)` |
| argmax | 命中 bin 存最大元素的行号; 未命中置 `src.size(0)` | wrapper `argmax.masked_fill_(argmax == -1, argmaxInvalidVal)`; kernel 用 `Compare(src,res,NE)+Select+atomic max` |
| 索引上下界 | 下界 `>= 0` 强制检查; 上界文档要求 `< 491520`, 源码未硬校验 | `TORCH_CHECK(idxMaxVal >= 0, "invalid index value.")`; doc 约束 |
| 规模约束 | `N * (M + 1) < 4,026,531,840`; 文档要求 M 32 字节对齐 | doc; tiling 用 `elemNumPerBlock = 32/srcDSize` |
| 运行核 | AIV (VectorCore) | 编译产物 kernel json: `"coreType": "VectorCore"`, `"core_type": "AIV"`, `magic: RT_DEV_BINARY_MAGIC_ELF_AIVEC` |

---

## 5. 最小编译探针 (已执行, 结果: 成功)

做法: 用 **CANN 8.5.1 自带的 `op_project_templates/ascendc/customize` 模板**建工程,
只放入 `scatter_max_v1.cpp/.h` (op_kernel) + `scatter_max_v1.cpp/.h` (op_host)
+ `common/op_host/common.h` (原样放成 `op_host/common/op_host/common.h` 以满足原 include 路径, **未改一行源码**)。
工程: `/root/zyg/build/scattermax_probe`, 日志 `/root/zyg/logs/probe_build*.log`。

过程中需要 3 个环境适配 (都不是源码移植):

1. `ASSEND_CANN_PACKAGE_PATH`: 模板 preset 默认 `/usr/local/Ascend/latest` (容器里**不存在**,
   只有 `/usr/local/Ascend/cann-8.5.1` 与 `/usr/local/Ascend/ascend-toolkit/latest`),
   第一次失败即 `register/tilingdata_base.h: No such file or directory`。改为 `cann-8.5.1` 后 host 侧全部编译通过。
2. 模板 `cmake/util/*.py` 是指向 `../../../common/util/*.py` 的**相对符号链接**, 复制出 CANN 目录树后断链
   → 补 `cp -r .../ascendc/common /root/zyg/build/common` 后恢复。
3. 内核编译阶段 `ModuleNotFoundError: No module named 'decorator'`
   (CANN `tbe/tvm/contrib/ccec.py` 需要) → **仅在本次构建命令内**注入
   `PYTHONPATH=/root/pyg_feasibility/R009-scattermax-raw-callability/deps:$PYTHONPATH`
   (该目录正是同事为 TorchAir/TBE 调试 vendor 的依赖; 全局 PYTHONPATH / 环境配置**未修改**)。

结果 (`bash build.sh` exit=0):

| 产物 | 路径 |
|---|---|
| 内核二进制 (ascend910b) | `build_out/op_kernel/binary/ascend910b/scatter_max_v1/ScatterMaxV1_7d55….o` (+ `.json`) |
| argmax 内核 | `build_out/op_kernel/binary/ascend910b/scatter_max_argmax_v1/ScatterMaxArgmaxV1_798f….o` |
| 自定义算子包 | `build_out/custom_opp_ubuntu_aarch64.run` (302KB) |
| aclnn 接口 (自动生成) | `build_out/autogen/aclnn_scatter_max_v1.h`, `aclnn_scatter_max_argmax_v1.h` (+ `.cpp`) |
| op info | `build_out/op_kernel/tbe/op_info_cfg/ai_core/ascend910b/aic-ascend910b-ops-info.json` (含 float32/int32 契约) |
| host 库 | `libcust_opsproto_rt2.0.so`, `libcust_opmaster_rt2.0.so`, `libcust_opapi.so` |

结论: **编译可移植** — 内核与 host 代码未改动一行, 仅需 3 项环境适配。

---

## 6. 得出 Stage 1A 结论

1. **能否从 DrivingSDK 提取最小 FP32 + INT32 ScatterMaxV1 并在当前环境独立编译?**
   **能** (已实测, §5)。最小闭包 5 个文件, 零源码修改。
2. **是否"源码真正支持" FP32 + INT32?**
   是。OpDef 只有 `DT_FLOAT`/`DT_INT32`, 无 fp16/bf16/int64 分支; tiling 里 dtype 只用来取字节数
   (`kDataSizeMap`), 内核通过 `DTYPE_*` 宏实例化, 不支持其它 dtype。
3. **主要风险点 (不是编译, 是行为)**
   - 运行期尚未验证 (Stage 1B): 需要装包 (`ASCEND_CUSTOM_OPP_PATH`) 或直接调 aclnn。
   - 语义需与 PyG/`torch_scatter` 对齐: 空 bin=0、`max(index)+1` 决定输出 dim0、argmax 的
     `Compare/Select` 写法是否等价于 torch_scatter 的 argmax(取最先出现的最大者?)。
   - 文档要求 M 32B 对齐 + `max(index) < 491520` + `N*(M+1) < 4.026e9`, 但这些约束在 host/内核里
     并未全部硬校验; PyG 的 global_max_pool 需要按实际 shape 校验。
   - TBE 内核编译依赖 `decorator` (镜像未预装) → 构建可复现性依赖那套 vendor deps 或后续补包。
4. **未做/未动**: 未跑设备执行、未安装任何包、未改全局 PYTHONPATH、未改 `main`、
   未修改 `/root/pyg_feasibility`、`/root/pyg_validation`、`/root/run_*.py` (仅只读引用 deps)。

## 7. 建议的 Stage 1B (下一步, 待确认)

1. 安装探针包并用最小 host 调用 (aclnn 或 GE 单算子) 跑一次 FP32/INT32 正向, 与 CPU golden 对比。
2. 若通过: 写 `mx_driving` 风格的 NPU aten 接口, 让 `aten::scatter_reduce(reduce="amax")` 命中该算子
   (Stage 2), 再回到 PyG `global_max_pool`。
3. 决定 argmax 是否纳入 (global_max_pool 不需要, 但保留可提高与 `torch_scatter` 的兼容度)。
