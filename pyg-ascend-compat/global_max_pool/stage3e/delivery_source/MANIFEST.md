# Stage 3E delivery source snapshot — hash manifest

Date: 2026-09-18T07:30:48Z  ·  container wio-pyg-cann851-pyg280  ·  CANN 8.5.1, ascend910b, vendor customize

These are the exact files compiled into the promoted formal delivery OPP
(`/root/zyg/build/scattermax_runtime_opp/vendors/customize`).

| file | md5 |
|---|---|
| `op_kernel/scatter_max_v1.cpp` | `c4e087beed4f430d2bfe0eacca86803d` |
| `op_kernel/scatter_max_v1.h` | `dde990670c8a14969869669b4949308e` |
| `op_kernel/scatter_max_argmax_v1.cpp` | `a93fe35e69ae0d62c0390ed916382c2a` |
| `op_kernel/CMakeLists.txt` | `042ba88541c8d62290aec9e3374d5dcc` |
| `op_host/scatter_max_v1.cpp` | `72d34d1ddd93ff98b37826855c651a1c` |
| `op_host/scatter_max_v1.h` | `281ec397861b9d55fb030a989ee5e23a` |

Repairs relative to the DrivingSDK reference
(`/root/zyg/DrivingSDK/kernels/scatter_max`, git 27375a9c…):

```diff
op_kernel/scatter_max_v1.cpp : } else { // TILING_KEY_LARGE_TAIL
                            -> } else if (TILING_KEY_IS(1)) { // TILING_KEY_LARGE_TAIL
op_kernel/CMakeLists.txt     : + add_ops_compile_options(ScatterMaxV1 OPTIONS --tiling_key=0,1)
op_kernel/scatter_max_v1.h   : - DTYPE_INDEX idxVal = _idxLocal.GetValue(idxOffset + k);
                               + DTYPE_INDEX idxVal = _idxLocal.GetValue(k);                (x2)
                               - DataCopyPad(_resGM[idxVal * _tailElemNum], _srcLocal,
                               + DataCopyPad(_resGM[idxVal * _tailElemNum + n * _srcBatchNum], _srcLocal,
                               index GM loads -> byte-exact DataCopyPad                          (x4)
```

Build (reproducible):

```bash
export PYTHONPATH=/root/pyg_feasibility/R009-scattermax-raw-callability/deps:$PYTHONPATH
cd /root/zyg/build/scattermax_probe && rm -rf build_out && bash build.sh
```

Resulting kernel package: `kernelList = [..._0, ..._1]`, `supportInfo.tilingKey = ["0","1"]`,
kernel `.o` md5 `5f27b82e8e0828d52e063c9c4a556845` (pre-Stage3E: `2f33d478509f7c6b4c95dadeb78eb050`).
