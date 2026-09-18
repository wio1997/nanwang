# Stage 3C — largeTail kernel entry repair + runtime closure

Goal: make the ScatterMaxV1 package contain a kernel entry for tiling key 1 (`LARGE_TAIL`) and
close the runtime question left open by Stage 3B.

**Result: PARTIAL** — the entry repair is **verified** (`kernelList` = `_0` + `_1`, the
`361001 BinaryGetFunctionByEntry(funcEntry=1)` failure is gone), but the large-tail kernel path
then **faults on the device** for every tested F (aligned and non-aligned):
`507035 = ACL_ERROR_RT_VECTOR_CORE_EXCEPTION` with
`errorStr: The DDR address of the MTE instruction is out of range`.
The frozen FP32 forward delivery (SMALL_TAIL envelope) stays fully green.

Full evidence: `../stage3c_large_tail_entry_repair.md`.

## Tools (all probe-only; the delivered package is never modified)

| file | purpose |
|---|---|
| `tools/apply_largetail_entry_fix.py` | probe copy: `else` → `else if (TILING_KEY_IS(1))` so the AscendC front-end registers entry `_1` |
| `tools/patch_kernel_task_type.py` | probe copy: adds `KERNEL_TASK_TYPE_DEFAULT(...)` (tried; not required) |
| `tools/run_stage3c_largetail_matrix.sh` | LT0..LT6 runtime matrix, one process per case, stops at the first failure |

## Reproduce the repaired package

```bash
# probe copy of the Stage 1A project (never the delivered one)
cp -r /root/zyg/build/scattermax_probe/cmake ... /root/zyg/build/stage3b_tiling_probe2/{op_host,op_kernel,...} /root/zyg/build/stage3c_fix5/
python3 tools/apply_largetail_entry_fix.py /root/zyg/build/stage3c_fix5/op_kernel/scatter_max_v1.cpp
echo 'add_ops_compile_options(ScatterMaxV1 OPTIONS --tiling_key=0,1)' >> /root/zyg/build/stage3c_fix5/op_kernel/CMakeLists.txt
cd /root/zyg/build/stage3c_fix5 && PYTHONPATH=/root/pyg_feasibility/R009-scattermax-raw-callability/deps:$PYTHONPATH bash build.sh
# -> kernelList contains _0 and _1 ; install into /root/zyg/build/stage3c_opp_fixed
```

## Run

```bash
bash tools/run_stage3c_largetail_matrix.sh      # LT0 PASS, LT1 fails with 507035 (expected)
```

## Key CANN facts learned (for the next stage)

* `opc --tiling_key=0,1` declares the key list; the CMake idiom is
  `add_ops_compile_options(<OP> OPTIONS --tiling_key=0,1)` (**without** `COMPUTE_UNIT`, otherwise
  `parse_op_debug_confg` drops it because it compares the full SoC against the short name)
* the per-key kernel entries themselves come from the kernel source's explicit
  `TILING_KEY_IS(<key>)` patterns — a bare `else` is not registered as a key
