#!/usr/bin/env python3
"""Stage 3C probe-only repair: declare the kernel task type in a COPY of the kernel entry.

Root cause chain (all evidence-based):
  1. op-api resolves the kernel function by tiling key: BinaryGetFunctionByEntry(binHandle, tilingKey)
  2. the kernel binary only registered entry `_0`, so tiling key 1 failed with 361001
  3. the entry list (`kernelList`) is built from `config.tiling_key_infos`, i.e. from the
     KERNEL_TASK_TYPE* declarations in the kernel entry source - NOT from --tiling_key alone
     (tbe/tikcpp/ascendc_compile_gen_json.py:_dynamic_kernel_list_to_json)
  4. every other DrivingSDK kernel declares KERNEL_TASK_TYPE_DEFAULT(...); scatter_max_v1.cpp does not

This script only adds that declaration (registration metadata). The kernel algorithm, the tiling
function and the delivered artifacts are untouched.

Usage: python3 patch_kernel_task_type.py <probe_copy>/op_kernel/scatter_max_v1.cpp
"""

import pathlib
import sys

MARKER = "extern \"C\" __global__ __aicore__ void scatter_max_v1("
DECL = """#if __CCE_AICORE__ == 220
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
#endif
"""


def main() -> int:
    path = pathlib.Path(sys.argv[1])
    text = path.read_text()
    if "KERNEL_TASK_TYPE" in text:
        raise SystemExit(f"{path} already declares KERNEL_TASK_TYPE")
    idx = text.index(MARKER)
    brace = text.index("{", idx) + 1
    # insert right after the opening brace of the kernel entry
    new = text[:brace] + "\n    // Stage 3C probe-only repair: register the kernel task type so the build emits\n" \
                           "    // one kernel entry per declared tiling key (_0 and _1)\n" + DECL + text[brace:]
    path.write_text(new)
    print(f"[patch] inserted KERNEL_TASK_TYPE_DEFAULT into {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
