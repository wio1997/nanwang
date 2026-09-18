#!/usr/bin/env python3
"""Stage 3C minimal repair (probe-only copy): make the LARGE_TAIL branch an explicit
`TILING_KEY_IS(1)` branch so the AscendC front-end registers a kernel entry for tiling key 1.

Evidence (all from this machine):
  * op-api resolves the kernel by tiling key: BinaryGetFunctionByEntry(binHandle, tilingKey)
  * the entry list comes from the kernel source's explicit TILING_KEY_IS(<key>) patterns
    (tbe/tikcpp/kernel_info_infer.py); a bare `else` branch is NOT registered as a key
  * the delivered package therefore only had `..._0` and any tiling-key-1 shape failed with 361001

Semantics: for keys 0 and 1 the behaviour is byte-for-byte identical to before; only *undefined*
keys change (they now do nothing instead of silently running the large-tail kernel). No algorithm
change, no tiling change, no semantic merge.

Usage: python3 apply_largetail_entry_fix.py <probe_copy>/op_kernel/scatter_max_v1.cpp
"""

import pathlib
import sys

OLD = """    } else { // TILING_KEY_LARGE_TAIL
        KernelScatterMaxV1<false> op(src, idx, res, argmax, &tiling_data, &pipe);
        op.Process();
    }"""

NEW = """    } else if (TILING_KEY_IS(1)) { // TILING_KEY_LARGE_TAIL
        KernelScatterMaxV1<false> op(src, idx, res, argmax, &tiling_data, &pipe);
        op.Process();
    }"""


def main() -> int:
    path = pathlib.Path(sys.argv[1])
    text = path.read_text()
    if "TILING_KEY_IS(1)" in text:
        raise SystemExit(f"{path} already has an explicit TILING_KEY_IS(1) branch")
    count = text.count(OLD)
    if count != 1:
        raise SystemExit(f"anchor not found exactly once in {path} (count={count})")
    path.write_text(text.replace(OLD, NEW))
    print(f"[fix] explicit TILING_KEY_IS(1) branch written to {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
