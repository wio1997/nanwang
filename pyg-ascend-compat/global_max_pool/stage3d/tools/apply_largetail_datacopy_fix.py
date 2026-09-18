#!/usr/bin/env python3
"""Stage 3D minimal repair (probe-only copy): fix the GM->UB src reads in the large-tail paths.

Root cause (proven in this stage):
  `DataCopy(LocalTensor, GlobalTensor, DataCopyParams)` passes `blockLen` straight to the hardware
  intrinsic `copy_gm_to_ubuf`, whose length unit is **32-byte blocks** (see
  `DataCopySliceGm2UBImpl`, which converts `blockLen * ONE_BLK_SIZE` into byte units before calling
  DataCopyPad). The large-tail code paths pass an **element count** (`srcLoadNumAlign`, e.g. 48824)
  instead, so the hardware reads 48824 * 32 B = 1.56 MB per call - far past the end of the tensor -
  and the device reports `507035` / "The DDR address of the MTE instruction is out of range".

Minimal fix: keep every other line identical and replace the two `DataCopy(<src GM> -> <UB>)` calls
with the byte-exact, unaligned-safe `DataCopyPad` form already used by the small-tail path.

Usage: python3 apply_largetail_datacopy_fix.py <probe_copy>/op_kernel/scatter_max_v1.h
"""

import pathlib
import sys

OLD = "DataCopy(_srcLocal, _srcGM[srcOffset], srcLoadNumAlign);"

NEW = ("DataCopyPad(_srcLocal, _srcGM[srcOffset],\n"
       "            {1, static_cast<uint32_t>(srcLoadNum * sizeof(DTYPE_SRC)), 0, 0, 0}, {0, 0, 0, 0});")


def main() -> int:
    path = pathlib.Path(sys.argv[1])
    text = path.read_text()
    count = text.count(OLD)
    if count == 0:
        raise SystemExit(f"anchor not found in {path}")
    if "srcLoadNum * sizeof(DTYPE_SRC)" in text:
        raise SystemExit(f"{path} already patched")
    path.write_text(text.replace(OLD, NEW))
    print(f"[fix] replaced {count} GM->UB src DataCopy call(s) with byte-exact DataCopyPad in {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
