#!/usr/bin/env python3
"""Stage 3D minimal repair #2 (probe-only copy): fix the index lookup in the large-tail path.

`batchProcess()` copies the index block starting at GM offset `idxOffset` into `_idxLocal[0..]`, so
the k-th entry of that block is `_idxLocal.GetValue(k)`. The large-tail `elemWiseBatchProcess()`
instead reads `_idxLocal.GetValue(idxOffset + k)` - a *local* read past the small index buffer
(`idxBatchNumAlign` = 8 entries for N=40) which returns garbage and is then used as the destination
row (`_resGM[idxVal * _tailElemNum]`), i.e. a wild GM write. For N=40 core i this reads
`GetValue(i)` (i up to 39) from an 8-entry buffer.

The small-tail `tailWisebatchProcess()` uses `GetValue(k * _tailBatchNum + n)` and is correct.

Usage: python3 apply_largetail_index_fix.py <probe_copy>/op_kernel/scatter_max_v1.h
"""

import pathlib
import sys

OLD = "DTYPE_INDEX idxVal = _idxLocal.GetValue(idxOffset + k);"
NEW = "DTYPE_INDEX idxVal = _idxLocal.GetValue(k);"


def main() -> int:
    path = pathlib.Path(sys.argv[1])
    text = path.read_text()
    count = text.count(OLD)
    if count == 0:
        raise SystemExit(f"anchor not found in {path}")
    if NEW in text:
        raise SystemExit(f"{path} already patched")
    path.write_text(text.replace(OLD, NEW))
    print(f"[fix] replaced {count} index lookup(s) in {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
