#!/usr/bin/env python3
"""Stage 3D minimal repair #3 (probe-only copy): add the missing chunk offset on the write side.

In the large-tail path the source row is read in chunks:

    srcOffset = (idxOffset + k) * _tailElemNum + n * _srcBatchNum

but the write-back omits the chunk offset, so every chunk after the first is written to the *start*
of the destination row:

    DataCopyPad(_resGM[idxVal * _tailElemNum], ...)            # missing  + n * _srcBatchNum

For F > srcBatchNum (i.e. whenever `_srcLoop >= 2`, exactly the large-tail regime) the tail of each
row therefore stays at its initial -inf and the row head is overwritten with tail data - observed as
`max_abs_diff=inf` / 10 mismatching elements for N=40 F=48825.

Usage: python3 apply_largetail_write_offset_fix.py <probe_copy>/op_kernel/scatter_max_v1.h
"""

import pathlib
import sys

OLD = "DataCopyPad(_resGM[idxVal * _tailElemNum], _srcLocal,"
NEW = "DataCopyPad(_resGM[idxVal * _tailElemNum + n * _srcBatchNum], _srcLocal,"


def main() -> int:
    path = pathlib.Path(sys.argv[1])
    text = path.read_text()
    count = text.count(OLD)
    if count == 0:
        raise SystemExit(f"anchor not found in {path}")
    if NEW in text:
        raise SystemExit(f"{path} already patched")
    path.write_text(text.replace(OLD, NEW))
    print(f"[fix] added n*_srcBatchNum chunk offset to {count} write site(s) in {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
