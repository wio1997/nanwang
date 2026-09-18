#!/usr/bin/env python3
"""Stage 3E / Task B+E: apply the validated largeTail repairs to the FORMAL DELIVERY SOURCE.

The script is deliberately strict: every replacement must match exactly the expected number of
times, and it refuses to run twice (idempotence guard) so the promoted diff cannot silently drift.

Repairs (see the Stage 3C / 3D reports for the root causes):

  Fix 1  op_kernel/scatter_max_v1.cpp : bare `else` -> explicit `TILING_KEY_IS(1)` branch
         (kernel entry `_1` registration; Stage 3C)
  Fix 1b op_kernel/CMakeLists.txt     : `add_ops_compile_options(ScatterMaxV1 OPTIONS --tiling_key=0,1)`
  Fix 2  op_kernel/scatter_max_v1.h   : `_idxLocal.GetValue(idxOffset + k)` -> `GetValue(k)` (x2)
         (largeTail local index lookup; Stage 3D)
  Fix 3  op_kernel/scatter_max_v1.h   : result write gains `+ n * _srcBatchNum` (x1)
         (largeTail feature-chunk offset; Stage 3D)
  Fix 4  op_kernel/scatter_max_v1.h   : index GM loads become byte-exact DataCopyPad (x4)
         (Stage 3E Task E - remove the 32B-granular index GM over-read; index semantics unchanged)

Usage:
  python3 01_apply_delivery_fixes.py --check <delivery_src_dir>
  python3 01_apply_delivery_fixes.py --apply <delivery_src_dir>
"""

from __future__ import annotations

import difflib
import pathlib
import re
import sys

KERNEL_CPP = "op_kernel/scatter_max_v1.cpp"
KERNEL_H = "op_kernel/scatter_max_v1.h"
KERNEL_CMAKE = "op_kernel/CMakeLists.txt"

ENTRY_OLD = """    } else { // TILING_KEY_LARGE_TAIL
        KernelScatterMaxV1<false> op(src, idx, res, argmax, &tiling_data, &pipe);
        op.Process();
    }"""
ENTRY_NEW = """    } else if (TILING_KEY_IS(1)) { // TILING_KEY_LARGE_TAIL
        KernelScatterMaxV1<false> op(src, idx, res, argmax, &tiling_data, &pipe);
        op.Process();
    }"""

CMAKE_ANCHOR = "add_kernels_compile()\n"
CMAKE_LINE = "\nadd_ops_compile_options(ScatterMaxV1 OPTIONS --tiling_key=0,1)\n"

INDEX_LOOKUP_OLD = "DTYPE_INDEX idxVal = _idxLocal.GetValue(idxOffset + k);"
INDEX_LOOKUP_NEW = "DTYPE_INDEX idxVal = _idxLocal.GetValue(k);"

WRITE_OLD = "DataCopyPad(_resGM[idxVal * _tailElemNum], _srcLocal,"
WRITE_NEW = "DataCopyPad(_resGM[idxVal * _tailElemNum + n * _srcBatchNum], _srcLocal,"

# Task E: the index loads carried AlignUp(count, 8) elements through the 32B-block DataCopy API,
# so the last core could read up to 28 B past the logical index tensor.  DataCopyPad transfers
# exactly blockLen bytes, which removes the over-read without touching any index value/semantics.
IDX_BATCH_OLD = "        DataCopy(_idxLocal, _idxGM[idxOffset], idxLoadNumAlgin);\n"
IDX_BATCH_NEW = (
    "        DataCopyExtParams idxCopyParams = {1, static_cast<uint32_t>(idxLoadNum * sizeof(DTYPE_INDEX)), 0, 0, 0};\n"
    "        DataCopyPad(_idxLocal, _idxGM[idxOffset], idxCopyParams, {0, 0, 0, 0});\n"
)

IDX_LEFT_OLD = "        DataCopy(_idxLocal, _idxGM[_leftSrcIdxPos], _elemNumPerBlock);\n"
IDX_LEFT_NEW = (
    "        DataCopyExtParams idxCopyParams = {1, static_cast<uint32_t>(sizeof(DTYPE_INDEX)), 0, 0, 0};\n"
    "        DataCopyPad(_idxLocal, _idxGM[_leftSrcIdxPos], idxCopyParams, {0, 0, 0, 0});\n"
)

EDITS = [
    (KERNEL_CPP, ENTRY_OLD, ENTRY_NEW, 1, "Fix 1  explicit TILING_KEY_IS(1) largeTail kernel entry"),
    (KERNEL_CMAKE, CMAKE_ANCHOR, CMAKE_ANCHOR + CMAKE_LINE, 1, "Fix 1b --tiling_key=0,1 build declaration"),
    (KERNEL_H, INDEX_LOOKUP_OLD, INDEX_LOOKUP_NEW, 2,
     "Fix 2  largeTail local index lookup GetValue(k) (ScatterMaxV1 + ScatterMaxArgmaxV1)"),
    (KERNEL_H, WRITE_OLD, WRITE_NEW, 1, "Fix 3  largeTail result write chunk offset + n*_srcBatchNum"),
    (KERNEL_H, IDX_BATCH_OLD, IDX_BATCH_NEW, 2, "Fix 4  batch index load -> byte-exact DataCopyPad (x2)"),
    (KERNEL_H, IDX_LEFT_OLD, IDX_LEFT_NEW, 2, "Fix 4  leftSrc index load -> byte-exact DataCopyPad (x2)"),
]


def main() -> int:
    if len(sys.argv) != 3 or sys.argv[1] not in ("--check", "--apply"):
        raise SystemExit(__doc__)
    mode, root = sys.argv[1], pathlib.Path(sys.argv[2])
    if not (root / KERNEL_H).is_file():
        raise SystemExit(f"not a delivery source tree: {root}")

    originals = {rel: (root / rel).read_text() for rel in {e[0] for e in EDITS}}
    texts = dict(originals)

    for rel, old, new, count, why in EDITS:
        got = texts[rel].count(old)
        if mode == "--check":
            status = "OK" if got == count else ("ALREADY-APPLIED" if got == 0 and new in texts[rel] else "ANCHOR-MISMATCH")
            print(f"[{status:>15}] {rel:32s} expect={count} found={got}  {why}")
            continue
        if got != count:
            raise SystemExit(f"ABORT: {rel}: expected {count} occurrence(s) of anchor, found {got}\n"
                             f"anchor: {old[:80]!r}")
        texts[rel] = texts[rel].replace(old, new)
        print(f"[fix] {rel:32s} x{count}  {why}")

    if mode == "--check":
        return 0

    for rel, text in texts.items():
        if text != originals[rel]:
            (root / rel).write_text(text)
    print()
    print("=== applied source diff (formal delivery source) ===")
    for rel in sorted(texts):
        if texts[rel] == originals[rel]:
            continue
        diff = difflib.unified_diff(originals[rel].splitlines(keepends=True),
                                    texts[rel].splitlines(keepends=True),
                                    fromfile=f"a/{rel}", tofile=f"b/{rel}")
        sys.stdout.writelines(diff)
    print("=== end diff ===")

    # sanity: the promoted source must contain all three mandated repairs and no stale anchors
    h = texts[KERNEL_H]
    cpp = texts[KERNEL_CPP]
    assert "TILING_KEY_IS(1)" in cpp
    assert "add_ops_compile_options(ScatterMaxV1 OPTIONS --tiling_key=0,1)" in texts[KERNEL_CMAKE]
    assert h.count("GetValue(idxOffset + k)") == 0
    assert h.count("_idxLocal.GetValue(k)") == 2
    assert "+ n * _srcBatchNum], _srcLocal," in h
    assert "DataCopy(_idxLocal," not in h, "a 32B-granular index load is still present"
    assert len(re.findall(r"DataCopyPad\(_idxLocal,", h)) == 4
    print("[ok] post-conditions verified (entry, tiling key, index lookup, write offset, "
          "4 byte-exact index loads)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
