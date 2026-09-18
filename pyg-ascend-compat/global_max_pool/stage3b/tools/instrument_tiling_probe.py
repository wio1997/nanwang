#!/usr/bin/env python3
"""Add a one-line stderr print of the *actual* tiling decision to a PROBE-ONLY copy of the op_host.

The delivered custom OPP / kernel is never modified: this script is applied to a throwaway copy of
the Stage 1A probe project so that Stage 3B can capture the real tiling key at runtime.

Usage: python3 instrument_tiling_probe.py <probe_copy>/op_host/scatter_max_v1.cpp
"""

import pathlib
import sys

ANCHOR = (
    "    ctx->SetBlockDim(coreNum);\n"
    "    if (idxNumPerCore != 0 && tailBatchNum == 0) {\n"
    "        ctx->SetTilingKey(TILING_KEY_LARGE_TAIL);\n"
    "    } else {\n"
    "        ctx->SetTilingKey(TILING_KEY_SMALL_TAIL);\n"
    "    }"
)

FIELDS = (
    "(unsigned long long)idxElemNum, (unsigned long long)tailElemNum, "
    "(unsigned long long)idxNumPerCore, (unsigned long long)idxBatchNum, "
    "(unsigned long long)tailBatchNum, (unsigned long long)srcBatchNum, "
    "(unsigned long long)leftIdxNum, (unsigned long long)coreNumPerTail, "
    "(unsigned long long)leftSrcBatchNum, (unsigned long long)ubSize, "
    "(unsigned long long)remainUbSize, (unsigned long long)tailSizeAlign"
)

REPLACEMENT = (
    "    ctx->SetBlockDim(coreNum);\n"
    "    if (idxNumPerCore != 0 && tailBatchNum == 0) {\n"
    "        ctx->SetTilingKey(TILING_KEY_LARGE_TAIL);\n"
    '        fprintf(stderr, "[stage3b-tiling] op=ScatterMaxV1 N=%llu F=%llu idxNumPerCore=%llu "\n'
    '                        "idxBatchNum=%llu tailBatchNum=%llu srcBatchNum=%llu leftIdxNum=%llu "\n'
    '                        "coreNumPerTail=%llu leftSrcBatchNum=%llu ubSize=%llu remainUbSize=%llu "\n'
    '                        "tailSizeAlign=%llu TILING_KEY=LARGE_TAIL(1)\\n",\n'
    "                " + FIELDS + ");\n"
    "    } else {\n"
    "        ctx->SetTilingKey(TILING_KEY_SMALL_TAIL);\n"
    '        fprintf(stderr, "[stage3b-tiling] op=ScatterMaxV1 N=%llu F=%llu idxNumPerCore=%llu "\n'
    '                        "idxBatchNum=%llu tailBatchNum=%llu srcBatchNum=%llu leftIdxNum=%llu "\n'
    '                        "coreNumPerTail=%llu leftSrcBatchNum=%llu ubSize=%llu remainUbSize=%llu "\n'
    '                        "tailSizeAlign=%llu TILING_KEY=SMALL_TAIL(0)\\n",\n'
    "                " + FIELDS + ");\n"
    "    }"
)


def main() -> int:
    path = pathlib.Path(sys.argv[1])
    text = path.read_text()
    count = text.count(ANCHOR)
    if count != 1:
        raise SystemExit(f"anchor not found exactly once in {path} (count={count})")
    if "[stage3b-tiling]" in text:
        raise SystemExit(f"{path} already instrumented")
    text = text.replace(ANCHOR, REPLACEMENT)
    if "#include <cstdio>" not in text:
        text = text.replace('#include "scatter_max_v1.h"',
                            '#include "scatter_max_v1.h"\n#include <cstdio>', 1)
    path.write_text(text)
    print(f"[instrument] patched {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
