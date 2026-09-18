#!/usr/bin/env python3
"""Remove the ScatterMaxArgmaxV1 OpDef block from a COPY of the op_host source.

Used only for the Stage 1B forward-only build probe; the DrivingSDK sources and the
Stage 1A probe are never modified.
"""
import pathlib
import sys


def main() -> int:
    path = pathlib.Path(sys.argv[1])
    text = path.read_text()

    marker = "class ScatterMaxArgmaxV1 : public OpDef {"
    idx = text.index(marker)
    ns_start = text.rindex("namespace ops {", 0, idx)
    add_marker = "OP_ADD(ScatterMaxArgmaxV1);"
    add_idx = text.index(add_marker, idx)
    ns_end = text.index("} // namespace ops", add_idx) + len("} // namespace ops\n")

    removed = text[ns_start:ns_end]
    path.write_text(text[:ns_start] + text[ns_end:])
    print(f"[strip] removed {len(removed)} bytes from {path}")
    print(f"[strip] remaining OP_ADD: {text[ns_start + len(removed):].count('OP_ADD')} declaration(s) after cut")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
