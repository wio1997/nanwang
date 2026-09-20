#!/usr/bin/env python3
"""Extract dataset_cascades.zip into the layout the PowerGraph loader expects.

The PowerGrid loader resolves::

    raw_dir       = <data_root>/<name>/<name>/raw
    processed_dir = <data_root>/<name>/<name>/processed_b

so ``ieee24/ieee24/raw/Bf.mat`` etc. must sit directly under the data root.

``Ef_nc.mat`` is skipped on purpose: the loader's ``raw_file_names`` does not
list it and nothing reads it, and it is ~1.1 GB of the 2.96 GB archive.

Usage:
    python3 extract_powergraph_data.py [<zip>] [<data_root>]

Both arguments fall back to the environment (``POWERGRAPH_DATA_ARCHIVE`` /
``POWERGRAPH_DATA_ROOT``) and then to package-relative defaults, so the script
can be run from anywhere:

    python3 extract_powergraph_data.py
"""

from __future__ import annotations

import os
import sys
import time
import zipfile


_VALIDATION_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_DEFAULT_ARCHIVE = os.path.join(
    _VALIDATION_ROOT, "data_download", "dataset_cascades.zip")
_DEFAULT_DATA_ROOT = os.path.join(_VALIDATION_ROOT, "data")


def main() -> int:
    zip_path = (sys.argv[1] if len(sys.argv) > 1
                else os.environ.get("POWERGRAPH_DATA_ARCHIVE", _DEFAULT_ARCHIVE))
    data_root = (sys.argv[2] if len(sys.argv) > 2
                 else os.environ.get("POWERGRAPH_DATA_ROOT", _DEFAULT_DATA_ROOT))
    if not os.path.isfile(zip_path):
        print(f"ERROR: archive not found: {zip_path}", file=sys.stderr)
        print("       run scripts/fetch_powergraph_data.sh first, or pass the "
              "archive path / set POWERGRAPH_DATA_ARCHIVE", file=sys.stderr)
        return 2
    os.makedirs(data_root, exist_ok=True)

    with zipfile.ZipFile(zip_path) as zf:
        members = [m for m in zf.infolist() if not m.filename.endswith("Ef_nc.mat")]
        skipped = len(zf.infolist()) - len(members)
        print(f"extracting {len(members)} members into {data_root} "
              f"(skipped {skipped} Ef_nc.mat)")
        t0 = time.time()
        for m in members:
            zf.extract(m, data_root)
        print(f"done in {time.time() - t0:.1f}s")

    for name in ("ieee24", "ieee39", "ieee118", "uk"):
        raw = os.path.join(data_root, name, name, "raw")
        print(f"{name}: raw_dir={'OK' if os.path.isdir(raw) else 'MISSING'} {raw}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
