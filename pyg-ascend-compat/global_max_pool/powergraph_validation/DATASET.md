# PowerGraph dataset provenance and acquisition

This file records exactly which upstream data the validation used, how to obtain
it, and how the upstream project describes its own license. **No PowerGraph data
is committed to this repository** — the extracted dataset is ~2.75 GiB.

---

## 1. Source

```text
PowerGraph-Graph repository : https://github.com/PowerGraph-Datasets/PowerGraph-Graph
PowerGraph-Graph commit used: eb100a2fd836bb8b6bd2d0b799af9c615eac8cb6
```

The repository is used **read-only**: this validation imports the unmodified
`code/dataset/powergrid.py` (`PowerGrid`) loader from that checkout and never
writes to it.

The datasets themselves are not stored in the git repository; the upstream
README points to a figshare object.

## 2. Data file actually used

```text
object : PowerGraph (figshare article 22820534)
file   : dataset_cascades.zip
file id: 46619158               <- the file the upstream README links to
URL    : https://figshare.com/ndownloader/files/46619158
size   : 61,628,977 bytes
md5    : 70b677416d2f377ccfee9f51d8369867
uncompressed: 2,958,249,040 bytes (2.75 GiB)
DOI    : 10.6084/m9.figshare.22820534  (article page: https://figshare.com/articles/dataset/PowerGraph/22820534)
```

Cross-check performed during the original validation: the newer article version
(v5) provides `dataset_cascades.zip` as file id `50083479`; that archive was also
downloaded and its md5 matched the value published by figshare
(`d4d144b9e720a760e1e077a31f34802d`). Both archives contain the same raw files
with identical sizes; v5 only adds one extra top-level directory. The v3 file
linked from the upstream README (`46619158`) is the one used for the reported
numbers.

## 3. Datasets

| dataset | graphs | nodes/graph | F | edges/graph (directed) |
|---|---|---|---|---|
| `ieee24` | 21,500 | 24 | 3 | 68–74 |
| `ieee39` | 28,000 | 39 | 3 | 86–90 |
| `ieee118` | 122,500 | 118 | 3 | 362–370 |
| `uk` | 64,000 | 29 | 3 | 190–196 |

Per-graph node count is constant per dataset; the edge count varies because each
graph has 1–5 tripped branches that the loader deletes before concatenating the
forward and reversed edges. Node features are always `float32 [N, 3]` (net
active power, net apparent power, voltage magnitude), so `F = 3` everywhere.
Machine-readable audit: `results/phase_a_raw_audit.json`.

## 4. Expected directory layout

The upstream loader resolves `raw_dir = <root>/<name>/<name>/raw`, where
`<root>` is `POWERGRAPH_DATA_ROOT`. After extraction:

```text
$POWERGRAPH_DATA_ROOT/
├── ieee24/ieee24/raw/{Bf.mat,blist.mat,Ef.mat,exp.mat,of_bi.mat,of_mc.mat,of_reg.mat}
├── ieee39/ieee39/raw/{...}
├── ieee118/ieee118/raw/{...}
└── uk/uk/raw/{...}
```

`Ef_nc.mat` present in the archive is intentionally **not** extracted: it is not
listed in the loader's `raw_file_names`, nothing reads it, and it accounts for
~1.1 GB of the 2.96 GB archive.

On first use the loader writes
`$POWERGRAPH_DATA_ROOT/<name>/<name>/processed_b/data.pt` (ieee24 ≈ 65 MB,
ieee39 ≈ 105 MB, ieee118 ≈ 1.81 GB, uk ≈ 0.5 GB). Those processed files are also
not committed.

## 5. Acquisition procedure

```bash
cd pyg-ascend-compat/global_max_pool/powergraph_validation

# 1) upstream loader checkout (read only)
git clone https://github.com/PowerGraph-Datasets/PowerGraph-Graph.git \
    upstream/PowerGraph-Graph

# 2) dataset archive (official figshare URL, md5 verified)
bash scripts/fetch_powergraph_data.sh

# 3) extraction into POWERGRAPH_DATA_ROOT
python3 scripts/extract_powergraph_data.py
```

Overrides understood by `fetch_powergraph_data.sh`:

| variable | meaning |
|---|---|
| `POWERGRAPH_DATA_URL` | archive URL (default: the official figshare ndownloader URL above) |
| `POWERGRAPH_DATA_ARCHIVE` | output zip path (default `<package>/data_download/dataset_cascades.zip`) |
| `POWERGRAPH_DATA_MD5` | expected md5 (default `70b677416d2f377ccfee9f51d8369867`; `skip` disables) |
| `POWERGRAPH_FIGSHARE_PROXY` | optional HTTP proxy used only if the direct download fails |
| `POWERGRAPH_FIGSHARE_FILE_ID` | figshare file id used by the proxy fallback (default `46619158`) |

### 5.1 Note on the original validation network

In the **original validation environment** `figshare.com` answered HTTP 403 for
every path (article page, `api.figshare.com`, `ndownloader.figshare.com`, both
IPv4 and IPv6). The data was obtained by asking figshare for its 302 redirect
through an HTTP proxy and then downloading the payload **directly from
`s3-eu-west-1.amazonaws.com/pfigshare-u-files/...`**; no proxy was used for the
payload transfer.

The packaged script reproduces that only as a fallback: it first tries the
official URL directly, and uses `POWERGRAPH_FIGSHARE_PROXY` (when set) only if
that fails. It never relies on a stored presigned URL — those expire within
seconds — and it re-derives a fresh one from the official figshare redirect on
every invocation. The expected md5 is verified in both paths.

## 6. License / attribution

PowerGraph is a third-party dataset. It is **not** redistributed here; only
derived benchmark numbers are stored. The upstream project does not ship a
standalone `LICENSE` file, and the statements found upstream are **not fully
consistent**, so both are quoted verbatim rather than resolved here:

| location | statement |
|---|---|
| `PowerGraph-Graph/README.md` (`## License`) | *"This work is licensed under a CC BY 4.0 license."* |
| `PowerGraph-Graph/code/dataset/powergrid.py` (file header) | *"PowerGrid dataset is licensed under a CC BY-SA 4.0 license."* |
| figshare record for article 22820534 (as returned by `api.figshare.com`) | license name `CC BY 4.0`, <https://creativecommons.org/licenses/by/4.0/> |

Reviewers who intend to redistribute the dataset itself should consult the
upstream repository and the figshare record directly:

* <https://github.com/PowerGraph-Datasets/PowerGraph-Graph> (see `README.md` and `code/dataset/powergrid.py`)
* <https://figshare.com/articles/dataset/PowerGraph/22820534>

This repository does not adjudicate between those statements.

## 7. What is committed vs regenerated

| item | committed? |
|---|---|
| raw `.mat` files (2.75 GiB uncompressed) | **no** — download with `fetch_powergraph_data.sh` |
| `dataset_cascades.zip` archive | **no** — git-ignored |
| PyG `processed_b/data.pt` | **no** — regenerated by the loader |
| upstream `PowerGraph-Graph` checkout | **no** — clone into `upstream/` (git-ignored) |
| benchmark result CSVs / JSONs | **yes** — under `results/` |
| parsed profiler gate records + per-case msprof CSVs | **yes** — under `evidence/profiler/` |
| raw msprof output directories | **no** — regenerate with `run_profiles.sh` |
