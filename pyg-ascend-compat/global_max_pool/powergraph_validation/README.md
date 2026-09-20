# PowerGraph real-workload validation — `global_max_pool`

Real-workload validation and single-operator performance measurement of the
**frozen** PyG Ascend `global_max_pool` implementation on real power-grid graph
data from [`PowerGraph-Datasets/PowerGraph-Graph`](https://github.com/PowerGraph-Datasets/PowerGraph-Graph).

> **This validation does not train a GNN.**
> PowerGraph is used **only as a real PyG graph data source**: graphs are loaded
> through the unmodified upstream `PowerGrid` `InMemoryDataset`, batched with
> `torch_geometric.loader.DataLoader`, and only `batch.x` / `batch.batch` are
> handed to `global_max_pool`. No convolution, no `Linear`, no optimizer, no
> accuracy evaluation.

Full technical report: [`POWERGRAPH_GLOBAL_MAX_POOL_VALIDATION.md`](POWERGRAPH_GLOBAL_MAX_POOL_VALIDATION.md)

---

## 1. What this validation is

It answers: *does the frozen Ascend `global_max_pool` work on real PyG graph
workloads, is it correct, and does it actually execute on the NPU?* The accepted
criteria for this round are functional and evidence-based, **not** speed-up
based:

| accepted | value |
|---|---|
| real data / real PyG batch | PASS |
| FP32 / FP16 / BF16 | PASS |
| forward | **96 / 96 PASS** |
| forward + first-order backward | **48 / 48 PASS** |
| CPU PyG oracle agreement | PASS (FP32 bit-exact) |
| `ScatterMaxV1` core type | `AI_VECTOR_CORE` |
| AI_CPU tasks | 0 |
| compat host fallback | NONE |
| compat `aten::scatter_reduce` | 0 (not called) |

Speed-up versus the upstream PyG fallback is recorded as workload
characterization only and is **not** a PASS/FAIL condition. PowerGraph's node
feature dimension is `F = 3`, i.e. an extremely small feature workload where the
adapter's fixed per-call cost dominates the end-to-end latency; that number does
not represent the ScatterMaxV1 kernel's capability at larger `F`.

## 2. Datasets measured

| dataset | graphs | nodes/graph | F | edges/graph (directed) |
|---|---|---|---|---|
| `ieee24` | 21,500 | 24 | 3 | 68–74 |
| `ieee39` | 28,000 | 39 | 3 | 86–90 |
| `ieee118` | 122,500 | 118 | 3 | 362–370 |
| `uk` | 64,000 | 29 | 3 | 190–196 |

Nodes per graph are fixed per dataset; the edge count varies because 1–5 branches
per graph are tripped and removed by the loader before the forward + reversed
edges are concatenated.

## 3. How to get the data

Raw data is **not** committed (2.75 GiB uncompressed). See [`DATASET.md`](DATASET.md)
for source, checksum and license attribution.

```bash
# upstream loader checkout (read only) + dataset archive + extraction
cd pyg-ascend-compat/global_max_pool/powergraph_validation

git clone https://github.com/PowerGraph-Datasets/PowerGraph-Graph.git \
    upstream/PowerGraph-Graph

# figshare is reachable from most networks; on the original validation network it
# returned HTTP 403, so a proxy fallback is supported (see DATASET.md)
bash scripts/fetch_powergraph_data.sh
python3 scripts/extract_powergraph_data.py
```

Expected layout after extraction:

```text
$POWERGRAPH_DATA_ROOT/<name>/<name>/raw/{Bf,blist,Ef,exp,of_bi,of_mc,of_reg}.mat
# e.g. data/ieee24/ieee24/raw/Bf.mat
```

On first use the unmodified `PowerGrid` loader writes
`$POWERGRAPH_DATA_ROOT/<name>/<name>/processed_b/data.pt`.

## 4. How to prepare the environment

```bash
# inside the CANN container
source scripts/bench_env.sh
```

`bench_env.sh` derives the repo root from its own location and exports the
frozen compat package, the adapter/autograd/stage5 module paths, the custom OPP
and the benchmark roots. Everything is overridable:

| variable | default |
|---|---|
| `GLOBAL_MAX_POOL_OPP` | this server's `scattermax_runtime_opp/vendors/customize` |
| `SCATTERMAXV1_BRIDGE` | this server's `stage2_ext/scattermaxv1_bridge.so` |
| `POWERGRAPH_UPSTREAM_DIR` | `<package>/upstream/PowerGraph-Graph` |
| `POWERGRAPH_DATA_ROOT` | `<package>/data` |
| `POWERGRAPH_RESULTS_ROOT` | `<package>/results` |
| `POWERGRAPH_PROFILE_ROOT` | `<package>/profiler_runs` |
| `ASCEND_RT_VISIBLE_DEVICES` | `0` |

Extra Python packages needed on top of the frozen Ascend environment are listed
in [`requirements-validation.txt`](requirements-validation.txt) —
**do not install PowerGraph's upstream `requirements.txt`** (it pins old CUDA
PyTorch/PyG).

## 5. How to run the forward benchmark

```bash
source scripts/bench_env.sh

python3 scripts/bench_forward.py \
    --datasets ieee24,ieee39,ieee118,uk \
    --batch-sizes 1,8,32,128 \
    --dtypes fp32,fp16,bf16 \
    --paths compat_ascend,original_pyg \
    --warmup 30 --iters 200 --tag repo_validation
```

`--paths compat_ascend` measures the frozen Ascend path; `original_pyg` measures
the upstream PyG implementation captured *before* `pyg_ascend_compat.enable()`
(which on this torch_npu runs on the **host CPU**).

## 6. How to run the backward benchmark

```bash
python3 scripts/bench_backward.py \
    --datasets ieee24,ieee39,ieee118,uk --batch-sizes 1,8,32,128 \
    --dtypes fp32,fp16,bf16 --paths compat_ascend \
    --warmup 30 --iters 200 --tag repo_validation
```

Each iteration is `x.grad = None; out = global_max_pool(x, batch); out.sum().backward()`.

## 7. How to run the profiler

```bash
source scripts/bench_env.sh
bash scripts/run_profiles.sh \
  "ieee24_b128_fp32 ieee24 128 fp32 compat_ascend" \
  "ieee24_b128_fp16 ieee24 128 fp16 compat_ascend" \
  "ieee24_b128_bf16 ieee24 128 bf16 compat_ascend" \
  "ieee118_b128_fp32 ieee118 128 fp32 compat_ascend" \
  "uk_b128_fp32 uk 128 fp32 compat_ascend" \
  "ieee24_b128_fp32_origpyg ieee24 128 fp32 original_pyg"
cat "${POWERGRAPH_PROFILE_ROOT:-profiler_runs}/profiler_summary.txt"
```

Gate per case: `ScatterMaxV1 -> AI_VECTOR_CORE: PASS`, `AI_CPU_task_types=[]`,
`aten::scatter_reduce occurrences = 0`, no fallback markers, and compat counters
`ascend_calls == total_calls` with `original_calls == 0`.

Everything in one go (syntax check → smoke → forward → forward+backward →
profiler → summaries):

```bash
bash scripts/run_validation.sh              # full run
bash scripts/run_validation.sh --smoke-only # fast post-migration check
```

## 8. Where the results are

```text
results/
  forward_phaseB_ieee24.csv                forward, ieee24                (24 rows)
  forward_phaseC.csv                       forward, ieee39/ieee118/uk     (72 rows)
  forward_backward_phaseC.csv              forward + first-order backward (48 rows)
  forward_per_iter_*.csv                   per-iteration latency (P50/P95/P99 source)
  forward_backward_per_iter_phaseC.csv
  performance_summary.csv                  derived: compat vs original PyG per cell
  profiler_summary.csv / profiler_summary.txt
  phase_a_raw_audit.json                   raw .mat / per-graph node+edge audit
  phase_a_loader.json, phase_a_loader_rest.json
  overhead_breakdown.json                  sync floor / trivial op / raw kernel
  adapter_step_breakdown.json              per-step cost of the adapter's device work
  baseline_stability.json                  repeatability of the host-CPU fallback
evidence/
  frozen_provenance.txt                    frozen SHAs + sha256 of every source
  environment/pip_freeze_{before,after}.txt
  profiler/<case>.gate.json                parsed profiler gate record
  profiler/<case>/mindstudio_profiler_output/{op_summary,op_statistic,api_statistic,task_time}.csv
```

Raw msprof directories (hundreds of MB of sqlite/JSON) are **not** committed;
regenerate them with `scripts/run_profiles.sh`.

## 9. Where the full technical report is

[`POWERGRAPH_GLOBAL_MAX_POOL_VALIDATION.md`](POWERGRAPH_GLOBAL_MAX_POOL_VALIDATION.md)
— positioning, dataset provenance, exact benchmark method, correctness sanity,
profiler evidence, latency interpretation, baseline comparison, limitations and
the reviewer reproduction procedure.

## 10. Script map

| script | purpose |
|---|---|
| `scripts/bench_env.sh` | environment bootstrap (OPP, adapter paths, roots, NPU pin) |
| `scripts/pg_env.py` | compat bootstrap before `torch_geometric` is imported + loader shims |
| `scripts/pg_dataset.py` | unmodified `PowerGrid` loader wrapper |
| `scripts/bench_forward.py` | forward benchmark (compat + original PyG) |
| `scripts/bench_backward.py` | forward + first-order backward benchmark |
| `scripts/profile_app.py` | msprof application for one representative case |
| `scripts/parse_profile.py` | msprof `PROF_*` parser → gate JSON |
| `scripts/run_profiles.sh` | profiler driver + gate checks |
| `scripts/run_validation.sh` | end-to-end driver |
| `scripts/fetch_powergraph_data.sh` | dataset download (official source + checksum) |
| `scripts/extract_powergraph_data.py` | extraction into the loader's expected layout |
| `scripts/analysis/phase_a_audit.py` | raw `.mat` audit |
| `scripts/analysis/phase_a_loader_check.py` | processed-dataset / loader verification |
| `scripts/analysis/smoke_compat.py` | import-order + 3-dtype smoke test |
| `scripts/analysis/probe_overhead.py` | latency decomposition vs sync floor / raw kernel |
| `scripts/analysis/probe_adapter_steps.py` | per-step adapter cost |
| `scripts/analysis/probe_baseline_stability.py` | host-CPU fallback repeatability |
| `scripts/analysis/make_summaries.py` | derives `performance_summary.csv`, `profiler_summary.csv` |
| `scripts/analysis/make_report.py` | regenerates the detailed evidence report |

All benchmark scripts keep the mandatory import order:

```python
import pyg_ascend_compat
pyg_ascend_compat.enable()
from torch_geometric.nn import global_max_pool
```
