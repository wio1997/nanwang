# Stage 5 — FP16 / BF16 forward + first-order backward

The delivered `ScatterMaxV1` OPP is FP32-only (measured: `aclnnScatterMaxV1GetWorkspaceSize`
returns `161002` for fp16/bf16). Stage 5 therefore adds an audited **device cast chain** around the
frozen fp32 kernel plus a dtype-exact tie-gradient backward:

```
x (fp16/bf16) --Cast--> fp32 --frozen ScatterMaxV1--> fp32 --Cast--> out (fp16/bf16)
backward: winner mask (dtype) * (grad_out[batch] / cast_to_dtype(exact_fp32_count))
```

| item | fp16 | bf16 |
|---|---|---|
| CPU oracle (real PyG, CPU) | 26/26 OK, out/grad dtype = input | same |
| count arithmetic (measured) | exact integer → rounded to dtype (21/21) | (22/22) |
| dtype division vs CPU | bit-exact (max ULP 0) | bit-exact |
| semantics matrix (31 cases) | 31/31 PASS, bit-exact | 31/31 PASS, bit-exact |
| real PyG E2E | PASS (out/grad dtype, counters, fallback NONE) | PASS |
| profiler (4 cases each) | ScatterMaxV1 AI_VECTOR_CORE, AI_CPU 0, no scatter_reduce | same |
| FP32 regression | 3A 34/34 · 3B 45/45 · 3D 9/9 · 3E 13/13 · Stage 6 20/20 · Stage 4 35/35+11/11 | unchanged |

```bash
export STAGE6_OPP=/root/zyg/build/scattermax_runtime_opp/vendors/customize
source /root/zyg/global_max_pool/stage6/env.sh
python3 tests/cpu_dtype_oracle.py            # CPU oracle, both dtypes
python3 tests/run_stage5_dtype_tests.py      # 62-case Ascend vs CPU matrix
python3 tests/run_stage5_pyg_e2e.py          # real PyG API E2E
bash    tools/run_stage5_profiler.sh         # 8 profiler runs (forward+loss+backward)
bash    tools/run_stage5_regressions.sh      # full regression incl. all FP32 suites
```

Full evidence: `../stage5_fp16_bf16.md`; contract: `../stage5_cpu_dtype_oracle.md`.
