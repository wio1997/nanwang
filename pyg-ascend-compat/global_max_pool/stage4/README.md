# Stage 4 — FP32 backward + tie-gradient

Adds a differentiable Ascend path for `torch_geometric.nn.global_max_pool` (FP32, first-order).
The forward is the frozen Stage 3 implementation; only the backward is new.

```
requires_grad=False  (or torch.no_grad())  -> frozen forward path (unchanged)
requires_grad=True   + grad enabled        -> autograd.Function (frozen forward + new backward)
```

| item | result |
|---|---|
| CPU oracle | frozen contract in `../stage4_cpu_gradient_oracle.md` (real PyG CPU, G0–G10 + batch=None) |
| contract model | exact on 300/300 randomized cases (`tests/cpu_contract_model.py`) |
| tie rule | equal split per (group, feature); zero-valued maxima add +1 to the denominator (PyTorch `include_self=False` quirk) |
| NaN | backward is `nan` for every row of a NaN cell (matched) |
| backward matrix | 35/35 PASS vs CPU oracle, 31/32 gradient cases bit-exact, max 1 ULP |
| real PyG E2E | 11/11 PASS, `ascend_calls=10 original_calls=0 autograd_calls=10` |
| profiler | ScatterMaxV1 forward AI_VECTOR_CORE; all backward primitives device; AI_CPU=0; scatter_reduce NOT CALLED |
| forward regressions | 3A 34/34 · 3B 45/45 · 3D 9/9 · 3E 13/13 · Stage 6 demo PASS · 6 20/20 · Stage 2 16/16 |
| first-order only | gradgrad not supported (documented) |

Reproduce (inside the container, after sourcing the formal OPP environment):

```bash
export STAGE6_OPP=/root/zyg/build/scattermax_runtime_opp/vendors/customize
source /root/zyg/global_max_pool/stage6/env.sh

python3 tests/cpu_gradient_oracle.py          # CPU oracle (real PyG, CPU)
python3 tests/cpu_tie_quirk_probe.py          # self-slot quirk experiment
python3 tests/cpu_contract_model.py           # 300-case model validation
python3 tests/run_stage4_backward_tests.py    # 35-case Ascend vs CPU matrix
python3 tests/run_stage4_pyg_e2e.py           # real PyG API backward E2E
bash    tools/run_stage4_profiler.sh          # P-BWD1..4 (forward+loss+backward)
python3 tools/stage4_primitive_audit.py       # primitive/fallback audit
bash    tools/run_stage4_regressions.sh       # full regression sweep
```

Full evidence: `../stage4_fp32_backward.md`.
