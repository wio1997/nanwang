# Stage 4 — FP32 backward + tie-gradient (evidence)

Date: 2026-09-18 · container `wio-pyg-cann851-pyg280` · CANN 8.5.1 · torch 2.9.0+cpu ·
torch_npu 2.9.0 · PyG 2.8.0.post1 · Ascend 910B3 (device 0) · formal delivery OPP.

Branch `feat/global-max-pool-scattermax-zyg`, history
`5816ef6 → 1a8df65 → 382301e → 028112b → 430bf81 → d506208 → <Stage 4 commits>`.

**Verdict: STAGE4 = PASS.**

---

## 1. What was added

```
torch_geometric.nn.global_max_pool   (x.requires_grad=True, NPU, fp32)
  -> pyg_ascend_compat dispatch                     (routes to the differentiable entry)
  -> global_max_pool_ascend_autograd                (NEW: autograd.Function)
       forward  = the frozen Stage 2/3A adapter      (unchanged; aclnnScatterMaxV1 + AI_VECTOR_CORE)
       backward = NEW tie-gradient implementation    (device primitives only)
  -> x.grad
```

The forward is *called*, never re-implemented: `GlobalMaxPoolAscendFunction.forward` invokes the
frozen `global_max_pool_ascend` adapter, so every Stage 3 guarantee (guards, padding/crop, leftSrc,
largeTail, empty-group/`-inf` semantics) is inherited unchanged. Inference
(`requires_grad=False`, or `torch.no_grad()`) keeps using exactly the old code path.

## 2. CPU oracle → frozen gradient contract

`stage4_cpu_gradient_oracle.md` is the measured contract (real PyG 2.8.0.post1 + torch 2.9.0 on
CPU, G0–G10 + `batch=None`, plus a controlled experiment that pins the PyTorch
`include_self=False` backward quirk). Executable form:

```
out[g,f]   = max over { x[i,f] : batch[i] == g }        empty group -> 0
winner     = (x == out[batch])
count[g,f] = sum_i winner[i,f] over the group + (1 if out[g,f] == 0 else 0)
grad_x     = winner * (grad_out[batch] / count[batch])  fp32
```

The model was validated against the real PyG API on **300/300 randomized cases** (ties, ±0, `-inf`,
NaN, multi-group) with exact forward and gradient agreement (`logs/stage4/cpu_contract_model.log`).

Key measured facts:

* ties split **equally**, per `(group, feature)`, in fp32 (`1/3 → 0.3333333432674408`);
* a **zero-valued maximum** gets an extra denominator (+1): `[[0],[-0]] → 1/3, 1/3`,
  `[[0]] → 1/2`, single zero row with upstream 11 → `5.5`. Controlled experiment (same data,
  destination initialised to `-1000`) gives `1/2` — the term is "self slot equals the result",
  the documented PyTorch `scatter_reduce(..., include_self=False)` backward quirk;
* **empty groups** are `0` in the forward, receive no input gradient, and upstream on them is
  ignored;
* a non-empty `-inf` group stays `-inf` and its gradient splits among the `-inf` rows;
* **NaN** is contagious in the backward: every row of a NaN `(group, feature)` cell gets `nan`
  (and also with zero upstream) — matched, not carved out;
* `batch=None` is **not** the scatter path at all (PyG: `x.max(dim=-2)`); see §4.

## 3. Backward design

```
out_rows = index_select(out, 0, batch)          # GatherV3
winner   = (x == out_rows)                      # Equal
w        = winner.to(float32)                   # Cast
count    = zeros(S,F).index_add_(0, batch, w)   # InplaceIndexAdd
count   += (out == 0).to(float32)               # the include_self=False quirk
scaled   = grad_out / count                     # RealDiv (fp32)
grad_x   = w * index_select(scaled, 0, batch)   # GatherV3 + Mul
grad_x.masked_fill_(count[batch] == 0, nan)     # MaskedFill — NaN contract, device-side
```

* no `[N, F]` int64 index is materialised (`index_add_`/`index_select` take the `[N]` batch);
* no host sync in the hot path (the diagnostics that read values are behind
  `STAGE4_BACKWARD_DEBUG=1`);
* memory: `[N,F]` fp32 mask + `[N,F]` fp32 grad + `[S,F]` temporaries — the minimum for this
  formula.

## 4. `batch=None`

PyG 2.8.0.post1 source (`torch_geometric/nn/pool/glob.py`): `if batch is None: return x.max(dim=dim,
keepdim=x.dim() <= 2)[0]`, i.e. a plain `x.max` (dim `-2` for 2-D, `-1` for 1-D).

Measured (`logs/stage4/batch_none_probe.log`):

* CPU vs NPU forward **and** backward are identical, including ties (`[[3,1],[3,2],[1,3]]` →
  gradient to the *first* maximal index, `[[1,0],[0,0],[0,1]]`) and `-inf`/1-D inputs;
* the NPU path is native (0 `npu_cpu_fallback` warnings);
* the compat wrapper keeps delegating (`ascend_calls=0`, `original_calls=1`).

**Conclusion: `batch=None` is explicitly outside the Stage 4 custom-backward scope** — different
operator, already correct and device-native; nothing was changed.

## 5. NPU primitive audit

`tools/stage4_primitive_audit.py` over the four P-BWD profiler runs
(`logs/stage4/stage4_primitive_audit.txt`): every operator in forward+loss+backward runs on
`AI_VECTOR_CORE` or `MIX_AIV`; **no CPU task type anywhere**.

| backward primitive | profiler op | core |
|---|---|---|
| `out[batch]`, `scaled[batch]` | `GatherV3` | AI_VECTOR_CORE |
| `x == out[batch]`, `out == 0` | `Equal` | AI_VECTOR_CORE |
| mask cast | `Cast` | AI_VECTOR_CORE |
| winner count | `InplaceIndexAdd` | AI_VECTOR_CORE |
| `grad_out / count` | `RealDiv` | AI_VECTOR_CORE |
| `winner * …` | `Mul` | AI_VECTOR_CORE |
| NaN fixup | `MaskedFill` | AI_VECTOR_CORE |

No `aten::scatter_reduce` (0 occurrences in every api_statistic), no host fallback.

## 6. Correctness matrix (`run_stage4_backward_tests.py`, 35/35 PASS)

Compared element-wise against the real PyG API on CPU (NaN-aware, ±0-aware) + gradient invariants:

| case | result |
|---|---|
| B1 unique max | PASS (bit-exact) |
| B2 two-way tie | PASS (bit-exact) |
| B3 three-way tie | PASS (bit-exact) |
| B4 per-feature different ties | PASS (bit-exact) |
| B5/B5b weighted upstream (incl. zero/negative) | PASS (bit-exact) |
| B6 multiple groups + repeated index | PASS (bit-exact) |
| B7 explicit size + empty groups | PASS (bit-exact) |
| B8 negative-only | PASS (bit-exact) |
| B9a/b/c true `-inf` (all `-inf`, mixed, vs empty) | PASS (bit-exact) |
| B10a/b/c ±0 and zero-value maxima | PASS (bit-exact) |
| B11a–e NaN (1, 2, NaN+inf, zero upstream) and `+inf` ties | PASS (bit-exact) |
| B_F{1,7,8,9,17,33} non-aligned F | PASS (bit-exact) |
| B_N39 / B_N40 / B_N41 (leftSrc) | PASS (bit-exact) |
| B_N4097 | PASS (244 cells 1 ULP, see §10) |
| B_LT_N40_F48825 / B_LT_N41_F48825 (largeTail) | PASS (bit-exact) |
| B12/B13/B14 graph, `no_grad`, inference | PASS |

`gradient-bitwise: 31/32 cases bit-exact, max ULP over all cases = 1`.

Invariants checked in every case: non-winner gradients are exactly `0`; the sum of the winner
gradients equals `grad_out` (or `grad_out·n/(n+1)` for zero-max cells) within fp32 rounding; empty
groups produce no input gradient.

## 7. Real PyG API end-to-end (`run_stage4_pyg_e2e.py`, 11/11 PASS)

```
import pyg_ascend_compat; pyg_ascend_compat.enable()
from torch_geometric.nn import global_max_pool
x.requires_grad_(True); out = global_max_pool(x, batch, size); (out*up).sum().backward()
```

E1 unique, E2 2-way tie, E3 per-feature ties, E4 multi-group, E5 explicit size + empty groups,
E6 ±0, E7 `-inf`, E8 NaN, E9 N=41 F=33 (leftSrc + non-aligned), E10 largeTail N=40 F=48825 —
forward **and** `x.grad` equal to the CPU oracle, `fallback=0`.

```
compat counters: total=10 ascend_calls=10 original_calls=0 autograd_calls=10 requires_grad_rejected=0
```

## 8. Profiler gate (`tools/run_stage4_profiler.sh`, 4/4 PASS)

| case | shape | ScatterMaxV1 forward | backward primitives | AI_CPU | scatter_reduce |
|---|---|---|---|---|---|
| P-BWD1 unique max | N=64 F=8 S=4 | AI_VECTOR_CORE | device | 0 | 0 |
| P-BWD2 tie + leftSrc | N=41 F=33 S=8 | AI_VECTOR_CORE | device | 0 | 0 |
| P-BWD3 non-aligned | N=64 F=17 S=4 | AI_VECTOR_CORE | device | 0 | 0 |
| P-BWD4 largeTail | N=40 F=48825 S=8 | AI_VECTOR_CORE | device | 0 | 0 |

Each run profiles forward + loss + backward; the app banner (`STAGE4_PROFILE … OK`) proves the
backward executed (grad shape and grad sum recorded).

## 9. Forward regressions (unchanged inference)

| suite | result |
|---|---|
| Stage 3A | **34/34 PASS** |
| Stage 3B | **45/45 PASS** |
| Stage 3D | **9/9 PASS** |
| Stage 3E largeTail | **13/13 PASS** |
| Stage 6 demo | **PASS** |
| Stage 6 matrix | **20/20 PASS** (`BEFORE=YES AFTER=NO`) |
| Stage 2 adapter suite (historical forward-only contract) | **16/16 PASS** |

The Stage 2 adapter still rejects `requires_grad=True` exactly as before; only the compat wrapper
and the new Stage 4 module provide the differentiable entry, so no forward behaviour changed.

## 10. Numerical note (the only deviation from bit-exactness)

The Ascend vector unit has **no correctly-rounded fp32 divide, and no fp64** (`torch_npu` logs
*"Device do not support double dtype now, dtype cast replace with float"*). Measured on the device:

* `1.0/122.0` → `0.008196721784770489` on NPU vs `0.008196720853447914` (correctly rounded) on CPU;
* over 10 000 random `(grad, count)` pairs: **max 1 ULP**, 6.3 % of cells off by 1 ULP;
* 256 cases `1/n, n=1..256`: max 1 ULP.

Consequences: every non-tie gradient cell is bit-exact (no division happens); tie cells may differ
from the CPU oracle by 1 ULP in the last bit, with identical winner masks, counts, zero structure
and NaN pattern. In the 35-case matrix the only case affected is `B_N4097` (a 122-way tie), and in
the corpus 31/32 gradient comparisons are bit-exact.

## 11. Scope limits / remaining risks

* **first-order only**: `torch.autograd.grad(..., create_graph=True)` / double backward is not
  implemented (the backward uses non-differentiable gather/scatter primitives); calling it raises
  or produces no second-order graph. Not a Stage 4 PASS blocker, explicitly out of scope.
* `batch=None` uses the original PyG `x.max` path (native, first-index tie rule) — intentional.
* tie-split cells are ≤1 ULP away from the IEEE-correct CPU division (§10).
* the Stage 3E source-row 32 B read hardening item stays **DEFER** (unchanged; no new fault seen).
* `requires_grad=True` with non-fp32 dtypes, `batch` on CPU, or 1-D x still falls through to the
  original PyG path (unchanged dispatch).
* guards unchanged: `index < 491520`, `N*(F+1) < 4,026,531,840`; extreme
  `N ≥ 163800` + largeTail runtime still not exercised (HBM).

## 12. Files

### Post-freeze sanity

After the Stage 4 commits, `tests/post_freeze_sanity.py` was re-run against the frozen HEAD
`ede301f696749943a6ab08b2b515d1979b30a5cf` through the real PyG API: tie case, largeTail
`N=40 F=48825`, NaN group and `±0` group all show `fwd_max_diff=0`, `grad_max_abs_diff=0`,
identical NaN patterns, `fallback=0` (`logs/stage4/stage4_post_freeze_sanity.log`).

```
global_max_pool/stage4/
  python/global_max_pool_ascend_autograd.py     NEW  autograd.Function + backward
  tests/cpu_gradient_oracle.py                  CPU oracle G0–G10 + batch=None
  tests/cpu_tie_quirk_probe.py                  controlled experiment for the self-slot quirk
  tests/cpu_contract_model.py                   executable spec, 300-case validation
  tests/batch_none_probe.py                     batch=None CPU/NPU/compat probe
  tests/run_stage4_backward_tests.py            35-case matrix + invariants
  tests/run_stage4_pyg_e2e.py                   real PyG API E2E backward
  tests/profile_stage4_backward.py              msprof app (forward+loss+backward)
  tests/post_freeze_sanity.py                   frozen-HEAD sanity (real PyG API backward)
  tools/run_stage4_profiler.sh                  P-BWD1..4 profiler gate
  tools/stage4_final_gate.sh                    consolidated Stage 4 hard gate
  tools/stage4_primitive_audit.py               NPU primitive/fallback audit
  tools/run_stage4_regressions.sh               full regression sweep
global_max_pool/stage4_cpu_gradient_oracle.md   frozen gradient contract
global_max_pool/stage4_fp32_backward.md         this report
global_max_pool/stage6/pyg_ascend_compat/__init__.py   dispatch update (requires_grad -> autograd)
```

Logs: `/root/zyg/logs/stage4/` · profiler raw: `/root/zyg/profiler/stage4/`.
