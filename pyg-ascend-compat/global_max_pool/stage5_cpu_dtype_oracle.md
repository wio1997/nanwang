# Stage 5 — FP16 / BF16 CPU gradient oracle (frozen contract)

Measured with the installed stack in `wio-pyg-cann851-pyg280` — python 3.11.14, torch 2.9.0+cpu,
PyG 2.8.0.post1 — on **CPU**, through the public `torch_geometric.nn.global_max_pool`. Nothing is
inherited from the FP32 oracle: every value below was measured for the dtype itself.

Scripts: `global_max_pool/stage5/tests/{cpu_dtype_oracle,cpu_dtype_arithmetic_probe,
cpu_dtype_count_model,cpu_dtype_div_search,cpu_dtype_div_decide}.py`
Logs: `/root/zyg/logs/stage5/cpu_dtype*.log`

## 1. Support and dtypes

Both dtypes are fully supported by the CPU oracle (26/26 cases each, forward + backward):

| dtype | forward | backward | `out.dtype` | `x.grad.dtype` |
|---|---|---|---|---|
| `torch.float16` | OK | OK | `torch.float16` | `torch.float16` |
| `torch.bfloat16` | OK | OK | `torch.bfloat16` | `torch.bfloat16` |

## 2. Contract (identical in structure to FP32, arithmetic in the input dtype)

```
out[g,f]    = max over { x[i,f] : batch[i] = g }          empty group -> 0
winner      = (x == out[batch])                           in the input dtype
count[g,f]  = (# winners) + (1 if out[g,f] == 0 else 0)   exact integer
count_dt    = cast_to_input_dtype(count)                  round-to-nearest-even
grad_x      = winner * (grad_out[batch] / count_dt[batch])
```

Measured highlights (both dtypes):

* ties split equally, **rounded to the dtype**: 3-way tie → `0.333251953125` (fp16) /
  `0.333984375` (bf16); 7-way → `0.142822265625` / `0.142578125`;
* the **zero-max self-slot quirk persists**: `[[0],[-0]] → 1/3` (dtype-rounded), `[[0]] → 0.5`,
  `[[4],[4],[0]] size=5 upstream=11 → 5.5`;
* empty group → `0`, no input gradient, upstream ignored;
* non-empty all-`-inf` group → `-inf`, ties split; `+inf` ties split normally;
* NaN → `nan` forward and `nan` for **every** row of that cell in the backward;
* `batch=None` → PyG `x.max` path with the first-max gradient rule (as in FP32).

## 3. Internal arithmetic — measured, not assumed

**Count.** Two candidates were tested against the real oracle: an integer count converted to the
input dtype (`M_round`) vs. an accumulation of `+1` performed *in* the dtype (`M_inc`, which
saturates: fp16 stops incrementing at 2048, bf16 at 256):

| dtype | cases probed | `M_round` hits | `M_inc` hits |
|---|---|---|---|
| fp16 | 21 (N = 2046…4103, 8193, 16385) | **21/21** | 4/21 |
| bf16 | 22 (N = 250…265, 510…514, 1025) | **22/22** | 8/22 |

Example (fp16, N=4100 identical rows, upstream 1): measured `0.00024390220642089844` = fp16(1/4100)
— the incremental dtype accumulation would have saturated at 2048 → `1/2048`.
**⇒ the denominator is the exact integer count rounded to the input dtype.**

**Division.** Candidates "divide in the input dtype" vs "divide in fp32, then cast". An exhaustive
search over dtype-representable gradients × counts (fp16 counts 1…5999, bf16 1…19999, 14 gradients,
exact `Fraction` arithmetic) found **no discriminating pair**, i.e. the two are observationally
identical for representable inputs (`cpu_dtype_div_search.log` / `cpu_dtype_div_decide.log`), and
the NPU dtype division was later measured bit-exact against CPU over 4096 random pairs
(`npu_primitive_dtype_audit.log`). The implementation therefore divides in the input dtype.

## 4. `batch=None`

PyG still routes `batch=None` to `x.max(dim=-2)` (1-D: `dim=-1`), **not** to the scatter path.
Measured on CPU *and* NPU for both dtypes: forward and backward identical, ties resolved to the
first maximal index, `-inf`/NaN patterns equal, zero host-fallback warnings, and the compat wrapper
keeps delegating (`logs/stage5/batch_none_dtype_probe.log`). Out of the Stage 5 custom scope.
