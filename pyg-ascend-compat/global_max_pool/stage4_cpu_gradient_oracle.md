# Stage 4 — CPU gradient oracle (frozen contract)

Measured with the *real* installed stack in `wio-pyg-cann851-pyg280`:

```
python 3.11.14 · torch 2.9.0+cpu · torch_geometric 2.8.0.post1
API     torch_geometric.nn.global_max_pool   (CPU only, CPU reference = final oracle)
scripts global_max_pool/stage4/tests/{cpu_gradient_oracle.py,cpu_tie_quirk_probe.py,cpu_contract_model.py}
logs    /root/zyg/logs/stage4/{cpu_gradient_oracle.log,cpu_tie_quirk_probe.log,cpu_contract_model.log}
```

PyG routes `global_max_pool(x, batch, size)` to
`scatter(x, batch, dim=-2, dim_size=size, reduce='max')`, which is

```python
out = src.new_zeros(size).scatter_reduce_(-2, index, src, reduce="amax", include_self=False)
```

## 1. The contract

```
out[g,f]   = max over { x[i,f] : batch[i] = g }          empty group -> 0
winner     = (x == out[batch])
count[g,f] = sum_i winner[i,f] over the group  +  (1 if out[g,f] == 0 else 0)
grad_x     = winner * (grad_out[batch] / count[batch])   computed in fp32
```

Everything below is measured, not assumed — see the oracle logs. The `+ (out == 0)` term is a
PyTorch `scatter_reduce(..., include_self=False)` backward quirk (the *forward-excluded* zero
self slot is still counted when it equals the result); §4 proves it with a controlled experiment.

The formula was validated as an executable model against the real PyG API on **300/300 randomized
cases** (kinds: ordinary, zeros, ties, `-inf`, NaN; 1–12 nodes, 1–4 groups, F ∈ {1,2,3,7,8,9},
upstream ∈ {0, ±1, 2.5}): exact match on forward *and* gradient, NaN treated as equal to NaN
(`cpu_contract_model.log`).

## 2. Element-wise semantics (G0–G5)

| case | data (one group) | forward | `x.grad` (loss `out.sum()`) |
|---|---|---|---|
| G0 unique max | `[[1,10],[3,20],[2,30]]` | `[[3,30]]` | `[[0,0],[1,0],[0,1]]` |
| G1 2-way tie | `[[3,1],[3,2],[1,3]]` | `[[3,3]]` | `[[0.5,0],[0.5,0],[0,1]]` |
| G2 3-way tie | `[[5,5],[5,5],[5,5]]` | `[[5,5]]` | all entries `0.3333333432674408` = fp32(1/3) |
| G3 per-feature ties | `[[5,1,7],[5,3,7],[2,3,0]]` | `[[5,3,7]]` | `[[.5,0,.5],[.5,.5,.5],[0,.5,0]]` |
| G5 multiple groups, repeated index | 2 groups / 6 rows | per group | per group, independent ties |

Ties split **equally**, per `(group, feature)` cell, using fp32 division (`1/3` is
`0.3333333432674408`, not `0.3333…` double).

## 3. Weighted upstream (G4)

`loss = (out * upstream).sum()` with `upstream=[[2,4]]` on `[[3,1],[3,2],[1,3]]` gives
`x.grad = [[1,0],[1,0],[0,4]]` — i.e. `grad_out[g,f] / count[g,f]` (2/2 for the tie, 4 for the
unique max). Zero upstream cells give exactly `0`.

## 4. Zero-valued maxima — the `include_self=False` quirk (G9 + controlled experiment)

| data | destination init | forward | grad |
|---|---|---|---|
| `[[0],[0]]` | `0.0` (PyG) | `[[0]]` | `1/3, 1/3` |
| `[[0],[0]]` | `-1000.0` | `[[0]]` | `0.5, 0.5` |
| `[[0],[0]]` | `+1000.0` / `2.0` | `[[0]]` | `0.5, 0.5` |
| `[[0]]` | `0.0` (PyG) | `[[0]]` | `0.5` |
| `[[0]]` | `-1000.0` | `[[0]]` | `1.0` |
| `[[0],[-0.0]]` | `0.0` (PyG) | `[[0]]` | `1/3, 1/3` |
| `[[-0.0]]` | `0.0` (PyG) | `[[-0.0]]` | `0.5` |
| `[[0],[-0],[-1]]` | `0.0` (PyG) | `[[0]]` | `1/3, 1/3, 0` |
| `[[5],[5]]` | `0.0` (PyG) | `[[5]]` | `0.5, 0.5` |

With the *same source data* and a non-zero destination init the quirk disappears — so the extra
denominator term is "the self slot equals the result", not "the value 0 exists in the data".
Signed zeros compare equal (`-0.0 == 0.0`), so `max(+0, -0) == 0` counts the self slot as well.
`global_max_pool(x, batch, size=5)` with a group whose single row is `0` therefore returns
`grad = upstream/2` for that row (measured: upstream `11` → `5.5`).

## 5. Empty groups / explicit size (G6)

`x=[[4,1],[4,2],[0,9]]`, `batch=[0,0,3]`, `size=5`, `upstream=[[1,1],[5,5],[7,7],[11,11],[13,13]]`:

```
forward : [[4,2],[0,0],[0,0],[0,9],[0,0]]      <- empty groups are exactly 0
x.grad  : [[0.5,0],[0.5,1],[5.5,11]]
```

* empty groups contribute **no** input gradient, and an upstream value on an empty group changes
  nothing (the group is occupied by no row);
* group 0 follows the ordinary rules; group 3 (single row, max `0`) follows §4 (`11/2 = 5.5`).

## 6. Negative-only data (G7)

`[[-5],[-2],[-2]]` → forward `[[-2]]`, grad `[[0],[0.5],[0.5]]`: the *least negative* value wins and
ties split normally (no special case for negatives).

## 7. True `-inf` (G8)

| data | batch / size | forward | grad |
|---|---|---|---|
| `[[-inf],[-inf]]` | `[0,0]`, size 1 | `[[-inf]]` | `0.5, 0.5` |
| `[[-inf],[-inf],[-1]]` | `[0,0,0]`, size 1 | `[[-1]]` | `0, 0, 1` |
| `[[-inf],[-inf],[5]]` | `[0,0,3]`, size 5 | `[[-inf],[0],[0],[5],[0]]` | `0.5, 0.5, 1` |

A non-empty group of `-inf` keeps `-inf` (forward, Stage 3 contract) and its backward splits the
gradient among the `-inf` rows; an **empty** group is `0` and produces no gradient. The two cases
are distinct and both are reproduced.

## 8. NaN (G10)

| data | forward | grad |
|---|---|---|
| `[[nan],[1],[2]]` | `[[nan]]` | `[[nan],[nan],[nan]]` |
| `[[nan],[nan],[2]]` | `[[nan]]` | `[[nan],[nan],[nan]]` |
| `[[nan],[+inf],[1]]` | `[[nan]]` | `[[nan],[nan],[nan]]` |
| `[[nan],[1]]` with upstream `0` | `[[nan]]` | `[[nan],[nan]]` |

NaN is **contagious in the backward**: *every* row of a NaN `(group, feature)` cell receives `nan`,
including rows whose own value is finite. This follows from the contract (`x == nan` is false
everywhere → `count = 0` → `0 * (grad/0) = nan`) and is *matched* by the Ascend implementation;
there is no "NaN unsupported" carve-out.

`+inf` rows behave like any other tie: `[[inf],[inf],[1]]` → `0.5, 0.5, 0`.

## 9. `batch=None`

PyG does **not** use scatter here:

```python
dim = -1 if x.dim() == 1 else -2
if batch is None:
    return x.max(dim=dim, keepdim=x.dim() <= 2)[0]
```

* 2-D: `out = x.max(dim=-2, keepdim=True)[0]` (shape `[1, F]`); 1-D: `x.max(dim=-1, keepdim=True)`.
* Backward: the **full** upstream goes to the *first* maximal index per column (not split):
  `[[3,1],[3,2],[1,3]]` → `x.grad = [[1,0],[0,0],[0,1]]`.
* On the NPU the original PyG path is native (no `npu_cpu_fallback` warning) and matches the CPU
  oracle exactly for forward *and* backward, including ties and `-inf`
  (`/root/zyg/logs/stage4/batch_none_probe.log`).
* The compat wrapper keeps delegating (`batch=None` → original PyG, `ascend_calls=0`,
  `original_calls=1`).

**Therefore `batch=None` is explicitly outside the Stage 4 custom-backward scope**: it is a
different operator (`x.max`), it is already correct and device-native, and Stage 4 neither
intercepts nor changes it.

## 10. Frozen contract summary (what the Ascend backward must reproduce)

```
unique max        : grad_out at the argmax, 0 elsewhere
2-way tie         : grad_out / 2 per winner
3-way tie         : grad_out / 3 per winner          (fp32 division)
per-feature ties  : independent per (group, feature)
weighted upstream : linear scaling by grad_out
zero maximum      : +1 to the denominator (self slot equals the result)
empty group       : out = 0, no input gradient, upstream ignored
negative-only     : ordinary rules
-inf (non-empty)  : visible in the output; ties split normally
NaN               : forward nan, backward nan for every row of that group/feature
+inf              : ordinary rules
batch=None        : NOT scatter -> x.max, first-index gradient, native, unchanged
```
