# Stage 3D — largeTail MTE root-cause + kernel repair

> **Stage 3E update:** these repairs are now promoted into the formal delivery source/OPP (plus a
> byte-exact index load closing the 32 B index GM over-read) and re-validated on that package — see
> `../stage3e_large_tail_delivery_promotion.md`.

**Result: PASS** — the large-tail path now runs correctly on device.

## Root cause (two defects, both in the large-tail `elemWiseBatchProcess`)

1. **Local index lookup off by the block offset (fatal).**
   `batchProcess()` copies the index block starting at GM `idxOffset` into `_idxLocal[0..]`, so entry
   *k* is `_idxLocal.GetValue(k)`. The large-tail code read `GetValue(idxOffset + k)`; with
   `idxBatchNum = 1` (N=40) the buffer holds 8 entries while `idxOffset` runs 0…39, so most cores read
   past the local buffer, obtained a garbage value and used it as the destination row
   (`_resGM[garbage * F]`) → `507035` / *MTE DDR address out of range*.
2. **Missing chunk offset on the write side.**
   The row is read in chunks `n = 0 … _srcLoop-1` at `row*F + n*srcBatchNum`, but the write-back
   omitted `+ n*srcBatchNum`, so every chunk after the first landed on the row head (`max_abs_diff=inf`).

## Fix (two lines, probe copy; behaviour for tiling keys 0/1 otherwise unchanged)

```c
-    DTYPE_INDEX idxVal = _idxLocal.GetValue(idxOffset + k);
+    DTYPE_INDEX idxVal = _idxLocal.GetValue(k);

-    DataCopyPad(_resGM[idxVal * _tailElemNum], _srcLocal, {1, srcLoadNum*sizeof(DTYPE_RES), …});
+    DataCopyPad(_resGM[idxVal * _tailElemNum + n * _srcBatchNum], _srcLocal, {…});
```

(Stage 3C's entry repair stays required: explicit `TILING_KEY_IS(1)` branch +
`add_ops_compile_options(ScatterMaxV1 OPTIONS --tiling_key=0,1)`.)

## Tools / tests

| file | purpose |
|---|---|
| `tools/apply_largetail_index_fix.py` | fix #1 (index lookup) |
| `tools/apply_largetail_write_offset_fix.py` | fix #2 (write chunk offset) |
| `tools/apply_largetail_datacopy_fix.py` | attempted change, **refuted** (kept for the record) |
| `tests/run_stage3d_largetail_tests.py` | tail-attack correctness + PyG E2E (9/9 PASS) |

## Evidence

* `STAGE3C_OPP=<fix4> bash ../stage3c/tools/run_stage3c_largetail_matrix.sh` → LT0–LT6 all PASS
* profiler P1/P2/P3 → `ScatterMaxV1` **AI_VECTOR_CORE** completed ×8, `AI_CPU=0`
* Stage 3B 45/45 · Stage 6 demo PASS · Stage 6 20/20 · no `507035` / MTE OOB / fallback

Full report: `../stage3d_large_tail_mte_repair.md`.
