# Stage 3E — delivery promotion + final freeze

Promotes the Stage 3C/3D largeTail repairs from the validated probe copy into the **formal delivery
OPP** and re-proves the whole FP32-forward path on it.

| item | result |
|---|---|
| formal delivery source | `/root/zyg/build/scattermax_probe` (CMakePresets `ascend910b`, vendor `customize`) |
| formal runtime OPP | `/root/zyg/build/scattermax_runtime_opp/vendors/customize` |
| kernel package | `kernelList = [_0,_1]`, `supportInfo.tilingKey = ["0","1"]` |
| runtime provenance | `LD_PRELOAD` open() trace: only the formal OPP kernel `.o` is opened |
| largeTail matrix | LT0–LT6 + LTA1/LTA2 = 9/9 PASS, `max_abs_diff=0` |
| tail attack + PyG E2E | 13/13 PASS, `ascend_calls=3 original_calls=0` |
| profiler | `ScatterMaxV1` AI_VECTOR_CORE ×8 per case, `AI_CPU=0`, entry `_1` |
| regressions | Stage 3A 34/34 · 3B 45/45 · 3D 9/9 · 6 demo PASS · 6 20/20 |
| index GM over-read | **REMOVED_BY_FIX** (byte-exact `DataCopyPad` index loads) |

Order of operations:

```bash
bash tools/00_pre_state.sh                       # freeze the pre-promotion state
python3 tools/01_apply_delivery_fixes.py --check /root/zyg/build/scattermax_probe
python3 tools/01_apply_delivery_fixes.py --apply /root/zyg/build/scattermax_probe
bash tools/02_build_delivery_opp.sh              # clean rebuild + kernel _0/_1 gate
bash tools/03_install_delivery_opp.sh            # install + quarantine probe OPPs
bash tools/04_prove_loaded_opp.sh                # Task D provenance proof
bash tools/run_stage3e_largetail_matrix.sh       # Task F
python3 tests/run_stage3e_largetail_tests.py     # Task G/H (source stage6/env.sh first)
bash tools/run_stage3e_profiler.sh               # Task I
bash tools/run_stage3e_regressions.sh            # Task J
bash tools/09_verify_delivery_package.sh         # Task C package evidence
bash tools/10_final_gate.sh                      # consolidated hard gate
```

Full evidence: `../stage3e_large_tail_delivery_promotion.md`.
