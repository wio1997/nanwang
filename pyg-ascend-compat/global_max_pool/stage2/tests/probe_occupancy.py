#!/usr/bin/env python3
"""Stage 2 probe: which candidate occupancy / post-process ops run natively on NPU?

Detection method follows the project's earlier convention (native_path_evidence.md):
torch_npu prints "The operator '<op>' is not currently supported on the NPU backend and will
fall back to run on the CPU (function npu_cpu_fallback)" for host fallbacks, which surfaces as a
Python warning we can capture with warnings.catch_warnings(record=True).

This probe only decides which ops the adapter may use; the final proof is the msprof run of the
complete adapter path (see run_stage2_tests.py / the Stage 2 report).
"""

import warnings

import torch
import torch_npu  # noqa: F401


def run_case(name, fn):
    torch.npu.synchronize()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        try:
            value = fn()
            torch.npu.synchronize()
            status = "OK"
        except Exception as exc:  # noqa: BLE001
            value = None
            status = f"ERROR {type(exc).__name__}: {exc}"
    msgs = [str(w.message) for w in caught]
    fallback = [m for m in msgs if "fall back" in m or "npu_cpu_fallback" in m]
    print(f"{name:34s} status={status}")
    for m in msgs:
        print(f"{'':34s}   warn: {m.strip()[:150]}")
    print(f"{'':34s} FALLBACK_WARNING={'YES' if fallback else 'no'}")
    return value


def main():
    dev = "npu:0"
    torch.npu.set_device(0)
    S, N = 8, 32
    batch64 = torch.randint(0, S, (N,), dtype=torch.int64, device=dev)
    out = torch.full((S, 8), float("-inf"), dtype=torch.float32, device=dev)

    print(f"device={dev} torch={torch.__version__} torch_npu={torch_npu.__version__}\n")

    x = run_case("batch.to(int32) [cast]",
                 lambda: batch64.to(torch.int32))
    batch32 = x if x is not None else batch64.to(torch.int32)

    run_case("torch.full(-inf)", lambda: torch.full((S, 8), float("-inf"), device=dev))
    run_case("torch.zeros(int32)", lambda: torch.zeros(S, dtype=torch.int32, device=dev))

    def c_scatter_i32():
        occ = torch.zeros(S, dtype=torch.int32, device=dev)
        occ.scatter_(0, batch32, 1)
        return occ

    occ_i32 = run_case("zeros().scatter_(int32 idx)", c_scatter_i32)

    def c_scatter_i64():
        occ = torch.zeros(S, dtype=torch.int32, device=dev)
        occ.scatter_(0, batch64, 1)
        return occ

    run_case("zeros().scatter_(int64 idx)", c_scatter_i64)
    run_case("torch.bincount(batch)", lambda: torch.bincount(batch64, minlength=S))

    if occ_i32 is not None:
        occ = occ_i32.clone()
        empty = (occ == 0).view(S, 1)
        run_case("occ == 0", lambda: (occ == 0))
        run_case("out.masked_fill_(empty,0)", lambda: out.clone().masked_fill_(empty, 0.0))
        run_case("torch.where(occ,out,0)",
                 lambda: torch.where(occ.view(S, 1) != 0, out, torch.zeros_like(out)))
        run_case("out[empty]=0 (index_put_)", lambda: _index_assign(out, empty))
        run_case("out.index_fill_(0,idx,0)",
                 lambda: out.clone().index_fill_(0, torch.nonzero(occ == 0).flatten(), 0.0))

    print("\n[note] a fallback warning here is decisive; absence of a warning is corroborated by "
          "msprof task types for the full adapter run.")


def _index_assign(out, empty):
    t = out.clone()
    t[empty] = 0.0
    return t


if __name__ == "__main__":
    main()
