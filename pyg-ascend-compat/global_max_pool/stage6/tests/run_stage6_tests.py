#!/usr/bin/env python3
"""Stage 6 tests: real ``torch_geometric.nn.global_max_pool`` on Ascend via pyg-ascend-compat.

Runs three phases:
  phase 0  BEFORE  - plain PyG on NPU, capture the host-fallback evidence
  phase 1  AFTER   - enable() the compat layer, call the same PyG API
  phase 2  counters/profile hooks and the PYG1..PYG8 correctness matrix

Every correctness case goes through the real PyG entry point (imported *after* enable(), which is
the documented usage); the adapter itself is never called directly here.
"""

from __future__ import annotations

import os
import subprocess
import sys
import time
import warnings

import torch
import torch_npu  # noqa: F401

DEV = "npu:0"
NEG_INF = float("-inf")
RESULTS = []


def fmt(v):
    if v != v:
        return "nan"
    if v == NEG_INF:
        return "-inf"
    return f"{v:g}"


def capture(fn):
    """Run fn, returning (value, warning messages, stderr-free fallback flag)."""
    torch.npu.synchronize()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        try:
            value = fn()
            torch.npu.synchronize()
            err = None
        except Exception as exc:  # noqa: BLE001
            value, err = None, exc
    msgs = [str(w.message) for w in caught]
    fallback = [m for m in msgs if "fall back" in m or "npu_cpu_fallback" in m]
    return value, msgs, fallback, err


def independent_reference(x, batch, size=None):
    x = x.detach().cpu().to(torch.float32)
    batch = batch.detach().cpu().to(torch.int64)
    n, f = x.shape
    if size is None:
        s = 0 if n == 0 else int(batch.max().item()) + 1
    else:
        s = int(size)
    out = torch.zeros((s, f), dtype=torch.float32)
    for g in range(s):
        mask = batch == g
        if bool(mask.any()):
            out[g] = x[mask].max(dim=0).values
    return out


def subprocess_probe(enable: bool):
    """Run the PyG API in a fresh process; return (stdout, fallback_seen).

    torch_npu reports the host fallback from C++ on stderr, which cannot be observed reliably from
    inside the process that triggers it, hence the subprocess.
    """
    script = os.path.join(os.path.dirname(os.path.abspath(__file__)), "pyg_fallback_probe.py")
    cmd = [sys.executable, script] + (["--enable"] if enable else [])
    proc = subprocess.run(cmd, capture_output=True, text=True, env=dict(os.environ))
    fallback = ("npu_cpu_fallback" in proc.stderr
                or "fall back to run on the CPU" in proc.stderr
                or "not currently supported on the NPU backend" in proc.stderr)
    return proc.stdout, proc.stderr, fallback


def record(name, ok, msg=""):
    RESULTS.append((name, "PASS" if ok else "FAIL", msg))
    print(f"{name:26s} {'PASS' if ok else 'FAIL'}  {msg}")
    return ok


def main():
    import torch_geometric
    import torch_geometric.nn as pyg_nn

    torch.npu.set_device(0)
    print(f"torch={torch.__version__} torch_npu={torch_npu.__version__} PyG={torch_geometric.__version__}")
    print(f"device={DEV}\n")

    # ------------------------------------------------------------------ phase 0: BEFORE
    print("=== phase 0: BEFORE compat (plain PyG on NPU) ===")
    x = torch.tensor([[1.0, -2.0], [3.0, -4.0], [5.0, -6.0]], dtype=torch.float32).to(DEV)
    batch = torch.tensor([0, 1, 0], dtype=torch.int64).to(DEV)
    before_out, before_msgs, before_fb, before_err = capture(lambda: pyg_nn.global_max_pool(x, batch))
    print(f"  output: {None if before_out is None else before_out.cpu().tolist()}")
    print(f"  error : {before_err}")
    before_stdout, before_stderr, before_fallback = subprocess_probe(enable=False)
    for ln in [l for l in before_stderr.splitlines() if "fall back" in l][:2]:
        print(f"  BEFORE stderr: {ln.strip()[:170]}")
    print(f"  BEFORE_HOST_CPU_FALLBACK = {'YES' if before_fallback else 'NO'}\n")
    record("BEFORE_fallback_expected", before_fallback,
           "npu_cpu_fallback on aten::scatter_reduce.two_out" if before_fallback
           else "NO fallback text observed")

    # ------------------------------------------------------------------ phase 1: AFTER
    print("=== phase 1: AFTER compat (enable + real PyG API) ===")
    import pyg_ascend_compat
    pyg_ascend_compat.enable(debug=True)
    # documented usage: import the PyG symbol *after* enabling
    from torch_geometric.nn import global_max_pool  # noqa: PLC0415

    wrapped = getattr(global_max_pool, "_pyg_ascend_compat", False)
    print(f"  torch_geometric.nn.global_max_pool is compat wrapper: {wrapped}")
    record("PYG_symbol_wrapped", bool(wrapped))

    after_out, after_msgs, after_fb, after_err = capture(lambda: global_max_pool(x, batch))
    print(f"  output: {None if after_out is None else after_out.cpu().tolist()}")
    print(f"  error : {after_err}")
    after_fallback = bool(after_fb)
    print(f"  AFTER_HOST_CPU_FALLBACK = {'YES' if after_fallback else 'NO'}")
    print(f"  counters: {pyg_ascend_compat.stats()}\n")
    record("AFTER_no_fallback", not after_fallback and after_err is None)
    record("AFTER_correct", torch.equal(after_out.cpu(), independent_reference(x, batch)))

    after_stdout, after_stderr, after_sub_fallback = subprocess_probe(enable=True)
    probe_line = (after_stdout.strip().splitlines() or [""])[-1]
    print(f"  AFTER subprocess: {probe_line}")
    print(f"  AFTER subprocess fallback text: {'YES' if after_sub_fallback else 'NO'}")
    record("AFTER_subprocess_no_fallback", not after_sub_fallback)
    record("AFTER_subprocess_used_adapter",
           "ScatterMaxV1 adapter" in (after_stdout + after_stderr))

    # ------------------------------------------------------------------ phase 2: matrix
    print("=== phase 2: PYG1..PYG8 (all through torch_geometric.nn.global_max_pool) ===")
    pyg_ascend_compat.reset_stats()
    calls = {"npu": 0, "cpu": 0}

    def case(name, x_cpu, batch_cpu, size=None, expect=None, extra=None):
        x_dev = x_cpu.to(DEV)
        b_dev = batch_cpu.to(DEV)
        calls["npu"] += 1
        t0 = time.perf_counter()
        out, msgs, fb, err = capture(lambda: global_max_pool(x_dev, b_dev, size))
        dt = (time.perf_counter() - t0) * 1000.0
        if expect is not None:
            ok = isinstance(err, expect)
            return record(name, ok, f"expected {expect.__name__}, got {err}")
        if err is not None:
            return record(name, False, f"unexpected {type(err).__name__}: {err}")
        ref = independent_reference(x_cpu, batch_cpu, size)
        if ref.shape != tuple(out.shape):
            return record(name, False, f"shape {tuple(out.shape)} != {tuple(ref.shape)}")
        got = out.detach().cpu()
        same = torch.equal(ref, got)
        finite = torch.isfinite(ref) & torch.isfinite(got)
        diff = float((ref[finite] - got[finite]).abs().max()) if bool(finite.any()) else 0.0
        inf_ok = torch.equal(torch.isinf(ref), torch.isinf(got))
        ok = same and diff == 0.0 and inf_ok
        detail = f"shape={tuple(got.shape)} max_diff={diff} exact={same} fallback_warn={bool(fb)}"
        if extra:
            detail += " " + extra(got)
        return record(name, ok, detail + f" t={dt:.1f}ms")

    torch.manual_seed(0)
    x8 = torch.round((torch.rand((8, 8)) * 20 - 10) * 4) / 4
    b8 = torch.tensor([0, 1, 0, 2, 1, 2, 0, 3], dtype=torch.int64)
    case("PYG1_baseline", x8, b8)

    x7 = torch.round((torch.rand((8, 7)) * 20 - 10) * 4) / 4
    case("PYG2_non_aligned_F7", x7, b8)

    x4 = torch.round((torch.rand((4, 8)) * 20 - 10) * 4) / 4
    case("PYG3_explicit_size", x4, torch.tensor([0, 2, 0, 1], dtype=torch.int64), size=5,
         extra=lambda g: f"rows3_4_zero={bool((g[3:] == 0).all())}")

    case("PYG4_repeated_index", torch.round((torch.rand((16, 8)) * 20 - 10) * 4) / 4,
         torch.zeros(16, dtype=torch.int64))

    case("PYG5_negative_only", -(torch.rand((8, 8)) * 10 + 1),
         torch.tensor([0, 0, 1, 1, 2, 2, 2, 2], dtype=torch.int64), size=4)

    x_inf = torch.full((3, 8), NEG_INF)
    x_inf[2] = torch.tensor([-3.0, -1.5, -2.25, -4.0, -0.5, -7.0, -6.5, -8.0])
    case("PYG6_true_neg_inf_vs_empty", x_inf, torch.tensor([0, 0, 2], dtype=torch.int64), size=3,
         extra=lambda g: f"g0=-inf:{bool((g[0] == NEG_INF).all())} g1=0:{bool((g[1] == 0).all())}")

    case("PYG7_F0", torch.zeros((4, 0)), torch.tensor([0, 2, 0, 1], dtype=torch.int64))

    case("PYG8a_empty_size_None", torch.zeros((0, 8)), torch.zeros((0,), dtype=torch.int64))
    case("PYG8b_empty_size_4", torch.zeros((0, 8)), torch.zeros((0,), dtype=torch.int64), size=4)

    print("\n=== external PyG CPU parity (same API on CPU) ===")
    for name, x_cpu, b_cpu, size in (
        ("cpu_parity_8", x8, b8, None),
        ("cpu_parity_f7", x7, b8, None),
        ("cpu_parity_size5", x4, torch.tensor([0, 2, 0, 1], dtype=torch.int64), 5),
        ("cpu_parity_neginf", x_inf, torch.tensor([0, 0, 2], dtype=torch.int64), 3),
    ):
        cpu_ref = pyg_nn.global_max_pool(x_cpu, b_cpu, size)
        calls["cpu"] += 1
        calls["npu"] += 1
        npu = global_max_pool(x_cpu.to(DEV), b_cpu.to(DEV), size).detach().cpu()
        record(name, torch.equal(cpu_ref, npu) or bool(torch.allclose(cpu_ref, npu, equal_nan=True)),
               "PyG CPU == PyG NPU(compat)")

    st = pyg_ascend_compat.stats()
    print(f"\nfinal counters: {st}")
    record("counters_match_dispatch",
           st["ascend_calls"] == calls["npu"] and st["original_calls"] == calls["cpu"]
           and st["ascend_calls"] > 0,
           f"ascend_calls={st['ascend_calls']} (NPU calls={calls['npu']}), "
           f"original_calls={st['original_calls']} (CPU calls={calls['cpu']})")

    print("\n=== summary ===")
    failed = [r for r in RESULTS if r[1] != "PASS"]
    for name, status, msg in RESULTS:
        print(f"{status:4s} {name:26s} {msg}")
    print(f"\nTOTAL {len(RESULTS)}  PASS {len(RESULTS) - len(failed)}  FAIL {len(failed)}")
    print(f"BEFORE_HOST_CPU_FALLBACK={'YES' if before_fallback else 'NO'} "
          f"AFTER_HOST_CPU_FALLBACK={'YES' if after_fallback else 'NO'}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
