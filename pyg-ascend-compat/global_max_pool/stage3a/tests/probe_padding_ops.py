#!/usr/bin/env python3
"""Stage 3A probe: which NPU-native way to build [N, F_pad] with -inf padding columns?

Candidates: F.pad(constant), new_full + slice assign, torch.cat with a -inf block.
Fallback detection follows the project convention (torch_npu npu_cpu_fallback warning);
the chosen op is additionally confirmed by msprof in the Stage 3A profile run (report §10).
"""

import warnings

import torch
import torch.nn.functional as F
import torch_npu  # noqa: F401


def run_case(name, fn):
    torch.npu.synchronize()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        try:
            out = fn()
            torch.npu.synchronize()
            status = "OK"
        except Exception as exc:  # noqa: BLE001
            out = None
            status = f"ERROR {type(exc).__name__}: {exc}"
    fb = [str(w.message) for w in caught if "fall back" in str(w.message)]
    print(f"{name:34s} status={status} FALLBACK={'YES' if fb else 'no'}")
    for m in fb:
        print(f"{'':34s}   warn: {m.strip()[:140]}")
    return out


def main():
    torch.npu.set_device(0)
    dev = "npu:0"
    n, f, f_pad = 4096, 33, 40
    x = (torch.rand((n, f), dtype=torch.float32) - 0.5).to(dev)

    def pad_fn():
        return F.pad(x, (0, f_pad - f), mode="constant", value=float("-inf"))

    def full_slice():
        y = x.new_full((n, f_pad), float("-inf"))
        y[:, :f] = x
        return y

    def cat_fn():
        pad_block = x.new_full((n, f_pad - f), float("-inf"))
        return torch.cat((x, pad_block), dim=1)

    a = run_case("F.pad(constant, -inf)", pad_fn)
    b = run_case("new_full + slice assign", full_slice)
    c = run_case("torch.cat(-inf block)", cat_fn)

    ref = torch.full((n, f_pad), float("-inf"), dtype=torch.float32)
    ref[:, :f] = x.detach().cpu()
    for name, out in (("F.pad", a), ("full+slice", b), ("cat", c)):
        if out is None:
            print(f"{name:34s} correctness: SKIPPED (failed)")
            continue
        ok = torch.equal(out.detach().cpu(), ref)
        print(f"{name:34s} correctness (first F cols == x, pads == -inf): {ok}")


if __name__ == "__main__":
    main()
