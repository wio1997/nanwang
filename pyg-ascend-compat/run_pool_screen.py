#!/usr/bin/env python3
import argparse
import json
import math
import os
import platform
import statistics
import time
import traceback
from pathlib import Path

import torch
import torch_npu
import torch_geometric
from torch_geometric.nn import (
    SAGPooling, TopKPooling, global_add_pool, global_max_pool,
    global_mean_pool, global_sort_pool,
)


DTYPES = {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}
GLOBAL_FUNCS = {
    "global_mean_pool": global_mean_pool,
    "global_add_pool": global_add_pool,
    "global_max_pool": global_max_pool,
    "global_sort_pool": global_sort_pool,
}


def metrics(actual, expected):
    a, e = actual.detach().float().cpu(), expected.detach().float().cpu()
    d = (a - e).abs()
    return {
        "shape_equal": list(a.shape) == list(e.shape),
        "max_abs": float(d.max()) if d.numel() else 0.0,
        "max_rel": float((d / e.abs().clamp_min(1e-12)).max()) if d.numel() else 0.0,
        "nan_equal": bool(torch.equal(torch.isnan(a), torch.isnan(e))),
        "inf_equal": bool(torch.equal(torch.isinf(a), torch.isinf(e))),
    }


def tol(dtype_name):
    return (1e-4, 1e-5) if dtype_name == "fp32" else (2e-2, 2e-2)


def base_x():
    return torch.tensor([
        [-1.2, .1, 1., 2., -.5, 3., .25, -.8],
        [.3, -.7, 2.2, 1., .5, -1., 1.25, 1.2],
        [1.1, .9, -.4, .3, 2., .5, -.75, .2],
        [2., 1., -1., 0., 3., -2., .5, 1.5],
        [1.5, -2., .5, 2., -1., 1., 2.5, -.3],
        [-.5, 3., 1.5, -1., 0., 2., -1.5, .7],
        [.7, 1.7, -2., 1.2, 2.2, -.8, 1.8, 2.1],
        [3., -1., 2., -2., 1., 0., -.2, .4],
        [-1., .5, 0., 3., -2., 1.5, .7, 1.1],
    ], dtype=torch.float32)


def manual_global(api, x, batch, size=2, k=4):
    rows = []
    for graph in range(size):
        part = x[batch == graph]
        if api == "global_add_pool":
            rows.append(part.sum(0))
        elif api == "global_mean_pool":
            rows.append(part.mean(0))
        elif api == "global_max_pool":
            rows.append(part.max(0).values)
        else:
            order = torch.argsort(part[:, -1], descending=True)
            part = part[order][:k]
            if part.size(0) < k:
                part = torch.cat([part, part.new_zeros(k - part.size(0), part.size(1))], 0)
            rows.append(part.reshape(-1))
    return torch.stack(rows)


def run_global_case(api, dtype_name, npu_device):
    fn, dtype = GLOBAL_FUNCS[api], DTYPES[dtype_name]
    x_master = base_x()
    batch = torch.tensor([0, 0, 0, 1, 1, 1, 1, 1, 1], dtype=torch.long)
    kwargs = {"k": 4} if api == "global_sort_pool" else {"size": 2}
    x_cpu = x_master.clone().requires_grad_(True)
    cpu_out = fn(x_cpu, batch, **kwargs)
    manual = manual_global(api, x_master, batch)
    upstream = torch.linspace(-.7, 1.3, cpu_out.numel()).reshape_as(cpu_out)
    cpu_out.backward(upstream)
    cpu_grad = x_cpu.grad.detach()

    x_npu = x_master.to(npu_device, dtype=dtype).requires_grad_(True)
    out = fn(x_npu, batch.to(npu_device), **kwargs)
    out.backward(upstream.to(npu_device, dtype=dtype))
    torch.npu.synchronize()
    rtol, atol = tol(dtype_name)
    out_m = metrics(out, cpu_out)
    grad_m = metrics(x_npu.grad, cpu_grad)
    out_m["allclose"] = bool(torch.allclose(out.float().cpu(), cpu_out.detach(), rtol=rtol, atol=atol))
    grad_m["allclose"] = bool(torch.allclose(x_npu.grad.float().cpu(), cpu_grad, rtol=rtol, atol=atol))
    manual_m = metrics(cpu_out, manual)
    return {
        "dtype": dtype_name, "mode": "eager_forward_backward",
        "input_shape": list(x_master.shape), "output_shape": list(out.shape),
        "output_device": str(out.device), "manual_reference": manual_m,
        "output_accuracy": {**out_m, "rtol": rtol, "atol": atol},
        "gradient_accuracy": grad_m,
        "success": out_m["allclose"] and grad_m["allclose"] and manual_m["max_abs"] <= 1e-6,
    }


def graph_inputs(features=8, device="cpu", dtype=torch.float32, graphs=(5, 7)):
    total = sum(graphs)
    x = torch.arange(total * features, dtype=torch.float32).reshape(total, features)
    x = torch.sin(x * .17) + torch.arange(total, dtype=torch.float32).unsqueeze(1) * .07
    batch, edges, offset = [], [], 0
    for gid, count in enumerate(graphs):
        batch.extend([gid] * count)
        for i in range(offset, offset + count - 1):
            edges.extend([(i, i + 1), (i + 1, i)])
        offset += count
    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
    edge_attr = torch.stack([
        torch.linspace(.1, 1.0, edge_index.size(1)),
        torch.linspace(-.5, .5, edge_index.size(1)),
    ], dim=1)
    return x.to(device, dtype=dtype), edge_index.to(device), edge_attr.to(device, dtype=dtype), torch.tensor(batch, dtype=torch.long, device=device)


def make_hier(api, features, dtype, device):
    torch.manual_seed(314159)
    module = TopKPooling(features, ratio=.5) if api == "TopKPooling" else SAGPooling(features, ratio=.5)
    if api == "TopKPooling":
        with torch.no_grad():
            module.select.weight.copy_(torch.linspace(.3, 1.1, features).reshape(1, -1))
    return module.eval().to(device=device, dtype=dtype)


def run_hier_case(api, dtype_name, npu_device):
    dtype = DTYPES[dtype_name]
    x_cpu, ei_cpu, ea_cpu, batch_cpu = graph_inputs()
    cpu_module = make_hier(api, 8, torch.float32, "cpu")
    cpu_out = cpu_module(x_cpu, ei_cpu, ea_cpu, batch_cpu)
    npu_module = make_hier(api, 8, dtype, npu_device)
    npu_module.load_state_dict({k: v.to(dtype=dtype) for k, v in cpu_module.state_dict().items()})
    x, ei, ea, batch = graph_inputs(device=npu_device, dtype=dtype)
    out = npu_module(x, ei, ea, batch)
    torch.npu.synchronize()
    rtol, atol = tol(dtype_name)
    names = ["x_pool", "edge_index", "edge_attr", "batch", "perm", "score"]
    comparisons, all_ok = {}, True
    devices = {}
    for name, actual, expected in zip(names, out, cpu_out):
        devices[name] = None if actual is None else str(actual.device)
        if actual is None or expected is None:
            ok = actual is None and expected is None
            comparisons[name] = {"equal": ok}
        elif actual.dtype in (torch.int32, torch.int64, torch.bool):
            ok = bool(torch.equal(actual.cpu(), expected))
            comparisons[name] = {"exact_equal": ok, "shape": list(actual.shape)}
        else:
            m = metrics(actual, expected)
            ok = bool(torch.allclose(actual.float().cpu(), expected.detach(), rtol=rtol, atol=atol))
            comparisons[name] = {**m, "allclose": ok, "rtol": rtol, "atol": atol}
        all_ok = all_ok and ok
    return {
        "dtype": dtype_name, "mode": "eager_forward", "input_shape": [12, 8],
        "edge_shape": list(ei.shape), "ratio": .5, "output_devices": devices,
        "comparisons": comparisons, "success": all_ok,
    }


def percentile(samples, q):
    v = sorted(samples)
    return v[max(0, min(len(v) - 1, math.ceil(q * len(v)) - 1))]


def make_bench(api, device, npu_device):
    target = npu_device if device == "npu" else "cpu"
    if api in GLOBAL_FUNCS:
        b, n, f = 32, 128, 64
        x = torch.linspace(-2, 2, b * n * f, device=target).reshape(b * n, f)
        batch = torch.arange(b, device=target).repeat_interleave(n)
        fn = GLOBAL_FUNCS[api]
        kwargs = {"k": 64} if api == "global_sort_pool" else {"size": b}
        call = lambda: fn(x, batch, **kwargs)
        shape = [b * n, f]
    else:
        b, n, f = 16, 64, 32
        x, ei, ea, batch = graph_inputs(f, target, torch.float32, (n,) * b)
        module = make_hier(api, f, torch.float32, target)
        call = lambda: module(x, ei, ea, batch)
        shape = [b * n, f]
    return call, shape


def benchmark(api, device, npu_device, warmup=10, iterations=50):
    call, shape = make_bench(api, device, npu_device)
    sync = torch.npu.synchronize if device == "npu" else lambda: None
    with torch.no_grad():
        for _ in range(warmup): call()
        sync()
        samples = []
        for _ in range(iterations):
            sync(); start = time.perf_counter_ns(); call(); sync()
            samples.append((time.perf_counter_ns() - start) / 1e6)
    return {"shape": shape, "warmup": warmup, "iterations": iterations,
            "median_ms": statistics.median(samples), "p90_ms": percentile(samples, .9),
            "min_ms": min(samples), "max_ms": max(samples)}


def collect_profile(api, output_dir, npu_device):
    from torch_npu.profiler import (AiCMetrics, ProfilerActivity, ProfilerLevel,
        _ExperimentalConfig, profile, schedule, tensorboard_trace_handler)
    profile_dir = output_dir / "profile_complete"
    profile_dir.mkdir(parents=True, exist_ok=True)
    call, _ = make_bench(api, "npu", npu_device)
    with torch.no_grad():
        for _ in range(5): call()
        torch.npu.synchronize()
        cfg = _ExperimentalConfig(profiler_level=ProfilerLevel.Level1,
            aic_metrics=AiCMetrics.PipeUtilization, data_simplification=False,
            export_type="text")
        with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.NPU],
             schedule=schedule(wait=0, warmup=0, active=1, repeat=1),
             record_shapes=True, experimental_config=cfg,
             on_trace_ready=tensorboard_trace_handler(str(profile_dir), analyse_flag=True)) as prof:
            with torch.autograd.profiler.record_function("case_" + api):
                call(); torch.npu.synchronize()
            prof.step()
    return {"success": True, "path": str(profile_dir),
            "files": [str(p.relative_to(output_dir)) for p in profile_dir.rglob("*") if p.is_file()]}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--api", required=True, choices=list(GLOBAL_FUNCS) + ["TopKPooling", "SAGPooling"])
    p.add_argument("--output-dir", required=True)
    p.add_argument("--device", default="npu:1")
    a = p.parse_args(); outdir = Path(a.output_dir); outdir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(20260916); torch.npu.set_device(a.device)
    result = {"api": a.api, "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "environment": {"platform": platform.platform(), "python": platform.python_version(),
        "torch": torch.__version__, "torch_npu": torch_npu.__version__,
        "torch_geometric": torch_geometric.__version__, "device": a.device,
        "ascend_home_path": os.environ.get("ASCEND_HOME_PATH")},
        "cases": [], "performance": {}, "profile": {}, "errors": []}
    runner = run_global_case if a.api in GLOBAL_FUNCS else run_hier_case
    for dtype_name in DTYPES:
        try: result["cases"].append(runner(a.api, dtype_name, a.device))
        except Exception as exc: result["cases"].append({"dtype": dtype_name, "success": False,
            "error": repr(exc), "traceback": traceback.format_exc()})
    try:
        result["performance"]["cpu"] = benchmark(a.api, "cpu", a.device)
        result["performance"]["npu_wall"] = benchmark(a.api, "npu", a.device)
        result["performance"]["speedup_cpu_over_npu"] = result["performance"]["cpu"]["median_ms"] / result["performance"]["npu_wall"]["median_ms"]
    except Exception as exc: result["errors"].append({"stage": "benchmark", "error": repr(exc), "traceback": traceback.format_exc()})
    try: result["profile"] = collect_profile(a.api, outdir, a.device)
    except Exception as exc: result["profile"] = {"success": False, "error": repr(exc), "traceback": traceback.format_exc()}
    result["functional_success"] = all(c.get("success", False) for c in result["cases"])
    (outdir / "result.json").write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__": main()
