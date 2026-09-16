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
from torch_geometric.nn import GraphNorm


DTYPES = {
    "fp32": torch.float32,
    "fp16": torch.float16,
    "bf16": torch.bfloat16,
}


def tensor_metrics(actual: torch.Tensor, expected: torch.Tensor):
    a = actual.detach().float().cpu()
    e = expected.detach().float().cpu()
    diff = (a - e).abs()
    denom = e.abs().clamp_min(1e-12)
    return {
        "shape_equal": list(a.shape) == list(e.shape),
        "max_abs": float(diff.max()) if diff.numel() else 0.0,
        "max_rel": float((diff / denom).max()) if diff.numel() else 0.0,
        "nan_equal": bool(torch.equal(torch.isnan(a), torch.isnan(e))),
        "inf_equal": bool(torch.equal(torch.isinf(a), torch.isinf(e))),
    }


def independent_graphnorm(x, batch, weight, bias, mean_scale, eps, batch_size):
    mean = torch.zeros(batch_size, x.size(1), dtype=x.dtype, device=x.device)
    mean.index_add_(0, batch, x)
    count = torch.bincount(batch, minlength=batch_size).to(x.dtype).unsqueeze(1)
    mean = mean / count
    centered = x - mean.index_select(0, batch) * mean_scale
    var = torch.zeros_like(mean)
    var.index_add_(0, batch, centered.square())
    var = var / count
    return weight * centered / torch.sqrt(var.index_select(0, batch) + eps) + bias


def make_module(dtype, device):
    module = GraphNorm(8, eps=1e-5).to(device=device, dtype=dtype)
    with torch.no_grad():
        module.weight.copy_(torch.linspace(0.75, 1.25, 8, dtype=dtype, device=device))
        module.bias.copy_(torch.linspace(-0.2, 0.2, 8, dtype=dtype, device=device))
        module.mean_scale.copy_(torch.linspace(0.6, 1.1, 8, dtype=dtype, device=device))
    return module


def tolerances(name):
    if name == "fp32":
        return 1e-4, 1e-5
    return 2e-2, 2e-2


def run_case(dtype_name, npu_device):
    dtype = DTYPES[dtype_name]
    batch_cpu = torch.tensor([0, 0, 0, 1, 1, 1, 1, 1, 1], dtype=torch.long)
    x_master = torch.tensor([
        [-1.2, 0.1, 1.0, 2.0, -0.5, 3.0, 0.25, -2.0],
        [0.3, -0.7, 2.2, 1.0, 0.5, -1.0, 1.25, 0.2],
        [1.1, 0.9, -0.4, 0.3, 2.0, 0.5, -0.75, 1.5],
        [2.0, 1.0, -1.0, 0.0, 3.0, -2.0, 0.5, 1.0],
        [1.5, -2.0, 0.5, 2.0, -1.0, 1.0, 2.5, -0.5],
        [-0.5, 3.0, 1.5, -1.0, 0.0, 2.0, -1.5, 2.5],
        [0.7, 1.7, -2.0, 1.2, 2.2, -0.8, 1.8, 0.6],
        [3.0, -1.0, 2.0, -2.0, 1.0, 0.0, -0.2, 1.7],
        [-1.0, 0.5, 0.0, 3.0, -2.0, 1.5, 0.7, -1.2],
    ], dtype=torch.float32)
    upstream = torch.linspace(-0.9, 1.1, x_master.numel()).reshape_as(x_master)

    cpu_module = make_module(torch.float32, "cpu")
    x_cpu = x_master.clone().requires_grad_(True)
    cpu_out = cpu_module(x_cpu, batch_cpu, batch_size=2)
    formula_out = independent_graphnorm(
        x_cpu, batch_cpu, cpu_module.weight, cpu_module.bias,
        cpu_module.mean_scale, cpu_module.eps, 2)
    formula_metrics = tensor_metrics(cpu_out, formula_out)
    cpu_out.backward(upstream)
    cpu_grads = {
        "x": x_cpu.grad.detach().clone(),
        "weight": cpu_module.weight.grad.detach().clone(),
        "bias": cpu_module.bias.grad.detach().clone(),
        "mean_scale": cpu_module.mean_scale.grad.detach().clone(),
    }

    npu_module = make_module(dtype, npu_device)
    x_npu = x_master.to(device=npu_device, dtype=dtype).requires_grad_(True)
    batch_npu = batch_cpu.to(npu_device)
    npu_out = npu_module(x_npu, batch_npu, batch_size=2)
    npu_out.backward(upstream.to(device=npu_device, dtype=dtype))
    torch.npu.synchronize()

    rtol, atol = tolerances(dtype_name)
    out_cpu = npu_out.detach().float().cpu()
    output_metrics = tensor_metrics(out_cpu, cpu_out.detach())
    output_allclose = bool(torch.allclose(out_cpu, cpu_out.detach(), rtol=rtol, atol=atol))
    grad_results = {}
    for name, actual in {
        "x": x_npu.grad,
        "weight": npu_module.weight.grad,
        "bias": npu_module.bias.grad,
        "mean_scale": npu_module.mean_scale.grad,
    }.items():
        actual_cpu = actual.detach().float().cpu()
        metrics = tensor_metrics(actual_cpu, cpu_grads[name])
        metrics["allclose"] = bool(torch.allclose(actual_cpu, cpu_grads[name], rtol=rtol, atol=atol))
        grad_results[name] = metrics

    return {
        "dtype": dtype_name,
        "mode": "eager_forward_backward",
        "input_shape": list(x_master.shape),
        "batch_graph_sizes": [3, 6],
        "output_shape": list(npu_out.shape),
        "output_device": str(npu_out.device),
        "formula_reference": formula_metrics,
        "output_accuracy": {**output_metrics, "allclose": output_allclose, "rtol": rtol, "atol": atol},
        "gradient_accuracy": grad_results,
        "success": output_allclose and all(x["allclose"] for x in grad_results.values()),
    }


def percentile(samples, q):
    values = sorted(samples)
    pos = max(0, min(len(values) - 1, math.ceil(q * len(values)) - 1))
    return values[pos]


def benchmark(device, npu_device, warmup=10, iterations=50):
    graph_count, nodes_per_graph, channels = 32, 128, 64
    x = torch.linspace(-2, 2, graph_count * nodes_per_graph * channels,
                       dtype=torch.float32).reshape(graph_count * nodes_per_graph, channels)
    batch = torch.arange(graph_count, dtype=torch.long).repeat_interleave(nodes_per_graph)
    module = GraphNorm(channels).eval()
    if device == "npu":
        module = module.to(npu_device)
        x = x.to(npu_device)
        batch = batch.to(npu_device)
        sync = torch.npu.synchronize
    else:
        sync = lambda: None

    with torch.no_grad():
        for _ in range(warmup):
            module(x, batch, batch_size=graph_count)
        sync()
        samples = []
        for _ in range(iterations):
            sync()
            start = time.perf_counter_ns()
            module(x, batch, batch_size=graph_count)
            sync()
            samples.append((time.perf_counter_ns() - start) / 1e6)
    return {
        "shape": [graph_count * nodes_per_graph, channels],
        "graphs": graph_count,
        "nodes_per_graph": nodes_per_graph,
        "warmup": warmup,
        "iterations": iterations,
        "median_ms": statistics.median(samples),
        "p90_ms": percentile(samples, 0.90),
        "min_ms": min(samples),
        "max_ms": max(samples),
    }


def collect_profile(output_dir, npu_device):
    from torch_npu.profiler import (
        AiCMetrics, ProfilerActivity, ProfilerLevel, _ExperimentalConfig,
        profile, schedule, tensorboard_trace_handler,
    )

    profile_dir = output_dir / "profile_complete"
    profile_dir.mkdir(parents=True, exist_ok=True)
    graph_count, nodes_per_graph, channels = 32, 128, 64
    x = torch.linspace(-2, 2, graph_count * nodes_per_graph * channels,
                       dtype=torch.float32, device=npu_device).reshape(-1, channels)
    batch = torch.arange(graph_count, dtype=torch.long, device=npu_device).repeat_interleave(nodes_per_graph)
    module = GraphNorm(channels).to(npu_device).eval()
    with torch.no_grad():
        for _ in range(5):
            module(x, batch, batch_size=graph_count)
        torch.npu.synchronize()
        config = _ExperimentalConfig(
            profiler_level=ProfilerLevel.Level1,
            aic_metrics=AiCMetrics.PipeUtilization,
            data_simplification=False,
            export_type="text",
        )
        with profile(
            activities=[ProfilerActivity.CPU, ProfilerActivity.NPU],
            schedule=schedule(wait=0, warmup=0, active=1, repeat=1),
            record_shapes=True,
            profile_memory=False,
            with_stack=False,
            experimental_config=config,
            on_trace_ready=tensorboard_trace_handler(str(profile_dir), analyse_flag=True),
        ) as prof:
            with torch.autograd.profiler.record_function("case_graphnorm_fp32"):
                module(x, batch, batch_size=graph_count)
                torch.npu.synchronize()
            prof.step()
    files = [str(p.relative_to(output_dir)) for p in profile_dir.rglob("*") if p.is_file()]
    return {"success": True, "path": str(profile_dir), "files": files}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default="npu:1")
    args = parser.parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    result = {
        "api": "GraphNorm",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "environment": {
            "platform": platform.platform(),
            "python": platform.python_version(),
            "torch": torch.__version__,
            "torch_npu": torch_npu.__version__,
            "torch_geometric": torch_geometric.__version__,
            "npu_available": bool(torch.npu.is_available()),
            "npu_count": int(torch.npu.device_count()),
            "device": args.device,
            "ascend_home_path": os.environ.get("ASCEND_HOME_PATH"),
        },
        "cases": [],
        "performance": {},
        "profile": {},
        "errors": [],
    }
    torch.manual_seed(20260916)
    torch.npu.set_device(args.device)
    for dtype_name in DTYPES:
        try:
            result["cases"].append(run_case(dtype_name, args.device))
        except Exception as exc:
            result["cases"].append({
                "dtype": dtype_name,
                "success": False,
                "error": repr(exc),
                "traceback": traceback.format_exc(),
            })
    try:
        result["performance"]["cpu"] = benchmark("cpu", args.device)
        result["performance"]["npu_wall"] = benchmark("npu", args.device)
        result["performance"]["speedup_cpu_over_npu"] = (
            result["performance"]["cpu"]["median_ms"] /
            result["performance"]["npu_wall"]["median_ms"])
    except Exception as exc:
        result["errors"].append({"stage": "benchmark", "error": repr(exc), "traceback": traceback.format_exc()})
    try:
        result["profile"] = collect_profile(output_dir, args.device)
    except Exception as exc:
        result["profile"] = {"success": False, "error": repr(exc), "traceback": traceback.format_exc()}

    result["functional_success"] = all(c.get("success", False) for c in result["cases"])
    result_path = output_dir / "result.json"
    result_path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
