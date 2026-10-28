#!/usr/bin/env python3
"""Capture one callable training iteration with PyTorch operator and CUDA traces."""
import argparse
import csv
import json
import os
import platform
import time
from pathlib import Path

import torch
from torch.profiler import profile, ProfilerActivity, record_function


def paper_operator_category(name):
    """Map ATen/CUDA operator names to the paper's five operator groups."""
    n = name.lower()
    if any(k in n for k in ("softmax", "log_softmax")):
        return "Softmax"
    if any(k in n for k in ("max_pool", "avg_pool", "adaptive_avg_pool", "adaptive_max_pool", "pooling")):
        return "Pool"
    if any(k in n for k in ("conv", "convolution", "cudnn_convolution")):
        return "Conv"
    if any(k in n for k in ("matmul", "mm", "addmm", "bmm", "linear", "gemm", "cublas")):
        return "GEMM"
    if any(k in n for k in ("relu", "sigmoid", "tanh", "gelu", "silu", "add", "mul", "div", "sub", "batch_norm", "native_batch_norm", "pointwise", "elementwise", "where", "clamp", "threshold", "copy_", "cat", "reshape", "view", "transpose", "contiguous")):
        return "Elemwise"
    return "Other"


def run_ncu_target(step_fn, warmup=3):
    """Run warmups then mark exactly one iteration for Nsight Compute NVTX filtering."""
    for _ in range(warmup):
        step_fn()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.nvtx.range_push("TARGET_ITERATION")
    try:
        step_fn()
    finally:
        if torch.cuda.is_available():
            torch.cuda.nvtx.range_pop()
            torch.cuda.synchronize()


def profile_iteration(step_fn, output_dir="profile_output", warmup=3, metadata=None,
                      prepare_fn=None, baseline_iterations=0):
    """Run warmups, then profile exactly one invocation of step_fn()."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    for _ in range(warmup):
        if prepare_fn is not None:
            prepare_fn()
        step_fn()
    # Source-only wall-time measurements separate instrumentation overhead
    # from latency. Preparation is excluded, exactly as in the traced step.
    if baseline_iterations < 0:
        raise ValueError("baseline_iterations must be nonnegative")
    baseline_ms = []
    for _ in range(baseline_iterations):
        if prepare_fn is not None:
            prepare_fn()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        baseline_start = time.perf_counter()
        step_fn()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        baseline_ms.append((time.perf_counter() - baseline_start) * 1000)
    if prepare_fn is not None:
        prepare_fn()
    if torch.cuda.is_available():
        torch.cuda.synchronize()

    activities = [ProfilerActivity.CPU]
    if torch.cuda.is_available():
        activities.append(ProfilerActivity.CUDA)
    start = time.perf_counter()
    with profile(activities=activities, record_shapes=True, profile_memory=True,
                 with_stack=False) as prof:
        with record_function("TARGET_ITERATION"):
            step_fn()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    elapsed_ms = (time.perf_counter() - start) * 1000

    trace_path = out / "trace.json"
    prof.export_chrome_trace(str(trace_path))

    # CUPTI kernel activity records in the Chrome trace expose launch geometry
    # and resource metadata without Nsight Compute hardware-counter access.
    trace = json.loads(trace_path.read_text())
    trace_events = trace.get("traceEvents", [])
    op_by_external_id = {}
    for event in trace_events:
        args = event.get("args") or {}
        external_id = args.get("External id")
        if external_id is not None and event.get("cat") == "cpu_op":
            op_by_external_id[external_id] = event.get("name", "")
    kernel_path = out / "kernels.csv"
    kernel_fields = ["kernel", "operator", "paper_category", "duration_us", "device", "context", "stream",
                     "correlation_id", "grid_x", "grid_y", "grid_z", "block_x", "block_y", "block_z",
                     "thread_blocks", "registers_per_thread", "shared_memory_bytes", "blocks_per_sm_est",
                     "warps_per_sm_est", "occupancy_est_pct"]
    with kernel_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=kernel_fields)
        writer.writeheader()
        for event in trace_events:
            if str(event.get("cat", "")).lower() != "kernel":
                continue
            args = event.get("args") or {}
            grid = args.get("grid", [None, None, None])
            block = args.get("block", [None, None, None])
            corr = args.get("correlation")
            op_name = op_by_external_id.get(args.get("External id"), op_by_external_id.get(corr, ""))
            writer.writerow({"kernel": event.get("name", ""), "operator": op_name,
                             "paper_category": paper_operator_category(op_name) if op_name else "Unmapped",
                             "duration_us": event.get("dur", 0), "device": args.get("device"),
                             "context": args.get("context"), "stream": args.get("stream"),
                             "correlation_id": corr, "grid_x": grid[0] if len(grid)>0 else None,
                             "grid_y": grid[1] if len(grid)>1 else None, "grid_z": grid[2] if len(grid)>2 else None,
                             "block_x": block[0] if len(block)>0 else None,
                             "block_y": block[1] if len(block)>1 else None, "block_z": block[2] if len(block)>2 else None,
                             "thread_blocks": (grid[0]*grid[1]*grid[2]) if all(v is not None for v in grid[:3]) else None,
                             "registers_per_thread": args.get("registers per thread"),
                             "shared_memory_bytes": args.get("shared memory"),
                             "blocks_per_sm_est": args.get("blocks per SM"),
                             "warps_per_sm_est": args.get("warps per SM"),
                             "occupancy_est_pct": args.get("est. achieved occupancy %")})

    events = sorted(prof.key_averages(group_by_input_shape=True),
                    key=lambda event: event.self_cuda_time_total, reverse=True)
    csv_path = out / "operators.csv"
    fields = ["record_type", "paper_category", "operator", "input_shapes", "calls", "self_cuda_us", "total_cuda_us",
              "self_cpu_us", "total_cpu_us", "cpu_memory_bytes", "cuda_memory_bytes"]
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for event in events:
            writer.writerow({"record_type": "aten_operator" if event.key.startswith("aten::") else "cuda_kernel_or_scope", "paper_category": paper_operator_category(event.key) if event.key.startswith("aten::") else "Kernel", "operator": event.key, "input_shapes": str(event.input_shapes),
                             "calls": event.count, "self_cuda_us": event.self_cuda_time_total,
                             "total_cuda_us": event.cuda_time_total,
                             "self_cpu_us": event.self_cpu_time_total,
                             "total_cpu_us": event.cpu_time_total,
                             "cpu_memory_bytes": event.cpu_memory_usage,
                             "cuda_memory_bytes": event.cuda_memory_usage})

    categories = ["Conv", "GEMM", "Elemwise", "Pool", "Softmax", "Other"]
    category_totals = {name: {"operator_calls": 0, "self_cuda_us": 0.0, "self_cpu_us": 0.0} for name in categories}
    for event in events:
        # Do not add CUDA kernel rows to the framework-operator rollup: that
        # would count the same GPU work twice.
        if not event.key.startswith("aten::"):
            continue
        category = paper_operator_category(event.key)
        category_totals[category]["operator_calls"] += event.count
        category_totals[category]["self_cuda_us"] += event.self_cuda_time_total
        category_totals[category]["self_cpu_us"] += event.self_cpu_time_total
    total_cuda_us = sum(v["self_cuda_us"] for v in category_totals.values())
    summary_path = out / "paper_operator_summary.csv"
    with summary_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["category", "operator_calls", "self_cuda_us", "cuda_time_share_pct", "self_cpu_us", "paper_priority"])
        for category in categories:
            item = category_totals[category]
            share = 100.0 * item["self_cuda_us"] / total_cuda_us if total_cuda_us else 0.0
            priority = "high" if category in ("Conv", "GEMM", "Elemwise") else "lower/diagnostic"
            writer.writerow([category, item["operator_calls"], item["self_cuda_us"], f"{share:.3f}", item["self_cpu_us"], priority])

    device_id = torch.cuda.current_device() if torch.cuda.is_available() else None
    device_properties = torch.cuda.get_device_properties(device_id) if device_id is not None else None
    info = {"torch": torch.__version__, "cuda_runtime": torch.version.cuda,
            "cuda_available": torch.cuda.is_available(), "device":
            torch.cuda.get_device_name(device_id) if device_id is not None else platform.processor(),
            "gpu_sm_count": device_properties.multi_processor_count if device_properties else None,
            "gpu_compute_capability": [device_properties.major, device_properties.minor] if device_properties else None,
            "gpu_total_memory_bytes": device_properties.total_memory if device_properties else None,
            "profiled_iteration_wall_ms": elapsed_ms, "warmup_iterations": warmup,
            "uninstrumented_iteration_ms": sum(baseline_ms) / len(baseline_ms) if baseline_ms else None,
            "uninstrumented_samples_ms": baseline_ms,
            "metadata": metadata or {}, "trace": str(trace_path), "operators_csv": str(csv_path), "paper_operator_summary_csv": str(summary_path), "kernels_csv": str(kernel_path),
            "kernel_metadata_source": "CUPTI activity fields embedded in the PyTorch Chrome trace; occupancy/blocks-per-SM values are estimates, not hardware-counter measurements. Exact block-to-SM placement and hardware utilization counters are unavailable while driver counter access is restricted."}
    (out / "metadata.json").write_text(json.dumps(info, indent=2))
    print(json.dumps(info, indent=2))
    print("Top CUDA operators:\n" + prof.key_averages().table(sort_by="self_cuda_time_total", row_limit=20))
    return prof


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="profile_output")
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--image-size", type=int, default=32)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--ncu-ready", action="store_true", help="Run one NVTX-marked iteration for an external ncu command; omit torch.profiler")
    args = parser.parse_args()

    import sys
    workloads = Path(__file__).resolve().parent.parent
    resnet_source = workloads / "ResNet-PyTorch"
    if (resnet_source / "ResNet.py").is_file():
        sys.path.insert(0, str(resnet_source))
        from ResNet import ResNet50
    else:
        # The source workload may not be mounted in some server images; use
        # torchvision's canonical ResNet-50 as a self-contained fallback.
        from torchvision.models import resnet50
        def ResNet50(num_classes):
            return resnet50(pretrained=False, num_classes=num_classes)

    device = torch.device(args.device)
    model = ResNet50(1000).to(device).train()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    inputs = torch.randn(args.batch_size, 3, args.image_size, args.image_size, device=device)
    labels = torch.randint(0, 1000, (args.batch_size,), device=device)
    scaler = torch.cuda.amp.GradScaler(enabled=args.amp)
    loss_fn = torch.nn.CrossEntropyLoss()

    def step():
        optimizer.zero_grad(set_to_none=True)
        with torch.cuda.amp.autocast(enabled=args.amp):
            loss = loss_fn(model(inputs), labels)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

    if args.ncu_ready:
        run_ncu_target(step, args.warmup)
        print("Completed one TARGET_ITERATION NVTX range for Nsight Compute.")
    else:
        profile_iteration(step, args.output, args.warmup,
                          {"model": "ResNet50", "batch_size": args.batch_size,
                           "image_size": args.image_size, "amp": args.amp})


if __name__ == "__main__":
    main()
