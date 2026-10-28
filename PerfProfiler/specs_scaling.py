"""Source-profile-only implementation of EcoSched Eq:specs and Eq:iterscaling.
"""
from __future__ import annotations

import csv
import json
import math
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path

DEFAULT_SPECS = Path(__file__).with_name("gpu_specs.json")
# These are fixed priors, NOT measured memory-bandwidth utilization.
# GEMM operators with usable shapes override the prior with a roofline proxy.
BETA_PRIORS = {"Conv": 0.10, "GEMM": 0.10, "Elemwise": 0.95,
               "Softmax": 0.80, "Pool": 0.70, "Norm": 0.85,
               "Reduce": 0.85, "Memory": 1.0, "Other": 0.50}


@dataclass(frozen=True)
class GPUSpec:
    name: str
    sm_count: int
    clock_mhz: float
    bandwidth_gbps: float
    fp32_tflops: float
    memory_gb: float
    threads_per_sm: int
    blocks_per_sm: int
    registers_per_sm: int
    shared_bytes_per_sm: int
    shared_bytes_per_block: int
    max_threads_per_block: int

    def __post_init__(self):
        for key, value in asdict(self).items():
            if key != "name" and (not math.isfinite(value) or value <= 0):
                raise ValueError(f"Invalid hardware parameter {key}={value}")


def load_specs(path=DEFAULT_SPECS):
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return {key: GPUSpec(**value) for key, value in data["gpus"].items()}


def category(operator, kernel, original="Other"):
    """Refine the original paper categories without using workload names."""
    name = (operator + " " + kernel).lower()
    if "gemv" in name:
        return "GEMM"
    if original in BETA_PRIORS and original != "Other":
        return original
    if "conv" in name or "cudnn" in name:
        return "Conv"
    if any(x in name for x in ("aten::mm", "aten::bmm", "addmm", "gemm")):
        return "GEMM"
    if any(x in name for x in ("norm", "batchnorm")):
        return "Norm"
    if "softmax" in name:
        return "Softmax"
    if any(x in name for x in ("reduce", "cumsum", "aten::sum")):
        return "Reduce"
    if any(x in name for x in ("copy", "fill", "indexselect", "gather", "scatter", "embedding")):
        return "Memory"
    if "elementwise" in name or "vectorized" in name:
        return "Elemwise"
    return "Other"


def gemm_work(shapes, operator):
    """Return FLOPs and elements moved for mm/bmm/addmm, including output.

    This is a compulsory-traffic estimate; it does not claim to measure DRAM
    transactions or cache hits. Operator records contain shapes, not counters.
    """
    offset = 1 if operator == "aten::addmm" else 0
    if operator not in {"aten::mm", "aten::bmm", "aten::addmm"}:
        return None
    try:
        a, b = shapes[offset], shapes[offset + 1]
        if len(a) < 2 or len(a) != len(b) or a[-1] != b[-2]:
            return None
        if a[:-2] != b[:-2] or any(x <= 0 for x in a + b):
            return None
        batch = math.prod(a[:-2])
        m, k, n = a[-2], a[-1], b[-1]
        return 2 * batch * m * k * n, batch * (m * k + k * n + m * n)
    except (TypeError, IndexError):
        return None


@dataclass(frozen=True)
class Kernel:
    name: str
    operator: str
    category: str
    duration_us: float
    blocks: int
    threads: int
    registers: int
    shared_bytes: int
    beta: float
    beta_method: str
    grid_known: bool


@dataclass
class Profile:
    workload: str
    batch_size: int
    source_gpu: str
    kernels: list[Kernel]
    kernel_sum_us: float
    busy_us: float
    span_us: float
    scope: str
    input_tokens: int
    output_tokens: int
    warnings: list[str]
    uninstrumented_ms: float | None = None
    dtype: str | None = None


def interval_union(intervals):
    if not intervals:
        return 0.0
    intervals = sorted(intervals)
    start, end = intervals[0]
    total = 0.0
    for a, b in intervals[1:]:
        if a > end:
            total += end - start
            start, end = a, b
        else:
            end = max(end, b)
    return total + end - start


def _number(row, key, default=0):
    value = row.get(key, "")
    number = float(value) if value not in ("", None) else default
    if not math.isfinite(number) or number < 0:
        raise ValueError(f"Invalid kernel field {key}={value}")
    return number


def load_profile(directory, workload, batch_size, source_gpu, spec):
    """Read only one explicitly selected source-GPU profile directory."""
    directory = Path(directory)
    warnings = []
    shapes = defaultdict(list)
    operators = directory / "operators.csv"
    with operators.open(encoding="utf-8-sig", newline="") as stream:
        for row in csv.DictReader(stream):
            # Never add ATen total time to CUDA time: that double-counts work.
            if row.get("record_type") != "aten_operator":
                continue
            try:
                shape = json.loads(row.get("input_shapes", "[]"))
                work = gemm_work(shape, row["operator"])
            except (json.JSONDecodeError, ValueError):
                work = None
            if work:
                weight = max(float(row.get("self_cuda_us") or 0), 0)
                shapes[row["operator"]].append((work, weight))

    kernels = []
    kernel_path = directory / "kernels.csv"
    with kernel_path.open(encoding="utf-8-sig", newline="") as stream:
        for row in csv.DictReader(stream):
            name, op = row["kernel"], row.get("operator", "")
            cat = category(op, name, row.get("paper_category", "Other"))
            beta, method = BETA_PRIORS[cat], "category_prior"
            candidates = shapes.get(op, [])
            half = any(s in name.lower() for s in ("half", "fp16", "bf16"))
            if candidates and sum(weight for _, weight in candidates) > 0:
                element_bytes = 2 if half else 4
                # Use a source-only roofline proxy: beta = M/(M+C).
                # FP32-equivalent peak is conservative for unknown instruction
                # paths. No Tensor Core use is inferred solely from dtype.
                weighted = []
                for (flops, elements), weight in candidates:
                    memory_s = elements * element_bytes / (spec.bandwidth_gbps * 1e9)
                    compute_s = flops / (spec.fp32_tflops * 1e12)
                    weighted.append((memory_s / (memory_s + compute_s), weight))
                beta = sum(b * w for b, w in weighted) / sum(w for _, w in weighted)
                method = "shape_roofline_proxy"
            elif "gemv" in name.lower():
                beta, method = 0.95, "gemv_prior"
            blocks = int(_number(row, "thread_blocks"))
            threads = math.prod(int(_number(row, f"block_{axis}", 1)) for axis in "xyz")
            kernels.append(Kernel(name, op, cat, _number(row, "duration_us"),
                                  max(blocks, 1), max(threads, 1),
                                  int(_number(row, "registers_per_thread")),
                                  int(_number(row, "shared_memory_bytes")), beta,
                                  method, blocks > 0 and threads > 0 and all(
                    row.get(f"block_{axis}") not in (None, "") for axis in "xyz")))
    if not kernels or sum(k.duration_us for k in kernels) <= 0:
        raise ValueError(f"No positive CUDA kernel time in {directory}")

    trace_path = directory / "trace.json"
    trace = json.loads(trace_path.read_text(encoding="utf-8"))
    events = [e for e in trace.get("traceEvents", [])
              if str(e.get("cat", "")).lower() == "kernel" and e.get("ph") == "X"]
    devices = {e.get("args", {}).get("device") for e in events}
    if len(devices) != 1:
        raise ValueError("Only single-device source profiles are supported")
    active_device = next(iter(devices))
    props = [p for p in trace.get("deviceProperties", []) if p.get("id") == active_device]
    if len(props) != 1 or source_gpu.lower() not in props[0].get("name", "").lower():
        raise ValueError(f"Trace must identify one active {source_gpu} device: {directory}")
    if props[0].get("numSms") != spec.sm_count:
        raise ValueError("Source SM count disagrees with configured hardware")
    if len(events) != len(kernels):
        raise ValueError("Kernel CSV and trace event counts disagree")
    trace_sum = sum(float(e["dur"]) for e in events)
    kernel_sum = sum(k.duration_us for k in kernels)
    if not math.isclose(trace_sum, kernel_sum, rel_tol=1e-6, abs_tol=1):
        raise ValueError("Kernel CSV and trace durations disagree")
    intervals = [(float(e["ts"]), float(e["ts"]) + float(e["dur"])) for e in events]
    busy = interval_union(intervals)
    span = max(end for _, end in intervals) - min(start for start, _ in intervals)
    # Legacy Qwen traces are a single decode token; other legacy traces are a
    # training iteration. New profiler metadata overrides this legacy mapping.
    scope = "decode" if workload == "Qwen" else "iteration"
    input_tokens, output_tokens = 64, 32
    dtype = "fp16" if workload == "Qwen" else None
    metadata_path = directory / "metadata.json"
    uninstrumented_ms = None
    if metadata_path.exists():
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        config = metadata.get("metadata", metadata.get("extra_metadata", metadata))
        if config.get("workload", workload) != workload or int(config.get("batch_size", batch_size)) != batch_size:
            raise ValueError("Profile metadata and requested workload/batch disagree")
        scope = config.get("qwen_profile_scope", scope) if workload == "Qwen" else scope
        scope = config.get("profile_scope", scope)
        dtype = config.get("qwen_dtype", dtype) if workload == "Qwen" else config.get("dtype", dtype)
        input_tokens = int(config.get("qwen_input_tokens", input_tokens))
        output_tokens = int(config.get("qwen_output_tokens", output_tokens))
        uninstrumented_ms = metadata.get("uninstrumented_iteration_ms")
        if uninstrumented_ms is not None:
            uninstrumented_ms = float(uninstrumented_ms)
            if not math.isfinite(uninstrumented_ms) or uninstrumented_ms <= 0:
                raise ValueError("Invalid source uninstrumented iteration time")
    else:
        warnings.append("Missing metadata: scope/token lengths inferred from repository profiler defaults.")
    if scope not in {"iteration", "decode", "prefill", "request"}:
        raise ValueError(f"Unsupported profile scope {scope}")
    if min(input_tokens, output_tokens) < 1:
        raise ValueError("Token counts must be positive")
    warnings.append("Trace gaps include profiler/host overhead; they are not uninstrumented latency measurements.")
    warnings.append("Target library kernel selection, clocks under load and cache behavior are unknown.")
    if any(not k.grid_known for k in kernels):
        warnings.append("Some launch dimensions are missing; affected kernels use saturated-wave scaling.")
    return Profile(workload, batch_size, source_gpu, kernels, kernel_sum,
                   busy, span, scope, input_tokens, output_tokens, warnings, uninstrumented_ms, dtype)


def capacity(kernel, spec):
    """Estimate B_g from CTA resources, not the CSV's grid/SM estimate.

    Register allocation is rounded to 256-register warp units. This is a static
    approximation, not CUDA occupancy API output for a compiled target binary.
    A zero capacity means the source launch cannot be transplanted as-is.
    """
    if kernel.threads > spec.max_threads_per_block or kernel.shared_bytes > spec.shared_bytes_per_block:
        return 0
    warps = math.ceil(kernel.threads / 32)
    limits = [spec.blocks_per_sm, spec.threads_per_sm // (warps * 32)]
    if kernel.registers:
        warp_registers = math.ceil(kernel.registers * 32 / 256) * 256
        limits.append(spec.registers_per_sm // (warp_registers * warps))
    if kernel.shared_bytes:
        shared = math.ceil(kernel.shared_bytes / 256) * 256
        limits.append(spec.shared_bytes_per_sm // shared)
    return spec.sm_count * min(limits)


def scaling_factor(kernel, source, target, variant="paper"):
    """EcoSched Eq:specs: scale_o = gamma_o * delta_o.

    gamma = ceil(B_o/B_target) / ceil(B_o/B_source)
    delta = (D_source/D_target)^beta * (B_target/B_source)^beta
            * (f_source/f_target)^(1-beta).
    The optional 'architecture' extension includes per-SM FP32 IPC differences;
    the default 'paper' variant uses the paper equation without that extension.
    """
    base, other = capacity(kernel, source), capacity(kernel, target)
    if not base:
        raise ValueError(f"Infeasible source launch: {kernel.name[:100]}")
    if variant not in {"paper", "architecture"}:
        raise ValueError(f"Unknown scaling variant {variant}")
    launch_model = "resource_limited_waves"
    if not other:
        # A source CTA may exceed the target shared-memory/register limits.
        # A target library must choose a different kernel. Its tile dimensions
        # cannot be recovered from the source trace. Use Eq:specs' saturated
        # limit with an equivalent one-CTA-per-SM assumption and flag it.
        base, other = source.sm_count, target.sm_count
        gamma = base / other
        launch_model = "saturated_equivalent_kernel_proxy"
    else:
        gamma = (math.ceil(kernel.blocks / other) / math.ceil(kernel.blocks / base)
                 if kernel.grid_known else base / other)
    beta = kernel.beta
    delta = ((source.bandwidth_gbps / target.bandwidth_gbps) ** beta
             * (other / base) ** beta
             * (source.clock_mhz / target.clock_mhz) ** (1 - beta))
    ipc_factor = 1.0
    if variant == "architecture":
        source_ipc = source.fp32_tflops / source.sm_count / source.clock_mhz
        target_ipc = target.fp32_tflops / target.sm_count / target.clock_mhz
        ipc_factor = (source_ipc / target_ipc) ** (1 - beta)
    return {"gamma": gamma, "delta": delta, "ipc_factor": ipc_factor,
            "scaling_factor": gamma * delta * ipc_factor,
            "source_capacity": base, "target_capacity": other,
            "launch_model": launch_model}


# # Calculate the absolute difference between estimate_iteration_time_results and measure_results
# difference_parallel_estimate_time_normal_results = {}
# for workload in workload_list:
#     difference_parallel_estimate_time_normal_results[workload] = {}
#     batch_size = workload_batch_size_mapping[workload][1]
#     difference_parallel_estimate_time_normal_results[workload][batch_size] = {}
#     # print(f"-----------------\n{workload}\n-----------------")
#     for gpu_model in GPU_model_list:
#         difference_parallel_estimate_time_normal_results[workload][batch_size][gpu_model] = {}
#         # print(f"-----------------\n{gpu_model}\n-----------------")
#         for gpu_num in gpu_parallel_list:
#             estimate_parallel_execute_time = parallel_estimate_time_normal_results[workload][batch_size][gpu_model][gpu_num]
#             measure_parallel_execute_time = parallel_measure_time_normal_results[workload][batch_size][gpu_model][gpu_num]
#             difference_parallel_estimate_time_normal_results[workload][batch_size][gpu_model][gpu_num] = (estimate_parallel_execute_time - measure_parallel_execute_time) / measure_parallel_execute_time
#             # print(difference_parallel_estimate_time_normal_results[workload][batch_size][gpu_model][gpu_num])

def predict(profile, source, target, variant="paper", output_tokens=None,
            input_tokens=None, prefill=None):
    """Predict an iteration, or approximate a request from a decode profile.

    Native scope: T_gpu' = sum_o(gamma_o*delta_o*T_o), Eq:iterscaling.
    Wall extension: T' = overlap_source*T_gpu' + unscaled_source_gap.
    Request extension: T_prefill' + (N_out-1)*T_decode'. The prefill forward
    supplies token 1. A decode-only trace cannot identify prefill accurately.
    """
    if output_tokens is not None and output_tokens < 1:
        raise ValueError("output_tokens must be positive")
    if input_tokens is not None and input_tokens < 1:
        raise ValueError("input_tokens must be positive")
    diagnostics = defaultdict(lambda: defaultdict(float))
    predicted_gpu_us = 0.0
    prefill_gpu_proxy_us = 0.0
    n_in = profile.input_tokens if input_tokens is None else input_tokens
    n_out = profile.output_tokens if output_tokens is None else output_tokens
    for kernel in profile.kernels:
        factor = scaling_factor(kernel, source, target, variant)
        time_us = kernel.duration_us * factor["scaling_factor"]
        predicted_gpu_us += time_us
        d = diagnostics[kernel.category]
        d["source_us"] += kernel.duration_us
        d["predicted_us"] += time_us
        d["kernel_count"] += 1
        d["shape_beta_kernel_count"] += kernel.beta_method == "shape_roofline_proxy"
        d["retiled_proxy_kernel_count"] += factor["launch_model"] != "resource_limited_waves"
        d["weighted_beta"] += kernel.beta * kernel.duration_us
        for key in ("gamma", "delta", "source_capacity", "target_capacity"):
            d["weighted_" + key] += factor[key] * kernel.duration_us

        prefill_gpu_proxy_us += time_us * (kernel.beta + (1 - kernel.beta) * n_in)
    for d in diagnostics.values():
        for key in list(d):
            if key.startswith("weighted_"):
                d[key.replace("weighted_", "mean_")] = d.pop(key) / max(d["source_us"], 1e-12)
        d["scaling_factor"] = d["predicted_us"] / max(d["source_us"], 1e-12)
    anchor_us = (profile.uninstrumented_ms * 1000 if profile.uninstrumented_ms is not None
                 else profile.span_us)
    effective_busy_us = min(profile.busy_us, anchor_us)
    overlap = effective_busy_us / profile.kernel_sum_us
    gap = max(0.0, anchor_us - effective_busy_us)
    native_us = overlap * predicted_gpu_us + gap
    warnings = list(profile.warnings)
    if any(d["retiled_proxy_kernel_count"] for d in diagnostics.values()):
        warnings.append(
            "Some source CTAs cannot fit the target; saturated equivalent-kernel proxies were used (see categories).")
    request_us = None
    prefill_ms = None
    status = "source_profile_estimate"
    if profile.scope == "decode":
        if prefill is not None:
            if (prefill.scope != "prefill" or prefill.workload != profile.workload
                    or prefill.batch_size != profile.batch_size
                    or prefill.source_gpu != profile.source_gpu
                    or prefill.input_tokens != n_in):
                raise ValueError("Prefill profile must match source, workload, batch and input length")
            prefill_us = predict(prefill, source, target, variant)["native_iteration_ms"] * 1000
        else:
            prefill_us = overlap * prefill_gpu_proxy_us + gap
            warnings.append("No prefill trace: prefill is a roofline-based proxy, not measured source work.")
        request_us = prefill_us + (n_out - 1) * native_us
        prefill_ms = prefill_us / 1000
        status = "request_extrapolation_from_decode"
        warnings.append(
            "Decode context growth, token selection and host changes are unmeasured; request accuracy is unverified.")
    elif profile.scope == "request":
        if n_in != profile.input_tokens or n_out != profile.output_tokens:
            raise ValueError("A full-request trace cannot be relabeled with different token lengths")
        request_us = native_us
    primary_ms = (request_us if request_us is not None else native_us) / 1000
    return {"workload": profile.workload, "batch_size": profile.batch_size,
            "source_gpu": profile.source_gpu, "target_gpu_name": target.name,
            "variant": variant, "scope": profile.scope, "status": status,
            "dtype": profile.dtype,
            "predicted_ms": primary_ms, "native_iteration_ms": native_us / 1000,
            "request_ms": request_us / 1000 if request_us is not None else None,
            "prefill_ms": prefill_ms, "input_tokens": n_in if profile.scope != "iteration" else None,
            "output_tokens": n_out if profile.scope != "iteration" else None,
            "source_kernel_sum_ms": profile.kernel_sum_us / 1000,
            "source_busy_ms": profile.busy_us / 1000,
            "source_span_ms": profile.span_us / 1000,
            "source_gap_ms": gap / 1000, "source_overlap_ratio": overlap,
            "source_latency_anchor_ms": anchor_us / 1000,
            "anchor_kind": "uninstrumented_source" if profile.uninstrumented_ms is not None else "instrumented_trace_span",
            "paper_kernel_sum_ms": predicted_gpu_us / 1000,
            "kernel_scaling_factor": predicted_gpu_us / profile.kernel_sum_us,
            "categories": dict(diagnostics), "warnings": warnings}
