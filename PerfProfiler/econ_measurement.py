"""Power-trace measurement for Eq:ECon, independent of timing predictors."""
from __future__ import annotations

import math
import os
import statistics
import subprocess
import ctypes
import ctypes.util
import threading
import time
import argparse
import importlib.util
import json
from datetime import datetime
from pathlib import Path


def finite(value, name, positive=False):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a number")
    if not math.isfinite(value) or value < 0 or (positive and value == 0):
        raise ValueError(f"Invalid {name}: {value}")
    return float(value)


def integrate(samples, start_s, end_s):
    """Integrate watts using trapezoids with interpolated window boundaries.

    Input timestamps are seconds in a shared monotonic clock domain. Samples
    must bracket the entire window; missing boundary coverage is never filled
    with an assumed power value. Idle and synchronization power are included.
    """
    finite(start_s, "start_s")
    finite(end_s, "end_s", True)
    if end_s <= start_s or len(samples) < 2:
        raise ValueError("A nonempty window and at least two samples are required")
    for t, p in samples:
        finite(t, "timestamp")
        finite(p, "watts")
    if any(b[0] <= a[0] for a, b in zip(samples, samples[1:])):
        raise ValueError("Timestamps must be strictly increasing")
    if samples[0][0] > start_s or samples[-1][0] < end_s:
        raise ValueError("Power samples must bracket the window")
    energy, gaps = 0.0, []
    for (ta, pa), (tb, pb) in zip(samples, samples[1:]):
        a, b = max(ta, start_s), min(tb, end_s)
        if b <= a:
            continue
        slope = (pb - pa) / (tb - ta)
        energy += (b - a) * (pa + slope * (a - ta) + pa + slope * (b - ta)) / 2
        gaps.append(tb - ta)
    return {"energy_j": energy, "mean_power_w": energy / (end_s - start_s),
            "max_sampling_gap_s": max(gaps), "integrated_segments": len(gaps)}


def measure(traces, windows, serial_iterations, baseline_j=None):
    """Sum device integrals, then scale by fixed serial work K / (n * steps).

    Each window is (start_s, end_s, synchronized_step_count). All devices share
    each window. Batch size is LOCAL to a device. Warmup must be outside these
    windows. Under a fixed global batch, supply actual distributed step counts
    instead of using this fixed-local-batch extrapolation.
    """
    finite(serial_iterations, "serial_iterations", True)
    if int(serial_iterations) != serial_iterations or not traces or not windows:
        raise ValueError("Integer work and nonempty traces/windows are required")
    rows, steps, total_j, previous_end = [], 0, 0.0, None
    for start, end, count in windows:
        finite(count, "step_count", True)
        if int(count) != count or (previous_end is not None and start < previous_end):
            raise ValueError("Use integer steps and ordered, non-overlapping windows")
        devices = [integrate(samples, start, end) for samples in traces]
        energy = sum(d["energy_j"] for d in devices)
        rows.append({"steps": count, "device_integrals": devices,
                     "energy_per_distributed_step_j": energy / count})
        total_j += energy
        steps += count
        previous_end = end
    energy = total_j * serial_iterations / (len(traces) * steps)
    step_energies = [row["energy_per_distributed_step_j"] for row in rows]
    return {"origin": "integrated_device_power_traces", "econ_j": energy,
            "econ_kwh": energy / 3_600_000, "sampled_energy_j": total_j,
            "sampled_steps": steps, "gpu_count": len(traces), "windows": rows,
            "between_window_step_energy_stdev_j": statistics.stdev(step_energies) if len(rows) > 1 else None,
            "normalized_econ": None if baseline_j is None else energy / finite(baseline_j, "baseline_j", True)}


class NVMLRecorder:
    """Sample NVML power and utilization using pynvml or the native NVML library.

    Power traces remain available as timestamp/watt pairs for ``measure``.
    Utilization is recorded alongside them as a diagnostic showing whether
    each selected GPU was active during the workload.
    """

    def __init__(self, devices, interval_s=0.01):
        if (not devices or any(isinstance(device, bool) or not isinstance(device, int)
                               or device < 0 for device in devices)
                or len(set(devices)) != len(devices)):
            raise ValueError("Provide distinct NVML device indices")
        self.devices = list(devices)
        self.interval = finite(interval_s, "interval_s", True)
        self.traces = [[] for _ in devices]
        self.utilization_traces = [[] for _ in devices]
        self.stop = threading.Event()
        self.error = None

    @staticmethod
    def _decode(value):
        return value.decode("utf-8", errors="replace") if isinstance(value, bytes) else str(value)

    def _native_check(self, status, operation):
        if status != 0:
            message = "NVML {} failed with status {}".format(operation, status)
            error_string = getattr(self.library, "nvmlErrorString", None)
            if error_string is not None:
                error_string.restype = ctypes.c_char_p
                detail = error_string(status)
                if detail:
                    message += ": " + detail.decode("utf-8", errors="replace")
            raise RuntimeError(message)

    def _load_native(self):
        if os.name == "nt":
            names = [ctypes.util.find_library("nvml"), "nvml.dll"]
        else:
            names = [ctypes.util.find_library("nvidia-ml"), "libnvidia-ml.so.1"]
        last_error = None
        for name in names:
            if not name:
                continue
            try:
                self.library = ctypes.CDLL(name)
                break
            except OSError as error:
                last_error = error
        else:
            raise RuntimeError("Live GPU sampling requires the NVIDIA NVML driver library") from last_error

        init = getattr(self.library, "nvmlInit_v2", None) or self.library.nvmlInit
        init.restype = ctypes.c_int
        self._native_check(init(), "initialization")

        get_handle = (getattr(self.library, "nvmlDeviceGetHandleByIndex_v2", None)
                      or self.library.nvmlDeviceGetHandleByIndex)
        get_handle.argtypes = [ctypes.c_uint, ctypes.POINTER(ctypes.c_void_p)]
        get_handle.restype = ctypes.c_int
        self.handles = []
        for index in self.devices:
            handle = ctypes.c_void_p()
            self._native_check(get_handle(index, ctypes.byref(handle)), "device lookup")
            self.handles.append(handle)

        self._native_power = self.library.nvmlDeviceGetPowerUsage
        self._native_power.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_uint)]
        self._native_power.restype = ctypes.c_int

        class Utilization(ctypes.Structure):
            _fields_ = [("gpu", ctypes.c_uint), ("memory", ctypes.c_uint)]

        self._native_utilization_type = Utilization
        self._native_utilization = getattr(self.library, "nvmlDeviceGetUtilizationRates", None)
        if self._native_utilization is not None:
            self._native_utilization.argtypes = [ctypes.c_void_p, ctypes.POINTER(Utilization)]
            self._native_utilization.restype = ctypes.c_int

        def read_text(symbol, handle):
            function = getattr(self.library, symbol)
            function.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_uint]
            function.restype = ctypes.c_int
            buffer = ctypes.create_string_buffer(96)
            self._native_check(function(handle, buffer, len(buffer)), symbol)
            return buffer.value.decode("utf-8", errors="replace")

        self.device_info = [
            {"index": index, "name": read_text("nvmlDeviceGetName", handle),
             "uuid": read_text("nvmlDeviceGetUUID", handle)}
            for index, handle in zip(self.devices, self.handles)
        ]

    def _load_pynvml(self, pynvml):
        self.nvml = pynvml
        pynvml.nvmlInit()
        self.handles = [pynvml.nvmlDeviceGetHandleByIndex(index) for index in self.devices]
        self.device_info = []
        for index, handle in zip(self.devices, self.handles):
            self.device_info.append({
                "index": index,
                "name": self._decode(pynvml.nvmlDeviceGetName(handle)),
                "uuid": self._decode(pynvml.nvmlDeviceGetUUID(handle)),
            })

    def _read_power(self, handle):
        if self.backend == "pynvml":
            return self.nvml.nvmlDeviceGetPowerUsage(handle) / 1000.0
        milliwatts = ctypes.c_uint()
        self._native_check(self._native_power(handle, ctypes.byref(milliwatts)), "power query")
        return milliwatts.value / 1000.0

    def _read_utilization(self, handle):
        try:
            if self.backend == "pynvml":
                value = self.nvml.nvmlDeviceGetUtilizationRates(handle)
                return {"gpu_pct": int(value.gpu), "memory_pct": int(value.memory)}
            if self._native_utilization is None:
                return None
            value = self._native_utilization_type()
            status = self._native_utilization(handle, ctypes.byref(value))
            if status != 0:
                return None
            return {"gpu_pct": int(value.gpu), "memory_pct": int(value.memory)}
        except Exception:
            return None

    def _sample(self):
        for handle, samples, utilization_samples in zip(
                self.handles, self.traces, self.utilization_traces):
            before = time.perf_counter()
            watts = self._read_power(handle)
            timestamp = (before + time.perf_counter()) / 2
            samples.append((timestamp, watts))
            utilization = self._read_utilization(handle)
            utilization_samples.append({"timestamp_s": timestamp, **(utilization or {})})

    def _run(self):
        try:
            while not self.stop.wait(self.interval):
                self._sample()
        except Exception as error:
            self.error = error

    def __enter__(self):
        try:
            import pynvml
            self.backend = "pynvml"
        except ImportError:
            self.backend = "ctypes"
        try:
            if self.backend == "pynvml":
                self._load_pynvml(pynvml)
            else:
                self._load_native()
            self._sample()
            self.worker = threading.Thread(target=self._run, daemon=True)
            self.worker.start()
        except Exception:
            self._shutdown()
            raise
        return self

    def __exit__(self, exc_type, exc, traceback):
        self.stop.set()
        self.worker.join()
        try:
            if self.error is not None:
                raise RuntimeError("Power sampler failed") from self.error
            self._sample()
        finally:
            self._shutdown()

    def _shutdown(self):
        if getattr(self, "backend", None) == "pynvml" and hasattr(self, "nvml"):
            self.nvml.nvmlShutdown()
        elif getattr(self, "backend", None) == "ctypes" and hasattr(self, "library"):
            shutdown = getattr(self.library, "nvmlShutdown", None)
            if shutdown is not None:
                shutdown()


def record_subprocess(command, gpu_indices, working_directory=None, environment=None,
                      interval_s=0.05, idle_window_s=1.0, baseline_j=None,
                      workload=None, batch_size=None, completed_steps=1):
    """Record per-GPU NVML power while a workload command runs.

    The command receives the selected physical devices through
    CUDA_VISIBLE_DEVICES. Workload startup, initialization, and warm-up remain
    inside the measured interval; use a command that runs only the intended
    measurement phase when those phases should be excluded.
    """
    if (not isinstance(command, (list, tuple)) or not command
            or any(not isinstance(item, str) or not item for item in command)):
        raise ValueError("command must be a nonempty list of command-line strings")
    if (not gpu_indices or any(isinstance(index, bool) or not isinstance(index, int)
                               or index < 0 for index in gpu_indices)
            or len(set(gpu_indices)) != len(gpu_indices)):
        raise ValueError("gpu_indices must contain distinct nonnegative physical GPU indices")
    finite(completed_steps, "completed_steps", True)
    if int(completed_steps) != completed_steps:
        raise ValueError("completed_steps must be a positive integer")
    if workload is not None and (not isinstance(workload, str) or not workload):
        raise ValueError("workload must be a nonempty string")
    if batch_size is not None:
        finite(batch_size, "batch_size", True)
        if int(batch_size) != batch_size:
            raise ValueError("batch_size must be a positive integer")
    finite(interval_s, "interval_s", True)
    finite(idle_window_s, "idle_window_s")
    if environment is not None and (not isinstance(environment, dict)
                                    or any(not isinstance(k, str) or not isinstance(v, str)
                                           for k, v in environment.items())):
        raise ValueError("environment must be a string-to-string mapping")

    child_env = os.environ.copy()
    child_env.update(environment or {})
    child_env["CUDA_VISIBLE_DEVICES"] = ",".join(str(index) for index in gpu_indices)
    traces = None
    with NVMLRecorder(gpu_indices, interval_s) as recorder:
        idle_start = time.perf_counter()
        if idle_window_s:
            time.sleep(idle_window_s)
        idle_end = time.perf_counter()
        start = time.perf_counter()
        completed = subprocess.run(command, cwd=working_directory, env=child_env, check=False)
        end = time.perf_counter()
        traces = recorder.traces
        utilization_traces = recorder.utilization_traces
        device_info = recorder.device_info

    if completed.returncode != 0:
        process_status = "failed"
    else:
        process_status = "completed"

    idle_values = []
    for trace in traces:
        values = [watts for stamp, watts in trace if idle_start <= stamp <= idle_end]
        idle_values.append(statistics.median(values) if values else None)

    result = measure(traces, [(start, end, completed_steps)],
                     len(gpu_indices) * completed_steps, baseline_j=baseline_j)
    duration = end - start
    device_rows = []
    for info, integral, idle_w, trace, utilization_trace in zip(
            device_info, result["windows"][0]["device_integrals"],
            idle_values, traces, utilization_traces):
        idle_energy = None if idle_w is None else idle_w * duration
        active_utilization = [sample for sample in utilization_trace
                              if start <= sample["timestamp_s"] <= end
                              and "gpu_pct" in sample]
        device_rows.append({
            **info,
            "energy_j": integral["energy_j"],
            "mean_power_w": integral["mean_power_w"],
            "idle_reference_power_w": idle_w,
            "idle_corrected_energy_j": (None if idle_energy is None
                                        else max(0.0, integral["energy_j"] - idle_energy)),
            "sample_count": len(trace),
            "max_sampling_gap_s": integral["max_sampling_gap_s"],
            "mean_gpu_utilization_pct": (None if not active_utilization else
                                         statistics.fmean(x["gpu_pct"] for x in active_utilization)),
            "mean_memory_utilization_pct": (None if not active_utilization else
                                            statistics.fmean(x["memory_pct"] for x in active_utilization)),
            "power_samples": [
                {"relative_time_s": stamp - start, "power_w": watts}
                for stamp, watts in trace
                if start - interval_s <= stamp <= end + interval_s
            ],
            "utilization_samples": [
                {"relative_time_s": sample["timestamp_s"] - start,
                 **{key: value for key, value in sample.items() if key != "timestamp_s"}}
                for sample in utilization_trace
                if start - interval_s <= sample["timestamp_s"] <= end + interval_s
            ],
        })

    return {
        **result,
        "origin": "nvml_subprocess_recording",
        "command": list(command),
        "return_code": completed.returncode,
        "process_status": process_status,
        "workload": workload,
        "batch_size": batch_size,
        "elapsed_s": duration,
        "completed_steps": completed_steps,
        "energy_per_distributed_step_j": result["windows"][0]["energy_per_distributed_step_j"],
        "sample_interval_s": interval_s,
        "idle_window_s": idle_window_s,
        "device_records": device_rows,
    }


def _load_workload_module(workload):
    directory, filename = {
        "ResNet50": ("ResNet50", "resnet50_benchmark.py"),
        "CycleGAN": ("CycleGAN", "cyclegan_benchmark.py"),
        "Transformers": ("Transformers", "transformers_benchmark.py"),
        "Qwen": ("Qwen", "qwen3_1_7b_benchmark.py"),
    }[workload]
    path = Path(__file__).resolve().parent / "Workloads" / directory / filename
    spec = importlib.util.spec_from_file_location("econ_trace_" + directory.lower(), path)
    if spec is None or spec.loader is None:
        raise RuntimeError("Cannot load workload implementation: {}".format(path))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _synchronize_all(torch, device_count):
    for index in range(device_count):
        torch.cuda.synchronize(index)


def _build_workload_step(args, torch, gpu_count):
    """Build the same synthetic training/request step used by the benchmarks."""
    import torch.nn as nn
    import torch.optim as optim

    module = _load_workload_module(args.workload)
    primary = torch.device("cuda:0")
    device_ids = list(range(gpu_count))
    local_batch = args.batch_size
    global_batch = local_batch * gpu_count
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    def parallel(network):
        network = network.to(primary)
        return (nn.DataParallel(network, device_ids=device_ids, output_device=0)
                if gpu_count > 1 else network)

    if args.workload == "ResNet50":
        torch.backends.cudnn.benchmark = True
        model = parallel(module.ResNet50(num_classes=1000).train())
        optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
        criterion = nn.CrossEntropyLoss()
        images = torch.randn(global_batch, 3, args.image_height, args.image_width,
                             device=primary)
        labels = torch.randint(0, 1000, (global_batch,), device=primary)

        def step():
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(images), labels)
            loss.backward()
            optimizer.step()

        metadata = {"input_shape": [global_batch, 3, args.image_height, args.image_width],
                    "batch_size_per_gpu": local_batch}

    elif args.workload == "CycleGAN":
        torch.backends.cudnn.benchmark = True
        model = module.CycleGAN(args.base_channels, args.residual_blocks).to(primary)
        if gpu_count > 1:
            for name in ("G_AB", "G_BA", "D_A", "D_B"):
                setattr(model, name, nn.DataParallel(
                    getattr(model, name), device_ids=device_ids, output_device=0))
        optim_g = optim.Adam(list(model.G_AB.parameters()) + list(model.G_BA.parameters()),
                             lr=args.learning_rate, betas=(0.5, 0.999))
        optim_d = optim.Adam(list(model.D_A.parameters()) + list(model.D_B.parameters()),
                             lr=args.learning_rate, betas=(0.5, 0.999))
        shape = (global_batch, 3, args.image_size, args.image_size)
        real_a = torch.rand(shape, device=primary).mul_(2).sub_(1)
        real_b = torch.rand(shape, device=primary).mul_(2).sub_(1)

        def step():
            module.train_iteration(model, optim_g, optim_d, real_a, real_b)

        metadata = {"input_shape": list(shape), "batch_size_per_gpu": local_batch,
                    "base_channels": args.base_channels,
                    "residual_blocks": args.residual_blocks}

    elif args.workload == "Transformers":
        model = parallel(module.TransformerClassifier(
            args.vocab_size, args.seq_len, args.d_model, args.nhead, args.layers,
            args.num_classes, args.dropout).train())
        optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate)
        criterion = nn.CrossEntropyLoss()
        tokens = torch.randint(args.vocab_size, (global_batch, args.seq_len),
                               device=primary)
        labels = torch.randint(args.num_classes, (global_batch,), device=primary)

        def step():
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(tokens), labels)
            loss.backward()
            optimizer.step()

        metadata = {"input_shape": [global_batch, args.seq_len],
                    "batch_size_per_gpu": local_batch, "d_model": args.d_model,
                    "layers": args.layers}

    else:
        from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

        if not args.model_dir.is_dir():
            raise FileNotFoundError("Qwen model directory does not exist: {}".format(args.model_dir))
        dtype = {"fp16": torch.float16, "bf16": torch.bfloat16,
                 "fp32": torch.float32}[args.dtype]
        tokenizer = AutoTokenizer.from_pretrained(args.model_dir, local_files_only=True)
        model_config = AutoConfig.from_pretrained(args.model_dir, local_files_only=True)
        if gpu_count == 1:
            device_map = None
            max_memory = None
        else:
            hidden = model_config.hidden_size
            intermediate = model_config.intermediate_size
            layers = model_config.num_hidden_layers
            vocab = model_config.vocab_size
            heads = model_config.num_attention_heads
            kv_heads = getattr(model_config, "num_key_value_heads", heads)
            head_dim = getattr(model_config, "head_dim", hidden // heads)
            parameter_estimate = (2 * vocab * hidden + layers * (
                    2 * hidden * hidden + 2 * hidden * head_dim * kv_heads +
                    3 * hidden * intermediate + 4 * hidden))
            bytes_per_parameter = {"fp16": 2, "bf16": 2, "fp32": 4}[args.dtype]
            weight_gib = parameter_estimate * bytes_per_parameter / (1024 ** 3) * 1.15
            per_gpu_gib = max(1.0, weight_gib / gpu_count * 1.08)
            max_memory = {index: "{:.2f}GiB".format(per_gpu_gib)
                          for index in device_ids}
            device_map = "auto"
        load_options = {"torch_dtype": dtype, "local_files_only": True,
                        "attn_implementation": "eager"}
        if device_map is not None:
            load_options["device_map"] = device_map
            load_options["max_memory"] = max_memory
        model = AutoModelForCausalLM.from_pretrained(args.model_dir, **load_options)
        if gpu_count == 1:
            model = model.to(torch.device("cuda:0"))
        model = model.eval()

        prompt = getattr(args, "prompt", "Explain how GPUs accelerate machine learning.")
        token_ids = tokenizer(prompt, add_special_tokens=True, truncation=True,
                              max_length=args.input_tokens)["input_ids"]
        if not token_ids:
            token_ids = [tokenizer.eos_token_id or tokenizer.bos_token_id or 0]
        token_ids = (token_ids * math.ceil(args.input_tokens / len(token_ids)))[:args.input_tokens]
        input_device = model.get_input_embeddings().weight.device
        input_ids = torch.tensor(token_ids, dtype=torch.long, device=input_device)
        input_ids = input_ids.unsqueeze(0).expand(global_batch, -1).contiguous()
        attention_mask = torch.ones_like(input_ids)

        def step():
            with torch.inference_mode():
                model.generate(
                    input_ids=input_ids, attention_mask=attention_mask,
                    do_sample=False, use_cache=True,
                    min_new_tokens=args.output_tokens,
                    max_new_tokens=args.output_tokens,
                    pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id)

        metadata = {"input_tokens": args.input_tokens,
                    "output_tokens": args.output_tokens,
                    "batch_size_per_gpu": local_batch,
                    "global_batch_size": global_batch,
                    "dtype": args.dtype,
                    "device_map": getattr(model, "hf_device_map", {})}

    return step, metadata


def record_workload(args):
    gpu_indices = [int(value.strip()) for value in args.gpu_indices.split(",") if value.strip()]
    if not gpu_indices or any(index < 0 for index in gpu_indices) or len(set(gpu_indices)) != len(gpu_indices):
        raise ValueError("--gpu-indices must be a comma-separated list of distinct nonnegative indices")
    if args.iterations < 1 or args.warmup < 0:
        raise ValueError("iterations must be positive and warmup must be nonnegative")
    if args.batch_size is None:
        args.batch_size = {"ResNet50": 256, "CycleGAN": 2,
                           "Transformers": 1024, "Qwen": 2}[args.workload]
    valid_batches = {"ResNet50": (128, 256, 512), "CycleGAN": (1, 2, 4),
                     "Transformers": (512, 1024, 2048), "Qwen": (1, 2, 4)}
    if args.batch_size not in valid_batches[args.workload]:
        raise ValueError("Unsupported batch size for {}: {} (choose {})".format(
            args.workload, args.batch_size, ", ".join(map(str, valid_batches[args.workload]))))

    # CUDA sees the selected physical devices remapped to logical indices 0..N-1.
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, gpu_indices))
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError("Live ECon tracing requires CUDA-enabled PyTorch")
    if torch.cuda.device_count() != len(gpu_indices):
        raise RuntimeError("The requested GPU selection is not fully visible to PyTorch")

    step, metadata = _build_workload_step(args, torch, len(gpu_indices))
    for _ in range(args.warmup):
        step()
    _synchronize_all(torch, len(gpu_indices))

    windows = []
    with NVMLRecorder(gpu_indices, args.sampling_ms / 1000.0) as recorder:
        for _ in range(args.iterations):
            _synchronize_all(torch, len(gpu_indices))
            start = time.perf_counter()
            step()
            _synchronize_all(torch, len(gpu_indices))
            windows.append((start, time.perf_counter(), 1))

    result = measure(recorder.traces, windows,
                     len(gpu_indices) * args.iterations)
    elapsed = sum(end - start for start, end, _ in windows)
    total_sampled_energy = result["econ_j"]
    per_gpu = []
    for position, info in enumerate(recorder.device_info):
        trace = recorder.traces[position]
        util = [row["gpu_pct"] for row in recorder.utilization_traces[position]
                if "gpu_pct" in row]
        energy = sum(window["device_integrals"][position]["energy_j"]
                     for window in result["windows"])
        per_gpu.append({
            **info, "energy_j": energy, "mean_power_w": energy / elapsed,
            "sample_count": len(trace),
            "max_sampling_gap_s": max((b[0] - a[0] for a, b in zip(trace, trace[1:])),
                                      default=None),
            "mean_gpu_utilization_pct": statistics.fmean(util) if util else None,
            "power_samples": [
                {"relative_time_s": stamp - windows[0][0], "power_w": watts}
                for stamp, watts in trace
                if windows[0][0] - args.sampling_ms / 1000 <= stamp <=
                   windows[-1][1] + args.sampling_ms / 1000],
        })

    output = args.output
    if output is None:
        output = (Path(__file__).resolve().parent / "example_measurements" /
                  "{}_batch_{}_gpu{}_{}.json".format(
                      args.workload, args.batch_size, len(gpu_indices),
                      datetime.now().strftime("%Y%m%d_%H%M%S")))
    output = output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    relative_output = os.path.relpath(output, Path(__file__).resolve().parent)
    output_result = {
        **result,
        "origin": "nvml_workload_iteration_recording",
        "workload": args.workload,
        "batch_size_per_gpu": args.batch_size,
        "global_batch_size": args.batch_size * len(gpu_indices),
        "gpu_indices": gpu_indices,
        "iterations": args.iterations,
        "warmup_iterations": args.warmup,
        "sample_interval_s": args.sampling_ms / 1000.0,
        "elapsed_s": elapsed,
        "iteration_durations_s": [end - start for start, end, _ in windows],
        "energy_per_iteration_j": [row["energy_per_distributed_step_j"]
                                   for row in result["windows"]],
        "mean_power_w": total_sampled_energy / elapsed,
        "workload_config": metadata,
        "device_records": per_gpu,
        "output": relative_output,
    }
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(json.dumps(output_result, indent=2, allow_nan=False) + "\n",
                         encoding="utf-8")
    temporary.replace(output)
    print(json.dumps({
        "workload": args.workload, "gpu_count": len(gpu_indices),
        "devices": [{"index": row["index"], "name": row["name"],
                     "energy_j": row["energy_j"]} for row in per_gpu],
        "batch_size_per_gpu": args.batch_size,
        "global_batch_size": args.batch_size * len(gpu_indices),
        "iterations": args.iterations,
        "econ_j": result["econ_j"],
        "energy_per_iteration_j": output_result["energy_per_iteration_j"],
        "mean_power_w": output_result["mean_power_w"],
        "elapsed_s": elapsed, "output": str(output),
    }, indent=2))
    return output_result


def main():
    parser = argparse.ArgumentParser(
        description="Measure per-GPU NVML energy for ResNet50, CycleGAN, Transformers, or Qwen.")
    parser.add_argument("--workload", required=True,
                        choices=("ResNet50", "CycleGAN", "Transformers", "Qwen"))
    parser.add_argument("--batch-size", type=int, default=None,
                        help="Current supported batch sizes: ResNet50 128/256/512; "
                             "CycleGAN and Qwen 1/2/4; Transformers 512/1024/2048. "
                             "This is the batch size per GPU.")
    parser.add_argument("--gpu-indices", default="0",
                        help="Comma-separated physical NVML indices, such as 0,1.")
    parser.add_argument("--iterations", type=int, default=10)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--sampling-ms", type=float, default=20)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--image-height", type=int, default=32)
    parser.add_argument("--image-width", type=int, default=32)
    parser.add_argument("--image-size", type=int, default=128)
    parser.add_argument("--base-channels", type=int, default=32)
    parser.add_argument("--residual-blocks", type=int, default=9)
    parser.add_argument("--seq-len", type=int, default=64)
    parser.add_argument("--vocab-size", type=int, default=4096)
    parser.add_argument("--num-classes", type=int, default=16)
    parser.add_argument("--d-model", type=int, default=128)
    parser.add_argument("--nhead", type=int, default=4)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--learning-rate", type=float, default=None)
    parser.add_argument("--model-dir", type=Path,
                        default=Path(__file__).resolve().parent / "Workloads" / "Qwen" / "model")
    parser.add_argument("--input-tokens", type=int, default=64)
    parser.add_argument("--output-tokens", type=int, default=32)
    parser.add_argument("--dtype", choices=("fp16", "bf16", "fp32"), default="fp16")
    parser.add_argument("--seed", type=int, default=1234)
    args = parser.parse_args()
    if args.sampling_ms <= 0:
        parser.error("--sampling-ms must be positive")
    if args.learning_rate is None:
        args.learning_rate = 2e-4 if args.workload == "CycleGAN" else 1e-3
    record_workload(args)


if __name__ == "__main__":
    main()
