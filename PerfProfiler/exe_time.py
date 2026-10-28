"""Generate execution-time results from the saved model parameters."""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parent
MODEL_FILE = ROOT / "Predict_result" / "exe_time_model.json"
RESULT_FILE = ROOT / "Predict_result" / "exe_time_results.json"
COUNTS = (1, 2, 4)
WORKLOAD_NAMES = {
    "cycleGAN-Pytorch": "CycleGAN",
    "ResNet50-Pytorch": "ResNet50",
    "Transformers-Pytorch": "Transformers",
    "Qwen": "Qwen",
}


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def save(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def positive_int(value):
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("Expected a positive integer")
    return parsed


def run_factors(rng, cv, repeats):
    """Stratify the run-duration distribution to cover its range evenly."""
    sigma = math.sqrt(math.log1p(cv * cv))
    normal = statistics.NormalDist()
    factors = [math.exp(sigma * normal.inv_cdf((i + 0.5) / repeats))
               for i in range(repeats)]
    average = statistics.mean(factors)
    factors = [value / average for value in factors]
    rng.shuffle(factors)
    return factors


def read_iterations(profile_root, workload, batch_size, gpu):
    """Read positive iteration durations in seconds from the benchmark record."""
    path = profile_root / WORKLOAD_NAMES[workload] / gpu / f"batch_{batch_size}" / "iterations.csv"
    with path.open(encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    if any(int(row["batch_size"]) != batch_size for row in rows):
        raise ValueError(f"Iteration-record batch size mismatch: {workload}/{gpu}")
    values = [float(row["seconds"]) for row in rows]
    if len(values) < 2 or any(not math.isfinite(value) or value <= 0 for value in values):
        raise ValueError(f"Expected at least two positive iteration times: {workload}/{gpu}")
    return values


def iteration_factor(rng, values, steps):
    """Resample adjacent iteration blocks, preserving short-range timing patterns."""
    block_size = max(1, round(math.sqrt(len(values))))
    total = 0.0
    for step in range(math.ceil(steps)):
        if step % block_size == 0:
            start = rng.randrange(len(values))
        value = values[(start + step % block_size) % len(values)]
        total += min(1.0, steps - step) * value
    return total / (steps * statistics.mean(values))


def execution_run_cv(group, gpu_count):
    """Combine baseline variability and correlated synchronization delays.

    A tree collective has log2(gpu_count) stages. Persistent network conditions
    can affect all stages of a run, so their relative deviation grows with the
    stage count. Its variance is added to the baseline run-condition variance.
    The resulting relative spread increases with GPU count.
    """
    synchronization_cv = math.log2(gpu_count) * group["communication_stage_cv"]
    return math.hypot(group["run_cv"], synchronization_cv)


def timing_coefficients(group):
    """Return normalized compute and per-stage communication times.

    Hardware scaling retains a serial fraction and scales the remaining work
    by a weighted mix of compute throughput and memory bandwidth ratios.
    Ratios express source capacity / target capacity. The configured fractions
    describe a projection from the source profile, not a target-GPU timing trace.
    """
    if "hardware_scaling" not in group:
        return group["compute_ratio"], group["communication_ratio"]
    scaling = group["hardware_scaling"]
    for field in ("serial_fraction", "compute_fraction", "communication_fraction"):
        if not math.isfinite(scaling[field]) or not 0 <= scaling[field] <= 1:
            raise ValueError(f"Invalid hardware-scaling {field}")
    for field in ("compute_time_ratio", "memory_time_ratio"):
        if not math.isfinite(scaling[field]) or scaling[field] <= 0:
            raise ValueError(f"Invalid hardware-scaling {field}")
    serial = scaling["serial_fraction"]
    weight = scaling["compute_fraction"]
    parallel_ratio = (weight * scaling["compute_time_ratio"]
                      + (1 - weight) * scaling["memory_time_ratio"])
    compute = serial + (1 - serial) * parallel_ratio
    return compute, compute * scaling["communication_fraction"]


def validate_model(model):
    if model.get("schema_version") != 1:
        raise ValueError("Unsupported execution-time model schema")
    if not model.get("groups"):
        raise ValueError("The execution-time model must contain at least one group")
    seen = set()
    for group in model["groups"]:
        key = group["workload"], group["batch_size"], group["gpu"]
        if key in seen:
            raise ValueError(f"Duplicate model configuration: {key}")
        seen.add(key)
        compute, communication = timing_coefficients(group)
        if not all(math.isfinite(value) and value >= 0 for value in (compute, communication)):
            raise ValueError(f"Invalid timing coefficients: {key}")
        for field in ("run_cv", "communication_stage_cv"):
            value = group[field]
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"Invalid {field}: {key}")
        if compute == 0:
            raise ValueError(f"Iteration times must be positive: {key}")
        if communication > compute:
            raise ValueError(f"The supported GPU counts would not provide speedup: {key}")


def generate(model, epochs=1, iterations_per_epoch=100, repeats=30, seed=42):
    """Apply the paper's fixed-work execution-time and communication equations.

    N = epochs * single-GPU iterations per epoch, a = compute/A16,
    b = communication-stage time/A16, and g = GPU count:
        T_iter(g) = source_iteration * (a + b * log2(g))
        T_exe(g) = (N / g) * T_iter(g)
    Thus normalized T_exe = (a + b * log2(g)) / g.

    Batch size is per GPU. N/g preserves the total example budget. Fractional
    last steps use proportional work, consistent with the continuous paper
    model. Startup, warmup, validation, and checkpoint time are excluded.
    """
    validate_model(model)
    if epochs < 1 or iterations_per_epoch < 1 or repeats < 2:
        raise ValueError("Use positive epoch/iteration counts and at least two repeats")
    budget = epochs * iterations_per_epoch
    rows = []
    profile_root = ROOT / model["iteration_profile_root"]
    profiles = {}
    for group in model["groups"]:
        profile_gpu = group.get("iteration_profile_gpu", group["gpu"].split("-")[0])
        for gpu in ("A16", profile_gpu):
            key = group["workload"], group["batch_size"], gpu
            if key not in profiles:
                profiles[key] = read_iterations(profile_root, *key)
    for index, group in enumerate(model["groups"]):
        profile_key = group["workload"], group["batch_size"]
        baseline_iteration = statistics.mean(profiles[(*profile_key, "A16")])
        profile_gpu = group.get("iteration_profile_gpu", group["gpu"].split("-")[0])
        iteration_values = profiles[(*profile_key, profile_gpu)]
        baseline_seconds = budget * baseline_iteration
        compute_ratio, communication_ratio = timing_coefficients(group)
        for count in COUNTS:
            # Separate stable streams keep existing configurations reproducible.
            rng = random.Random(seed + 1009 * index + 9176 * count)
            steps = budget / count
            stages = math.log2(count)
            compute = compute_ratio / count
            communication = communication_ratio * stages / count
            duration_cv = execution_run_cv(group, count)
            factors = run_factors(rng, duration_cv, repeats)
            trials = []
            for repeat, factor in enumerate(factors):
                compute_sample = compute * factor * iteration_factor(rng, iteration_values, steps)
                communication_sample = communication * factor
                total = compute_sample + communication_sample
                trials.append({
                    "repeat": repeat + 1,
                    "compute_seconds": compute_sample * baseline_seconds,
                    "communication_seconds": communication_sample * baseline_seconds,
                    "exe_seconds": total * baseline_seconds,
                    "normalized_exe_time": total,
                })
            values = [trial["normalized_exe_time"] for trial in trials]
            mean = statistics.mean(values)
            std = statistics.stdev(values)
            row = {
                **{key: group[key] for key in ("workload", "batch_size", "gpu")},
                "gpu_count": count,
                "baseline_iteration_ms": baseline_iteration * 1000,
                "profile_iteration_mean_ms": statistics.mean(iteration_values) * 1000,
                "profile_iteration_std_ms": statistics.stdev(iteration_values) * 1000,
                "profile_iteration_count": len(iteration_values),
                "parallel_steps": steps,
                "communication_stages": stages,
                "baseline_exe_seconds": baseline_seconds,
                "compute_seconds": compute * baseline_seconds,
                "communication_seconds": communication * baseline_seconds,
                "predicted_exe_seconds": (compute + communication) * baseline_seconds,
                "predicted_normalized_exe_time": compute + communication,
                "mean_exe_seconds": mean * baseline_seconds,
                "std_exe_seconds": std * baseline_seconds,
                "mean_normalized_exe_time": mean,
                "std_normalized_exe_time": std,
                "relative_std_exe_time": std / mean,
                "trials": trials,
            }
            if "hardware_scaling" in group:
                row["iteration_profile_gpu"] = profile_gpu
                row["compute_ratio"] = compute_ratio
                row["communication_ratio"] = communication_ratio
            rows.append(row)
    return {
        "schema_version": 1,
        "data_kind": "model_generated",
        "normalization": "single_A16_same_workload_batch_and_example_budget",
        "error_bar": "one_sample_standard_deviation_of_model_trials",
        "epochs": epochs,
        "iterations_per_epoch": iterations_per_epoch,
        "repeats": repeats,
        "seed": seed,
        "rows": rows,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=MODEL_FILE)
    parser.add_argument("--output", type=Path, default=RESULT_FILE)
    parser.add_argument("--epochs", type=positive_int, default=1)
    parser.add_argument("--iterations-per-epoch", type=positive_int, default=100)
    parser.add_argument("--repeats", type=positive_int, default=30)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    try:
        result = generate(read(args.model), args.epochs, args.iterations_per_epoch,
                          args.repeats, args.seed)
        save(args.output, result)
        report = {
            "status": "generated", "data_kind": result["data_kind"],
            "points": len(result["rows"]), "repeats": args.repeats,
            "single_gpu_iteration_budget": args.epochs * args.iterations_per_epoch,
            "error_bar": result["error_bar"],
            "relative_std_pct_by_gpu_count": {
                str(count): {
                    "min": min(100 * row["relative_std_exe_time"] for row in result["rows"]
                               if row["gpu_count"] == count),
                    "max": max(100 * row["relative_std_exe_time"] for row in result["rows"]
                               if row["gpu_count"] == count),
                }
                for count in COUNTS
            },
            "output": Path(os.path.relpath(args.output)).as_posix(),
        }
    except (ValueError, KeyError, OSError) as error:
        parser.exit(1, f"Execution-time error: {error}\n")
    print(json.dumps(report, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
