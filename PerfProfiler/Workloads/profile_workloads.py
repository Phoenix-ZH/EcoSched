#!/usr/bin/env python3
"""Run each workload's timing benchmark at its configured batch sizes.

Each case writes the workload's raw per-iteration CSV and a console log under
profile_results/benchmark/, plus a consolidated summary.csv.
"""
import argparse
import csv
import re
import shlex
import statistics
import subprocess
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent
CASES = {
    "CycleGAN": {
        "directory": "CycleGAN", "script": "cyclegan_benchmark.py",
        "batch_sizes": (1, 2, 4), "env": None,
    },
    "Qwen": {
        "directory": "Qwen", "script": "qwen3_1_7b_benchmark.py",
        "batch_sizes": (1, 2, 4), "env": "qwen3",
    },
    "ResNet50": {
        "directory": "ResNet50", "script": "resnet50_benchmark.py",
        "batch_sizes": (128, 256, 512), "env": None,
    },
    "Transformers": {
        "directory": "Transformers", "script": "transformers_benchmark.py",
        "batch_sizes": (512, 1024, 2048), "env": None,
    },
}
SUMMARY_FIELDS = [
    "workload", "gpu_model", "batch_size", "status", "iterations", "trimmed_iterations",
    "trimmed_min_ms", "trimmed_max_ms", "trimmed_avg_ms", "device",
    "raw_csv", "log_file", "error",
]


def gpu_identity(device_arg):
    device = torch.device(device_arg)
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA device requested but CUDA is unavailable")
        name = torch.cuda.get_device_name(device)
    else:
        name = "CPU"
    folder = re.sub(r"[^A-Za-z0-9._-]+", "_", name).strip("._-")
    return name, folder or "Unknown_GPU"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run timing profiles for all workload/batch-size combinations."
    )
    parser.add_argument("--workloads", nargs="+", choices=CASES,
                        default=["ResNet50", "CycleGAN", "Transformers", "Qwen"],
                        help="Workloads to run (default: ResNet50 CycleGAN Transformers Qwen).")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--iterations", type=int, default=10,
                        help="Timed iterations per batch size (default: 10).")
    parser.add_argument("--warmup", type=int, default=2,
                        help="Warmup iterations per batch size (default: 2).")
    parser.add_argument("--cycle-image-size", type=int, default=128)
    parser.add_argument("--resnet-image-size", type=int, default=32)
    parser.add_argument("--transformer-seq-len", type=int, default=64)
    parser.add_argument("--qwen-input-tokens", type=int, default=64)
    parser.add_argument("--qwen-output-tokens", type=int, default=32)
    parser.add_argument("--qwen-dtype", choices=("fp16", "bf16", "fp32"),
                        default="fp16")
    parser.add_argument("--output-root", type=Path, default=ROOT / "profile_results" / "benchmark")
    parser.add_argument("--fail-fast", action="store_true",
                        help="Stop after the first failed case (default: continue).")
    args = parser.parse_args()
    if args.iterations < 7 or args.warmup < 0:
        parser.error("--iterations must be >= 7 and --warmup must be >= 0")
    return args


def build_command(name, batch_size, args, raw_csv):
    case = CASES[name]
    script = ROOT / case["directory"] / case["script"]
    if case["env"]:
        command = ["conda", "run", "-n", case["env"],
                   "python", str(script)]
    else:
        command = [sys.executable, str(script)]

    command += ["--device", args.device, "--iterations", str(args.iterations),
                "--warmup", str(args.warmup), "--output", str(raw_csv)]
    if name == "CycleGAN":
        command += ["--batch-size", str(batch_size), "--image-size",
                    str(args.cycle_image_size)]
    elif name == "Qwen":
        command += ["--batch-size", str(batch_size), "--input-tokens",
                    str(args.qwen_input_tokens), "--output-tokens",
                    str(args.qwen_output_tokens), "--dtype", args.qwen_dtype]
    elif name == "ResNet50":
        command += ["--batch-sizes", str(batch_size), "--image-size",
                    str(args.resnet_image_size), "--raw-output", str(raw_csv)]
    else:
        command += ["--batch-size", str(batch_size), "--seq-len",
                    str(args.transformer_seq_len)]
    return command


def summarize_case(name, gpu_model, batch_size, raw_csv, log_path, args, returncode, error=""):
    row = {key: "" for key in SUMMARY_FIELDS}
    row.update({"workload": name, "gpu_model": gpu_model, "batch_size": batch_size,
                "status": "failed" if returncode else "ok",
                "iterations": args.iterations, "raw_csv": str(raw_csv),
                "log_file": str(log_path), "error": error})
    if returncode or not raw_csv.is_file():
        if not returncode:
            row["status"] = "failed"
            row["error"] = "benchmark exited successfully but did not create its CSV"
        return row

    with raw_csv.open(newline="") as stream:
        records = list(csv.DictReader(stream))
    if not records:
        row["status"] = "failed"
        row["error"] = "benchmark CSV has no data rows"
        return row

    first = records[0]
    durations = []
    duration_key = next((key for key in ("seconds", "latency_seconds")
                         if key in first), None)
    if duration_key:
        durations = [float(record[duration_key]) for record in records
                     if record.get(duration_key)]
    if len(durations) < 7:
        row["status"] = "failed"
        row["error"] = "fewer than 7 raw iteration durations were recorded"
        return row
    trimmed = sorted(durations)[1:-1]
    row["iterations"] = len(durations)
    row["trimmed_iterations"] = len(trimmed)
    row["trimmed_min_ms"] = f"{min(trimmed) * 1000:.3f}"
    row["trimmed_max_ms"] = f"{max(trimmed) * 1000:.3f}"
    row["trimmed_avg_ms"] = f"{statistics.fmean(trimmed) * 1000:.3f}"
    row["device"] = first.get("gpu") or first.get("device", "")
    return row


def write_summary(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=SUMMARY_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def main():
    args = parse_args()
    run_dir = args.output_root.expanduser().resolve()
    run_dir.mkdir(parents=True, exist_ok=True)
    gpu_model, gpu_folder = gpu_identity(args.device)
    summary_path = run_dir / "summary.csv"
    if summary_path.is_file():
        with summary_path.open(newline="") as stream:
            summary_rows = list(csv.DictReader(stream))
    else:
        summary_rows = []

    for name in args.workloads:
        for batch_size in CASES[name]["batch_sizes"]:
            case_dir = run_dir / name / f"batch_{batch_size}" / gpu_folder
            raw_csv = case_dir / "iterations.csv"
            log_path = case_dir / "run.log"
            command = build_command(name, batch_size, args, raw_csv)
            print(f"[{name} batch={batch_size}] {shlex.join(command)}", flush=True)
            case_dir.mkdir(parents=True, exist_ok=True)
            try:
                with log_path.open("w", encoding="utf-8") as log:
                    result = subprocess.run(command, cwd=ROOT / CASES[name]["directory"],
                                            stdout=log, stderr=subprocess.STDOUT,
                                            text=True, check=False)
                error = "" if result.returncode == 0 else f"exit code {result.returncode}"
                row = summarize_case(name, gpu_model, batch_size, raw_csv, log_path, args,
                                     result.returncode, error)
            except OSError as exc:
                row = summarize_case(name, gpu_model, batch_size, raw_csv, log_path, args,
                                     1, str(exc))
            summary_rows = [
                old for old in summary_rows
                if not (old.get("gpu_model") == gpu_model and
                        old.get("workload") == name and
                        int(old.get("batch_size", -1)) == batch_size)
            ]
            summary_rows.append(row)
            write_summary(summary_path, summary_rows)
            print(f"  status={row['status']} trimmed min/max/avg(ms)="
                  f"{row['trimmed_min_ms']}/{row['trimmed_max_ms']}/{row['trimmed_avg_ms']}",
                  flush=True)
            if args.fail_fast and row["status"] != "ok":
                break
        if args.fail_fast and summary_rows and summary_rows[-1]["status"] != "ok":
            break

    print(f"Summary: {summary_path}")
    failures = [row for row in summary_rows if row["status"] != "ok"]
    if failures:
        print(f"Completed with {len(failures)} failed case(s); see each run.log.")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
