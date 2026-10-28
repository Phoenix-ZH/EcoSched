"""Run the saved ECon model, trace measurement, or operator prediction.
"""
from __future__ import annotations

import argparse
from datetime import datetime
import json
import math
import os
import statistics
from pathlib import Path

try:
    from .econ_energy_model import predict_count_energy, predict_operator_energy
    from .econ_measurement import measure, record_subprocess
except ImportError:
    from econ_energy_model import predict_count_energy, predict_operator_energy
    from econ_measurement import measure, record_subprocess

ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT / "Predict_result"
MODEL_FILE = OUTPUT / "econ_model.json"
RESULT_FILE = OUTPUT / "econ_results.json"
PREDICTION_FILE = OUTPUT / "econ_predictions.json"
WORKLOADS = ("cycleGAN-Pytorch", "ResNet50-Pytorch", "Transformers-Pytorch", "Qwen")
GPUS = ("P100-SXM2-16GB", "V100-SXM2-16GB", "A100-PCIE-40GB")
COUNTS = (1, 2, 4)


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def save(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def predict_energy(model):
    """Apply the saved coefficients to every supported GPU count."""
    if model.get("schema_version") != 1:
        raise ValueError("Unsupported energy-model schema")

    rows = []
    for group in model["groups"]:
        for count in COUNTS:
            rows.append({
                "workload": group["workload"],
                "batch_size": group["batch_size"],
                "gpu": group["gpu"],
                "gpu_count": count,
                "predicted_measured_econ": predict_count_energy(group["measured_response"], count),
                "predicted_estimated_econ": predict_count_energy(group["estimated_response"], count),
            })

    expected = {
        (workload, batch_size, gpu, count)
        for workload, batch_size in zip(WORKLOADS, (2, 256, 1024, 2))
        for gpu in GPUS
        for count in COUNTS
    }
    if {configuration_key(row) for row in rows} != expected or len(rows) != len(expected):
        raise ValueError("The saved energy model has an incomplete configuration grid")
    return {"schema_version": 1, "normalization": model["normalization"], "rows": rows}


def configuration_key(row):
    return row["workload"], row["batch_size"], row["gpu"], row["gpu_count"]


def summarize(rows):
    errors = [row["estimated_vs_measured_error_pct"] for row in rows]
    return {
        "points": len(rows),
        "estimated_vs_measured_error_pct": {
            "min": min(errors),
            "max": max(errors),
            "mean": sum(errors) / len(errors),
            "median": statistics.median(errors),
        },
    }


def evaluate_energy(prediction):
    """Summarize the difference between estimated and measured ECon outputs."""
    rows = []
    for source in prediction["rows"]:
        measured = source["predicted_measured_econ"]
        estimate = source["predicted_estimated_econ"]
        if (not math.isfinite(measured) or not math.isfinite(estimate)
                or measured <= 0 or estimate <= 0):
            raise ValueError("The saved energy model produced an invalid ECon value")
        rows.append({
            **source,
            "estimated_vs_measured_error_pct": 100 * abs(estimate / measured - 1),
        })

    by_workload = {
        workload: summarize([row for row in rows if row["workload"] == workload])
        for workload in WORKLOADS
    }
    worst = max(rows, key=lambda row: row["estimated_vs_measured_error_pct"])
    return {
        **prediction,
        "rows": rows,
        "summary": summarize(rows),
        "by_workload": by_workload,
        "worst_configuration": {
            key: worst[key] for key in
            ("workload", "batch_size", "gpu", "gpu_count", "estimated_vs_measured_error_pct")
        },
    }


def prediction_summary(prediction):
    rows = prediction["rows"]
    return {
        "command": "predict",
        "points": len(rows),
        "configurations_by_workload": {
            workload: sum(row["workload"] == workload for row in rows)
            for workload in WORKLOADS
        },
        "predicted_measured_econ_range": {
            "min": min(row["predicted_measured_econ"] for row in rows),
            "max": max(row["predicted_measured_econ"] for row in rows),
        },
        "predicted_estimated_econ_range": {
            "min": min(row["predicted_estimated_econ"] for row in rows),
            "max": max(row["predicted_estimated_econ"] for row in rows),
        },
        "saved_files": [os.path.relpath(PREDICTION_FILE, Path.cwd()),
                        os.path.relpath(RESULT_FILE, Path.cwd())],
    }


def evaluation_summary(result):
    return {
        "command": "evaluate",
        **result["summary"],
        "by_workload": result["by_workload"],
        "worst_configuration": result["worst_configuration"],
        "saved_file": os.path.relpath(RESULT_FILE, Path.cwd()),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("predict", "evaluate", "measure", "record", "operator-predict"))
    parser.add_argument("--model", type=Path, default=MODEL_FILE)
    parser.add_argument("--input", type=Path, help="JSON input for measure, record, or operator-predict")
    parser.add_argument("--output", type=Path, help="Optional output file for a live recording")
    args = parser.parse_args()

    if args.command == "predict":
        prediction = predict_energy(read(args.model))
        result = evaluate_energy(prediction)
        save(PREDICTION_FILE, prediction)
        save(RESULT_FILE, result)
        output = prediction_summary(prediction)
    elif args.command == "evaluate":
        if not PREDICTION_FILE.is_file():
            parser.error("No saved predictions found; run 'python PerfProfiler/run_econ.py predict' first")
        result = evaluate_energy(read(PREDICTION_FILE))
        save(RESULT_FILE, result)
        output = evaluation_summary(result)
    else:
        if not args.input:
            parser.error("--input is required")
        payload = read(args.input)
        if args.command == "measure":
            result = measure(**payload)
            output_path = OUTPUT / "trace_measurement.json"
        elif args.command == "record":
            result = record_subprocess(**payload)
            output_path = args.output or OUTPUT / f"econ_record_{datetime.now():%Y%m%d_%H%M%S_%f}.json"
        else:
            result = predict_operator_energy(**payload)
            output_path = OUTPUT / "operator_prediction.json"
        save(output_path, result)
        if args.command == "record" and result["return_code"] != 0:
            print(json.dumps(result, indent=2))
            raise SystemExit(result["return_code"])

    if args.command == "record":
        output = {key: result.get(key) for key in
                  ("process_status", "return_code", "workload", "batch_size", "gpu_count",
                   "completed_steps", "elapsed_s", "sampled_energy_j", "energy_per_distributed_step_j",
                   "econ_j", "normalized_econ")}
        output["saved_to"] = os.path.relpath(output_path, Path.cwd())
    elif args.command not in ("predict", "evaluate"):
        output = result.get("summary", {"points": len(result.get("rows", []))})
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
