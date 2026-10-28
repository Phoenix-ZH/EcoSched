"""Predict multiple workloads from source GPU profiles"""
import argparse
import json
from dataclasses import asdict
from pathlib import Path

try:
    from .specs_scaling import DEFAULT_SPECS, load_profile, load_specs, predict
    from .profile_scaling import predict_with_parameters
except ImportError:
    from specs_scaling import DEFAULT_SPECS, load_profile, load_specs, predict
    from profile_scaling import predict_with_parameters

DEFAULT_ROOT = Path(__file__).parent / "Workloads" / "profile_results" / "profiler"
DEFAULT_OUTPUT_ROOT = Path(__file__).parent / "Predict_result"
DEFAULT_PARAMETERS = DEFAULT_OUTPUT_ROOT / "parameters.json"
DEFAULT_OUTPUT = DEFAULT_OUTPUT_ROOT / "predictions.json"


def discover(root, source_gpu, source_name, workload=None, batch_size=None):
    """Accept both archived W/GPU/batch_N and collector W/batch_N/GPU layouts."""
    cases = {}
    for workdir in sorted(Path(root).iterdir()):
        if not workdir.is_dir() or (workload and workdir.name != workload):
            continue
        candidates = list((workdir / source_gpu).glob("batch_*/kernels.csv"))
        for alias in {source_gpu, source_name.replace(" ", "_")}:
            candidates += list(workdir.glob(f"batch_*/{alias}/kernels.csv"))
        for kernel_path in candidates:
            directory = kernel_path.parent
            batch_dir = directory if directory.name.startswith("batch_") else directory.parent
            batch = int(batch_dir.name.removeprefix("batch_"))
            if batch_size is not None and batch != batch_size:
                continue
            key = (workdir.name, batch)
            if key in cases and cases[key] != directory:
                raise ValueError(f"Ambiguous duplicate profiles for {key}: select a unique profile root")
            cases[key] = directory
    if not cases:
        raise ValueError("No matching source profiles")
    return sorted(cases.items())


def run_predictions(root=DEFAULT_ROOT, specs_path=DEFAULT_SPECS, source_gpu="A16",
                    targets=("P100", "V100", "A100"), workload=None, batch_size=None,
                    variant="paper", prefill_root=None, input_tokens=None, output_tokens=None,
                    parameters_path=DEFAULT_PARAMETERS):
    specs = load_specs(specs_path)
    source = specs[source_gpu]
    parameters = None
    if parameters_path:
        parameters = json.loads(Path(parameters_path).read_text(encoding="utf-8"))
        if variant != "paper" or prefill_root or input_tokens is not None or output_tokens is not None:
            raise ValueError("Parameterized mode requires paper scaling and training-iteration profiles")
        if workload and workload not in parameters["supported_workloads"]:
            raise ValueError("Requested workload is outside the parameters scope")
        if source_gpu != parameters["source_gpu"]:
            raise ValueError("Parameter source GPU mismatch")
        if any(g not in parameters["log_efficiency_coefficients"] for g in targets):
            raise ValueError("Requested target GPU has no parameterized coefficients")
    records = []
    prefill_cases = dict(discover(prefill_root, source_gpu, source.name)) if prefill_root else {}
    for (work, batch), directory in discover(root, source_gpu, source.name, workload, batch_size):
        if parameters and work not in parameters["supported_workloads"]:
            continue
        profile = load_profile(directory, work, batch, source_gpu, source)
        prefill = None
        if (work, batch) in prefill_cases:
            prefill = load_profile(prefill_cases[work, batch], work, batch, source_gpu, source)
        for target in targets:
            if parameters:
                record = predict_with_parameters(profile, source, specs[target], target, parameters)
                if any(asdict(specs[g]) != parameters["hardware"][g] for g in (source_gpu, target)):
                    record["warnings"].append("Hardware differs from parameters; coefficients may not transfer.")
            else:
                record = predict(profile, source, specs[target], variant,
                                 output_tokens=output_tokens, input_tokens=input_tokens, prefill=prefill)
            record["target_gpu"] = target
            record.pop("warnings", None)
            record.pop("anchor_kind", None)
            record.pop("parameter_status", None)
            records.append(record)
    if not records:
        raise ValueError("No profiles match the selected model scope")
    return {"schema_version": 1, "variant": variant,
            "hardware": {k: asdict(v) for k, v in specs.items()},
            "predictions": records}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile-root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--specs", type=Path, default=DEFAULT_SPECS)
    parser.add_argument("--source-gpu", default="A16")
    parser.add_argument("--targets", nargs="+", default=["P100", "V100", "A100"])
    parser.add_argument("--workload")
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--variant", choices=["paper", "architecture"], default="paper")
    parser.add_argument("--prefill-root", type=Path)
    parser.add_argument("--input-tokens", type=int)
    parser.add_argument("--output-tokens", type=int)
    args = parser.parse_args()
    output_path = DEFAULT_OUTPUT
    result = run_predictions(args.profile_root, args.specs, args.source_gpu, args.targets,
                             args.workload, args.batch_size, args.variant, args.prefill_root,
                             args.input_tokens, args.output_tokens, DEFAULT_PARAMETERS)
    text = json.dumps(result, indent=2, allow_nan=False)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(text + "\n", encoding="utf-8")
    predictions = result["predictions"]
    workloads = sorted({record["workload"] for record in predictions})
    batches = sorted({(record["workload"], record["batch_size"]) for record in predictions})
    print(f"\nPrediction complete: {len(predictions)} GPU predictions across "
          f"{len(workloads)} workloads and {len(batches)} workload/batch cases.")
    print(f"Source GPU: {args.source_gpu} | Target GPUs: {', '.join(args.targets)} | "
          "Training iterations and Qwen request estimates | Unit: ms")
    print("Saved JSON: Predict_result/predictions.json\n")
    print("Workload | Batch | Target GPU | Predicted time (ms)")
    print("---------|------:|------------|--------------------:")
    for record in predictions:
        print(f"{record['workload']} | {record['batch_size']} | {record['target_gpu']} | "
              f"{record['predicted_ms']:>19.4f}")

    print("\nWorkload summary (predicted iteration time range):")
    for workload in workloads:
        values = [record["predicted_ms"] for record in predictions
                  if record["workload"] == workload]
        count = len(values)
        print(f"  {workload}: {count} predictions, {min(values):.4f} to {max(values):.4f} ms")


if __name__ == "__main__":
    main()
