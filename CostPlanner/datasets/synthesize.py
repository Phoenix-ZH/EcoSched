#!/usr/bin/env python3
"""Build a chronological 24-hour EcoSched workload dataset.

Alibaba task, instance, job, group and machine records provide task identity,
execution order, GPU reservations, hardware context and workload-family signals.
PerfProfiler benchmark CSVs provide iteration-time inputs for ResNet50,
CycleGAN, Transformers and Qwen across the supported GPU types. The synthesis
configuration sets the 24-hour arrival volume, workload-size profile, epoch
corpora, deadline budgets, regions, power values and transfer assumptions.
Export rows combine source-trace provenance with EcoSched iterations, epochs,
deadlines, per-GPU execution candidates, energy and regional electricity costs.
"""

import argparse
import csv
import filecmp
import hashlib
import json
import math
import random
import statistics
import sys
import tempfile
from collections import Counter, defaultdict
from contextlib import ExitStack
from pathlib import Path
from typing import NamedTuple

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parent.parent
COST_PLANNER = HERE.parent
WORKLOADS = PROJECT / "PerfProfiler" / "Workloads"
GPUS = ("A16", "P100", "V100", "A100")
TASK_COLUMNS = (
    "job_name", "task_name", "inst_num", "status", "start_time", "end_time",
    "plan_cpu", "plan_mem", "plan_gpu", "gpu_type",
)
LEGACY_COLUMNS = ["arrival_time", "deadline"] + [
    field for gpu in GPUS for field in (gpu + "_num", gpu + "_exec_time")
]
REGION_COLUMNS = ["region"] + [g + "_num" for g in GPUS] + ["electricity_cost"]
# Exactly representable as float and above every schedulable GPU capacity;
# the matching runtime also exceeds the task's deadline budget.
INFEASIBLE_GPU_COUNT = 2**53 - 1
OUTPUT_NAMES = ("workloads.csv", "ML_workloads.csv", "df_regions.csv")
OVERHEAD_TASK_COUNTS = (1000, 3000, 5000, 7000, 9000)


class TraceTask(NamedTuple):
    row_number: int
    job_name: str
    task_name: str
    start: float
    end: float
    inst_num: str
    plan_cpu: str
    plan_mem: str
    plan_gpu: float
    gpu_type: str


def rng_for(seed, purpose):
    digest = hashlib.sha256((str(seed) + ":" + purpose).encode()).digest()
    return random.Random(int.from_bytes(digest, "big"))


def positive(value, label):
    number = float(value)
    if not math.isfinite(number) or number <= 0:
        raise ValueError(label + " must be finite and positive")
    return number


def integer(value, label, minimum=1):
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(label + " must be an integer >= " + str(minimum))
    return value


def load_templates(args):
    """Load benchmark profiles and configuration for the EcoSched workloads."""
    config = json.loads(args.config.read_text(encoding="utf-8"))
    if config.get("schema_version", 2) != 2:
        raise ValueError("Unsupported synthesize_config.json schema_version")
    args.hourly_task_counts = config["hourly_task_counts"]
    if len(args.hourly_task_counts) != 24:
        raise ValueError("hourly_task_counts must contain exactly 24 bins")
    for count in args.hourly_task_counts:
        integer(count, "hourly task count")
    args.work_budget = config["work_budget"]
    low = positive(args.work_budget["reference_runtime_min_s"], "reference runtime minimum")
    high = positive(args.work_budget["reference_runtime_max_s"], "reference runtime maximum")
    if low > high:
        raise ValueError("Reference runtime minimum must not exceed maximum")
    for name in ("hourly_runtime_scale", "hourly_inference_weight"):
        values = args.work_budget[name]
        if len(values) != 24:
            raise ValueError(name + " must contain exactly 24 values")
        for value in values:
            positive(value, name)
    args.trace_workload_templates = config["trace_workload_templates"]
    args.trace_gpu_profiles = config["trace_gpu_profiles"]
    args.unmatched_trace_gpu_profile = config["unmatched_trace_gpu_profile"]
    if any(g not in GPUS for g in list(args.trace_gpu_profiles.values()) + [args.unmatched_trace_gpu_profile]):
        raise ValueError("Trace GPU mappings must refer to supported profile GPUs")
    args.gpu_power_w = config["gpu_power_w"]
    for gpu in GPUS:
        positive(args.gpu_power_w[gpu], gpu + " power")
    templates = config["templates"]
    if not templates or len({t["name"] for t in templates}) != len(templates):
        raise ValueError("Templates must be nonempty and have unique names")
    if set(args.trace_workload_templates.values()) - {t["name"] for t in templates}:
        raise ValueError("Trace workload mapping refers to an unknown template")
    inputs = [args.config]
    for t in templates:
        positive(t["weight"], "template weight")
        integer(t["epoch_samples"], "epoch_samples")
        integer(t["optimizer_updates_per_iteration"], "optimizer updates", 0)
        if t["workload_type"] not in ("training", "inference"):
            raise ValueError("Unsupported workload_type")
        if t["gradient_accumulation_steps"] != 1:
            raise ValueError("Current benchmark adapters require accumulation=1")
        if not t["batch_sizes"] or len(set(t["batch_sizes"])) != len(t["batch_sizes"]):
            raise ValueError("batch_sizes must be nonempty and unique")
        script = args.workloads_root / t["script"]
        if not script.is_file():
            raise ValueError("Missing workload implementation: " + str(script))
        inputs.append(script)
        t["script_path"] = str(script)
        t["profiles"] = {}
        t["profile_issues"] = {}
        for batch in t["batch_sizes"]:
            integer(batch, "batch_size")
            cases, issues = {}, []
            for gpu in GPUS:
                path = args.profiles / t["name"] / gpu / ("batch_%d" % batch) / "iterations.csv"
                with path.open(encoding="utf-8-sig", newline="") as stream:
                    rows = list(csv.DictReader(stream))
                if len(rows) < 3:
                    raise ValueError("Need >=3 recorded timings: " + str(path))
                key = "latency_seconds" if "latency_seconds" in rows[0] else "seconds"
                values = [positive(r[key], "recorded duration") for r in rows]
                if any(int(r["batch_size"]) != batch for r in rows):
                    raise ValueError("Profile batch mismatch: " + str(path))
                metadata = {}
                for field in rows[0]:
                    if field not in ("iteration", "seconds", "latency_seconds"):
                        unique = sorted({r[field] for r in rows if r[field] != ""})
                        if len(unique) == 1:
                            metadata[field] = unique[0]
                        elif unique:
                            issues.append("%s has varying %s: %s" % (gpu, field, unique))
                missing = [k for k in t["profile_config_fields"] if k not in metadata]
                if missing:
                    issues.append(gpu + " missing configuration: " + ", ".join(missing))
                for r in rows:
                    if r.get("generated_tokens") and r.get("output_tokens"):
                        if int(r["generated_tokens"]) != batch * int(r["output_tokens"]):
                            issues.append(gpu + " generated_tokens != batch_size * output_tokens")
                            break
                trimmed = sorted(values)[1:-1]
                cases[gpu] = {
                    "path": str(path),
                    "recorded_gpu": metadata.get("gpu", gpu + "_folder_label_only"),
                    "iteration_time_s": statistics.mean(trimmed),
                    "statistic": "mean_after_dropping_one_minimum_and_one_maximum",
                    "observations": len(values), "retained_observations": len(trimmed),
                    "duration_column": key, "recorded_metadata": metadata,
                }
                inputs.append(path)
            for field in t["profile_config_fields"]:
                known = {g: c["recorded_metadata"][field] for g, c in cases.items()
                         if field in c["recorded_metadata"]}
                if len(set(known.values())) > 1:
                    issues.append("Cross-GPU configuration differs: %s=%s" % (field, known))
            t["profiles"][str(batch)] = cases
            t["profile_issues"][str(batch)] = issues
            if args.require_measured and issues:
                raise ValueError("Cannot verify comparable profiles for %s/%s: %s"
                                 % (t["name"], batch, "; ".join(issues)))
        t["timing_source"] = "recorded_benchmark_csv_simulation_input"
        t["configuration_source"] = "current_script_assumptions_plus_recorded_metadata"
    launcher = args.workloads_root / "profile_workloads.py"
    if launcher.is_file():
        inputs.append(launcher)
    return templates, inputs


def read_tasks(path, requested, rng):
    """Unbiased reservoir sampling without replacement; zero retains all."""
    header = path.with_suffix(".header")
    if tuple(header.read_text(encoding="utf-8-sig").strip().split(",")) != TASK_COLUMNS:
        raise ValueError("Unexpected task header: " + str(header))
    selected, stats, earliest = [], Counter(), math.inf
    with path.open(encoding="utf-8-sig", newline="") as stream:
        for line, row in enumerate(csv.reader(stream), 1):
            stats["source_rows"] += 1
            if len(row) != len(TASK_COLUMNS):
                stats["excluded_malformed_row"] += 1
                continue
            if row[3] != "Terminated":
                stats["excluded_status_" + row[3]] += 1
                continue
            try:
                start, end = float(row[4]), float(row[5])
                if not all(map(math.isfinite, (start, end))) or start < 0 or end <= start:
                    raise ValueError("invalid timestamps")
            except ValueError:
                stats["excluded_invalid_time"] += 1
                continue
            try:
                gpu = positive(row[8], "plan_gpu")
            except ValueError:
                stats["excluded_nonpositive_or_missing_gpu"] += 1
                continue
            try:
                instances = float(row[2])
                if not math.isfinite(instances) or instances < 1 or not instances.is_integer():
                    raise ValueError("invalid instance count")
            except ValueError:
                stats["excluded_invalid_instance_count"] += 1
                continue
            stats["eligible_rows"] += 1
            earliest = min(earliest, start)
            item = TraceTask(line, row[0], row[1], start, end, row[2], row[6], row[7], gpu, row[9])
            if requested == 0 or len(selected) < requested:
                selected.append(item)
            else:
                index = rng.randrange(stats["eligible_rows"])
                if index < requested:
                    selected[index] = item
    if not selected or requested > len(selected):
        raise ValueError("Requested %d tasks; found %d eligible" % (requested, len(selected)))
    selected.sort(key=lambda task: task.row_number)
    return selected, dict(stats), earliest


def trace_rows(path, columns):
    header = path.with_suffix(".header").read_text(encoding="utf-8-sig").strip().split(",")
    if header != columns.split(","):
        raise ValueError("Unexpected trace header: " + str(path))
    with path.open(encoding="utf-8-sig", newline="") as stream:
        for line, row in enumerate(csv.reader(stream), 1):
            if len(row) != len(header):
                raise ValueError("Malformed trace row %d: %s" % (line, path))
            yield line, row


def lookup_context(args, tasks):
    """Join selected tasks with job, group, worker and machine records by key,
    streaming large trace tables as context is resolved."""
    root = args.tasks.parent
    paths = {name: root / (name + ".csv") for name in
             ("pai_group_tag_table", "pai_instance_table", "pai_machine_spec")}
    inputs = [p for path in paths.values() for p in (path, path.with_suffix(".header"))]
    wanted_jobs = {task.job_name for task in tasks}
    jobs = {}
    for line, row in trace_rows(args.jobs, "job_name,inst_id,user,status,start_time,end_time"):
        if row[0] in wanted_jobs:
            value = dict(inst_id=row[1], user=row[2], start=row[4], end=row[5], row=line)
            if row[0] in jobs and {k: v for k, v in jobs[row[0]].items() if k != "row"} != {k: v for k, v in value.items() if k != "row"}:
                raise ValueError("Ambiguous job key: " + row[0])
            jobs[row[0]] = value
    wanted_ids = {job["inst_id"] for job in jobs.values()}
    tags = {}
    for line, row in trace_rows(paths["pai_group_tag_table"], "inst_id,user,gpu_type_spec,group,workload"):
        if row[0] in wanted_ids:
            value = dict(user=row[1], gpu_type_spec=row[2], group=row[3], workload=row[4], row=line)
            if row[0] in tags and {k: v for k, v in tags[row[0]].items() if k != "row"} != {k: v for k, v in value.items() if k != "row"}:
                raise ValueError("Ambiguous group-tag key: " + row[0])
            tags[row[0]] = value
    machines = {}
    for _, row in trace_rows(paths["pai_machine_spec"], "machine,gpu_type,cap_cpu,cap_mem,cap_gpu"):
        if row[0] in machines and machines[row[0]] != row[1]:
            raise ValueError("Ambiguous machine type: " + row[0])
        machines[row[0]] = row[1]
    selected = {(t.job_name, t.task_name): t for t in tasks}
    if len(selected) != len(tasks):
        raise ValueError("Selected task keys must be unique")
    workers = defaultdict(dict)
    stats = Counter()
    print("Joining selected tasks to worker lifetimes and machine types...", flush=True)
    for line, row in trace_rows(paths["pai_instance_table"],
                               "job_name,task_name,inst_name,worker_name,inst_id,status,start_time,end_time,machine"):
        key = (row[0], row[1])
        if key not in selected:
            continue
        stats["matched_instance_rows"] += 1
        # Group snapshots by worker_name and retain that worker's latest valid
        # terminated interval to derive one consistent execution record.
        try:
            start, end = float(row[6]), float(row[7])
            if row[5] != "Terminated" or not all(map(math.isfinite, (start, end))) or start <= 0 or end <= start:
                raise ValueError("invalid completed interval")
        except ValueError:
            stats["excluded_instance_intervals"] += 1
            continue
        worker = row[3]
        if not worker:
            stats["excluded_missing_worker_name"] += 1
            continue
        value = (start, end, line, row[8])
        previous = workers[key].get(worker)
        if previous is not None:
            stats["deduplicated_worker_rows"] += 1
        if previous is None or value[:3] > previous[:3]:
            workers[key][worker] = value
    context = {}
    for key, task in selected.items():
        job = jobs.get(task.job_name, {})
        tag = tags.get(job.get("inst_id", ""), {})
        user = job.get("user") or tag.get("user", "")
        instances = int(float(task.inst_num))
        intervals = workers.get(key, {})
        # Convert each worker's plan_gpu percentage into its GPU share and
        # combine all planned instances for this task.
        per_worker_gpus = task.plan_gpu / 100
        complete = len(intervals) == instances
        gpu_seconds = Counter()
        if complete:
            for start, end, _, machine in intervals.values():
                gpu_seconds[machines.get(machine, "UNKNOWN")] += (end - start) * per_worker_gpus
            execution_time = sum(end - start for start, end, _, _ in intervals.values()) / instances
            source = "completed_worker_lifetimes_times_plan_gpu_fraction"
        else:
            # When worker coverage differs from inst_num, derive the task's
            # reservation proxy consistently from its task-level record.
            execution_time = task.end - task.start
            gpu_seconds[task.gpu_type or "UNKNOWN"] = execution_time * instances * per_worker_gpus
            source = "task_wall_time_times_planned_instances_and_gpu_fraction"
        first_start = min((v[0] for v in intervals.values()), default=None)
        wait = first_start - task.start if first_start is not None and first_start >= task.start else ""
        context[task.row_number] = dict(
            user=user, job=job, tag=tag, valid_workers=len(intervals),
            complete=complete, execution_time=execution_time, work_source=source,
            gpu_seconds=dict(gpu_seconds), total_gpus=instances * per_worker_gpus,
            first_start=first_start if first_start is not None else "", observed_wait=wait,
            # Retain the anonymized group key for consistent workload assignment.
            template_key="group:" + user + ":" + tag["group"] if tag.get("group") else "job:" + task.job_name)
        stats["tasks_with_complete_worker_coverage" if complete else "tasks_using_task_reservation_fallback"] += 1
        stats["tasks_with_job_join"] += bool(job)
        stats["tasks_with_group_tag_join"] += bool(tag)
        stats["tasks_with_workload_label"] += bool(tag.get("workload"))
    group_models = defaultdict(set)
    for source in context.values():
        mapped = args.trace_workload_templates.get(source["tag"].get("workload", "").lower())
        if mapped:
            group_models[source["template_key"]].add(mapped)
    for source in context.values():
        candidates = group_models[source["template_key"]]
        source["mapped_model"] = next(iter(candidates)) if len(candidates) == 1 else None
        source["model_source"] = ("trace_group_family_to_benchmark_proxy" if len(candidates) == 1
                                  else "synthetic_group_or_job_template_assignment")
        source["group_model_conflict"] = int(len(candidates) > 1)
    return context, dict(stats), inputs


def load_regions(args):
    if args.regions_file:
        with args.regions_file.open(encoding="utf-8-sig", newline="") as stream:
            rows = list(csv.DictReader(stream))
        if not rows or len({r["region"] for r in rows}) != len(rows):
            raise ValueError("Region labels must be nonempty and unique")
        for r in rows:
            for g in GPUS:
                n = int(r[g + "_num"])
                if not 0 <= n < INFEASIBLE_GPU_COUNT:
                    raise ValueError("Unsupported regional GPU capacity")
                r[g + "_num"] = n
            r["electricity_cost"] = positive(r["electricity_cost"], "electricity_cost")
        args.regions = len(rows)
        return rows
    return [dict(region="region %d" % i, A16_num=i*200, P100_num=i*100,
                 V100_num=i*50, A100_num=i*20, electricity_cost=float(30+i*3))
            for i in range(1, args.regions+1)]


def make_arrivals(args, tasks):
    # Order jobs by Alibaba start time, assign the configured number of jobs
    # to each hour, and map source times while preserving within-hour gaps/ties.
    ordered = sorted(tasks, key=lambda t: (t.start, t.row_number))
    if sum(args.hourly_task_counts) != len(tasks):
        raise ValueError("Hourly count mismatch")
    times = {}
    offset = 0
    for hour, count in enumerate(args.hourly_task_counts):
        block = ordered[offset:offset + count]
        low, high = block[0].start, block[-1].start
        for task in block:
            fraction = (task.start - low) / (high - low) if high > low else .5
            times[task.row_number] = 3600 * (hour + 1e-6 + fraction * (1 - 2e-6))
        offset += count
    return [times[t.row_number] for t in tasks]


def reservation_work(args, context, times):
    # For each worker, multiply its observed interval by plan_gpu/100 and map
    # the source device through the assigned model's GPU iteration profiles.
    # Sum the resulting work proxy and map T4/MISC/unknown devices via A16.
    iterations, fallback_seconds = 0.0, 0.0
    mappings = {}
    for source_gpu, seconds in context["gpu_seconds"].items():
        gpu = args.trace_gpu_profiles.get(source_gpu, args.unmatched_trace_gpu_profile)
        if source_gpu not in args.trace_gpu_profiles:
            fallback_seconds += seconds
        mappings[source_gpu] = gpu
        iterations += seconds / times[gpu]
    return iterations, fallback_seconds, mappings


def minimum_gpus(iterations, iteration_time, budget, maximum, comm_ratio):
    """Find the first deadline-feasible integer GPU count using profile time."""
    for count in range(1, min(maximum, iterations) + 1):
        runtime = ((iterations + count - 1) // count) * iteration_time * (
            1 + comm_ratio * math.log2(count))
        if runtime <= budget:
            return count, runtime
    return None, None


def assigned_profile(args, source, templates, templates_by_name, weights):
    assignment_rng = rng_for(args.seed, "model:" + source["template_key"])
    mapped_model = source["mapped_model"]
    template = (templates_by_name[mapped_model] if mapped_model
                else assignment_rng.choices(templates, weights=weights, k=1)[0])
    batch = rng_for(args.seed, "batch:" + source["template_key"] + ":" + template["name"]).choice(template["batch_sizes"])
    cases = template["profiles"][str(batch)]
    times = {gpu: cases[gpu]["iteration_time_s"] for gpu in GPUS}
    return template, batch, cases, times


def workload_size_quantiles(args, tasks, templates, context):
    # Rank reservation-derived work proxies within each benchmark family.
    # These quantiles preserve the trace's relative workload sizes and map them
    # onto the configured EcoSched execution-time range.
    groups = defaultdict(list)
    by_name = {t["name"]: t for t in templates}
    weights = [t["weight"] for t in templates]
    for task in tasks:
        source = context[task.row_number]
        template, _, _, times = assigned_profile(args, source, templates, by_name, weights)
        iterations, _, _ = reservation_work(args, source, times)
        groups[template["name"]].append((iterations * times["A16"], task.row_number))
    quantiles = {}
    for group in groups.values():
        for rank, (_, row_number) in enumerate(sorted(group)):
            quantiles[row_number] = (rank + .5) / len(group)
    return quantiles


def synthesize_rows(args, tasks, templates, context, region_rows):
    weights = [t["weight"] for t in templates]
    arrivals = make_arrivals(args, tasks)
    templates_by_name = {t["name"]: t for t in templates}
    quantiles = workload_size_quantiles(args, tasks, templates, context)
    schedule = sorted(zip(tasks, arrivals), key=lambda pair: (pair[1], pair[0].row_number))
    # Prefixes are literal first-N rows of this one chronological dataset.
    for rank, (task, arrival) in enumerate(schedule, 1):
        # Per-source, per-attribute streams keep nested experiments and deadline
        # sensitivity runs paired, even if arrival order or task count changes.
        def stream(purpose):
            return rng_for(args.seed, purpose + ":" + str(task.row_number))
        source = context[task.row_number]
        tag = source["tag"]
        label = tag.get("workload", "").lower()
        # Map recognized workload labels to their benchmark family; assign
        # other groups a seeded template and a stable batch size.
        t, batch, cases, times = assigned_profile(args, source, templates, templates_by_name, weights)
        duration = task.end - task.start
        reserved_iterations, fallback_seconds, profile_mapping = reservation_work(args, source, times)
        source_runtime_proxy = reserved_iterations * times["A16"]
        hour = int(arrival // 3600)
        quantile = quantiles[task.row_number]
        parameters = args.work_budget
        base_runtime = (parameters["reference_runtime_min_s"] + quantile *
                        (parameters["reference_runtime_max_s"] - parameters["reference_runtime_min_s"]))
        runtime_scale = parameters["hourly_runtime_scale"][hour]
        inference_weight = (parameters["hourly_inference_weight"][hour]
                            if t["workload_type"] == "inference" else 1.0)
        # Apply the configured hourly workload scale and inference weight to
        # the rank-based reference runtime to define this task's synthetic work.
        target_runtime = base_runtime * runtime_scale * inference_weight
        unrounded_iterations = target_runtime / times["A16"]
        iterations = max(1, math.floor(unrounded_iterations + 0.5))
        # Use the template's corpus size and batch size to express the generated
        # iteration budget as complete epochs and any remaining iterations.
        per_epoch = (t["epoch_samples"] + batch - 1) // batch
        full_epochs, partial = divmod(iterations, per_epoch)
        epochs = full_epochs + int(partial > 0)
        factor = ""
        # Generate a reproducible deadline budget from the selected scenario;
        # duration-factor mode can scale it to the source task duration.
        if args.deadline_mode == "duration-factor":
            factor = stream("deadline").choice(args.deadline_factors)
            budget = max(args.deadline_min_s, factor * source["execution_time"])
        else:
            budget = stream("deadline").randint(args.deadline_min_s, args.deadline_max_s)
        volume = round(math.exp(stream("volume").uniform(
            math.log(args.data_min_mb * 1e6), math.log(args.data_max_mb * 1e6))))
        user = source["user"]
        origin = rng_for(args.seed, "region:" + (user or task.job_name)).randrange(args.regions)
        overhead = args.comm_ratio if t["workload_type"] == "training" else 0.0
        issues = t["profile_issues"][str(batch)]
        # Combine Alibaba provenance with the generated deadline, iteration
        # budget, epoch structure and profile-based execution candidates.
        row = {
            "workload_id": "syn_row_%09d" % task.row_number, "selection_rank": rank,
            "source_row": task.row_number, "source_job_name": task.job_name,
            "source_task_name": task.task_name, "user_id": user,
            "source_status": "Terminated", "source_start_time": task.start,
            "source_end_time": task.end, "source_duration_s": duration,
            "source_inst_num": task.inst_num, "source_plan_cpu": task.plan_cpu,
            "source_plan_mem": task.plan_mem, "source_plan_gpu": task.plan_gpu,
            "source_gpu_type": task.gpu_type,
            "source_job_row": source["job"].get("row", ""),
            "source_inst_id": source["job"].get("inst_id", ""),
            "source_job_start_time": source["job"].get("start", ""),
            "source_job_end_time": source["job"].get("end", ""),
            "source_group_tag_row": tag.get("row", ""),
            "source_group": tag.get("group", ""), "source_workload_label": label,
            "source_gpu_type_spec": tag.get("gpu_type_spec", ""),
            "template_assignment_key": source["template_key"],
            "source_group_model_conflict": source["group_model_conflict"],
            "source_valid_worker_count": source["valid_workers"],
            "source_complete_worker_coverage": int(source["complete"]),
            "source_first_worker_start_time": source["first_start"],
            "source_observed_wait_s": source["observed_wait"],
            "source_execution_time_s": source["execution_time"],
            "source_planned_total_gpus": source["total_gpus"],
            "source_reservation_gpu_seconds": sum(source["gpu_seconds"].values()),
            "source_work_proxy_basis": source["work_source"],
            "source_gpu_seconds_by_type": json.dumps(source["gpu_seconds"], sort_keys=True),
            "source_gpu_profile_mapping": json.dumps(profile_mapping, sort_keys=True),
            "unmatched_source_gpu_seconds": fallback_seconds,
            "arrival_mode": "hourly", "arrival_time": arrival,
            "arrival_source": "trace_start_order_and_within_bin_gaps_warped_to_fixed_counts",
            "deadline": arrival + budget, "deadline_budget_s": budget,
            "deadline_factor": factor, "deadline_source": "synthetic_" + args.deadline_mode,
            "workload_name": t["name"], "workload_type": t["workload_type"],
            "model_source": source["model_source"], "execution_mode": t["execution_mode"],
            "batch_size": batch, "batch_unit": t["batch_unit"], "batch_scope": "per_gpu",
            "precision_assumption": t["precision"],
            "input_assumptions": json.dumps(t["input_assumptions"], sort_keys=True),
            "input_configuration_source": "synthetic_template_not_verified_for_all_profiles",
            "dataset_samples": t["epoch_samples"], "dataset_source": "synthetic_epoch_corpus",
            "epoch_unit": t["epoch_unit"], "epoch_tail_policy": "pad_dataset_tail_to_full_batch",
            "padded_samples_per_epoch": per_epoch * batch,
            "iterations_per_epoch": per_epoch, "epochs": epochs,
            "effective_epochs": iterations / per_epoch, "full_epochs": full_epochs,
            "partial_epoch_iterations": partial,
            "last_epoch_iterations": (iterations - 1) % per_epoch + 1,
            "total_iterations": iterations,
            "iteration_unit": "single_gpu_training_minibatch" if t["workload_type"] == "training"
                              else "single_gpu_batched_inference_request",
            "iterations_source": "round_synthetic_reference_runtime_over_A16_iteration_time",
            "source_reference_runtime_proxy_s": source_runtime_proxy,
            "workload_size_quantile": quantile,
            "reference_runtime_scale": runtime_scale,
            "inference_work_weight": inference_weight,
            "target_reference_runtime_s": target_runtime,
            "work_budget_source": "trace_rank_within_model_and_configured_hourly_runtime",
            "epoch_source": "decompose_iteration_budget_using_synthetic_corpus_size",
            "processed_sample_slots": iterations * batch,
            "gradient_accumulation_steps": 1,
            "reference_optimizer_updates": iterations * t["optimizer_updates_per_iteration"],
            "reference_runtime_s": iterations * times["A16"],
            "runtime_quantization_error_s": iterations * times["A16"] - target_runtime,
            "timing_source": t["timing_source"],
            "profile_comparability": "unverified_or_conflicting" if issues else "metadata_consistent",
            "profile_issue_count": len(issues), "parallel_comm_ratio": overhead,
            "parallel_timing_source": "synthetic_fixed_work_ceil_rounds",
            "origin_region": origin + 1, "origin_region_label": region_rows[origin]["region"],
            "region_source": "synthetic_seeded_user_hash",
            "data_volume_bytes": volume, "data_volume_source": "synthetic_loguniform_uncached_input",
            "bandwidth_gbps": args.bandwidth_gbps,
            "transfer_time_s": volume * 8 / (args.bandwidth_gbps * 1e9),
            "feasibility_scope": "execution_time_only",
            "legacy_row_index": "",
        }
        for gpu in GPUS:
            count, runtime = minimum_gpus(iterations, times[gpu], budget, args.max_gpus, overhead)
            row[gpu + "_profile_gpu"] = cases[gpu]["recorded_gpu"]
            row[gpu + "_iteration_time_s"] = times[gpu]
            row[gpu + "_single_gpu_time_s"] = iterations * times[gpu]
            row[gpu + "_num"] = count if count is not None else ""
            row[gpu + "_exec_time"] = runtime if runtime is not None else ""
            row[gpu + "_time_feasible"] = int(count is not None)
            # Derive active-GPU ECon from configured power and profile runtime,
            # then calculate an ECost candidate for every regional tariff.
            # ECon[MWh] = GPUs * power[W] * runtime[s] / 3.6e9.
            # ECost[USD] = ECon[MWh] * regional tariff[USD/MWh].
            row[gpu + "_power_w"] = args.gpu_power_w[gpu]
            energy = count * runtime * args.gpu_power_w[gpu] / 3.6e9 if count is not None else None
            row[gpu + "_ECon_MWh"] = energy if energy is not None else ""
            row[gpu + "_origin_ECost_USD"] = energy * region_rows[origin]["electricity_cost"] if energy is not None else ""
            for region_index, region in enumerate(region_rows, 1):
                row[gpu + "_region_%d_ECost_USD" % region_index] = energy * region["electricity_cost"] if energy is not None else ""
        row["any_gpu_time_feasible"] = int(any(row[g + "_time_feasible"] for g in GPUS))
        row["all_gpu_time_feasible"] = int(all(row[g + "_time_feasible"] for g in GPUS))
        yield row


def legacy_row(row):
    result = {k: row[k] for k in LEGACY_COLUMNS}
    for g in GPUS:
        if not row[g + "_time_feasible"]:
            result[g + "_num"] = INFEASIBLE_GPU_COUNT
            result[g + "_exec_time"] = row["deadline_budget_s"] + max(1, row["deadline_budget_s"] * .01)
    return result


def overhead_row(row, args, region_rows):
    rng = rng_for(args.seed, "migration:" + row["workload_id"])
    migrate = args.regions > 1 and rng.random() < args.migration_fraction
    origin = row["origin_region"] - 1
    target = rng.choice([i for i in range(args.regions) if i != origin]) if migrate else origin
    profile_seconds = row["A100_iteration_time_s"] * args.profile_iterations
    profile_energy = profile_seconds * args.gpu_power_w["A100"] / 3.6e9
    return {
        "workload_id": row["workload_id"], "workload_name": row["workload_name"],
        "workload_type": row["workload_type"], "batch_size": row["batch_size"],
        "origin_region": row["origin_region_label"],
        "overhead_target_region": region_rows[target]["region"],
        "migration_required": int(migrate),
        "migration_assignment_source": "synthetic_overhead_scenario_not_scheduler_decision",
        "data_volume_bytes": row["data_volume_bytes"],
        "transfer_service_time_s": row["transfer_time_s"] if migrate else 0.0,
        "bandwidth_gbps": args.bandwidth_gbps,
        "profile_gpu": "A100", "profile_iterations": args.profile_iterations,
        "profile_service_time_proxy_s": profile_seconds,
        "profile_region": region_rows[origin]["region"],
        "profile_ECon_proxy_MWh": profile_energy,
        "profile_ECost_proxy_USD": profile_energy * region_rows[origin]["electricity_cost"],
        "profile_time_source": "recorded_iteration_proxy_not_measured_profiling_overhead",
        "profile_comparability": row["profile_comparability"],
    }


def write_dataset(path, rows, args, region_rows, label, overhead=False):
    path.parent.mkdir(parents=True, exist_ok=True)
    infeasible = generated = 0
    with tempfile.TemporaryDirectory(prefix=".synthesize-", dir=path.parent) as temporary:
        stage = Path(temporary)
        with ExitStack() as stack:
            def writer(name, fields):
                handle = stack.enter_context((stage / name).open("w", encoding="utf-8", newline=""))
                result = csv.DictWriter(handle, fieldnames=fields)
                result.writeheader()
                return result
            rich_writer = None
            numeric_writer = writer("ML_workloads.csv", LEGACY_COLUMNS)
            overhead_writer = None
            for source in rows:
                row = dict(source)
                generated += 1
                row["legacy_row_index"] = generated
                if rich_writer is None:
                    rich_writer = writer("workloads.csv", list(row))
                rich_writer.writerow(row)
                numeric_writer.writerow(legacy_row(row))
                if overhead:
                    item = overhead_row(row, args, region_rows)
                    if overhead_writer is None:
                        overhead_writer = writer("overhead_inputs.csv", list(item))
                    overhead_writer.writerow(item)
                infeasible += not row["any_gpu_time_feasible"]
            if not generated:
                raise ValueError("Cannot export an empty dataset")
            region_writer = writer("df_regions.csv", REGION_COLUMNS)
            region_writer.writerows({k: r[k] for k in REGION_COLUMNS} for r in region_rows)
        path.mkdir(parents=True, exist_ok=True)
        for file in stage.iterdir():
            destination = path / file.name
            # Avoid replacing a byte-identical CSV that may be open in another
            # application on Windows. Still replace it when its contents differ.
            if destination.is_file() and filecmp.cmp(file, destination, shallow=False):
                file.unlink()
            else:
                file.replace(destination)
    print("%s: %d rows; infeasible on all GPUs=%d" % (label, generated, infeasible), flush=True)


def write_prefix_views(args):
    # No task copies or retimed arrivals: each scenario is an iloc[:N] view.
    cases = []
    for count in OVERHEAD_TASK_COUNTS + (6000,):
        cases.append(dict(task_count=count, first_data_row=1, last_data_row=count,
                          workload_file="workloads.csv", legacy_file="ML_workloads.csv",
                          overhead_file="overhead_inputs.csv"))
    with (args.output_dir / "prefix_views.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(cases[0]))
        writer.writeheader()
        writer.writerows(sorted(cases, key=lambda row: row["task_count"]))
    cases = [dict(task_count=count, profiling_gpus_per_region=n,
                  profiling_gpus_total=n * args.regions, profiling_gpu_model="A100",
                  first_data_row=1, last_data_row=count, overhead_file="overhead_inputs.csv")
             for count in OVERHEAD_TASK_COUNTS for n in (2, 4, 6, 8)]
    with (args.output_dir / "overhead_scenarios.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(cases[0]))
        writer.writeheader()
        writer.writerows(cases)


def parser():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--tasks", type=Path, default=COST_PLANNER / "cluster_trace/data/pai_task_table.csv")
    p.add_argument("--jobs", type=Path, default=COST_PLANNER / "cluster_trace/data/pai_job_table.csv")
    p.add_argument("--config", type=Path, default=HERE / "synthesize_config.json")
    p.add_argument("--workloads-root", type=Path, default=WORKLOADS)
    p.add_argument("--profiles", type=Path, help="Root containing model/GPU/batch_N/iterations.csv")
    p.add_argument("--regions-file", type=Path, help="Existing region table, electricity_cost in USD/MWh")
    p.add_argument("--output-dir", type=Path)
    p.add_argument("--seed", type=int, default=2026)
    p.add_argument("--deadline-mode", choices=("uniform", "duration-factor"), default="uniform")
    p.add_argument("--deadline-min-s", type=int, default=1200)
    p.add_argument("--deadline-max-s", type=int, default=2400)
    p.add_argument("--deadline-factors", type=float, nargs="+", default=[1.2, 1.5, 2.0])
    p.add_argument("--regions", type=int, default=10)
    p.add_argument("--data-min-mb", type=float, default=100)
    p.add_argument("--data-max-mb", type=float, default=10000)
    p.add_argument("--bandwidth-gbps", type=float, default=50)
    p.add_argument("--comm-ratio", type=float, default=.05, help="Synthetic training communication overhead")
    p.add_argument("--max-gpus", type=int, default=256)
    p.add_argument("--migration-fraction", type=float, default=.2,
                   help="Synthetic migration share in the overhead input sets")
    p.add_argument("--profile-iterations", type=int, default=1,
                   help="Recorded benchmark iterations used as an overhead proxy")
    p.add_argument("--require-measured", action="store_true",
                   help="Fail if comparable profile configuration metadata cannot be verified")
    return p


def main(argv=None):
    p = parser()
    args = p.parse_args(argv)
    try:
        for name in ("regions", "max_gpus", "profile_iterations", "deadline_min_s", "deadline_max_s"):
            integer(getattr(args, name), name)
        if args.deadline_min_s > args.deadline_max_s:
            raise ValueError("deadline_min_s must not exceed deadline_max_s")
        for value in args.deadline_factors:
            positive(value, "deadline factor")
        for name in ("data_min_mb", "data_max_mb", "bandwidth_gbps"):
            positive(getattr(args, name), name)
        if args.data_min_mb > args.data_max_mb:
            raise ValueError("data_min_mb must not exceed data_max_mb")
        if not math.isfinite(args.comm_ratio) or args.comm_ratio < 0:
            raise ValueError("comm_ratio must be finite and nonnegative")
        if not math.isfinite(args.migration_fraction) or not 0 <= args.migration_fraction <= 1:
            raise ValueError("migration_fraction must be in [0,1]")
        args.profiles = args.profiles or args.workloads_root / "profile_results/benchmark"
        args.output_dir = args.output_dir or HERE / "workloads_24h"
        for name in ("tasks", "jobs", "profiles", "workloads_root", "config", "output_dir", "regions_file"):
            if getattr(args, name) is not None:
                setattr(args, name, getattr(args, name).resolve())
        outputs = [args.output_dir / name for name in OUTPUT_NAMES + (
            "overhead_inputs.csv", "prefix_views.csv", "overhead_scenarios.csv")]
        templates, inputs = load_templates(args)
        region_rows = load_regions(args)
        inputs += [args.tasks, args.tasks.with_suffix(".header"), args.jobs, args.jobs.with_suffix(".header")]
        if args.regions_file:
            inputs.append(args.regions_file)
        if set(outputs) & set(inputs):
            raise ValueError("Output paths must not replace inputs")
        requested = sum(args.hourly_task_counts)
        print("Sampling trace and resolving users...", flush=True)
        tasks, _, _ = read_tasks(args.tasks, requested, rng_for(args.seed, "sampling"))
        context, _, trace_inputs = lookup_context(args, tasks)
        inputs += trace_inputs
        write_dataset(
            args.output_dir, synthesize_rows(args, tasks, templates, context, region_rows),
            args, region_rows, "chronological_24h", overhead=True)
        write_prefix_views(args)
        print("Output: " + str(args.output_dir), flush=True)
        return 0
    except (ValueError, OSError, KeyError, TypeError, OverflowError, csv.Error) as error:
        p.error(str(error))


if __name__ == "__main__":
    sys.exit(main())
