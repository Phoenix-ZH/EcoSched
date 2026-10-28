"""Shared fixed-work execution, resource reservation and accounting.

Hourly scenarios start with fresh capacity. Full execution energy/cost is
charged to the arrival hour. Waiting is deadline-bounded; reservations are
non-preemptive and do not backfill earlier gaps. Only active GPU execution
energy is included. Results contain scheduler measurements and workload totals.
"""

import csv
import math
from bisect import bisect_right
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path


HERE = Path(__file__).resolve().parent
DATA = HERE / "datasets" / "workloads_24h"
RESULTS_DIR = HERE / "schedule_results"
INPUT_FILE = DATA / "workloads.csv"
REGION_FILE = DATA / "df_regions.csv"
GPU_TYPES = ("A16", "P100", "V100", "A100")
# Match synthesize.py's default per-task limit; regional capacities also apply.
MAX_GPUS = 256
COMM_LOG = tuple([0.0] + [math.log2(n) for n in range(1, MAX_GPUS + 1)])
RESULT_FIELDS = (
    "Hour", "Input_Task_NUM", "Comp_Task_NUM", "Failed_Task_NUM",
    "ECon_MWh", "ECost", "Waited_Task_NUM", "Mean_Wait_s", "Max_Wait_s",
    "Ideal_ECost_Lower_Bound", "Deadline_ECost_Lower_Bound",
)
SCHEDULE_FIELDS = (
    "Hour", "workload_id", "legacy_row_index", "total_iterations", "epochs",
    "arrival_time", "deadline", "status", "gpu_type", "region", "gpu_count",
    "iteration_time_s", "parallel_comm_ratio", "power_w", "start_time",
    "finish_time", "wait_s", "runtime_s", "ECon_MWh",
    "electricity_cost_USD_per_MWh", "ECost_USD",
)


@dataclass(frozen=True)
class Task:
    workload_id: str
    row_index: int
    arrival: float
    deadline: float
    iterations: int
    epochs: int
    comm_ratio: float
    times: tuple
    powers: tuple


@dataclass(frozen=True)
class Region:
    name: str
    price: float
    capacities: tuple


@dataclass(frozen=True)
class Placement:
    gpu: int
    region: int
    count: int
    start: float
    runtime: float
    energy: float
    cost: float

    @property
    def finish(self):
        return self.start + self.runtime

    @property
    def rank(self):
        # Energy remains primary. Remaining ties have deterministic ordering.
        return (self.energy, self.cost, self.count, self.finish, self.gpu, self.region)


def load_inputs():
    tasks_by_hour = defaultdict(list)
    with INPUT_FILE.open(encoding="utf-8-sig", newline="") as stream:
        for row in csv.DictReader(stream):
            task = Task(
                row["workload_id"], int(row["legacy_row_index"]),
                float(row["arrival_time"]), float(row["deadline"]),
                int(row["total_iterations"]), int(row["epochs"]),
                float(row["parallel_comm_ratio"]),
                tuple(float(row[gpu + "_iteration_time_s"]) for gpu in GPU_TYPES),
                tuple(float(row[gpu + "_power_w"]) for gpu in GPU_TYPES),
            )
            if (not 0 <= task.arrival < 86400
                    or not math.isfinite(task.deadline) or task.deadline <= task.arrival
                    or task.iterations < 1 or task.epochs < 1
                    or not math.isfinite(task.comm_ratio) or task.comm_ratio < 0
                    or any(not math.isfinite(v) or v <= 0 for v in task.times + task.powers)):
                raise ValueError(f"Invalid synthesized workload: {task.workload_id}")
            tasks_by_hour[int(task.arrival // 3600)].append(task)

    regions = []
    with REGION_FILE.open(encoding="utf-8-sig", newline="") as stream:
        for row in csv.DictReader(stream):
            region = Region(row["region"], float(row["electricity_cost"]),
                            tuple(int(row[gpu + "_num"]) for gpu in GPU_TYPES))
            if (not math.isfinite(region.price) or region.price <= 0
                    or any(n < 0 for n in region.capacities)):
                raise ValueError(f"Invalid region: {region.name}")
            regions.append(region)
    if not tasks_by_hour or not regions:
        raise ValueError("The synthesized task and region files must be nonempty")
    regions.sort(key=lambda region: (region.price, region.name))
    if len({region.name for region in regions}) != len(regions):
        raise ValueError("Region names must be unique")
    return tasks_by_hour, regions


def execution(task, gpu, count):
    # K already includes the full/partial epochs; multiplying by epochs again
    # would double-count training. Keep the generator's fixed-work timing model.
    rounds = (task.iterations + count - 1) // count
    factor = 1 + task.comm_ratio * COMM_LOG[count]
    runtime = rounds * task.times[gpu] * factor
    # Multiply the integer work first so equivalent partitions have identical
    # floating-point energy, preserving the cost tie-breaker for inference.
    energy = (rounds * count * task.times[gpu] * task.powers[gpu] / 3.6e9) * factor
    return runtime, energy


def energy_bounds(task):
    """Relax capacity/queueing, but retain fixed work and the deadline."""
    ideal = tuple(task.iterations * t * p / 3.6e9
                  for t, p in zip(task.times, task.powers))
    feasible = []
    budget = task.deadline - task.arrival
    for gpu in range(len(GPU_TYPES)):
        best = math.inf
        for count in range(1, min(MAX_GPUS, task.iterations) + 1):
            # ceil(K/G)*G >= K. This monotone bound safely prunes larger counts,
            # even though the actual integer-round runtime is not monotone.
            floor = ideal[gpu] * (1 + task.comm_ratio * COMM_LOG[count])
            if floor > best * (1 + 1e-12):
                break
            runtime, energy = execution(task, gpu, count)
            if runtime <= budget:
                best = min(best, energy)
                if count == 1 or (task.comm_ratio == 0 and task.iterations % count == 0):
                    break
        feasible.append(best)
    return ideal, tuple(feasible)


def reserve(placement, ready):
    pool = ready[placement.region][placement.gpu]
    del pool[:placement.count]
    index = bisect_right(pool, placement.finish)
    pool[index:index] = [placement.finish] * placement.count


def schedule_hour(hour, tasks, regions, schedule_writer, selector):
    # Historical hourly experiments reset all capacity, even for tasks whose
    # execution extends into another hour. This is not a continuous-day replay.
    ready = [[[0.0] * count for count in region.capacities] for region in regions]
    energy_values, cost_values, waits, ideals, bounds = [], [], [], [], []
    for task in sorted(tasks, key=lambda t: (t.arrival, t.row_index)):
        ideal, feasible = energy_bounds(task)
        ideals.append(min(ideal) * regions[0].price)
        bounds.append(min(feasible) * regions[0].price)
        placement = selector(task, regions, ready, ideal, feasible)
        detail = {
            "Hour": hour, "workload_id": task.workload_id,
            "legacy_row_index": task.row_index, "total_iterations": task.iterations,
            "epochs": task.epochs, "arrival_time": task.arrival,
            "deadline": task.deadline,
            "status": "scheduled" if placement is not None else "unscheduled",
        }
        if placement is not None:
            reserve(placement, ready)
            wait = placement.start - task.arrival
            energy_values.append(placement.energy)
            cost_values.append(placement.cost)
            waits.append(wait)
            detail.update({
                "gpu_type": GPU_TYPES[placement.gpu],
                "region": regions[placement.region].name, "gpu_count": placement.count,
                "iteration_time_s": task.times[placement.gpu],
                "parallel_comm_ratio": task.comm_ratio, "power_w": task.powers[placement.gpu],
                "start_time": placement.start, "finish_time": placement.finish,
                "wait_s": wait, "runtime_s": placement.runtime,
                "ECon_MWh": placement.energy,
                "electricity_cost_USD_per_MWh": regions[placement.region].price,
                "ECost_USD": placement.cost,
            })
        schedule_writer.writerow(detail)
    return {
        "Hour": hour, "Input_Task_NUM": len(tasks), "Comp_Task_NUM": len(waits),
        "Failed_Task_NUM": len(tasks) - len(waits),
        "ECon_MWh": math.fsum(energy_values), "ECost": math.fsum(cost_values),
        "Waited_Task_NUM": sum(wait > 0 for wait in waits),
        "Mean_Wait_s": math.fsum(waits) / len(waits) if waits else 0.0,
        "Max_Wait_s": max(waits, default=0.0),
        "Ideal_ECost_Lower_Bound": math.fsum(ideals),
        "Deadline_ECost_Lower_Bound": math.fsum(bounds),
    }


def run_algorithm(algorithm, selector):
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    output_file = RESULTS_DIR / (algorithm + "_results.csv")
    schedule_file = RESULTS_DIR / (algorithm + "_schedule.csv")
    tasks_by_hour, regions = load_inputs()
    rows = []
    schedule_tmp = schedule_file.with_suffix(".csv.tmp")
    results_tmp = output_file.with_suffix(".csv.tmp")
    try:
        with schedule_tmp.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=SCHEDULE_FIELDS)
            writer.writeheader()
            for hour in range(24):
                row = schedule_hour(hour, tasks_by_hour[hour], regions, writer, selector)
                rows.append(row)
                print(f"Hour {hour:02d}: completed {row['Comp_Task_NUM']}/{row['Input_Task_NUM']}; "
                      f"ECon={row['ECon_MWh']:.6f} MWh; ECost=${row['ECost']:.6f}; "
                      f"waited={row['Waited_Task_NUM']}", flush=True)
        with results_tmp.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=RESULT_FIELDS)
            writer.writeheader()
            writer.writerows(rows)
        schedule_tmp.replace(schedule_file)
        results_tmp.replace(output_file)
    finally:
        schedule_tmp.unlink(missing_ok=True)
        results_tmp.unlink(missing_ok=True)
    print(f"Hourly results: {output_file}")
    print(f"Task schedule: {schedule_file}")
