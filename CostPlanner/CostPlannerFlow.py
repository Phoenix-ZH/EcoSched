"""Cost-aware GPU planning with batched minimum-cost flow.

The Cost-aware GPU Resource Planner constructs a workload-to-pool
flow graph from a GPU demand matrix. This implementation uses successive
shortest augmenting paths for the assignment problem, then checks the actual
multi-GPU reservations before committing them. Conflicting proposals are
replanned against the updated resource calendars. Exact interval calendars
retain idle gaps before future bookings, allowing safe non-preemptive backfill.

One unit of flow represents one complete workload. GPU gang sizes differ
between workloads, so a pool's task-count capacity alone cannot enforce its
physical GPU limit. The flow problem is an assignment relaxation; the calendar
checks provide physical feasibility. The complete resource-constrained planner
is an approximation, rather than a globally optimal gang scheduling solver.

Execution and accounting follow scheduling_common: total_iterations already
contains the epoch work, and each hourly experiment starts with fresh capacity.
Cross-region input transfer delays execution; active GPU execution determines
the electricity bill. Transfer does not reserve GPUs while data is in flight.
"""

import csv
import math
import tempfile
from collections import defaultdict
from dataclasses import dataclass
from itertools import product
from pathlib import Path

from costplanner_flow import MinCostFlow
from GeoCollector import CapacityCalendar
from scheduling_common import (
    GPU_TYPES, MAX_GPUS, INPUT_FILE, RESULTS_DIR, RESULT_FIELDS, SCHEDULE_FIELDS,
    Task, Placement, energy_bounds, execution, load_inputs,
)


BATCH_SIZE = 64
MAX_BATCH_DELAY_S = 10.0
# Integer edge costs preserve nanodollar resolution. Exported bills are always
# recomputed from the unrounded physical execution energy and regional tariff.
FLOW_COST_UNITS = 1_000_000_000
HOURLY_FIELDS = RESULT_FIELDS + (
    "Batch_NUM", "Flow_Solve_NUM", "Replanned_Task_NUM", "Migrated_Task_NUM",
    "Mean_Migration_s", "Mean_Decision_Wait_s", "Capacity_ECost_Lower_Bound",
)
DETAIL_FIELDS = SCHEDULE_FIELDS + (
    "batch_id", "decision_time", "origin_region", "migration_time_s",
)


@dataclass(frozen=True)
class Migration:
    origin: str
    seconds: float


@dataclass(frozen=True)
class Candidate:
    task: Task
    gpu: int
    region: int
    count: int
    start: float
    runtime: float
    energy: float
    cost: float
    migration: float


class ExecutionTable:
    """Reuse profiled gang runtimes across geographic candidate pools."""

    def __init__(self, task):
        self.task = task
        self.values = [[None] for _ in GPU_TYPES]

    def get(self, gpu, count):
        values = self.values[gpu]
        while len(values) <= count:
            values.append(execution(self.task, gpu, len(values)))
        return values[count]


def load_migration(tasks_by_hour, regions):
    """Read the existing workload input-volume and origin-region columns."""
    identities = {task.workload_id for group in tasks_by_hour.values() for task in group}
    region_names = {region.name for region in regions}
    metadata = {}
    with INPUT_FILE.open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        required = {"workload_id", "origin_region_label", "data_volume_bytes", "bandwidth_gbps"}
        if not required.issubset(reader.fieldnames or ()):
            raise ValueError("Workload input is missing migration columns")
        for row in reader:
            identity = row["workload_id"]
            if identity in metadata:
                raise ValueError("Duplicate workload identity: " + identity)
            origin = row["origin_region_label"]
            volume = float(row["data_volume_bytes"])
            bandwidth = float(row["bandwidth_gbps"])
            if (origin not in region_names or not math.isfinite(volume) or volume < 0
                    or not math.isfinite(bandwidth) or bandwidth <= 0):
                raise ValueError("Invalid migration inputs: " + identity)
            seconds = volume * 8 / (bandwidth * 1e9)
            if not math.isfinite(seconds):
                raise ValueError("Invalid transfer duration: " + identity)
            metadata[identity] = Migration(origin, seconds)
    if set(metadata) != identities:
        raise ValueError("Task and migration inputs contain different workload identities")
    return metadata


def arrival_batches(hour, tasks, batch_size, maximum_delay):
    """Flush when the queue fills or its first workload's batching timer expires."""
    ordered = sorted(tasks, key=lambda task: (task.arrival, task.row_index))
    position = 0
    while position < len(ordered):
        flush_time = min(ordered[position].arrival + maximum_delay, (hour + 1) * 3600.0)
        end = position
        while (end < len(ordered) and end - position < batch_size
               and ordered[end].arrival <= flush_time):
            end += 1
        group = ordered[position:end]
        decision_time = group[-1].arrival if len(group) == batch_size else flush_time
        yield decision_time, group
        position = end


def minimum_gang(task, gpu, region_index, regions, pool, decision_time, migration, timing):
    """Construct one demand-matrix entry with the smallest feasible GPU count."""
    transfer = 0.0 if regions[region_index].name == migration.origin else migration.seconds
    earliest = max(decision_time, task.arrival) + transfer
    if earliest >= task.deadline:
        return None
    ideal_work = task.iterations * task.times[gpu]
    available_work = pool.available_device_time(earliest, task.deadline)
    tolerance = 1e-9 * max(1.0, available_work, ideal_work)
    if available_work + tolerance < ideal_work:
        return None
    # Total GPU-seconds are a necessary feasibility condition. This bound
    # avoids searching fragmented pools that cannot contain even ideal work.
    first_count = max(1, int(ideal_work / (task.deadline - earliest)))
    # Integer rounds and queue availability can make feasibility non-monotone
    # in GPU count, so enumerate in increasing order rather than binary-search.
    for count in range(first_count, min(MAX_GPUS, task.iterations, pool.capacity) + 1):
        runtime, energy = timing.get(gpu, count)
        if count * runtime > available_work + tolerance:
            continue
        start = pool.earliest_start(earliest, runtime, count, task.deadline)
        if start is not None:
            return Candidate(task, gpu, region_index, count, start, runtime,
                             energy, energy * regions[region_index].price, transfer)
    return None


def demand_matrix(tasks, regions, ready, decision_time, migration, timings):
    options = {}
    for task in tasks:
        candidates = []
        for region_index in range(len(regions)):
            for gpu in range(len(GPU_TYPES)):
                candidate = minimum_gang(
                    task, gpu, region_index, regions, ready[region_index][gpu],
                    decision_time, migration[task.workload_id], timings[task.workload_id],
                )
                if candidate is not None:
                    candidates.append(candidate)
        options[task.workload_id] = candidates
    return options


def available_device_time(pool, decision_time, horizon):
    """Integrate unreserved GPU time, including gaps before future bookings."""
    return pool.available_device_time(decision_time, horizon)


def flow_assignment(tasks, options, regions, ready, decision_time):
    """Solve the workload/pool assignment relaxation and read final residual flow."""
    task_count = len(tasks)
    pool_count = len(regions) * len(GPU_TYPES)
    source = 0
    pool_offset = 1 + task_count
    sink = pool_offset + pool_count
    network = MinCostFlow(sink + 1)
    handles = []
    pool_candidates = defaultdict(list)
    for index, task in enumerate(tasks, 1):
        network.add_edge(source, index, 1, 0)
        for candidate in options[task.workload_id]:
            pool_index = candidate.region * len(GPU_TYPES) + candidate.gpu
            units = int(round(candidate.cost * FLOW_COST_UNITS))
            handle = network.add_edge(index, pool_offset + pool_index, 1, units)
            handles.append((handle, candidate))
            pool_candidates[pool_index].append(candidate)
    for pool_index, candidates in pool_candidates.items():
        region_index, gpu = divmod(pool_index, len(GPU_TYPES))
        horizon = max(candidate.task.deadline for candidate in candidates)
        available = available_device_time(ready[region_index][gpu], decision_time, horizon)
        minimum_work = min(candidate.count * candidate.runtime for candidate in candidates)
        # This task-count bound relaxes gang overlap and individual deadlines.
        # Actual GPU quantities and simultaneous reservations are checked later.
        capacity = min(task_count, int(available / minimum_work + 1e-9))
        network.add_edge(pool_offset + pool_index, sink, capacity, 0)
    assigned_count, _ = network.solve(source, sink, task_count)
    assignments = [candidate for handle, candidate in handles if network.flow_on(handle)]
    if len(assignments) != assigned_count:
        raise RuntimeError("Residual flow does not match the assignment count")
    if len({candidate.task.workload_id for candidate in assignments}) != assigned_count:
        raise RuntimeError("A workload was assigned more than once")
    return assignments


def commit_order(candidate, options):
    """Preserve urgent deadlines and scarce low-cost alternatives."""
    alternatives = [other.cost for other in options[candidate.task.workload_id]
                    if (other.region, other.gpu) != (candidate.region, candidate.gpu)]
    regret = min(alternatives, default=math.inf) - candidate.cost
    return (candidate.task.deadline, -regret, candidate.task.arrival, candidate.task.row_index)


def commit_candidate(candidate, ready, decision_time):
    """Reserve an entire GPU gang atomically, or leave every pool untouched."""
    pool = ready[candidate.region][candidate.gpu]
    if candidate.count > pool.capacity:
        return None
    earliest = max(decision_time, candidate.task.arrival) + candidate.migration
    start = pool.earliest_start(earliest, candidate.runtime, candidate.count,
                                candidate.task.deadline)
    if start is None:
        return None
    placement = Placement(candidate.gpu, candidate.region, candidate.count, start,
                          candidate.runtime, candidate.energy, candidate.cost)
    pool.reserve(placement.start, placement.finish, placement.count)
    return placement


def plan_batch(tasks, regions, ready, decision_time, migration):
    """Rebuild the flow graph after proposals conflict with exact GPU calendars."""
    pending = list(tasks)
    timings = {task.workload_id: ExecutionTable(task) for task in tasks}
    committed = {}
    solve_count = 0
    retried = set()
    while pending:
        options = demand_matrix(pending, regions, ready, decision_time, migration, timings)
        assignments = flow_assignment(pending, options, regions, ready, decision_time)
        solve_count += 1
        if not assignments:
            break
        progress = 0
        for candidate in sorted(assignments, key=lambda item: commit_order(item, options)):
            placement = commit_candidate(candidate, ready, decision_time)
            if placement is None:
                retried.add(candidate.task.workload_id)
                continue
            committed[candidate.task.workload_id] = (placement, candidate.migration)
            progress += 1
        # The first proposal comes from the unchanged snapshot and must fit.
        # Rebuilding only after accepting reservations guarantees termination.
        if not progress:
            raise RuntimeError("A feasible flow proposal could not be committed")
        pending = [task for task in pending if task.workload_id not in committed]
    return committed, solve_count, len(retried)


def capacity_cost_bound(tasks, regions):
    """Certify a lower bound by relaxing task indivisibility and time windows.

    A pool supplies at most capacity * (last deadline - first arrival) GPU-hours.
    Communication, integer padding, migration and individual deadlines are
    omitted. If profiled power varies by task, its per-type minimum is used.
    These relaxations can only reduce the bill required to execute all work.

    For nonnegative GPU-hour prices lambda, the allocation dual is
    sum_m min_p(cost[m,p] + lambda[p]*work[m,p]) - sum_p lambda[p]*capacity[p].
    Each enumerated cutoff defines lambda[p] = max(0, cutoff[g] - rate[p]).
    Every evaluated dual value is a valid bound; no convergence claim is needed.
    This diagnostic is calculated after scheduling and does not use future
    arrivals to make placement decisions.
    """
    if not tasks:
        return 0.0
    width = len(GPU_TYPES)
    profiles = defaultdict(int)
    for task in tasks:
        profiles[task.times] += task.iterations
    power = [min(task.powers[gpu] for task in tasks) for gpu in range(width)]
    hours = (max(task.deadline for task in tasks)
             - min(task.arrival for task in tasks)) / 3600.0
    work = [tuple(iterations * times[gpu] / 3600.0 for gpu in range(width))
            for times, iterations in profiles.items()]
    rates = [sorted({power[gpu] * region.price / 1e6 for region in regions})
             for gpu in range(width)]
    # Bound the diagnostic's enumeration even when a larger fleet is supplied.
    rates = [values if len(values) <= 10 else
             [values[round(index * (len(values) - 1) / 9)] for index in range(10)]
             for values in rates]
    penalties = [[math.fsum(
        region.capacities[gpu] * hours
        * max(0.0, cutoff - power[gpu] * region.price / 1e6)
        for region in regions) for cutoff in rates[gpu]] for gpu in range(width)]
    best = 0.0
    for indices in product(*(range(len(values)) for values in rates)):
        cutoffs = tuple(rates[gpu][indices[gpu]] for gpu in range(width))
        value = math.fsum(min(demand[gpu] * cutoffs[gpu] for gpu in range(width))
                          for demand in work)
        value -= math.fsum(penalties[gpu][indices[gpu]] for gpu in range(width))
        best = max(best, value)
    return max(0.0, best - 1e-9 * max(1.0, best))


def schedule_hour(hour, tasks, regions, migration, writer,
                  batch_size=BATCH_SIZE, maximum_delay=MAX_BATCH_DELAY_S):
    ready = [[CapacityCalendar(count) for count in region.capacities] for region in regions]
    costs, energy_values, waits, decisions, transfers = [], [], [], [], []
    ideals, bounds = [], []
    batches = solves = replanned = migrated = 0
    for decision_time, group in arrival_batches(hour, tasks, batch_size, maximum_delay):
        batches += 1
        plans, count, retried = plan_batch(group, regions, ready, decision_time, migration)
        solves += count
        replanned += retried
        for task in group:
            ideal, feasible = energy_bounds(task)
            ideals.append(min(ideal) * min(region.price for region in regions))
            bounds.append(min(feasible) * min(region.price for region in regions))
            detail = {
                "Hour": hour, "workload_id": task.workload_id,
                "legacy_row_index": task.row_index, "total_iterations": task.iterations,
                "epochs": task.epochs, "arrival_time": task.arrival, "deadline": task.deadline,
                "batch_id": batches, "decision_time": decision_time,
                "origin_region": migration[task.workload_id].origin,
                "status": "scheduled" if task.workload_id in plans else "unscheduled",
            }
            if task.workload_id in plans:
                placement, transfer = plans[task.workload_id]
                wait = placement.start - task.arrival
                costs.append(placement.cost)
                energy_values.append(placement.energy)
                waits.append(wait)
                decisions.append(decision_time - task.arrival)
                transfers.append(transfer)
                migrated += int(regions[placement.region].name != migration[task.workload_id].origin)
                detail.update({
                    "gpu_type": GPU_TYPES[placement.gpu], "region": regions[placement.region].name,
                    "gpu_count": placement.count, "iteration_time_s": task.times[placement.gpu],
                    "parallel_comm_ratio": task.comm_ratio, "power_w": task.powers[placement.gpu],
                    "start_time": placement.start, "finish_time": placement.finish,
                    "wait_s": wait, "runtime_s": placement.runtime, "ECon_MWh": placement.energy,
                    "electricity_cost_USD_per_MWh": regions[placement.region].price,
                    "ECost_USD": placement.cost, "migration_time_s": transfer,
                })
            writer.writerow(detail)
    completed = len(costs)
    return {
        "Hour": hour, "Input_Task_NUM": len(tasks), "Comp_Task_NUM": completed,
        "Failed_Task_NUM": len(tasks) - completed, "ECon_MWh": math.fsum(energy_values),
        "ECost": math.fsum(costs), "Waited_Task_NUM": sum(wait > 0 for wait in waits),
        "Mean_Wait_s": math.fsum(waits) / completed if completed else 0.0,
        "Max_Wait_s": max(waits, default=0.0), "Ideal_ECost_Lower_Bound": math.fsum(ideals),
        "Deadline_ECost_Lower_Bound": math.fsum(bounds), "Batch_NUM": batches,
        "Flow_Solve_NUM": solves, "Replanned_Task_NUM": replanned, "Migrated_Task_NUM": migrated,
        "Mean_Migration_s": math.fsum(transfers) / completed if completed else 0.0,
        "Mean_Decision_Wait_s": math.fsum(decisions) / completed if completed else 0.0,
        "Capacity_ECost_Lower_Bound": capacity_cost_bound(tasks, regions),
    }


def run(output_dir=RESULTS_DIR, batch_size=BATCH_SIZE, maximum_delay=MAX_BATCH_DELAY_S):
    if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size < 1:
        raise ValueError("batch_size must be a positive integer")
    if not math.isfinite(maximum_delay) or maximum_delay < 0:
        raise ValueError("maximum_delay must be finite and nonnegative")
    tasks, regions = load_inputs()
    migration = load_migration(tasks, regions)
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    result_file = output_dir / "CostPlannerFlow_results.csv"
    schedule_file = output_dir / "CostPlannerFlow_schedule.csv"
    rows = []
    with tempfile.TemporaryDirectory(prefix=".CostPlannerFlow-", dir=str(output_dir)) as temporary:
        temporary = Path(temporary).resolve()
        if temporary.parent != output_dir:
            raise RuntimeError("Unexpected temporary output directory")
        schedule_tmp = temporary / schedule_file.name
        results_tmp = temporary / result_file.name
        with schedule_tmp.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=DETAIL_FIELDS)
            writer.writeheader()
            for hour in range(24):
                row = schedule_hour(hour, tasks[hour], regions, migration, writer,
                                    batch_size, maximum_delay)
                rows.append(row)
                print(f"Hour {hour:02d}: completed {row['Comp_Task_NUM']}/{row['Input_Task_NUM']}; "
                      f"ECon={row['ECon_MWh']:.6f} MWh; ECost=${row['ECost']:.6f}; "
                      f"flow solves={row['Flow_Solve_NUM']}", flush=True)
        with results_tmp.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=HOURLY_FIELDS)
            writer.writeheader()
            writer.writerows(rows)
        schedule_tmp.replace(schedule_file)
        results_tmp.replace(result_file)
    print("Hourly results:", result_file)
    print("Task schedule:", schedule_file)
    return rows


if __name__ == "__main__":
    run()
