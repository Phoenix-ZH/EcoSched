"""Measure network-transfer sensitivity for the 6,000-workload hourly case.

The workload trace does not contain network-device power. For this experiment,
an explicit network-path power assumption converts transfer service time to
network energy. GPU electricity and network electricity are reported
separately, and ECost is their combined electricity bill.
"""

import argparse
import csv
import math
from dataclasses import dataclass, replace
from pathlib import Path

import CostPlannerFlow
import LowestCostFirst
import LowestEConFirst
import MAST
import TanGo
import baseline_selection
import scheduling_common


HERE = Path(__file__).resolve().parent
WORKLOAD_FILE = scheduling_common.INPUT_FILE
REGION_FILE = scheduling_common.REGION_FILE
WORKLOAD_HOUR = 14
WORKLOAD_COUNT = 6000
VOLUME_MULTIPLIERS = (1, 2, 3, 4, 5)
BANDWIDTH_LEVELS_GBPS = (10, 20, 30, 40)
VOLUME_EXPERIMENT_BANDWIDTH_GBPS = 50
# Assumed active power allocated to an inter-region transfer path.
NETWORK_ACTIVE_POWER_W = 300_000.0
DEFAULT_OUTPUT_DIR = scheduling_common.RESULTS_DIR / "transfer_results"
ALGORITHMS = (
    ("LowestCostFirst", LowestCostFirst, "select_lcf_placement"),
    ("LowestEConFirst", LowestEConFirst, "select_lef_placement"),
    ("TanGo", TanGo, "select_tango_placement"),
    ("MAST", MAST, "select_mast_placement"),
)

FLOW_EXTRA_FIELDS = (
    "Batch_NUM", "Flow_Solve_NUM", "Replanned_Task_NUM",
    "Migrated_Task_NUM", "Mean_Migration_s", "Mean_Decision_Wait_s",
    "GPU_Capacity_ECost_Lower_Bound",
)
GPU_ONLY_LOWER_BOUNDS = {
    "Ideal_ECost_Lower_Bound",
    "Deadline_ECost_Lower_Bound",
    "Capacity_ECost_Lower_Bound",
}
HOURLY_RESULT_FIELDS = tuple(
    "GPU_" + field if field in GPU_ONLY_LOWER_BOUNDS else field
    for field in scheduling_common.RESULT_FIELDS
)
HOURLY_FIELDS = (
    "Workload_Hour", "Workload_Count", "Experiment", "Scenario_Value",
    "Scenario_Unit", "Data_Volume_Multiplier", "Bandwidth_Gbps",
    "Network_Active_Power_W", "Algorithm", *HOURLY_RESULT_FIELDS,
    *FLOW_EXTRA_FIELDS, "GPU_ECon_MWh", "GPU_ECost_USD",
    "Network_ECon_MWh", "Network_ECost_USD", "Total_Cost_USD",
)
SUMMARY_FIELDS = (
    "Workload_Hour", "Workload_Count", "Experiment", "Scenario_Value",
    "Scenario_Unit", "Data_Volume_Multiplier", "Bandwidth_Gbps",
    "Network_Active_Power_W", "Algorithm",
    "Input_Task_NUM", "Comp_Task_NUM", "Failed_Task_NUM",
    "Completion_Rate", "ECon_MWh", "ECost", "GPU_ECon_MWh",
    "GPU_ECost_USD", "Network_ECon_MWh", "Network_ECost_USD",
    "Total_Cost_USD", "ECost_per_completed_task",
    "GPU_ECost_per_completed_task", "Migrated_Task_NUM",
    "Transferred_GB", "Mean_Migration_s", "Mean_Wait_s",
)


@dataclass(frozen=True)
class TransferInput:
    origin: str
    volume_bytes: float
    bandwidth_gbps: float


class ScheduleMetrics:
    """Collect migration totals while hourly schedulers emit task rows."""

    def __init__(self, transfer_inputs, volume_multiplier, bandwidth_override,
                 region_prices, network_active_power_w):
        self.transfer_inputs = transfer_inputs
        self.volume_multiplier = volume_multiplier
        self.bandwidth_override = bandwidth_override
        self.region_prices = region_prices
        self.network_active_power_w = network_active_power_w
        self.completed = 0
        self.migrated = 0
        self.migration_seconds = 0.0
        self.transferred_bytes = 0.0
        self.network_energy_mwh = 0.0
        self.network_cost_usd = 0.0

    def transfer_seconds(self, workload_id, destination):
        source = self.transfer_inputs[workload_id]
        if destination == source.origin:
            return 0.0
        bandwidth = self.bandwidth_override or source.bandwidth_gbps
        return (source.volume_bytes * self.volume_multiplier * 8.0
                / (bandwidth * 1e9))

    def writerow(self, row):
        if row.get("status") != "scheduled":
            return
        self.completed += 1
        workload_id = row["workload_id"]
        destination = row["region"]
        origin = row.get("origin_region")
        if origin is None:
            origin = self.transfer_inputs[workload_id].origin
        if destination != origin:
            self.migrated += 1
            seconds = row.get("migration_time_s")
            if seconds is None:
                seconds = self.transfer_seconds(workload_id, destination)
            seconds = float(seconds)
            self.migration_seconds += seconds
            source = self.transfer_inputs[workload_id]
            energy_mwh = self.network_active_power_w * seconds / 3.6e9
            tariff = (self.region_prices[source.origin]
                      + self.region_prices[destination]) / 2.0
            self.network_energy_mwh += energy_mwh
            self.network_cost_usd += energy_mwh * tariff
            self.transferred_bytes += source.volume_bytes * self.volume_multiplier

    @property
    def mean_migration_seconds(self):
        return self.migration_seconds / self.completed if self.completed else 0.0


def read_transfer_inputs(path, task_ids, region_names):
    required = {
        "workload_id", "origin_region_label", "data_volume_bytes", "bandwidth_gbps",
    }
    values = {}
    with Path(path).open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        if not required.issubset(reader.fieldnames or ()):
            raise ValueError("Workload file is missing migration input columns")
        for row in reader:
            workload_id = row["workload_id"]
            if workload_id not in task_ids:
                continue
            origin = row["origin_region_label"]
            volume = float(row["data_volume_bytes"])
            bandwidth = float(row["bandwidth_gbps"])
            if (workload_id in values or origin not in region_names
                    or not math.isfinite(volume) or volume < 0
                    or not math.isfinite(bandwidth) or bandwidth <= 0):
                raise ValueError("Invalid migration inputs for " + workload_id)
            values[workload_id] = TransferInput(origin, volume, bandwidth)
    if set(values) != task_ids:
        raise ValueError("Workload and transfer input identities do not match")
    return values


def make_network_feasible_placements(transfer_inputs, volume_multiplier,
                                     bandwidth_override):
    """Add migration delay to candidate start times without reserving GPUs in flight."""

    def placements(task, regions, ready):
        source = transfer_inputs[task.workload_id]
        bandwidth = bandwidth_override or source.bandwidth_gbps
        for candidate in baseline_selection.feasible_placements(task, regions, ready):
            destination = regions[candidate.region].name
            if destination == source.origin:
                transfer = 0.0
            else:
                transfer = source.volume_bytes * volume_multiplier * 8.0 / (bandwidth * 1e9)
            gpu_ready = ready[candidate.region][candidate.gpu][candidate.count - 1]
            start = max(task.arrival + transfer, gpu_ready)
            if start + candidate.runtime <= task.deadline:
                yield replace(candidate, start=start)

    return placements


def scenario_migration_map(flow_module, transfer_inputs, volume_multiplier,
                           bandwidth_override):
    migration = {}
    for workload_id, source in transfer_inputs.items():
        bandwidth = bandwidth_override or source.bandwidth_gbps
        seconds = source.volume_bytes * volume_multiplier * 8.0 / (bandwidth * 1e9)
        migration[workload_id] = flow_module.Migration(source.origin, seconds)
    return migration


def run_baseline(name, module, selector_name, tasks, regions, transfer_inputs,
                 volume_multiplier, bandwidth_override, region_prices,
                 network_active_power_w, hourly_writer, scenario_name,
                 scenario_value, scenario_unit):
    metrics = ScheduleMetrics(
        transfer_inputs, volume_multiplier, bandwidth_override,
        region_prices, network_active_power_w,
    )
    network_candidates = make_network_feasible_placements(
        transfer_inputs, volume_multiplier, bandwidth_override
    )
    original_generator = module.feasible_placements
    module.feasible_placements = network_candidates
    try:
        row = scheduling_common.schedule_hour(
            WORKLOAD_HOUR, tasks, regions, metrics, getattr(module, selector_name)
        )
    finally:
        module.feasible_placements = original_generator
    row["Migrated_Task_NUM"] = metrics.migrated
    row["Mean_Migration_s"] = metrics.mean_migration_seconds
    summary = make_summary(row, metrics)
    record_hour(
        hourly_writer, scenario_name, scenario_value, scenario_unit,
        volume_multiplier, bandwidth_override, network_active_power_w,
        name, row, metrics,
    )
    return summary


def network_cost_for_task(task, destination, transfer_inputs, volume_multiplier,
                          bandwidth_gbps, region_prices, network_active_power_w):
    source = transfer_inputs[task.workload_id]
    if destination == source.origin:
        return 0.0
    seconds = source.volume_bytes * volume_multiplier * 8.0 / (bandwidth_gbps * 1e9)
    energy_mwh = network_active_power_w * seconds / 3.6e9
    tariff = (region_prices[source.origin] + region_prices[destination]) / 2.0
    return energy_mwh * tariff


def run_costplanner(tasks, regions, transfer_inputs, volume_multiplier,
                    bandwidth_override, region_prices, network_active_power_w,
                    hourly_writer, scenario_name, scenario_value, scenario_unit):
    metrics = ScheduleMetrics(
        transfer_inputs, volume_multiplier, bandwidth_override,
        region_prices, network_active_power_w,
    )
    migration = scenario_migration_map(
        CostPlannerFlow, transfer_inputs, volume_multiplier, bandwidth_override
    )
    original_demand_matrix = CostPlannerFlow.demand_matrix
    original_commit_candidate = CostPlannerFlow.commit_candidate
    task_by_id = {task.workload_id: task for task in tasks}

    def network_aware_demand_matrix(batch_tasks, candidate_regions, ready,
                                    decision_time, migration_map, timings):
        options = original_demand_matrix(
            batch_tasks, candidate_regions, ready, decision_time,
            migration_map, timings,
        )
        adjusted = {}
        for workload_id, candidates in options.items():
            task = task_by_id[workload_id]
            adjusted[workload_id] = [
                replace(
                    candidate,
                    cost=candidate.cost + network_cost_for_task(
                        task, candidate_regions[candidate.region].name,
                        transfer_inputs, volume_multiplier, bandwidth_override,
                        region_prices, network_active_power_w,
                    ),
                )
                for candidate in candidates
            ]
        return adjusted

    def commit_compute_bill(candidate, ready, decision_time):
        placement = original_commit_candidate(candidate, ready, decision_time)
        if placement is None:
            return None
        compute_cost = placement.energy * regions[placement.region].price
        return replace(placement, cost=compute_cost)

    CostPlannerFlow.demand_matrix = network_aware_demand_matrix
    CostPlannerFlow.commit_candidate = commit_compute_bill
    try:
        row = CostPlannerFlow.schedule_hour(
            WORKLOAD_HOUR, tasks, regions, migration, metrics,
            CostPlannerFlow.BATCH_SIZE, CostPlannerFlow.MAX_BATCH_DELAY_S,
        )
    finally:
        CostPlannerFlow.demand_matrix = original_demand_matrix
        CostPlannerFlow.commit_candidate = original_commit_candidate
    row["Migrated_Task_NUM"] = metrics.migrated
    row["Mean_Migration_s"] = metrics.mean_migration_seconds
    summary = make_summary(row, metrics)
    record_hour(
        hourly_writer, scenario_name, scenario_value, scenario_unit,
        volume_multiplier, bandwidth_override, network_active_power_w,
        "CostPlanner", row, metrics,
    )
    return summary


def make_summary(row, metrics):
    completed = int(row["Comp_Task_NUM"])
    input_count = int(row["Input_Task_NUM"])
    gpu_energy = float(row["ECon_MWh"])
    gpu_cost = float(row["ECost"])
    total_energy = gpu_energy + metrics.network_energy_mwh
    total_cost = gpu_cost + metrics.network_cost_usd
    return {
        "Input_Task_NUM": input_count,
        "Comp_Task_NUM": completed,
        "Failed_Task_NUM": int(row["Failed_Task_NUM"]),
        "Completion_Rate": completed / input_count if input_count else 0.0,
        "ECon_MWh": total_energy,
        "ECost": total_cost,
        "GPU_ECon_MWh": gpu_energy,
        "GPU_ECost_USD": gpu_cost,
        "Network_ECon_MWh": metrics.network_energy_mwh,
        "Network_ECost_USD": metrics.network_cost_usd,
        "Total_Cost_USD": total_cost,
        "ECost_per_completed_task": total_cost / completed if completed else 0.0,
        "GPU_ECost_per_completed_task": gpu_cost / completed if completed else 0.0,
        "Migrated_Task_NUM": metrics.migrated,
        "Transferred_GB": metrics.transferred_bytes / 1e9,
        "Mean_Migration_s": metrics.mean_migration_seconds / completed if completed else 0.0,
        "Mean_Wait_s": float(row["Mean_Wait_s"]),
    }


def experiment_metadata(experiment, value, unit, volume_multiplier,
                        bandwidth_gbps, network_active_power_w, algorithm):
    return {
        "Workload_Hour": WORKLOAD_HOUR,
        "Workload_Count": WORKLOAD_COUNT,
        "Experiment": experiment,
        "Scenario_Value": value,
        "Scenario_Unit": unit,
        "Data_Volume_Multiplier": volume_multiplier,
        "Bandwidth_Gbps": bandwidth_gbps,
        "Network_Active_Power_W": network_active_power_w,
        "Algorithm": algorithm,
    }


def record_hour(writer, experiment, value, unit, volume_multiplier,
                bandwidth_gbps, network_active_power_w, algorithm, row, metrics):
    gpu_energy = float(row["ECon_MWh"])
    gpu_cost = float(row["ECost"])
    total_energy = gpu_energy + metrics.network_energy_mwh
    total_cost = gpu_cost + metrics.network_cost_usd
    output = dict(row)
    for field in GPU_ONLY_LOWER_BOUNDS:
        if field in output:
            output["GPU_" + field] = output.pop(field)
    output.update(experiment_metadata(
        experiment, value, unit, volume_multiplier, bandwidth_gbps,
        network_active_power_w, algorithm,
    ))
    output.update({
        "GPU_ECon_MWh": gpu_energy,
        "GPU_ECost_USD": gpu_cost,
        "Network_ECon_MWh": metrics.network_energy_mwh,
        "Network_ECost_USD": metrics.network_cost_usd,
        "ECon_MWh": total_energy,
        "ECost": total_cost,
        "Total_Cost_USD": total_cost,
    })
    writer.writerow(output)


def record_summary(writer, summary, experiment, value, unit,
                   volume_multiplier, bandwidth_gbps,
                   network_active_power_w, algorithm):
    output = dict(summary)
    output.update(experiment_metadata(
        experiment, value, unit, volume_multiplier, bandwidth_gbps,
        network_active_power_w, algorithm,
    ))
    writer.writerow(output)


def scenarios():
    for multiplier in VOLUME_MULTIPLIERS:
        yield ("data_volume_multiplier", multiplier, "x", multiplier,
               VOLUME_EXPERIMENT_BANDWIDTH_GBPS)
    for bandwidth in BANDWIDTH_LEVELS_GBPS:
        yield "inter_region_bandwidth", bandwidth, "Gbps", 1, bandwidth


def run_experiments(output_root=DEFAULT_OUTPUT_DIR,
                    network_active_power_w=NETWORK_ACTIVE_POWER_W):
    tasks_by_hour, regions = scheduling_common.load_inputs()
    tasks = tasks_by_hour.get(WORKLOAD_HOUR, [])
    if len(tasks) != WORKLOAD_COUNT:
        raise ValueError(
            "Hour {} must contain exactly {} workloads; found {}".format(
                WORKLOAD_HOUR, WORKLOAD_COUNT, len(tasks)
            )
        )
    if not math.isfinite(network_active_power_w) or network_active_power_w <= 0:
        raise ValueError("Network active power must be finite and positive")
    task_ids = {task.workload_id for task in tasks}
    if len(task_ids) != WORKLOAD_COUNT:
        raise ValueError("The selected hourly workload identities are not unique")
    region_names = {region.name for region in regions}
    transfer_inputs = read_transfer_inputs(WORKLOAD_FILE, task_ids, region_names)
    region_prices = {region.name: region.price for region in regions}
    output_root = Path(output_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    hourly_path = output_root / "transfer_hourly_results.csv"
    summary_path = output_root / "transfer_summary.csv"

    with hourly_path.open("w", encoding="utf-8", newline="") as hourly_stream, \
            summary_path.open("w", encoding="utf-8", newline="") as summary_stream:
        hourly_writer = csv.DictWriter(hourly_stream, fieldnames=HOURLY_FIELDS,
                                       extrasaction="ignore")
        summary_writer = csv.DictWriter(summary_stream, fieldnames=SUMMARY_FIELDS)
        hourly_writer.writeheader()
        summary_writer.writeheader()

        for experiment, value, unit, volume_multiplier, bandwidth_override in scenarios():
            print(
                "Scenario: {}={}{}; volume={}x; bandwidth={}Gbps".format(
                    experiment, value, unit, volume_multiplier, bandwidth_override
                ),
                flush=True,
            )
            summaries = []
            for name, module, selector_name in ALGORITHMS:
                result = run_baseline(
                    name, module, selector_name, tasks, regions, transfer_inputs,
                    volume_multiplier, bandwidth_override, region_prices,
                    network_active_power_w, hourly_writer, experiment, value, unit,
                )
                summaries.append((name, result))
            result = run_costplanner(
                tasks, regions, transfer_inputs, volume_multiplier,
                bandwidth_override, region_prices, network_active_power_w,
                hourly_writer, experiment, value, unit,
            )
            summaries.append(("CostPlanner", result))
            for name, summary in summaries:
                record_summary(
                    summary_writer, summary, experiment, value, unit,
                    volume_multiplier, bandwidth_override,
                    network_active_power_w, name,
                )
            hourly_stream.flush()
            summary_stream.flush()
            for name, summary in summaries:
                print(
                    "  {}: ECost=${:.3f} (GPU=${:.3f}, network=${:.3f}); "
                    "completed={}/{}; migrated={}".format(
                        name, summary["ECost"], summary["GPU_ECost_USD"],
                        summary["Network_ECost_USD"], summary["Comp_Task_NUM"],
                        summary["Input_Task_NUM"], summary["Migrated_Task_NUM"],
                    ),
                    flush=True,
                )

    print("Hourly results:", hourly_path)
    print("Scenario summaries:", summary_path)
    return output_root


def main():
    parser = argparse.ArgumentParser(
        description="Run data-volume and inter-region-bandwidth sensitivity experiments."
    )
    parser.add_argument(
        "--output-root", type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Output directory for the hourly and scenario results.",
    )
    parser.add_argument(
        "--network-active-power-w", type=float,
        default=NETWORK_ACTIVE_POWER_W,
        help="Assumed active network-path power used for transfer energy accounting.",
    )
    args = parser.parse_args()
    if not math.isfinite(args.network_active_power_w) or args.network_active_power_w <= 0:
        parser.error("--network-active-power-w must be finite and positive")
    run_experiments(args.output_root.resolve(), args.network_active_power_w)


if __name__ == "__main__":
    main()
