"""Shared candidate generation and ranking for the scheduler baselines."""

import json
import math
from pathlib import Path

from scheduling_common import (
    GPU_TYPES, MAX_GPUS, Placement, execution,
)

PARAMETERS_FILE = Path(__file__).resolve().parent / "schedule_results" / "baseline_parameters.json"


def load_baseline_parameters(algorithm):
    """Load and validate the editable parameters for one scheduler."""
    with PARAMETERS_FILE.open(encoding="utf-8") as stream:
        config = json.load(stream)
    if algorithm not in config or not isinstance(config[algorithm], dict):
        raise ValueError("Missing baseline parameters for " + algorithm)
    parameters = dict(config[algorithm])
    weights = parameters.get("BILL_WEIGHT_BY_HOUR")
    if not isinstance(weights, list) or len(weights) != 24:
        raise ValueError(algorithm + " must define 24 hourly bill weights")
    weights = tuple(float(value) for value in weights)
    if any(not math.isfinite(value) or not 0.0 <= value <= 1.0 for value in weights):
        raise ValueError(algorithm + " bill weights must be finite values in [0, 1]")
    parameters["BILL_WEIGHT_BY_HOUR"] = weights
    for name in ("GPU_SCARCITY_EXPONENT", "SUPPLY_EXPONENT"):
        if name in parameters:
            value = float(parameters[name])
            if not math.isfinite(value) or value < 0.0:
                raise ValueError(algorithm + " " + name + " must be finite and nonnegative")
            parameters[name] = value
    return parameters


def feasible_placements(task, regions, ready):
    """Enumerate deadline-feasible GPU and regional placements."""
    for gpu in range(len(GPU_TYPES)):
        for region_index, region in enumerate(regions):
            pool = ready[region_index][gpu]
            limit = min(MAX_GPUS, task.iterations, len(pool))
            if limit < 1:
                continue
            counts = set(range(1, min(limit, 8) + 1))
            count = 16
            while count < limit:
                counts.add(count)
                count *= 2
            counts.add(limit)
            for count in sorted(counts):
                start = max(task.arrival, pool[count - 1])
                if start >= task.deadline:
                    break
                runtime, energy = execution(task, gpu, count)
                if start + runtime <= task.deadline:
                    yield Placement(gpu, region_index, count, start, runtime,
                                    energy, energy * region.price)


def weighted_choice(candidates, primary_score, bill_weight):
    """Balance a normalized policy score with the resulting electricity bill."""
    candidates = list(candidates)
    if not candidates:
        return None
    bills = [candidate.cost for candidate in candidates]
    primary = [primary_score(candidate) for candidate in candidates]
    bill_min, bill_max = min(bills), max(bills)
    primary_min, primary_max = min(primary), max(primary)
    bill_span = bill_max - bill_min
    primary_span = primary_max - primary_min

    def rank(candidate):
        bill_score = ((candidate.cost - bill_min) / bill_span
                      if bill_span else 0.0)
        value = primary_score(candidate)
        policy_score = ((primary_max - value) / primary_span
                        if primary_span else 0.0)
        return (bill_weight * bill_score + (1 - bill_weight) * policy_score,
                candidate.cost, -candidate.energy, -candidate.count,
                -candidate.start, -candidate.gpu, -candidate.region)

    return max(candidates, key=rank)
