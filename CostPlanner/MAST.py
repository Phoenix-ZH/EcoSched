"""Supply-scaled regional auctions for fixed-work GPU workloads."""

from baseline_selection import feasible_placements, load_baseline_parameters
from scheduling_common import run_algorithm


PARAMETERS = load_baseline_parameters("MAST")
SUPPLY_EXPONENT = PARAMETERS["SUPPLY_EXPONENT"]
BILL_WEIGHT_BY_HOUR = PARAMETERS["BILL_WEIGHT_BY_HOUR"]


def rank_mast_candidates(candidates, policy_score, bill_weight):
    """Combine supply-scaled GPU-time with the electricity bill."""
    candidates = list(candidates)
    if not candidates:
        return None
    bills = [candidate.cost for candidate in candidates]
    policy_values = [policy_score(candidate) for candidate in candidates]
    bill_min, bill_max = min(bills), max(bills)
    policy_min, policy_max = min(policy_values), max(policy_values)
    bill_span = bill_max - bill_min
    policy_span = policy_max - policy_min

    def rank(candidate):
        bill_score = ((candidate.cost - bill_min) / bill_span if bill_span else 0.0)
        value = policy_score(candidate)
        policy_component = ((policy_max - value) / policy_span if policy_span else 0.0)
        return (bill_weight * bill_score + (1.0 - bill_weight) * policy_component,
                candidate.cost, -candidate.energy, -candidate.count,
                -candidate.start, -candidate.gpu, -candidate.region)

    return max(candidates, key=rank)


def region_key(name):
    """Use stable region identities for equally good auction bids."""
    prefix, _, suffix = name.rpartition(" ")
    return (prefix, 0, int(suffix)) if suffix.isdigit() else (name, 1, 0)


def select_mast_placement(task, regions, ready, ideal, feasible):
    """Rank feasible placements by supply-scaled GPU-time and electricity bill."""
    supply = [sum(region.capacities[gpu] for region in regions)
              for gpu in range(4)]
    total_supply = sum(supply)

    def supply_scaled_time(choice):
        scale = (supply[choice.gpu] / total_supply) ** SUPPLY_EXPONENT
        return task.iterations * task.times[choice.gpu] / scale

    return rank_mast_candidates(
        feasible_placements(task, regions, ready),
        supply_scaled_time,
        BILL_WEIGHT_BY_HOUR[int(task.arrival // 3600)],
    )


if __name__ == "__main__":
    run_algorithm("MAST", select_mast_placement)
