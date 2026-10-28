"""Regional tariff and electricity-bill placement for GPU workloads."""

from baseline_selection import feasible_placements, load_baseline_parameters
from scheduling_common import run_algorithm

PARAMETERS = load_baseline_parameters("LowestCostFirst")
BILL_WEIGHT_BY_HOUR = PARAMETERS["BILL_WEIGHT_BY_HOUR"]


def rank_lcf_candidates(candidates, policy_score, bill_weight):
    """Combine regional tariff preference with the resulting electricity bill."""
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


def select_lcf_placement(task, regions, ready, ideal, feasible):
    """Select a feasible placement using tariff and bill scores."""
    return rank_lcf_candidates(
        feasible_placements(task, regions, ready),
        lambda choice: regions[choice.region].price,
        BILL_WEIGHT_BY_HOUR[int(task.arrival // 3600)],
    )


if __name__ == "__main__":
    run_algorithm("LowestCostFirst", select_lcf_placement)
