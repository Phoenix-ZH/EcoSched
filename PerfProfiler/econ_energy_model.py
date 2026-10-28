"""Saved ECon prediction and operator energy-scaling models."""
from __future__ import annotations

import math

try:
    from .econ_measurement import finite
except ImportError:
    from econ_measurement import finite


def predict_operator_energy(operators, source, target, gpu_count, serial_iterations,
                            baseline_j, communication_s=0.0, communication_power_w=0.0,
                            host_gap_s=0.0, idle_power_w=0.0):
    """Estimate normalized ECon using operator time and average power.

    Operator power follows the paper's zeta equation:
    zeta_o = alpha_o * clock_ratio**(1-beta_o) * bandwidth_ratio**beta_o.
    P'_o = P_o / P_peak * P'_peak * zeta_o.
    E_compute = sum_o T'_o * P'_o.
    E/K = E_compute + P_idle*T_gap + P_comm*T_comm*log2(n).

    Device count cancels with K/n for fixed total work and local batch size.
    Operator durations must partition busy time, with overlap accounted for
    upstream. alpha_o describes relative hardware efficiency. Peak power limits
    the modeled device power to a physically possible value.
    """
    finite(gpu_count, "gpu_count", True)
    finite(serial_iterations, "serial_iterations", True)
    if int(gpu_count) != gpu_count or int(serial_iterations) != serial_iterations or not operators:
        raise ValueError("Integer GPU/work counts and nonempty operators are required")
    for spec in (source, target):
        for name in ("clock_mhz", "bandwidth_gbps", "peak_power_w"):
            finite(spec[name], name, True)

    operator_rows = []
    for operator in operators:
        beta = finite(operator["beta"], "beta")
        if beta > 1:
            raise ValueError("beta must be in [0, 1]")
        alpha = finite(operator.get("alpha", 1.0), "alpha", True)
        source_power = finite(operator["source_power_w"], "source_power_w")
        target_seconds = finite(operator["target_seconds"], "target_seconds")

        zeta = alpha * (target["clock_mhz"] / source["clock_mhz"]) ** (1 - beta)
        zeta *= (target["bandwidth_gbps"] / source["bandwidth_gbps"]) ** beta
        predicted_power = source_power / source["peak_power_w"] * target["peak_power_w"] * zeta
        predicted_power = min(predicted_power, target["peak_power_w"])
        operator_rows.append({"name": operator["name"], "power_w": predicted_power,
                              "zeta": zeta, "energy_j": target_seconds * predicted_power})

    communication_j = (finite(communication_s, "communication_s") *
                       finite(communication_power_w, "communication_power_w") * math.log2(gpu_count))
    idle_j = finite(host_gap_s, "host_gap_s") * finite(idle_power_w, "idle_power_w")
    compute_j = sum(operator["energy_j"] for operator in operator_rows)
    energy_j = serial_iterations * (compute_j + idle_j + communication_j)
    return {"origin": "operator_power_scaling", "econ_j": energy_j,
            "econ_kwh": energy_j / 3_600_000,
            "normalized_econ": energy_j / finite(baseline_j, "baseline_j", True),
            "compute_energy_per_serial_iteration_j": compute_j,
            "host_energy_per_serial_iteration_j": idle_j,
            "communication_energy_per_serial_iteration_j": communication_j,
            "operators": operator_rows}


def predict_count_energy(model, gpu_count):
    """Evaluate a count-response model inside its supported GPU-count range."""
    finite(gpu_count, "gpu_count", True)
    if int(gpu_count) != gpu_count or not model["count_domain"][0] <= gpu_count <= model["count_domain"][1]:
        raise ValueError("GPU count is outside the model's supported range")
    log_count = math.log2(gpu_count)
    baseline, first_stage, marginal_stage = model["coefficients"]
    energy = baseline + first_stage * log_count + marginal_stage * log_count * (log_count - 1) / 2
    return finite(energy, "predicted_energy", True)
