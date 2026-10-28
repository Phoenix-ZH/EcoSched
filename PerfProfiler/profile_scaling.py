"""Profile-conditioned efficiency scaling based on EcoSched Eq:specs and Eq:iterscaling.

Efficiency coefficients are estimated from training timings. Prediction consumes
saved coefficients, source-GPU profiles and hardware specifications.
No workload ID, batch-size ID or per-case correction is a regression feature.
"""
from dataclasses import replace
import math

try:
    from .specs_scaling import predict as predict_paper, scaling_factor
except ImportError:
    from specs_scaling import predict as predict_paper, scaling_factor

FEATURE_VERSION = 2
FAMILIES = (
    "conv_layout", "conv_winograd", "conv_tensor_fprop", "conv_tensor_dgrad",
    "conv_tensor_wgrad", "conv_simt", "gemm", "elementwise", "normalization",
    "softmax", "pool", "reduction", "memory", "other", "host_gap",
)


def kernel_family(kernel):
    """Group implementation families by kernel/operator semantics, not workload.

    Layout conversions inherit a convolution category in the source exporter,
    but their work is memory movement. Assign beta=1 for those kernels before
    applying Eq:specs. All other betas come from the source-profile estimator.
    """
    name = kernel.name.lower()
    op = kernel.operator.lower()
    if any(token in name for token in ("nchwtonhwc", "nhwctonchw", "nhwctonc", "nctonhw")):
        return "conv_layout", replace(kernel, beta=1.0)
    if kernel.category == "Conv":
        if "winograd" in name:
            return "conv_winograd", kernel
        tensor = any(token in name for token in ("tensorop", "xmma", "cutlass", "tf32"))
        if tensor:
            if "wgrad" in name:
                return "conv_tensor_wgrad", kernel
            if "dgrad" in name:
                return "conv_tensor_dgrad", kernel
            return "conv_tensor_fprop", kernel
        return "conv_simt", kernel
    if "norm" in op or "norm" in name:
        return "normalization", kernel
    return {"GEMM": "gemm", "Elemwise": "elementwise", "Softmax": "softmax",
            "Pool": "pool", "Reduce": "reduction", "Memory": "memory",
            "Norm": "normalization"}.get(kernel.category, "other"), kernel


def profile_features(profile, source, target):
    """X_c = overlap * sum_{o in c}(T_o * gamma_o * delta_o), in ms.

    Eq:specs supplies gamma and delta using source kernel geometry and hardware.
    Eq:iterscaling sums the scaled kernel durations. Source overlap and idle
    time extend that kernel sum to wall time. The learned correction is applied
    separately, so hardware scaling and empirical efficiency remain visible.
    """
    if profile.scope not in {"iteration", "decode"}:
        raise ValueError("This model supports iteration and decode profiles")
    anchor = profile.uninstrumented_ms * 1000 if profile.uninstrumented_ms is not None else profile.span_us
    overlap = min(profile.busy_us, anchor) / profile.kernel_sum_us
    features = dict.fromkeys(FAMILIES, 0.0)
    counts = dict.fromkeys(FAMILIES, 0)
    source_ms = dict.fromkeys(FAMILIES, 0.0)
    diagnostics = {c: {key: 0.0 for key in ("beta", "gamma", "delta", "source_capacity", "target_capacity")}
                   for c in FAMILIES}
    retiled = dict.fromkeys(FAMILIES, 0)
    for original in profile.kernels:
        family, kernel = kernel_family(original)
        factors = scaling_factor(kernel, source, target, "paper")
        features[family] += kernel.duration_us * factors["scaling_factor"] * overlap / 1000
        source_ms[family] += kernel.duration_us * overlap / 1000
        counts[family] += 1
        duration_ms = kernel.duration_us * overlap / 1000
        for key in diagnostics[family]:
            diagnostics[family][key] += (kernel.beta if key == "beta" else factors[key]) * duration_ms
        retiled[family] += factors["launch_model"] != "resource_limited_waves"
    features["host_gap"] = max(0.0, anchor - profile.busy_us) / 1000
    source_ms["host_gap"] = features["host_gap"]
    total = sum(features.values())
    if not math.isfinite(total) or total <= 0:
        raise ValueError("Profile must have a finite positive scaled duration")
    for family, values in diagnostics.items():
        diagnostics[family] = {"mean_" + key: value / source_ms[family] if source_ms[family] else None
                               for key, value in values.items()}
        if family == "host_gap":
            diagnostics[family] = {}
        diagnostics[family]["retiled_proxy_kernel_count"] = retiled[family]
    return {"features_ms": features, "source_family_ms": source_ms, "kernel_counts": counts,
            "diagnostics": diagnostics,
            "shares": {family: value / total for family, value in features.items()}}


def predict_with_parameters(profile, source, target, target_key, parameters):
    """Apply saved log-efficiency weights; references are absent from this API.

    Let q_c = X_c / sum(X). A geometric mixture models profile-conditioned
    efficiency: eta_g(q) = exp(sum_c theta_gc * q_c).
    T_hat = eta_g(q) * sum_c X_c; hence the effective kernel scaling factor is
    eta_g(q) * gamma_o * delta_o. This empirical residual is an extension to
    Eq:specs, NOT the paper's power-model alpha and NOT a hardware measurement.
    Signed log-weights allow speedups or slowdowns while eta stays positive.
    Workload and batch labels are used for scope checks/reporting only.
    """
    if parameters["feature_version"] != FEATURE_VERSION or parameters["families"] != list(FAMILIES):
        raise ValueError("Parameter feature schema mismatch; regenerate parameters")
    if profile.source_gpu != parameters["source_gpu"]:
        raise ValueError("Parameter fitting source GPU mismatch")
    if profile.workload not in parameters["supported_workloads"]:
        raise ValueError("Workload is outside the supported scope")
    if profile.scope == "decode":
        weights = parameters.get("decode_log_efficiency_coefficients", {}).get(target_key)
        if weights is None:
            raise ValueError(f"No decode efficiency coefficients are configured for {target_key}")
    else:
        weights = parameters["log_efficiency_coefficients"][target_key]
    detail = profile_features(profile, source, target)
    if set(weights) != set(FAMILIES) or any(not math.isfinite(float(v)) for v in weights.values()):
        raise ValueError("Invalid log-efficiency coefficients")
    log_eta = sum(float(weights[c]) * detail["shares"][c] for c in FAMILIES)
    # Fail explicitly on extreme extrapolation rather than silently clipping
    # output to a reference value or pretending an overflow is a prediction.
    if not -50 < log_eta < 50:
        raise ValueError("Log-efficiency extrapolation is numerically unsafe")
    eta = math.exp(log_eta)
    contributions = {}
    for family, value in detail["features_ms"].items():
        contributions[family] = {"paper_scaled_ms": value, "efficiency_factor": eta,
                                 "profile_share": detail["shares"][family],
                                 "log_efficiency_weight": weights[family],
                                 "log_efficiency_contribution": weights[family] * detail["shares"][family],
                                 "predicted_ms": eta * value,
                                 "source_ms": detail["source_family_ms"][family],
                                 "kernel_count": detail["kernel_counts"][family],
                                 "paper_scaling_factor": value / detail["source_family_ms"][family]
                                 if detail["source_family_ms"][family] else None,
                                 **detail["diagnostics"][family]}
    decode_step_ms = sum(c["predicted_ms"] for c in contributions.values())
    paper_request = None
    if profile.scope == "decode":
        paper_request = predict_paper(profile, source, target, variant="paper")
        total = eta * paper_request["predicted_ms"]
    else:
        total = decode_step_ms
    warnings = list(profile.warnings) + [
        "Accuracy for unseen batches or changed hardware is not guaranteed."]
    if paper_request is not None:
        warnings.extend(paper_request["warnings"])
    if any(c["retiled_proxy_kernel_count"] for c in contributions.values()):
        warnings.append("Some source CTAs cannot fit the target; saturated equivalent-kernel proxies were used.")
    result = {"workload": profile.workload, "batch_size": profile.batch_size,
              "source_gpu": profile.source_gpu, "target_gpu": target_key,
              "predicted_ms": total, "paper_features_sum_ms": sum(detail["features_ms"].values()),
              "efficiency_scaling_factor": eta,
              "overall_scaling_factor": total / sum(detail["source_family_ms"].values()),
              "source_iteration_anchor_ms": sum(detail["source_family_ms"].values()),
              "anchor_kind": "uninstrumented_source" if profile.uninstrumented_ms is not None else "instrumented_trace_span",
              "scope": "iteration", "families": contributions,
              "parameter_status": ("profile_conditioned_decode_coefficients" if paper_request is not None
                                   else "profile_conditioned_iteration_coefficients"),
              "warnings": warnings}
    if paper_request is not None:
        result.update({"scope": "request_from_decode_profile",
                       "paper_request_ms": paper_request["predicted_ms"],
                       "prefill_ms": eta * paper_request["prefill_ms"],
                       "decode_iteration_ms": eta * paper_request["native_iteration_ms"],
                       "input_tokens": profile.input_tokens,
                       "output_tokens": profile.output_tokens})
    return result
