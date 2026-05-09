from __future__ import annotations

import contextlib
import io
import logging
from typing import Any

import numpy as np
import pandas as pd
from scipy.integrate import trapezoid
from scipy.optimize import least_squares, linear_sum_assignment

from astroagent.agent.policy import SOURCE_WORK_WINDOW_KMS
from astroagent.spectra.line_catalog import load_line_catalog, transition_definitions
from astroagent.spectra.peak_detection import detect_absorption_peaks
from astroagent.spectra.voigt_model import (
    column_density_to_tau_scale,
    combined_physical_flux_model,
    physical_component_flux_model,
    voigt_velocity_profile,
)

try:
    from ultranest import ReactiveNestedSampler
except Exception:  # pragma: no cover - optional dependency
    ReactiveNestedSampler = None


C_KMS = 299792.458


def transition_velocity_kms(wavelength_A: np.ndarray, observed_center_A: float) -> np.ndarray:
    """Local velocity coordinate for one spectral transition."""
    return C_KMS * (np.asarray(wavelength_A, dtype=float) / float(observed_center_A) - 1.0)


def _prepare_arrays(plot_window: pd.DataFrame) -> dict[str, np.ndarray]:
    wavelength = plot_window["wavelength"].to_numpy(dtype=float)
    normalized_flux = plot_window["normalized_flux"].to_numpy(dtype=float)
    normalized_ivar = plot_window["normalized_ivar"].to_numpy(dtype=float)
    good = plot_window["is_good_pixel"].to_numpy(dtype=bool) if "is_good_pixel" in plot_window.columns else np.ones(len(plot_window), dtype=bool)
    finite = np.isfinite(wavelength) & np.isfinite(normalized_flux) & np.isfinite(normalized_ivar)
    good = good & finite & (normalized_ivar > 0.0)
    return {
        "wavelength": wavelength,
        "normalized_flux": normalized_flux,
        "normalized_ivar": normalized_ivar,
        "good": good,
    }


def _empty_model_columns(output: pd.DataFrame) -> pd.DataFrame:
    output["voigt_model"] = np.nan
    output["voigt_residual"] = np.nan
    output["voigt_residual_sigma"] = np.nan
    output["is_voigt_fit_pixel"] = 0
    output["voigt_component_index"] = -1
    output["voigt_transition_id"] = ""
    output["transition_velocity_kms"] = np.nan
    return output


def _fit_peak_group(
    velocity: np.ndarray,
    flux: np.ndarray,
    ivar: np.ndarray,
    good: np.ndarray,
    seeds: list[dict[str, Any]],
    local_fit_half_width_kms: float,
    center_shift_limit_kms: float,
    rest_wavelength_A: float,
    oscillator_strength: float,
    damping_gamma_kms: float,
    merge_separation_kms: float = 70.0,
) -> tuple[list[dict[str, Any]], np.ndarray, np.ndarray]:
    if not seeds:
        return [], np.full(len(velocity), np.nan, dtype=float), np.zeros(len(velocity), dtype=bool)

    seed_duplicates: set[int] = set()
    close_seed_merge_separation_kms = min(15.0, 0.25 * merge_separation_kms)
    for i in range(len(seeds)):
        for j in range(i + 1, len(seeds)):
            try:
                center_i = float(seeds[i].get("seed_velocity_kms", seeds[i].get("center_velocity_kms")))
                center_j = float(seeds[j].get("seed_velocity_kms", seeds[j].get("center_velocity_kms")))
            except (TypeError, ValueError):
                continue
            if not np.isfinite(center_i) or not np.isfinite(center_j):
                continue
            same_explicit_status = bool(seeds[i].get("explicit_fit_control_source")) == bool(seeds[j].get("explicit_fit_control_source"))
            if abs(center_i - center_j) < close_seed_merge_separation_kms:
                score_i = float(seeds[i].get("score", 0.0))
                score_j = float(seeds[j].get("score", 0.0))
                explicit_i = bool(seeds[i].get("explicit_fit_control_source"))
                explicit_j = bool(seeds[j].get("explicit_fit_control_source"))
                if explicit_i != explicit_j:
                    seed_duplicates.add(j if explicit_i else i)
                elif same_explicit_status:
                    seed_duplicates.add(j if score_i >= score_j else i)
    if seed_duplicates and len(seeds) - len(seed_duplicates) >= 1:
        seeds = [seed for index, seed in enumerate(seeds) if index not in seed_duplicates]

    fit_mask = good.copy()
    min_fit_pixels = max(8, 3 * len(seeds) + 3)
    if int(fit_mask.sum()) < min_fit_pixels:
        failed = [
            {
                **seed,
                "fit_success": False,
                "reason": f"too few good pixels in transition-frame simultaneous fit: {int(fit_mask.sum())} < {min_fit_pixels}",
                "fit_pixel_mask_local": fit_mask,
                "model_local": np.full(len(velocity), np.nan, dtype=float),
            }
            for seed in seeds
        ]
        return failed, np.full(len(velocity), np.nan, dtype=float), fit_mask

    fitted, model_local, fit_mask = _fit_peak_group_once(
        velocity,
        flux,
        ivar,
        fit_mask,
        seeds,
        local_fit_half_width_kms=local_fit_half_width_kms,
        center_shift_limit_kms=center_shift_limit_kms,
        rest_wavelength_A=rest_wavelength_A,
        oscillator_strength=oscillator_strength,
        damping_gamma_kms=damping_gamma_kms,
    )
    successful = [peak for peak in fitted if peak.get("fit_success")]
    centers = [float(peak.get("center_velocity_kms", np.nan)) for peak in successful]
    duplicates: set[int] = set()
    close_merge_separation_kms = min(12.0, 0.20 * merge_separation_kms)
    for i in range(len(centers)):
        for j in range(i + 1, len(centers)):
            if not np.isfinite(centers[i]) or not np.isfinite(centers[j]):
                continue
            separation = abs(centers[i] - centers[j])
            has_explicit = bool(successful[i].get("explicit_fit_control_source") or successful[j].get("explicit_fit_control_source"))
            if has_explicit and separation >= close_merge_separation_kms:
                continue
            if separation < merge_separation_kms:
                score_i = float(successful[i].get("score", 0.0))
                score_j = float(successful[j].get("score", 0.0))
                duplicates.add(successful[j]["_seed_order"] if score_i >= score_j else successful[i]["_seed_order"])

    if duplicates and len(seeds) - len(duplicates) >= 1:
        reduced_seeds = [seed for index, seed in enumerate(seeds) if index not in duplicates]
        fitted, model_local, fit_mask = _fit_peak_group_once(
            velocity,
            flux,
            ivar,
            good.copy(),
            reduced_seeds,
            local_fit_half_width_kms=local_fit_half_width_kms,
            center_shift_limit_kms=center_shift_limit_kms,
            rest_wavelength_A=rest_wavelength_A,
            oscillator_strength=oscillator_strength,
            damping_gamma_kms=damping_gamma_kms,
        )
        for peak in fitted:
            peak["duplicate_merge_applied"] = True

    return fitted, model_local, fit_mask


def _run_ultranest_physical_posterior(
    vel_fit: np.ndarray,
    flux_fit: np.ndarray,
    err_fit: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    center_prior_mean: np.ndarray,
    center_prior_sigma: np.ndarray,
    *,
    rest_wavelength_A: float,
    oscillator_strength: float,
    damping_gamma_kms: float,
    max_ncalls: int | None = None,
) -> dict[str, Any] | None:
    if ReactiveNestedSampler is None:
        return None

    lower = np.asarray(lower, dtype=float)
    upper = np.asarray(upper, dtype=float)
    span = upper - lower
    if lower.shape != upper.shape or np.any(~np.isfinite(lower)) or np.any(~np.isfinite(upper)) or np.any(span <= 0.0):
        return None

    param_names = [
        name
        for index in range(len(lower) // 3)
        for name in (
            f"logN_{index}",
            f"b_kms_{index}",
            f"center_velocity_kms_{index}",
        )
    ]

    def transform(unit_cube: np.ndarray) -> np.ndarray:
        return lower + span * np.asarray(unit_cube, dtype=float)

    def loglike(params: np.ndarray) -> float:
        params = np.asarray(params, dtype=float)
        model = combined_physical_flux_model(
            vel_fit,
            params,
            rest_wavelength_A=rest_wavelength_A,
            oscillator_strength=oscillator_strength,
            damping_gamma_kms=damping_gamma_kms,
        )
        resid = (flux_fit - model) / err_fit
        logp = -0.5 * float(np.sum(np.square(resid)))
        centers = params[2::3]
        sigma = np.maximum(center_prior_sigma, 1.0)
        logp += -0.5 * float(np.sum(np.square((centers - center_prior_mean) / sigma)))
        return logp

    sampler = ReactiveNestedSampler(param_names, loglike, transform=transform)
    logger_names = ["ultranest", "ultranest.integrator", "ultranest.mlfriends", "ultranest.netiter"]
    loggers = [logging.getLogger(name) for name in logger_names]
    old_disabled = [logger.disabled for logger in loggers]
    try:
        for logger in loggers:
            logger.disabled = True
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            return sampler.run(
                show_status=False,
                viz_callback=False,
                max_ncalls=int(max_ncalls or max(360, 70 * len(param_names))),
                min_num_live_points=max(48, 5 * len(param_names)),
            )
    except Exception:
        return None
    finally:
        for logger, disabled in zip(loggers, old_disabled, strict=True):
            logger.disabled = disabled


def _weighted_quantile(values: np.ndarray, weights: np.ndarray, quantiles: list[float]) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    weights = np.asarray(weights, dtype=float)
    finite = np.isfinite(values) & np.isfinite(weights) & (weights >= 0.0)
    if not finite.any():
        return np.full(len(quantiles), np.nan, dtype=float)
    values = values[finite]
    weights = weights[finite]
    if float(np.sum(weights)) <= 0.0:
        weights = np.ones_like(values, dtype=float)
    order = np.argsort(values)
    values = values[order]
    weights = weights[order]
    cdf = np.cumsum(weights)
    cdf = cdf / cdf[-1]
    return np.interp(quantiles, cdf, values)


def _parameter_summary_from_posterior(
    nested_result: dict[str, Any] | None,
    *,
    bounds: dict[str, tuple[float, float]],
    center_prior_mean: np.ndarray | None = None,
    center_prior_sigma: np.ndarray | None = None,
) -> dict[str, Any]:
    if nested_result is None:
        return {"backend": "least_squares", "posterior": {}}

    param_names = [str(name) for name in nested_result.get("paramnames", [])]
    points = np.asarray(nested_result.get("weighted_samples", {}).get("points", []), dtype=float)
    weights = np.asarray(nested_result.get("weighted_samples", {}).get("weights", []), dtype=float)
    points, relabelled = _relabel_posterior_points_by_center_prior(points, center_prior_mean)
    center_prior_mean = None if center_prior_mean is None else np.asarray(center_prior_mean, dtype=float)
    center_prior_sigma = None if center_prior_sigma is None else np.asarray(center_prior_sigma, dtype=float)
    posterior = nested_result.get("posterior", {})
    posterior_map: dict[str, Any] = {}
    for i, name in enumerate(param_names):
        if points.ndim == 2 and points.shape[0] and points.shape[1] > i and len(weights) == points.shape[0]:
            sample_mask = _component_center_condition_mask(name, points, center_prior_mean, center_prior_sigma)
            sample_points = points[sample_mask, i] if sample_mask is not None else points[:, i]
            sample_weights = weights[sample_mask] if sample_mask is not None else weights
            q16, q50, q84 = _weighted_quantile(sample_points, sample_weights, [0.16, 0.50, 0.84])
        else:
            median = posterior.get("median", [np.nan] * len(param_names))[i]
            errlo = posterior.get("errlo", [np.nan] * len(param_names))[i]
            errup = posterior.get("errup", [np.nan] * len(param_names))[i]
            q16, q50, q84 = float(errlo), float(median), float(errup)
        posterior_map[name] = {
            "q16": float(q16),
            "median": float(q50),
            "q84": float(q84),
            "mean": float(posterior.get("mean", [np.nan] * len(param_names))[i]),
            "stdev": float(posterior.get("stdev", [np.nan] * len(param_names))[i]),
            "prior_bounds": [float(bounds[name][0]), float(bounds[name][1])] if name in bounds else None,
        }
    return {
        "backend": "ultranest",
        "logz": float(nested_result.get("logz", np.nan)),
        "logzerr": float(nested_result.get("logzerr", np.nan)),
        "ncall": int(nested_result.get("ncall", 0) or 0),
        "posterior_component_labels": "center_prior_order" if relabelled else "sampler_order",
        "maximum_likelihood": nested_result.get("maximum_likelihood", {}),
        "posterior": posterior_map,
    }


def _component_center_condition_mask(
    name: str,
    points: np.ndarray,
    center_prior_mean: np.ndarray | None,
    center_prior_sigma: np.ndarray | None,
) -> np.ndarray | None:
    if center_prior_mean is None or center_prior_sigma is None or "_" not in name:
        return None
    try:
        component_index = int(str(name).rsplit("_", 1)[1])
    except ValueError:
        return None
    if component_index < 0 or component_index >= len(center_prior_mean) or component_index >= len(center_prior_sigma):
        return None
    center_column = component_index * 3 + 2
    if points.ndim != 2 or points.shape[1] <= center_column:
        return None
    mean = float(center_prior_mean[component_index])
    sigma = float(center_prior_sigma[component_index])
    if not np.isfinite(mean) or not np.isfinite(sigma):
        return None
    half_width = max(20.0, sigma)
    mask = np.isfinite(points[:, center_column]) & (np.abs(points[:, center_column] - mean) <= half_width)
    if int(mask.sum()) < max(16, int(0.05 * points.shape[0])):
        return None
    return mask


def _relabel_posterior_points_by_center_prior(
    points: np.ndarray,
    center_prior_mean: np.ndarray | None,
) -> tuple[np.ndarray, bool]:
    points = np.asarray(points, dtype=float)
    if points.ndim != 2 or not points.shape[0] or points.shape[1] < 6:
        return points, False
    if center_prior_mean is None:
        return points, False
    center_prior_mean = np.asarray(center_prior_mean, dtype=float)
    n_components = points.shape[1] // 3
    if n_components < 2 or len(center_prior_mean) != n_components or np.any(~np.isfinite(center_prior_mean)):
        return points, False

    relabelled = np.full_like(points, np.nan, dtype=float)
    for row_index, row in enumerate(points):
        centers = row[2::3]
        if len(centers) != n_components or np.any(~np.isfinite(centers)):
            relabelled[row_index] = row
            continue
        cost = np.abs(centers[:, None] - center_prior_mean[None, :])
        source_indices, target_indices = linear_sum_assignment(cost)
        reordered = np.empty_like(row, dtype=float)
        for source_index, target_index in zip(source_indices, target_indices, strict=True):
            reordered[target_index * 3 : target_index * 3 + 3] = row[source_index * 3 : source_index * 3 + 3]
        relabelled[row_index] = reordered
    return relabelled, True


def _component_interval(summary: dict[str, Any], name: str) -> dict[str, float]:
    item = summary.get("posterior", {}).get(name, {})
    return {
        "q16": float(item.get("q16", np.nan)),
        "median": float(item.get("median", np.nan)),
        "q84": float(item.get("q84", np.nan)),
    }


def _posterior_median_params(parameter_summary: dict[str, Any], fallback_params: np.ndarray) -> np.ndarray:
    params = np.full_like(np.asarray(fallback_params, dtype=float), np.nan, dtype=float)
    if parameter_summary.get("backend") != "ultranest":
        return params
    posterior = parameter_summary.get("posterior", {})
    for index in range(len(params) // 3):
        for offset, field in enumerate(("logN", "b_kms", "center_velocity_kms")):
            item = posterior.get(f"{field}_{index}", {})
            median = float(item.get("median", np.nan))
            if np.isfinite(median):
                params[index * 3 + offset] = median
    return params


def _component_parameter_diagnostics(
    *,
    index: int,
    eval_logN: float,
    eval_b_kms: float,
    eval_center_kms: float,
    map_logN: float,
    map_b_kms: float,
    map_center_kms: float,
    initializer_success: bool,
    parameter_summary: dict[str, Any],
    bounds_by_name: dict[str, tuple[float, float]],
) -> tuple[list[str], list[str], dict[str, Any]]:
    flags: list[str] = []
    warnings: list[str] = []
    details: dict[str, Any] = {}
    posterior = parameter_summary.get("posterior", {})
    eval_values = {
        f"logN_{index}": eval_logN,
        f"b_kms_{index}": eval_b_kms,
        f"center_velocity_kms_{index}": eval_center_kms,
    }
    map_values = {
        f"logN_{index}": map_logN,
        f"b_kms_{index}": map_b_kms,
        f"center_velocity_kms_{index}": map_center_kms,
    }
    mismatches: dict[str, dict[str, Any]] = {}
    at_bounds: list[str] = []
    for name, eval_value in eval_values.items():
        median = float(posterior.get(name, {}).get("median", np.nan))
        if initializer_success and np.isfinite(median):
            map_value = float(map_values[name])
            tolerance = 1.0
            if name.startswith("b_kms"):
                tolerance = max(30.0, 0.35 * max(abs(float(map_value)), 1.0))
            elif name.startswith("center_velocity"):
                tolerance = 45.0
            if abs(float(map_value) - median) > tolerance:
                mismatches[name] = {
                    "posterior_median": median,
                    "initializer_offset_exceeds_tolerance": True,
                }
        bounds = bounds_by_name.get(name)
        if bounds is None:
            continue
        lower, upper = float(bounds[0]), float(bounds[1])
        span = max(upper - lower, 1e-12)
        if abs(float(eval_value) - lower) <= 0.01 * span or abs(float(eval_value) - upper) <= 0.01 * span:
            at_bounds.append(name)
    if mismatches:
        flags.append("component_parameter_posterior_degenerate")
        warnings.append("component posterior is degenerate relative to the initialization; saturated/blended fit is weakly constrained")
        details["posterior_initializer_degeneracy"] = mismatches
    if at_bounds:
        flags.append("component_at_parameter_bound")
        warnings.append("component parameter is pinned near a fit bound")
        details["parameters_at_bounds"] = at_bounds
    return flags, warnings, details


def _guess_b_kms(seed: dict[str, Any]) -> float:
    if np.isfinite(float(seed.get("b_kms", np.nan))):
        return float(seed["b_kms"])
    width = float(seed.get("trough_velocity_max_kms", np.nan)) - float(seed.get("trough_velocity_min_kms", np.nan))
    if np.isfinite(width) and width > 0.0:
        return float(np.clip(width / 2.355, 8.0, 120.0))
    return 25.0


def _guess_logN(depth: float, b_kms: float, rest_wavelength_A: float, oscillator_strength: float, damping_gamma_kms: float) -> float:
    tau_center = -np.log(max(1.0 - float(np.clip(depth, 0.005, 0.98)), 0.02))
    sigma = max(float(b_kms) / np.sqrt(2.0), 1e-3)
    profile_peak = float(voigt_velocity_profile(np.asarray([0.0]), 0.0, sigma, damping_gamma_kms)[0])
    scale = 2.654e-15 * float(oscillator_strength) * float(rest_wavelength_A) * max(profile_peak, 1e-12)
    column_density = max(tau_center / scale, 1.0e8)
    return float(np.clip(np.log10(column_density), 10.0, 18.5))


def _fit_peak_group_once(
    velocity: np.ndarray,
    flux: np.ndarray,
    ivar: np.ndarray,
    fit_mask: np.ndarray,
    seeds: list[dict[str, Any]],
    *,
    local_fit_half_width_kms: float,
    center_shift_limit_kms: float,
    rest_wavelength_A: float,
    oscillator_strength: float,
    damping_gamma_kms: float,
) -> tuple[list[dict[str, Any]], np.ndarray, np.ndarray]:
    evaluation_mask = np.isfinite(velocity)
    vel_fit = velocity[fit_mask]
    flux_fit = flux[fit_mask]
    err_fit = np.zeros_like(flux_fit, dtype=float)
    np.divide(1.0, np.sqrt(ivar[fit_mask]), out=err_fit, where=ivar[fit_mask] > 0)
    finite_err = err_fit[np.isfinite(err_fit) & (err_fit > 0.0)]
    fallback_err = float(np.nanmedian(finite_err)) if len(finite_err) else 0.05
    err_fit = np.where(np.isfinite(err_fit) & (err_fit > 0.0), err_fit, fallback_err)
    err_fit = np.clip(err_fit, fallback_err * 0.10, fallback_err * 40.0)

    p0: list[float] = []
    lower: list[float] = []
    upper: list[float] = []
    center_prior_mean: list[float] = []
    center_prior_sigma: list[float] = []
    bounds_by_name: dict[str, tuple[float, float]] = {}

    finite_fit_velocity = velocity[fit_mask & np.isfinite(velocity)]
    frame_velocity_min = float(np.nanmin(finite_fit_velocity)) if len(finite_fit_velocity) else float(np.nanmin(velocity))
    frame_velocity_max = float(np.nanmax(finite_fit_velocity)) if len(finite_fit_velocity) else float(np.nanmax(velocity))

    for index, seed in enumerate(seeds):
        seed_velocity = float(seed["seed_velocity_kms"])
        depth = float(np.clip(seed["depth_below_continuum"], 0.005, 0.98))
        b_guess = float(np.clip(_guess_b_kms(seed), 5.0, 200.0))
        logN_guess = float(seed.get("logN", _guess_logN(depth, b_guess, rest_wavelength_A, oscillator_strength, damping_gamma_kms)))
        prior_sigma = float(seed.get("center_prior_sigma_kms", max(30.0, 0.35 * center_shift_limit_kms)))
        prior_sigma = float(np.clip(prior_sigma, 5.0, 500.0))
        prior_half_width = float(seed.get("center_prior_half_width_kms", max(center_shift_limit_kms, 3.0 * prior_sigma)))
        prior_half_width = float(np.clip(prior_half_width, 20.0, 1200.0))
        center_lower = max(seed_velocity - prior_half_width, frame_velocity_min)
        center_upper = min(seed_velocity + prior_half_width, frame_velocity_max)
        if center_upper <= center_lower:
            center_lower = seed_velocity - 1.0
            center_upper = seed_velocity + 1.0

        logN_lower = float(seed.get("logN_lower", 10.0))
        logN_upper = float(seed.get("logN_upper", 18.5))
        b_lower = float(seed.get("b_kms_lower", 5.0))
        b_upper = float(seed.get("b_kms_upper", 200.0))
        p0.extend(
            [
                np.clip(logN_guess, logN_lower, logN_upper),
                np.clip(b_guess, b_lower, b_upper),
                np.clip(seed_velocity, center_lower, center_upper),
            ]
        )
        lower.extend([logN_lower, b_lower, center_lower])
        upper.extend([logN_upper, b_upper, center_upper])
        center_prior_mean.append(seed_velocity)
        center_prior_sigma.append(prior_sigma)
        bounds_by_name[f"logN_{index}"] = (logN_lower, logN_upper)
        bounds_by_name[f"b_kms_{index}"] = (b_lower, b_upper)
        bounds_by_name[f"center_velocity_kms_{index}"] = (center_lower, center_upper)

    lower_array = np.asarray(lower, dtype=float)
    upper_array = np.asarray(upper, dtype=float)
    p0_array = np.clip(np.asarray(p0, dtype=float), lower_array + 1e-8, upper_array - 1e-8)
    center_prior_mean_array = np.asarray(center_prior_mean, dtype=float)
    center_prior_sigma_array = np.asarray(center_prior_sigma, dtype=float)

    def model_for(params: np.ndarray, x: np.ndarray) -> np.ndarray:
        return combined_physical_flux_model(
            x,
            params,
            rest_wavelength_A=rest_wavelength_A,
            oscillator_strength=oscillator_strength,
            damping_gamma_kms=damping_gamma_kms,
        )

    def residual(params: np.ndarray) -> np.ndarray:
        model = model_for(params, vel_fit)
        data_resid = (flux_fit - model) / err_fit
        prior_resid = (params[2::3] - center_prior_mean_array) / np.maximum(center_prior_sigma_array, 1.0)
        return np.concatenate([data_resid, prior_resid])

    try:
        ls_result = least_squares(residual, p0_array, bounds=(lower_array, upper_array), max_nfev=5000)
        map_params = np.asarray(ls_result.x, dtype=float)
        initializer_success = bool(ls_result.success)
        initializer_reason = "" if initializer_success else str(ls_result.message)
    except Exception as exc:
        map_params = p0_array.copy()
        initializer_success = False
        initializer_reason = f"least-squares initializer failed: {exc}"

    nested_result = _run_ultranest_physical_posterior(
        vel_fit,
        flux_fit,
        err_fit,
        lower_array,
        upper_array,
        center_prior_mean_array,
        center_prior_sigma_array,
        rest_wavelength_A=rest_wavelength_A,
        oscillator_strength=oscillator_strength,
        damping_gamma_kms=damping_gamma_kms,
    )
    if nested_result is not None:
        fit_backend = "ultranest"
        fit_method = "ultranest_physical_transition_frame"
    else:
        fit_backend = "least_squares"
        fit_method = "least_squares_physical_transition_frame"

    parameter_summary = _parameter_summary_from_posterior(
        nested_result,
        bounds=bounds_by_name,
        center_prior_mean=center_prior_mean_array,
        center_prior_sigma=center_prior_sigma_array,
    )
    fit_success = parameter_summary.get("backend") == "ultranest"
    fit_reason = "" if fit_success else initializer_reason
    params = _posterior_median_params(parameter_summary, map_params)
    if not np.all(np.isfinite(params)):
        failed = []
        for index, seed in enumerate(seeds):
            offset = index * 3
            failed.append(
                {
                    **seed,
                    "_seed_order": index,
                    "fit_success": False,
                    "reason": "bayesian posterior unavailable; least-squares MAP kept only as initialization",
                    "fit_backend": fit_backend,
                    "fit_method": fit_method,
                    "fit_model": "multi_component_physical_voigt",
                    "parameter_estimator": "none_posterior_unavailable",
                    "fit_pixel_mask_local": fit_mask,
                    "model_local": np.full(len(velocity), np.nan, dtype=float),
                    "combined_model_local": np.full(len(velocity), np.nan, dtype=float),
                    "diagnostic_flags": ["bayesian_posterior_unavailable"],
                    "diagnostic_warnings": ["least-squares MAP is not used as a public fit result"],
                    "diagnostic_details": {"posterior_backend": parameter_summary.get("backend")},
                }
            )
        return failed, np.full(len(velocity), np.nan, dtype=float), fit_mask
    combined_fit = model_for(params, vel_fit)
    norm_residual = (flux_fit - combined_fit) / err_fit
    chi2 = float(np.sum(np.square(norm_residual)))
    dof = int(len(vel_fit) - len(params))
    reduced_chi2 = float(chi2 / dof) if dof > 0 else float("nan")
    fit_rms = float(np.sqrt(np.mean(np.square(norm_residual)))) if len(norm_residual) else float("nan")

    combined_model_local = np.full(len(velocity), np.nan, dtype=float)
    combined_model_local[evaluation_mask] = model_for(params, velocity[evaluation_mask])

    fitted: list[dict[str, Any]] = []
    for index, seed in enumerate(seeds):
        offset = index * 3
        logN = float(params[offset])
        b_kms = float(params[offset + 1])
        center_kms = float(params[offset + 2])
        map_logN = float(map_params[offset])
        map_b_kms = float(map_params[offset + 1])
        map_center_kms = float(map_params[offset + 2])
        component_model_local = np.full(len(velocity), np.nan, dtype=float)
        component_model_local[evaluation_mask] = physical_component_flux_model(
            velocity[evaluation_mask],
            logN=logN,
            b_kms=b_kms,
            center_kms=center_kms,
            rest_wavelength_A=rest_wavelength_A,
            oscillator_strength=oscillator_strength,
            damping_gamma_kms=damping_gamma_kms,
        )
        ew_half_width = max(float(local_fit_half_width_kms), 6.0 * b_kms)
        fine_velocity = np.linspace(center_kms - ew_half_width, center_kms + ew_half_width, 1000)
        fine_model = physical_component_flux_model(
            fine_velocity,
            logN=logN,
            b_kms=b_kms,
            center_kms=center_kms,
            rest_wavelength_A=rest_wavelength_A,
            oscillator_strength=oscillator_strength,
            damping_gamma_kms=damping_gamma_kms,
        )
        equivalent_width_velocity_kms = float(trapezoid(1.0 - fine_model, fine_velocity))
        diagnostic_flags, diagnostic_warnings, diagnostic_details = _component_parameter_diagnostics(
            index=index,
            eval_logN=logN,
            eval_b_kms=b_kms,
            eval_center_kms=center_kms,
            map_logN=map_logN,
            map_b_kms=map_b_kms,
            map_center_kms=map_center_kms,
            initializer_success=initializer_success,
            parameter_summary=parameter_summary,
            bounds_by_name=bounds_by_name,
        )
        explicit_saturated_siblings = [
            int(other_index)
            for other_index, other_seed in enumerate(seeds)
            if other_index != index
            and bool(seed.get("explicit_fit_control_source"))
            and bool(other_seed.get("explicit_fit_control_source"))
            and abs(float(other_seed.get("seed_velocity_kms", np.nan)) - float(seed.get("seed_velocity_kms", np.nan)))
            <= max(160.0, 2.5 * max(float(seed.get("b_kms", b_kms)), float(other_seed.get("b_kms", b_kms)), 1.0))
        ]
        if len(seeds) > 1:
            other_params = np.concatenate([params[:offset], params[offset + 3 :]])
            other_model_fit = combined_physical_flux_model(
                vel_fit,
                other_params,
                rest_wavelength_A=rest_wavelength_A,
                oscillator_strength=oscillator_strength,
                damping_gamma_kms=damping_gamma_kms,
            )
            component_fit = physical_component_flux_model(
                vel_fit,
                logN=logN,
                b_kms=b_kms,
                center_kms=center_kms,
                rest_wavelength_A=rest_wavelength_A,
                oscillator_strength=oscillator_strength,
                damping_gamma_kms=damping_gamma_kms,
            )
            significant_component = np.isfinite(component_fit) & (component_fit < 0.65)
            if significant_component.any():
                component_min_flux = float(np.nanmin(component_fit[significant_component]))
                other_median_flux = float(np.nanmedian(other_model_fit[significant_component]))
                other_min_flux = float(np.nanmin(other_model_fit[significant_component]))
                combined_min_flux = float(np.nanmin(combined_fit[significant_component]))
                combined_saturated_pixels = np.isfinite(combined_fit) & (combined_fit < 0.08)
                component_in_saturated_pixels = (
                    np.isfinite(component_fit)
                    & combined_saturated_pixels
                    & (component_fit < 0.65)
                )
                other_in_saturated_pixels = (
                    np.isfinite(other_model_fit)
                    & combined_saturated_pixels
                    & (other_model_fit < 0.65)
                )
                shared_saturated_pixels = int(np.sum(component_in_saturated_pixels & other_in_saturated_pixels))
                diagnostic_details["saturation_overlap"] = {
                    "component_min_flux": component_min_flux,
                    "other_components_median_flux_on_component_absorption": other_median_flux,
                    "other_components_min_flux_on_component_absorption": other_min_flux,
                    "combined_min_flux": combined_min_flux,
                    "n_fit_pixels": int(significant_component.sum()),
                    "shared_saturated_fit_pixels": shared_saturated_pixels,
                    "interpretation": (
                        "Voigt components are transmission curves; total flux is a product, so saturated redundant "
                        "components do not add linearly or make the combined model negative."
                    ),
                }
                if (
                    component_min_flux < 0.35
                    and (other_median_flux < 0.45 or other_min_flux < 0.35 or shared_saturated_pixels >= 3)
                ):
                    diagnostic_flags.append("saturated_redundant_component")
                    diagnostic_warnings.append(
                        "component is fitted on pixels already made saturated by other components; residuals may not penalize it"
                    )
        if explicit_saturated_siblings and logN >= 15.0 and b_kms >= 5.0:
            seed_velocity = float(seed.get("seed_velocity_kms", center_kms))
            seed_half_width = max(45.0, 0.75 * float(seed.get("b_kms", b_kms)))
            local_seed_region = np.isfinite(vel_fit) & (np.abs(vel_fit - seed_velocity) <= seed_half_width)
            local_seed_min_flux = float(np.nanmin(flux_fit[local_seed_region])) if local_seed_region.any() else float("nan")
            diagnostic_details.setdefault("saturation_overlap", {})
            diagnostic_details["saturation_overlap"].update(
                {
                    "explicit_saturated_sibling_seed_orders": explicit_saturated_siblings,
                    "local_seed_region_min_flux": local_seed_min_flux,
                    "explicit_sibling_interpretation": (
                        "Multiple explicit sources were placed on one saturated trough; in black/near-black pixels "
                        "residuals cannot reliably distinguish extra Voigt components."
                    ),
                }
            )
            if np.isfinite(local_seed_min_flux) and local_seed_min_flux < 0.12:
                diagnostic_flags.append("saturated_redundant_component")
                diagnostic_warnings.append(
                    "explicit source sits on a saturated trough already represented by nearby explicit sources"
                )
        fitted.append(
            {
                **seed,
                "_seed_order": index,
                "fit_success": bool(fit_success),
                "reason": fit_reason,
                "fit_backend": fit_backend,
                "fit_method": fit_method,
                "parameter_estimator": "posterior_median" if parameter_summary.get("backend") == "ultranest" else "none_posterior_unavailable",
                "fit_model": "multi_component_physical_voigt",
                "fit_window_half_width_kms": float(local_fit_half_width_kms),
                "center_velocity_kms": center_kms,
                "center_offset_from_seed_kms": float(center_kms - float(seed["seed_velocity_kms"])),
                "center_prior": {
                    "mean_kms": float(center_prior_mean_array[index]),
                    "sigma_kms": float(center_prior_sigma_array[index]),
                    "bounds_kms": [float(lower_array[offset + 2]), float(upper_array[offset + 2])],
                },
                "logN": logN,
                "b_kms": b_kms,
                "damping_gamma_kms": float(damping_gamma_kms),
                "tau_scale": column_density_to_tau_scale(logN, rest_wavelength_A, oscillator_strength),
                "parameter_intervals": {
                    "logN": _component_interval(parameter_summary, f"logN_{index}"),
                    "b_kms": _component_interval(parameter_summary, f"b_kms_{index}"),
                    "center_velocity_kms": _component_interval(parameter_summary, f"center_velocity_kms_{index}"),
                },
                "chi2": chi2,
                "reduced_chi2": reduced_chi2,
                "fit_rms": fit_rms,
                "n_fit_pixels": int(fit_mask.sum()),
                "equivalent_width_velocity_kms": equivalent_width_velocity_kms,
                "fit_pixel_mask_local": fit_mask,
                "model_local": component_model_local,
                "combined_model_local": combined_model_local,
                "diagnostic_flags": sorted(set(diagnostic_flags)),
                "diagnostic_warnings": sorted(set(diagnostic_warnings)),
                "diagnostic_details": diagnostic_details,
            }
        )
    return fitted, combined_model_local, fit_mask


def _public_peak(peak: dict[str, Any]) -> dict[str, Any]:
    private = {
        "fit_pixel_mask_local",
        "model_local",
        "combined_model_local",
        "_seed_order",
        "map_parameters",
        "fit_parameter_summary",
    }
    return {key: value for key, value in peak.items() if key not in private}


def _frame_quality(peaks: list[dict[str, Any]], flux: np.ndarray, good: np.ndarray) -> tuple[str, list[str], list[str]]:
    warnings: list[str] = []
    reasons: list[str] = []
    successful = [peak for peak in peaks if peak.get("fit_success")]
    if not successful:
        warnings.append("no peak fit succeeded in this transition frame")
        reasons.append("transition_frame_fit_failed")
    if len(successful) > 3:
        warnings.append("many components in one transition frame; model needs review")
        reasons.append("many_components_in_transition_frame")
    for peak in successful:
        for flag in peak.get("diagnostic_flags", []):
            if flag == "bayesian_posterior_unavailable":
                warnings.append("bayesian posterior unavailable for this transition frame")
                reasons.append("bayesian_posterior_unavailable")
            elif flag == "saturated_redundant_component":
                warnings.append("redundant component in already saturated model region")
                reasons.append("saturated_redundant_component")
            elif flag in {"component_parameter_posterior_degenerate", "component_parameter_posterior_disagrees_with_map"}:
                warnings.append("component posterior is degenerate relative to the initialization")
                reasons.append("component_parameter_posterior_degenerate")
            elif flag == "component_at_parameter_bound":
                warnings.append("component parameter is near a fit bound")
                reasons.append("component_at_parameter_bound")
        if float(peak.get("fit_rms", 0.0)) > 3.0:
            warnings.append("high residual around a fitted peak")
            reasons.append("high_residual_transition_peak")
            break
        intervals = peak.get("parameter_intervals", {})
        logN_interval = intervals.get("logN", {})
        if float(logN_interval.get("q84", peak.get("logN", 0.0))) - float(logN_interval.get("q16", peak.get("logN", 0.0))) > 1.5:
            warnings.append("column density is weakly constrained")
            reasons.append("weak_logN_constraint")
            break
    if good.any() and float(np.nanmin(flux[good])) < -0.1:
        warnings.append("negative normalized flux in transition frame")
        reasons.append("negative_normalized_flux")
    if good.any() and float(np.nanmin(flux[good])) < 0.08:
        warnings.append("saturated or near-black trough; column density can be degenerate")
        reasons.append("saturated_peak")
    quality = "good" if not warnings else ("failed" if not successful else "inspect")
    return quality, warnings, reasons


def fit_voigt_absorption(
    plot_window: pd.DataFrame,
    window_metadata: dict[str, Any] | None,
    transition_half_width_kms: float = 600.0,
    local_fit_half_width_kms: float = 180.0,
    center_shift_limit_kms: float = 120.0,
    min_peak_depth: float = 0.025,
    max_peaks_per_transition: int = 4,
    min_peak_separation_kms: float = 100.0,
    fit_control: dict[str, Any] | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Fit physical Voigt components inside each transition velocity frame.

    `line_id` and `z_sys` define transition frames. Detected peaks or LLM-added
    sources provide priors, not fixed centers. Each transition is fit in its
    own velocity coordinate; sibling transition projections are masked to avoid
    fitting the other doublet member as a local multi-peak solution.
    """
    output = _empty_model_columns(plot_window.copy())
    arrays = _prepare_arrays(output)
    wavelength = arrays["wavelength"]
    normalized_flux = arrays["normalized_flux"]
    normalized_ivar = arrays["normalized_ivar"]
    good_all = arrays["good"]

    if good_all.sum() < 8:
        return output, _failed_summary("too few good pixels for transition-frame fit")

    metadata = window_metadata or {}
    controls = fit_control or {}
    line_id = str(metadata.get("line_id", ""))
    z_sys = float(metadata.get("z_sys", np.nan))
    if not line_id or not np.isfinite(z_sys):
        return output, _failed_summary("window metadata must include line_id and z_sys")

    catalog = load_line_catalog(metadata.get("catalog_path"))
    transitions = transition_definitions(line_id, catalog)
    transition_frames: list[dict[str, Any]] = []
    global_model = np.full(len(output), np.nan, dtype=float)
    global_fit_mask = np.zeros(len(output), dtype=bool)
    frame_fit_residuals: list[np.ndarray] = []
    frame_work_residuals: list[np.ndarray] = []
    component_index = 0
    review_reasons: list[str] = []

    for transition in transitions:
        transition_line_id = str(transition["transition_line_id"])
        window_control = controls.get("fit_windows", {}).get(transition_line_id, {})
        frame_half_width_kms = float(window_control.get("transition_half_width_kms", transition_half_width_kms))
        frame_local_half_width_kms = float(window_control.get("local_fit_half_width_kms", local_fit_half_width_kms))
        rest_A = float(transition["rest_wavelength_A"])
        oscillator_strength = float(transition.get("oscillator_strength", 1.0))
        damping_gamma_kms = float(transition.get("damping_gamma_kms", 0.001))
        observed_center_A = rest_A * (1.0 + z_sys)
        velocity = transition_velocity_kms(wavelength, observed_center_A)
        frame_mask = np.abs(velocity) <= frame_half_width_kms
        source_fit_half_width_kms = float(window_control.get("source_fit_half_width_kms", SOURCE_WORK_WINDOW_KMS[1]))
        source_fit_half_width_kms = float(np.clip(source_fit_half_width_kms, 80.0, frame_half_width_kms))
        source_fit_mask = np.abs(velocity) <= source_fit_half_width_kms
        frame_good = good_all & frame_mask
        fit_good = frame_good & source_fit_mask
        sibling_mask_mode = str(window_control.get("sibling_mask_mode", "exclude"))
        if sibling_mask_mode == "allow_overlap":
            sibling_intervals = []
        else:
            sibling_intervals = _sibling_transition_intervals(
                transition,
                transitions,
                frame_half_width_kms=frame_half_width_kms,
                exclusion_half_width_kms=max(frame_local_half_width_kms, min(240.0, 0.35 * frame_half_width_kms)),
            )
        frame_good = _apply_fit_mask_intervals(velocity, frame_good, transition_line_id, sibling_intervals)
        frame_good = _apply_fit_mask_intervals(velocity, frame_good, transition_line_id, controls.get("fit_mask_intervals", []))
        fit_good = _apply_fit_mask_intervals(velocity, fit_good, transition_line_id, sibling_intervals)
        fit_good = _apply_fit_mask_intervals(velocity, fit_good, transition_line_id, controls.get("fit_mask_intervals", []))
        output.loc[frame_mask, "voigt_transition_id"] = transition_line_id
        output.loc[frame_mask, "transition_velocity_kms"] = velocity[frame_mask]

        frame_record: dict[str, Any] = {
            "transition_line_id": transition_line_id,
            "family": transition.get("family", line_id),
            "rest_wavelength_A": rest_A,
            "oscillator_strength": oscillator_strength,
            "damping_gamma_kms": damping_gamma_kms,
            "atomic_label": transition.get("atomic_label"),
            "observed_center_A": float(observed_center_A),
            "velocity_frame": {
                "zero": "transition observed center from rest_wavelength_A * (1 + z_sys)",
                "bounds_kms": [-frame_half_width_kms, frame_half_width_kms],
                "n_pixels": int(frame_mask.sum()),
                "n_good_pixels": int(frame_good.sum()),
            },
            "simultaneous_fit_window": {
                "bounds_kms": [-frame_half_width_kms, frame_half_width_kms],
                "masked_by": "good pixels, sibling transition projections, and fit_control mask intervals",
                "source_fit_bounds_kms": [-source_fit_half_width_kms, source_fit_half_width_kms],
                "source_fit_policy": "Voigt parameters are fit in the source work window by default; context pixels are diagnostic unless set_fit_window expands source_fit_half_width_kms.",
                "model": "all retained components in this transition frame are fit together with physical logN/b/v parameters",
                "sibling_mask_mode": sibling_mask_mode,
            },
            "sibling_transition_masks": sibling_intervals,
            "peaks": [],
        }
        if int(fit_good.sum()) < 8:
            frame_record.update(
                {
                    "success": False,
                    "quality": "failed",
                    "agent_review_required": True,
                    "agent_review_reasons": ["too_few_good_pixels_in_transition_frame"],
                    "reason": "too few good pixels in transition velocity frame",
                }
            )
            review_reasons.extend(frame_record["agent_review_reasons"])
            transition_frames.append(frame_record)
            continue

        source_detection_mode = str(controls.get("source_detection_mode", "auto"))
        if source_detection_mode == "controlled":
            seeds = []
        else:
            seeds = detect_absorption_peaks(
                velocity[frame_mask],
                normalized_flux[frame_mask],
                min_peak_depth=min_peak_depth,
                max_peaks=max_peaks_per_transition,
                min_separation_kms=min_peak_separation_kms,
                good=fit_good[frame_mask],
            )
        seeds = _merge_control_source_seeds(
            seeds,
            transition_line_id=transition_line_id,
            controls=controls,
            min_peak_depth=min_peak_depth,
            min_peak_separation_kms=min_peak_separation_kms,
        )
        seeds = _remove_suppressed_source_seeds(
            seeds,
            transition_line_id=transition_line_id,
            controls=controls,
        )
        frame_indices = np.flatnonzero(frame_mask)
        frame_velocity = velocity[frame_mask]
        frame_flux = normalized_flux[frame_mask]
        frame_ivar = normalized_ivar[frame_mask]
        frame_good_local = frame_good[frame_mask]
        fit_good_local = fit_good[frame_mask]
        if not seeds:
            null_model_local = np.ones_like(frame_flux, dtype=float)
            null_fit_mask = fit_good_local & np.isfinite(frame_flux) & np.isfinite(frame_ivar) & (frame_ivar > 0.0)
            if null_fit_mask.any():
                global_fit_mask[frame_indices[null_fit_mask]] = True
                fill = ~np.isfinite(global_model[frame_indices[null_fit_mask]])
                target_indices = frame_indices[null_fit_mask]
                global_model[target_indices[fill]] = null_model_local[null_fit_mask][fill]
                global_model[target_indices[~fill]] = np.minimum(
                    global_model[target_indices[~fill]],
                    null_model_local[null_fit_mask][~fill],
                )
            residual_samples = _frame_residual_samples(
                wavelength[frame_mask],
                frame_velocity,
                frame_flux,
                frame_ivar,
                null_model_local,
                null_fit_mask,
            )
            diagnostic_residual_samples = _frame_diagnostic_residual_samples(
                wavelength[frame_mask],
                frame_velocity,
                frame_flux,
                frame_ivar,
                null_model_local,
                frame_good_local & np.isfinite(frame_flux) & np.isfinite(frame_ivar) & (frame_ivar > 0.0),
                null_fit_mask,
            )
            _append_frame_residuals(residual_samples, frame_fit_residuals, frame_work_residuals)
            frame_residual_metrics = _residual_metrics_from_samples(residual_samples)
            frame_work_metrics = _residual_metrics_from_samples(
                sample
                for sample in residual_samples
                if SOURCE_WORK_WINDOW_KMS[0] <= float(sample.get("velocity_kms", np.nan)) <= SOURCE_WORK_WINDOW_KMS[1]
            )
            frame_record.update(
                {
                    "success": False,
                    "quality": "failed",
                    "agent_review_required": True,
                    "agent_review_reasons": ["no_absorption_peak_detected_in_transition_frame"],
                    "reason": "no absorption peak above depth threshold after sibling/mask exclusions; residuals use a flat no-source model",
                    "peak_detection": {
                        "method": "find_peaks_on_smoothed_absorption_depth_with_center_priors",
                        "min_peak_depth": float(min_peak_depth),
                        "max_peaks_per_transition": int(max_peaks_per_transition),
                        "source_detection_mode": source_detection_mode,
                    },
                    "n_successful_peak_fits": 0,
                    "fit_metrics": frame_residual_metrics,
                    "source_work_window_metrics": frame_work_metrics,
                    "residual_model": "flat_no_source_model",
                    "residual_samples": residual_samples,
                    "diagnostic_residual_samples": diagnostic_residual_samples,
                }
            )
            review_reasons.extend(frame_record["agent_review_reasons"])
            transition_frames.append(frame_record)
            continue

        fitted_peaks, combined_model_local, combined_fit_mask = _fit_peak_group(
            frame_velocity,
            frame_flux,
            frame_ivar,
            fit_good_local,
            seeds=seeds,
            local_fit_half_width_kms=frame_local_half_width_kms,
            center_shift_limit_kms=center_shift_limit_kms,
            rest_wavelength_A=rest_A,
            oscillator_strength=oscillator_strength,
            damping_gamma_kms=damping_gamma_kms,
            merge_separation_kms=0.70 * min_peak_separation_kms,
        )
        combined_finite = np.isfinite(combined_model_local)
        if combined_finite.any():
            full_indices = frame_indices[combined_finite]
            full_model = combined_model_local[combined_finite]
            fill = ~np.isfinite(global_model[full_indices])
            global_model[full_indices[fill]] = full_model[fill]
            global_model[full_indices[~fill]] = np.minimum(global_model[full_indices[~fill]], full_model[~fill])
        if combined_fit_mask.any():
            global_fit_mask[frame_indices[combined_fit_mask]] = True
        residual_samples = _frame_residual_samples(
            wavelength[frame_mask],
            frame_velocity,
            frame_flux,
            frame_ivar,
            combined_model_local,
            combined_fit_mask,
        )
        diagnostic_residual_samples = _frame_diagnostic_residual_samples(
            wavelength[frame_mask],
            frame_velocity,
            frame_flux,
            frame_ivar,
            combined_model_local,
            frame_good_local,
            combined_fit_mask,
        )
        _append_frame_residuals(residual_samples, frame_fit_residuals, frame_work_residuals)
        frame_residual_metrics = _residual_metrics_from_samples(residual_samples)
        frame_work_metrics = _residual_metrics_from_samples(
            sample
            for sample in residual_samples
            if SOURCE_WORK_WINDOW_KMS[0] <= float(sample.get("velocity_kms", np.nan)) <= SOURCE_WORK_WINDOW_KMS[1]
        )

        for fitted in fitted_peaks:
            fitted["component_index"] = int(component_index)
            fitted["transition_line_id"] = transition_line_id
            fitted["rest_wavelength_A"] = rest_A
            fitted["oscillator_strength"] = oscillator_strength
            center_velocity = float(fitted.get("center_velocity_kms", fitted.get("seed_velocity_kms", np.nan)))
            fitted["center_wavelength_A"] = float(observed_center_A * (1.0 + center_velocity / C_KMS))
            fitted["observed_equivalent_width_mA"] = float(
                fitted.get("equivalent_width_velocity_kms", 0.0) * rest_A * (1.0 + z_sys) / C_KMS * 1000.0
            )
            component_index += 1

        successful_peaks = [peak for peak in fitted_peaks if peak.get("fit_success")]
        quality, warnings, frame_review_reasons = _frame_quality(fitted_peaks, frame_flux, fit_good_local)
        review_reasons.extend(frame_review_reasons)
        frame_record.update(
            {
                "success": bool(successful_peaks),
                "quality": quality,
                "quality_warnings": warnings,
                "agent_review_required": bool(frame_review_reasons),
                "agent_review_reasons": frame_review_reasons,
                "peak_detection": {
                    "method": "find_peaks_on_smoothed_absorption_depth_with_center_priors",
                    "min_peak_depth": float(min_peak_depth),
                    "min_peak_separation_kms": float(min_peak_separation_kms),
                    "n_detected_peaks": int(len(seeds)),
                    "center_use": "detected centers are prior means, not fixed fit centers",
                    "source_detection_mode": source_detection_mode,
                },
                "n_successful_peak_fits": int(len(successful_peaks)),
                "fit_metrics": frame_residual_metrics,
                "source_work_window_metrics": frame_work_metrics,
                "residual_samples": residual_samples,
                "diagnostic_residual_samples": diagnostic_residual_samples,
                "peaks": [_public_peak(peak) for peak in fitted_peaks],
            }
        )
        transition_frames.append(frame_record)

    output["voigt_model"] = global_model
    output["voigt_residual"] = np.nan
    output.loc[global_fit_mask, "voigt_residual"] = normalized_flux[global_fit_mask] - global_model[global_fit_mask]
    output["voigt_residual_sigma"] = np.nan
    with np.errstate(divide="ignore", invalid="ignore"):
        err_all = np.where(normalized_ivar > 0.0, 1.0 / np.sqrt(normalized_ivar), np.nan)
    output.loc[global_fit_mask, "voigt_residual_sigma"] = (
        normalized_flux[global_fit_mask] - global_model[global_fit_mask]
    ) / err_all[global_fit_mask]
    output["is_voigt_fit_pixel"] = global_fit_mask.astype(int)

    fitted_public_peaks = [
        peak
        for frame in transition_frames
        for peak in frame.get("peaks", [])
        if peak.get("fit_success")
    ]
    for peak in fitted_public_peaks:
        peak_half_width = float(peak.get("fit_window_half_width_kms", local_fit_half_width_kms))
        peak_mask = (
            (output["voigt_transition_id"].to_numpy() == peak["transition_line_id"])
            & np.isfinite(output["transition_velocity_kms"].to_numpy(dtype=float))
            & (np.abs(output["transition_velocity_kms"].to_numpy(dtype=float) - float(peak["center_velocity_kms"])) <= peak_half_width)
        )
        output.loc[peak_mask, "voigt_component_index"] = int(peak["component_index"])

    if frame_fit_residuals:
        finite_resid = np.concatenate(frame_fit_residuals)
        finite_resid = finite_resid[np.isfinite(finite_resid)]
    else:
        finite_resid = np.array([], dtype=float)
    n_fit_pixels = int(len(finite_resid))
    if n_fit_pixels:
        chi2 = float(np.sum(np.square(finite_resid)))
        reduced_chi2 = float(chi2 / n_fit_pixels)
        fit_rms = float(np.sqrt(reduced_chi2))
    else:
        chi2 = float("nan")
        reduced_chi2 = float("nan")
        fit_rms = float("nan")
    work_window_metrics = _source_work_window_metrics_from_residuals(frame_work_residuals)

    unique_review_reasons = sorted(set(review_reasons))
    success = any(frame.get("success") for frame in transition_frames)
    status = "fit" if success else "failed"
    quality = "inspect" if unique_review_reasons else ("good" if success else "failed")
    total_ew_mA = float(sum(float(peak.get("observed_equivalent_width_mA", 0.0)) for peak in fitted_public_peaks))
    fit_methods = sorted({str(peak.get("fit_method")) for peak in fitted_public_peaks if peak.get("fit_method")})
    fit_backends = sorted({str(peak.get("fit_backend")) for peak in fitted_public_peaks if peak.get("fit_backend")})
    fit_method = fit_methods[0] if len(fit_methods) == 1 else ("mixed_transition_frame_fit" if fit_methods else "ultranest_physical_transition_frame")

    return output, {
        "fit_type": "transition_frame_peak_voigt",
        "fit_method": fit_method,
        "fit_backends": fit_backends,
        "fit_model": "per_transition_multi_component_physical_voigt",
        "model_equation": "normalized_flux = product_i VoigtFlux(logN_i, b_i, v_i; transition rest_A, f, gamma)",
        "success": bool(success),
        "status": status,
        "quality": quality,
        "n_good_pixels": int(good_all.sum()),
        "n_fit_pixels": n_fit_pixels,
        "n_transition_frames": int(len(transition_frames)),
        "n_components_fitted": int(len(fitted_public_peaks)),
        "chi2": chi2,
        "reduced_chi2": reduced_chi2,
        "fit_rms": fit_rms,
        "source_work_window_kms": list(SOURCE_WORK_WINDOW_KMS),
        "source_work_window_metrics": work_window_metrics,
        "transition_half_width_kms": float(transition_half_width_kms),
        "local_fit_half_width_kms": float(local_fit_half_width_kms),
        "center_shift_limit_kms": float(center_shift_limit_kms),
        "line_id": line_id,
        "z_sys": z_sys,
        "transition_frames": transition_frames,
        "components": [{"kind": "transition_frame_peak", **peak} for peak in fitted_public_peaks],
        "observed_equivalent_width_mA": total_ew_mA,
        "post_window_metadata_used": True,
        "metadata_use": "line_id and z_sys define transition frames only; peak centers become posterior priors",
        "line_identity_used_after_window": "transition frame construction and transition atomic constants only",
        "agent_review": {
            "required": bool(unique_review_reasons),
            "reviewer": "llm_or_human",
            "reasons": unique_review_reasons,
            "suggested_actions": ["inspect_transition_frames"] if unique_review_reasons else [],
        },
        "fit_control_applied": _fit_control_summary(controls),
        "model_columns": [
            "voigt_model",
            "voigt_residual",
            "voigt_residual_sigma",
            "is_voigt_fit_pixel",
            "voigt_component_index",
            "voigt_transition_id",
            "transition_velocity_kms",
        ],
    }


def _failed_summary(reason: str) -> dict[str, Any]:
    return {
        "fit_type": "transition_frame_peak_voigt",
        "fit_method": "ultranest_physical_transition_frame",
        "success": False,
        "status": "failed",
        "reason": reason,
        "post_window_metadata_used": True,
        "metadata_use": "line_id and z_sys define transition frames only",
        "transition_frames": [],
    }


def _sibling_transition_intervals(
    transition: dict[str, Any],
    transitions: list[dict[str, Any]],
    *,
    frame_half_width_kms: float,
    exclusion_half_width_kms: float,
) -> list[dict[str, Any]]:
    current_id = str(transition["transition_line_id"])
    current_rest = float(transition["rest_wavelength_A"])
    intervals: list[dict[str, Any]] = []
    for other in transitions:
        other_id = str(other["transition_line_id"])
        if other_id == current_id:
            continue
        other_rest = float(other["rest_wavelength_A"])
        projected_velocity = C_KMS * (other_rest / current_rest - 1.0)
        if abs(projected_velocity) > frame_half_width_kms + exclusion_half_width_kms:
            continue
        intervals.append(
            {
                "transition_line_id": current_id,
                "start_velocity_kms": float(projected_velocity - exclusion_half_width_kms),
                "end_velocity_kms": float(projected_velocity + exclusion_half_width_kms),
                "mask_kind": "exclude",
                "reason": f"projected sibling transition {other_id}",
                "mask_source": "sibling_transition_projection",
                "sibling_transition_line_id": other_id,
            }
        )
    return intervals


def _apply_fit_mask_intervals(
    velocity: np.ndarray,
    good: np.ndarray,
    transition_line_id: str,
    intervals: list[dict[str, Any]],
) -> np.ndarray:
    updated = good.copy()
    finite_velocity = np.isfinite(velocity)
    for interval in intervals:
        interval_transition = str(interval.get("transition_line_id", transition_line_id))
        if interval_transition and interval_transition != transition_line_id:
            continue
        start = float(interval.get("start_velocity_kms", np.nan))
        end = float(interval.get("end_velocity_kms", np.nan))
        if not np.isfinite(start) or not np.isfinite(end):
            continue
        lower, upper = sorted((start, end))
        selected = finite_velocity & (velocity >= lower) & (velocity <= upper)
        if str(interval.get("mask_kind", "exclude")) == "include":
            updated[selected] = True
        else:
            updated[selected] = False
    return updated


def _merge_control_source_seeds(
    seeds: list[dict[str, Any]],
    *,
    transition_line_id: str,
    controls: dict[str, Any],
    min_peak_depth: float,
    min_peak_separation_kms: float,
) -> list[dict[str, Any]]:
    merged = list(seeds)
    for source in controls.get("source_seeds", []):
        if str(source.get("transition_line_id", transition_line_id)) != transition_line_id:
            continue
        center = float(source.get("center_velocity_kms", np.nan))
        if not np.isfinite(center):
            continue
        explicit = bool(source.get("explicit_fit_control_source"))
        replace_center = float(source.get("replace_center_velocity_kms", np.nan))
        replace_tolerance = float(source.get("replace_tolerance_kms", 45.0))
        if np.isfinite(replace_center):
            merged = [
                seed
                for seed in merged
                if abs(float(seed.get("seed_velocity_kms", np.inf)) - replace_center) > replace_tolerance
            ]
        if not explicit and any(abs(float(seed.get("seed_velocity_kms", np.inf)) - center) < 0.35 * min_peak_separation_kms for seed in merged):
            continue
        depth = float(np.clip(source.get("depth_below_continuum", min_peak_depth), min_peak_depth, 0.98))
        seed = {
            "seed_velocity_kms": center,
            "depth_below_continuum": depth,
            "prominence": depth,
            "score": float(depth + (2.0 if explicit else 0.0)),
            "seed_source": source.get("seed_source", "fit_control"),
            "explicit_fit_control_source": explicit,
            "control_reason": source.get("reason", ""),
        }
        if "replace_component_index" in source:
            seed["replace_component_index"] = int(source["replace_component_index"])
        if np.isfinite(replace_center):
            seed["replace_center_velocity_kms"] = float(replace_center)
        for key in (
            "logN",
            "logN_lower",
            "logN_upper",
            "b_kms",
            "b_kms_lower",
            "b_kms_upper",
            "center_prior_sigma_kms",
            "center_prior_half_width_kms",
        ):
            if key in source:
                seed[key] = float(source[key])
        merged.append(seed)
    merged.sort(key=lambda item: float(item.get("score", 0.0)), reverse=True)
    return merged


def _remove_suppressed_source_seeds(
    seeds: list[dict[str, Any]],
    *,
    transition_line_id: str,
    controls: dict[str, Any],
) -> list[dict[str, Any]]:
    removed_sources = [
        source
        for source in controls.get("removed_sources", [])
        if str(source.get("transition_line_id", transition_line_id)) == transition_line_id
    ]
    if not removed_sources:
        return seeds

    remove_indices: set[int] = set()
    for source in removed_sources:
        center = float(source.get("center_velocity_kms", np.nan))
        tolerance = float(source.get("center_tolerance_kms", 45.0))
        if not np.isfinite(center) or not seeds:
            continue
        distances = np.asarray([abs(float(seed.get("seed_velocity_kms", np.nan)) - center) for seed in seeds], dtype=float)
        finite = np.isfinite(distances)
        if not finite.any():
            continue
        best_index = int(np.nanargmin(np.where(finite, distances, np.inf)))
        if float(distances[best_index]) <= tolerance:
            remove_indices.add(best_index)

    return [seed for index, seed in enumerate(seeds) if index not in remove_indices]


def _fit_control_summary(controls: dict[str, Any]) -> dict[str, Any]:
    if not controls:
        return {"applied": False}
    return {
        "applied": True,
        "n_source_seeds": int(len(controls.get("source_seeds", []))),
        "n_removed_sources": int(len(controls.get("removed_sources", []))),
        "n_fit_mask_intervals": int(len(controls.get("fit_mask_intervals", []))),
        "n_fit_windows": int(len(controls.get("fit_windows", {}))),
        "fit_mask_intervals": controls.get("fit_mask_intervals", []),
        "continuum_anchor_remove_wavelengths_A": controls.get("continuum_anchor_remove_wavelengths_A", []),
        "continuum_mask_intervals_A": controls.get("continuum_mask_intervals_A", []),
    }


def _frame_residual_samples(
    wavelength: np.ndarray,
    velocity: np.ndarray,
    flux: np.ndarray,
    ivar: np.ndarray,
    model: np.ndarray,
    fit_mask: np.ndarray,
) -> list[dict[str, float]]:
    with np.errstate(divide="ignore", invalid="ignore"):
        err = np.where(ivar > 0.0, 1.0 / np.sqrt(ivar), np.nan)
        residual = flux - model
        residual_sigma = residual / err
    selected = fit_mask & np.isfinite(wavelength) & np.isfinite(velocity) & np.isfinite(model) & np.isfinite(residual_sigma)
    samples: list[dict[str, float]] = []
    for wave_A, vel_kms, model_flux, resid, resid_sigma in zip(
        wavelength[selected],
        velocity[selected],
        model[selected],
        residual[selected],
        residual_sigma[selected],
        strict=True,
    ):
        samples.append(
            {
                "wavelength_A": float(wave_A),
                "velocity_kms": float(vel_kms),
                "model_flux": float(model_flux),
                "residual": float(resid),
                "residual_sigma": float(resid_sigma),
            }
        )
    return samples


def _frame_diagnostic_residual_samples(
    wavelength: np.ndarray,
    velocity: np.ndarray,
    flux: np.ndarray,
    ivar: np.ndarray,
    model: np.ndarray,
    good_mask: np.ndarray,
    fit_mask: np.ndarray,
) -> list[dict[str, float]]:
    samples = _frame_residual_samples(wavelength, velocity, flux, ivar, model, good_mask)
    fit_mask = np.asarray(fit_mask, dtype=bool)
    selected_indices = np.flatnonzero(
        np.asarray(good_mask, dtype=bool)
        & np.isfinite(wavelength)
        & np.isfinite(velocity)
        & np.isfinite(model)
    )
    for sample, index in zip(samples, selected_indices, strict=True):
        sample["used_in_fit"] = bool(fit_mask[index])
    return samples


def _append_frame_residuals(
    residual_samples: list[dict[str, float]],
    frame_fit_residuals: list[np.ndarray],
    frame_work_residuals: list[np.ndarray],
) -> None:
    if not residual_samples:
        return
    sample_residuals = np.asarray([sample["residual_sigma"] for sample in residual_samples], dtype=float)
    sample_velocities = np.asarray([sample["velocity_kms"] for sample in residual_samples], dtype=float)
    finite_sample_residuals = sample_residuals[np.isfinite(sample_residuals)]
    if not len(finite_sample_residuals):
        return
    frame_fit_residuals.append(finite_sample_residuals)
    in_work = (
        np.isfinite(sample_velocities)
        & np.isfinite(sample_residuals)
        & (sample_velocities >= SOURCE_WORK_WINDOW_KMS[0])
        & (sample_velocities <= SOURCE_WORK_WINDOW_KMS[1])
    )
    if in_work.any():
        frame_work_residuals.append(sample_residuals[in_work])


def _source_work_window_metrics_from_residuals(frame_work_residuals: list[np.ndarray]) -> dict[str, Any]:
    if not frame_work_residuals:
        return {
            "n_fit_pixels": 0,
            "chi2": float("nan"),
            "reduced_chi2": float("nan"),
            "fit_rms": float("nan"),
            "max_abs_residual_sigma": float("nan"),
        }
    values = np.concatenate(frame_work_residuals)
    values = values[np.isfinite(values)]
    if len(values) == 0:
        return {
            "n_fit_pixels": 0,
            "chi2": float("nan"),
            "reduced_chi2": float("nan"),
            "fit_rms": float("nan"),
            "max_abs_residual_sigma": float("nan"),
        }
    chi2 = float(np.sum(np.square(values)))
    reduced_chi2 = float(chi2 / len(values))
    return {
        "n_fit_pixels": int(len(values)),
        "chi2": chi2,
        "reduced_chi2": reduced_chi2,
        "fit_rms": float(np.sqrt(reduced_chi2)),
        "max_abs_residual_sigma": float(np.nanmax(np.abs(values))),
    }


def _residual_metrics_from_samples(samples: Any) -> dict[str, Any]:
    residuals: list[float] = []
    for sample in samples:
        if not isinstance(sample, dict):
            continue
        try:
            residual_sigma = float(sample.get("residual_sigma"))
        except (TypeError, ValueError):
            continue
        if np.isfinite(residual_sigma):
            residuals.append(residual_sigma)
    values = np.asarray(residuals, dtype=float)
    if len(values) == 0:
        return {
            "n_fit_pixels": 0,
            "chi2": float("nan"),
            "reduced_chi2": float("nan"),
            "fit_rms": float("nan"),
            "max_abs_residual_sigma": float("nan"),
        }
    chi2 = float(np.sum(np.square(values)))
    reduced_chi2 = float(chi2 / len(values))
    return {
        "n_fit_pixels": int(len(values)),
        "chi2": chi2,
        "reduced_chi2": reduced_chi2,
        "fit_rms": float(np.sqrt(reduced_chi2)),
        "max_abs_residual_sigma": float(np.nanmax(np.abs(values))),
    }


def _source_work_window_metrics(output: pd.DataFrame) -> dict[str, Any]:
    if "transition_velocity_kms" not in output.columns or "voigt_residual_sigma" not in output.columns:
        return {
            "n_fit_pixels": 0,
            "chi2": float("nan"),
            "reduced_chi2": float("nan"),
            "fit_rms": float("nan"),
            "max_abs_residual_sigma": float("nan"),
        }
    velocity = output["transition_velocity_kms"].to_numpy(dtype=float)
    residual = output["voigt_residual_sigma"].to_numpy(dtype=float)
    fit_pixel = output["is_voigt_fit_pixel"].to_numpy(dtype=bool) if "is_voigt_fit_pixel" in output.columns else np.isfinite(residual)
    selected = (
        fit_pixel
        & np.isfinite(velocity)
        & np.isfinite(residual)
        & (velocity >= SOURCE_WORK_WINDOW_KMS[0])
        & (velocity <= SOURCE_WORK_WINDOW_KMS[1])
    )
    if not selected.any():
        return {
            "n_fit_pixels": 0,
            "chi2": float("nan"),
            "reduced_chi2": float("nan"),
            "fit_rms": float("nan"),
            "max_abs_residual_sigma": float("nan"),
        }
    values = residual[selected]
    chi2 = float(np.sum(np.square(values)))
    reduced_chi2 = float(chi2 / len(values))
    return {
        "n_fit_pixels": int(len(values)),
        "chi2": chi2,
        "reduced_chi2": reduced_chi2,
        "fit_rms": float(np.sqrt(reduced_chi2)),
        "max_abs_residual_sigma": float(np.nanmax(np.abs(values))),
    }
