from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.signal import find_peaks

from astroagent.spectra.line_catalog import load_line_catalog, primary_rest_wavelength_A, rest_wavelengths_A, transition_definitions
from astroagent.agent.policy import SOURCE_CORE_WINDOW_KMS, SOURCE_WORK_WINDOW_KMS
from astroagent.review.continuum import C_KMS, build_plot_data, observed_wavelength_A, validate_spectrum_table, velocity_kms
from astroagent.review.plot import build_smooth_voigt_model_data, save_review_plot, save_window_overview_plot
from astroagent.spectra.voigt_fit import fit_voigt_absorption


def cut_local_window(
    spectrum: pd.DataFrame,
    line_id: str,
    z_sys: float,
    half_width_kms: float = 1500.0,
    catalog_path: str | Path | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    validate_spectrum_table(spectrum)
    catalog = load_line_catalog(catalog_path)
    transitions = transition_definitions(line_id, catalog)
    rests = [float(transition["rest_wavelength_A"]) for transition in transitions]
    centers = [observed_wavelength_A(rest, z_sys) for rest in rests]
    primary_center = observed_wavelength_A(primary_rest_wavelength_A(line_id, catalog), z_sys)

    lower_edges = [center * (1.0 - half_width_kms / C_KMS) for center in centers]
    upper_edges = [center * (1.0 + half_width_kms / C_KMS) for center in centers]
    window_min = min(lower_edges)
    window_max = max(upper_edges)

    mask = (spectrum["wavelength"] >= window_min) & (spectrum["wavelength"] <= window_max)
    window = spectrum.loc[mask].copy()
    if window.empty:
        raise ValueError(
            f"no pixels found for {line_id} at z_sys={z_sys}; "
            f"window={window_min:.3f}-{window_max:.3f} A"
        )

    window_reference_A = 0.5 * (window_min + window_max)
    window["velocity_kms"] = velocity_kms(window["wavelength"].to_numpy(), window_reference_A)
    metadata = {
        "line_id": line_id,
        "z_sys": float(z_sys),
        "rest_wavelengths_A": rests,
        "observed_centers_A": centers,
        "transitions": [
            {
                **transition,
                "observed_center_A": observed_wavelength_A(float(transition["rest_wavelength_A"]), z_sys),
            }
            for transition in transitions
        ],
        "primary_observed_center_A": primary_center,
        "window_reference_A": float(window_reference_A),
        "window_min_A": float(window_min),
        "window_max_A": float(window_max),
        "half_width_kms": float(half_width_kms),
        "catalog_path": str(Path(catalog_path)) if catalog_path is not None else None,
        "post_window_rule": "line metadata defines the coarse observed window and each transition velocity frame only",
    }
    return window.reset_index(drop=True), metadata


def summarize_window(window: pd.DataFrame) -> dict[str, Any]:
    validate_spectrum_table(window)
    good = (window["ivar"] > 0) & (window["pipeline_mask"] == 0)
    snr = np.abs(window.loc[good, "flux"].to_numpy()) * np.sqrt(window.loc[good, "ivar"].to_numpy())
    return {
        "n_pixels": int(len(window)),
        "n_good_pixels": int(good.sum()),
        "bad_pixel_fraction": float(1.0 - good.mean()),
        "wavelength_min_A": float(window["wavelength"].min()),
        "wavelength_max_A": float(window["wavelength"].max()),
        "flux_median": float(window["flux"].median()),
        "flux_p10": float(window["flux"].quantile(0.10)),
        "flux_p90": float(window["flux"].quantile(0.90)),
        "snr_median": float(np.median(snr)) if len(snr) else 0.0,
    }


def assess_absorber_hypothesis(window: pd.DataFrame, window_metadata: dict[str, Any] | None = None) -> dict[str, Any]:
    validate_spectrum_table(window)
    flux = window["flux"].to_numpy()
    ivar = window["ivar"].to_numpy()
    pipeline_mask = window["pipeline_mask"].to_numpy()
    good = np.isfinite(flux) & np.isfinite(ivar) & (ivar > 0) & (pipeline_mask == 0)
    continuum_level = float(np.median(flux[good])) if good.any() else float(np.median(flux))
    noise_level = float(np.median(1.0 / np.sqrt(ivar[good]))) if good.any() else 0.0
    depth_threshold = max(0.05 * max(abs(continuum_level), 1e-6), 3.0 * noise_level)

    peak_records: list[dict[str, float]] = []
    if good.sum() >= 5:
        wavelength = window["wavelength"].to_numpy(dtype=float)
        wave_good = wavelength[good]
        depth = continuum_level - flux[good]
        peaks, properties = find_peaks(depth, height=depth_threshold, distance=3, prominence=0.5 * depth_threshold)
        peak_heights = properties.get("peak_heights", depth[peaks] if len(peaks) else np.array([], dtype=float))
        peak_records = [
            {
                "wavelength_A": float(wave_good[peak]),
                "depth_below_window_median": float(height),
            }
            for peak, height in zip(peaks, peak_heights, strict=True)
        ]

    detected_count = len(peak_records)
    status = "plausible" if detected_count >= 1 else "weak_or_absent"
    reason = (
        "coarse window contains data-driven absorption-like troughs"
        if detected_count >= 1
        else "coarse window has no absorption-like trough above the data-driven threshold"
    )
    return {
        "status": status,
        "absorption_signal_present": status == "plausible",
        "detected_absorption_peaks": int(detected_count),
        "continuum_level": continuum_level,
        "noise_level": noise_level,
        "depth_threshold": float(depth_threshold),
        "peak_checks": peak_records,
        "rationale": reason,
        "method": "coarse_window_data_driven_absorption_check",
        "post_window_metadata_used": False,
    }


def human_adjudication_policy() -> dict[str, Any]:
    return {
        "required": True,
        "agent_role": "organize_evidence_not_final_science_judge",
        "human_decision_fields": [
            "final_system_acceptance",
            "physical_interpretation",
            "outflow_or_intervening_class",
            "cross_element_consistency",
        ],
        "escalation_flags": [
            "velocity_exceeds_virial_expectation",
            "cross_element_inconsistency",
            "single_line_only",
            "blend_or_contamination",
            "ambiguous_redshift_solution",
            "low_snr_or_bad_pixels",
        ],
        "fit_result_policy": "preserve_all_component_measurements_for_review",
    }


def _contiguous_intervals(wavelength: np.ndarray, selected: np.ndarray) -> list[list[float]]:
    if len(wavelength) == 0 or not selected.any():
        return []
    selected_indices = np.flatnonzero(selected)
    groups = np.split(selected_indices, np.where(np.diff(selected_indices) != 1)[0] + 1)
    return [[float(wavelength[group[0]]), float(wavelength[group[-1]])] for group in groups]


def build_plot_and_fit_data(
    window: pd.DataFrame,
    window_metadata: dict[str, Any] | None = None,
    fit_control: dict[str, Any] | None = None,
) -> tuple[pd.DataFrame, dict[str, Any], dict[str, Any]]:
    plot_window, plot_summary = build_plot_data(window, window_metadata, fit_control=fit_control)
    fit_window, fit_summary = fit_voigt_absorption(plot_window, window_metadata, fit_control=fit_control)
    return fit_window, plot_summary, fit_summary


def suggest_task_a_labels(window: pd.DataFrame) -> dict[str, Any]:
    validate_spectrum_table(window)
    wavelength = window["wavelength"].to_numpy()
    flux = window["flux"].to_numpy()
    ivar = window["ivar"].to_numpy()
    pipeline_mask = window["pipeline_mask"].to_numpy()

    finite = np.isfinite(flux) & np.isfinite(ivar)
    good = finite & (ivar > 0) & (pipeline_mask == 0)
    median = float(np.median(flux[good])) if good.any() else float(np.median(flux[finite]))
    scatter = float(1.4826 * np.median(np.abs(flux[good] - median))) if good.any() else 0.0
    scatter = max(scatter, 1e-6)

    bad_pixels = (~good) | (flux < median - 2.5 * scatter)
    intervals = _contiguous_intervals(wavelength, bad_pixels)

    if good.any():
        anchor_threshold = np.quantile(flux[good], 0.55)
        anchor_pool = good & (~bad_pixels) & (flux >= anchor_threshold)
    else:
        anchor_pool = np.zeros_like(good, dtype=bool)
    anchor_wavelengths = wavelength[anchor_pool]
    if len(anchor_wavelengths) >= 4:
        positions = np.linspace(0, len(anchor_wavelengths) - 1, 4).round().astype(int)
        anchors = [float(anchor_wavelengths[index]) for index in positions]
    else:
        fallback = wavelength[good] if good.any() else wavelength
        positions = np.linspace(0, len(fallback) - 1, min(4, len(fallback))).round().astype(int)
        anchors = [float(fallback[index]) for index in positions]

    bad_fraction = float(bad_pixels.mean())
    quality = "low" if bad_fraction > 0.35 or len(anchors) < 3 else "medium"
    return {
        "task": "local_mask_continuum",
        "window_action": "keep",
        "analysis_mask_intervals_A": intervals,
        "continuum_anchor_points_A": anchors,
        "quality": quality,
        "rationale": "Rule baseline: mask flagged pixels and unusually low flux, then anchor continuum on high good-flux pixels.",
    }


def build_review_record(
    spectrum: pd.DataFrame,
    line_id: str,
    z_sys: float,
    sample_id: str,
    source: dict[str, Any],
    half_width_kms: float = 1500.0,
    catalog_path: str | Path | None = None,
) -> tuple[dict[str, Any], pd.DataFrame]:
    window, window_metadata = cut_local_window(
        spectrum=spectrum,
        line_id=line_id,
        z_sys=z_sys,
        half_width_kms=half_width_kms,
        catalog_path=catalog_path,
    )
    record = build_review_record_from_window(
        window=window,
        window_metadata=window_metadata,
        sample_id=sample_id,
        source=source,
    )
    return record, window


def build_review_record_from_window(
    window: pd.DataFrame,
    window_metadata: dict[str, Any],
    sample_id: str,
    source: dict[str, Any],
    fit_control: dict[str, Any] | None = None,
) -> dict[str, Any]:
    plot_window, plot_summary, fit_summary = build_plot_and_fit_data(window, window_metadata, fit_control=fit_control)
    record = {
        "sample_id": sample_id,
        "source": source,
        "input": window_metadata,
        "window_summary": summarize_window(window),
        "absorber_hypothesis_check": assess_absorber_hypothesis(window, window_metadata),
        "human_adjudication": human_adjudication_policy(),
        "fit_results": [{"task": "voigt_profile_fit", **fit_summary}],
        "plot_data": _plot_data_record(sample_id, plot_summary, fit_summary),
        "fit_control_hints": build_fit_control_hints(plot_window, fit_summary),
        "task_a_rule_suggestion": suggest_task_a_labels(window),
        "human_review": {
            "status": "needs_review",
            "reviewer": "",
            "notes": "",
            "absorber_hypothesis_notes": "",
            "final_system_acceptance": None,
            "physical_interpretation": None,
            "cross_element_consistency": None,
            "accepted_task_a": None,
            "corrected_task_a": None,
        },
    }
    if fit_control:
        record["fit_control_application"] = {
            "applied": True,
            "summary": {
                "n_source_seeds": int(len(fit_control.get("source_seeds", []))),
                "n_removed_sources": int(len(fit_control.get("removed_sources", []))),
                "n_fit_mask_intervals": int(len(fit_control.get("fit_mask_intervals", []))),
                "n_fit_windows": int(len(fit_control.get("fit_windows", {}))),
                "n_continuum_anchors": int(len(fit_control.get("continuum_anchor_wavelengths_A", []))),
                "n_continuum_anchor_nodes": int(len(fit_control.get("continuum_anchor_nodes", []))),
                "n_continuum_mask_intervals": int(len(fit_control.get("continuum_mask_intervals_A", []))),
            },
        }
    record["_plot_window"] = plot_window
    record["_plot_summary"] = plot_summary
    record["_fit_summary"] = fit_summary
    return record


def build_fit_control_hints(plot_window: pd.DataFrame, fit_summary: dict[str, Any]) -> dict[str, Any]:
    """Build deterministic hints for the LLM tool loop from fit residuals."""
    hints: list[dict[str, Any]] = []

    transition_half_width = float(fit_summary.get("transition_half_width_kms", 600.0))
    overlap_lookup = _frame_overlap_lookup(fit_summary)
    for frame in fit_summary.get("transition_frames", []):
        transition_line_id = str(frame.get("transition_line_id", ""))
        if not transition_line_id:
            continue
        velocity, residual = _frame_residual_arrays(frame, plot_window, transition_line_id)
        if len(velocity) == 0:
            continue
        finite = np.isfinite(velocity) & np.isfinite(residual)
        strong_negative = finite & (residual <= -3.0)
        for group in _boolean_groups(strong_negative):
            depth = float(np.nanmin(residual[group])) if len(group) else float("nan")
            if len(group) < 2 and (not np.isfinite(depth) or depth > -5.0):
                continue
            lower_v = float(np.nanmin(velocity[group]))
            upper_v = float(np.nanmax(velocity[group]))
            center = float(np.nanmedian(velocity[group]))
            width = float(upper_v - lower_v)
            in_core_window = SOURCE_CORE_WINDOW_KMS[0] <= center <= SOURCE_CORE_WINDOW_KMS[1]
            touches_work_window = upper_v >= SOURCE_WORK_WINDOW_KMS[0] and lower_v <= SOURCE_WORK_WINDOW_KMS[1]
            in_boundary_band = touches_work_window and not in_core_window
            overlap_matches = _matching_velocity_overlap_intervals(transition_line_id, center, overlap_lookup)
            in_velocity_overlap = bool(overlap_matches)
            high_sigma_context = (not in_core_window) and depth <= -5.0
            kind = (
                "source_work_window_residual"
                if in_core_window
                else (
                    "source_boundary_residual"
                    if in_boundary_band
                    else ("high_sigma_context_mask_candidate" if high_sigma_context else "context_only_residual")
                )
            )
            suggested_action = "inspect wavelength overview first; only add/update a source if the feature is target absorption"
            preferred_action = "add_or_update_absorption_source" if in_core_window and not in_velocity_overlap else "set_fit_mask_interval_or_human_review"
            payload: dict[str, Any] = {
                "kind": kind,
                "transition_line_id": transition_line_id,
                "center_velocity_kms": center,
                "velocity_range_kms": [lower_v, upper_v],
                "width_kms": width,
                "min_residual_sigma": depth,
                "in_velocity_frame_overlap": in_velocity_overlap,
                "overlap_intervals": overlap_matches,
                "preferred_action": preferred_action,
                "suggested_action": suggested_action,
            }
            if in_boundary_band:
                pad = max(12.0, min(35.0, 0.5 * width + 8.0))
                payload.update(
                    {
                        "suggested_fit_mask_interval_kms": [center - pad, center + pad],
                        "preferred_action": "set_fit_mask_interval_or_human_review",
                        "suggested_action": (
                            "boundary band; inspect wavelength overview first. If the feature is projected/overlap/another-line "
                            "contamination, use a narrow set_fit_mask_interval; add/update a source only if it is visibly target absorption"
                        ),
                    }
                )
            elif not in_core_window:
                pad = max(12.0, min(35.0, 0.5 * width + 8.0))
                context_kind = "high_sigma_context_mask_candidate" if high_sigma_context else "context_only_mask_candidate"
                action = (
                    "high-sigma context residual; use a narrow set_fit_mask_interval exclude or explicitly explain why this red residual is acceptable"
                    if high_sigma_context
                    else "context only; consider a narrow set_fit_mask_interval exclude only if it contaminates the source work window"
                )
                payload.update(
                    {
                        "kind": context_kind,
                        "suggested_fit_mask_interval_kms": [center - pad, center + pad],
                        "preferred_action": "set_fit_mask_interval_or_human_review",
                        "suggested_action": action,
                    }
                )
            hints.append(payload)

    work_hints = [hint for hint in hints if hint["kind"] == "source_work_window_residual"]
    boundary_hints = [hint for hint in hints if hint["kind"] == "source_boundary_residual"]
    context_hints = [hint for hint in hints if hint["kind"] not in {"source_work_window_residual", "source_boundary_residual"}]
    work_hints.sort(key=lambda item: abs(float(item.get("min_residual_sigma", 0.0))), reverse=True)
    boundary_hints.sort(key=lambda item: abs(float(item.get("min_residual_sigma", 0.0))), reverse=True)
    context_hints.sort(key=lambda item: abs(float(item.get("min_residual_sigma", 0.0))), reverse=True)
    selected = [*work_hints[:8], *boundary_hints[:3], *context_hints[:3]]
    selected.sort(key=lambda item: float(item.get("center_velocity_kms", 0.0)))
    return {
        "source_work_window_kms": list(SOURCE_WORK_WINDOW_KMS),
        "source_core_window_kms": list(SOURCE_CORE_WINDOW_KMS),
        "priority_order": [
            "source_work_window_residual",
            "source_boundary_residual",
            "high_sigma_context_mask_candidate",
            "context_only_mask_candidate",
        ],
        "hints": selected,
    }


def _frame_residual_arrays(
    frame: dict[str, Any],
    plot_window: pd.DataFrame,
    transition_line_id: str,
) -> tuple[np.ndarray, np.ndarray]:
    samples = frame.get("residual_samples", [])
    if isinstance(samples, list) and samples:
        rows = [
            (float(sample.get("velocity_kms")), float(sample.get("residual_sigma")))
            for sample in samples
            if _finite_number(sample.get("velocity_kms")) and _finite_number(sample.get("residual_sigma"))
        ]
        rows.sort(key=lambda item: item[0])
        return (
            np.asarray([item[0] for item in rows], dtype=float),
            np.asarray([item[1] for item in rows], dtype=float),
        )
    if "transition_velocity_kms" not in plot_window.columns or "voigt_residual_sigma" not in plot_window.columns:
        return np.asarray([], dtype=float), np.asarray([], dtype=float)
    selected = (
        (plot_window["voigt_transition_id"].astype(str) == transition_line_id)
        & (plot_window.get("is_voigt_fit_pixel", 0).astype(bool))
    )
    if not selected.any():
        return np.asarray([], dtype=float), np.asarray([], dtype=float)
    frame_data = plot_window.loc[selected].sort_values("transition_velocity_kms")
    return (
        frame_data["transition_velocity_kms"].to_numpy(dtype=float),
        frame_data["voigt_residual_sigma"].to_numpy(dtype=float),
    )


def _finite_number(value: Any) -> bool:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return False
    return bool(np.isfinite(parsed))


def _frame_overlap_lookup(fit_summary: dict[str, Any]) -> dict[str, list[dict[str, Any]]]:
    frames: list[dict[str, Any]] = []
    for frame in fit_summary.get("transition_frames", []):
        transition_line_id = str(frame.get("transition_line_id", ""))
        observed_center = _finite_or_none(frame.get("observed_center_A"))
        bounds = frame.get("velocity_frame", {}).get("bounds_kms")
        if not transition_line_id or observed_center is None or not isinstance(bounds, list) or len(bounds) != 2:
            continue
        lower_v = _finite_or_none(bounds[0])
        upper_v = _finite_or_none(bounds[1])
        if lower_v is None or upper_v is None:
            continue
        lower_v, upper_v = sorted((lower_v, upper_v))
        frames.append(
            {
                "transition_line_id": transition_line_id,
                "observed_center_A": observed_center,
                "wavelength_interval_A": _velocity_interval_to_wavelength_A(observed_center, lower_v, upper_v),
            }
        )
    lookup: dict[str, list[dict[str, Any]]] = {}
    for index, left in enumerate(frames):
        for right in frames[index + 1 :]:
            lower_A = max(left["wavelength_interval_A"][0], right["wavelength_interval_A"][0])
            upper_A = min(left["wavelength_interval_A"][1], right["wavelength_interval_A"][1])
            if lower_A >= upper_A:
                continue
            for current, other in ((left, right), (right, left)):
                current_center = current["observed_center_A"]
                lookup.setdefault(current["transition_line_id"], []).append(
                    {
                        "other_transition_line_id": other["transition_line_id"],
                        "overlap_wavelength_A": [lower_A, upper_A],
                        "overlap_velocity_kms": [
                            _wavelength_to_velocity_kms(current_center, lower_A),
                            _wavelength_to_velocity_kms(current_center, upper_A),
                        ],
                    }
                )
    return lookup


def _matching_velocity_overlap_intervals(
    transition_line_id: str,
    center_velocity_kms: float,
    overlap_lookup: dict[str, list[dict[str, Any]]],
) -> list[dict[str, Any]]:
    matches: list[dict[str, Any]] = []
    for interval in overlap_lookup.get(transition_line_id, []):
        lower, upper = sorted(float(value) for value in interval.get("overlap_velocity_kms", []))
        if lower <= center_velocity_kms <= upper:
            matches.append(interval)
    return matches


def _velocity_interval_to_wavelength_A(observed_center_A: float, lower_velocity_kms: float, upper_velocity_kms: float) -> list[float]:
    return [
        observed_center_A * (1.0 + lower_velocity_kms / 299792.458),
        observed_center_A * (1.0 + upper_velocity_kms / 299792.458),
    ]


def _wavelength_to_velocity_kms(observed_center_A: float, wavelength_A: float) -> float:
    return 299792.458 * (wavelength_A / observed_center_A - 1.0)


def _finite_or_none(value: Any) -> float | None:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if np.isfinite(parsed) else None


def _boolean_groups(mask: np.ndarray) -> list[np.ndarray]:
    indices = np.flatnonzero(mask)
    if len(indices) == 0:
        return []
    return [group for group in np.split(indices, np.where(np.diff(indices) != 1)[0] + 1) if len(group)]


def _plot_data_record(sample_id: str, plot_summary: dict[str, Any], fit_summary: dict[str, Any]) -> dict[str, Any]:
    return {
        **plot_summary,
        "columns": [
            "wavelength",
            "flux",
            "ivar",
            "pipeline_mask",
            "velocity_kms",
            "continuum_model",
            "normalized_flux",
            "normalized_ivar",
            "is_good_pixel",
            "is_continuum_pixel",
            "is_continuum_excluded_pixel",
            "voigt_model",
            "voigt_lsf_model",
            "voigt_residual",
            "voigt_lsf_residual",
            "voigt_residual_sigma",
            "voigt_lsf_residual_sigma",
            "is_voigt_fit_pixel",
            "voigt_component_index",
            "voigt_transition_id",
            "transition_velocity_kms",
        ],
        "model_columns": [
            "curve_kind",
            "component_index",
            "transition_line_id",
            "wavelength",
            "transition_velocity_kms",
            "smooth_voigt_model",
            "smooth_lsf_model",
            "center_velocity_kms",
            "center_wavelength_A",
            "logN",
            "b_kms",
            "damping_gamma_kms",
        ],
        "image_space": "per_transition_velocity_frame",
        "image_file": f"{sample_id}.plot.png",
        "overview_image_space": "observed_wavelength_window",
        "overview_image_file": f"{sample_id}.overview.png",
        "model_file": f"{sample_id}.model.csv",
        "continuum_exclusion_intervals_A": plot_summary["continuum_exclusion_intervals_A"],
        "voigt_fit": fit_summary,
    }


def write_review_packet(record: dict[str, Any], window: pd.DataFrame, output_dir: str | Path) -> dict[str, Path]:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    json_path = output_path / f"{record['sample_id']}.review.json"
    csv_path = output_path / f"{record['sample_id']}.window.csv"
    plot_csv_path = output_path / f"{record['sample_id']}.plot.csv"
    model_csv_path = output_path / f"{record['sample_id']}.model.csv"
    plot_png_path = output_path / f"{record['sample_id']}.plot.png"
    overview_png_path = output_path / f"{record['sample_id']}.overview.png"
    readme_path = output_path / "README.md"

    plot_window = record.get("_plot_window")
    plot_summary = record.get("_plot_summary")
    fit_summary = record.get("_fit_summary")
    if plot_window is None or plot_summary is None or fit_summary is None:
        plot_window, plot_summary = build_plot_data(window, record.get("input"))
        fit_summary = _record_fit_summary(record)
        if fit_summary is None:
            plot_window, plot_summary, fit_summary = build_plot_and_fit_data(window, record.get("input"))
        else:
            plot_window = _apply_fit_summary_to_plot_window(plot_window, fit_summary)

    public_record = _public_review_record(record)
    with json_path.open("w", encoding="utf-8") as handle:
        json.dump(public_record, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    window.to_csv(csv_path, index=False)
    plot_window.to_csv(plot_csv_path, index=False)
    build_smooth_voigt_model_data(fit_summary).to_csv(model_csv_path, index=False)
    save_window_overview_plot(public_record, window, overview_png_path, plot_window=plot_window)
    save_review_plot(public_record, window, plot_png_path, plot_window=plot_window, plot_summary=plot_summary, fit_summary=fit_summary)
    readme_path.write_text(_review_packet_readme(), encoding="utf-8")
    return {
        "json": json_path,
        "csv": csv_path,
        "plot_csv": plot_csv_path,
        "model_csv": model_csv_path,
        "overview_png": overview_png_path,
        "plot_png": plot_png_path,
        "readme": readme_path,
    }


def _public_review_record(record: dict[str, Any]) -> dict[str, Any]:
    public = deepcopy(record)
    public.pop("_plot_window", None)
    public.pop("_plot_summary", None)
    public.pop("_fit_summary", None)
    return public


def _record_fit_summary(record: dict[str, Any]) -> dict[str, Any] | None:
    for result in record.get("fit_results", []):
        if result.get("task") == "voigt_profile_fit" or result.get("fit_type"):
            return result
    return None


def _apply_fit_summary_to_plot_window(plot_window: pd.DataFrame, fit_summary: dict[str, Any]) -> pd.DataFrame:
    """Attach model/residual columns from a stored public fit summary.

    This fallback is used when a review JSON has been reloaded from disk and no
    private `_plot_window` cache is available. It keeps rewritten packet images
    and model CSVs consistent with the public `fit_results` instead of silently
    rerunning a fresh default fit.
    """
    output = plot_window.copy()
    output["voigt_model"] = np.nan
    output["voigt_lsf_model"] = np.nan
    output["voigt_residual"] = np.nan
    output["voigt_lsf_residual"] = np.nan
    output["voigt_residual_sigma"] = np.nan
    output["voigt_lsf_residual_sigma"] = np.nan
    output["is_voigt_fit_pixel"] = 0
    output["voigt_component_index"] = -1
    output["voigt_transition_id"] = ""
    output["transition_velocity_kms"] = np.nan

    wavelength = output["wavelength"].to_numpy(dtype=float)
    normalized_flux = output["normalized_flux"].to_numpy(dtype=float)
    normalized_ivar = output["normalized_ivar"].to_numpy(dtype=float)
    model_data = build_smooth_voigt_model_data(fit_summary)
    if model_data.empty:
        return output

    for frame in fit_summary.get("transition_frames", []):
        transition_line_id = str(frame.get("transition_line_id", ""))
        observed_center_A = float(frame.get("observed_center_A", np.nan))
        if not transition_line_id or not np.isfinite(observed_center_A):
            continue
        bounds = frame.get("velocity_frame", {}).get("bounds_kms", [-np.inf, np.inf])
        try:
            lower_bound, upper_bound = float(bounds[0]), float(bounds[1])
        except (TypeError, ValueError, IndexError):
            lower_bound, upper_bound = -np.inf, np.inf
        velocity = velocity_kms(wavelength, observed_center_A)
        frame_mask = np.isfinite(velocity) & (velocity >= lower_bound) & (velocity <= upper_bound)
        combined_rows = model_data[
            (model_data["transition_line_id"] == transition_line_id)
            & (model_data["curve_kind"] == "combined")
        ].sort_values("transition_velocity_kms")
        if combined_rows.empty:
            continue
        model = np.interp(
            velocity[frame_mask],
            combined_rows["transition_velocity_kms"].to_numpy(dtype=float),
            combined_rows["smooth_voigt_model"].to_numpy(dtype=float),
            left=np.nan,
            right=np.nan,
        )
        lsf_model = (
            np.interp(
                velocity[frame_mask],
                combined_rows["transition_velocity_kms"].to_numpy(dtype=float),
                combined_rows["smooth_lsf_model"].to_numpy(dtype=float),
                left=np.nan,
                right=np.nan,
            )
            if "smooth_lsf_model" in combined_rows.columns
            else np.full_like(model, np.nan, dtype=float)
        )
        frame_indices = np.flatnonzero(frame_mask)
        finite_model = np.isfinite(model)
        target_indices = frame_indices[finite_model]
        existing = output.loc[target_indices, "voigt_model"].to_numpy(dtype=float)
        updated_model = model[finite_model]
        output.loc[target_indices, "voigt_model"] = np.where(
            np.isfinite(existing),
            np.minimum(existing, updated_model),
            updated_model,
        )
        finite_lsf_model = np.isfinite(lsf_model)
        lsf_target_indices = frame_indices[finite_lsf_model]
        existing_lsf = output.loc[lsf_target_indices, "voigt_lsf_model"].to_numpy(dtype=float)
        updated_lsf_model = lsf_model[finite_lsf_model]
        output.loc[lsf_target_indices, "voigt_lsf_model"] = np.where(
            np.isfinite(existing_lsf),
            np.minimum(existing_lsf, updated_lsf_model),
            updated_lsf_model,
        )
        output.loc[frame_indices, "voigt_transition_id"] = transition_line_id
        output.loc[frame_indices, "transition_velocity_kms"] = velocity[frame_mask]

    finite = np.isfinite(output["voigt_model"].to_numpy(dtype=float))
    output.loc[finite, "is_voigt_fit_pixel"] = 1
    output.loc[finite, "voigt_residual"] = normalized_flux[finite] - output.loc[finite, "voigt_model"].to_numpy(dtype=float)
    finite_lsf = np.isfinite(output["voigt_lsf_model"].to_numpy(dtype=float))
    output.loc[finite_lsf, "voigt_lsf_residual"] = normalized_flux[finite_lsf] - output.loc[finite_lsf, "voigt_lsf_model"].to_numpy(dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        err = np.where(normalized_ivar > 0.0, 1.0 / np.sqrt(normalized_ivar), np.nan)
    output.loc[finite, "voigt_residual_sigma"] = output.loc[finite, "voigt_residual"].to_numpy(dtype=float) / err[finite]
    output.loc[finite_lsf, "voigt_lsf_residual_sigma"] = output.loc[finite_lsf, "voigt_lsf_residual"].to_numpy(dtype=float) / err[finite_lsf]

    for component in fit_summary.get("components", []):
        transition_line_id = str(component.get("transition_line_id", ""))
        center_velocity = float(component.get("center_velocity_kms", np.nan))
        half_width = float(component.get("fit_window_half_width_kms", fit_summary.get("local_fit_half_width_kms", 180.0)))
        if not transition_line_id or not np.isfinite(center_velocity) or not np.isfinite(half_width):
            continue
        selected = (
            (output["voigt_transition_id"].astype(str) == transition_line_id)
            & np.isfinite(output["transition_velocity_kms"].to_numpy(dtype=float))
            & (np.abs(output["transition_velocity_kms"].to_numpy(dtype=float) - center_velocity) <= half_width)
        )
        output.loc[selected, "voigt_component_index"] = int(component.get("component_index", -1))
    return output


def _review_packet_readme() -> str:
    return "\n".join(
        [
            "# 人工审查包",
            "",
            "这个目录里的文件是脚本生成的中间产物。",
            "",
            "- `*.review.json`：一条结构化样本，留给人工审查和修正。",
            "- `*.window.csv`：局域观测波长谱窗。",
            "- `*.plot.csv`：画图专用数据，包含连续谱拟合和归一化谱。",
            "- `*.model.csv`：速度空间平滑 Voigt 曲线，`component` 是单峰，`combined` 是同一 transition 内的总模型；有 LSF 时会保留 LSF 诊断列。",
            "- `*.overview.png`：观测波长空间的局域窗口总览图，显示 raw flux、continuum、normalized flux 和 transition centers。",
            "- `*.plot.png`：每条 transition 一个局部速度坐标子图。",
            "",
            "先看 `absorber_hypothesis_check`、`fit_results` 和 rule suggestion，再填写 `human_review.notes`、",
            "`human_review.accepted_task_a` 或 `human_review.corrected_task_a`。",
            "",
        ]
    )


def make_demo_quasar_spectrum(z_sys: float = 2.6, line_id: str = "CIV_doublet") -> pd.DataFrame:
    catalog = load_line_catalog()
    centers = [observed_wavelength_A(rest, z_sys) for rest in rest_wavelengths_A(line_id, catalog)]
    wavelength = np.arange(min(centers) - 35.0, max(centers) + 35.0, 0.25)
    continuum = 1.0 + 0.015 * (wavelength - wavelength.mean()) / (wavelength.max() - wavelength.min())
    flux = continuum.copy()

    for index, center in enumerate(centers):
        depth = 0.42 if index == 0 else 0.24
        sigma_A = 0.9 if index == 0 else 1.1
        flux -= depth * np.exp(-0.5 * ((wavelength - center) / sigma_A) ** 2)

    sky_residual = (wavelength > centers[0] - 5.5) & (wavelength < centers[0] - 4.5)
    flux[sky_residual] += 0.18
    ivar = np.full_like(wavelength, 400.0)
    pipeline_mask = np.zeros_like(wavelength, dtype=int)
    pipeline_mask[sky_residual] = 1

    return pd.DataFrame(
        {
            "wavelength": wavelength,
            "flux": flux,
            "ivar": ivar,
            "pipeline_mask": pipeline_mask,
        }
    )
