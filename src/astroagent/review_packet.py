from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.signal import find_peaks

from astroagent.line_catalog import load_line_catalog, primary_rest_wavelength_A, rest_wavelengths_A, transition_definitions
from astroagent.review_continuum import C_KMS, build_plot_data, observed_wavelength_A, validate_spectrum_table, velocity_kms
from astroagent.review_plot import build_smooth_voigt_model_data, save_review_plot
from astroagent.voigt_fit import fit_voigt_absorption


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
            "voigt_residual",
            "voigt_residual_sigma",
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
            "center_velocity_kms",
            "center_wavelength_A",
            "logN",
            "b_kms",
            "damping_gamma_kms",
        ],
        "image_space": "per_transition_velocity_frame",
        "image_file": f"{sample_id}.plot.png",
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
    readme_path = output_path / "README.md"

    plot_window = record.pop("_plot_window", None)
    plot_summary = record.pop("_plot_summary", None)
    fit_summary = record.pop("_fit_summary", None)
    if plot_window is None or plot_summary is None or fit_summary is None:
        plot_window, plot_summary, fit_summary = build_plot_and_fit_data(window, record.get("input"))

    with json_path.open("w", encoding="utf-8") as handle:
        json.dump(record, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    window.to_csv(csv_path, index=False)
    plot_window.to_csv(plot_csv_path, index=False)
    build_smooth_voigt_model_data(fit_summary).to_csv(model_csv_path, index=False)
    save_review_plot(record, window, plot_png_path, plot_window=plot_window, plot_summary=plot_summary, fit_summary=fit_summary)
    readme_path.write_text(_review_packet_readme(), encoding="utf-8")
    return {
        "json": json_path,
        "csv": csv_path,
        "plot_csv": plot_csv_path,
        "model_csv": model_csv_path,
        "plot_png": plot_png_path,
        "readme": readme_path,
    }


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
            "- `*.model.csv`：速度空间平滑 Voigt 曲线，`component` 是单峰，`combined` 是同一 transition 内的总模型。",
            "- `*.plot.png`：每条 transition 一个局部速度坐标子图；不再输出波长空间总览图。",
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
