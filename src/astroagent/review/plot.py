from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from astroagent.agent.policy import SOURCE_WORK_WINDOW_KMS
from astroagent.review.continuum import C_KMS, _contiguous_intervals, velocity_kms
from astroagent.spectra.voigt_model import physical_component_flux_model


def _configure_matplotlib_cache() -> None:
    cache_dir = Path(tempfile.gettempdir()) / "astroagent-matplotlib"
    cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(cache_dir))


def _smooth_voigt_curve(
    velocity_kms_values: np.ndarray,
    center_velocity_kms: float,
    logN: float,
    b_kms: float,
    rest_wavelength_A: float,
    oscillator_strength: float,
    damping_gamma_kms: float,
) -> np.ndarray:
    return physical_component_flux_model(
        velocity_kms_values,
        logN=float(logN),
        b_kms=float(b_kms),
        center_kms=float(center_velocity_kms),
        rest_wavelength_A=float(rest_wavelength_A),
        oscillator_strength=float(oscillator_strength),
        damping_gamma_kms=float(damping_gamma_kms),
    )


def build_smooth_voigt_model_data(
    fit_summary: dict[str, Any],
    points_per_peak: int = 240,
    half_width_factor: float = 4.5,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    component_curves: dict[str, list[tuple[int, np.ndarray, np.ndarray]]] = {}
    for component in fit_summary.get("components", []):
        if not component.get("fit_success"):
            continue
        required = [
            "transition_line_id",
            "center_velocity_kms",
            "center_wavelength_A",
            "logN",
            "b_kms",
            "damping_gamma_kms",
        ]
        numeric_required = [key for key in required if key != "transition_line_id"]
        if any(key not in component for key in required):
            continue
        if any(not np.isfinite(float(component[key])) for key in numeric_required):
            continue

        center_velocity = float(component["center_velocity_kms"])
        center_wavelength = float(component["center_wavelength_A"])
        logN = float(component["logN"])
        b_kms = float(component["b_kms"])
        gamma = float(component["damping_gamma_kms"])
        width = max(80.0, half_width_factor * max(b_kms, 8.0))
        velocity = np.linspace(center_velocity - width, center_velocity + width, int(points_per_peak))
        rest_A = float(component.get("rest_wavelength_A", np.nan))
        oscillator_strength = float(component.get("oscillator_strength", np.nan))
        if not np.isfinite(rest_A) or not np.isfinite(oscillator_strength):
            continue
        model = _smooth_voigt_curve(
            velocity,
            center_velocity_kms=center_velocity,
            logN=logN,
            b_kms=b_kms,
            rest_wavelength_A=rest_A,
            oscillator_strength=oscillator_strength,
            damping_gamma_kms=gamma,
        )

        observed_center_A = center_wavelength / (1.0 + center_velocity / C_KMS)
        wavelength = observed_center_A * (1.0 + velocity / C_KMS)
        transition_line_id = str(component["transition_line_id"])
        component_index = int(component.get("component_index", -1))
        component_curves.setdefault(transition_line_id, []).append((component_index, velocity, model))
        for wave_A, vel_kms, model_flux in zip(wavelength, velocity, model, strict=True):
            rows.append(
                {
                    "curve_kind": "component",
                    "component_index": component_index,
                    "transition_line_id": transition_line_id,
                    "wavelength": float(wave_A),
                    "transition_velocity_kms": float(vel_kms),
                    "smooth_voigt_model": float(model_flux),
                    "center_velocity_kms": center_velocity,
                    "center_wavelength_A": center_wavelength,
                    "logN": logN,
                    "b_kms": b_kms,
                    "damping_gamma_kms": gamma,
                }
            )

    frames_by_id = {
        str(frame.get("transition_line_id")): frame
        for frame in fit_summary.get("transition_frames", [])
        if frame.get("transition_line_id")
    }
    for transition_line_id, curves in component_curves.items():
        frame = frames_by_id.get(transition_line_id, {})
        bounds = frame.get("velocity_frame", {}).get("bounds_kms")
        if isinstance(bounds, list) and len(bounds) == 2:
            vmin, vmax = float(bounds[0]), float(bounds[1])
        else:
            centers = np.asarray([curve[1][len(curve[1]) // 2] for curve in curves], dtype=float)
            vmin = float(np.nanmin(centers) - 350.0)
            vmax = float(np.nanmax(centers) + 350.0)
        velocity_grid = np.linspace(vmin, vmax, max(int(points_per_peak) * 2, 480))
        total_model = np.ones_like(velocity_grid, dtype=float)
        for _, velocity, model in curves:
            interp_model = np.interp(velocity_grid, velocity, model, left=1.0, right=1.0)
            total_model *= np.clip(interp_model, 0.0, 1.5)

        observed_center_A = float(frame.get("observed_center_A", np.nan))
        wavelength = observed_center_A * (1.0 + velocity_grid / C_KMS) if np.isfinite(observed_center_A) else np.full_like(velocity_grid, np.nan)
        for wave_A, vel_kms, model_flux in zip(wavelength, velocity_grid, total_model, strict=True):
            rows.append(
                {
                    "curve_kind": "combined",
                    "component_index": -1,
                    "transition_line_id": transition_line_id,
                    "wavelength": float(wave_A),
                    "transition_velocity_kms": float(vel_kms),
                    "smooth_voigt_model": float(model_flux),
                    "center_velocity_kms": np.nan,
                    "center_wavelength_A": np.nan,
                    "logN": np.nan,
                    "b_kms": np.nan,
                    "damping_gamma_kms": np.nan,
                }
            )

    columns = [
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
    ]
    return pd.DataFrame(rows, columns=columns)


def _component_posterior_band(
    component: dict[str, Any],
    velocity_grid: np.ndarray,
    *,
    n_samples: int = 128,
) -> np.ndarray | None:
    if component.get("fit_backend") != "ultranest":
        return None
    if component.get("parameter_estimator") != "posterior_median":
        return None
    if set(component.get("diagnostic_flags", [])) & {
        "component_parameter_posterior_degenerate",
        "component_parameter_posterior_disagrees_with_map",
    }:
        return None
    intervals = component.get("parameter_intervals", {})
    logN_i = intervals.get("logN", {})
    b_i = intervals.get("b_kms", {})
    center_i = intervals.get("center_velocity_kms", {})
    if not logN_i or not b_i or not center_i:
        return None
    q16 = np.array([logN_i.get("q16"), b_i.get("q16"), center_i.get("q16")], dtype=float)
    q50 = np.array([logN_i.get("median"), b_i.get("median"), center_i.get("median")], dtype=float)
    q84 = np.array([logN_i.get("q84"), b_i.get("q84"), center_i.get("q84")], dtype=float)
    if not np.all(np.isfinite(q16)) or not np.all(np.isfinite(q50)) or not np.all(np.isfinite(q84)):
        return None
    if np.any(q84 < q16):
        return None
    scales = (q84 - q16) / 2.0
    samples = []
    rng = np.random.default_rng(12345 + int(component.get("component_index", 0)))
    rest_A = float(component.get("rest_wavelength_A", np.nan))
    oscillator_strength = float(component.get("oscillator_strength", np.nan))
    damping_gamma_kms = float(component.get("damping_gamma_kms", np.nan))
    if not np.isfinite(rest_A) or not np.isfinite(oscillator_strength) or not np.isfinite(damping_gamma_kms):
        return None
    for _ in range(int(n_samples)):
        draw = np.where(scales > 0.0, rng.normal(q50, scales), q50)
        draw = np.clip(draw, q16, q84)
        samples.append(
            physical_component_flux_model(
                velocity_grid,
                logN=float(draw[0]),
                b_kms=float(draw[1]),
                center_kms=float(draw[2]),
                rest_wavelength_A=rest_A,
                oscillator_strength=oscillator_strength,
                damping_gamma_kms=damping_gamma_kms,
            )
        )
    return np.asarray(samples, dtype=float)


def _combined_posterior_band(
    components: list[dict[str, Any]],
    transition_line_id: str,
    velocity_grid: np.ndarray,
    *,
    n_samples: int = 128,
) -> tuple[np.ndarray, np.ndarray] | None:
    transition_components = [
        component
        for component in components
        if component.get("fit_success") and str(component.get("transition_line_id")) == transition_line_id
        and component.get("fit_backend") == "ultranest"
        and component.get("parameter_estimator") == "posterior_median"
        and not (
            set(component.get("diagnostic_flags", []))
            & {"component_parameter_posterior_degenerate", "component_parameter_posterior_disagrees_with_map"}
        )
    ]
    if not transition_components:
        return None
    combined_samples = np.ones((int(n_samples), len(velocity_grid)), dtype=float)
    used = 0
    for component in transition_components:
        samples = _component_posterior_band(component, velocity_grid, n_samples=n_samples)
        if samples is None:
            continue
        combined_samples *= np.clip(samples, 0.0, 1.5)
        used += 1
    if used == 0:
        return None
    q16, q84 = np.nanpercentile(combined_samples, [16.0, 84.0], axis=0)
    return q16, q84


def save_review_plot(
    record: dict[str, Any],
    window: pd.DataFrame,
    output_path: str | Path,
    plot_window: pd.DataFrame | None = None,
    plot_summary: dict[str, Any] | None = None,
    fit_summary: dict[str, Any] | None = None,
) -> Path:
    if plot_window is None or plot_summary is None or fit_summary is None:
        raise ValueError("save_review_plot requires precomputed plot_window, plot_summary, and fit_summary")

    plot_path = Path(output_path)
    _configure_matplotlib_cache()

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    wavelength = plot_window["wavelength"].to_numpy(dtype=float)
    normalized_flux = plot_window["normalized_flux"].to_numpy(dtype=float)
    normalized_ivar = plot_window["normalized_ivar"].to_numpy(dtype=float)
    good = plot_window["is_good_pixel"].to_numpy(dtype=bool)
    smooth_model = build_smooth_voigt_model_data(fit_summary)

    with np.errstate(divide="ignore", invalid="ignore"):
        normalized_err = np.where(normalized_ivar > 0, 1.0 / np.sqrt(normalized_ivar), np.nan)

    frame_by_id = {
        str(frame.get("transition_line_id")): frame
        for frame in fit_summary.get("transition_frames", [])
        if frame.get("transition_line_id")
    }
    transitions = record.get("input", {}).get("transitions", [])
    if not transitions:
        transitions = list(frame_by_id.values())
    has_frame_residual_schema = any("residual_samples" in frame for frame in frame_by_id.values())
    n_panels = max(1, len(transitions))
    fig_height = max(4.0, 3.25 * n_panels + 0.8)
    height_ratios = [2.7, 1.0] * n_panels
    fig, axes = plt.subplots(
        2 * n_panels,
        1,
        figsize=(10.5, fig_height),
        sharex=True,
        squeeze=False,
        gridspec_kw={"height_ratios": height_ratios, "hspace": 0.08},
    )
    axes_flat = axes.ravel()

    transition_half_width = float(fit_summary.get("transition_half_width_kms", record.get("input", {}).get("half_width_kms", 600.0)))
    component_color = "#2b6cb0"
    combined_color = "#c1121f"
    excluded_color = "#d97706"
    fit_mask_color = "#7c3aed"
    fit_mask_intervals = _fit_mask_intervals_by_transition(record, fit_summary)
    legacy_residual_sigma = (
        plot_window["voigt_residual_sigma"].to_numpy(dtype=float)
        if "voigt_residual_sigma" in plot_window.columns
        else np.full(len(plot_window), np.nan, dtype=float)
    )
    legacy_fit_pixel = (
        plot_window["is_voigt_fit_pixel"].to_numpy(dtype=bool)
        if "is_voigt_fit_pixel" in plot_window.columns
        else np.isfinite(legacy_residual_sigma)
    )

    flux_axes: list[Any] = []
    residual_axes: list[Any] = []
    for panel_index, transition in enumerate(transitions):
        axis = axes_flat[2 * panel_index]
        residual_axis = axes_flat[2 * panel_index + 1]
        flux_axes.append(axis)
        residual_axes.append(residual_axis)
        transition_line_id = str(transition.get("transition_line_id", transition.get("line_id", "")))
        frame = frame_by_id.get(transition_line_id, {})
        observed_center_A = float(transition.get("observed_center_A", frame.get("observed_center_A", np.nan)))
        rest_A = float(transition.get("rest_wavelength_A", frame.get("rest_wavelength_A", np.nan)))
        if not np.isfinite(observed_center_A):
            axis.text(0.5, 0.5, f"{transition_line_id}: missing transition center", ha="center", va="center")
            residual_axis.set_visible(False)
            continue

        local_velocity = velocity_kms(wavelength, observed_center_A)
        panel_mask = np.abs(local_velocity) <= transition_half_width
        order = np.argsort(local_velocity[panel_mask])
        v_panel = local_velocity[panel_mask][order]
        f_panel = normalized_flux[panel_mask][order]
        err_panel = normalized_err[panel_mask][order]
        good_panel = good[panel_mask][order]
        residual_velocity, residual_panel = _frame_residual_series(frame)
        masked_residual_velocity, masked_residual_panel = _frame_diagnostic_masked_residual_series(frame)
        if len(residual_velocity) == 0 and not has_frame_residual_schema:
            legacy_residual_panel = legacy_residual_sigma[panel_mask][order]
            legacy_fit_panel = legacy_fit_pixel[panel_mask][order] & np.isfinite(legacy_residual_panel)
            residual_velocity = v_panel[legacy_fit_panel]
            residual_panel = legacy_residual_panel[legacy_fit_panel]

        if len(v_panel) == 0:
            axis.text(0.5, 0.5, f"{transition_line_id}: no pixels in velocity frame", ha="center", va="center")
            residual_axis.set_visible(False)
            continue

        _draw_source_work_window(axis, residual_axis)

        axis.fill_between(
            v_panel,
            f_panel - err_panel,
            f_panel + err_panel,
            step="mid",
            color="0.82",
            alpha=0.45,
            linewidth=0,
            label="1 sigma",
        )
        axis.step(v_panel, f_panel, where="mid", color="black", linewidth=1.0, label="normalized flux")
        if (~good_panel).any():
            axis.scatter(v_panel[~good_panel], f_panel[~good_panel], s=16, color="#6b46c1", zorder=4, label="bad pixel")

        excluded = plot_window["is_continuum_excluded_pixel"].to_numpy(dtype=bool)[panel_mask][order]
        for start_v, end_v in _contiguous_intervals(v_panel, excluded):
            axis.axvspan(start_v, end_v, color=excluded_color, alpha=0.12, linewidth=0)
        _draw_fit_mask_intervals(axis, residual_axis, fit_mask_intervals.get(transition_line_id, []), fit_mask_color)

        model_for_transition = smooth_model[smooth_model["transition_line_id"] == transition_line_id] if not smooth_model.empty else smooth_model
        if not model_for_transition.empty:
            component_rows = model_for_transition[model_for_transition["curve_kind"] == "component"]
            for component_index, group in component_rows.groupby("component_index", sort=True):
                group = group.sort_values("transition_velocity_kms")
                label = "Voigt components" if component_index == component_rows["component_index"].min() else None
                axis.plot(
                    group["transition_velocity_kms"],
                    group["smooth_voigt_model"],
                    color=component_color,
                    linewidth=1.0,
                    alpha=0.75,
                    label=label,
                )
            combined_rows = model_for_transition[model_for_transition["curve_kind"] == "combined"].sort_values("transition_velocity_kms")
            if not combined_rows.empty:
                combined_velocity = combined_rows["transition_velocity_kms"].to_numpy(dtype=float)
                combined_band = _combined_posterior_band(
                    fit_summary.get("components", []),
                    transition_line_id,
                    combined_velocity,
                )
                if combined_band is not None:
                    lower, upper = combined_band
                    axis.fill_between(
                        combined_velocity,
                        lower,
                        upper,
                        color="#ef4444",
                        alpha=0.28,
                        linewidth=0,
                        zorder=2.4,
                        label="combined 68% band",
                    )
                axis.plot(
                    combined_rows["transition_velocity_kms"],
                    combined_rows["smooth_voigt_model"],
                    color=combined_color,
                    linewidth=1.7,
                    zorder=3.0,
                    label="combined model",
                )

        for component in fit_summary.get("components", []):
            if component.get("transition_line_id") == transition_line_id and component.get("fit_success"):
                center_v = float(component.get("center_velocity_kms", np.nan))
                if np.isfinite(center_v):
                    axis.axvline(center_v, color=combined_color, linewidth=0.8, alpha=0.35)

        finite_flux = f_panel[np.isfinite(f_panel)]
        if len(finite_flux):
            lo, hi = np.nanpercentile(finite_flux, [2.0, 98.0])
            ymin = max(-0.5, min(0.0, float(lo) - 0.15))
            ymax = min(1.8, max(1.2, float(hi) + 0.15))
            if ymax <= ymin:
                ymin, ymax = -0.1, 1.3
            axis.set_ylim(ymin, ymax)
        axis.set_xlim(-transition_half_width, transition_half_width)
        axis.axhline(1.0, color="0.35", linewidth=0.8, linestyle="--")
        axis.axvline(0.0, color="0.55", linewidth=0.8, linestyle=":")
        axis.set_ylabel("Norm. flux")
        axis.set_title(f"{transition_line_id}  rest={rest_A:.3f} A  center={observed_center_A:.2f} A", fontsize=10, loc="left")
        axis.grid(alpha=0.18)

        residual_axis.axhline(0.0, color="0.25", linewidth=0.8)
        residual_axis.axhline(3.0, color="#c1121f", linewidth=0.7, linestyle="--", alpha=0.65)
        residual_axis.axhline(-3.0, color="#c1121f", linewidth=0.7, linestyle="--", alpha=0.65)
        residual_axis.step(residual_velocity, residual_panel, where="mid", color="#334155", linewidth=0.95, label="residual sigma")
        if len(masked_residual_velocity):
            residual_axis.scatter(
                masked_residual_velocity,
                masked_residual_panel,
                s=10,
                color="#7c3aed",
                alpha=0.55,
                zorder=3,
                label="masked residual",
            )
        significant = np.isfinite(residual_panel) & (np.abs(residual_panel) >= 3.0)
        if significant.any():
            residual_axis.scatter(
                residual_velocity[significant],
                residual_panel[significant],
                s=12,
                color="#c1121f",
                zorder=4,
                label="|resid| >= 3",
            )
        residual_axis.axvline(0.0, color="0.55", linewidth=0.8, linestyle=":")
        residual_axis.set_ylabel("Resid.\nsigma")
        residual_axis.grid(alpha=0.18)
        finite_residual = np.concatenate(
            [
                residual_panel[np.isfinite(residual_panel)],
                masked_residual_panel[np.isfinite(masked_residual_panel)],
            ]
        )
        if len(finite_residual):
            lo, hi = np.nanpercentile(finite_residual, [2.0, 98.0])
            bound = max(4.0, min(25.0, float(max(abs(lo), abs(hi))) + 1.0))
            residual_axis.set_ylim(-bound, bound)
        else:
            residual_axis.set_ylim(-4.0, 4.0)

    if flux_axes:
        flux_axes[0].legend(loc="best", fontsize=8)
    for axis in flux_axes[:-1]:
        axis.tick_params(labelbottom=False)
    for axis in residual_axes[:-1]:
        axis.tick_params(labelbottom=False)
    axes_flat[-1].set_xlabel("Velocity relative to each transition center (km/s)")
    fig.suptitle(
        f"{record['sample_id']} | {record['input']['line_id']} z={record['input']['z_sys']:.4f} | "
        f"quality={fit_summary.get('quality', fit_summary.get('status', 'fit'))}",
        fontsize=11,
    )
    fig.subplots_adjust(top=0.88, bottom=0.11, left=0.08, right=0.98, hspace=0.10)
    fig.savefig(plot_path, dpi=160)
    plt.close(fig)
    return plot_path


def _fit_mask_intervals_by_transition(record: dict[str, Any], fit_summary: dict[str, Any]) -> dict[str, list[dict[str, Any]]]:
    intervals: dict[str, list[dict[str, Any]]] = {}
    for patch in record.get("fit_control_patches", []):
        for call in patch.get("tool_calls", []):
            if call.get("name") != "set_fit_mask_interval":
                continue
            args = call.get("arguments", {})
            if not isinstance(args, dict) or str(args.get("mask_kind", "exclude")) != "exclude":
                continue
            transition_line_id = str(args.get("transition_line_id", ""))
            start = _finite_float(args.get("start_velocity_kms"))
            end = _finite_float(args.get("end_velocity_kms"))
            if not transition_line_id or start is None or end is None:
                continue
            lower, upper = sorted((start, end))
            intervals.setdefault(transition_line_id, []).append(
                {
                    "start_velocity_kms": lower,
                    "end_velocity_kms": upper,
                    "reason": str(args.get("reason", "")),
                }
            )

    controls = fit_summary.get("fit_control_applied", {})
    for interval in controls.get("fit_mask_intervals", []) if isinstance(controls, dict) else []:
        transition_line_id = str(interval.get("transition_line_id", ""))
        start = _finite_float(interval.get("start_velocity_kms"))
        end = _finite_float(interval.get("end_velocity_kms"))
        if transition_line_id and start is not None and end is not None:
            lower, upper = sorted((start, end))
            intervals.setdefault(transition_line_id, []).append(
                {
                    "start_velocity_kms": lower,
                    "end_velocity_kms": upper,
                    "reason": str(interval.get("reason", "")),
                }
            )
    return intervals


def _frame_residual_series(frame: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    samples = frame.get("residual_samples", [])
    if not isinstance(samples, list) or not samples:
        return np.asarray([], dtype=float), np.asarray([], dtype=float)
    rows = [
        (float(sample.get("velocity_kms")), float(sample.get("residual_sigma")))
        for sample in samples
        if _finite_float(sample.get("velocity_kms")) is not None and _finite_float(sample.get("residual_sigma")) is not None
    ]
    if not rows:
        return np.asarray([], dtype=float), np.asarray([], dtype=float)
    rows.sort(key=lambda item: item[0])
    return np.asarray([item[0] for item in rows], dtype=float), np.asarray([item[1] for item in rows], dtype=float)


def _frame_diagnostic_masked_residual_series(frame: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    samples = frame.get("diagnostic_residual_samples", [])
    if not isinstance(samples, list) or not samples:
        return np.asarray([], dtype=float), np.asarray([], dtype=float)
    rows = [
        (float(sample.get("velocity_kms")), float(sample.get("residual_sigma")))
        for sample in samples
        if sample.get("used_in_fit") is False
        and _finite_float(sample.get("velocity_kms")) is not None
        and _finite_float(sample.get("residual_sigma")) is not None
    ]
    if not rows:
        return np.asarray([], dtype=float), np.asarray([], dtype=float)
    rows.sort(key=lambda item: item[0])
    return np.asarray([item[0] for item in rows], dtype=float), np.asarray([item[1] for item in rows], dtype=float)


def _draw_fit_mask_intervals(axis: Any, residual_axis: Any, intervals: list[dict[str, Any]], color: str) -> None:
    for index, interval in enumerate(intervals):
        start = float(interval["start_velocity_kms"])
        end = float(interval["end_velocity_kms"])
        label = "fit mask" if index == 0 else None
        axis.axvspan(start, end, color=color, alpha=0.14, linewidth=0, label=label)
        residual_axis.axvspan(start, end, color=color, alpha=0.14, linewidth=0)
        axis.axvline(start, color=color, linewidth=0.6, alpha=0.55)
        axis.axvline(end, color=color, linewidth=0.6, alpha=0.55)
        residual_axis.axvline(start, color=color, linewidth=0.6, alpha=0.55)
        residual_axis.axvline(end, color=color, linewidth=0.6, alpha=0.55)


def _draw_source_work_window(axis: Any, residual_axis: Any) -> None:
    start, end = SOURCE_WORK_WINDOW_KMS
    for target_axis in (axis, residual_axis):
        target_axis.axvspan(start, end, color="#22c55e", alpha=0.035, linewidth=0, label="source work window" if target_axis is axis else None)
        target_axis.axvline(start, color="#16a34a", linewidth=0.75, linestyle="--", alpha=0.55)
        target_axis.axvline(end, color="#16a34a", linewidth=0.75, linestyle="--", alpha=0.55)


def _finite_float(value: Any) -> float | None:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if np.isfinite(parsed) else None


def save_window_overview_plot(
    record: dict[str, Any],
    window: pd.DataFrame,
    output_path: str | Path,
    plot_window: pd.DataFrame,
) -> Path:
    """Save wavelength-space context for the same local review window."""
    plot_path = Path(output_path)
    _configure_matplotlib_cache()

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    wavelength = plot_window["wavelength"].to_numpy(dtype=float)
    flux = plot_window["flux"].to_numpy(dtype=float)
    ivar = plot_window["ivar"].to_numpy(dtype=float)
    continuum = plot_window["continuum_model"].to_numpy(dtype=float)
    normalized_flux = plot_window["normalized_flux"].to_numpy(dtype=float)
    good = plot_window["is_good_pixel"].to_numpy(dtype=bool)
    with np.errstate(divide="ignore", invalid="ignore"):
        flux_err = np.where(ivar > 0, 1.0 / np.sqrt(ivar), np.nan)

    fig, axes = plt.subplots(2, 1, figsize=(10.5, 5.0), sharex=True, squeeze=False)
    raw_axis, norm_axis = axes.ravel()
    raw_axis.fill_between(wavelength, flux - flux_err, flux + flux_err, step="mid", color="0.82", alpha=0.45, linewidth=0, label="1 sigma")
    raw_axis.step(wavelength, flux, where="mid", color="black", linewidth=1.0, label="flux")
    raw_axis.plot(wavelength, continuum, color="#c1121f", linewidth=1.4, label="continuum")
    if (~good).any():
        raw_axis.scatter(wavelength[~good], flux[~good], s=14, color="#6b46c1", zorder=4, label="bad pixel")

    norm_axis.step(wavelength, normalized_flux, where="mid", color="black", linewidth=1.0, label="normalized flux")
    norm_axis.axhline(1.0, color="0.35", linewidth=0.8, linestyle="--")
    norm_axis.axhline(0.0, color="0.65", linewidth=0.7, linestyle=":")

    for transition in record.get("input", {}).get("transitions", []):
        center = float(transition.get("observed_center_A", np.nan))
        label = str(transition.get("transition_line_id", "transition"))
        if np.isfinite(center):
            for axis in (raw_axis, norm_axis):
                axis.axvline(center, color="#2b6cb0", linewidth=0.9, alpha=0.75)
                axis.text(center, axis.get_ylim()[1], label, rotation=90, va="top", ha="right", fontsize=8, color="#2b6cb0")

    raw_axis.set_ylabel("Flux")
    norm_axis.set_ylabel("Norm. flux")
    norm_axis.set_xlabel("Observed wavelength (A)")
    raw_axis.legend(loc="best", fontsize=8)
    raw_axis.grid(alpha=0.18)
    norm_axis.grid(alpha=0.18)
    finite_norm = normalized_flux[np.isfinite(normalized_flux)]
    if len(finite_norm):
        lo, hi = np.nanpercentile(finite_norm, [1.0, 99.0])
        norm_axis.set_ylim(max(-0.5, min(-0.05, float(lo) - 0.15)), min(2.0, max(1.2, float(hi) + 0.15)))
    fig.suptitle(f"{record['sample_id']} | wavelength-space local window", fontsize=11)
    fig.tight_layout()
    fig.savefig(plot_path, dpi=160)
    plt.close(fig)
    return plot_path
