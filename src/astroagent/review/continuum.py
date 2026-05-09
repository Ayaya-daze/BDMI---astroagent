from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from scipy.interpolate import PchipInterpolator
from scipy.signal import find_peaks


C_KMS = 299792.458
REQUIRED_SPECTRUM_COLUMNS = ("wavelength", "flux", "ivar", "pipeline_mask")
CONTINUUM_LINE_CORE_EXCLUSION_KMS = 250.0


def observed_wavelength_A(rest_wavelength_A: float, z_sys: float) -> float:
    return float(rest_wavelength_A) * (1.0 + float(z_sys))


def velocity_kms(wavelength_A: np.ndarray, center_wavelength_A: float) -> np.ndarray:
    return C_KMS * (np.asarray(wavelength_A, dtype=float) / float(center_wavelength_A) - 1.0)


def validate_spectrum_table(spectrum: pd.DataFrame) -> None:
    missing = [column for column in REQUIRED_SPECTRUM_COLUMNS if column not in spectrum.columns]
    if missing:
        raise ValueError(f"spectrum is missing required columns: {missing}")
    if spectrum.empty:
        raise ValueError("spectrum table is empty")


def _contiguous_intervals(wavelength: np.ndarray, selected: np.ndarray) -> list[list[float]]:
    intervals: list[list[float]] = []
    if len(wavelength) == 0 or not selected.any():
        return intervals

    selected_indices = np.flatnonzero(selected)
    groups = np.split(selected_indices, np.where(np.diff(selected_indices) != 1)[0] + 1)
    for group in groups:
        intervals.append([float(wavelength[group[0]]), float(wavelength[group[-1]])])
    return intervals


def _data_driven_absorption_exclusion_mask(
    wavelength: np.ndarray,
    flux: np.ndarray,
    good: np.ndarray,
    half_width_kms: float = CONTINUUM_LINE_CORE_EXCLUSION_KMS,
) -> tuple[np.ndarray, list[list[float]]]:
    if good.sum() < 5:
        return np.zeros(len(wavelength), dtype=bool), []

    good_indices = np.flatnonzero(good)
    wave_good = wavelength[good]
    flux_good = flux[good]
    baseline = float(np.nanmedian(flux_good))
    scatter = float(1.4826 * np.nanmedian(np.abs(flux_good - baseline)))
    if not np.isfinite(scatter) or scatter <= 0.0:
        scatter = float(np.nanstd(flux_good))
    if not np.isfinite(scatter) or scatter <= 0.0:
        scatter = max(abs(baseline) * 0.03, 1e-6)

    depth = baseline - flux_good
    threshold = max(2.0 * scatter, 0.04 * max(abs(baseline), 1e-6))
    peaks, _ = find_peaks(depth, height=threshold, distance=3, prominence=0.5 * threshold)
    if len(peaks) == 0:
        return np.zeros(len(wavelength), dtype=bool), []

    exclusion_mask = np.zeros(len(wavelength), dtype=bool)
    for peak in peaks:
        center_A = float(wave_good[peak])
        half_width_A = center_A * float(half_width_kms) / C_KMS
        exclusion_mask |= (wavelength >= center_A - half_width_A) & (wavelength <= center_A + half_width_A)

        full_index = int(good_indices[peak])
        left = full_index
        right = full_index
        trough_limit = baseline - 0.75 * scatter
        while left > 0 and good[left - 1] and flux[left - 1] < trough_limit:
            left -= 1
        while right < len(flux) - 1 and good[right + 1] and flux[right + 1] < trough_limit:
            right += 1
        exclusion_mask[left : right + 1] = True

    return exclusion_mask, _contiguous_intervals(wavelength, exclusion_mask)


def _smooth_anchor_fluxes(anchor_flux: np.ndarray, anchor_weight: np.ndarray) -> np.ndarray:
    if len(anchor_flux) < 5:
        return anchor_flux.copy()

    smoothed = anchor_flux.copy()
    for index in range(len(anchor_flux)):
        left = max(0, index - 1)
        right = min(len(anchor_flux), index + 2)
        local_weight = anchor_weight[left:right]
        local_flux = anchor_flux[left:right]
        if np.isfinite(local_weight).all() and float(local_weight.sum()) > 0.0:
            neighbor_average = float(np.average(local_flux, weights=local_weight))
        else:
            neighbor_average = float(np.median(local_flux))
        smoothed[index] = 0.70 * float(anchor_flux[index]) + 0.30 * neighbor_average
    return smoothed


def _local_anchor_continuum(
    wavelength: np.ndarray,
    anchor_wave: np.ndarray,
    anchor_flux: np.ndarray,
    anchor_weight: np.ndarray,
    smooth_nodes: bool = True,
) -> np.ndarray:
    floor = _continuum_floor(anchor_flux)
    order = np.argsort(anchor_wave)
    x = anchor_wave[order]
    y = anchor_flux[order] if not smooth_nodes else _smooth_anchor_fluxes(anchor_flux[order], anchor_weight[order])

    unique_x, unique_indices = np.unique(x, return_index=True)
    unique_y = y[unique_indices]
    if len(unique_x) == 0:
        return np.full_like(wavelength, floor, dtype=float)
    if len(unique_x) == 1:
        return np.full_like(wavelength, max(float(unique_y[0]), floor), dtype=float)
    if len(unique_x) == 2:
        continuum = np.interp(wavelength, unique_x, unique_y)
    else:
        interpolator = PchipInterpolator(unique_x, unique_y, extrapolate=True)
        continuum = interpolator(wavelength)
        left = wavelength < unique_x[0]
        right = wavelength > unique_x[-1]
        if left.any():
            continuum[left] = np.interp(wavelength[left], unique_x[:2], unique_y[:2])
        if right.any():
            continuum[right] = np.interp(wavelength[right], unique_x[-2:], unique_y[-2:])

    return np.clip(np.asarray(continuum, dtype=float), floor, None)


def _continuum_floor(values: np.ndarray | list[float]) -> float:
    array = np.asarray(values, dtype=float)
    finite_positive = array[np.isfinite(array) & (array > 0.0)]
    if len(finite_positive):
        scale = float(np.nanmedian(finite_positive))
    else:
        finite_abs = np.abs(array[np.isfinite(array)])
        scale = float(np.nanmedian(finite_abs)) if len(finite_abs) else 1.0
    if not np.isfinite(scale) or scale <= 0.0:
        scale = 1.0
    return float(max(scale * 1.0e-6, np.finfo(float).tiny))


def _rolling_upper_envelope_continuum(
    wavelength: np.ndarray,
    flux: np.ndarray,
    good: np.ndarray,
    exclude_mask: np.ndarray | None,
    n_bins: int | None = None,
    percentile: float = 82.0,
    extra_anchor_wavelengths_A: list[float] | None = None,
    extra_anchor_nodes: list[dict[str, Any]] | None = None,
    remove_anchor_indices: list[int] | None = None,
    remove_anchor_wavelengths_A: list[float] | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    fit_mask = good.copy() if exclude_mask is None else good & ~exclude_mask
    if fit_mask.sum() < max(8, int(good.sum() * 0.30)):
        fit_mask = good.copy()

    if n_bins is None:
        n_bins = max(8, min(18, int(np.ceil(max(int(fit_mask.sum()), 1) / 5.0))))
    n_bins = max(4, int(n_bins))

    wave_fit = wavelength[fit_mask]
    if len(wave_fit) == 0:
        floor = _continuum_floor(flux[good] if good.any() else flux)
        level = max(float(np.nanmedian(flux[good])) if good.any() else float(np.nanmedian(flux)), floor)
        return np.full_like(flux, level), good.copy(), np.array([float(np.nanmedian(wavelength))]), {
            "anchor_strategy": "rolling_upper_envelope_fallback",
            "n_anchor_bins": 1,
        }

    edges = np.linspace(float(wave_fit.min()), float(wave_fit.max()), n_bins + 1)
    anchor_wave: list[float] = []
    anchor_flux: list[float] = []
    anchor_weight: list[float] = []
    min_pixels = 3
    for index, (lower, upper) in enumerate(zip(edges[:-1], edges[1:], strict=True)):
        if index == n_bins - 1:
            bin_mask = fit_mask & (wavelength >= lower) & (wavelength <= upper)
        else:
            bin_mask = fit_mask & (wavelength >= lower) & (wavelength < upper)
        if int(bin_mask.sum()) < min_pixels:
            continue
        values = flux[bin_mask]
        anchor_wave.append(float(np.median(wavelength[bin_mask])))
        anchor_flux.append(float(np.percentile(values, percentile)))
        anchor_weight.append(float(bin_mask.sum()))

    if len(anchor_wave) < 4:
        selected_wave = wavelength[fit_mask]
        selected_flux = flux[fit_mask]
        positions = np.linspace(0, len(selected_wave) - 1, min(6, len(selected_wave))).round().astype(int)
        anchor_wave = [float(selected_wave[i]) for i in positions]
        anchor_flux = [float(selected_flux[i]) for i in positions]
        anchor_weight = [1.0 for _ in positions]

    manual_anchors: list[dict[str, float]] = []
    if extra_anchor_wavelengths_A:
        finite_flux = np.isfinite(flux)
        anchor_pool = fit_mask & finite_flux
        if not anchor_pool.any():
            anchor_pool = good & finite_flux
        if not anchor_pool.any():
            anchor_pool = finite_flux
        pool_indices = np.flatnonzero(anchor_pool)
        for anchor_A in extra_anchor_wavelengths_A:
            anchor_A = float(anchor_A)
            if not np.isfinite(anchor_A) or len(pool_indices) == 0:
                continue
            nearest = int(pool_indices[np.argmin(np.abs(wavelength[pool_indices] - anchor_A))])
            anchor_wave.append(float(wavelength[nearest]))
            anchor_flux.append(float(flux[nearest]))
            anchor_weight.append(1.0)
            manual_anchors.append(
                {
                    "wavelength_A": float(wavelength[nearest]),
                    "continuum_flux": float(flux[nearest]),
                    "source": "nearest_good_pixel",
                }
            )

    if extra_anchor_nodes:
        for node in extra_anchor_nodes:
            anchor_A = float(node.get("wavelength_A", np.nan))
            continuum_flux = float(node.get("continuum_flux", np.nan))
            if not np.isfinite(anchor_A) or not np.isfinite(continuum_flux):
                continue
            anchor_wave.append(anchor_A)
            anchor_flux.append(continuum_flux)
            anchor_weight.append(float(node.get("weight", 1.0)))
            manual_anchors.append(
                {
                    "wavelength_A": anchor_A,
                    "continuum_flux": continuum_flux,
                    "source": str(node.get("source", "fit_control_node")),
                }
            )

    if remove_anchor_indices:
        remove = {int(index) for index in remove_anchor_indices if int(index) >= 0}
        kept = [
            (wave, flux_value, weight)
            for index, (wave, flux_value, weight) in enumerate(zip(anchor_wave, anchor_flux, anchor_weight, strict=True))
            if index not in remove
        ]
        anchor_wave = [item[0] for item in kept]
        anchor_flux = [item[1] for item in kept]
        anchor_weight = [item[2] for item in kept]

    removed_anchor_wavelengths: list[float] = []
    if remove_anchor_wavelengths_A:
        targets = [float(value) for value in remove_anchor_wavelengths_A if np.isfinite(float(value))]
        for target in targets:
            if not anchor_wave:
                break
            distances = np.abs(np.asarray(anchor_wave, dtype=float) - target)
            nearest = int(np.argmin(distances))
            tolerance_A = max(0.5, 0.25 * float(np.nanmedian(np.diff(np.sort(wavelength)))) if len(wavelength) > 1 else 0.5)
            if float(distances[nearest]) <= tolerance_A:
                removed_anchor_wavelengths.append(float(anchor_wave[nearest]))
                del anchor_wave[nearest]
                del anchor_flux[nearest]
                del anchor_weight[nearest]

    anchors_w = np.asarray(anchor_wave, dtype=float)
    anchors_f = np.asarray(anchor_flux, dtype=float)
    anchors_weight = np.asarray(anchor_weight, dtype=float)
    continuum = _local_anchor_continuum(
        wavelength,
        anchors_w,
        anchors_f,
        anchors_weight,
        smooth_nodes=False,
    )

    residuals = flux - continuum
    pool = fit_mask if fit_mask.any() else good
    scatter = (
        float(1.4826 * np.nanmedian(np.abs(residuals[pool] - np.nanmedian(residuals[pool]))))
        if pool.any()
        else 1.0
    )
    if not np.isfinite(scatter) or scatter <= 0.0:
        scatter = float(np.nanstd(residuals[pool])) if pool.any() else 1.0
    if not np.isfinite(scatter) or scatter <= 0.0:
        scatter = 1.0
    continuum_mask = fit_mask & (flux > continuum - 2.5 * scatter)

    return continuum, continuum_mask, anchors_w, {
        "anchor_strategy": "rolling_upper_envelope",
        "n_anchor_bins": int(len(anchors_w)),
        "upper_envelope_percentile": float(percentile),
        "anchor_wavelengths_A": [float(value) for value in anchors_w],
        "anchor_fluxes": [float(value) for value in anchors_f],
        "manual_anchor_nodes": manual_anchors,
        "manual_anchor_wavelengths_A": [float(node["wavelength_A"]) for node in manual_anchors],
        "removed_anchor_indices": [int(index) for index in remove_anchor_indices or []],
        "removed_anchor_wavelengths_A": removed_anchor_wavelengths,
    }


def _egent_like_continuum(
    wavelength: np.ndarray,
    flux: np.ndarray,
    good: np.ndarray,
    ivar: np.ndarray | None = None,
    exclude_mask: np.ndarray | None = None,
    sigma_clip: float = 2.5,
    top_percentile: float = 85.0,
    extra_anchor_wavelengths_A: list[float] | None = None,
    extra_anchor_nodes: list[dict[str, Any]] | None = None,
    remove_anchor_indices: list[int] | None = None,
    remove_anchor_wavelengths_A: list[float] | None = None,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    if exclude_mask is None:
        fit_mask = good.copy()
    else:
        fit_mask = good & ~exclude_mask

    if fit_mask.sum() < max(3, int(good.sum() * 0.25)):
        fit_mask = good.copy()

    if good.sum() == 0:
        floor = _continuum_floor(flux)
        continuum = np.full_like(flux, max(float(np.nanmedian(flux)), floor), dtype=float)
        return continuum, good.copy(), {
            "fit_type": "egent_iterative_continuum",
            "method": "constant_fallback",
            "polynomial_order": 0,
            "iterations": 0,
            "top_percentile": float(top_percentile),
            "sigma_clip": float(sigma_clip),
            "n_continuum_pixels": 0,
            "coefficients_high_to_low": [float(continuum[0])],
            "wave_center_A": float(np.nanmean(wavelength)),
        }

    continuum, continuum_mask, envelope_anchors, envelope_summary = _rolling_upper_envelope_continuum(
        wavelength,
        flux,
        good,
        exclude_mask=exclude_mask,
        extra_anchor_wavelengths_A=extra_anchor_wavelengths_A,
        extra_anchor_nodes=extra_anchor_nodes,
        remove_anchor_indices=remove_anchor_indices,
        remove_anchor_wavelengths_A=remove_anchor_wavelengths_A,
    )

    residuals = flux - continuum
    residual_pool = fit_mask if fit_mask.any() else good
    scatter = (
        float(1.4826 * np.nanmedian(np.abs(residuals[residual_pool] - np.nanmedian(residuals[residual_pool]))))
        if residual_pool.any()
        else 1.0
    )
    if not np.isfinite(scatter) or scatter <= 0.0:
        scatter = float(np.nanstd(residuals[residual_pool])) if residual_pool.any() else 1.0
    if not np.isfinite(scatter) or scatter <= 0.0:
        scatter = 1.0

    return np.clip(continuum, _continuum_floor(continuum), None), continuum_mask, {
        "fit_type": "egent_iterative_continuum",
        "method": "rolling_upper_envelope_pchip",
        "polynomial_order": 0,
        "iterations": 1,
        "top_percentile": float(top_percentile),
        "sigma_clip": float(sigma_clip),
        "n_continuum_pixels": int(continuum_mask.sum()),
        "coefficients_high_to_low": [float(np.nanmedian(continuum))],
        "wave_center_A": float(np.nanmean(envelope_anchors)) if len(envelope_anchors) else float(np.nanmean(wavelength)),
        "continuum_exclusion_kms": float(CONTINUUM_LINE_CORE_EXCLUSION_KMS if exclude_mask is not None else 0.0),
        "continuum_scatter": float(scatter),
        **envelope_summary,
    }


def build_plot_data(
    window: pd.DataFrame,
    window_metadata: dict[str, Any] | None = None,
    fit_control: dict[str, Any] | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    validate_spectrum_table(window)
    wavelength = window["wavelength"].to_numpy(dtype=float)
    flux = window["flux"].to_numpy(dtype=float)
    ivar = window["ivar"].to_numpy(dtype=float)
    pipeline_mask = window["pipeline_mask"].to_numpy()

    finite = np.isfinite(wavelength) & np.isfinite(flux) & np.isfinite(ivar)
    good = finite & (ivar > 0) & (pipeline_mask == 0)
    continuum_exclusion_mask, continuum_exclusion_intervals = _data_driven_absorption_exclusion_mask(wavelength, flux, good)
    controls = fit_control or {}
    continuum_exclusion_mask = _apply_continuum_mask_intervals(
        wavelength,
        continuum_exclusion_mask,
        controls.get("continuum_mask_intervals_A", []),
    )
    continuum_exclusion_intervals = _contiguous_intervals(wavelength, continuum_exclusion_mask)
    provided_continuum = window["CONTINUUM"].to_numpy(dtype=float) if "CONTINUUM" in window.columns else None
    if provided_continuum is not None and np.isfinite(provided_continuum).any():
        continuum = np.asarray(provided_continuum, dtype=float)
        continuum = np.where(np.isfinite(continuum) & (continuum > 0.0), continuum, np.nan)
        if np.isfinite(continuum[good]).any():
            baseline = float(np.nanmedian(continuum[good]))
        elif np.isfinite(continuum).any():
            baseline = float(np.nanmedian(continuum[np.isfinite(continuum)]))
        else:
            baseline = 1.0
        if not np.isfinite(baseline) or baseline <= 0.0:
            baseline = 1.0
        fill_value = max(baseline * 1.0e-6, np.finfo(float).tiny)
        continuum = np.where(np.isfinite(continuum) & (continuum > 0.0), continuum, fill_value)
        continuum_mask = good & ~continuum_exclusion_mask
        summary = {
            "fit_type": "provided_continuum",
            "method": "input_continuum_column",
            "polynomial_order": 0,
            "iterations": 0,
            "top_percentile": None,
            "sigma_clip": None,
            "n_continuum_pixels": int(continuum_mask.sum()),
            "coefficients_high_to_low": [float(np.nanmedian(continuum))],
            "wave_center_A": float(np.nanmean(wavelength)),
            "continuum_exclusion_kms": float(CONTINUUM_LINE_CORE_EXCLUSION_KMS if continuum_exclusion_mask is not None else 0.0),
            "continuum_scatter": float(np.nanstd((flux[good] / continuum[good]) - 1.0)) if good.any() else 1.0,
            "anchor_strategy": "provided_continuum_column",
            "n_anchor_bins": 0,
            "provided_continuum_column": True,
        }
    else:
        continuum, continuum_mask, summary = _egent_like_continuum(
            wavelength,
            flux,
            good,
            ivar=ivar,
            exclude_mask=continuum_exclusion_mask,
            extra_anchor_wavelengths_A=controls.get("continuum_anchor_wavelengths_A", []),
            extra_anchor_nodes=controls.get("continuum_anchor_nodes", []),
            remove_anchor_indices=controls.get("continuum_anchor_remove_indices", []),
            remove_anchor_wavelengths_A=controls.get("continuum_anchor_remove_wavelengths_A", []),
        )
    normalized_flux = flux / continuum
    normalized_ivar = ivar * np.square(continuum)

    plot_window = window.copy()
    plot_window["continuum_model"] = continuum
    plot_window["normalized_flux"] = normalized_flux
    plot_window["normalized_ivar"] = normalized_ivar
    plot_window["is_good_pixel"] = good.astype(int)
    plot_window["is_continuum_pixel"] = continuum_mask.astype(int)
    plot_window["is_continuum_excluded_pixel"] = continuum_exclusion_mask.astype(int)
    summary["n_good_pixels"] = int(good.sum())
    summary["continuum_exclusion_intervals_A"] = continuum_exclusion_intervals
    if controls:
        summary["fit_control_applied"] = {
            "n_manual_continuum_anchors": int(len(controls.get("continuum_anchor_wavelengths_A", []))),
            "n_manual_continuum_nodes": int(len(controls.get("continuum_anchor_nodes", []))),
            "n_continuum_mask_intervals": int(len(controls.get("continuum_mask_intervals_A", []))),
        }
    return plot_window, summary


def _apply_continuum_mask_intervals(
    wavelength: np.ndarray,
    mask: np.ndarray,
    intervals: list[dict[str, Any]],
) -> np.ndarray:
    updated = mask.copy()
    for interval in intervals:
        start = float(interval.get("start_wavelength_A", np.nan))
        end = float(interval.get("end_wavelength_A", np.nan))
        if not np.isfinite(start) or not np.isfinite(end):
            continue
        lower, upper = sorted((start, end))
        selected = (wavelength >= lower) & (wavelength <= upper)
        if str(interval.get("mask_kind", "exclude")) == "include":
            updated[selected] = False
        else:
            updated[selected] = True
    return updated
