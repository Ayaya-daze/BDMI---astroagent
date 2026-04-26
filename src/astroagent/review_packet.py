from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from astroagent.line_catalog import (
    load_line_catalog,
    primary_rest_wavelength_A,
    rest_wavelengths_A,
)


C_KMS = 299792.458
REQUIRED_SPECTRUM_COLUMNS = ("wavelength", "flux", "ivar", "pipeline_mask")


def observed_wavelength_A(rest_wavelength_A: float, z_sys: float) -> float:
    return float(rest_wavelength_A) * (1.0 + float(z_sys))


def velocity_kms(wavelength_A: np.ndarray, center_wavelength_A: float) -> np.ndarray:
    return C_KMS * (wavelength_A / float(center_wavelength_A) - 1.0)


def validate_spectrum_table(spectrum: pd.DataFrame) -> None:
    missing = [column for column in REQUIRED_SPECTRUM_COLUMNS if column not in spectrum.columns]
    if missing:
        raise ValueError(f"spectrum is missing required columns: {missing}")
    if spectrum.empty:
        raise ValueError("spectrum table is empty")


def cut_local_window(
    spectrum: pd.DataFrame,
    line_id: str,
    z_sys: float,
    half_width_kms: float = 1500.0,
    catalog_path: str | Path | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Cut the local observed-frame window for a known line hypothesis."""
    validate_spectrum_table(spectrum)
    catalog = load_line_catalog(catalog_path)
    rests = rest_wavelengths_A(line_id, catalog)
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

    window["velocity_kms"] = velocity_kms(window["wavelength"].to_numpy(), primary_center)
    metadata = {
        "line_id": line_id,
        "z_sys": float(z_sys),
        "rest_wavelengths_A": rests,
        "observed_centers_A": centers,
        "primary_observed_center_A": primary_center,
        "window_min_A": float(window_min),
        "window_max_A": float(window_max),
        "half_width_kms": float(half_width_kms),
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


def assess_absorber_hypothesis(window: pd.DataFrame, window_metadata: dict[str, Any]) -> dict[str, Any]:
    """Check whether expected line centers are supported by absorption-like pixels."""
    validate_spectrum_table(window)
    wavelength = window["wavelength"].to_numpy()
    flux = window["flux"].to_numpy()
    ivar = window["ivar"].to_numpy()
    pipeline_mask = window["pipeline_mask"].to_numpy()
    good = np.isfinite(flux) & np.isfinite(ivar) & (ivar > 0) & (pipeline_mask == 0)
    continuum_level = float(np.median(flux[good])) if good.any() else float(np.median(flux))
    noise_level = float(np.median(1.0 / np.sqrt(ivar[good]))) if good.any() else 0.0
    depth_threshold = max(0.05, 3.0 * noise_level)

    line_checks = []
    for rest_A, center_A in zip(
        window_metadata["rest_wavelengths_A"],
        window_metadata["observed_centers_A"],
        strict=True,
    ):
        half_width_A = center_A * 150.0 / C_KMS
        local = (wavelength >= center_A - half_width_A) & (wavelength <= center_A + half_width_A) & good
        if local.any():
            local_min_flux = float(np.min(flux[local]))
            depth = max(0.0, continuum_level - local_min_flux)
            detected = depth >= depth_threshold
            n_good_pixels = int(local.sum())
        else:
            local_min_flux = None
            depth = 0.0
            detected = False
            n_good_pixels = 0
        line_checks.append(
            {
                "rest_wavelength_A": float(rest_A),
                "observed_center_A": float(center_A),
                "local_min_flux": local_min_flux,
                "depth_below_continuum": float(depth),
                "depth_threshold": float(depth_threshold),
                "n_good_pixels_near_center": n_good_pixels,
                "absorption_like": bool(detected),
            }
        )

    detected_count = sum(1 for item in line_checks if item["absorption_like"])
    if len(line_checks) == 1:
        status = "plausible" if detected_count == 1 else "weak_or_absent"
        reason = "single expected line has an absorption-like trough" if detected_count == 1 else "expected line center is weak or absent"
    else:
        depths = [item["depth_below_continuum"] for item in line_checks]
        primary_depth = max(depths[0], 1e-6)
        secondary_ratio = depths[1] / primary_depth if len(depths) > 1 else 0.0
        ratio_ok = 0.25 <= secondary_ratio <= 1.30
        if detected_count == len(line_checks) and ratio_ok:
            status = "plausible"
            reason = "expected doublet members are both absorption-like with a plausible depth ratio"
        elif detected_count > 0:
            status = "needs_review"
            reason = "only part of the expected absorber pattern is supported"
        else:
            status = "weak_or_absent"
            reason = "expected absorber centers are weak or absent"

    return {
        "status": status,
        "candidate_absorber_reasonable": status == "plausible",
        "detected_expected_lines": int(detected_count),
        "expected_lines": int(len(line_checks)),
        "continuum_level": continuum_level,
        "noise_level": noise_level,
        "line_checks": line_checks,
        "rationale": reason,
    }


def _contiguous_intervals(wavelength: np.ndarray, selected: np.ndarray) -> list[list[float]]:
    intervals: list[list[float]] = []
    if len(wavelength) == 0 or not selected.any():
        return intervals

    selected_indices = np.flatnonzero(selected)
    groups = np.split(selected_indices, np.where(np.diff(selected_indices) != 1)[0] + 1)
    for group in groups:
        intervals.append([float(wavelength[group[0]]), float(wavelength[group[-1]])])
    return intervals


def suggest_task_a_labels(window: pd.DataFrame) -> dict[str, Any]:
    """Create simple rule labels that a human can accept, reject, or edit."""
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
    record = {
        "sample_id": sample_id,
        "source": source,
        "input": window_metadata,
        "window_summary": summarize_window(window),
        "absorber_hypothesis_check": assess_absorber_hypothesis(window, window_metadata),
        "task_a_rule_suggestion": suggest_task_a_labels(window),
        "human_review": {
            "status": "needs_review",
            "reviewer": "",
            "notes": "",
            "absorber_hypothesis_notes": "",
            "accepted_task_a": None,
            "corrected_task_a": None,
        },
    }
    return record, window


def write_review_packet(record: dict[str, Any], window: pd.DataFrame, output_dir: str | Path) -> dict[str, Path]:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    json_path = output_path / f"{record['sample_id']}.review.json"
    csv_path = output_path / f"{record['sample_id']}.window.csv"
    readme_path = output_path / "README.md"

    with json_path.open("w", encoding="utf-8") as handle:
        json.dump(record, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    window.to_csv(csv_path, index=False)
    readme_path.write_text(
        "\n".join(
            [
                "# 人工审查包",
                "",
                "这个目录里的文件是脚本生成的中间产物。",
                "",
                "- `*.review.json`：一条结构化样本，留给人工审查和修正。",
                "- `*.window.csv`：局域观测波长谱窗。",
                "",
                "先看 `absorber_hypothesis_check` 和 rule suggestion，再填写 `human_review.notes`、",
                "`human_review.accepted_task_a` 或 `human_review.corrected_task_a`。",
                "",
            ]
        ),
        encoding="utf-8",
    )
    return {"json": json_path, "csv": csv_path, "readme": readme_path}


def make_demo_quasar_spectrum(z_sys: float = 2.6, line_id: str = "CIV_doublet") -> pd.DataFrame:
    """Generate a deterministic toy spectrum for local development only."""
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
