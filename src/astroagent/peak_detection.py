from __future__ import annotations

from typing import Any

import numpy as np
from scipy.signal import find_peaks


def detect_absorption_peaks(
    velocity: np.ndarray,
    flux: np.ndarray,
    min_peak_depth: float,
    max_peaks: int,
    min_separation_kms: float,
    good: np.ndarray | None = None,
) -> list[dict[str, Any]]:
    depth = 1.0 - flux
    finite = np.isfinite(velocity) & np.isfinite(depth)
    if good is not None:
        finite &= np.asarray(good, dtype=bool)
    if finite.sum() < 5:
        return []

    vel_good = velocity[finite]
    depth_good = depth[finite]
    order = np.argsort(vel_good)
    vel_good = vel_good[order]
    depth_good = depth_good[order]
    step = float(np.nanmedian(np.diff(vel_good))) if len(vel_good) > 1 else min_separation_kms
    step = max(abs(step), 1.0)
    smooth_depth = smooth_depth_profile(depth_good, step)
    noise = robust_depth_noise(depth_good)
    core_threshold = max(float(min_peak_depth), 3.0 * noise, 0.05)
    island_threshold = max(0.45 * core_threshold, float(min_peak_depth))
    absorption = smooth_depth >= island_threshold
    if not absorption.any():
        return []

    islands = contiguous_index_groups(np.flatnonzero(absorption))
    islands = merge_nearby_islands(islands, vel_good, max_gap_kms=0.60 * float(min_separation_kms))
    distance_pixels = max(1, int(round(float(min_separation_kms) / step)))
    prominence_threshold = max(0.70 * core_threshold, 2.5 * noise, 0.035)

    out: list[dict[str, Any]] = []
    for island in islands:
        local_depth = depth_good[island]
        local_smooth = smooth_depth[island]
        if float(np.nanmax(local_smooth)) < core_threshold:
            continue
        local_peaks, properties = find_peaks(
            local_smooth,
            height=core_threshold,
            distance=distance_pixels,
            prominence=prominence_threshold,
        )
        if len(local_peaks) == 0:
            local_peaks = np.asarray([int(np.nanargmax(local_depth))], dtype=int)
            heights = local_depth[local_peaks]
            prominences = np.asarray([float(np.nanmax(local_smooth) - island_threshold)], dtype=float)
        else:
            heights = properties.get("peak_heights", local_smooth[local_peaks])
            prominences = properties.get("prominences", np.zeros(len(local_peaks), dtype=float))

        for local_peak, height, prominence in zip(local_peaks, heights, prominences, strict=True):
            full_peak = int(island[int(local_peak)])
            out.append(
                {
                    "seed_velocity_kms": float(vel_good[full_peak]),
                    "depth_below_continuum": float(max(depth_good[full_peak], height)),
                    "prominence": float(prominence),
                    "score": float(max(depth_good[full_peak], height) + prominence),
                    "trough_velocity_min_kms": float(vel_good[island[0]]),
                    "trough_velocity_max_kms": float(vel_good[island[-1]]),
                    "seed_source": "absorption_trough",
                }
            )

    out.sort(key=lambda item: item["score"], reverse=True)
    return out[: int(max_peaks)]


def smooth_depth_profile(depth: np.ndarray, step_kms: float) -> np.ndarray:
    width = int(round(45.0 / max(float(step_kms), 1.0)))
    width = max(3, min(11, width))
    if width % 2 == 0:
        width += 1
    if len(depth) < width:
        return depth.astype(float, copy=True)
    kernel = np.ones(width, dtype=float) / float(width)
    padded = np.pad(depth.astype(float), width // 2, mode="edge")
    return np.convolve(padded, kernel, mode="valid")


def robust_depth_noise(depth: np.ndarray) -> float:
    if len(depth) < 4:
        return 0.02
    diffs = np.diff(depth.astype(float))
    finite = diffs[np.isfinite(diffs)]
    if len(finite) == 0:
        return 0.02
    noise = 1.4826 * float(np.nanmedian(np.abs(finite - np.nanmedian(finite)))) / np.sqrt(2.0)
    if not np.isfinite(noise) or noise <= 0.0:
        noise = 0.02
    return float(max(noise, 0.005))


def contiguous_index_groups(indices: np.ndarray) -> list[np.ndarray]:
    if len(indices) == 0:
        return []
    return [group for group in np.split(indices, np.where(np.diff(indices) != 1)[0] + 1) if len(group)]


def merge_nearby_islands(islands: list[np.ndarray], velocity: np.ndarray, max_gap_kms: float) -> list[np.ndarray]:
    if not islands:
        return []
    merged: list[np.ndarray] = [islands[0]]
    for island in islands[1:]:
        previous = merged[-1]
        gap = float(velocity[island[0]] - velocity[previous[-1]])
        if gap <= float(max_gap_kms):
            merged[-1] = np.arange(previous[0], island[-1] + 1)
        else:
            merged.append(island)
    return merged
