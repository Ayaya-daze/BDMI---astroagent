from __future__ import annotations

import numpy as np
from scipy.special import voigt_profile


def voigt_velocity_profile(
    velocity_kms: np.ndarray,
    center_kms: float,
    sigma_kms: float,
    gamma_kms: float,
) -> np.ndarray:
    """Area-normalized Voigt profile in velocity units."""
    return voigt_profile(
        np.asarray(velocity_kms, dtype=float) - float(center_kms),
        max(float(sigma_kms), 1e-3),
        max(float(gamma_kms), 1e-6),
    )


def column_density_to_tau_scale(logN: float, rest_wavelength_A: float, oscillator_strength: float) -> float:
    """ABSpec N2tau relation for one transition."""
    return float((10.0 ** float(logN)) * 2.654e-15 * float(oscillator_strength) * float(rest_wavelength_A))


def physical_component_flux_model(
    velocity_kms: np.ndarray,
    *,
    logN: float,
    b_kms: float,
    center_kms: float,
    rest_wavelength_A: float,
    oscillator_strength: float,
    damping_gamma_kms: float,
    covering_fraction: float = 1.0,
) -> np.ndarray:
    """ABSpec-style single-transition Voigt absorption model.

    The first-stage fitter samples `logN`, `b_kms`, and center velocity
    directly. Covering fraction is fixed to one by default; it remains an
    explicit argument so later saturated-line work can expose it cleanly.
    """
    velocity = np.asarray(velocity_kms, dtype=float)
    sigma_kms = max(float(b_kms) / np.sqrt(2.0), 1e-3)
    tau_scale = column_density_to_tau_scale(logN, rest_wavelength_A, oscillator_strength)
    tau = tau_scale * voigt_velocity_profile(
        velocity,
        center_kms=float(center_kms),
        sigma_kms=sigma_kms,
        gamma_kms=float(damping_gamma_kms),
    )
    covered_flux = np.exp(-np.clip(tau, 0.0, 80.0))
    covering = float(np.clip(covering_fraction, 0.0, 1.0))
    return covering * covered_flux + (1.0 - covering)


def combined_physical_flux_model(
    velocity_kms: np.ndarray,
    params: np.ndarray,
    *,
    rest_wavelength_A: float,
    oscillator_strength: float,
    damping_gamma_kms: float,
) -> np.ndarray:
    """Product of all physical Voigt components in one transition frame."""
    velocity = np.asarray(velocity_kms, dtype=float)
    model = np.ones_like(velocity, dtype=float)
    n_lines = len(params) // 3
    for index in range(n_lines):
        offset = index * 3
        logN, b_kms, center_kms = params[offset : offset + 3]
        model *= physical_component_flux_model(
            velocity,
            logN=float(logN),
            b_kms=float(b_kms),
            center_kms=float(center_kms),
            rest_wavelength_A=rest_wavelength_A,
            oscillator_strength=oscillator_strength,
            damping_gamma_kms=damping_gamma_kms,
        )
    return model


def normalized_voigt_velocity(
    velocity_kms: np.ndarray,
    center_kms: float,
    sigma_kms: float,
    gamma_kms: float,
) -> np.ndarray:
    profile = voigt_velocity_profile(velocity_kms, center_kms, sigma_kms, gamma_kms)
    peak = float(np.nanmax(profile)) if len(profile) else 0.0
    if not np.isfinite(peak) or peak <= 0.0:
        return np.zeros_like(velocity_kms, dtype=float)
    return profile / peak


def combined_optical_depth_model(velocity_kms: np.ndarray, params: np.ndarray) -> np.ndarray:
    """Legacy toy-model helper kept for old fixtures only."""
    total_tau = np.zeros_like(np.asarray(velocity_kms, dtype=float), dtype=float)
    n_lines = len(params) // 4
    for index in range(n_lines):
        offset = index * 4
        log_tau0, center_kms, log_sigma_kms, log_gamma_kms = params[offset : offset + 4]
        tau0 = float(np.exp(log_tau0))
        sigma_kms = float(np.exp(log_sigma_kms))
        gamma_kms = float(np.exp(log_gamma_kms))
        total_tau += tau0 * normalized_voigt_velocity(velocity_kms, center_kms, sigma_kms, gamma_kms)
    return np.exp(-np.clip(total_tau, 0.0, 80.0))


def component_optical_depth_model(
    velocity_kms: np.ndarray,
    *,
    tau0: float,
    center_kms: float,
    sigma_kms: float,
    gamma_kms: float,
) -> np.ndarray:
    """Legacy toy-model helper kept for old fixtures only."""
    tau = float(tau0) * normalized_voigt_velocity(velocity_kms, center_kms, sigma_kms, gamma_kms)
    return np.exp(-np.clip(tau, 0.0, 80.0))

