"""Spectral line catalogs, peak detection, and Voigt fitting."""

from importlib import import_module
from typing import Any

__all__ = [
    "column_density_to_tau_scale",
    "combined_optical_depth_model",
    "combined_physical_flux_model",
    "component_optical_depth_model",
    "detect_absorption_peaks",
    "fit_voigt_absorption",
    "get_line_definition",
    "load_line_catalog",
    "normalized_voigt_velocity",
    "oscillator_strengths",
    "physical_component_flux_model",
    "primary_rest_wavelength_A",
    "rest_wavelengths_A",
    "transition_definitions",
    "voigt_velocity_profile",
]

_EXPORTS = {
    "get_line_definition": "astroagent.spectra.line_catalog",
    "load_line_catalog": "astroagent.spectra.line_catalog",
    "oscillator_strengths": "astroagent.spectra.line_catalog",
    "primary_rest_wavelength_A": "astroagent.spectra.line_catalog",
    "rest_wavelengths_A": "astroagent.spectra.line_catalog",
    "transition_definitions": "astroagent.spectra.line_catalog",
    "detect_absorption_peaks": "astroagent.spectra.peak_detection",
    "fit_voigt_absorption": "astroagent.spectra.voigt_fit",
    "column_density_to_tau_scale": "astroagent.spectra.voigt_model",
    "combined_optical_depth_model": "astroagent.spectra.voigt_model",
    "combined_physical_flux_model": "astroagent.spectra.voigt_model",
    "component_optical_depth_model": "astroagent.spectra.voigt_model",
    "normalized_voigt_velocity": "astroagent.spectra.voigt_model",
    "physical_component_flux_model": "astroagent.spectra.voigt_model",
    "voigt_velocity_profile": "astroagent.spectra.voigt_model",
}


def __getattr__(name: str) -> Any:
    module_name = _EXPORTS.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(import_module(module_name), name)
    globals()[name] = value
    return value
