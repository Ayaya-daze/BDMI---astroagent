"""Review-packet construction, continuum handling, and diagnostics."""

from importlib import import_module
from typing import Any

__all__ = [
    "C_KMS",
    "assess_absorber_hypothesis",
    "build_fit_control_hints",
    "build_plot_and_fit_data",
    "build_plot_data",
    "build_review_record",
    "build_review_record_from_window",
    "build_smooth_voigt_model_data",
    "cut_local_window",
    "human_adjudication_policy",
    "make_demo_quasar_spectrum",
    "observed_wavelength_A",
    "save_review_plot",
    "save_window_overview_plot",
    "summarize_window",
    "suggest_task_a_labels",
    "validate_spectrum_table",
    "velocity_kms",
    "write_review_packet",
]

_EXPORTS = {
    "C_KMS": "astroagent.review.continuum",
    "build_plot_data": "astroagent.review.continuum",
    "observed_wavelength_A": "astroagent.review.continuum",
    "validate_spectrum_table": "astroagent.review.continuum",
    "velocity_kms": "astroagent.review.continuum",
    "assess_absorber_hypothesis": "astroagent.review.packet",
    "build_fit_control_hints": "astroagent.review.packet",
    "build_plot_and_fit_data": "astroagent.review.packet",
    "build_review_record": "astroagent.review.packet",
    "build_review_record_from_window": "astroagent.review.packet",
    "cut_local_window": "astroagent.review.packet",
    "human_adjudication_policy": "astroagent.review.packet",
    "make_demo_quasar_spectrum": "astroagent.review.packet",
    "summarize_window": "astroagent.review.packet",
    "suggest_task_a_labels": "astroagent.review.packet",
    "write_review_packet": "astroagent.review.packet",
    "build_smooth_voigt_model_data": "astroagent.review.plot",
    "save_review_plot": "astroagent.review.plot",
    "save_window_overview_plot": "astroagent.review.plot",
}


def __getattr__(name: str) -> Any:
    module_name = _EXPORTS.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(import_module(module_name), name)
    globals()[name] = value
    return value
