import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from astroagent.review.packet import (
    assess_absorber_hypothesis,
    build_fit_control_hints,
    build_plot_data,
    build_plot_and_fit_data,
    build_review_record,
    make_demo_quasar_spectrum,
    observed_wavelength_A,
    write_review_packet,
)
from astroagent.review import plot as review_plot
from astroagent.spectra import voigt_fit
from astroagent.spectra.voigt_model import apply_lsf_matrix, physical_component_flux_model


class ReviewPacketTest(unittest.TestCase):
    def test_observed_wavelength_scales_with_redshift(self):
        self.assertEqual(observed_wavelength_A(1548.204, 2.6), 1548.204 * 3.6)

    def test_build_review_record_for_demo_civ_doublet(self):
        spectrum = make_demo_quasar_spectrum(z_sys=2.6, line_id="CIV_doublet")
        record, window = build_review_record(
            spectrum=spectrum,
            line_id="CIV_doublet",
            z_sys=2.6,
            sample_id="unit_demo",
            source={"kind": "unit_test"},
        )

        self.assertIsInstance(window, pd.DataFrame)
        self.assertGreater(len(window), 50)
        self.assertEqual(record["sample_id"], "unit_demo")
        self.assertEqual(record["input"]["line_id"], "CIV_doublet")
        self.assertEqual(record["input"]["line_family_context"]["ion"], "C IV")
        self.assertEqual(record["input"]["line_family_context"]["multiplet_type"], "doublet")
        self.assertIn("outlier_policy", record["input"]["line_family_context"]["soft_background"])
        self.assertEqual(record["input"]["transitions"][0]["partner_line_id"], "CIV_1550")
        self.assertGreater(record["window_summary"]["n_good_pixels"], 0)
        self.assertTrue(record["absorber_hypothesis_check"]["absorption_signal_present"])
        self.assertTrue(record["human_adjudication"]["required"])
        self.assertIn("velocity_exceeds_virial_expectation", record["human_adjudication"]["escalation_flags"])
        self.assertEqual(record["fit_results"][0]["task"], "voigt_profile_fit")
        self.assertTrue(record["fit_results"][0]["success"])
        self.assertEqual(record["fit_results"][0]["fit_type"], "transition_frame_peak_voigt")
        self.assertEqual(record["fit_results"][0]["fit_method"], "ultranest_physical_transition_frame")
        self.assertEqual(record["fit_results"][0]["n_transition_frames"], 2)
        self.assertEqual(record["fit_results"][0]["source_work_window_kms"], [-400.0, 400.0])
        self.assertIn("chi2", record["fit_results"][0])
        self.assertIn("reduced_chi2", record["fit_results"][0])
        self.assertIn("chi2", record["fit_results"][0]["source_work_window_metrics"])
        self.assertIn("reduced_chi2", record["fit_results"][0]["source_work_window_metrics"])
        self.assertIn("fit_rms", record["fit_results"][0]["source_work_window_metrics"])
        self.assertIn("transition_frames", record["fit_results"][0])
        components = record["fit_results"][0]["components"]
        self.assertGreaterEqual(len(components), 1)
        self.assertTrue(any(component["logN"] > 10.0 for component in components))
        self.assertTrue(any(component["b_kms"] > 0.0 for component in components))
        self.assertEqual(record["fit_results"][0]["metadata_use"], "line_id and z_sys define transition frames only; peak centers become posterior priors")
        self.assertNotIn("center_selection", record["fit_results"][0])
        self.assertNotIn("peak_diagnostics", record["fit_results"][0])
        for frame in record["fit_results"][0]["transition_frames"]:
            self.assertIn("transition_line_id", frame)
            self.assertEqual(frame["ion"], "C IV")
            self.assertIn("family_context", frame)
            self.assertIn("velocity_frame", frame)
            self.assertGreaterEqual(frame["velocity_frame"]["n_good_pixels"], 1)
            self.assertIn("peaks", frame)
        self.assertEqual(record["plot_data"]["fit_type"], "egent_iterative_continuum")
        self.assertEqual(record["plot_data"]["method"], "rolling_upper_envelope_pchip")
        self.assertIn("normalized_flux", record["plot_data"]["columns"])
        self.assertIn("is_continuum_pixel", record["plot_data"]["columns"])
        self.assertIn("voigt_model", record["plot_data"]["columns"])
        self.assertIn("is_voigt_fit_pixel", record["plot_data"]["columns"])
        self.assertNotIn("voigt_best_model", record["plot_data"]["columns"])
        self.assertNotIn("is_voigt_best_fit_pixel", record["plot_data"]["columns"])
        self.assertEqual(record["plot_data"]["image_space"], "per_transition_velocity_frame")
        self.assertIn("curve_kind", record["plot_data"]["model_columns"])
        self.assertNotIn("covering_fraction", record["plot_data"]["model_columns"])
        self.assertIn("logN", record["plot_data"]["model_columns"])
        self.assertIn("b_kms", record["plot_data"]["model_columns"])

        suggestion = record["task_a_rule_suggestion"]
        self.assertEqual(suggestion["task"], "local_mask_continuum")
        self.assertTrue(suggestion["analysis_mask_intervals_A"])
        self.assertGreaterEqual(len(suggestion["continuum_anchor_points_A"]), 3)

    def test_absorber_check_rejects_flat_window(self):
        spectrum = make_demo_quasar_spectrum(z_sys=2.6, line_id="CIV_doublet")
        spectrum["flux"] = 1.0
        record, window = build_review_record(
            spectrum=spectrum,
            line_id="CIV_doublet",
            z_sys=2.6,
            sample_id="flat_demo",
            source={"kind": "unit_test"},
        )

        check = assess_absorber_hypothesis(window, record["input"])
        self.assertFalse(check["absorption_signal_present"])
        self.assertEqual(check["status"], "weak_or_absent")

    def test_fit_control_hints_keep_boundary_residuals_below_core_priority(self):
        velocity = np.array([-250.0, -245.0, 0.0, 388.0, 392.0, 406.0, 410.0])
        plot_window = pd.DataFrame(
            {
                "voigt_transition_id": ["HI_LYA"] * len(velocity),
                "transition_velocity_kms": velocity,
                "voigt_residual_sigma": [-5.0, -6.0, 0.0, -7.0, -8.0, -9.0, -10.0],
                "is_voigt_fit_pixel": [1] * len(velocity),
            }
        )
        fit_summary = {"transition_frames": [{"transition_line_id": "HI_LYA"}]}

        hints = build_fit_control_hints(plot_window, fit_summary)

        kinds = [hint["kind"] for hint in hints["hints"]]
        self.assertEqual(hints["priority_order"][0], "source_work_window_residual")
        self.assertIn("source_work_window_residual", kinds)
        self.assertIn("source_boundary_residual", kinds)

    def test_plot_data_and_files_are_written(self):
        spectrum = make_demo_quasar_spectrum(z_sys=2.6, line_id="CIV_doublet")
        record, window = build_review_record(
            spectrum=spectrum,
            line_id="CIV_doublet",
            z_sys=2.6,
            sample_id="plot_demo",
            source={"kind": "unit_test"},
        )
        plot_window, summary = build_plot_data(window)

        self.assertIn("continuum_model", plot_window.columns)
        self.assertIn("normalized_flux", plot_window.columns)
        self.assertEqual(summary["fit_type"], "egent_iterative_continuum")
        self.assertEqual(len(plot_window), len(window))
        self.assertIn("is_continuum_pixel", plot_window.columns)

        with tempfile.TemporaryDirectory(prefix="astroagent_review_packet_") as output_dir:
            paths = write_review_packet(record, window, output_dir)
            self.assertTrue(paths["plot_csv"].exists())
            self.assertTrue(paths["model_csv"].exists())
            self.assertTrue(paths["overview_png"].exists())
            self.assertTrue(paths["plot_png"].exists())

            plot_csv = pd.read_csv(paths["plot_csv"])
            model_csv = pd.read_csv(paths["model_csv"])
        self.assertIn("continuum_model", plot_csv.columns)
        self.assertIn("normalized_flux", plot_csv.columns)
        self.assertIn("voigt_model", plot_csv.columns)
        self.assertIn("voigt_residual_sigma", plot_csv.columns)
        self.assertIn("is_voigt_fit_pixel", plot_csv.columns)
        self.assertNotIn("voigt_best_model", plot_csv.columns)
        self.assertNotIn("is_voigt_best_fit_pixel", plot_csv.columns)
        self.assertIn("voigt_transition_id", plot_csv.columns)
        self.assertIn("transition_velocity_kms", plot_csv.columns)
        self.assertIn("smooth_voigt_model", model_csv.columns)
        self.assertIn("curve_kind", model_csv.columns)
        self.assertIn("component_index", model_csv.columns)
        self.assertNotIn("covering_fraction", model_csv.columns)
        self.assertIn("logN", model_csv.columns)
        self.assertIn("b_kms", model_csv.columns)
        self.assertIn("damping_gamma_kms", model_csv.columns)
        self.assertTrue({"component", "combined"}.issubset(set(model_csv["curve_kind"])))

    def test_write_review_packet_is_repeatable_and_keeps_json_model_consistent(self):
        spectrum = make_demo_quasar_spectrum(z_sys=2.6, line_id="CIV_doublet")
        record, window = build_review_record(
            spectrum=spectrum,
            line_id="CIV_doublet",
            z_sys=2.6,
            sample_id="repeatable_packet",
            source={"kind": "unit_test"},
        )

        with tempfile.TemporaryDirectory(prefix="astroagent_review_repeatable_") as output_dir:
            write_review_packet(record, window, output_dir)
            self.assertIn("_plot_window", record)
            self.assertIn("_fit_summary", record)

            paths = write_review_packet(record, window, output_dir)
            saved = json.loads(paths["json"].read_text(encoding="utf-8"))
            model_csv = pd.read_csv(paths["model_csv"])
            reloaded_paths = write_review_packet(saved, window, output_dir)
            reloaded_model_csv = pd.read_csv(reloaded_paths["model_csv"])

        json_centers = sorted(
            round(float(component["center_velocity_kms"]), 3)
            for component in saved["fit_results"][0]["components"]
        )
        model_centers = sorted(
            round(float(center), 3)
            for center in model_csv.loc[model_csv["curve_kind"] == "component", "center_velocity_kms"].dropna().unique()
        )
        self.assertEqual(model_centers, json_centers)
        reloaded_model_centers = sorted(
            round(float(center), 3)
            for center in reloaded_model_csv.loc[
                reloaded_model_csv["curve_kind"] == "component",
                "center_velocity_kms",
            ].dropna().unique()
        )
        self.assertEqual(reloaded_model_centers, json_centers)
        self.assertNotIn("_plot_window", saved)
        self.assertNotIn("_fit_summary", saved)

    def test_voigt_fit_recovers_demo_absorption_components(self):
        spectrum = make_demo_quasar_spectrum(z_sys=2.6, line_id="CIV_doublet")
        record, window = build_review_record(
            spectrum=spectrum,
            line_id="CIV_doublet",
            z_sys=2.6,
            sample_id="voigt_demo",
            source={"kind": "unit_test"},
        )
        plot_window, _, fit_summary = build_plot_and_fit_data(window, record["input"])

        self.assertTrue(fit_summary["success"])
        data_components = [item for item in fit_summary["components"] if item["kind"] == "transition_frame_peak"]
        self.assertGreaterEqual(len(data_components), 1)
        self.assertGreater(fit_summary["observed_equivalent_width_mA"], 100.0)
        self.assertEqual(fit_summary["metadata_use"], "line_id and z_sys define transition frames only; peak centers become posterior priors")
        self.assertEqual(fit_summary["n_transition_frames"], 2)
        self.assertTrue(plot_window["voigt_model"].notna().any())
        self.assertGreater(plot_window["is_voigt_fit_pixel"].sum(), 0)
        self.assertLess(plot_window["is_voigt_fit_pixel"].sum(), len(plot_window))
        self.assertIn("transition_velocity_kms", plot_window.columns)
        for component in data_components:
            self.assertIn("center_wavelength_A", component)
            self.assertIn("center_velocity_kms", component)
            self.assertEqual(component["fit_method"], "ultranest_physical_transition_frame")
            self.assertIn("logN", component)
            self.assertIn("b_kms", component)
            self.assertIn("parameter_intervals", component)
            self.assertIn("logN", component["parameter_intervals"])
            self.assertIn("center_prior", component)

    def test_default_fit_reports_lsf_unavailable_without_changing_model_columns(self):
        spectrum = make_demo_quasar_spectrum(z_sys=2.6, line_id="CIV_doublet")
        record, window = build_review_record(
            spectrum=spectrum,
            line_id="CIV_doublet",
            z_sys=2.6,
            sample_id="no_lsf_demo",
            source={"kind": "unit_test"},
        )
        plot_window, _, fit_summary = build_plot_and_fit_data(window, record["input"])

        self.assertFalse(fit_summary["lsf"]["available"])
        self.assertFalse(fit_summary["instrument_lsf_applied"])
        self.assertIn("voigt_lsf_model", plot_window.columns)
        self.assertFalse(plot_window["voigt_lsf_model"].notna().any())
        for frame in fit_summary["transition_frames"]:
            self.assertFalse(frame["lsf"]["available"])
            self.assertFalse(frame["lsf_diagnostic"]["available"])

    def test_lsf_matrix_adds_parallel_diagnostic_without_replacing_intrinsic_fit(self):
        z_sys = 0.0
        rest_A = 2796.352
        wavelength = np.arange(rest_A - 2.0, rest_A + 2.0, 0.08)
        velocity = (wavelength / rest_A - 1.0) * 299792.458
        flux = 1.0 - 0.35 * np.exp(-0.5 * (velocity / 16.0) ** 2)
        spectrum = pd.DataFrame(
            {
                "wavelength": wavelength,
                "flux": flux,
                "ivar": np.full_like(wavelength, 900.0),
                "pipeline_mask": np.zeros_like(wavelength, dtype=int),
            }
        )
        record, window = build_review_record(
            spectrum=spectrum,
            line_id="MGII_2796",
            z_sys=z_sys,
            sample_id="lsf_diagnostic_demo",
            source={"kind": "unit_test"},
            half_width_kms=220.0,
        )
        frame_size = int(len(window))
        offsets = np.arange(frame_size)
        sigma_pix = 1.4
        lsf_matrix = np.exp(-0.5 * ((offsets[:, None] - offsets[None, :]) / sigma_pix) ** 2)
        metadata = {
            **record["input"],
            "lsf_matrices_by_transition": {"MGII_2796": lsf_matrix.tolist()},
            "lsf_source": "unit_resolution_matrix",
        }

        plot_window, _, fit_summary = build_plot_and_fit_data(window, metadata)
        model_data = review_plot.build_smooth_voigt_model_data(fit_summary)

        self.assertTrue(fit_summary["lsf"]["available"])
        self.assertTrue(fit_summary["instrument_lsf_applied"])
        self.assertFalse(fit_summary["instrument_lsf_applied_to_fit_likelihood"])
        self.assertEqual(fit_summary["lsf"]["transition_frames_with_lsf"], ["MGII_2796"])
        frame = fit_summary["transition_frames"][0]
        self.assertTrue(frame["lsf"]["available"])
        self.assertFalse(frame["lsf"]["applied_to_fit_likelihood"])
        self.assertTrue(frame["lsf_diagnostic"]["available"])
        self.assertIn("intrinsic_fit_window_metrics", frame["lsf_diagnostic"])
        self.assertIn("lsf_fit_window_metrics", frame["lsf_diagnostic"])
        self.assertGreater(len(frame["residual_samples"]), 0)
        self.assertEqual(len(frame["residual_samples"]), len(frame["lsf_diagnostic"]["residual_samples"]))
        self.assertTrue(plot_window["voigt_model"].notna().any())
        self.assertTrue(plot_window["voigt_lsf_model"].notna().any())
        self.assertTrue(model_data["smooth_lsf_model"].notna().any())
        self.assertFalse(
            np.allclose(
                plot_window["voigt_model"].dropna().to_numpy(dtype=float),
                plot_window["voigt_lsf_model"].dropna().to_numpy(dtype=float),
            )
        )

    def test_apply_lsf_matrix_preserves_continuum_and_broadens_narrow_absorption(self):
        velocity = np.linspace(-80.0, 80.0, 81)
        intrinsic = physical_component_flux_model(
            velocity,
            logN=13.2,
            b_kms=8.0,
            center_kms=0.0,
            rest_wavelength_A=2796.352,
            oscillator_strength=0.6123,
            damping_gamma_kms=0.001,
        )
        pixels = np.arange(len(velocity))
        matrix = np.exp(-0.5 * ((pixels[:, None] - pixels[None, :]) / 2.0) ** 2)

        convolved = apply_lsf_matrix(intrinsic, matrix)

        self.assertTrue(np.allclose(apply_lsf_matrix(np.ones_like(intrinsic), matrix), 1.0))
        self.assertGreater(float(np.nanmin(convolved)), float(np.nanmin(intrinsic)))
        self.assertLess(float(convolved[30]), float(intrinsic[30]))

    def test_ultranest_posterior_success_overrides_initializer_failure(self):
        class FailedInitializer:
            success = False
            message = "maximum function evaluations exceeded"

            def __init__(self, x):
                self.x = np.asarray(x, dtype=float)

        def fake_least_squares(_residual, p0, bounds, max_nfev):
            return FailedInitializer(p0)

        def fake_posterior(*_args, **_kwargs):
            return {
                "paramnames": ["logN_0", "b_kms_0", "center_velocity_kms_0"],
                "weighted_samples": {
                    "points": np.asarray(
                        [
                            [13.0, 28.0, -2.0],
                            [13.1, 30.0, 0.0],
                            [13.2, 32.0, 2.0],
                        ],
                        dtype=float,
                    ),
                    "weights": np.asarray([1.0, 2.0, 1.0], dtype=float),
                },
                "posterior": {
                    "mean": [13.1, 30.0, 0.0],
                    "stdev": [0.1, 2.0, 2.0],
                },
                "logz": -12.0,
                "logzerr": 0.1,
                "ncall": 24,
            }

        old_least_squares = voigt_fit.least_squares
        old_posterior = voigt_fit._run_ultranest_physical_posterior
        voigt_fit.least_squares = fake_least_squares
        voigt_fit._run_ultranest_physical_posterior = fake_posterior
        try:
            velocity = np.linspace(-120.0, 120.0, 49)
            flux = 1.0 - 0.25 * np.exp(-0.5 * (velocity / 25.0) ** 2)
            ivar = np.full_like(velocity, 900.0)
            seeds = [{"seed_velocity_kms": 0.0, "depth_below_continuum": 0.25, "score": 1.0}]

            fitted, model, _lsf_model, fit_mask = voigt_fit._fit_peak_group_once(
                velocity,
                flux,
                ivar,
                np.ones_like(velocity, dtype=bool),
                seeds,
                local_fit_half_width_kms=180.0,
                center_shift_limit_kms=120.0,
                rest_wavelength_A=2796.352,
                oscillator_strength=0.6123,
                damping_gamma_kms=0.001,
            )
        finally:
            voigt_fit.least_squares = old_least_squares
            voigt_fit._run_ultranest_physical_posterior = old_posterior

        self.assertTrue(fitted[0]["fit_success"])
        self.assertEqual(fitted[0]["fit_backend"], "ultranest")
        self.assertEqual(fitted[0]["parameter_estimator"], "posterior_median")
        self.assertNotIn("component_parameter_posterior_degenerate", fitted[0].get("diagnostic_flags", []))
        self.assertTrue(np.isfinite(model[fit_mask]).all())

    def test_ultranest_posterior_still_runs_when_initializer_raises(self):
        def raising_least_squares(_residual, _p0, _bounds, _max_nfev):
            raise RuntimeError("initializer failed")

        def fake_posterior(*_args, **_kwargs):
            return {
                "paramnames": ["logN_0", "b_kms_0", "center_velocity_kms_0"],
                "weighted_samples": {
                    "points": np.asarray(
                        [
                            [13.0, 28.0, -2.0],
                            [13.1, 30.0, 0.0],
                            [13.2, 32.0, 2.0],
                        ],
                        dtype=float,
                    ),
                    "weights": np.asarray([1.0, 2.0, 1.0], dtype=float),
                },
                "posterior": {"mean": [13.1, 30.0, 0.0], "stdev": [0.1, 2.0, 2.0]},
                "logz": -12.0,
                "logzerr": 0.1,
                "ncall": 24,
            }

        old_least_squares = voigt_fit.least_squares
        old_posterior = voigt_fit._run_ultranest_physical_posterior
        voigt_fit.least_squares = raising_least_squares
        voigt_fit._run_ultranest_physical_posterior = fake_posterior
        try:
            velocity = np.linspace(-120.0, 120.0, 49)
            flux = 1.0 - 0.25 * np.exp(-0.5 * (velocity / 25.0) ** 2)
            ivar = np.full_like(velocity, 900.0)
            seeds = [{"seed_velocity_kms": 0.0, "depth_below_continuum": 0.25, "score": 1.0}]

            fitted, model, _lsf_model, fit_mask = voigt_fit._fit_peak_group_once(
                velocity,
                flux,
                ivar,
                np.ones_like(velocity, dtype=bool),
                seeds,
                local_fit_half_width_kms=180.0,
                center_shift_limit_kms=120.0,
                rest_wavelength_A=2796.352,
                oscillator_strength=0.6123,
                damping_gamma_kms=0.001,
            )
        finally:
            voigt_fit.least_squares = old_least_squares
            voigt_fit._run_ultranest_physical_posterior = old_posterior

        self.assertTrue(fitted[0]["fit_success"])
        self.assertEqual(fitted[0]["fit_backend"], "ultranest")
        self.assertEqual(fitted[0]["parameter_estimator"], "posterior_median")
        self.assertTrue(np.isfinite(model[fit_mask]).all())

    def test_incomplete_posterior_does_not_fall_back_to_initializer_parameters(self):
        def fake_posterior(*_args, **_kwargs):
            return {
                "paramnames": ["logN_0", "b_kms_0"],
                "weighted_samples": {
                    "points": np.asarray([[13.0, 28.0], [13.1, 30.0], [13.2, 32.0]], dtype=float),
                    "weights": np.asarray([1.0, 2.0, 1.0], dtype=float),
                },
                "posterior": {"mean": [13.1, 30.0], "stdev": [0.1, 2.0]},
                "logz": -12.0,
                "logzerr": 0.1,
                "ncall": 24,
            }

        old_posterior = voigt_fit._run_ultranest_physical_posterior
        voigt_fit._run_ultranest_physical_posterior = fake_posterior
        try:
            velocity = np.linspace(-120.0, 120.0, 49)
            flux = 1.0 - 0.25 * np.exp(-0.5 * (velocity / 25.0) ** 2)
            ivar = np.full_like(velocity, 900.0)
            seeds = [{"seed_velocity_kms": 0.0, "depth_below_continuum": 0.25, "score": 1.0}]

            fitted, model, _lsf_model, _fit_mask = voigt_fit._fit_peak_group_once(
                velocity,
                flux,
                ivar,
                np.ones_like(velocity, dtype=bool),
                seeds,
                local_fit_half_width_kms=180.0,
                center_shift_limit_kms=120.0,
                rest_wavelength_A=2796.352,
                oscillator_strength=0.6123,
                damping_gamma_kms=0.001,
            )
        finally:
            voigt_fit._run_ultranest_physical_posterior = old_posterior

        self.assertFalse(fitted[0]["fit_success"])
        self.assertEqual(fitted[0]["parameter_estimator"], "none_posterior_unavailable")
        self.assertIn("bayesian_posterior_unavailable", fitted[0]["diagnostic_flags"])
        self.assertFalse(np.isfinite(model).any())

    def test_component_interval_missing_posterior_values_are_not_filled_from_fit(self):
        interval = voigt_fit._component_interval({"backend": "ultranest", "posterior": {}}, "logN_0")

        self.assertTrue(np.isnan(interval["q16"]))
        self.assertTrue(np.isnan(interval["median"]))
        self.assertTrue(np.isnan(interval["q84"]))

    def test_posterior_band_does_not_invent_uncertainty_for_degenerate_intervals(self):
        component = {
            "component_index": 0,
            "fit_backend": "ultranest",
            "parameter_estimator": "posterior_median",
            "fit_success": True,
            "transition_line_id": "MGII_2796",
            "rest_wavelength_A": 2796.352,
            "oscillator_strength": 0.6123,
            "damping_gamma_kms": 0.001,
            "parameter_intervals": {
                "logN": {"q16": 13.0, "median": 13.0, "q84": 13.0},
                "b_kms": {"q16": 30.0, "median": 30.0, "q84": 30.0},
                "center_velocity_kms": {"q16": 0.0, "median": 0.0, "q84": 0.0},
            },
            "diagnostic_flags": [],
        }
        velocity = np.linspace(-80.0, 80.0, 21)

        samples = review_plot._component_posterior_band(component, velocity, n_samples=8)

        self.assertIsNotNone(samples)
        self.assertTrue(np.allclose(samples, samples[0]))

    def test_doublet_velocity_frames_keep_independent_residual_samples(self):
        spectrum = make_demo_quasar_spectrum(z_sys=2.6, line_id="CIV_doublet")
        record, window = build_review_record(
            spectrum=spectrum,
            line_id="CIV_doublet",
            z_sys=2.6,
            sample_id="frame_residual_demo",
            source={"kind": "unit_test"},
        )
        _, _, fit_summary = build_plot_and_fit_data(window, record["input"])

        frames = fit_summary["transition_frames"]
        self.assertEqual(len(frames), 2)
        residual_counts = [len(frame.get("residual_samples", [])) for frame in frames]
        self.assertTrue(all(count > 0 for count in residual_counts))
        self.assertEqual(fit_summary["n_fit_pixels"], sum(residual_counts))
        self.assertAlmostEqual(fit_summary["reduced_chi2"], fit_summary["fit_rms"] ** 2)
        self.assertAlmostEqual(fit_summary["chi2"], fit_summary["reduced_chi2"] * fit_summary["n_fit_pixels"])
        for frame in frames:
            samples = frame["residual_samples"]
            velocities = [sample["velocity_kms"] for sample in samples]
            self.assertGreaterEqual(min(velocities), -fit_summary["transition_half_width_kms"])
            self.assertLessEqual(max(velocities), fit_summary["transition_half_width_kms"])
            self.assertTrue(all("residual_sigma" in sample for sample in samples))

    def test_voigt_fit_handles_multiple_peaks_in_one_transition_frame_simultaneously(self):
        z_sys = 0.0
        rest_A = 2796.352
        wavelength = np.arange(rest_A - 7.0, rest_A + 7.0, 0.08)
        velocity = (wavelength / rest_A - 1.0) * 299792.458
        continuum = np.ones_like(wavelength)
        flux = continuum.copy()
        for center_v, depth, sigma_v in [(-120.0, 0.35, 28.0), (135.0, 0.24, 35.0)]:
            flux -= depth * np.exp(-0.5 * ((velocity - center_v) / sigma_v) ** 2)
        spectrum = pd.DataFrame(
            {
                "wavelength": wavelength,
                "flux": flux,
                "ivar": np.full_like(wavelength, 900.0),
                "pipeline_mask": np.zeros_like(wavelength, dtype=int),
            }
        )
        record, window = build_review_record(
            spectrum=spectrum,
            line_id="MGII_2796",
            z_sys=z_sys,
            sample_id="two_peak_single_transition",
            source={"kind": "unit_test"},
            half_width_kms=700.0,
        )
        plot_window, _, fit_summary = build_plot_and_fit_data(window, record["input"])

        frame = fit_summary["transition_frames"][0]
        successful = [peak for peak in frame["peaks"] if peak["fit_success"]]
        centers = sorted(float(peak["center_velocity_kms"]) for peak in successful)
        self.assertEqual(frame["transition_line_id"], "MGII_2796")
        self.assertEqual(frame["simultaneous_fit_window"]["model"], "all retained components in this transition frame are fit together with physical logN/b/v parameters")
        self.assertGreaterEqual(len(successful), 2)
        self.assertTrue(any(abs(center + 120.0) < 55.0 for center in centers))
        self.assertTrue(any(abs(center - 135.0) < 55.0 for center in centers))
        self.assertTrue(plot_window["voigt_model"].notna().any())
        self.assertLessEqual(max(abs(center) for center in centers), 700.0)

    def test_voigt_fit_merges_nearly_duplicate_explicit_sources(self):
        z_sys = 0.0
        rest_A = 2796.352
        wavelength = np.arange(rest_A - 3.0, rest_A + 3.0, 0.08)
        velocity = (wavelength / rest_A - 1.0) * 299792.458
        flux = 1.0 - 0.35 * np.exp(-0.5 * (velocity / 18.0) ** 2)
        spectrum = pd.DataFrame(
            {
                "wavelength": wavelength,
                "flux": flux,
                "ivar": np.full_like(wavelength, 900.0),
                "pipeline_mask": np.zeros_like(wavelength, dtype=int),
            }
        )
        record, window = build_review_record(
            spectrum=spectrum,
            line_id="MGII_2796",
            z_sys=z_sys,
            sample_id="duplicate_seed_single_transition",
            source={"kind": "unit_test"},
            half_width_kms=320.0,
        )
        fit_control = {
            "source_seeds": [
                {"transition_line_id": "MGII_2796", "center_velocity_kms": -4.0, "explicit_fit_control_source": True},
                {"transition_line_id": "MGII_2796", "center_velocity_kms": 5.0, "explicit_fit_control_source": True},
            ],
            "fit_mask_intervals": [],
            "fit_windows": {},
            "removed_sources": [],
        }
        _, _, fit_summary = build_plot_and_fit_data(window, record["input"], fit_control=fit_control)

        centers = sorted(
            float(peak["center_velocity_kms"])
            for peak in fit_summary["components"]
            if peak.get("fit_success") and abs(float(peak["center_velocity_kms"])) < 20.0
        )
        self.assertEqual(len(centers), 1)

    def test_voigt_fit_flags_saturated_redundant_explicit_sources(self):
        z_sys = 0.0
        rest_A = 2796.352
        wavelength = np.arange(rest_A - 4.0, rest_A + 4.0, 0.08)
        velocity = (wavelength / rest_A - 1.0) * 299792.458
        flux = 1.0 - 0.96 * np.exp(-0.5 * (velocity / 55.0) ** 2)
        spectrum = pd.DataFrame(
            {
                "wavelength": wavelength,
                "flux": np.clip(flux, 0.01, None),
                "ivar": np.full_like(wavelength, 900.0),
                "pipeline_mask": np.zeros_like(wavelength, dtype=int),
            }
        )
        record, window = build_review_record(
            spectrum=spectrum,
            line_id="MGII_2796",
            z_sys=z_sys,
            sample_id="saturated_redundant_sources",
            source={"kind": "unit_test"},
            half_width_kms=420.0,
        )
        fit_control = {
            "source_detection_mode": "controlled",
            "source_seeds": [
                {
                    "transition_line_id": "MGII_2796",
                    "center_velocity_kms": -60.0,
                    "explicit_fit_control_source": True,
                    "depth_below_continuum": 0.85,
                    "b_kms": 70.0,
                    "center_prior_half_width_kms": 40.0,
                    "logN_lower": 15.0,
                    "logN_upper": 18.8,
                },
                {
                    "transition_line_id": "MGII_2796",
                    "center_velocity_kms": 60.0,
                    "explicit_fit_control_source": True,
                    "depth_below_continuum": 0.85,
                    "b_kms": 70.0,
                    "center_prior_half_width_kms": 40.0,
                    "logN_lower": 15.0,
                    "logN_upper": 18.8,
                },
            ],
            "fit_mask_intervals": [],
            "fit_windows": {},
            "removed_sources": [],
        }
        _, _, fit_summary = build_plot_and_fit_data(window, record["input"], fit_control=fit_control)

        flags = [
            flag
            for component in fit_summary["components"]
            for flag in component.get("diagnostic_flags", [])
        ]
        self.assertIn("saturated_redundant_component", flags)
        self.assertIn("saturated_redundant_component", fit_summary["agent_review"]["reasons"])

    def test_voigt_fit_infers_nonzero_absorber_velocity(self):
        z_catalog = 2.6
        z_abs = 2.6015
        expected_velocity = ((1.0 + z_abs) / (1.0 + z_catalog) - 1.0) * 299792.458
        spectrum = make_demo_quasar_spectrum(z_sys=z_abs, line_id="CIV_doublet")
        record, window = build_review_record(
            spectrum=spectrum,
            line_id="CIV_doublet",
            z_sys=z_catalog,
            sample_id="velocity_offset_demo",
            source={"kind": "unit_test"},
        )
        _, _, fit_summary = build_plot_and_fit_data(window, record["input"])

        self.assertTrue(fit_summary["success"])
        self.assertEqual(fit_summary["fit_type"], "transition_frame_peak_voigt")
        self.assertNotIn("center_selection", fit_summary)
        peak_velocities = [
            peak["center_velocity_kms"]
            for frame in fit_summary["transition_frames"]
            for peak in frame.get("peaks", [])
            if peak.get("fit_success")
        ]
        self.assertTrue(any(abs(velocity - expected_velocity) < 80.0 for velocity in peak_velocities))

    def test_voigt_fit_keeps_far_absorption_as_context_diagnostic_not_source_fit(self):
        z_catalog = 2.6
        z_far = 2.6066
        spectrum = make_demo_quasar_spectrum(z_sys=z_far, line_id="CIV_doublet")
        record, window = build_review_record(
            spectrum=spectrum,
            line_id="CIV_doublet",
            z_sys=z_catalog,
            sample_id="far_velocity_demo",
            source={"kind": "unit_test"},
        )
        _, _, fit_summary = build_plot_and_fit_data(window, record["input"])

        self.assertTrue(fit_summary["success"])
        self.assertNotIn("velocity_prior_scale_kms", fit_summary)
        self.assertNotIn("hard_velocity_window", fit_summary)
        peak_velocities = [
            peak["center_velocity_kms"]
            for frame in fit_summary["transition_frames"]
            for peak in frame.get("peaks", [])
            if peak.get("fit_success")
        ]
        self.assertTrue(all(abs(velocity) <= 400.0 for velocity in peak_velocities))
        diagnostic_context_residuals = [
            sample
            for frame in fit_summary["transition_frames"]
            for sample in frame.get("diagnostic_residual_samples", [])
            if abs(float(sample.get("velocity_kms", 0.0))) > 500.0
            and abs(float(sample.get("residual_sigma", 0.0))) > 3.0
            and sample.get("used_in_fit") is False
        ]
        self.assertTrue(diagnostic_context_residuals)
        self.assertTrue(fit_summary["agent_review"]["required"])
        self.assertIn("no_absorption_peak_detected_in_transition_frame", fit_summary["agent_review"]["reasons"])

    def test_continuum_does_not_bias_high_on_demo_spectrum(self):
        spectrum = make_demo_quasar_spectrum(z_sys=2.6, line_id="CIV_doublet")
        record, window = build_review_record(
            spectrum=spectrum,
            line_id="CIV_doublet",
            z_sys=2.6,
            sample_id="continuum_demo",
            source={"kind": "unit_test"},
        )
        plot_window, summary = build_plot_data(window)
        good = plot_window["is_good_pixel"].astype(bool)
        ratio = (plot_window.loc[good, "flux"] / plot_window.loc[good, "continuum_model"]).median()
        self.assertLess(ratio, 1.05)
        self.assertGreater(ratio, 0.85)
        self.assertIn("anchor_strategy", summary)
        self.assertEqual(summary["anchor_strategy"], "rolling_upper_envelope")

    def test_continuum_excludes_data_driven_absorption_troughs(self):
        spectrum = make_demo_quasar_spectrum(z_sys=2.6, line_id="CIV_doublet")
        record, window = build_review_record(
            spectrum=spectrum,
            line_id="CIV_doublet",
            z_sys=2.6,
            sample_id="exclude_demo",
            source={"kind": "unit_test"},
        )
        plot_window, summary = build_plot_data(window, record["input"])
        excluded = summary["continuum_exclusion_intervals_A"]
        self.assertGreaterEqual(len(excluded), 1)
        self.assertGreater(int(plot_window["is_continuum_excluded_pixel"].sum()), 0)
        for start, end in excluded:
            sub = plot_window[(plot_window["wavelength"] >= start) & (plot_window["wavelength"] <= end)]
            self.assertEqual(int(sub["is_continuum_pixel"].sum()), 0)

    def test_mgii_continuum_exclusion_does_not_mask_whole_doublet_gap(self):
        spectrum = make_demo_quasar_spectrum(z_sys=1.68, line_id="MGII_doublet")
        record, window = build_review_record(
            spectrum=spectrum,
            line_id="MGII_doublet",
            z_sys=1.68,
            sample_id="mgii_exclusion_width_demo",
            source={"kind": "unit_test"},
        )
        plot_window, summary = build_plot_data(window, record["input"])
        excluded_fraction = float(plot_window["is_continuum_excluded_pixel"].mean())
        intervals = summary["continuum_exclusion_intervals_A"]

        self.assertLess(excluded_fraction, 0.45)
        self.assertLess(intervals[0][1], intervals[1][0])
        self.assertEqual(summary["continuum_exclusion_kms"], 250.0)

    def test_fit_control_can_add_and_remove_continuum_anchor_nodes(self):
        wavelength = np.linspace(5000.0, 5100.0, 401)
        x = (wavelength - 5050.0) / 50.0
        true_continuum = 1.0 + 0.16 * x + 0.08 * x**2
        flux = true_continuum.copy()
        flux -= 0.28 * np.exp(-0.5 * ((wavelength - 5050.0) / 1.6) ** 2)
        spectrum = pd.DataFrame(
            {
                "wavelength": wavelength,
                "flux": flux,
                "ivar": np.full_like(wavelength, 2500.0),
                "pipeline_mask": np.zeros_like(wavelength, dtype=int),
            }
        )

        base_window, base_summary = build_plot_data(spectrum)
        center_index = int(np.argmin(np.abs(wavelength - 5050.0)))
        bad_anchor_control = {
            "continuum_anchor_nodes": [
                {
                    "wavelength_A": 5050.0,
                    "continuum_flux": float(flux[center_index]),
                    "source": "unit_bad_anchor",
                }
            ]
        }
        bad_window, bad_summary = build_plot_data(spectrum, fit_control=bad_anchor_control)
        removed_window, removed_summary = build_plot_data(
            spectrum,
            fit_control={**bad_anchor_control, "continuum_anchor_remove_indices": [len(base_summary["anchor_wavelengths_A"])]},
        )
        corrected_window, corrected_summary = build_plot_data(
            spectrum,
            fit_control={
                "continuum_anchor_nodes": [
                    {
                        "wavelength_A": 5050.0,
                        "continuum_flux": float(true_continuum[center_index]),
                        "source": "unit_correct_anchor",
                    }
                ]
            },
        )

        base_error = abs(float(base_window.loc[center_index, "continuum_model"]) - float(true_continuum[center_index]))
        bad_error = abs(float(bad_window.loc[center_index, "continuum_model"]) - float(true_continuum[center_index]))
        removed_error = abs(float(removed_window.loc[center_index, "continuum_model"]) - float(true_continuum[center_index]))
        corrected_error = abs(float(corrected_window.loc[center_index, "continuum_model"]) - float(true_continuum[center_index]))

        self.assertGreater(bad_error, base_error + 0.10)
        self.assertLess(removed_error, bad_error - 0.10)
        self.assertLess(corrected_error, 0.02)
        self.assertEqual(len(corrected_summary["manual_anchor_nodes"]), 1)
        self.assertAlmostEqual(corrected_summary["manual_anchor_nodes"][0]["continuum_flux"], float(true_continuum[center_index]))
        self.assertEqual(removed_summary["removed_anchor_indices"], [len(base_summary["anchor_wavelengths_A"])])

    def test_provided_continuum_column_is_used_for_cos_like_low_flux_data(self):
        wavelength = np.linspace(1210.0, 1225.0, 64)
        continuum = np.full_like(wavelength, 3.0e-16, dtype=float)
        flux = continuum.copy()
        flux[20:24] *= 0.45
        spectrum = pd.DataFrame(
            {
                "wavelength": wavelength,
                "flux": flux,
                "ivar": np.full_like(wavelength, 1.0e28, dtype=float),
                "pipeline_mask": np.zeros_like(wavelength, dtype=int),
                "CONTINUUM": continuum,
            }
        )

        window, summary = build_plot_data(spectrum)

        self.assertEqual(summary["method"], "input_continuum_column")
        self.assertTrue(summary["provided_continuum_column"])
        normalized = window["normalized_flux"].to_numpy(dtype=float)
        ratio = np.median(normalized[np.isfinite(normalized)])
        self.assertAlmostEqual(float(ratio), 1.0, places=6)
        self.assertGreater(float(window["continuum_model"].min()), 0.0)


if __name__ == "__main__":
    unittest.main()
