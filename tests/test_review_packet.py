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

from astroagent.review_packet import (
    assess_absorber_hypothesis,
    build_plot_data,
    build_plot_and_fit_data,
    build_review_record,
    make_demo_quasar_spectrum,
    observed_wavelength_A,
    write_review_packet,
)


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
        self.assertGreater(record["window_summary"]["n_good_pixels"], 0)
        self.assertTrue(record["absorber_hypothesis_check"]["absorption_signal_present"])
        self.assertTrue(record["human_adjudication"]["required"])
        self.assertIn("velocity_exceeds_virial_expectation", record["human_adjudication"]["escalation_flags"])
        self.assertEqual(record["fit_results"][0]["task"], "voigt_profile_fit")
        self.assertTrue(record["fit_results"][0]["success"])
        self.assertEqual(record["fit_results"][0]["fit_type"], "transition_frame_peak_voigt")
        self.assertEqual(record["fit_results"][0]["fit_method"], "ultranest_physical_transition_frame")
        self.assertEqual(record["fit_results"][0]["n_transition_frames"], 2)
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

    def test_voigt_fit_keeps_far_absorption_inside_transition_frame_without_soft_prior(self):
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
        self.assertTrue(any(abs(velocity) > 500.0 for velocity in peak_velocities))
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


if __name__ == "__main__":
    unittest.main()
