import unittest

import pandas as pd

from astroagent.review_packet import (
    assess_absorber_hypothesis,
    build_review_record,
    make_demo_quasar_spectrum,
    observed_wavelength_A,
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
        self.assertTrue(record["absorber_hypothesis_check"]["candidate_absorber_reasonable"])
        self.assertTrue(record["human_adjudication"]["required"])
        self.assertIn("velocity_exceeds_virial_expectation", record["human_adjudication"]["escalation_flags"])
        self.assertEqual(record["candidate_results"], [])

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
        self.assertFalse(check["candidate_absorber_reasonable"])
        self.assertEqual(check["status"], "weak_or_absent")


if __name__ == "__main__":
    unittest.main()
