import base64
import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from astroagent.data.desi_public import CATALOGS, _merge_structured_arrays, choose_best_absorbers, parse_viewer_html


def _ndarray_payload(values):
    array = np.asarray(values, dtype="<f8")
    return {
        "__ndarray__": base64.b64encode(array.tobytes()).decode("ascii"),
        "dtype": "float64",
        "shape": [len(array)],
    }


class DesiPublicTest(unittest.TestCase):
    def test_civ_catalog_url_points_to_versioned_vac_file(self):
        self.assertIn("/civ-absorber/v1.0/", CATALOGS["CIV"]["catalog_url"])
        self.assertTrue(CATALOGS["CIV"]["catalog_url"].endswith("CIV-Absorbers-dr1-v1.0.fits"))
        self.assertEqual(CATALOGS["CIV"]["metadata_extname"], "METADATA")

    def test_merge_civ_absorber_rows_with_metadata_rows(self):
        absorbers = np.array(
            [(2.1, 0.4), (2.2, 1.7)],
            dtype=[("Z_ABS", "f8"), ("CIV_EW_TOTAL", "f8")],
        )
        metadata = np.array(
            [(11, 3.1), (22, 4.2)],
            dtype=[("TARGETID", "i8"), ("MEANSNR", "f8")],
        )

        merged = _merge_structured_arrays(absorbers, metadata)

        self.assertEqual(merged.dtype.names, ("Z_ABS", "CIV_EW_TOTAL", "TARGETID", "MEANSNR"))
        self.assertEqual(merged["TARGETID"].tolist(), [11, 22])
        self.assertEqual(merged["Z_ABS"].tolist(), [2.1, 2.2])

    def test_choose_best_civ_absorbers_scores_and_sorts_rows(self):
        table = np.array(
            [
                (11, 2.1, 0.4),
                (22, 2.2, 1.7),
                (33, np.nan, 3.0),
                (44, 2.4, 0.9),
            ],
            dtype=[("TARGETID", "i8"), ("Z_ABS", "f8"), ("CIV_EW_TOTAL", "f8")],
        )

        choices = choose_best_absorbers(table, "CIV", top_n=2)

        self.assertEqual([choice.targetid for choice in choices], [22, 44])
        self.assertEqual([choice.rank for choice in choices], [1, 2])
        self.assertEqual(choices[0].line_id, "CIV_doublet")

    def test_parse_viewer_html_extracts_spectrum_arrays(self):
        document = {
            "roots": [
                {
                    "type": "ColumnDataSource",
                    "attributes": {
                        "name": "brz",
                        "data": {
                            "type": "map",
                            "entries": [
                                ["origwave", _ndarray_payload([5000.0, 5001.0, 5001.0, 5002.0])],
                                ["origflux0", _ndarray_payload([1.0, 0.8, 0.9, 1.1])],
                                ["orignoise0", _ndarray_payload([0.5, 0.25, 0.3, 0.0])],
                            ],
                        },
                    },
                }
            ]
        }
        html = (
            '<html><body><script type="application/json" id="unit">'
            + json.dumps(document)
            + "</script></body></html>"
        )

        with tempfile.TemporaryDirectory(prefix="astroagent_viewer_") as tmpdir:
            html_path = Path(tmpdir) / "target.html"
            html_path.write_text(html, encoding="utf-8")
            spectrum = parse_viewer_html(html_path)

        self.assertEqual(list(spectrum.columns), ["wavelength", "flux", "ivar", "pipeline_mask", "band"])
        self.assertEqual(spectrum["wavelength"].tolist(), [5000.0, 5001.0, 5002.0])
        self.assertAlmostEqual(float(spectrum.loc[0, "ivar"]), 4.0)
        self.assertAlmostEqual(float(spectrum.loc[1, "ivar"]), 16.0)
        self.assertEqual(int(spectrum.loc[2, "ivar"]), 0)
        self.assertTrue((spectrum["pipeline_mask"] == 0).all())


if __name__ == "__main__":
    unittest.main()
