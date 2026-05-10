import json
import os
import subprocess
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

from astroagent.agent.fit_control import (
    append_fit_control_patch,
    build_fit_control_patch,
    summarize_pending_fit_control,
)
from astroagent.agent.fit_control import (
    _with_controlled_source_seeds,
    build_fit_control_overrides,
    evaluate_refit_patch,
    merge_fit_control_overrides,
    refit_record_with_patch,
)
from astroagent.agent.llm import (
    LLMMessage,
    LLMResult,
    OfflineReviewClient,
    build_fit_control_messages,
    _chat_completions_url,
    run_fit_review,
    run_fit_control,
)
from astroagent.review.packet import build_review_record, make_demo_quasar_spectrum, write_review_packet


class ToolCallingClient:
    def complete(self, messages, *, temperature=0.0, tools=None):
        return LLMResult(
            content="",
            model="unit-tool-model",
            raw={"unit": True},
            tool_calls=[
                {
                    "id": "call_1",
                    "type": "function",
                    "function": {
                        "name": "add_continuum_anchor",
                        "arguments": json.dumps(
                            {
                                "wavelength_A": 5575.0,
                                "reason": "restore curved continuum support",
                            }
                        ),
                    },
                },
                {
                    "id": "call_2",
                    "type": "function",
                    "function": {
                        "name": "request_refit",
                        "arguments": json.dumps({"reason": "apply continuum anchor edit"}),
                    },
                },
            ],
        )


class LLMInterfaceTest(unittest.TestCase):
    def _demo_record(self):
        spectrum = make_demo_quasar_spectrum(z_sys=2.6, line_id="CIV_doublet")
        record, _ = build_review_record(
            spectrum=spectrum,
            line_id="CIV_doublet",
            z_sys=2.6,
            sample_id="llm_demo",
            source={"kind": "unit_test"},
        )
        return record

    def test_fit_control_messages_can_include_plot_image(self):
        record = self._demo_record()
        with tempfile.TemporaryDirectory(prefix="astroagent_llm_image_") as tmpdir:
            image_path = Path(tmpdir) / "fit.png"
            image_path.write_bytes(b"\x89PNG\r\n\x1a\nunit-test")

            messages = build_fit_control_messages(record, image_path)

        self.assertEqual(messages[0].role, "system")
        self.assertIsInstance(messages[1], LLMMessage)
        self.assertIsInstance(messages[1].content, list)
        self.assertEqual(messages[1].content[0]["type"], "text")
        self.assertIn("Task: fit_control", messages[1].content[0]["text"])
        self.assertEqual(messages[1].content[1]["type"], "text")
        self.assertIn("fit.png", messages[1].content[1]["text"])
        self.assertEqual(messages[1].content[2]["type"], "image_url")
        self.assertTrue(messages[1].content[2]["image_url"]["url"].startswith("data:image/png;base64,"))

    def test_fit_control_messages_can_include_overview_and_fit_images(self):
        record = self._demo_record()
        with tempfile.TemporaryDirectory(prefix="astroagent_llm_images_") as tmpdir:
            overview_path = Path(tmpdir) / "overview.png"
            fit_path = Path(tmpdir) / "fit.png"
            overview_path.write_bytes(b"\x89PNG\r\n\x1a\noverview")
            fit_path.write_bytes(b"\x89PNG\r\n\x1a\nfit")

            messages = build_fit_control_messages(record, [overview_path, fit_path])

        content = messages[1].content
        self.assertIsInstance(content, list)
        image_parts = [part for part in content if part.get("type") == "image_url"]
        label_parts = [part for part in content if part.get("type") == "text" and str(part.get("text", "")).startswith("Image ")]
        self.assertEqual(len(image_parts), 2)
        self.assertEqual([part["text"] for part in label_parts], ["Image 1: overview.png", "Image 2: fit.png"])

    def test_fit_control_messages_include_velocity_frame_overlap_hints(self):
        messages = build_fit_control_messages(self._demo_record())
        text = messages[1].content
        self.assertIsInstance(text, str)
        prefix = "Task: fit_control. Inspect"
        self.assertTrue(text.startswith(prefix))
        payload = json.loads(text[text.index("{") :])

        diagnostic_hints = payload["fit_control_context"]["diagnostic_hints"]
        overlaps = diagnostic_hints["velocity_frame_wavelength_overlaps"]
        projections = diagnostic_hints["sibling_work_window_projections"]
        plot_data = payload["plot_data"]

        self.assertTrue(overlaps["frames"])
        self.assertTrue(overlaps["overlaps"])
        self.assertTrue(projections)
        self.assertIn("visible_overlap_in_current_frame_kms", projections[0])
        self.assertIn("continuum_anchor_table", plot_data)
        self.assertIn("continuum_anchor_diagnostics", plot_data)
        self.assertIn("anchor_index", plot_data["continuum_anchor_table"][0])
        self.assertIn("continuum is trustworthy", text)
        self.assertIn("do not add or retune sources merely to absorb a continuum error", text)
        self.assertIn("remove those biased anchor indices first", text)
        self.assertIn("Velocity panels are not independent evidence", text)
        self.assertIn("general wavelength-overlap rule", text)
        self.assertIn("explicitly attribute the worse fit to the previous round's tool calls", text)
        self.assertIn("Never pair a new edge mask with multiple new sources in the partner transition frame", text)
        self.assertIn("wavelength data and may overlap", text)
        self.assertIn("A mask only changes which pixels constrain the next fit", text)

    def test_openai_compatible_url_builder_accepts_base_or_full_api_url(self):
        self.assertEqual(
            _chat_completions_url("https://llmapi.paratera.com"),
            "https://llmapi.paratera.com/chat/completions",
        )
        self.assertEqual(
            _chat_completions_url("https://llmapi.paratera.com/v1"),
            "https://llmapi.paratera.com/v1/chat/completions",
        )
        self.assertEqual(
            _chat_completions_url("https://ignored.example/v1", "https://llmapi.paratera.com/v1/chat/completions"),
            "https://llmapi.paratera.com/v1/chat/completions",
        )

    def test_run_fit_control_normalizes_provider_tool_calls(self):
        record = self._demo_record()
        control = run_fit_control(record, ToolCallingClient())

        self.assertEqual(control["task"], "fit_control")
        self.assertEqual(control["status"], "tool_calls")
        self.assertEqual(control["_llm_metadata"]["model"], "unit-tool-model")
        self.assertEqual([call["name"] for call in control["tool_calls"]], ["add_continuum_anchor", "request_refit"])
        self.assertEqual(control["tool_calls"][0]["arguments"]["wavelength_A"], 5575.0)

    def test_run_fit_control_defaults_missing_rationale_for_json_response(self):
        class MissingRationaleClient:
            def complete(self, messages, *, temperature=0.0, tools=None):
                return LLMResult(
                    content=json.dumps(
                        {
                            "task": "fit_control",
                            "status": "no_action",
                            "tool_calls": [],
                        }
                    ),
                    model="unit-missing-rationale-model",
                    raw={"unit": True},
                )

        control = run_fit_control(self._demo_record(), MissingRationaleClient())

        self.assertEqual(control["rationale"], "Model produced no rationale.")
        self.assertEqual(control["status"], "no_action")

    def test_run_fit_control_accepts_duplicated_json_object_prefix(self):
        payload = json.dumps(
            {
                "task": "fit_control",
                "status": "no_action",
                "tool_calls": [],
                "rationale": "first object",
            }
        )

        class DuplicatedJsonClient:
            def complete(self, messages, *, temperature=0.0, tools=None):
                return LLMResult(
                    content=payload + payload,
                    model="unit-duplicated-json-model",
                    raw={"unit": True},
                )

        control = run_fit_control(self._demo_record(), DuplicatedJsonClient())

        self.assertEqual(control["rationale"], "first object")
        self.assertEqual(control["status"], "no_action")

    def test_run_fit_control_normalizes_loose_status_labels(self):
        class LooseStatusClient:
            def complete(self, messages, *, temperature=0.0, tools=None):
                return LLMResult(
                    content=json.dumps(
                        {
                            "task": "fit_control",
                            "status": "needs_human_review",
                            "tool_calls": [],
                            "rationale": "label is loose",
                        }
                    ),
                    model="unit-loose-status-model",
                    raw={"unit": True},
                )

        control = run_fit_control(self._demo_record(), LooseStatusClient())

        self.assertEqual(control["status"], "inspect")

    def test_offline_fit_review_uses_second_stage_schema(self):
        record = self._demo_record()
        review = run_fit_review(record, OfflineReviewClient())

        self.assertEqual(review["task"], "fit_review")
        self.assertIn("system_plausible", review)
        self.assertNotIn("absorber_reasonable", review)

    def test_build_fit_control_patch_tracks_refit_requirement(self):
        control = {
            "task": "fit_control",
            "status": "tool_calls",
            "rationale": "continuum anchor is missing",
            "tool_calls": [
                {
                    "id": "call_1",
                    "name": "add_continuum_anchor",
                    "arguments": {"wavelength_A": 5575.0, "reason": "anchor the continuum shoulder"},
                }
            ],
            "_llm_metadata": {"model": "unit-model"},
        }

        patch = build_fit_control_patch(control, source="unit")

        self.assertEqual(patch["task"], "fit_control_patch")
        self.assertEqual(patch["source"], "unit")
        self.assertTrue(patch["requires_refit"])
        self.assertFalse(patch["applied"])
        self.assertEqual(patch["tool_calls"][0]["sequence_index"], 0)
        self.assertTrue(patch["tool_calls"][0]["validated"])

    def test_request_refit_alone_is_not_a_concrete_patch(self):
        control = {
            "task": "fit_control",
            "status": "tool_calls",
            "rationale": "try again",
            "tool_calls": [
                {
                    "name": "request_refit",
                    "arguments": {"reason": "no concrete edit"},
                }
            ],
        }

        patch = build_fit_control_patch(control, source="unit")

        self.assertFalse(patch["requires_refit"])

    def test_append_and_summarize_pending_fit_control_patches(self):
        record = {"sample_id": "patch_demo", "human_review": {}}
        control = {
            "task": "fit_control",
            "status": "tool_calls",
            "rationale": "remove spurious source",
            "tool_calls": [
                {
                    "name": "remove_absorption_source",
                    "arguments": {"component_index": 2, "reason": "duplicate component"},
                }
            ],
        }

        updated = append_fit_control_patch(record, control)
        summary = summarize_pending_fit_control(updated)

        self.assertEqual(summary["n_patches"], 1)
        self.assertEqual(summary["n_pending_patches"], 1)
        self.assertEqual(summary["pending_tool_counts"]["remove_absorption_source"], 1)
        self.assertTrue(summary["requires_refit"])
        self.assertIn("fit_control_notes", updated["human_review"])
        self.assertNotIn("fit_control_patches", record)

    def test_run_fit_review_cli_writes_control_and_patch_defaults(self):
        with tempfile.TemporaryDirectory(prefix="astroagent_llm_cli_") as tmpdir:
            outdir = Path(tmpdir)
            spectrum = make_demo_quasar_spectrum(z_sys=2.6, line_id="CIV_doublet")
            record, window = build_review_record(
                spectrum=spectrum,
                line_id="CIV_doublet",
                z_sys=2.6,
                sample_id="cli_demo",
                source={"kind": "unit_test"},
            )
            paths = write_review_packet(record, window, outdir)

            result = subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "scripts" / "run_fit_review.py"),
                    "--review-json",
                    str(paths["json"]),
                    "--client",
                    "offline",
                    "--mode",
                    "fit_control",
                    "--plot-image",
                    str(paths["plot_png"]),
                ],
                cwd=ROOT,
                env={**os.environ, "PYTHONPATH": str(SRC)},
                check=True,
                capture_output=True,
                text=True,
            )

            control_path = outdir / "cli_demo.llm_control.json"
            patch_path = outdir / "cli_demo.fit_control_patch.json"
            self.assertIn(str(control_path), result.stdout)
            self.assertTrue(control_path.exists())
            self.assertTrue(patch_path.exists())
            patch = json.loads(patch_path.read_text(encoding="utf-8"))
            self.assertEqual(patch["task"], "fit_control_patch")

    def test_unified_cli_runs_llm_with_default_plot_image(self):
        with tempfile.TemporaryDirectory(prefix="astroagent_unified_llm_cli_") as tmpdir:
            outdir = Path(tmpdir)
            spectrum = make_demo_quasar_spectrum(z_sys=2.6, line_id="CIV_doublet")
            record, window = build_review_record(
                spectrum=spectrum,
                line_id="CIV_doublet",
                z_sys=2.6,
                sample_id="unified_cli_demo",
                source={"kind": "unit_test"},
            )
            paths = write_review_packet(record, window, outdir)

            result = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "astroagent.cli.main",
                    "llm",
                    "--review-json",
                    str(paths["json"]),
                    "--client",
                    "offline",
                    "--mode",
                    "fit_control",
                ],
                cwd=ROOT,
                env={**os.environ, "PYTHONPATH": str(SRC)},
                check=True,
                capture_output=True,
                text=True,
            )

            self.assertIn("unified_cli_demo.llm_control.json", result.stdout)
            self.assertTrue((outdir / "unified_cli_demo.llm_control.json").exists())
            self.assertTrue((outdir / "unified_cli_demo.fit_control_patch.json").exists())

    def test_fit_control_patch_can_be_applied_to_refit_record(self):
        spectrum = make_demo_quasar_spectrum(z_sys=2.6, line_id="CIV_doublet")
        record, window = build_review_record(
            spectrum=spectrum,
            line_id="CIV_doublet",
            z_sys=2.6,
            sample_id="apply_demo",
            source={"kind": "unit_test"},
        )
        transition_id = record["input"]["transitions"][0]["transition_line_id"]
        patch = {
            "task": "fit_control_patch",
            "source": "unit",
            "status": "tool_calls",
            "rationale": "add a missing component and narrow the frame",
            "tool_calls": [
                {
                    "sequence_index": 0,
                    "name": "add_absorption_source",
                    "arguments": {
                        "transition_line_id": transition_id,
                        "center_velocity_kms": 140.0,
                        "logN": 13.8,
                        "b_kms": 30.0,
                        "center_prior_sigma_kms": 45.0,
                        "reason": "visible residual shoulder",
                    },
                },
                {
                    "sequence_index": 1,
                    "name": "set_fit_window",
                    "arguments": {
                        "transition_line_id": transition_id,
                        "transition_half_width_kms": 320.0,
                        "local_fit_half_width_kms": 120.0,
                        "sibling_mask_mode": "allow_overlap",
                        "reason": "focus on local virial-scale frame",
                    },
                },
                {
                    "sequence_index": 2,
                    "name": "add_continuum_anchor",
                    "arguments": {
                        "wavelength_A": record["input"]["observed_centers_A"][0] + 3.0,
                        "continuum_flux": 1.08,
                        "reason": "support curved local continuum",
                    },
                },
                {
                    "sequence_index": 3,
                    "name": "request_refit",
                    "arguments": {"reason": "apply patch"},
                },
            ],
            "requires_refit": True,
            "applied": False,
        }

        overrides = build_fit_control_overrides(record, patch)
        refit_record, applied_overrides = refit_record_with_patch(record, window, patch, sample_id="apply_demo_refit")

        self.assertEqual(overrides["fit_windows"][transition_id]["transition_half_width_kms"], 320.0)
        self.assertEqual(overrides["fit_windows"][transition_id]["sibling_mask_mode"], "allow_overlap")
        self.assertEqual(applied_overrides["source_seeds"][0]["center_velocity_kms"], 140.0)
        self.assertEqual(applied_overrides["source_detection_mode"], "controlled")
        self.assertEqual(overrides["continuum_anchor_nodes"][0]["continuum_flux"], 1.08)
        self.assertEqual(refit_record["sample_id"], "apply_demo_refit")
        self.assertTrue(refit_record["fit_control_patches"][0]["applied"])
        self.assertIn(refit_record["fit_control_patches"][0]["application_result"]["decision"], {"accepted", "needs_human_review", "rejected"})
        self.assertIn("fit_control_evaluation", refit_record)
        self.assertTrue(refit_record["fit_control_application"]["applied"])
        self.assertIn("evaluation", refit_record["fit_control_application"])
        self.assertGreaterEqual(refit_record["fit_results"][0]["fit_control_applied"]["n_source_seeds"], 1)
        refit_frame = next(
            frame
            for frame in refit_record["fit_results"][0]["transition_frames"]
            if frame["transition_line_id"] == transition_id
        )
        self.assertEqual(refit_frame["velocity_frame"]["bounds_kms"], [-320.0, 320.0])
        self.assertEqual(refit_frame["simultaneous_fit_window"]["sibling_mask_mode"], "allow_overlap")
        self.assertEqual(refit_frame["sibling_transition_masks"], [])

    def test_refit_evaluation_rejects_failed_or_degraded_patch(self):
        original_fit = {
            "success": True,
            "quality": "good",
            "fit_rms": 1.0,
            "n_components_fitted": 2,
            "n_fit_pixels": 100,
            "agent_review": {"required": False, "reasons": []},
            "transition_frames": [],
        }
        degraded_fit = {
            "success": True,
            "quality": "inspect",
            "fit_rms": 1.4,
            "n_components_fitted": 8,
            "n_fit_pixels": 20,
            "agent_review": {"required": True, "reasons": ["high_residual_transition_peak"]},
            "transition_frames": [],
        }

        evaluation = evaluate_refit_patch(original_fit, degraded_fit, {"source_seeds": [], "removed_sources": []})

        self.assertEqual(evaluation["decision"], "rejected")
        self.assertFalse(evaluation["accepted"])
        self.assertTrue(evaluation["requires_human_review"])
        self.assertIn("fit_rms_worsened", evaluation["warnings"])
        self.assertIn("component_count_exploded", evaluation["warnings"])
        self.assertIn("fit_window_or_mask_removed_too_many_pixels", evaluation["reasons"])
        self.assertIn("fit_quality_regressed", evaluation["warnings"])

    def test_refit_evaluation_accepts_clean_improvement(self):
        original_fit = {
            "success": True,
            "quality": "inspect",
            "fit_rms": 2.0,
            "n_components_fitted": 2,
            "n_fit_pixels": 100,
            "agent_review": {"required": True, "reasons": ["high_residual_transition_peak"]},
            "transition_frames": [],
        }
        improved_fit = {
            "success": True,
            "quality": "good",
            "fit_rms": 1.7,
            "n_components_fitted": 2,
            "n_fit_pixels": 95,
            "agent_review": {"required": False, "reasons": []},
            "transition_frames": [],
        }

        evaluation = evaluate_refit_patch(original_fit, improved_fit, {"source_seeds": [], "removed_sources": []})

        self.assertEqual(evaluation["decision"], "accepted")
        self.assertTrue(evaluation["accepted"])
        self.assertFalse(evaluation["requires_human_review"])
        self.assertEqual(evaluation["warnings"], [])
        self.assertIn("chi2", evaluation["metrics"]["delta"])
        self.assertIn("reduced_chi2", evaluation["metrics"]["delta"])
        self.assertIn("source_work_window_reduced_chi2", evaluation["metrics"]["delta"])

    def test_refit_evaluation_downgrades_global_rms_worsening_when_work_window_improves(self):
        original_fit = {
            "success": True,
            "quality": "inspect",
            "fit_rms": 1.0,
            "n_components_fitted": 3,
            "n_fit_pixels": 100,
            "source_work_window_metrics": {"fit_rms": 1.0, "n_fit_pixels": 80, "max_abs_residual_sigma": 5.0},
            "agent_review": {"required": True, "reasons": ["high_residual_transition_peak"]},
            "transition_frames": [],
        }
        refit_fit = {
            "success": True,
            "quality": "inspect",
            "fit_rms": 1.12,
            "n_components_fitted": 4,
            "n_fit_pixels": 100,
            "source_work_window_metrics": {"fit_rms": 0.90, "n_fit_pixels": 80, "max_abs_residual_sigma": 4.0},
            "agent_review": {"required": True, "reasons": ["high_residual_transition_peak"]},
            "transition_frames": [],
        }

        evaluation = evaluate_refit_patch(original_fit, refit_fit, {"source_seeds": [{"transition_line_id": "MGII_2796"}], "removed_sources": []})

        self.assertEqual(evaluation["decision"], "needs_human_review")
        self.assertNotIn("fit_rms_worsened", evaluation["reasons"])
        self.assertIn("fit_rms_worsened_but_source_work_window_improved", evaluation["warnings"])

    def test_refit_evaluation_does_not_hard_reject_continuum_only_rms_worsening(self):
        original_fit = {
            "success": True,
            "quality": "inspect",
            "fit_rms": 1.0,
            "n_components_fitted": 2,
            "n_fit_pixels": 100,
            "source_work_window_metrics": {"fit_rms": 1.0, "n_fit_pixels": 80, "max_abs_residual_sigma": 3.0},
            "agent_review": {"required": True, "reasons": ["saturated_peak"]},
            "transition_frames": [],
        }
        refit_fit = {
            "success": True,
            "quality": "inspect",
            "fit_rms": 1.9,
            "n_components_fitted": 2,
            "n_fit_pixels": 100,
            "source_work_window_metrics": {"fit_rms": 1.8, "n_fit_pixels": 80, "max_abs_residual_sigma": 5.0},
            "agent_review": {"required": True, "reasons": ["saturated_peak", "component_parameter_posterior_degenerate"]},
            "transition_frames": [],
        }

        evaluation = evaluate_refit_patch(
            original_fit,
            refit_fit,
            {
                "source_seeds": [],
                "removed_sources": [],
                "continuum_anchor_remove_wavelengths_A": [9775.6, 9780.4],
            },
        )

        self.assertEqual(evaluation["decision"], "needs_human_review")
        self.assertNotIn("fit_rms_worsened", evaluation["reasons"])
        self.assertIn("continuum_changed_fit_not_directly_comparable", evaluation["warnings"])
        self.assertIn("continuum_refit_needs_source_mask_followup", evaluation["warnings"])

    def test_fit_control_prompt_omits_map_initializer_fields(self):
        record = self._demo_record()
        fit_summary = record["fit_results"][0]
        fit_summary.setdefault("components", [{}])[0]["map_parameters"] = {"center_velocity_kms": 123.0}
        fit_summary["components"][0]["fit_parameter_summary"] = {"backend": "least_squares"}
        fit_summary.setdefault("transition_frames", [{"peaks": [{}]}])[0].setdefault("peaks", [{}])[0][
            "map_parameters"
        ] = {"center_velocity_kms": 123.0}
        fit_summary["transition_frames"][0]["peaks"][0]["fit_parameter_summary"] = {"backend": "least_squares"}

        messages = build_fit_control_messages(record)
        text = messages[1].content

        self.assertNotIn("map_parameters", text)
        self.assertNotIn("fit_parameter_summary", text)

    def test_fit_control_prompt_includes_lsf_parallel_diagnostic(self):
        record = self._demo_record()
        fit_summary = record["fit_results"][0]
        fit_summary["lsf"] = {
            "available": True,
            "applied_to_fit_likelihood": False,
            "used_for_diagnostic_model": True,
            "transition_frames_with_lsf": ["CIV_1548"],
        }
        fit_summary["instrument_lsf_applied"] = True
        fit_summary["instrument_lsf_applied_to_fit_likelihood"] = False
        fit_summary["transition_frames"][0]["lsf_diagnostic"] = {
            "available": True,
            "comparison": "same fitted intrinsic Voigt parameters evaluated after LSF/resolution-matrix convolution",
            "intrinsic_fit_window_metrics": {"fit_rms": 1.4},
            "lsf_fit_window_metrics": {"fit_rms": 1.1},
        }

        messages = build_fit_control_messages(record)
        text = messages[1].content

        self.assertIn("lsf", text)
        self.assertIn("lsf_diagnostic", text)
        self.assertIn("applied_to_fit_likelihood", text)
        self.assertIn("instrument_lsf_applied_to_fit_likelihood", text)
        self.assertIn("intrinsic_fit_window_metrics", text)
        self.assertIn("lsf_fit_window_metrics", text)

    def test_fit_control_prompt_includes_soft_doublet_context_and_outlier_policy(self):
        record = self._demo_record()

        messages = build_fit_control_messages(record)
        text = messages[1].content

        self.assertIn("line_family_context", text)
        self.assertIn("soft physical background", text)
        self.assertIn("not an automatic failure", text)
        self.assertIn("Preserve and report single-member", text)
        self.assertIn("outlier_policy", text)

    def test_continuum_changed_carryover_sources_are_soft_position_priors(self):
        original_fit = {
            "components": [
                {
                    "fit_success": True,
                    "component_index": 3,
                    "transition_line_id": "MGII_2803",
                    "center_velocity_kms": 42.0,
                    "logN": 18.8,
                    "b_kms": 180.0,
                }
            ]
        }
        controls = {
            "continuum_anchor_remove_wavelengths_A": [9806.0],
            "source_seeds": [],
            "removed_sources": [],
            "fit_mask_intervals": [],
            "fit_windows": {},
            "tool_calls": [
                {
                    "name": "remove_continuum_anchor",
                    "arguments": {"wavelength_A": 9806.0, "reason": "bad continuum"},
                }
            ],
        }

        controlled = _with_controlled_source_seeds(original_fit, controls)
        carried = controlled["source_seeds"][0]

        self.assertEqual(carried["seed_source"], "previous_fit_component")
        self.assertFalse(carried["explicit_fit_control_source"])
        self.assertEqual(carried["center_velocity_kms"], 42.0)
        self.assertEqual(carried["center_prior_sigma_kms"], 90.0)
        self.assertEqual(carried["center_prior_half_width_kms"], 240.0)
        self.assertNotIn("logN", carried)
        self.assertNotIn("b_kms", carried)
        self.assertNotIn("logN_lower", carried)
        self.assertNotIn("logN_upper", carried)
        self.assertNotIn("b_kms_lower", carried)
        self.assertNotIn("b_kms_upper", carried)

    def test_continuum_changed_existing_source_seeds_are_soft_position_priors(self):
        controls = {
            "continuum_anchor_remove_wavelengths_A": [9806.0],
            "source_seeds": [
                {
                    "transition_line_id": "MGII_2803",
                    "center_velocity_kms": 42.0,
                    "explicit_fit_control_source": True,
                    "seed_source": "add_absorption_source",
                    "logN": 18.8,
                    "b_kms": 180.0,
                    "logN_lower": 15.0,
                    "logN_upper": 19.0,
                    "b_kms_lower": 20.0,
                    "b_kms_upper": 250.0,
                }
            ],
            "removed_sources": [],
            "fit_mask_intervals": [],
            "fit_windows": {},
            "tool_calls": [
                {
                    "name": "remove_continuum_anchor",
                    "arguments": {"wavelength_A": 9806.0, "reason": "bad continuum"},
                }
            ],
        }

        controlled = _with_controlled_source_seeds({"components": []}, controls)
        seed = controlled["source_seeds"][0]

        self.assertFalse(seed["explicit_fit_control_source"])
        self.assertEqual(seed["center_prior_sigma_kms"], 90.0)
        self.assertEqual(seed["center_prior_half_width_kms"], 240.0)
        self.assertNotIn("logN", seed)
        self.assertNotIn("b_kms", seed)
        self.assertNotIn("logN_lower", seed)
        self.assertNotIn("logN_upper", seed)
        self.assertNotIn("b_kms_lower", seed)
        self.assertNotIn("b_kms_upper", seed)

    def test_refit_evaluation_reports_high_context_diagnostic_residual(self):
        original_fit = {
            "success": True,
            "quality": "inspect",
            "fit_rms": 1.4,
            "n_components_fitted": 1,
            "n_fit_pixels": 30,
            "agent_review": {"required": True, "reasons": []},
            "transition_frames": [],
        }
        refit_fit = {
            "success": True,
            "quality": "inspect",
            "fit_rms": 1.2,
            "n_components_fitted": 1,
            "n_fit_pixels": 30,
            "agent_review": {"required": True, "reasons": []},
            "transition_frames": [
                {
                    "transition_line_id": "MGII_2796",
                    "diagnostic_residual_samples": [
                        {"velocity_kms": 470.0, "residual_sigma": 4.5, "used_in_fit": False},
                        {"velocity_kms": 510.0, "residual_sigma": -2.0, "used_in_fit": False},
                        {"velocity_kms": 520.0, "residual_sigma": 7.0, "used_in_fit": True},
                    ],
                }
            ],
        }

        evaluation = evaluate_refit_patch(original_fit, refit_fit, {"source_seeds": [], "removed_sources": []})
        context = evaluation["metrics"]["delta"]["context_residual"]

        self.assertIn("high_context_diagnostic_residual", evaluation["warnings"])
        self.assertEqual(context["n_high_context_diagnostic_residual_pixels"], 1)
        self.assertEqual(context["details"][0]["n_context_diagnostic_pixels"], 2)

    def test_refit_evaluation_penalizes_added_source_in_saturated_redundant_region(self):
        original_fit = {
            "success": True,
            "quality": "inspect",
            "fit_rms": 1.2,
            "n_components_fitted": 1,
            "n_fit_pixels": 80,
            "agent_review": {"required": True, "reasons": ["saturated_peak"]},
            "transition_frames": [],
            "components": [
                {
                    "component_index": 0,
                    "transition_line_id": "MGII_2803",
                    "diagnostic_flags": [],
                }
            ],
        }
        refit_fit = {
            "success": True,
            "quality": "inspect",
            "fit_rms": 1.0,
            "n_components_fitted": 2,
            "n_fit_pixels": 80,
            "agent_review": {"required": True, "reasons": ["saturated_peak", "saturated_redundant_component"]},
            "transition_frames": [],
            "components": [
                {
                    "component_index": 0,
                    "transition_line_id": "MGII_2803",
                    "diagnostic_flags": [],
                },
                {
                    "component_index": 1,
                    "transition_line_id": "MGII_2803",
                    "diagnostic_flags": ["saturated_redundant_component"],
                },
            ],
        }

        evaluation = evaluate_refit_patch(
            original_fit,
            refit_fit,
            {"source_seeds": [{"transition_line_id": "MGII_2803"}], "removed_sources": []},
        )

        self.assertEqual(evaluation["decision"], "needs_human_review")
        self.assertIn("added_source_in_already_saturated_region", evaluation["warnings"])
        self.assertGreater(
            evaluation["metrics"]["delta"]["component_diagnostic_regression"]["new_saturated_redundant_components"],
            0,
        )

    def test_refit_evaluation_keeps_hard_reject_for_large_global_rms_worsening(self):
        original_fit = {
            "success": True,
            "quality": "inspect",
            "fit_rms": 1.0,
            "n_components_fitted": 3,
            "n_fit_pixels": 100,
            "source_work_window_metrics": {"fit_rms": 1.0, "n_fit_pixels": 80, "max_abs_residual_sigma": 5.0},
            "agent_review": {"required": True, "reasons": []},
            "transition_frames": [],
        }
        refit_fit = {
            "success": True,
            "quality": "inspect",
            "fit_rms": 1.35,
            "n_components_fitted": 4,
            "n_fit_pixels": 100,
            "source_work_window_metrics": {"fit_rms": 0.90, "n_fit_pixels": 80, "max_abs_residual_sigma": 4.0},
            "agent_review": {"required": True, "reasons": []},
            "transition_frames": [],
        }

        evaluation = evaluate_refit_patch(original_fit, refit_fit, {"source_seeds": [{"transition_line_id": "MGII_2796"}], "removed_sources": []})

        self.assertEqual(evaluation["decision"], "needs_human_review")
        self.assertIn("fit_rms_worsened", evaluation["warnings"])

    def test_refit_evaluation_marks_global_rms_incomparable_when_new_transition_frame_is_fit(self):
        original_fit = {
            "success": True,
            "quality": "inspect",
            "chi2": 43.08,
            "reduced_chi2": 0.916,
            "fit_rms": 0.957,
            "n_components_fitted": 4,
            "n_fit_pixels": 47,
            "source_work_window_metrics": {"fit_rms": 0.964, "n_fit_pixels": 32, "max_abs_residual_sigma": 2.4},
            "agent_review": {"required": True, "reasons": ["saturated_peak"]},
            "transition_frames": [
                {
                    "transition_line_id": "MGII_2796",
                    "observed_center_A": 9780.52,
                    "velocity_frame": {"bounds_kms": [-600.0, 600.0]},
                    "fit_metrics": {"n_fit_pixels": 47, "chi2": 43.08, "reduced_chi2": 0.916, "fit_rms": 0.957},
                },
                {
                    "transition_line_id": "MGII_2803",
                    "observed_center_A": 9805.63,
                    "velocity_frame": {"bounds_kms": [-600.0, 600.0]},
                    "fit_metrics": {"n_fit_pixels": 0, "chi2": float("nan"), "reduced_chi2": float("nan"), "fit_rms": float("nan")},
                },
            ],
        }
        refit_fit = {
            "success": True,
            "quality": "inspect",
            "chi2": 227.51,
            "reduced_chi2": 2.446,
            "fit_rms": 1.564,
            "n_components_fitted": 6,
            "n_fit_pixels": 93,
            "source_work_window_metrics": {"fit_rms": 0.903, "n_fit_pixels": 65, "max_abs_residual_sigma": 2.45},
            "agent_review": {"required": True, "reasons": ["saturated_peak"]},
            "transition_frames": [
                {
                    "transition_line_id": "MGII_2796",
                    "observed_center_A": 9780.52,
                    "velocity_frame": {"bounds_kms": [-600.0, 600.0]},
                    "fit_metrics": {"n_fit_pixels": 47, "chi2": 44.0, "reduced_chi2": 0.936, "fit_rms": 0.967},
                },
                {
                    "transition_line_id": "MGII_2803",
                    "observed_center_A": 9805.63,
                    "velocity_frame": {"bounds_kms": [-600.0, 600.0]},
                    "fit_metrics": {"n_fit_pixels": 46, "chi2": 183.5, "reduced_chi2": 3.99, "fit_rms": 1.997},
                },
            ],
        }

        evaluation = evaluate_refit_patch(
            original_fit,
            refit_fit,
            {"source_seeds": [{"transition_line_id": "MGII_2803", "center_velocity_kms": -220.0}], "removed_sources": []},
        )

        self.assertEqual(evaluation["decision"], "needs_human_review")
        self.assertNotIn("fit_rms_worsened", evaluation["reasons"])
        self.assertIn("fit_pixel_coverage_changed_between_rounds", evaluation["warnings"])
        self.assertIn("global_fit_rms_not_directly_comparable", evaluation["warnings"])
        frame_comparison = evaluation["metrics"]["delta"]["transition_frame_fit_comparison"]
        self.assertEqual(frame_comparison["added_fitted_transition_line_ids"], ["MGII_2803"])
        self.assertEqual(frame_comparison["common_fitted_transition_line_ids"], ["MGII_2796"])
        overlap_policy = evaluation["metrics"]["delta"]["overlap_edit_policy"]
        self.assertTrue(overlap_policy["edits_in_overlapping_velocity_frames"])
        self.assertTrue(overlap_policy["edit_annotations"])

    def test_refit_evaluation_flags_overlap_source_removal_without_mask(self):
        original_fit = {
            "success": True,
            "quality": "inspect",
            "fit_rms": 1.0,
            "n_components_fitted": 2,
            "n_fit_pixels": 80,
            "agent_review": {"required": True, "reasons": []},
            "transition_frames": [
                {
                    "transition_line_id": "LINE_A",
                    "observed_center_A": 1000.0,
                    "velocity_frame": {"bounds_kms": [-600.0, 600.0]},
                    "fit_metrics": {"n_fit_pixels": 40, "chi2": 40.0, "reduced_chi2": 1.0, "fit_rms": 1.0},
                },
                {
                    "transition_line_id": "LINE_B",
                    "observed_center_A": 1002.0,
                    "velocity_frame": {"bounds_kms": [-600.0, 600.0]},
                    "fit_metrics": {"n_fit_pixels": 40, "chi2": 40.0, "reduced_chi2": 1.0, "fit_rms": 1.0},
                },
            ],
        }
        refit_fit = dict(original_fit)

        evaluation = evaluate_refit_patch(
            original_fit,
            refit_fit,
            {
                "removed_sources": [
                    {
                        "component_index": 3,
                        "transition_line_id": "LINE_A",
                        "center_velocity_kms": 560.0,
                    }
                ],
                "source_seeds": [],
                "fit_mask_intervals": [],
            },
        )

        self.assertIn("edits_in_overlapping_velocity_frames", evaluation["warnings"])
        self.assertIn("missing_overlap_mask_after_source_removal", evaluation["warnings"])
        self.assertIn("overlap_affected_velocity_frames_unaddressed", evaluation["warnings"])
        overlap_policy = evaluation["metrics"]["delta"]["overlap_edit_policy"]
        self.assertTrue(overlap_policy["removed_source_violations"])
        self.assertEqual(overlap_policy["unaddressed_affected_transition_line_ids"][0]["transition_line_id"], "LINE_B")

    def test_refit_evaluation_annotates_overlap_mask_edits(self):
        fit = {
            "success": True,
            "quality": "inspect",
            "fit_rms": 1.0,
            "n_components_fitted": 2,
            "n_fit_pixels": 80,
            "agent_review": {"required": True, "reasons": []},
            "transition_frames": [
                {
                    "transition_line_id": "LINE_A",
                    "observed_center_A": 1000.0,
                    "velocity_frame": {"bounds_kms": [-600.0, 600.0]},
                    "fit_metrics": {"n_fit_pixels": 40, "chi2": 40.0, "reduced_chi2": 1.0, "fit_rms": 1.0},
                },
                {
                    "transition_line_id": "LINE_B",
                    "observed_center_A": 1002.0,
                    "velocity_frame": {"bounds_kms": [-600.0, 600.0]},
                    "fit_metrics": {"n_fit_pixels": 40, "chi2": 40.0, "reduced_chi2": 1.0, "fit_rms": 1.0},
                },
            ],
        }

        evaluation = evaluate_refit_patch(
            fit,
            fit,
            {
                "removed_sources": [],
                "source_seeds": [],
                "fit_mask_intervals": [
                    {
                        "transition_line_id": "LINE_A",
                        "start_velocity_kms": 520.0,
                        "end_velocity_kms": 590.0,
                        "mask_kind": "exclude",
                    }
                ],
            },
        )

        self.assertIn("edits_in_overlapping_velocity_frames", evaluation["warnings"])
        self.assertIn("overlap_affected_velocity_frames_unaddressed", evaluation["warnings"])
        overlap_policy = evaluation["metrics"]["delta"]["overlap_edit_policy"]
        self.assertEqual(overlap_policy["edit_annotations"][0]["tool_kind"], "set_fit_mask_interval:exclude")

    def test_refit_evaluation_flags_affected_overlap_frame_without_fit_pixels(self):
        original_fit = {
            "success": True,
            "quality": "inspect",
            "fit_rms": 1.0,
            "n_components_fitted": 2,
            "n_fit_pixels": 40,
            "agent_review": {"required": True, "reasons": []},
            "transition_frames": [
                {
                    "transition_line_id": "LINE_A",
                    "observed_center_A": 1000.0,
                    "velocity_frame": {"bounds_kms": [-600.0, 600.0]},
                    "fit_metrics": {"n_fit_pixels": 40, "chi2": 40.0, "reduced_chi2": 1.0, "fit_rms": 1.0},
                },
                {
                    "transition_line_id": "LINE_B",
                    "observed_center_A": 1002.0,
                    "velocity_frame": {"bounds_kms": [-600.0, 600.0]},
                    "fit_metrics": {"n_fit_pixels": 0, "chi2": float("nan"), "reduced_chi2": float("nan"), "fit_rms": float("nan")},
                },
            ],
        }

        evaluation = evaluate_refit_patch(
            original_fit,
            original_fit,
            {
                "removed_sources": [],
                "source_seeds": [],
                "fit_mask_intervals": [
                    {
                        "transition_line_id": "LINE_A",
                        "start_velocity_kms": 520.0,
                        "end_velocity_kms": 590.0,
                        "mask_kind": "exclude",
                    }
                ],
            },
        )

        self.assertIn("overlap_affected_velocity_frames_without_fit_pixels", evaluation["warnings"])
        overlap_policy = evaluation["metrics"]["delta"]["overlap_edit_policy"]
        self.assertEqual(overlap_policy["affected_transition_line_ids_without_fit_pixels"][0]["transition_line_id"], "LINE_B")

    def test_refit_evaluation_does_not_hard_reject_light_quality_regression_with_clear_rms_gain(self):
        original_fit = {
            "success": True,
            "quality": "good",
            "fit_rms": 1.25,
            "n_components_fitted": 3,
            "n_fit_pixels": 120,
            "agent_review": {"required": False, "reasons": []},
            "transition_frames": [],
        }
        improved_but_annotated_fit = {
            "success": True,
            "quality": "inspect",
            "fit_rms": 1.00,
            "n_components_fitted": 4,
            "n_fit_pixels": 120,
            "source_work_window_metrics": {"fit_rms": 0.92, "n_fit_pixels": 88, "max_abs_residual_sigma": 2.4},
            "agent_review": {"required": True, "reasons": ["many_components_in_transition_frame"]},
            "transition_frames": [],
        }

        evaluation = evaluate_refit_patch(original_fit, improved_but_annotated_fit, {"source_seeds": [{"transition_line_id": "HI_LYA"}], "removed_sources": []})

        self.assertEqual(evaluation["decision"], "needs_human_review")
        self.assertFalse(evaluation["accepted"])
        self.assertTrue(evaluation["requires_human_review"])
        self.assertIn("fit_quality_regressed", evaluation["warnings"])

    def test_refit_evaluation_reports_region_aware_fit_mask_metrics(self):
        original_fit = {
            "success": True,
            "quality": "inspect",
            "fit_rms": 2.0,
            "n_components_fitted": 2,
            "n_fit_pixels": 100,
            "agent_review": {"required": True, "reasons": []},
            "transition_frames": [],
        }
        improved_fit = {
            "success": True,
            "quality": "good",
            "fit_rms": 1.8,
            "n_components_fitted": 3,
            "n_fit_pixels": 90,
            "agent_review": {"required": False, "reasons": []},
            "transition_frames": [],
        }
        overrides = {
            "source_seeds": [],
            "removed_sources": [],
            "fit_mask_intervals": [
                {
                    "transition_line_id": "HI_LYA",
                    "start_velocity_kms": -380.0,
                    "end_velocity_kms": -350.0,
                    "mask_kind": "exclude",
                },
                {
                    "transition_line_id": "HI_LYA",
                    "start_velocity_kms": 460.0,
                    "end_velocity_kms": 520.0,
                    "mask_kind": "exclude",
                },
            ],
        }

        evaluation = evaluate_refit_patch(original_fit, improved_fit, overrides)
        fit_mask = evaluation["metrics"]["delta"]["fit_mask"]

        self.assertEqual(fit_mask["n_intervals"], 2)
        self.assertEqual(fit_mask["n_core_intervals"], 1)
        self.assertEqual(fit_mask["core_width_kms"], 10.0)
        self.assertEqual(fit_mask["boundary_width_kms"], 20.0)
        self.assertEqual(fit_mask["context_width_kms"], 60.0)
        self.assertEqual(fit_mask["max_interval_width_kms"], 60.0)
        self.assertNotIn("fit_mask_total_width_large", evaluation["warnings"])

    def test_refit_evaluation_rejects_excessive_core_fit_mask(self):
        original_fit = {
            "success": True,
            "quality": "inspect",
            "fit_rms": 2.0,
            "n_components_fitted": 2,
            "n_fit_pixels": 100,
            "agent_review": {"required": True, "reasons": []},
            "transition_frames": [],
        }
        improved_fit = {
            "success": True,
            "quality": "good",
            "fit_rms": 1.8,
            "n_components_fitted": 3,
            "n_fit_pixels": 80,
            "agent_review": {"required": False, "reasons": []},
            "transition_frames": [],
        }
        overrides = {
            "source_seeds": [],
            "removed_sources": [],
            "fit_mask_intervals": [
                {
                    "transition_line_id": "HI_LYA",
                    "start_velocity_kms": -80.0,
                    "end_velocity_kms": 80.0,
                    "mask_kind": "exclude",
                }
            ],
        }

        evaluation = evaluate_refit_patch(original_fit, improved_fit, overrides)

        self.assertEqual(evaluation["decision"], "rejected")
        self.assertIn("fit_mask_core_removed_too_much", evaluation["reasons"])
        self.assertIn("fit_mask_core_width_large", evaluation["warnings"])

    def test_fit_control_overrides_sanitize_tool_arguments(self):
        record = self._demo_record()
        anchor_wavelength = record["plot_data"]["anchor_wavelengths_A"][1]
        patch = {
            "task": "fit_control_patch",
            "source": "unit",
            "status": "tool_calls",
            "rationale": "sanitize bad model arguments",
            "tool_calls": [
                {
                    "name": "add_absorption_source",
                    "arguments": {
                        "transition_line_id": "CIV_1548",
                        "center_velocity_kms": 1.0e9,
                        "logN": -4.0,
                        "b_kms": 1.0e6,
                        "center_prior_sigma_kms": -5.0,
                        "reason": "bad raw values",
                    },
                },
                {
                    "name": "remove_continuum_anchor",
                    "arguments": {
                        "anchor_index": 1,
                        "reason": "remove biased current anchor",
                    },
                },
                {
                    "name": "set_fit_window",
                    "arguments": {
                        "transition_line_id": "CIV_1548",
                        "transition_half_width_kms": 1.0e6,
                        "local_fit_half_width_kms": -1.0,
                        "reason": "bad window",
                    },
                },
                {
                    "name": "set_fit_mask_interval",
                    "arguments": {
                        "transition_line_id": "CIV_1548",
                        "start_velocity_kms": -1.0e9,
                        "end_velocity_kms": 1.0e9,
                        "mask_kind": "invalid",
                        "reason": "bad mask",
                    },
                },
            ],
            "requires_refit": True,
            "applied": False,
        }

        overrides = build_fit_control_overrides(record, patch)

        source = overrides["source_seeds"][0]
        self.assertEqual(source["center_velocity_kms"], 5000.0)
        self.assertEqual(source["logN"], 10.0)
        self.assertEqual(source["b_kms"], 200.0)
        self.assertEqual(source["center_prior_sigma_kms"], 5.0)
        self.assertEqual(overrides["fit_windows"]["CIV_1548"]["transition_half_width_kms"], 1500.0)
        self.assertEqual(overrides["fit_windows"]["CIV_1548"]["local_fit_half_width_kms"], 40.0)
        self.assertEqual(overrides["fit_mask_intervals"][0]["start_velocity_kms"], -5000.0)
        self.assertEqual(overrides["fit_mask_intervals"][0]["end_velocity_kms"], 5000.0)
        self.assertEqual(overrides["fit_mask_intervals"][0]["mask_kind"], "exclude")
        self.assertEqual(overrides["continuum_anchor_remove_indices"], [])
        self.assertEqual(overrides["continuum_anchor_remove_wavelengths_A"], [anchor_wavelength])

    def test_merge_fit_control_overrides_dedupes_repeated_sources_and_masks(self):
        first = {
            "source_seeds": [
                {
                    "transition_line_id": "HI_LYA",
                    "center_velocity_kms": -379.0,
                    "seed_source": "add_absorption_source",
                    "reason": "first",
                }
            ],
            "fit_mask_intervals": [
                {
                    "transition_line_id": "HI_LYA",
                    "start_velocity_kms": -581.0,
                    "end_velocity_kms": -548.0,
                    "mask_kind": "exclude",
                    "reason": "first mask",
                }
            ],
        }
        second = {
            "source_seeds": [
                {
                    "transition_line_id": "HI_LYA",
                    "center_velocity_kms": -378.5,
                    "seed_source": "add_absorption_source",
                    "reason": "repeat",
                }
            ],
            "fit_mask_intervals": [
                {
                    "transition_line_id": "HI_LYA",
                    "start_velocity_kms": -580.0,
                    "end_velocity_kms": -549.0,
                    "mask_kind": "exclude",
                    "reason": "repeat mask",
                }
            ],
        }

        merged = merge_fit_control_overrides(first, second)

        self.assertEqual(len(merged["source_seeds"]), 1)
        self.assertEqual(len(merged["fit_mask_intervals"]), 1)
        self.assertEqual(merged["source_seeds"][0]["reason"], "repeat")

    def test_merge_fit_control_overrides_dedupes_removed_continuum_anchor_wavelengths(self):
        merged = merge_fit_control_overrides(
            {"continuum_anchor_remove_wavelengths_A": [9775.6], "tool_calls": []},
            {"continuum_anchor_remove_wavelengths_A": [9775.65, 9780.4], "tool_calls": []},
        )

        self.assertEqual(merged["continuum_anchor_remove_wavelengths_A"], [9775.6, 9780.4])

    def test_merge_fit_control_overrides_drops_source_overlapping_exclude_mask(self):
        merged = merge_fit_control_overrides(
            {
                "source_seeds": [
                    {
                        "transition_line_id": "MGII_2796",
                        "center_velocity_kms": 547.0,
                        "seed_source": "previous_fit_component",
                    },
                    {
                        "transition_line_id": "MGII_2796",
                        "center_velocity_kms": 474.0,
                        "center_prior_half_width_kms": 45.0,
                        "seed_source": "previous_fit_component",
                    },
                    {
                        "transition_line_id": "MGII_2796",
                        "center_velocity_kms": 64.0,
                        "seed_source": "previous_fit_component",
                    },
                    {
                        "transition_line_id": "MGII_2803",
                        "center_velocity_kms": 547.0,
                        "seed_source": "previous_fit_component",
                    },
                ],
                "tool_calls": [],
            },
            {
                "fit_mask_intervals": [
                    {
                        "transition_line_id": "MGII_2796",
                        "start_velocity_kms": 495.0,
                        "end_velocity_kms": 590.0,
                        "mask_kind": "exclude",
                    }
                ],
                "tool_calls": [],
            },
        )

        centers = [(item["transition_line_id"], item["center_velocity_kms"]) for item in merged["source_seeds"]]
        self.assertNotIn(("MGII_2796", 547.0), centers)
        self.assertNotIn(("MGII_2796", 474.0), centers)
        self.assertIn(("MGII_2796", 64.0), centers)
        self.assertIn(("MGII_2803", 547.0), centers)

    def test_remove_source_patch_removes_only_target_seed(self):
        spectrum = make_demo_quasar_spectrum(z_sys=2.6, line_id="CIV_doublet")
        record, window = build_review_record(
            spectrum=spectrum,
            line_id="CIV_doublet",
            z_sys=2.6,
            sample_id="remove_demo",
            source={"kind": "unit_test"},
        )
        fitted = record["fit_results"][0]["components"]
        self.assertGreaterEqual(len(fitted), 2)
        target = fitted[0]
        patch = {
            "task": "fit_control_patch",
            "source": "unit",
            "status": "tool_calls",
            "rationale": "remove one duplicate component",
            "tool_calls": [
                {
                    "sequence_index": 0,
                    "name": "remove_absorption_source",
                    "arguments": {
                        "component_index": target["component_index"],
                        "transition_line_id": target["transition_line_id"],
                        "reason": "duplicate component",
                    },
                }
            ],
            "requires_refit": True,
            "applied": False,
        }

        refit_record, overrides = refit_record_with_patch(record, window, patch, sample_id="remove_demo_refit")

        self.assertEqual(len(overrides["removed_sources"]), 1)
        self.assertEqual(overrides["removed_sources"][0]["component_index"], target["component_index"])
        self.assertTrue(refit_record["fit_results"][0]["success"])
        self.assertGreater(refit_record["fit_results"][0]["n_components_fitted"], 0)
        self.assertEqual(refit_record["fit_results"][0]["fit_control_applied"]["n_removed_sources"], 1)

    def test_apply_fit_control_patch_cli_writes_refit_outputs(self):
        with tempfile.TemporaryDirectory(prefix="astroagent_apply_patch_cli_") as tmpdir:
            outdir = Path(tmpdir)
            spectrum = make_demo_quasar_spectrum(z_sys=2.6, line_id="CIV_doublet")
            record, window = build_review_record(
                spectrum=spectrum,
                line_id="CIV_doublet",
                z_sys=2.6,
                sample_id="apply_cli_demo",
                source={"kind": "unit_test"},
            )
            paths = write_review_packet(record, window, outdir)
            transition_id = record["input"]["transitions"][0]["transition_line_id"]
            patch_path = outdir / "apply_cli_demo.fit_control_patch.json"
            patch_path.write_text(
                json.dumps(
                    {
                        "task": "fit_control_patch",
                        "source": "unit",
                        "status": "tool_calls",
                        "rationale": "add source",
                        "tool_calls": [
                            {
                                "name": "add_absorption_source",
                                "arguments": {
                                    "transition_line_id": transition_id,
                                    "center_velocity_kms": 130.0,
                                    "reason": "test seed",
                                },
                            }
                        ],
                        "requires_refit": True,
                        "applied": False,
                    }
                )
                + "\n",
                encoding="utf-8",
            )

            result = subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "scripts" / "apply_fit_control_patch.py"),
                    "--review-json",
                    str(paths["json"]),
                    "--window-csv",
                    str(paths["csv"]),
                    "--patch-json",
                    str(patch_path),
                    "--sample-id",
                    "apply_cli_demo_refit",
                    "--output-dir",
                    str(outdir),
                ],
                cwd=ROOT,
                check=True,
                capture_output=True,
                text=True,
            )

            refit_json = outdir / "apply_cli_demo_refit.review.json"
            self.assertIn(str(refit_json), result.stdout)
            self.assertIn("fit-control decision:", result.stdout)
            self.assertTrue(refit_json.exists())
            self.assertTrue((outdir / "apply_cli_demo_refit.plot.png").exists())
            refit = json.loads(refit_json.read_text(encoding="utf-8"))
            self.assertGreaterEqual(refit["fit_control_application"]["summary"]["n_source_seeds"], 1)
            self.assertEqual(refit["fit_control_application"]["controls"]["source_detection_mode"], "controlled")
            self.assertIn("fit_control_evaluation", refit)

    def test_unified_cli_apply_patch_infers_window_csv(self):
        with tempfile.TemporaryDirectory(prefix="astroagent_unified_apply_cli_") as tmpdir:
            outdir = Path(tmpdir)
            spectrum = make_demo_quasar_spectrum(z_sys=2.6, line_id="CIV_doublet")
            record, window = build_review_record(
                spectrum=spectrum,
                line_id="CIV_doublet",
                z_sys=2.6,
                sample_id="unified_apply_demo",
                source={"kind": "unit_test"},
            )
            paths = write_review_packet(record, window, outdir)
            transition_id = record["input"]["transitions"][0]["transition_line_id"]
            patch_path = outdir / "unified_apply_demo.fit_control_patch.json"
            patch_path.write_text(
                json.dumps(
                    {
                        "task": "fit_control_patch",
                        "source": "unit",
                        "status": "tool_calls",
                        "rationale": "add source",
                        "tool_calls": [
                            {
                                "name": "add_absorption_source",
                                "arguments": {
                                    "transition_line_id": transition_id,
                                    "center_velocity_kms": 130.0,
                                    "reason": "test seed",
                                },
                            }
                        ],
                        "requires_refit": True,
                        "applied": False,
                    }
                )
                + "\n",
                encoding="utf-8",
            )

            result = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "astroagent.cli.main",
                    "apply-patch",
                    "--review-json",
                    str(paths["json"]),
                    "--patch-json",
                    str(patch_path),
                    "--sample-id",
                    "unified_apply_demo_refit",
                    "--output-dir",
                    str(outdir),
                ],
                cwd=ROOT,
                check=True,
                capture_output=True,
                text=True,
            )

            refit_json = outdir / "unified_apply_demo_refit.review.json"
            self.assertIn(str(refit_json), result.stdout)
            self.assertTrue(refit_json.exists())

    def test_agent_refit_does_not_auto_detect_new_sources_after_llm_seed(self):
        z_sys = 0.0
        rest_A = 2796.352
        wavelength = np.arange(rest_A - 7.0, rest_A + 7.0, 0.08)
        velocity = (wavelength / rest_A - 1.0) * 299792.458
        flux = 1.0 - 0.35 * np.exp(-0.5 * (velocity / 22.0) ** 2)
        flux -= 0.28 * np.exp(-0.5 * ((velocity - 180.0) / 20.0) ** 2)
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
            sample_id="controlled_refit_demo",
            source={"kind": "unit_test"},
            half_width_kms=700.0,
        )
        original_components = len(record["fit_results"][0]["components"])
        patch = {
            "task": "fit_control_patch",
            "source": "unit",
            "status": "tool_calls",
            "rationale": "controlled seed",
            "tool_calls": [
                {
                    "name": "add_absorption_source",
                    "arguments": {
                        "transition_line_id": "MGII_2796",
                        "center_velocity_kms": -180.0,
                        "reason": "explicit added seed",
                    },
                }
            ],
            "requires_refit": True,
            "applied": False,
        }

        refit_record, controls = refit_record_with_patch(record, window, patch)

        peaks = refit_record["fit_results"][0]["transition_frames"][0]["peaks"]
        self.assertEqual(refit_record["fit_results"][0]["transition_frames"][0]["peak_detection"]["source_detection_mode"], "controlled")
        self.assertEqual(len(peaks), original_components + 1)
        self.assertEqual(controls["source_detection_mode"], "controlled")
        self.assertTrue(all(peak["seed_source"] != "absorption_trough" for peak in peaks))

    def test_controlled_refit_does_not_carry_previous_source_inside_new_fit_mask(self):
        z_sys = 0.0
        rest_A = 2796.352
        wavelength = np.arange(rest_A - 7.0, rest_A + 7.0, 0.08)
        velocity = (wavelength / rest_A - 1.0) * 299792.458
        flux = 1.0 - 0.35 * np.exp(-0.5 * (velocity / 22.0) ** 2)
        flux -= 0.30 * np.exp(-0.5 * ((velocity - 520.0) / 18.0) ** 2)
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
            sample_id="controlled_mask_demo",
            source={"kind": "unit_test"},
            half_width_kms=700.0,
        )
        patch = {
            "task": "fit_control_patch",
            "source": "unit",
            "status": "tool_calls",
            "rationale": "mask edge",
            "tool_calls": [
                {
                    "name": "set_fit_mask_interval",
                    "arguments": {
                        "transition_line_id": "MGII_2796",
                        "start_velocity_kms": 480.0,
                        "end_velocity_kms": 560.0,
                        "mask_kind": "exclude",
                        "reason": "edge contaminant",
                    },
                }
            ],
            "requires_refit": True,
            "applied": False,
        }

        refit_record, controls = refit_record_with_patch(record, window, patch)

        centers = [source["center_velocity_kms"] for source in controls["source_seeds"] if source["transition_line_id"] == "MGII_2796"]
        self.assertTrue(all(not (480.0 <= center <= 560.0) for center in centers))
        peaks = refit_record["fit_results"][0]["transition_frames"][0]["peaks"]
        self.assertTrue(all(not (480.0 <= float(peak.get("seed_velocity_kms", 0.0)) <= 560.0) for peak in peaks))


if __name__ == "__main__":
    unittest.main()
