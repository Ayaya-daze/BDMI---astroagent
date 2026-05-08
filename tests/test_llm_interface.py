import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from astroagent.agent.fit_control import (
    append_fit_control_patch,
    build_fit_control_patch,
    summarize_pending_fit_control,
)
from astroagent.agent.fit_control import build_fit_control_overrides, evaluate_refit_patch, merge_fit_control_overrides, refit_record_with_patch
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
        self.assertEqual(overrides["continuum_anchor_nodes"][0]["continuum_flux"], 1.08)
        self.assertEqual(refit_record["sample_id"], "apply_demo_refit")
        self.assertTrue(refit_record["fit_control_patches"][0]["applied"])
        self.assertIn(refit_record["fit_control_patches"][0]["application_result"]["decision"], {"accepted", "needs_human_review", "rejected"})
        self.assertIn("fit_control_evaluation", refit_record)
        self.assertTrue(refit_record["fit_control_application"]["applied"])
        self.assertIn("evaluation", refit_record["fit_control_application"])
        self.assertEqual(refit_record["fit_results"][0]["fit_control_applied"]["n_source_seeds"], 1)
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
        self.assertIn("fit_rms_worsened", evaluation["reasons"])
        self.assertIn("component_count_exploded", evaluation["reasons"])
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
            self.assertEqual(refit["fit_control_application"]["summary"]["n_source_seeds"], 1)
            self.assertIn("fit_control_evaluation", refit)


if __name__ == "__main__":
    unittest.main()
