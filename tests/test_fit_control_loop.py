import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from astroagent.agent.fit_control import empty_fit_control_overrides, merge_fit_control_overrides
from astroagent.agent.loop import (
    _candidate_score,
    _coerce_distinct_repeated_updates_to_adds,
    _prepare_round_controls,
    run_fit_control_loop,
)
from astroagent.agent.llm import LLMResult, OfflineReviewClient
from astroagent.review.packet import build_review_record, make_demo_quasar_spectrum


class OneToolCallClient:
    def complete(self, messages, *, temperature=0.0, tools=None):
        return LLMResult(
            content="",
            model="unit-loop-tool-client",
            raw={"unit": True},
            tool_calls=[
                {
                    "id": "call_1",
                    "type": "function",
                    "function": {
                        "name": "add_absorption_source",
                        "arguments": json.dumps(
                            {
                                "transition_line_id": "CIV_1548",
                                "center_velocity_kms": 120.0,
                                "reason": "unit loop refit seed",
                            }
                        ),
                    },
                },
                {
                    "id": "call_2",
                    "type": "function",
                    "function": {
                        "name": "request_refit",
                        "arguments": json.dumps({"reason": "unit loop refit"}),
                    },
                },
            ],
        )


class SequentialToolCallClient:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = 0

    def complete(self, messages, *, temperature=0.0, tools=None):
        index = min(self.calls, len(self.responses) - 1)
        self.calls += 1
        return LLMResult(content="", model="unit-sequential-tool-client", raw={"unit": True}, tool_calls=self.responses[index])


class FitControlLoopTest(unittest.TestCase):
    def _demo_record_and_window(self):
        spectrum = make_demo_quasar_spectrum(z_sys=2.6, line_id="CIV_doublet")
        return build_review_record(
            spectrum=spectrum,
            line_id="CIV_doublet",
            z_sys=2.6,
            sample_id="loop_demo",
            source={"kind": "unit_test"},
        )

    def test_offline_loop_stops_without_refit(self):
        record, window = self._demo_record_and_window()
        with tempfile.TemporaryDirectory(prefix="astroagent_loop_offline_") as tmpdir:
            result = run_fit_control_loop(
                record=record,
                window=window,
                client=OfflineReviewClient(),
                output_dir=tmpdir,
                max_rounds=2,
                force=True,
            )

        self.assertEqual(result.stop_reason, "no_refit_requested")
        self.assertEqual(result.final_record["sample_id"], "loop_demo")
        self.assertEqual(result.final_record["fit_control_loop"]["n_rounds"], 1)
        self.assertEqual(result.final_record["fit_control_loop"]["rounds"][0]["control_status"], "no_action")
        self.assertEqual(result.output_paths, [])
        self.assertNotIn("fit_control_loop", record)

    def test_good_fit_is_skipped_unless_forced(self):
        record, window = self._demo_record_and_window()
        record["fit_results"][0]["quality"] = "good"
        record["fit_results"][0]["agent_review"] = {"required": False, "reasons": []}
        with tempfile.TemporaryDirectory(prefix="astroagent_loop_good_") as tmpdir:
            result = run_fit_control_loop(
                record=record,
                window=window,
                client=OneToolCallClient(),
                output_dir=tmpdir,
                max_rounds=1,
            )

        self.assertEqual(result.stop_reason, "not_needed_good_fit")
        self.assertEqual(result.history, [])
        self.assertEqual(result.output_paths, [])

    def test_loop_applies_tool_patch_and_writes_refit_record(self):
        record, window = self._demo_record_and_window()
        with tempfile.TemporaryDirectory(prefix="astroagent_loop_tool_") as tmpdir:
            result = run_fit_control_loop(
                record=record,
                window=window,
                client=OneToolCallClient(),
                output_dir=tmpdir,
                max_rounds=1,
                force=True,
            )
            output_paths = result.output_paths[0]
            final_json = output_paths["json"]
            saved = json.loads(final_json.read_text(encoding="utf-8"))
            self.assertTrue(output_paths["overview_png"].exists())
            self.assertTrue(output_paths["plot_png"].exists())

        self.assertIn(
            result.stop_reason,
            {
                "max_rounds_reached",
                "needs_human_review",
                "refit_rejected",
                "needs_human_review_best_candidate_selected",
                "refit_rejected_best_candidate_selected",
            },
        )
        self.assertEqual(result.final_record["sample_id"], "loop_demo_loop1")
        self.assertEqual(saved["fit_control_loop"]["task"], "fit_control_loop")
        self.assertEqual(saved["fit_control_loop"]["n_rounds"], 1)
        self.assertEqual(saved["fit_control_loop"]["rounds"][0]["n_tool_calls"], 2)
        self.assertIn(saved["fit_control_loop"]["rounds"][0]["decision"], {"accepted", "needs_human_review", "rejected"})
        if saved["fit_control_loop"]["rounds"][0]["decision"] == "accepted":
            self.assertTrue(saved["fit_control_loop"]["rounds"][0]["advanced"])
        else:
            self.assertFalse(saved["fit_control_loop"]["rounds"][0]["advanced"])

    def test_repeated_distinct_component_updates_are_coerced_to_added_sources(self):
        patch = {
            "task": "fit_control_patch",
            "tool_calls": [
                {
                    "name": "update_absorption_source",
                    "arguments": {
                        "component_index": 1,
                        "transition_line_id": "HI_LYA",
                        "center_velocity_kms": -249.0,
                        "reason": "tighten existing component",
                    },
                },
                {
                    "name": "update_absorption_source",
                    "arguments": {
                        "component_index": 1,
                        "transition_line_id": "HI_LYA",
                        "center_velocity_kms": -380.0,
                        "reason": "separate shoulder",
                    },
                },
            ],
            "requires_refit": True,
        }

        coerced = _coerce_distinct_repeated_updates_to_adds(patch)

        self.assertEqual(coerced["tool_calls"][0]["name"], "update_absorption_source")
        self.assertEqual(coerced["tool_calls"][1]["name"], "add_absorption_source")
        self.assertNotIn("component_index", coerced["tool_calls"][1]["arguments"])

    def test_round_preparation_does_not_reinterpret_committed_component_indexes(self):
        record, window = self._demo_record_and_window()
        transition_id = record["input"]["transitions"][0]["transition_line_id"]
        active_controls = merge_fit_control_overrides(
            empty_fit_control_overrides(),
            {
                "source_seeds": [
                    {
                        "transition_line_id": transition_id,
                        "center_velocity_kms": 80.0,
                        "seed_source": "update_absorption_source",
                        "replace_component_index": 0,
                        "replace_center_velocity_kms": 12.0,
                    }
                ],
                "tool_calls": [
                    {
                        "name": "update_absorption_source",
                        "arguments": {
                            "component_index": 0,
                            "transition_line_id": transition_id,
                            "center_velocity_kms": 80.0,
                            "reason": "already interpreted in round one",
                        },
                    }
                ],
            },
        )
        control = {
            "task": "fit_control",
            "status": "tool_calls",
            "rationale": "round two",
            "tool_calls": [
                {
                    "name": "add_absorption_source",
                    "arguments": {
                        "transition_line_id": transition_id,
                        "center_velocity_kms": -80.0,
                        "reason": "round two source",
                    },
                }
            ],
        }

        _, round_overrides, controls, effective_patch = _prepare_round_controls(
            record,
            control,
            active_controls,
            round_index=2,
        )

        self.assertEqual(len(round_overrides["source_seeds"]), 1)
        self.assertNotIn("replace_component_index", round_overrides["source_seeds"][0])
        self.assertEqual(len(effective_patch["tool_calls"]), 2)
        replaced = [source for source in controls["source_seeds"] if "replace_component_index" in source]
        self.assertEqual(len(replaced), 1)
        self.assertEqual(replaced[0]["replace_component_index"], 0)

    def test_candidate_score_penalizes_core_mask_more_than_context_mask(self):
        base = {
            "metrics": {
                "refit": {
                    "fit_rms": 2.0,
                    "n_components": 4,
                    "source_work_window": {
                        "fit_rms": 2.0,
                        "max_abs_residual_sigma": 8.0,
                    },
                },
                "delta": {
                    "n_fit_pixels": -20,
                    "fit_mask": {
                        "n_intervals": 1,
                        "n_core_intervals": 1,
                        "core_width_kms": 80.0,
                        "boundary_width_kms": 0.0,
                        "context_width_kms": 0.0,
                        "max_interval_width_kms": 80.0,
                    },
                },
            }
        }
        context = json.loads(json.dumps(base))
        context["metrics"]["delta"]["fit_mask"].update(
            {
                "n_core_intervals": 0,
                "core_width_kms": 0.0,
                "context_width_kms": 80.0,
            }
        )

        self.assertLess(_candidate_score(context), _candidate_score(base))


if __name__ == "__main__":
    unittest.main()
