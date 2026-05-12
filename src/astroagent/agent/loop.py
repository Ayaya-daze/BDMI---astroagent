from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import pandas as pd

from astroagent.agent.audit import render_fit_control_audit_report
from astroagent.agent.fit_control import (
    build_fit_control_overrides,
    build_fit_control_patch,
    empty_fit_control_overrides,
    merge_fit_control_overrides,
    refit_record_with_overrides,
)
from astroagent.agent.llm import LLMClient, run_fit_control
from astroagent.review.packet import write_review_packet


@dataclass(frozen=True)
class FitControlLoopResult:
    final_record: dict[str, Any]
    final_image_paths: list[Path] | None
    history: list[dict[str, Any]]
    stop_reason: str
    output_paths: list[dict[str, Path]]
    audit_paths: dict[str, Path] | None = None


@dataclass(frozen=True)
class FitControlAgentState:
    """Committed mainline state for the bounded fit-control agent."""

    record: dict[str, Any]
    image_paths: list[Path] | None
    controls: dict[str, Any]


@dataclass(frozen=True)
class FitControlCandidate:
    """One executed candidate branch produced from a single round decision."""

    record: dict[str, Any]
    image_paths: list[Path] | None
    controls: dict[str, Any]
    output_paths: dict[str, Path]
    evaluation: dict[str, Any]
    decision: str
    score: tuple[float, float, int]
    commit_to_mainline: bool
    eligible_for_best: bool


def run_fit_control_loop(
    *,
    record: dict[str, Any],
    window: pd.DataFrame,
    client: LLMClient,
    output_dir: str | Path,
    initial_plot_image_path: str | Path | Sequence[str | Path] | None = None,
    max_rounds: int = 3,
    hard_max_rounds: int | None = None,
    temperature: float = 0.0,
    sample_id_prefix: str | None = None,
    force: bool = False,
) -> FitControlLoopResult:
    """Run a bounded fit-control agent/tool/refit loop.

    The loop is intentionally small: the LLM proposes tool calls, the patch
    layer normalizes them, the deterministic fitter refits, and the gate decides
    whether the refit can become the next state. Final science adjudication stays
    outside this loop.
    """
    if max_rounds < 1:
        raise ValueError("max_rounds must be >= 1")
    effective_hard_max_rounds = int(hard_max_rounds) if hard_max_rounds is not None else max(int(max_rounds), 6)
    if effective_hard_max_rounds < int(max_rounds):
        raise ValueError("hard_max_rounds must be >= max_rounds")

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    state = FitControlAgentState(
        record=deepcopy(record),
        image_paths=_image_paths(initial_plot_image_path),
        controls=empty_fit_control_overrides(),
    )
    history: list[dict[str, Any]] = []
    output_paths: list[dict[str, Path]] = []
    stop_reason = "max_rounds_reached"
    approved_round_limit = int(max_rounds)
    base_id = sample_id_prefix or str(record.get("sample_id", "fit_control_loop"))
    best_record: dict[str, Any] | None = None
    best_plot: list[Path] | None = None
    best_output_paths: dict[str, Path] | None = None
    best_score: tuple[float, float, int] | None = None
    trial_controls = state.controls

    if not force and not _fit_control_needed(state.record):
        stop_reason = "not_needed_good_fit"
        state.record["fit_control_loop"] = _loop_record(history, stop_reason=stop_reason)
        audit_paths = _write_loop_audit(
            record=state.record,
            history=history,
            stop_reason=stop_reason,
            output_path=output_path,
            output_paths=output_paths,
        )
        return FitControlLoopResult(
            final_record=state.record,
            final_image_paths=state.image_paths,
            history=history,
            stop_reason=stop_reason,
            output_paths=output_paths,
            audit_paths=audit_paths,
        )

    round_index = 1
    while round_index <= approved_round_limit:
        image_paths = state.image_paths
        rejected_feedback = state.record.get("fit_control_last_rejected_refit")
        if image_paths and isinstance(rejected_feedback, dict):
            extra_images = [
                value
                for key in ("overview_png", "plot_png")
                if (value := rejected_feedback.get(key))
            ]
            if extra_images:
                image_paths = [*image_paths, *_image_paths(extra_images)]
        control = run_fit_control(
            state.record,
            client,
            temperature=temperature,
            plot_image_path=image_paths,
        )
        patch, round_overrides, candidate_controls, effective_patch = _prepare_round_controls(
            state.record,
            control,
            trial_controls,
            round_index=round_index,
        )
        round_entry: dict[str, Any] = {
            "round_index": round_index,
            "input_sample_id": state.record.get("sample_id"),
            "control": control,
            "decision_control": control,
            "patch": effective_patch,
            "new_patch": patch,
            "round_overrides": round_overrides,
            "decision": None,
            "output_sample_id": None,
            "advanced": False,
            "candidate_score": None,
            "selected_candidate": False,
            "budget_request": None,
            "budget_decision": None,
            "assessment_control": None,
            "assessment_summary": None,
        }

        if not patch.get("requires_refit"):
            budget_decision = _decide_budget_request(
                patch,
                round_index=round_index,
                approved_round_limit=approved_round_limit,
                hard_max_rounds=effective_hard_max_rounds,
                decision="no_refit",
            )
            round_entry["budget_decision"] = budget_decision
            stop_reason = "no_refit_requested"
            history.append(round_entry)
            break

        refit_sample_id = f"{base_id}_loop{round_index}"
        candidate = _execute_candidate(
            state,
            window,
            effective_patch,
            candidate_controls,
            round_overrides,
            output_path,
            sample_id=refit_sample_id,
        )

        evaluation = candidate.evaluation
        decision = candidate.decision
        round_entry.update(
            {
                "decision": decision,
                "output_sample_id": candidate.record.get("sample_id"),
                "gate_reasons": evaluation.get("reasons", []),
                "gate_warnings": evaluation.get("warnings", []),
                "gate_metrics": evaluation.get("metrics", {}),
                "overrides_summary": _overrides_summary(candidate.controls),
            }
        )

        assessment_control, assessment_patch = _run_round_assessment(
            candidate=candidate,
            client=client,
            temperature=temperature,
            patch=patch,
            evaluation=evaluation,
            history=[*history, round_entry],
            round_index=round_index,
            approved_round_limit=approved_round_limit,
            hard_max_rounds=effective_hard_max_rounds,
        )
        round_entry["assessment_control"] = assessment_control
        round_entry["assessment_summary"] = assessment_control.get("rationale")
        budget_patch = _budget_only_patch(assessment_patch)
        if budget_patch != assessment_patch:
            round_entry["assessment_ignored_tool_calls"] = _non_budget_tool_calls(assessment_patch)
        round_entry["budget_request"] = _budget_request_from_patch(budget_patch)
        budget_decision = _decide_budget_request(
            budget_patch,
            round_index=round_index,
            approved_round_limit=approved_round_limit,
            hard_max_rounds=effective_hard_max_rounds,
            decision=decision,
        )
        if budget_decision.get("approved"):
            approved_round_limit = int(budget_decision["new_round_limit"])
        round_entry["budget_decision"] = budget_decision
        is_last_round = round_index == approved_round_limit
        candidate_score = candidate.score
        round_entry["candidate_score"] = candidate_score
        history.append(round_entry)
        output_paths.append(candidate.output_paths)
        if not candidate.commit_to_mainline:
            stop_reason = "refit_rejected" if is_last_round else "retry_after_rejected_refit"
            candidate.record["fit_control_loop"] = _loop_record(history, stop_reason=stop_reason)
            write_review_packet(candidate.record, window, output_path)
            state.record["fit_control_loop"] = _loop_record(history, stop_reason=None)
            state.record["fit_control_last_rejected_refit"] = {
                "sample_id": candidate.record.get("sample_id"),
                "evaluation": evaluation,
                "patch": patch,
                "overview_png": str(candidate.output_paths.get("overview_png", "")),
                "plot_png": str(candidate.output_paths.get("plot_png", "")),
            }
            state.record["fit_control_last_refit_feedback"] = {
                "sample_id": candidate.record.get("sample_id"),
                "evaluation": evaluation,
                "patch": patch,
                "fit_summary": candidate.record.get("fit_results", [{}])[0],
                "assessment_control": assessment_control,
            }
            trial_controls = candidate.controls
            if is_last_round:
                stop_reason = "refit_rejected"
                break
            round_index += 1
            continue
        if decision == "needs_human_review":
            stop_reason = "needs_human_review" if is_last_round else "continue_after_human_review_warning"
            if candidate.eligible_for_best and _is_better_candidate(candidate_score, best_score):
                _clear_selected_candidate(history)
                round_entry["selected_candidate"] = True
                best_score = candidate_score
                best_record = candidate.record
                best_plot = candidate.image_paths
                best_output_paths = candidate.output_paths
            state = _commit_candidate(candidate)
            trial_controls = state.controls
            if is_last_round:
                stop_reason = "needs_human_review"
                break
            state.record["fit_control_loop"] = _loop_record(history, stop_reason=None)
            state.record["fit_control_last_refit_feedback"] = {
                "sample_id": candidate.record.get("sample_id"),
                "evaluation": evaluation,
                "patch": patch,
                "fit_summary": candidate.record.get("fit_results", [{}])[0],
                "budget_decision": budget_decision,
                "assessment_control": assessment_control,
            }
            round_index += 1
            continue

        round_entry["advanced"] = True
        if candidate.eligible_for_best and _is_better_candidate(candidate_score, best_score):
            _clear_selected_candidate(history)
            round_entry["selected_candidate"] = True
            best_score = candidate_score
            best_record = candidate.record
            best_plot = candidate.image_paths
            best_output_paths = candidate.output_paths
        state = _commit_candidate(candidate)
        state.record["fit_control_last_refit_feedback"] = {
            "sample_id": candidate.record.get("sample_id"),
            "evaluation": evaluation,
            "patch": patch,
            "fit_summary": candidate.record.get("fit_results", [{}])[0],
            "budget_decision": budget_decision,
            "assessment_control": assessment_control,
        }
        trial_controls = state.controls
        round_index += 1

    else:
        stop_reason = "max_rounds_reached"

    current_record = state.record
    current_plot = state.image_paths
    if best_record is not None and best_record.get("sample_id") != current_record.get("sample_id"):
        current_record = best_record
        current_plot = best_plot
        stop_reason = f"{stop_reason}_best_candidate_selected"
        current_record["fit_control_selected_candidate"] = {
            "sample_id": current_record.get("sample_id"),
            "selection_policy": "minimize fit_rms with penalty for excessive fit-pixel removal",
            "score": list(best_score) if best_score is not None else None,
        }
        if best_output_paths is not None and output_paths:
            output_paths[-1] = best_output_paths

    current_record["fit_control_loop"] = _loop_record(history, stop_reason=stop_reason)
    if output_paths and current_record.get("sample_id") != record.get("sample_id"):
        final_paths = write_review_packet(current_record, window, output_path)
        output_paths[-1] = final_paths
        current_plot = _image_paths([final_paths["overview_png"], final_paths["plot_png"]])
    audit_paths = _write_loop_audit(
        record=current_record,
        history=history,
        stop_reason=stop_reason,
        output_path=output_path,
        output_paths=output_paths,
    )
    if output_paths and current_record.get("sample_id") != record.get("sample_id"):
        final_paths = write_review_packet(current_record, window, output_path)
        output_paths[-1] = final_paths
    return FitControlLoopResult(
        final_record=current_record,
        final_image_paths=current_plot,
        history=history,
        stop_reason=stop_reason,
        output_paths=output_paths,
        audit_paths=audit_paths,
    )


def _write_loop_audit(
    *,
    record: dict[str, Any],
    history: list[dict[str, Any]],
    stop_reason: str,
    output_path: Path,
    output_paths: list[dict[str, Path]],
) -> dict[str, Path]:
    audit_paths = render_fit_control_audit_report(
        record=record,
        history=history,
        stop_reason=stop_reason,
        output_dir=output_path,
        output_paths=output_paths,
    )
    loop = record.setdefault("fit_control_loop", _loop_record(history, stop_reason=stop_reason))
    loop["audit_report_paths"] = {key: str(value) for key, value in audit_paths.items()}
    return audit_paths


def _run_round_assessment(
    *,
    candidate: FitControlCandidate,
    client: LLMClient,
    temperature: float,
    patch: dict[str, Any],
    evaluation: dict[str, Any],
    history: list[dict[str, Any]],
    round_index: int,
    approved_round_limit: int,
    hard_max_rounds: int,
) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
    """Let the agent inspect this same round's refit result and decide whether to continue."""
    feedback_record = deepcopy(candidate.record)
    feedback_record["fit_control_loop"] = _loop_record(history, stop_reason="round_assessment")
    feedback_record["fit_control_last_refit_feedback"] = {
        "sample_id": candidate.record.get("sample_id"),
        "evaluation": evaluation,
        "patch": patch,
        "fit_summary": candidate.record.get("fit_results", [{}])[0],
        "round_assessment": {
            "round_index": int(round_index),
            "approved_round_limit": int(approved_round_limit),
            "hard_max_rounds": int(hard_max_rounds),
            "instruction": (
                "This is the assessment step of the same experiment round. Briefly analyze whether the refit improved, "
                "worsened, or needs human review. Return no_action/inspect if the loop should stop. Call request_more_budget "
                "only if the next round has a concrete justified experiment."
            ),
        },
    }
    control = run_fit_control(
        feedback_record,
        client,
        temperature=temperature,
        plot_image_path=candidate.image_paths,
    )
    assessment_patch = build_fit_control_patch(control, source=f"round_assessment_{round_index}")
    return control, assessment_patch


def _loop_record(history: list[dict[str, Any]], stop_reason: str | None) -> dict[str, Any]:
    selected_sample_id = None
    selected_entries = [entry for entry in history if entry.get("selected_candidate")]
    if selected_entries:
        selected_sample_id = selected_entries[-1].get("output_sample_id")
    return {
        "task": "fit_control_loop",
        "n_rounds": int(len(history)),
        "stop_reason": stop_reason,
        "selected_sample_id": selected_sample_id,
        "rounds": [
            {
                "round_index": int(entry["round_index"]),
                "input_sample_id": entry.get("input_sample_id"),
                "output_sample_id": entry.get("output_sample_id"),
                "control_status": entry.get("control", {}).get("status"),
                "decision_control_status": entry.get("decision_control", {}).get("status")
                if isinstance(entry.get("decision_control"), dict)
                else None,
                "n_tool_calls": int(len(entry.get("patch", {}).get("tool_calls", []))),
                "n_new_tool_calls": int(len(entry.get("new_patch", {}).get("tool_calls", []))),
                "new_tool_calls": _public_tool_calls(entry.get("new_patch", {}).get("tool_calls", [])),
                "requires_refit": bool(entry.get("patch", {}).get("requires_refit")),
                "decision": entry.get("decision"),
                "advanced": bool(entry.get("advanced")),
                "rationale": entry.get("new_patch", {}).get("rationale") or entry.get("patch", {}).get("rationale"),
                "overrides_summary": entry.get("overrides_summary", {}),
                "candidate_score": list(entry["candidate_score"]) if entry.get("candidate_score") is not None else None,
                "selected_candidate": bool(entry.get("selected_candidate")),
                "budget_request": entry.get("budget_request"),
                "budget_decision": entry.get("budget_decision"),
                "assessment_control_status": entry.get("assessment_control", {}).get("status")
                if isinstance(entry.get("assessment_control"), dict)
                else None,
                "assessment_summary": entry.get("assessment_summary"),
                "gate_reasons": entry.get("gate_reasons", []),
                "gate_warnings": entry.get("gate_warnings", []),
                "gate_metrics": entry.get("gate_metrics", {}),
            }
            for entry in history
        ],
    }


def _public_tool_calls(tool_calls: list[dict[str, Any]]) -> list[dict[str, Any]]:
    public: list[dict[str, Any]] = []
    for call in tool_calls:
        args = call.get("arguments", {})
        item = {
            "name": call.get("name"),
            "reason": args.get("reason") if isinstance(args, dict) else None,
            "arguments": args if isinstance(args, dict) else {},
        }
        public.append(item)
    return public


def _budget_request_from_patch(patch: dict[str, Any]) -> dict[str, Any] | None:
    for call in patch.get("tool_calls", []):
        if call.get("name") == "request_more_budget":
            args = call.get("arguments", {})
            return args if isinstance(args, dict) else {}
    return None


def _budget_only_patch(patch: dict[str, Any] | None) -> dict[str, Any]:
    if not isinstance(patch, dict):
        return {"tool_calls": []}
    updated = deepcopy(patch)
    updated["tool_calls"] = [
        call
        for call in patch.get("tool_calls", [])
        if isinstance(call, dict) and call.get("name") == "request_more_budget"
    ]
    updated["requires_refit"] = False
    return updated


def _non_budget_tool_calls(patch: dict[str, Any] | None) -> list[dict[str, Any]]:
    if not isinstance(patch, dict):
        return []
    return [
        call
        for call in patch.get("tool_calls", [])
        if isinstance(call, dict) and call.get("name") != "request_more_budget"
    ]


def _decide_budget_request(
    patch: dict[str, Any],
    *,
    round_index: int,
    approved_round_limit: int,
    hard_max_rounds: int,
    decision: str | None,
) -> dict[str, Any]:
    request = _budget_request_from_patch(patch)
    if request is None:
        return {
            "requested": False,
            "approved": False,
            "round_limit": int(approved_round_limit),
            "hard_max_rounds": int(hard_max_rounds),
        }
    requested_rounds = max(1, int(_score_float(request.get("requested_rounds"), default=1.0)))
    if approved_round_limit >= hard_max_rounds:
        return {
            "requested": True,
            "approved": False,
            "reason": "hard_max_rounds_reached",
            "round_limit": int(approved_round_limit),
            "hard_max_rounds": int(hard_max_rounds),
            "request": request,
        }
    if decision == "rejected":
        return {
            "requested": True,
            "approved": False,
            "reason": "last_refit_rejected",
            "round_limit": int(approved_round_limit),
            "hard_max_rounds": int(hard_max_rounds),
            "request": request,
        }
    if decision == "no_refit":
        return {
            "requested": True,
            "approved": False,
            "reason": "no_refit_result_to_review",
            "round_limit": int(approved_round_limit),
            "hard_max_rounds": int(hard_max_rounds),
            "request": request,
        }
    new_limit = min(hard_max_rounds, max(approved_round_limit, round_index) + requested_rounds)
    return {
        "requested": True,
        "approved": new_limit > approved_round_limit,
        "reason": "approved_for_agent_experiment" if new_limit > approved_round_limit else "no_additional_rounds_available",
        "round_limit": int(approved_round_limit),
        "new_round_limit": int(new_limit),
        "hard_max_rounds": int(hard_max_rounds),
        "request": request,
    }


def _image_paths(value: str | Path | Sequence[str | Path] | None) -> list[Path] | None:
    if value is None:
        return None
    if isinstance(value, (list, tuple)):
        return [Path(item) for item in value]
    return [Path(value)]


def _prepare_round_controls(
    record: dict[str, Any],
    control: dict[str, Any],
    active_controls: dict[str, Any],
    *,
    round_index: int,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]]:
    """Normalize one LLM decision and interpret it exactly once."""
    patch = build_fit_control_patch(control, source=f"llm_fit_control_round_{round_index}")
    patch = _coerce_distinct_repeated_updates_to_adds(patch)
    patch = _drop_regressive_broad_updates(patch, active_controls)
    round_overrides = build_fit_control_overrides(record, patch)
    candidate_controls = merge_fit_control_overrides(active_controls, round_overrides)
    effective_patch = deepcopy(patch)
    effective_patch["applied_controls"] = _public_fit_control_controls(candidate_controls)
    effective_patch["requires_refit"] = bool(patch.get("requires_refit"))
    return patch, round_overrides, candidate_controls, effective_patch


def _public_fit_control_controls(controls: dict[str, Any]) -> dict[str, Any]:
    public = deepcopy(controls)
    public.pop("tool_calls", None)
    return public


def _execute_candidate(
    state: FitControlAgentState,
    window: pd.DataFrame,
    patch: dict[str, Any],
    controls: dict[str, Any],
    evaluation_controls: dict[str, Any],
    output_path: Path,
    *,
    sample_id: str,
) -> FitControlCandidate:
    record, applied_controls = refit_record_with_overrides(
        state.record,
        window,
        patch,
        controls,
        sample_id=sample_id,
        evaluation_overrides=evaluation_controls,
    )
    evaluation = record.get("fit_control_evaluation", {})
    decision = str(evaluation.get("decision", "unknown"))
    paths = write_review_packet(record, window, output_path)
    image_paths = _image_paths([paths["overview_png"], paths["plot_png"]])
    return FitControlCandidate(
        record=record,
        image_paths=image_paths,
        controls=applied_controls,
        output_paths=paths,
        evaluation=evaluation,
        decision=decision,
        score=_candidate_score(evaluation),
        commit_to_mainline=decision != "rejected",
        eligible_for_best=decision != "rejected",
    )


def _commit_candidate(candidate: FitControlCandidate) -> FitControlAgentState:
    return FitControlAgentState(
        record=candidate.record,
        image_paths=candidate.image_paths,
        controls=candidate.controls,
    )


def _overrides_summary(overrides: dict[str, Any]) -> dict[str, int]:
    return {
        "n_source_seeds": int(len(overrides.get("source_seeds", []))),
        "n_removed_sources": int(len(overrides.get("removed_sources", []))),
        "n_fit_mask_intervals": int(len(overrides.get("fit_mask_intervals", []))),
        "n_fit_windows": int(len(overrides.get("fit_windows", {}))),
        "n_continuum_edits": int(
            len(overrides.get("continuum_anchor_wavelengths_A", []))
            + len(overrides.get("continuum_anchor_nodes", []))
            + len(overrides.get("continuum_anchor_remove_indices", []))
            + len(overrides.get("continuum_mask_intervals_A", []))
        ),
    }


def _drop_regressive_broad_updates(patch: dict[str, Any], active_overrides: dict[str, Any]) -> dict[str, Any]:
    """Drop broad updates that would merge a region already split by explicit sources."""
    protected_regions = _protected_split_regions(active_overrides)
    if not protected_regions:
        return patch
    updated = deepcopy(patch)
    kept: list[dict[str, Any]] = []
    for call in updated.get("tool_calls", []):
        if call.get("name") != "update_absorption_source":
            kept.append(call)
            continue
        args = call.get("arguments", {})
        if not isinstance(args, dict):
            kept.append(call)
            continue
        transition = str(args.get("transition_line_id", ""))
        center = _score_float(args.get("center_velocity_kms"), default=float("nan"))
        b_kms = _score_float(args.get("b_kms"), default=0.0)
        half_width = _score_float(args.get("center_prior_half_width_kms"), default=0.0)
        regressive = False
        for region in protected_regions:
            if transition != region["transition_line_id"]:
                continue
            if region["start_velocity_kms"] <= center <= region["end_velocity_kms"] and max(b_kms, half_width) >= region["max_width_kms"]:
                regressive = True
                break
        if not regressive:
            kept.append(call)
    updated["tool_calls"] = kept
    updated["requires_refit"] = any(call.get("name") != "request_refit" for call in kept)
    return updated


def _coerce_distinct_repeated_updates_to_adds(patch: dict[str, Any]) -> dict[str, Any]:
    """Treat repeated updates of one component at distinct centers as added source seeds.

    Vision models sometimes say "separate exploratory component" but call
    update_absorption_source against the nearest existing component more than
    once. Keeping only replacement semantics loses the intended multi-source
    plan, so the first update remains an update and later well-separated
    updates become add_absorption_source calls.
    """
    updated = deepcopy(patch)
    seen: dict[tuple[str, int], list[float]] = {}
    coerced: list[dict[str, Any]] = []
    for call in updated.get("tool_calls", []):
        if call.get("name") != "update_absorption_source":
            coerced.append(call)
            continue
        args = call.get("arguments", {})
        if not isinstance(args, dict):
            coerced.append(call)
            continue
        try:
            component_index = int(args.get("component_index"))
        except (TypeError, ValueError):
            coerced.append(call)
            continue
        center = _score_float(args.get("center_velocity_kms"), default=float("nan"))
        if center != center:
            coerced.append(call)
            continue
        transition = str(args.get("transition_line_id", ""))
        key = (transition, component_index)
        prior_centers = seen.setdefault(key, [])
        if any(abs(center - previous) >= 45.0 for previous in prior_centers):
            add_args = deepcopy(args)
            add_args.pop("component_index", None)
            add_args["reason"] = str(add_args.get("reason", "")) + " Converted from repeated update to added source because the requested center is distinct from another update of the same component in this round."
            coerced.append({**call, "name": "add_absorption_source", "arguments": add_args})
        else:
            coerced.append(call)
        prior_centers.append(center)
    updated["tool_calls"] = coerced
    return updated


def _protected_split_regions(active_overrides: dict[str, Any]) -> list[dict[str, Any]]:
    centers_by_transition: dict[str, list[float]] = {}
    for source in active_overrides.get("source_seeds", []):
        if not isinstance(source, dict):
            continue
        center = _score_float(source.get("center_velocity_kms"), default=float("nan"))
        if center != center:
            continue
        transition = str(source.get("transition_line_id", ""))
        centers_by_transition.setdefault(transition, []).append(center)

    regions: list[dict[str, Any]] = []
    for transition, centers in centers_by_transition.items():
        centers = sorted(centers)
        for left, right in zip(centers[:-1], centers[1:], strict=False):
            if 20.0 <= right - left <= 90.0:
                regions.append(
                    {
                        "transition_line_id": transition,
                        "start_velocity_kms": left - 20.0,
                        "end_velocity_kms": right + 20.0,
                        "max_width_kms": 45.0,
                    }
                )
    return regions


def _candidate_score(evaluation: dict[str, Any]) -> tuple[float, float, int]:
    metrics = evaluation.get("metrics", {})
    refit = metrics.get("refit", {})
    delta = metrics.get("delta", {})
    fit_rms = _score_float(refit.get("fit_rms"), default=1.0e9)
    work_window = refit.get("source_work_window", {}) if isinstance(refit.get("source_work_window", {}), dict) else {}
    work_rms = _score_float(work_window.get("fit_rms"), default=fit_rms)
    max_work_residual = _score_float(work_window.get("max_abs_residual_sigma"), default=0.0)
    transition_frames = refit.get("transition_frames", {}) if isinstance(refit.get("transition_frames", {}), dict) else {}
    max_frame_residual = 0.0
    for frame in transition_frames.values():
        if isinstance(frame, dict):
            max_frame_residual = max(max_frame_residual, _score_float(frame.get("max_abs_residual_sigma"), default=0.0))
    fit_pixel_delta = _score_float(delta.get("n_fit_pixels"), default=0.0)
    fit_mask = delta.get("fit_mask", {}) if isinstance(delta.get("fit_mask", {}), dict) else {}
    mask_boundary = delta.get("fit_mask_boundary_residual", {}) if isinstance(delta.get("fit_mask_boundary_residual", {}), dict) else {}
    context_residual = delta.get("context_residual", {}) if isinstance(delta.get("context_residual", {}), dict) else {}
    core_mask_width = _score_float(fit_mask.get("core_width_kms"), default=0.0)
    boundary_mask_width = _score_float(fit_mask.get("boundary_width_kms"), default=0.0)
    context_mask_width = _score_float(fit_mask.get("context_width_kms"), default=0.0)
    max_mask_width = _score_float(fit_mask.get("max_interval_width_kms", fit_mask.get("max_width_kms")), default=0.0)
    n_mask_intervals = int(fit_mask.get("n_intervals", 0) or 0)
    n_core_intervals = int(fit_mask.get("n_core_intervals", 0) or 0)
    n_components = int(refit.get("n_components", 0) or 0)
    # Lower is better. Prefer lower RMS, but penalize aggressive pixel removal
    # so a later over-masked branch does not win just by hiding core residuals.
    removed_pixel_penalty = max(0.0, -fit_pixel_delta) * 0.08
    mask_width_penalty = core_mask_width * 0.12 + boundary_mask_width * 0.035 + context_mask_width * 0.012
    broad_mask_penalty = max(0.0, max_mask_width - 70.0) * 0.12
    mask_count_penalty = max(0, n_mask_intervals - 6) * 1.0 + max(0, n_core_intervals - 1) * 3.0
    work_residual_penalty = max(0.0, max_work_residual - 12.0) * 0.20
    frame_residual_penalty = max(0.0, max_frame_residual - 3.5) * 0.45
    mask_boundary_penalty = max(0.0, _score_float(mask_boundary.get("max_abs_residual_sigma_near_mask"), default=0.0) - 3.0) * 0.60
    context_residual_penalty = max(0.0, _score_float(context_residual.get("max_abs_residual_sigma_context"), default=0.0) - 3.0) * 0.75
    return (
        0.55 * fit_rms
        + 0.45 * work_rms
        + work_residual_penalty
        + frame_residual_penalty
        + mask_boundary_penalty
        + context_residual_penalty
        + removed_pixel_penalty
        + mask_width_penalty
        + broad_mask_penalty
        + mask_count_penalty,
        fit_rms,
        n_components,
    )


def _is_better_candidate(candidate: tuple[float, float, int], current: tuple[float, float, int] | None) -> bool:
    return current is None or candidate < current


def _clear_selected_candidate(history: list[dict[str, Any]]) -> None:
    for entry in history:
        entry["selected_candidate"] = False


def _score_float(value: Any, *, default: float) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return default
    return parsed if parsed == parsed else default


def _fit_control_needed(record: dict[str, Any]) -> bool:
    fit = record.get("fit_results", [{}])[0]
    if str(fit.get("quality", "")).lower() != "good":
        return True
    if bool(fit.get("agent_review", {}).get("required")):
        return True
    return False
