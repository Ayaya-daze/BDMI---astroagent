from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import pandas as pd

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
    base_id = sample_id_prefix or str(record.get("sample_id", "fit_control_loop"))
    best_record: dict[str, Any] | None = None
    best_plot: list[Path] | None = None
    best_output_paths: dict[str, Path] | None = None
    best_score: tuple[float, float, int] | None = None

    if not force and not _fit_control_needed(state.record):
        stop_reason = "not_needed_good_fit"
        state.record["fit_control_loop"] = _loop_record(history, stop_reason=stop_reason)
        return FitControlLoopResult(
            final_record=state.record,
            final_image_paths=state.image_paths,
            history=history,
            stop_reason=stop_reason,
            output_paths=output_paths,
        )

    for round_index in range(1, int(max_rounds) + 1):
        control = run_fit_control(
            state.record,
            client,
            temperature=temperature,
            plot_image_path=state.image_paths,
        )
        patch, round_overrides, candidate_controls, effective_patch = _prepare_round_controls(
            state.record,
            control,
            state.controls,
            round_index=round_index,
        )
        round_entry: dict[str, Any] = {
            "round_index": round_index,
            "input_sample_id": state.record.get("sample_id"),
            "control": control,
            "patch": effective_patch,
            "new_patch": patch,
            "round_overrides": round_overrides,
            "decision": None,
            "output_sample_id": None,
            "advanced": False,
            "candidate_score": None,
            "selected_candidate": False,
        }

        if not patch.get("requires_refit"):
            stop_reason = "no_refit_requested"
            history.append(round_entry)
            break

        refit_sample_id = f"{base_id}_loop{round_index}"
        candidate = _execute_candidate(
            state,
            window,
            effective_patch,
            candidate_controls,
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

        is_last_round = round_index == int(max_rounds)
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
            }
            if is_last_round:
                stop_reason = "refit_rejected"
                break
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
            if is_last_round:
                stop_reason = "needs_human_review"
                break
            state.record["fit_control_loop"] = _loop_record(history, stop_reason=None)
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
    return FitControlLoopResult(
        final_record=current_record,
        final_image_paths=current_plot,
        history=history,
        stop_reason=stop_reason,
        output_paths=output_paths,
    )


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
    effective_patch["tool_calls"] = deepcopy(candidate_controls.get("tool_calls", []))
    effective_patch["requires_refit"] = bool(patch.get("requires_refit"))
    return patch, round_overrides, candidate_controls, effective_patch


def _execute_candidate(
    state: FitControlAgentState,
    window: pd.DataFrame,
    patch: dict[str, Any],
    controls: dict[str, Any],
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
    fit_pixel_delta = _score_float(delta.get("n_fit_pixels"), default=0.0)
    fit_mask = delta.get("fit_mask", {}) if isinstance(delta.get("fit_mask", {}), dict) else {}
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
    return (
        0.55 * fit_rms
        + 0.45 * work_rms
        + work_residual_penalty
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
