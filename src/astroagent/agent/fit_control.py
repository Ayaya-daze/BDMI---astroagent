from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timezone
from typing import Any

import numpy as np
import pandas as pd

from astroagent.agent.policy import SOURCE_CORE_WINDOW_KMS, SOURCE_WORK_WINDOW_KMS
from astroagent.review.packet import build_review_record_from_window


FIT_CONTROL_TOOL_NAMES = {
    "add_absorption_source",
    "remove_absorption_source",
    "update_absorption_source",
    "set_fit_mask_interval",
    "set_fit_window",
    "request_refit",
    "request_more_budget",
    "add_continuum_anchor",
    "remove_continuum_anchor",
    "update_continuum_mask",
}

SOURCE_EXCLUDE_MASK_MIN_MARGIN_KMS = 35.0
SOURCE_EXCLUDE_MASK_MAX_MARGIN_KMS = 90.0
OVERLAP_MASK_REQUIRED_MARGIN_A = 1.25


def build_fit_control_patch(
    control: dict[str, Any],
    *,
    source: str = "llm_fit_control",
) -> dict[str, Any]:
    """Convert one LLM fit-control result into an auditable patch record."""
    if control.get("task") != "fit_control":
        raise ValueError("control task must be 'fit_control'")
    tool_calls = control.get("tool_calls", [])
    if not isinstance(tool_calls, list):
        raise ValueError("control tool_calls must be a list")

    normalized_calls = [_normalize_patch_call(call, index=index) for index, call in enumerate(tool_calls)]
    return {
        "task": "fit_control_patch",
        "source": source,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": control.get("status", "inspect"),
        "rationale": str(control.get("rationale", "")),
        "llm_metadata": control.get("_llm_metadata", {}),
        "tool_calls": normalized_calls,
        "requires_refit": _requires_refit(normalized_calls),
        "applied": False,
        "application_result": None,
    }


def append_fit_control_patch(
    record: dict[str, Any],
    control: dict[str, Any],
    *,
    source: str = "llm_fit_control",
) -> dict[str, Any]:
    """Return a copy of a review record with one fit-control patch appended."""
    updated = deepcopy(record)
    patches = updated.setdefault("fit_control_patches", [])
    patches.append(build_fit_control_patch(control, source=source))
    updated.setdefault("human_review", {}).setdefault("fit_control_notes", "")
    return updated


def summarize_pending_fit_control(record: dict[str, Any]) -> dict[str, Any]:
    patches = record.get("fit_control_patches", [])
    pending = [patch for patch in patches if not patch.get("applied")]
    tool_counts: dict[str, int] = {}
    for patch in pending:
        for call in patch.get("tool_calls", []):
            name = str(call.get("name", ""))
            tool_counts[name] = tool_counts.get(name, 0) + 1
    return {
        "n_patches": int(len(patches)),
        "n_pending_patches": int(len(pending)),
        "pending_tool_counts": tool_counts,
        "requires_refit": any(bool(patch.get("requires_refit")) for patch in pending),
    }


def build_fit_control_overrides(
    record: dict[str, Any],
    patch: dict[str, Any],
) -> dict[str, Any]:
    """Translate an auditable fit-control patch into deterministic refit overrides."""
    if patch.get("task") != "fit_control_patch":
        raise ValueError("patch task must be 'fit_control_patch'")

    overrides: dict[str, Any] = {
        "continuum_anchor_wavelengths_A": [],
        "continuum_anchor_nodes": [],
        "continuum_anchor_remove_indices": [],
        "continuum_anchor_remove_wavelengths_A": [],
        "continuum_mask_intervals_A": [],
        "fit_mask_intervals": [],
        "fit_windows": {},
        "source_seeds": [],
        "removed_sources": [],
        "request_refit": False,
        "tool_calls": deepcopy(patch.get("tool_calls", [])),
    }

    for call in patch.get("tool_calls", []):
        name = str(call.get("name", ""))
        args = call.get("arguments", {})
        if not isinstance(args, dict):
            continue
        if name == "add_continuum_anchor":
            node = _continuum_anchor_node_from_args(args)
            if node:
                overrides["continuum_anchor_nodes"].append(node)
            else:
                _append_number(overrides["continuum_anchor_wavelengths_A"], args.get("wavelength_A"))
        elif name == "remove_continuum_anchor":
            anchor_wavelength = _bounded_float(args.get("wavelength_A"), 0.0, 1.0e7)
            anchor_index = _bounded_int(args.get("anchor_index"), 0, 10_000)
            if anchor_wavelength is None:
                anchor_wavelength = _continuum_anchor_wavelength_from_index(record, anchor_index)
            if anchor_wavelength is not None:
                _append_number(overrides["continuum_anchor_remove_wavelengths_A"], anchor_wavelength)
            elif anchor_index is not None:
                _append_int(overrides["continuum_anchor_remove_indices"], anchor_index)
        elif name == "update_continuum_mask":
            start = _bounded_float(args.get("start_wavelength_A"), 0.0, 1.0e7)
            end = _bounded_float(args.get("end_wavelength_A"), 0.0, 1.0e7)
            if start is None or end is None:
                continue
            lower, upper = sorted((start, end))
            overrides["continuum_mask_intervals_A"].append(
                {
                    "start_wavelength_A": lower,
                    "end_wavelength_A": upper,
                    "mask_kind": _clean_mask_kind(args.get("mask_kind", "exclude")),
                    "reason": args.get("reason", ""),
                }
            )
        elif name == "set_fit_mask_interval":
            start = _bounded_float(args.get("start_velocity_kms"), -5000.0, 5000.0)
            end = _bounded_float(args.get("end_velocity_kms"), -5000.0, 5000.0)
            if start is None or end is None:
                continue
            lower, upper = sorted((start, end))
            overrides["fit_mask_intervals"].append(
                {
                    "transition_line_id": args.get("transition_line_id"),
                    "start_velocity_kms": lower,
                    "end_velocity_kms": upper,
                    "mask_kind": _clean_mask_kind(args.get("mask_kind", "exclude")),
                    "reason": args.get("reason", ""),
                }
            )
        elif name == "set_fit_window":
            transition_line_id = str(args.get("transition_line_id", ""))
            if transition_line_id:
                window = overrides["fit_windows"].setdefault(transition_line_id, {})
                transition_half_width = _bounded_float(args.get("transition_half_width_kms"), 80.0, 1500.0)
                local_fit_half_width = _bounded_float(args.get("local_fit_half_width_kms"), 40.0, 800.0)
                if transition_half_width is not None:
                    window["transition_half_width_kms"] = transition_half_width
                if local_fit_half_width is not None:
                    window["local_fit_half_width_kms"] = local_fit_half_width
                    window["source_fit_half_width_kms"] = local_fit_half_width
                sibling_mask_mode = str(args.get("sibling_mask_mode", ""))
                if sibling_mask_mode in {"exclude", "allow_overlap"}:
                    window["sibling_mask_mode"] = sibling_mask_mode
                window["reason"] = args.get("reason", "")
        elif name == "add_absorption_source":
            source = _source_seed_from_args(args, seed_source="add_absorption_source")
            if source:
                overrides["source_seeds"].append(source)
        elif name == "update_absorption_source":
            source = _updated_source_seed(record, args)
            if source:
                overrides["source_seeds"].append(source)
        elif name == "remove_absorption_source":
            source = _removed_source(record, args)
            if source:
                overrides["removed_sources"].append(source)
        elif name == "request_refit":
            overrides["request_refit"] = True

    return overrides


def refit_record_with_patch(
    record: dict[str, Any],
    window: pd.DataFrame,
    patch: dict[str, Any],
    *,
    sample_id: str | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Apply one patch to a review packet and rerun continuum + Voigt fitting."""
    overrides = build_fit_control_overrides(record, patch)
    return refit_record_with_overrides(record, window, patch, overrides, sample_id=sample_id)


def refit_record_with_overrides(
    record: dict[str, Any],
    window: pd.DataFrame,
    patch: dict[str, Any],
    overrides: dict[str, Any],
    *,
    sample_id: str | None = None,
    evaluation_overrides: dict[str, Any] | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Apply already-canonical fit-control overrides and rerun the fit."""
    original_fit = _primary_fit(record)
    controlled_overrides = _with_controlled_source_seeds(original_fit, overrides)
    controlled_evaluation_overrides = (
        _with_controlled_source_seeds(original_fit, evaluation_overrides)
        if evaluation_overrides is not None
        else controlled_overrides
    )
    refit_record = build_review_record_from_window(
        window=window,
        window_metadata=record["input"],
        sample_id=sample_id or f"{record['sample_id']}_refit",
        source={**record.get("source", {}), "fit_control_source": patch.get("source", "fit_control_patch")},
        fit_control=controlled_overrides,
    )
    refit_fit = _primary_fit(refit_record)
    evaluation = evaluate_refit_patch(original_fit, refit_fit, controlled_evaluation_overrides)
    applied_patch = deepcopy(patch)
    applied_patch["applied"] = True
    applied_patch["accepted"] = bool(evaluation["accepted"])
    applied_patch["application_result"] = {
        "sample_id": refit_record["sample_id"],
        "success": bool(refit_fit.get("success")),
        "fit_rms": refit_fit.get("fit_rms"),
        "decision": evaluation["decision"],
        "accepted": bool(evaluation["accepted"]),
        "requires_human_review": bool(evaluation["requires_human_review"]),
        "reasons": evaluation["reasons"],
    }
    refit_record["fit_control_patches"] = [applied_patch]
    refit_record["fit_control_evaluation"] = evaluation
    application = refit_record.setdefault("fit_control_application", {})
    application["evaluation"] = evaluation
    application["controls"] = _public_fit_control_overrides(controlled_overrides)
    refit_record.setdefault("human_review", {})["fit_control_decision"] = evaluation["decision"]
    return refit_record, controlled_overrides


def merge_fit_control_overrides(*items: dict[str, Any]) -> dict[str, Any]:
    """Merge canonical override objects without reinterpreting component indexes."""
    merged = {
        "continuum_anchor_wavelengths_A": [],
        "continuum_anchor_nodes": [],
        "continuum_anchor_remove_indices": [],
        "continuum_anchor_remove_wavelengths_A": [],
        "continuum_mask_intervals_A": [],
        "fit_mask_intervals": [],
        "fit_windows": {},
        "source_seeds": [],
        "removed_sources": [],
        "request_refit": False,
        "tool_calls": [],
    }
    for item in items:
        if not item:
            continue
        for key in (
            "continuum_anchor_wavelengths_A",
            "continuum_anchor_nodes",
            "continuum_anchor_remove_indices",
            "continuum_anchor_remove_wavelengths_A",
            "continuum_mask_intervals_A",
            "fit_mask_intervals",
            "source_seeds",
            "removed_sources",
            "tool_calls",
        ):
            merged[key].extend(deepcopy(item.get(key, [])))
        for transition_line_id, control in item.get("fit_windows", {}).items():
            merged["fit_windows"].setdefault(transition_line_id, {}).update(deepcopy(control))
        merged["request_refit"] = bool(merged["request_refit"] or item.get("request_refit"))
    merged["source_seeds"] = _dedupe_source_seeds(merged["source_seeds"])
    merged["fit_mask_intervals"] = _dedupe_fit_mask_intervals(merged["fit_mask_intervals"])
    merged["source_seeds"] = _drop_sources_overlapping_exclude_masks(merged["source_seeds"], merged["fit_mask_intervals"])
    merged["continuum_anchor_remove_wavelengths_A"] = _dedupe_numbers(
        merged["continuum_anchor_remove_wavelengths_A"],
        tolerance=0.25,
    )
    for item in items:
        if item and item.get("source_detection_mode"):
            merged["source_detection_mode"] = item.get("source_detection_mode")
    return merged


def empty_fit_control_overrides() -> dict[str, Any]:
    return merge_fit_control_overrides({})


def _with_controlled_source_seeds(original_fit: dict[str, Any], overrides: dict[str, Any]) -> dict[str, Any]:
    controlled = deepcopy(overrides)
    controlled["source_detection_mode"] = "controlled"
    if _edit_profile(controlled)["continuum_changed"]:
        controlled["source_seeds"] = [_relax_source_seed_after_continuum_edit(seed) for seed in controlled.get("source_seeds", [])]
    base_sources = _source_seeds_from_fit_components(original_fit, controlled)
    controlled["source_seeds"] = _dedupe_source_seeds([*controlled.get("source_seeds", []), *base_sources])
    return controlled


def _relax_source_seed_after_continuum_edit(seed: dict[str, Any]) -> dict[str, Any]:
    relaxed = deepcopy(seed)
    for key in ("logN", "b_kms", "logN_lower", "logN_upper", "b_kms_lower", "b_kms_upper"):
        relaxed.pop(key, None)
    relaxed["explicit_fit_control_source"] = False
    relaxed["center_prior_sigma_kms"] = max(float(relaxed.get("center_prior_sigma_kms", 0.0) or 0.0), 90.0)
    relaxed["center_prior_half_width_kms"] = max(float(relaxed.get("center_prior_half_width_kms", 0.0) or 0.0), 240.0)
    relaxed["reason"] = (
        str(relaxed.get("reason", "")).strip()
        + " Continuum changed; prior source topology is treated as a soft position hint only."
    ).strip()
    return relaxed


def _source_seeds_from_fit_components(fit: dict[str, Any], controls: dict[str, Any]) -> list[dict[str, Any]]:
    seeds: list[dict[str, Any]] = []
    continuum_changed = _edit_profile(controls)["continuum_changed"]
    for component in fit.get("components", []):
        if not component.get("fit_success"):
            continue
        transition_line_id = str(component.get("transition_line_id", ""))
        center = _bounded_float(component.get("center_velocity_kms"), -5000.0, 5000.0)
        if not transition_line_id or center is None:
            continue
        if _source_is_removed_by_control(transition_line_id, center, component, controls):
            continue
        seed = {
            "transition_line_id": transition_line_id,
            "center_velocity_kms": center,
            "seed_source": "previous_fit_component",
            "explicit_fit_control_source": not continuum_changed,
            "reason": "carried from previous accepted/current fit; agent refit does not auto-detect new sources",
            "center_prior_sigma_kms": 90.0 if continuum_changed else 35.0,
            "center_prior_half_width_kms": 240.0 if continuum_changed else 90.0,
        }
        if not continuum_changed:
            for key in ("logN", "b_kms", "logN_lower", "logN_upper", "b_kms_lower", "b_kms_upper"):
                value = _finite_float(component.get(key))
                if np.isfinite(value):
                    seed[key] = value
        if not continuum_changed:
            if "logN_lower" not in seed:
                seed["logN_lower"] = 8.0
            if "logN_upper" not in seed:
                seed["logN_upper"] = 22.0
            if "b_kms_lower" not in seed:
                seed["b_kms_lower"] = 1.0
            if "b_kms_upper" not in seed:
                seed["b_kms_upper"] = 500.0
        seeds.append(seed)
    return seeds


def _source_is_removed_by_control(
    transition_line_id: str,
    center_velocity_kms: float,
    component: dict[str, Any],
    controls: dict[str, Any],
) -> bool:
    for interval in controls.get("fit_mask_intervals", []):
        if str(interval.get("mask_kind", "exclude")) != "exclude":
            continue
        if str(interval.get("transition_line_id", transition_line_id)) != transition_line_id:
            continue
        start = _finite_float(interval.get("start_velocity_kms"))
        end = _finite_float(interval.get("end_velocity_kms"))
        if not (np.isfinite(start) and np.isfinite(end)):
            continue
        lower, upper = sorted((start, end))
        if _source_overlaps_exclude_fit_mask_interval(center_velocity_kms, component, lower, upper):
            return True
    for removed in controls.get("removed_sources", []):
        if str(removed.get("transition_line_id", transition_line_id)) != transition_line_id:
            continue
        removed_center = _finite_float(removed.get("center_velocity_kms"))
        tolerance = _finite_float(removed.get("center_tolerance_kms"))
        if not np.isfinite(tolerance):
            tolerance = 45.0
        if np.isfinite(removed_center) and abs(center_velocity_kms - removed_center) <= tolerance:
            return True
        try:
            if int(removed.get("component_index")) == int(component.get("component_index")):
                return True
        except (TypeError, ValueError):
            pass
    return False


def _component_diagnostic_counts(fit: dict[str, Any]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for component in fit.get("components", []):
        for flag in component.get("diagnostic_flags", []):
            key = str(flag)
            counts[key] = counts.get(key, 0) + 1
    for frame in fit.get("transition_frames", []):
        for reason in frame.get("agent_review_reasons", []):
            key = str(reason)
            counts[key] = max(counts.get(key, 0), 1)
    return counts


def _component_diagnostic_regression(original_fit: dict[str, Any], refit_fit: dict[str, Any]) -> dict[str, Any]:
    original_counts = _component_diagnostic_counts(original_fit)
    refit_counts = _component_diagnostic_counts(refit_fit)

    def delta(name: str) -> int:
        return max(0, int(refit_counts.get(name, 0)) - int(original_counts.get(name, 0)))

    return {
        "original_counts": original_counts,
        "refit_counts": refit_counts,
        "new_saturated_redundant_components": delta("saturated_redundant_component"),
        "new_posterior_degenerate_components": max(
            delta("component_parameter_posterior_degenerate"),
            delta("component_parameter_posterior_disagrees_with_map"),
        ),
        "new_parameter_bound_components": delta("component_at_parameter_bound"),
    }


def _edit_profile(overrides: dict[str, Any]) -> dict[str, Any]:
    tool_calls = [call for call in overrides.get("tool_calls", []) if isinstance(call, dict)]
    if tool_calls:
        names = [str(call.get("name", "")) for call in tool_calls]
        continuum_count = sum(
            1
            for name in names
            if name in {"add_continuum_anchor", "remove_continuum_anchor", "update_continuum_mask"}
        )
        added_source_count = names.count("add_absorption_source")
        updated_source_count = names.count("update_absorption_source")
        removed_source_count = names.count("remove_absorption_source")
        source_count = added_source_count + updated_source_count + removed_source_count
        mask_window_count = sum(1 for name in names if name in {"set_fit_mask_interval", "set_fit_window"})
    else:
        continuum_count = (
            int(len(overrides.get("continuum_anchor_wavelengths_A", [])))
            + int(len(overrides.get("continuum_anchor_nodes", [])))
            + int(len(overrides.get("continuum_anchor_remove_indices", [])))
            + int(len(overrides.get("continuum_anchor_remove_wavelengths_A", [])))
            + int(len(overrides.get("continuum_mask_intervals_A", [])))
        )
        added_source_count = int(
            sum(
                1
                for source in overrides.get("source_seeds", [])
                if str(source.get("seed_source", "add_absorption_source")) != "previous_fit_component"
            )
        )
        updated_source_count = int(
            sum(1 for source in overrides.get("source_seeds", []) if str(source.get("seed_source")) == "update_absorption_source")
        )
        removed_source_count = int(len(overrides.get("removed_sources", [])))
        source_count = added_source_count + updated_source_count + removed_source_count
        mask_window_count = int(len(overrides.get("fit_mask_intervals", []))) + int(len(overrides.get("fit_windows", {})))
    return {
        "continuum_edit_count": continuum_count,
        "source_edit_count": source_count,
        "added_source_tool_count": added_source_count,
        "updated_source_tool_count": updated_source_count,
        "removed_source_tool_count": removed_source_count,
        "fit_mask_window_edit_count": mask_window_count,
        "continuum_changed": continuum_count > 0,
        "source_or_mask_changed": source_count > 0 or mask_window_count > 0,
        "continuum_only": continuum_count > 0 and source_count == 0 and mask_window_count == 0,
    }


def evaluate_refit_patch(
    original_fit: dict[str, Any],
    refit_fit: dict[str, Any],
    overrides: dict[str, Any],
) -> dict[str, Any]:
    """Decide whether a deterministic refit should be trusted automatically."""
    original_metrics = _fit_metrics(original_fit)
    refit_metrics = _fit_metrics(refit_fit)
    reasons: list[str] = []
    warnings: list[str] = []

    if not refit_metrics["success"]:
        reasons.append("refit_failed")
    if not original_metrics["success"] and refit_metrics["success"]:
        warnings.append("original_fit_failed_refit_succeeded")

    original_rms = original_metrics["fit_rms"]
    refit_rms = refit_metrics["fit_rms"]
    rms_worsened = False
    if np.isfinite(original_rms) and np.isfinite(refit_rms):
        rms_ratio = float(refit_rms / max(original_rms, 1e-6))
        if rms_ratio > 1.05:
            rms_worsened = True
    else:
        rms_ratio = float("nan")
        if np.isfinite(original_rms) and not np.isfinite(refit_rms):
            reasons.append("refit_rms_not_finite")

    original_reduced_chi2 = original_metrics["reduced_chi2"]
    refit_reduced_chi2 = refit_metrics["reduced_chi2"]
    if np.isfinite(original_reduced_chi2) and np.isfinite(refit_reduced_chi2):
        reduced_chi2_delta = float(refit_reduced_chi2 - original_reduced_chi2)
        reduced_chi2_ratio = float(refit_reduced_chi2 / max(original_reduced_chi2, 1e-6))
    else:
        reduced_chi2_delta = float("nan")
        reduced_chi2_ratio = float("nan")

    original_work_rms = original_metrics["source_work_window"]["fit_rms"]
    refit_work_rms = refit_metrics["source_work_window"]["fit_rms"]
    if np.isfinite(original_work_rms) and np.isfinite(refit_work_rms):
        work_rms_delta = float(refit_work_rms - original_work_rms)
        work_rms_ratio = float(refit_work_rms / max(original_work_rms, 1e-6))
    else:
        work_rms_delta = float("nan")
        work_rms_ratio = float("nan")
    original_work_reduced_chi2 = original_metrics["source_work_window"]["reduced_chi2"]
    refit_work_reduced_chi2 = refit_metrics["source_work_window"]["reduced_chi2"]
    if np.isfinite(original_work_reduced_chi2) and np.isfinite(refit_work_reduced_chi2):
        work_reduced_chi2_delta = float(refit_work_reduced_chi2 - original_work_reduced_chi2)
        work_reduced_chi2_ratio = float(refit_work_reduced_chi2 / max(original_work_reduced_chi2, 1e-6))
    else:
        work_reduced_chi2_delta = float("nan")
        work_reduced_chi2_ratio = float("nan")

    original_components = int(original_metrics["n_components"])
    refit_components = int(refit_metrics["n_components"])
    edit_profile = _edit_profile(overrides)
    added_sources = int(edit_profile["added_source_tool_count"])
    removed_sources = int(edit_profile["removed_source_tool_count"])
    diagnostic_regression = _component_diagnostic_regression(original_fit, refit_fit)
    mask_metrics = _fit_mask_metrics(overrides)
    material_edit_count = (
        added_sources
        + int(edit_profile["updated_source_tool_count"])
        + removed_sources
        + int(edit_profile["fit_mask_window_edit_count"])
        + int(edit_profile["continuum_edit_count"])
    )
    if original_components > 0 and refit_components > max(original_components + max(added_sources, 1), 2 * original_components):
        warnings.append("component_count_exploded")
    if original_components > 0 and refit_components == 0:
        reasons.append("all_components_removed")
    if removed_sources == 0 and original_components > 0 and refit_components < max(1, original_components - 2):
        warnings.append("component_count_dropped_without_remove_request")
    if added_sources > 0 and diagnostic_regression["new_saturated_redundant_components"] > 0:
        warnings.append("added_source_in_already_saturated_region")
    elif diagnostic_regression["new_saturated_redundant_components"] > 0:
        warnings.append("new_saturated_redundant_component")
    if diagnostic_regression["new_posterior_degenerate_components"] > 0:
        warnings.append("new_component_parameter_degeneracy")

    original_fit_pixels = int(original_metrics["n_fit_pixels"])
    refit_fit_pixels = int(refit_metrics["n_fit_pixels"])
    frame_comparison = _compare_frame_fit_metrics(original_metrics["transition_frames"], refit_metrics["transition_frames"])
    overlap_policy = _overlap_edit_policy(original_fit, refit_fit, overrides)
    mask_boundary_residual = _fit_mask_boundary_residual_diagnostics(refit_fit, overrides)
    context_residual = _context_residual_diagnostics(refit_fit)
    if original_fit_pixels > 0 and refit_fit_pixels < 0.35 * original_fit_pixels:
        reasons.append("fit_window_or_mask_removed_too_many_pixels")
    if frame_comparison["coverage_changed"]:
        warnings.append("fit_pixel_coverage_changed_between_rounds")
    if overlap_policy["edits_in_overlapping_velocity_frames"]:
        warnings.append("edits_in_overlapping_velocity_frames")
    if overlap_policy["overlap_affected_velocity_frames_unaddressed"]:
        warnings.append("overlap_affected_velocity_frames_unaddressed")
    if overlap_policy["overlap_affected_velocity_frames_without_fit_pixels"]:
        warnings.append("overlap_affected_velocity_frames_without_fit_pixels")
    if overlap_policy["missing_overlap_mask_after_source_removal"]:
        warnings.append("missing_overlap_mask_after_source_removal")
    if overlap_policy["missing_overlap_mask_after_source_update"]:
        warnings.append("missing_overlap_mask_after_source_update")

    if rms_worsened:
        if frame_comparison["coverage_changed"]:
            warnings.append("global_fit_rms_not_directly_comparable")
        elif edit_profile["continuum_changed"] and not edit_profile["source_or_mask_changed"]:
            warnings.append("continuum_changed_fit_not_directly_comparable")
            warnings.append("continuum_refit_needs_source_mask_followup")
        elif _work_window_improvement_offsets_global_rms_worsening(
            rms_ratio=rms_ratio,
            work_rms_ratio=work_rms_ratio,
            original_fit_pixels=original_fit_pixels,
            refit_fit_pixels=refit_fit_pixels,
            hard_reasons=reasons,
        ):
            warnings.append("fit_rms_worsened_but_source_work_window_improved")
        else:
            warnings.append("fit_rms_worsened")
    if mask_metrics["max_interval_width_kms"] > 100.0:
        warnings.append("fit_mask_interval_too_broad")
    if mask_boundary_residual["n_high_residual_fit_pixels_near_mask"] > 0:
        warnings.append("fit_mask_boundary_left_residual_in_fit_pixels")
    if context_residual["n_high_context_diagnostic_residual_pixels"] > 0:
        warnings.append("high_context_diagnostic_residual")
    if mask_metrics["core_width_kms"] > 50.0:
        warnings.append("fit_mask_core_width_large")
    if mask_metrics["core_width_kms"] > 120.0:
        reasons.append("fit_mask_core_removed_too_much")
    if mask_metrics["boundary_width_kms"] > 90.0:
        warnings.append("fit_mask_boundary_width_large")
    if mask_metrics["context_width_kms"] > 320.0:
        warnings.append("fit_mask_context_width_large")
    if mask_metrics["total_width_kms"] > 320.0 and mask_metrics["core_width_kms"] > 0.0:
        warnings.append("fit_mask_total_width_large")
    if original_fit_pixels > 0 and refit_fit_pixels < 0.70 * original_fit_pixels and mask_metrics["n_intervals"] > 0:
        warnings.append("fit_mask_removed_many_pixels")
    if material_edit_count > 0:
        component_delta = abs(refit_components - original_components)
        fit_pixel_delta = abs(refit_fit_pixels - original_fit_pixels)
        rms_delta = abs(_finite_delta(refit_rms, original_rms))
        if component_delta == 0 and fit_pixel_delta == 0 and (not np.isfinite(rms_delta) or rms_delta < 1e-4):
            warnings.append("refit_had_no_material_effect")
        if (
            len(overrides.get("fit_mask_intervals", [])) >= 3
            and added_sources == 0
            and removed_sources == 0
            and refit_fit_pixels < 0.80 * max(original_fit_pixels, 1)
        ):
            warnings.append("mask_heavy_refit_without_source_change")

    original_rank = _quality_rank(original_metrics["quality"])
    refit_rank = _quality_rank(refit_metrics["quality"])
    if refit_rank < original_rank:
        warnings.append("fit_quality_regressed")
    elif refit_rank == original_rank and refit_metrics["quality"] == "inspect":
        warnings.append("refit_still_requires_inspection")

    original_review_reasons = set(original_metrics["agent_review_reasons"])
    refit_review_reasons = set(refit_metrics["agent_review_reasons"])
    new_review_reasons = sorted(refit_review_reasons - original_review_reasons)
    if new_review_reasons:
        warnings.append("new_agent_review_reasons")

    hard_reasons = [
        reason
        for reason in reasons
        if reason in {
            "refit_failed",
            "refit_rms_not_finite",
            "all_components_removed",
            "fit_window_or_mask_removed_too_many_pixels",
            "fit_mask_core_removed_too_much",
        }
    ]
    if hard_reasons:
        decision = "rejected"
        outcome = "refit_worsened"
        accepted = False
        requires_human_review = True
    elif warnings:
        decision = "needs_human_review"
        outcome = "needs_followup"
        accepted = False
        requires_human_review = True
    else:
        decision = "accepted"
        outcome = "accepted"
        accepted = True
        requires_human_review = False

    return {
        "task": "fit_control_refit_evaluation",
        "decision": decision,
        "outcome": outcome,
        "accepted": accepted,
        "requires_human_review": requires_human_review,
        "reasons": reasons,
        "warnings": warnings,
        "metrics": {
            "original": original_metrics,
            "refit": refit_metrics,
            "delta": {
                "fit_rms": _finite_delta(refit_rms, original_rms),
                "fit_rms_ratio": rms_ratio,
                "reduced_chi2": reduced_chi2_delta,
                "reduced_chi2_ratio": reduced_chi2_ratio,
                "chi2": _finite_delta(refit_metrics["chi2"], original_metrics["chi2"]),
                "source_work_window_fit_rms": work_rms_delta,
                "source_work_window_fit_rms_ratio": work_rms_ratio,
                "source_work_window_reduced_chi2": work_reduced_chi2_delta,
                "source_work_window_reduced_chi2_ratio": work_reduced_chi2_ratio,
                "source_work_window_chi2": _finite_delta(
                    refit_metrics["source_work_window"]["chi2"],
                    original_metrics["source_work_window"]["chi2"],
                ),
                "n_components": refit_components - original_components,
                "n_fit_pixels": refit_fit_pixels - original_fit_pixels,
                "transition_frame_fit_comparison": frame_comparison,
                "overlap_edit_policy": overlap_policy,
                "new_agent_review_reasons": new_review_reasons,
                "component_diagnostic_regression": diagnostic_regression,
                "edit_profile": edit_profile,
                "material_edit_count": material_edit_count,
                "fit_mask": mask_metrics,
                "fit_mask_boundary_residual": mask_boundary_residual,
                "context_residual": context_residual,
            },
        },
        "policy": {
            "reject_if_rms_ratio_gt": 1.05,
            "downgrade_rms_worsening_to_review_if_source_work_window_ratio_lt": 0.95,
            "downgrade_rms_worsening_to_review_if_fit_rms_ratio_lte": 1.25,
            "reject_if_fit_pixels_lt_fraction": 0.35,
            "reject_quality_regression": True,
            "source_work_window_rms_is_scored_separately": True,
            "accepted_means_no_new_warnings": True,
            "global_rms_requires_same_transition_frame_fit_coverage": True,
            "continuum_only_refit_rms_is_not_directly_comparable": True,
            "overlap_source_removal_requires_overlap_mask": True,
            "added_source_in_saturated_redundant_region_requires_human_review": True,
        },
    }


def _work_window_improvement_offsets_global_rms_worsening(
    *,
    rms_ratio: float,
    work_rms_ratio: float,
    original_fit_pixels: int,
    refit_fit_pixels: int,
    hard_reasons: list[str],
) -> bool:
    if not (np.isfinite(rms_ratio) and np.isfinite(work_rms_ratio)):
        return False
    if rms_ratio > 1.25 or work_rms_ratio >= 0.95:
        return False
    if "component_count_exploded" in hard_reasons or "fit_window_or_mask_removed_too_many_pixels" in hard_reasons:
        return False
    if original_fit_pixels > 0 and refit_fit_pixels < 0.70 * original_fit_pixels:
        return False
    return True


def _normalize_patch_call(call: dict[str, Any], *, index: int) -> dict[str, Any]:
    name = str(call.get("name", ""))
    if name not in FIT_CONTROL_TOOL_NAMES:
        raise ValueError(f"unknown fit-control tool: {name}")
    arguments = call.get("arguments", {})
    if not isinstance(arguments, dict):
        raise ValueError(f"fit-control tool arguments must be an object for {name}")
    _validate_patch_call_arguments(name, arguments)
    return {
        "sequence_index": int(index),
        "id": call.get("id"),
        "name": name,
        "arguments": arguments,
        "validated": True,
    }


def _validate_patch_call_arguments(name: str, arguments: dict[str, Any]) -> None:
    required_by_tool = {
        "add_absorption_source": {"transition_line_id", "center_velocity_kms", "reason"},
        "remove_absorption_source": {"component_index", "reason"},
        "update_absorption_source": {"component_index", "reason"},
        "set_fit_mask_interval": {"transition_line_id", "start_velocity_kms", "end_velocity_kms", "mask_kind", "reason"},
        "set_fit_window": {"transition_line_id", "reason"},
        "request_refit": {"reason"},
        "request_more_budget": {"requested_rounds", "next_experiment", "reason"},
        "add_continuum_anchor": {"wavelength_A", "reason"},
        "remove_continuum_anchor": {"anchor_index", "reason"},
        "update_continuum_mask": {"start_wavelength_A", "end_wavelength_A", "mask_kind", "reason"},
    }
    missing = sorted(required_by_tool.get(name, set()) - set(arguments))
    if missing:
        raise ValueError(f"fit-control tool {name} missing required arguments: {missing}")
    numeric_keys = {
        "center_velocity_kms",
        "center_prior_sigma_kms",
        "center_prior_half_width_kms",
        "logN",
        "logN_lower",
        "logN_upper",
        "b_kms",
        "b_kms_lower",
        "b_kms_upper",
        "start_velocity_kms",
        "end_velocity_kms",
        "transition_half_width_kms",
        "local_fit_half_width_kms",
        "wavelength_A",
        "continuum_flux",
        "start_wavelength_A",
        "end_wavelength_A",
    }
    for key in numeric_keys & set(arguments):
        if not _finite(arguments.get(key)):
            raise ValueError(f"fit-control tool {name} argument {key} must be a finite number")
    for key in {"component_index", "anchor_index", "requested_rounds"} & set(arguments):
        try:
            parsed = int(arguments.get(key))
        except (TypeError, ValueError) as exc:
            raise ValueError(f"fit-control tool {name} argument {key} must be an integer") from exc
        if isinstance(arguments.get(key), float) and not float(arguments[key]).is_integer():
            raise ValueError(f"fit-control tool {name} argument {key} must be an integer")
        if key == "requested_rounds" and parsed < 1:
            raise ValueError("fit-control tool request_more_budget argument requested_rounds must be >= 1")
    if arguments.get("mask_kind") is not None and arguments.get("mask_kind") not in {"include", "exclude"}:
        raise ValueError(f"fit-control tool {name} argument mask_kind must be include or exclude")
    if arguments.get("sibling_mask_mode") is not None and arguments.get("sibling_mask_mode") not in {"exclude", "allow_overlap"}:
        raise ValueError(f"fit-control tool {name} argument sibling_mask_mode must be exclude or allow_overlap")
    for key in {"transition_line_id", "reason", "next_experiment"} & set(arguments):
        if not isinstance(arguments.get(key), str):
            raise ValueError(f"fit-control tool {name} argument {key} must be a string")


def _requires_refit(tool_calls: list[dict[str, Any]]) -> bool:
    refit_tools = {
        "add_absorption_source",
        "remove_absorption_source",
        "update_absorption_source",
        "set_fit_mask_interval",
        "set_fit_window",
        "add_continuum_anchor",
        "remove_continuum_anchor",
        "update_continuum_mask",
    }
    return any(call.get("name") in refit_tools for call in tool_calls)


def _dedupe_source_seeds(sources: list[dict[str, Any]]) -> list[dict[str, Any]]:
    kept: list[dict[str, Any]] = []
    for source in sources:
        if not isinstance(source, dict):
            continue
        transition = str(source.get("transition_line_id", ""))
        center = _finite_float(source.get("center_velocity_kms"))
        seed_source = str(source.get("seed_source", ""))
        replace_index = source.get("replace_component_index")
        duplicate_index: int | None = None
        for index, existing in enumerate(kept):
            if transition != str(existing.get("transition_line_id", "")):
                continue
            existing_center = _finite_float(existing.get("center_velocity_kms"))
            if not np.isfinite(center) or not np.isfinite(existing_center):
                continue
            same_replace = replace_index is not None and replace_index == existing.get("replace_component_index")
            same_kind = seed_source == str(existing.get("seed_source", ""))
            if (same_replace or same_kind) and abs(center - existing_center) <= 8.0:
                duplicate_index = index
                break
        if duplicate_index is None:
            kept.append(deepcopy(source))
        else:
            kept[duplicate_index] = _merge_source_seed(kept[duplicate_index], source)
    return kept


def _merge_source_seed(old: dict[str, Any], new: dict[str, Any]) -> dict[str, Any]:
    merged = deepcopy(old)
    for key, value in new.items():
        if value is not None:
            merged[key] = deepcopy(value)
    return merged


def _dedupe_fit_mask_intervals(intervals: list[dict[str, Any]]) -> list[dict[str, Any]]:
    kept: list[dict[str, Any]] = []
    for interval in intervals:
        if not isinstance(interval, dict):
            continue
        transition = str(interval.get("transition_line_id", ""))
        start = _finite_float(interval.get("start_velocity_kms"))
        end = _finite_float(interval.get("end_velocity_kms"))
        kind = str(interval.get("mask_kind", "exclude"))
        duplicate = False
        for existing in kept:
            if transition != str(existing.get("transition_line_id", "")):
                continue
            if kind != str(existing.get("mask_kind", "exclude")):
                continue
            existing_start = _finite_float(existing.get("start_velocity_kms"))
            existing_end = _finite_float(existing.get("end_velocity_kms"))
            if (
                np.isfinite(start)
                and np.isfinite(end)
                and np.isfinite(existing_start)
                and np.isfinite(existing_end)
                and abs(start - existing_start) <= 5.0
                and abs(end - existing_end) <= 5.0
            ):
                duplicate = True
                break
        if not duplicate:
            kept.append(deepcopy(interval))
    return kept


def _drop_sources_overlapping_exclude_masks(
    sources: list[dict[str, Any]],
    intervals: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    kept: list[dict[str, Any]] = []
    for source in sources:
        transition = str(source.get("transition_line_id", ""))
        center = _finite_float(source.get("center_velocity_kms"))
        if not transition or not np.isfinite(center):
            kept.append(source)
            continue
        if _source_overlaps_exclude_fit_mask(transition, center, source, intervals):
            continue
        kept.append(source)
    return kept


def _source_overlaps_exclude_fit_mask(
    transition_line_id: str,
    center_velocity_kms: float,
    source: dict[str, Any],
    intervals: list[dict[str, Any]],
) -> bool:
    for interval in intervals:
        if str(interval.get("mask_kind", "exclude")) != "exclude":
            continue
        if str(interval.get("transition_line_id", transition_line_id)) != transition_line_id:
            continue
        start = _finite_float(interval.get("start_velocity_kms"))
        end = _finite_float(interval.get("end_velocity_kms"))
        if not (np.isfinite(start) and np.isfinite(end)):
            continue
        lower, upper = sorted((start, end))
        if _source_overlaps_exclude_fit_mask_interval(center_velocity_kms, source, lower, upper):
            return True
    return False


def _source_overlaps_exclude_fit_mask_interval(
    center_velocity_kms: float,
    source: dict[str, Any],
    mask_start_velocity_kms: float,
    mask_end_velocity_kms: float,
) -> bool:
    margin = _source_exclude_mask_margin_kms(source)
    return bool(mask_start_velocity_kms - margin <= center_velocity_kms <= mask_end_velocity_kms + margin)


def _source_exclude_mask_margin_kms(source: dict[str, Any]) -> float:
    candidates: list[float] = []
    prior_half_width = _finite_float(source.get("center_prior_half_width_kms"))
    if np.isfinite(prior_half_width):
        candidates.append(prior_half_width)
    prior_sigma = _finite_float(source.get("center_prior_sigma_kms"))
    if np.isfinite(prior_sigma):
        candidates.append(2.0 * prior_sigma)
    b_kms = _finite_float(source.get("b_kms"))
    if np.isfinite(b_kms):
        candidates.append(2.0 * b_kms)
    if not candidates:
        return 45.0
    return float(max(SOURCE_EXCLUDE_MASK_MIN_MARGIN_KMS, min(SOURCE_EXCLUDE_MASK_MAX_MARGIN_KMS, max(candidates))))


def _dedupe_numbers(values: list[float], *, tolerance: float) -> list[float]:
    kept: list[float] = []
    for value in values:
        parsed = _bounded_float(value, -1.0e12, 1.0e12)
        if parsed is None:
            continue
        if any(abs(parsed - existing) <= tolerance for existing in kept):
            continue
        kept.append(parsed)
    return kept


def _append_number(values: list[float], value: Any) -> None:
    if _finite(value):
        values.append(float(value))


def _append_int(values: list[int], value: Any) -> None:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return
    if parsed >= 0:
        values.append(parsed)


def _finite(value: Any) -> bool:
    try:
        return bool(np.isfinite(float(value)))
    except (TypeError, ValueError):
        return False


def _clean_mask_kind(value: Any) -> str:
    return "include" if str(value) == "include" else "exclude"


def _continuum_anchor_node_from_args(args: dict[str, Any]) -> dict[str, Any] | None:
    wavelength_A = _bounded_float(args.get("wavelength_A"), 0.0, 1.0e7)
    continuum_flux = _bounded_float(args.get("continuum_flux"), 1.0e-6, 1.0e7)
    if wavelength_A is None or continuum_flux is None:
        return None
    return {
        "wavelength_A": wavelength_A,
        "continuum_flux": continuum_flux,
        "source": "add_continuum_anchor",
        "reason": args.get("reason", ""),
    }


def _continuum_anchor_wavelength_from_index(record: dict[str, Any], anchor_index: int | None) -> float | None:
    if anchor_index is None:
        return None
    anchors = record.get("plot_data", {}).get("anchor_wavelengths_A", [])
    try:
        wavelength = float(anchors[int(anchor_index)])
    except (TypeError, ValueError, IndexError):
        return None
    if not np.isfinite(wavelength):
        return None
    return wavelength


def _bounded_float(value: Any, lower: float, upper: float) -> float | None:
    if not _finite(value):
        return None
    return float(np.clip(float(value), lower, upper))


def _bounded_int(value: Any, lower: int, upper: int) -> int | None:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return None
    return int(np.clip(parsed, lower, upper))


def _primary_fit(record: dict[str, Any]) -> dict[str, Any]:
    for result in record.get("fit_results", []):
        if result.get("task") == "voigt_profile_fit" or result.get("fit_type"):
            return result
    return {}


def _public_fit_control_overrides(overrides: dict[str, Any]) -> dict[str, Any]:
    public = deepcopy(overrides)
    return public


def _fit_metrics(fit: dict[str, Any]) -> dict[str, Any]:
    review_reasons = set(fit.get("agent_review", {}).get("reasons", []))
    for frame in fit.get("transition_frames", []):
        review_reasons.update(str(reason) for reason in frame.get("agent_review_reasons", []))
    return {
        "success": bool(fit.get("success")),
        "quality": str(fit.get("quality", fit.get("status", "unknown"))),
        "chi2": _finite_float(fit.get("chi2")),
        "reduced_chi2": _finite_float(fit.get("reduced_chi2")),
        "fit_rms": _finite_float(fit.get("fit_rms")),
        "n_components": int(fit.get("n_components_fitted", 0) or 0),
        "n_fit_pixels": int(fit.get("n_fit_pixels", 0) or 0),
        "transition_frames": _transition_frame_fit_metrics(fit),
        "source_work_window": _source_work_window_metric(fit),
        "agent_review_required": bool(fit.get("agent_review", {}).get("required")),
        "agent_review_reasons": sorted(review_reasons),
    }


def _transition_frame_fit_metrics(fit: dict[str, Any]) -> dict[str, dict[str, Any]]:
    frames: dict[str, dict[str, Any]] = {}
    for frame in fit.get("transition_frames", []):
        transition_line_id = str(frame.get("transition_line_id", ""))
        if not transition_line_id:
            continue
        metrics = frame.get("fit_metrics", {})
        if not isinstance(metrics, dict):
            metrics = {}
        work_metrics = frame.get("source_work_window_metrics", {})
        if not isinstance(work_metrics, dict):
            work_metrics = {}
        frames[transition_line_id] = {
            "n_fit_pixels": int(metrics.get("n_fit_pixels", 0) or 0),
            "chi2": _finite_float(metrics.get("chi2")),
            "reduced_chi2": _finite_float(metrics.get("reduced_chi2")),
            "fit_rms": _finite_float(metrics.get("fit_rms")),
            "max_abs_residual_sigma": _finite_float(metrics.get("max_abs_residual_sigma")),
            "source_work_window": {
                "n_fit_pixels": int(work_metrics.get("n_fit_pixels", 0) or 0),
                "chi2": _finite_float(work_metrics.get("chi2")),
                "reduced_chi2": _finite_float(work_metrics.get("reduced_chi2")),
                "fit_rms": _finite_float(work_metrics.get("fit_rms")),
                "max_abs_residual_sigma": _finite_float(work_metrics.get("max_abs_residual_sigma")),
            },
        }
    return frames


def _compare_frame_fit_metrics(
    original_frames: dict[str, dict[str, Any]],
    refit_frames: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    original_fitted = {line_id for line_id, metrics in original_frames.items() if int(metrics.get("n_fit_pixels", 0) or 0) > 0}
    refit_fitted = {line_id for line_id, metrics in refit_frames.items() if int(metrics.get("n_fit_pixels", 0) or 0) > 0}
    common = sorted(original_fitted & refit_fitted)
    added = sorted(refit_fitted - original_fitted)
    removed = sorted(original_fitted - refit_fitted)
    common_chi2_original = 0.0
    common_chi2_refit = 0.0
    common_pixels_original = 0
    common_pixels_refit = 0
    per_frame: dict[str, Any] = {}
    for line_id in common:
        original = original_frames[line_id]
        refit = refit_frames[line_id]
        original_n = int(original.get("n_fit_pixels", 0) or 0)
        refit_n = int(refit.get("n_fit_pixels", 0) or 0)
        original_chi2 = _finite_float(original.get("chi2"))
        refit_chi2 = _finite_float(refit.get("chi2"))
        if np.isfinite(original_chi2):
            common_chi2_original += original_chi2
            common_pixels_original += original_n
        if np.isfinite(refit_chi2):
            common_chi2_refit += refit_chi2
            common_pixels_refit += refit_n
        per_frame[line_id] = {
            "original_n_fit_pixels": original_n,
            "refit_n_fit_pixels": refit_n,
            "fit_rms_delta": _finite_delta(_finite_float(refit.get("fit_rms")), _finite_float(original.get("fit_rms"))),
            "reduced_chi2_delta": _finite_delta(
                _finite_float(refit.get("reduced_chi2")),
                _finite_float(original.get("reduced_chi2")),
            ),
        }
    common_original_reduced = float(common_chi2_original / common_pixels_original) if common_pixels_original else float("nan")
    common_refit_reduced = float(common_chi2_refit / common_pixels_refit) if common_pixels_refit else float("nan")
    return {
        "coverage_changed": bool(added or removed),
        "original_fitted_transition_line_ids": sorted(original_fitted),
        "refit_fitted_transition_line_ids": sorted(refit_fitted),
        "added_fitted_transition_line_ids": added,
        "removed_fitted_transition_line_ids": removed,
        "common_fitted_transition_line_ids": common,
        "common_original_n_fit_pixels": common_pixels_original,
        "common_refit_n_fit_pixels": common_pixels_refit,
        "common_original_reduced_chi2": common_original_reduced,
        "common_refit_reduced_chi2": common_refit_reduced,
        "common_reduced_chi2_delta": _finite_delta(common_refit_reduced, common_original_reduced),
        "per_frame": per_frame,
    }


def _overlap_edit_policy(original_fit: dict[str, Any], refit_fit: dict[str, Any], overrides: dict[str, Any]) -> dict[str, Any]:
    original_frames = _frame_geometry(original_fit)
    refit_frames = _frame_geometry(refit_fit)
    frames = {**original_frames, **refit_frames}
    overlap_intervals = _velocity_frame_overlap_intervals_A(frames)
    refit_frame_metrics = _transition_frame_fit_metrics(refit_fit)
    removed_violations = _overlap_source_edits_missing_masks(
        source_edits=overrides.get("removed_sources", []),
        frames=frames,
        overlap_intervals=overlap_intervals,
        mask_intervals=overrides.get("fit_mask_intervals", []),
    )
    updated_violations = _overlap_source_edits_missing_masks(
        source_edits=[source for source in overrides.get("source_seeds", []) if "replace_component_index" in source],
        frames=frames,
        overlap_intervals=overlap_intervals,
        mask_intervals=overrides.get("fit_mask_intervals", []),
        center_key="replace_center_velocity_kms",
    )
    edit_annotations = _overlap_edit_annotations(overrides, frames, overlap_intervals)
    unaddressed_frames = _overlap_affected_frames_without_direct_edits(edit_annotations, overrides)
    unfit_frames = _overlap_affected_frames_without_fit_pixels(edit_annotations, refit_frame_metrics)
    return {
        "edits_in_overlapping_velocity_frames": bool(edit_annotations),
        "edit_annotations": edit_annotations,
        "overlap_affected_velocity_frames_unaddressed": bool(unaddressed_frames),
        "unaddressed_affected_transition_line_ids": unaddressed_frames,
        "overlap_affected_velocity_frames_without_fit_pixels": bool(unfit_frames),
        "affected_transition_line_ids_without_fit_pixels": unfit_frames,
        "missing_overlap_mask_after_source_removal": bool(removed_violations),
        "missing_overlap_mask_after_source_update": bool(updated_violations),
        "removed_source_violations": removed_violations,
        "updated_source_violations": updated_violations,
        "overlap_intervals_A": overlap_intervals,
    }


def _overlap_edit_annotations(
    overrides: dict[str, Any],
    frames: dict[str, dict[str, float]],
    overlap_intervals: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    annotations: list[dict[str, Any]] = []
    for source in overrides.get("source_seeds", []):
        if str(source.get("seed_source", "")) == "previous_fit_component":
            continue
        center_key = "replace_center_velocity_kms" if "replace_component_index" in source else "center_velocity_kms"
        annotation = _point_edit_overlap_annotation(
            tool_kind=str(source.get("seed_source", "source_seed")),
            transition_line_id=str(source.get("transition_line_id", "")),
            center_velocity_kms=_finite_float(source.get(center_key)),
            frames=frames,
            overlap_intervals=overlap_intervals,
            component_index=source.get("replace_component_index"),
        )
        if annotation:
            annotations.append(annotation)
    for removed in overrides.get("removed_sources", []):
        annotation = _point_edit_overlap_annotation(
            tool_kind="remove_absorption_source",
            transition_line_id=str(removed.get("transition_line_id", "")),
            center_velocity_kms=_finite_float(removed.get("center_velocity_kms")),
            frames=frames,
            overlap_intervals=overlap_intervals,
            component_index=removed.get("component_index"),
        )
        if annotation:
            annotations.append(annotation)
    for mask in overrides.get("fit_mask_intervals", []):
        annotation = _interval_edit_overlap_annotation(mask, frames, overlap_intervals)
        if annotation:
            annotations.append(annotation)
    return annotations


def _overlap_affected_frames_without_direct_edits(
    edit_annotations: list[dict[str, Any]],
    overrides: dict[str, Any],
) -> list[dict[str, Any]]:
    edited = _directly_edited_transition_line_ids(overrides)
    missing: dict[str, dict[str, Any]] = {}
    for annotation in edit_annotations:
        current = str(annotation.get("transition_line_id", ""))
        affected = _annotation_affected_transition_line_ids(annotation)
        for transition_line_id in affected:
            if not transition_line_id or transition_line_id == current or transition_line_id in edited:
                continue
            missing.setdefault(
                transition_line_id,
                {
                    "transition_line_id": transition_line_id,
                    "caused_by_transition_line_ids": [],
                    "overlap_intervals_A": [],
                    "suggested_action": (
                        "inspect this velocity frame because a same-wavelength overlap was edited in another frame; "
                        "add a corresponding mask/window/source edit or explicitly leave it for human review"
                    ),
                },
            )
            entry = missing[transition_line_id]
            if current and current not in entry["caused_by_transition_line_ids"]:
                entry["caused_by_transition_line_ids"].append(current)
            for interval in annotation.get("overlap_intervals_A", []):
                if interval not in entry["overlap_intervals_A"]:
                    entry["overlap_intervals_A"].append(interval)
    return [missing[key] for key in sorted(missing)]


def _overlap_affected_frames_without_fit_pixels(
    edit_annotations: list[dict[str, Any]],
    refit_frame_metrics: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    missing: dict[str, dict[str, Any]] = {}
    for annotation in edit_annotations:
        for transition_line_id in _annotation_affected_transition_line_ids(annotation):
            metrics = refit_frame_metrics.get(transition_line_id, {})
            if int(metrics.get("n_fit_pixels", 0) or 0) > 0:
                continue
            missing.setdefault(
                transition_line_id,
                {
                    "transition_line_id": transition_line_id,
                    "overlap_intervals_A": [],
                    "suggested_action": (
                        "this affected velocity frame has no refit residual samples, so the next round cannot judge the overlap there from RMS alone"
                    ),
                },
            )
            entry = missing[transition_line_id]
            for interval in annotation.get("overlap_intervals_A", []):
                if interval not in entry["overlap_intervals_A"]:
                    entry["overlap_intervals_A"].append(interval)
    return [missing[key] for key in sorted(missing)]


def _directly_edited_transition_line_ids(overrides: dict[str, Any]) -> set[str]:
    edited: set[str] = set()
    for key in ("source_seeds", "removed_sources", "fit_mask_intervals"):
        for item in overrides.get(key, []):
            transition_line_id = str(item.get("transition_line_id", ""))
            if transition_line_id:
                edited.add(transition_line_id)
    for transition_line_id in overrides.get("fit_windows", {}):
        if str(transition_line_id):
            edited.add(str(transition_line_id))
    return edited


def _annotation_affected_transition_line_ids(annotation: dict[str, Any]) -> set[str]:
    affected: set[str] = set()
    for interval in annotation.get("overlap_intervals_A", []):
        for transition_line_id in interval.get("transition_line_ids", []):
            if str(transition_line_id):
                affected.add(str(transition_line_id))
    return affected


def _point_edit_overlap_annotation(
    *,
    tool_kind: str,
    transition_line_id: str,
    center_velocity_kms: float,
    frames: dict[str, dict[str, float]],
    overlap_intervals: list[dict[str, Any]],
    component_index: Any = None,
) -> dict[str, Any] | None:
    frame = frames.get(transition_line_id)
    if not frame or not np.isfinite(center_velocity_kms):
        return None
    center_wavelength = _velocity_to_wavelength_A(frame["observed_center_A"], center_velocity_kms)
    overlaps = _matching_overlap_intervals(transition_line_id, center_wavelength, overlap_intervals)
    if not overlaps:
        return None
    return {
        "tool_kind": tool_kind,
        "transition_line_id": transition_line_id,
        "component_index": component_index,
        "center_velocity_kms": float(center_velocity_kms),
        "center_wavelength_A": float(center_wavelength),
        "overlap_intervals_A": overlaps,
        "feedback": "this edit touches pixels shared by multiple transition velocity frames; inspect all affected panels next round",
    }


def _interval_edit_overlap_annotation(
    mask: dict[str, Any],
    frames: dict[str, dict[str, float]],
    overlap_intervals: list[dict[str, Any]],
) -> dict[str, Any] | None:
    transition_line_id = str(mask.get("transition_line_id", ""))
    mask_interval = _mask_interval_to_wavelength_A(mask, frames)
    if mask_interval is None:
        return None
    overlaps = [
        interval
        for interval in overlap_intervals
        if transition_line_id in interval.get("transition_line_ids", [])
        and _overlap_width(mask_interval[0], mask_interval[1], interval["start_wavelength_A"], interval["end_wavelength_A"]) > 0.0
    ]
    if not overlaps:
        return None
    return {
        "tool_kind": f"set_fit_mask_interval:{mask.get('mask_kind', 'exclude')}",
        "transition_line_id": transition_line_id,
        "velocity_interval_kms": [mask.get("start_velocity_kms"), mask.get("end_velocity_kms")],
        "wavelength_interval_A": [float(mask_interval[0]), float(mask_interval[1])],
        "overlap_intervals_A": overlaps,
        "feedback": "this mask is in an observed-wavelength region shown by multiple velocity frames; inspect all affected panels next round",
    }


def _matching_overlap_intervals(
    transition_line_id: str,
    wavelength_A: float,
    overlap_intervals: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    return [
        interval
        for interval in overlap_intervals
        if transition_line_id in interval.get("transition_line_ids", [])
        and interval["start_wavelength_A"] - OVERLAP_MASK_REQUIRED_MARGIN_A
        <= wavelength_A
        <= interval["end_wavelength_A"] + OVERLAP_MASK_REQUIRED_MARGIN_A
    ]


def _frame_geometry(fit: dict[str, Any]) -> dict[str, dict[str, float]]:
    frames: dict[str, dict[str, float]] = {}
    for frame in fit.get("transition_frames", []):
        transition_line_id = str(frame.get("transition_line_id", ""))
        observed_center = _finite_float(frame.get("observed_center_A"))
        if not transition_line_id or not np.isfinite(observed_center):
            continue
        bounds = frame.get("velocity_frame", {}).get("bounds_kms", [])
        if not isinstance(bounds, list) or len(bounds) != 2:
            continue
        lower = _finite_float(bounds[0])
        upper = _finite_float(bounds[1])
        if not (np.isfinite(lower) and np.isfinite(upper)):
            continue
        frames[transition_line_id] = {
            "observed_center_A": float(observed_center),
            "velocity_lower_kms": float(min(lower, upper)),
            "velocity_upper_kms": float(max(lower, upper)),
        }
    return frames


def _velocity_frame_overlap_intervals_A(frames: dict[str, dict[str, float]]) -> list[dict[str, Any]]:
    items = sorted(frames.items())
    overlaps: list[dict[str, Any]] = []
    for index, (left_id, left) in enumerate(items):
        left_interval = _velocity_interval_to_wavelength_A(
            left["observed_center_A"],
            left["velocity_lower_kms"],
            left["velocity_upper_kms"],
        )
        for right_id, right in items[index + 1 :]:
            right_interval = _velocity_interval_to_wavelength_A(
                right["observed_center_A"],
                right["velocity_lower_kms"],
                right["velocity_upper_kms"],
            )
            lower = max(left_interval[0], right_interval[0])
            upper = min(left_interval[1], right_interval[1])
            if lower < upper:
                overlaps.append(
                    {
                        "transition_line_ids": [left_id, right_id],
                        "start_wavelength_A": float(lower),
                        "end_wavelength_A": float(upper),
                    }
                )
    return overlaps


def _overlap_source_edits_missing_masks(
    *,
    source_edits: list[dict[str, Any]],
    frames: dict[str, dict[str, float]],
    overlap_intervals: list[dict[str, Any]],
    mask_intervals: list[dict[str, Any]],
    center_key: str = "center_velocity_kms",
) -> list[dict[str, Any]]:
    violations: list[dict[str, Any]] = []
    for edit in source_edits:
        transition_line_id = str(edit.get("transition_line_id", ""))
        frame = frames.get(transition_line_id)
        center = _finite_float(edit.get(center_key))
        if not frame or not np.isfinite(center):
            continue
        center_wavelength = _velocity_to_wavelength_A(frame["observed_center_A"], center)
        matching_overlaps = _matching_overlap_intervals(transition_line_id, center_wavelength, overlap_intervals)
        if not matching_overlaps:
            continue
        if _has_overlap_fit_mask_for_edit(transition_line_id, matching_overlaps, frames, mask_intervals):
            continue
        violations.append(
            {
                "transition_line_id": transition_line_id,
                "component_index": edit.get("component_index", edit.get("replace_component_index")),
                "center_velocity_kms": float(center),
                "center_wavelength_A": float(center_wavelength),
                "overlap_intervals_A": matching_overlaps,
                "suggested_action": "add a tight set_fit_mask_interval for the same observed-wavelength overlap or explain why those pixels are no longer fitted",
            }
        )
    return violations


def _has_overlap_fit_mask_for_edit(
    transition_line_id: str,
    overlap_intervals: list[dict[str, Any]],
    frames: dict[str, dict[str, float]],
    mask_intervals: list[dict[str, Any]],
) -> bool:
    frame = frames.get(transition_line_id)
    if not frame:
        return False
    mask_wavelengths = [
        _mask_interval_to_wavelength_A(mask, frames)
        for mask in mask_intervals
        if str(mask.get("mask_kind", "exclude")) == "exclude"
        and str(mask.get("transition_line_id", transition_line_id)) == transition_line_id
    ]
    for mask_interval in mask_wavelengths:
        if mask_interval is None:
            continue
        for overlap in overlap_intervals:
            if _overlap_width(mask_interval[0], mask_interval[1], overlap["start_wavelength_A"], overlap["end_wavelength_A"]) > 0.0:
                return True
    return False


def _mask_interval_to_wavelength_A(mask: dict[str, Any], frames: dict[str, dict[str, float]]) -> tuple[float, float] | None:
    transition_line_id = str(mask.get("transition_line_id", ""))
    frame = frames.get(transition_line_id)
    if not frame:
        return None
    start = _finite_float(mask.get("start_velocity_kms"))
    end = _finite_float(mask.get("end_velocity_kms"))
    if not (np.isfinite(start) and np.isfinite(end)):
        return None
    return _velocity_interval_to_wavelength_A(frame["observed_center_A"], start, end)


def _velocity_interval_to_wavelength_A(observed_center_A: float, start_velocity_kms: float, end_velocity_kms: float) -> tuple[float, float]:
    start_A = _velocity_to_wavelength_A(observed_center_A, start_velocity_kms)
    end_A = _velocity_to_wavelength_A(observed_center_A, end_velocity_kms)
    return (float(min(start_A, end_A)), float(max(start_A, end_A)))


def _velocity_to_wavelength_A(observed_center_A: float, velocity_kms: float) -> float:
    return float(observed_center_A * (1.0 + velocity_kms / 299792.458))


def _source_work_window_metric(fit: dict[str, Any]) -> dict[str, Any]:
    metrics = fit.get("source_work_window_metrics", {})
    if not isinstance(metrics, dict):
        metrics = {}
    return {
        "n_fit_pixels": int(metrics.get("n_fit_pixels", 0) or 0),
        "chi2": _finite_float(metrics.get("chi2")),
        "reduced_chi2": _finite_float(metrics.get("reduced_chi2")),
        "fit_rms": _finite_float(metrics.get("fit_rms")),
        "max_abs_residual_sigma": _finite_float(metrics.get("max_abs_residual_sigma")),
    }


def _fit_mask_boundary_residual_diagnostics(
    fit: dict[str, Any],
    overrides: dict[str, Any],
    *,
    margin_kms: float = 80.0,
    threshold_sigma: float = 3.0,
) -> dict[str, Any]:
    intervals = [
        interval
        for interval in overrides.get("fit_mask_intervals", [])
        if isinstance(interval, dict) and str(interval.get("mask_kind", "exclude")) == "exclude"
    ]
    details: list[dict[str, Any]] = []
    high_count = 0
    max_abs = 0.0
    if not intervals:
        return {
            "n_high_residual_fit_pixels_near_mask": 0,
            "max_abs_residual_sigma_near_mask": 0.0,
            "details": [],
        }
    frames = {
        str(frame.get("transition_line_id")): frame
        for frame in fit.get("transition_frames", [])
        if frame.get("transition_line_id")
    }
    for interval in intervals:
        transition = str(interval.get("transition_line_id", ""))
        frame = frames.get(transition)
        if not frame:
            continue
        start = _finite_float(interval.get("start_velocity_kms"))
        end = _finite_float(interval.get("end_velocity_kms"))
        if not (np.isfinite(start) and np.isfinite(end)):
            continue
        lower, upper = sorted((float(start), float(end)))
        samples = [
            sample
            for sample in frame.get("residual_samples", [])
            if isinstance(sample, dict)
            and np.isfinite(_finite_float(sample.get("velocity_kms")))
            and np.isfinite(_finite_float(sample.get("residual_sigma")))
        ]
        near = [
            sample
            for sample in samples
            if (lower - margin_kms <= float(sample["velocity_kms"]) < lower)
            or (upper < float(sample["velocity_kms"]) <= upper + margin_kms)
        ]
        high = [sample for sample in near if abs(float(sample["residual_sigma"])) >= threshold_sigma]
        if not near:
            continue
        local_max = max(abs(float(sample["residual_sigma"])) for sample in near)
        max_abs = max(max_abs, local_max)
        high_count += len(high)
        details.append(
            {
                "transition_line_id": transition,
                "mask_interval_kms": [lower, upper],
                "n_fit_pixels_near_mask": len(near),
                "n_high_residual_fit_pixels_near_mask": len(high),
                "max_abs_residual_sigma_near_mask": float(local_max),
                "interpretation": (
                    "Residuals immediately outside a fit mask still participate in the fit; if these are unmasked "
                    "blend/overlap pixels they can pull source parameters."
                ),
            }
        )
    return {
        "n_high_residual_fit_pixels_near_mask": int(high_count),
        "max_abs_residual_sigma_near_mask": float(max_abs),
        "details": details,
    }


def _context_residual_diagnostics(
    fit: dict[str, Any],
    *,
    threshold_sigma: float = 3.0,
) -> dict[str, Any]:
    details: list[dict[str, Any]] = []
    high_count = 0
    max_abs = 0.0
    for frame in fit.get("transition_frames", []):
        transition = str(frame.get("transition_line_id", ""))
        has_diagnostic_samples = bool(frame.get("diagnostic_residual_samples"))
        raw_samples = frame.get("diagnostic_residual_samples") or frame.get("residual_samples", [])
        samples = [
            sample
            for sample in raw_samples
            if isinstance(sample, dict)
            and np.isfinite(_finite_float(sample.get("velocity_kms")))
            and np.isfinite(_finite_float(sample.get("residual_sigma")))
            and not (SOURCE_WORK_WINDOW_KMS[0] <= float(sample["velocity_kms"]) <= SOURCE_WORK_WINDOW_KMS[1])
            and (not has_diagnostic_samples or not bool(sample.get("used_in_fit", False)))
        ]
        if not samples:
            continue
        high = [sample for sample in samples if abs(float(sample["residual_sigma"])) >= threshold_sigma]
        local_max = max(abs(float(sample["residual_sigma"])) for sample in samples)
        max_abs = max(max_abs, local_max)
        high_count += len(high)
        if high:
            high_velocities = [float(sample["velocity_kms"]) for sample in high]
            details.append(
                {
                    "transition_line_id": transition,
                    "n_context_diagnostic_pixels": len(samples),
                    "n_high_context_diagnostic_pixels": len(high),
                    "n_high_context_diagnostic_residual_pixels": len(high),
                    "n_context_fit_pixels": 0,
                    "n_high_context_residual_fit_pixels": len(high),
                    "max_abs_residual_sigma_context": float(local_max),
                    "high_residual_velocity_range_kms": [float(min(high_velocities)), float(max(high_velocities))],
                    "interpretation": (
                        "A high residual outside the source work window is diagnostic context, not a source-fit pixel. "
                        "It should usually be handled with a mask/overlap explanation instead of adding a source there."
                    ),
                }
            )
    return {
        "n_high_context_diagnostic_residual_pixels": int(high_count),
        "n_high_context_residual_fit_pixels": int(high_count),
        "max_abs_residual_sigma_context": float(max_abs),
        "details": details,
    }


def _fit_mask_metrics(overrides: dict[str, Any]) -> dict[str, Any]:
    widths: list[float] = []
    core_width = 0.0
    boundary_width = 0.0
    context_width = 0.0
    core_intervals = 0
    for interval in overrides.get("fit_mask_intervals", []):
        if str(interval.get("mask_kind", "exclude")) != "exclude":
            continue
        start = _finite_float(interval.get("start_velocity_kms"))
        end = _finite_float(interval.get("end_velocity_kms"))
        if np.isfinite(start) and np.isfinite(end):
            lower, upper = sorted((float(start), float(end)))
            widths.append(abs(upper - lower))
            interval_core = _overlap_width(lower, upper, SOURCE_CORE_WINDOW_KMS[0], SOURCE_CORE_WINDOW_KMS[1])
            interval_work = _overlap_width(lower, upper, SOURCE_WORK_WINDOW_KMS[0], SOURCE_WORK_WINDOW_KMS[1])
            interval_boundary = max(0.0, interval_work - interval_core)
            interval_context = max(0.0, (upper - lower) - interval_work)
            core_width += interval_core
            boundary_width += interval_boundary
            context_width += interval_context
            if interval_core > 0.0:
                core_intervals += 1
    return {
        "n_intervals": int(len(widths)),
        "n_core_intervals": int(core_intervals),
        "total_width_kms": float(sum(widths)),
        "core_width_kms": float(core_width),
        "boundary_width_kms": float(boundary_width),
        "context_width_kms": float(context_width),
        "max_interval_width_kms": float(max(widths) if widths else 0.0),
        "max_width_kms": float(max(widths) if widths else 0.0),
        "widths_kms": [float(width) for width in widths],
    }


def _overlap_width(start: float, end: float, region_start: float, region_end: float) -> float:
    return float(max(0.0, min(end, region_end) - max(start, region_start)))


def _finite_float(value: Any) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return float("nan")
    return parsed if np.isfinite(parsed) else float("nan")


def _finite_delta(new_value: float, old_value: float) -> float:
    if np.isfinite(new_value) and np.isfinite(old_value):
        return float(new_value - old_value)
    return float("nan")


def _quality_rank(quality: str) -> int:
    return {
        "failed": 0,
        "unknown": 0,
        "inspect": 1,
        "fit": 1,
        "good": 2,
    }.get(str(quality), 0)


def _source_seed_from_args(args: dict[str, Any], *, seed_source: str) -> dict[str, Any] | None:
    center_velocity = _bounded_float(args.get("center_velocity_kms"), -5000.0, 5000.0)
    if center_velocity is None:
        return None
    source = {
        "transition_line_id": str(args.get("transition_line_id", "")),
        "center_velocity_kms": center_velocity,
        "seed_source": seed_source,
        "explicit_fit_control_source": seed_source in {"add_absorption_source", "update_absorption_source"},
        "reason": args.get("reason", ""),
    }
    bounds = {
        "center_prior_sigma_kms": (5.0, 500.0),
        "center_prior_half_width_kms": (20.0, 1200.0),
        "logN": (10.0, 18.5),
        "b_kms": (5.0, 200.0),
        "logN_lower": (8.0, 18.5),
        "logN_upper": (10.0, 22.0),
        "b_kms_lower": (1.0, 200.0),
        "b_kms_upper": (5.0, 500.0),
    }
    for key, (lower, upper) in bounds.items():
        value = _bounded_float(args.get(key), lower, upper)
        if value is not None:
            source[key] = value
    if "logN_lower" not in source:
        source["logN_lower"] = bounds["logN_lower"][0]
    if "logN_upper" not in source:
        source["logN_upper"] = bounds["logN_upper"][1]
    if "b_kms_lower" not in source:
        source["b_kms_lower"] = bounds["b_kms_lower"][0]
    if "b_kms_upper" not in source:
        source["b_kms_upper"] = bounds["b_kms_upper"][1]
    return source


def _updated_source_seed(record: dict[str, Any], args: dict[str, Any]) -> dict[str, Any] | None:
    component = _find_component(record, args.get("component_index"), args.get("transition_line_id"))
    merged = {**component, **{key: value for key, value in args.items() if value is not None}}
    source = _source_seed_from_args(merged, seed_source="update_absorption_source")
    if source and component:
        source["replace_component_index"] = int(component.get("component_index", args.get("component_index", -1)))
        source["replace_center_velocity_kms"] = float(component.get("center_velocity_kms", source["center_velocity_kms"]))
        source["replace_tolerance_kms"] = max(35.0, 2.5 * float(component.get("b_kms", 10.0) or 10.0))
    return source


def _removed_source(record: dict[str, Any], args: dict[str, Any]) -> dict[str, Any] | None:
    component = _find_component(record, args.get("component_index"), args.get("transition_line_id"))
    if not component or not _finite(component.get("center_velocity_kms")):
        return None
    return {
        "component_index": int(component.get("component_index", args.get("component_index", -1))),
        "transition_line_id": str(component.get("transition_line_id", args.get("transition_line_id", ""))),
        "center_velocity_kms": float(component["center_velocity_kms"]),
        "center_tolerance_kms": 45.0,
        "reason": args.get("reason", ""),
    }


def _find_component(
    record: dict[str, Any],
    component_index: Any,
    transition_line_id: Any = None,
) -> dict[str, Any]:
    try:
        target_index = int(component_index)
    except (TypeError, ValueError):
        return {}
    target_transition = str(transition_line_id or "")
    for result in record.get("fit_results", []):
        for component in result.get("components", []):
            if int(component.get("component_index", -1)) != target_index:
                continue
            if target_transition and str(component.get("transition_line_id", "")) != target_transition:
                continue
            return dict(component)
    return {}
