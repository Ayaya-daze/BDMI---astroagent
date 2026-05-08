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
    "add_continuum_anchor",
    "remove_continuum_anchor",
    "update_continuum_mask",
}


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
            _append_int(overrides["continuum_anchor_remove_indices"], args.get("anchor_index"))
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
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Apply already-canonical fit-control overrides and rerun the fit."""
    original_fit = _primary_fit(record)
    refit_record = build_review_record_from_window(
        window=window,
        window_metadata=record["input"],
        sample_id=sample_id or f"{record['sample_id']}_refit",
        source={**record.get("source", {}), "fit_control_source": patch.get("source", "fit_control_patch")},
        fit_control=overrides,
    )
    refit_fit = _primary_fit(refit_record)
    evaluation = evaluate_refit_patch(original_fit, refit_fit, overrides)
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
    application["controls"] = _public_fit_control_overrides(overrides)
    refit_record.setdefault("human_review", {})["fit_control_decision"] = evaluation["decision"]
    return refit_record, overrides


def merge_fit_control_overrides(*items: dict[str, Any]) -> dict[str, Any]:
    """Merge canonical override objects without reinterpreting component indexes."""
    merged = {
        "continuum_anchor_wavelengths_A": [],
        "continuum_anchor_nodes": [],
        "continuum_anchor_remove_indices": [],
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
    return merged


def empty_fit_control_overrides() -> dict[str, Any]:
    return merge_fit_control_overrides({})


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
    if np.isfinite(original_rms) and np.isfinite(refit_rms):
        rms_ratio = float(refit_rms / max(original_rms, 1e-6))
        if rms_ratio > 1.05:
            reasons.append("fit_rms_worsened")
    else:
        rms_ratio = float("nan")
        if np.isfinite(original_rms) and not np.isfinite(refit_rms):
            reasons.append("refit_rms_not_finite")

    original_work_rms = original_metrics["source_work_window"]["fit_rms"]
    refit_work_rms = refit_metrics["source_work_window"]["fit_rms"]
    if np.isfinite(original_work_rms) and np.isfinite(refit_work_rms):
        work_rms_delta = float(refit_work_rms - original_work_rms)
        work_rms_ratio = float(refit_work_rms / max(original_work_rms, 1e-6))
    else:
        work_rms_delta = float("nan")
        work_rms_ratio = float("nan")

    original_components = int(original_metrics["n_components"])
    refit_components = int(refit_metrics["n_components"])
    added_sources = int(len(overrides.get("source_seeds", [])))
    removed_sources = int(len(overrides.get("removed_sources", [])))
    mask_metrics = _fit_mask_metrics(overrides)
    material_edit_count = (
        added_sources
        + removed_sources
        + int(len(overrides.get("fit_mask_intervals", [])))
        + int(len(overrides.get("fit_windows", {})))
        + int(len(overrides.get("continuum_anchor_wavelengths_A", [])))
        + int(len(overrides.get("continuum_anchor_nodes", [])))
        + int(len(overrides.get("continuum_anchor_remove_indices", [])))
        + int(len(overrides.get("continuum_mask_intervals_A", [])))
    )
    if original_components > 0 and refit_components > max(original_components + max(added_sources, 1), 2 * original_components):
        reasons.append("component_count_exploded")
    if original_components > 0 and refit_components == 0:
        reasons.append("all_components_removed")
    if removed_sources == 0 and original_components > 0 and refit_components < max(1, original_components - 2):
        warnings.append("component_count_dropped_without_remove_request")

    original_fit_pixels = int(original_metrics["n_fit_pixels"])
    refit_fit_pixels = int(refit_metrics["n_fit_pixels"])
    if original_fit_pixels > 0 and refit_fit_pixels < 0.35 * original_fit_pixels:
        reasons.append("fit_window_or_mask_removed_too_many_pixels")
    if mask_metrics["max_interval_width_kms"] > 100.0:
        warnings.append("fit_mask_interval_too_broad")
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

    if reasons:
        decision = "rejected"
        accepted = False
        requires_human_review = True
    elif warnings:
        decision = "needs_human_review"
        accepted = False
        requires_human_review = True
    else:
        decision = "accepted"
        accepted = True
        requires_human_review = False

    return {
        "task": "fit_control_refit_evaluation",
        "decision": decision,
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
                "source_work_window_fit_rms": work_rms_delta,
                "source_work_window_fit_rms_ratio": work_rms_ratio,
                "n_components": refit_components - original_components,
                "n_fit_pixels": refit_fit_pixels - original_fit_pixels,
                "new_agent_review_reasons": new_review_reasons,
                "material_edit_count": material_edit_count,
                "fit_mask": mask_metrics,
            },
        },
        "policy": {
            "reject_if_rms_ratio_gt": 1.05,
            "reject_if_fit_pixels_lt_fraction": 0.35,
            "reject_quality_regression": True,
            "source_work_window_rms_is_scored_separately": True,
            "accepted_means_no_new_warnings": True,
        },
    }


def _normalize_patch_call(call: dict[str, Any], *, index: int) -> dict[str, Any]:
    name = str(call.get("name", ""))
    if name not in FIT_CONTROL_TOOL_NAMES:
        raise ValueError(f"unknown fit-control tool: {name}")
    arguments = call.get("arguments", {})
    if not isinstance(arguments, dict):
        raise ValueError(f"fit-control tool arguments must be an object for {name}")
    return {
        "sequence_index": int(index),
        "id": call.get("id"),
        "name": name,
        "arguments": arguments,
        "validated": True,
    }


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


def _bounded_float(value: Any, lower: float, upper: float) -> float | None:
    if not _finite(value):
        return None
    return float(np.clip(float(value), lower, upper))


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
        "fit_rms": _finite_float(fit.get("fit_rms")),
        "n_components": int(fit.get("n_components_fitted", 0) or 0),
        "n_fit_pixels": int(fit.get("n_fit_pixels", 0) or 0),
        "source_work_window": _source_work_window_metric(fit),
        "agent_review_required": bool(fit.get("agent_review", {}).get("required")),
        "agent_review_reasons": sorted(review_reasons),
    }


def _source_work_window_metric(fit: dict[str, Any]) -> dict[str, Any]:
    metrics = fit.get("source_work_window_metrics", {})
    if not isinstance(metrics, dict):
        metrics = {}
    return {
        "n_fit_pixels": int(metrics.get("n_fit_pixels", 0) or 0),
        "fit_rms": _finite_float(metrics.get("fit_rms")),
        "max_abs_residual_sigma": _finite_float(metrics.get("max_abs_residual_sigma")),
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
