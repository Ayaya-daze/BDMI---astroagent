from __future__ import annotations

from copy import deepcopy
from typing import Any

import numpy as np
import pandas as pd

from astroagent.review_packet import build_review_record_from_window


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
            overrides["continuum_mask_intervals_A"].append(
                {
                    "start_wavelength_A": start,
                    "end_wavelength_A": end,
                    "mask_kind": _clean_mask_kind(args.get("mask_kind", "exclude")),
                    "reason": args.get("reason", ""),
                }
            )
        elif name == "set_fit_mask_interval":
            start = _bounded_float(args.get("start_velocity_kms"), -5000.0, 5000.0)
            end = _bounded_float(args.get("end_velocity_kms"), -5000.0, 5000.0)
            if start is None or end is None:
                continue
            overrides["fit_mask_intervals"].append(
                {
                    "transition_line_id": args.get("transition_line_id"),
                    "start_velocity_kms": start,
                    "end_velocity_kms": end,
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
    refit_record.setdefault("fit_control_application", {})["evaluation"] = evaluation
    refit_record.setdefault("human_review", {})["fit_control_decision"] = evaluation["decision"]
    return refit_record, overrides


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

    original_components = int(original_metrics["n_components"])
    refit_components = int(refit_metrics["n_components"])
    added_sources = int(len(overrides.get("source_seeds", [])))
    removed_sources = int(len(overrides.get("removed_sources", [])))
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

    original_rank = _quality_rank(original_metrics["quality"])
    refit_rank = _quality_rank(refit_metrics["quality"])
    if refit_rank < original_rank:
        reasons.append("fit_quality_regressed")
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
                "n_components": refit_components - original_components,
                "n_fit_pixels": refit_fit_pixels - original_fit_pixels,
                "new_agent_review_reasons": new_review_reasons,
            },
        },
        "policy": {
            "reject_if_rms_ratio_gt": 1.05,
            "reject_if_fit_pixels_lt_fraction": 0.35,
            "reject_quality_regression": True,
            "accepted_means_no_new_warnings": True,
        },
    }


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
        "agent_review_required": bool(fit.get("agent_review", {}).get("required")),
        "agent_review_reasons": sorted(review_reasons),
    }


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
    return _source_seed_from_args(merged, seed_source="update_absorption_source")


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
