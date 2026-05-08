from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timezone
from typing import Any


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
        "request_refit",
        "add_continuum_anchor",
        "remove_continuum_anchor",
        "update_continuum_mask",
    }
    return any(call.get("name") in refit_tools for call in tool_calls)

