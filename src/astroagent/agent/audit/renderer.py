from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def render_fit_control_audit_report(
    *,
    record: dict[str, Any],
    history: list[dict[str, Any]],
    stop_reason: str,
    output_dir: str | Path,
    output_paths: list[dict[str, Path]] | None = None,
) -> dict[str, Path]:
    """Write human and machine audit summaries for one fit-control loop run."""
    output_path = Path(output_dir)
    sample_id = str(record.get("sample_id", "fit_control_loop"))
    audit_dir = output_path / f"{sample_id}.audit"
    audit_dir.mkdir(parents=True, exist_ok=True)

    payload = build_fit_control_audit_payload(
        record=record,
        history=history,
        stop_reason=stop_reason,
        output_paths=output_paths or [],
    )
    json_path = audit_dir / "audit_report.json"
    md_path = audit_dir / "audit_report.md"
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    md_path.write_text(_render_markdown(payload), encoding="utf-8")
    return {"audit_json": json_path, "audit_markdown": md_path}


def build_fit_control_audit_payload(
    *,
    record: dict[str, Any],
    history: list[dict[str, Any]],
    stop_reason: str,
    output_paths: list[dict[str, Path]],
) -> dict[str, Any]:
    fit = _primary_fit(record)
    key_metrics = _key_metrics(fit)
    rounds = [_round_summary(entry) for entry in history]
    findings = _findings_from_rounds(rounds)
    budget_events = _budget_events_from_rounds(rounds)
    human_action_items = _human_action_items(stop_reason, rounds, fit)
    return {
        "schema_version": 1,
        "task": "fit_control_audit_report",
        "sample_id": record.get("sample_id"),
        "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "stop_reason": stop_reason,
        "overall_status": _overall_status(stop_reason, fit),
        "executive_summary": _executive_summary(record, stop_reason, rounds, key_metrics),
        "key_metrics": key_metrics,
        "rounds": rounds,
        "budget_events": budget_events,
        "findings": findings,
        "human_action_items": human_action_items,
        "final_outputs": _public_output_paths(output_paths[-1] if output_paths else {}),
    }


def _primary_fit(record: dict[str, Any]) -> dict[str, Any]:
    for result in record.get("fit_results", []):
        if isinstance(result, dict) and (result.get("task") == "voigt_profile_fit" or result.get("fit_type")):
            return result
    return {}


def _key_metrics(fit: dict[str, Any]) -> dict[str, Any]:
    work = fit.get("source_work_window_metrics", {})
    if not isinstance(work, dict):
        work = {}
    return {
        "quality": fit.get("quality"),
        "success": fit.get("success"),
        "fit_rms": _finite_or_none(fit.get("fit_rms")),
        "reduced_chi2": _finite_or_none(fit.get("reduced_chi2")),
        "chi2": _finite_or_none(fit.get("chi2")),
        "n_components": _int_or_none(fit.get("n_components_fitted", fit.get("n_components"))),
        "work_window_fit_rms": _finite_or_none(work.get("fit_rms")),
        "work_window_reduced_chi2": _finite_or_none(work.get("reduced_chi2")),
        "work_window_max_abs_residual_sigma": _finite_or_none(work.get("max_abs_residual_sigma")),
    }


def _round_summary(entry: dict[str, Any]) -> dict[str, Any]:
    control = entry.get("control", {}) if isinstance(entry.get("control"), dict) else {}
    patch = entry.get("new_patch") if isinstance(entry.get("new_patch"), dict) else entry.get("patch", {})
    if not isinstance(patch, dict):
        patch = {}
    return {
        "round_index": _int_or_none(entry.get("round_index")),
        "input_sample_id": entry.get("input_sample_id"),
        "output_sample_id": entry.get("output_sample_id"),
        "control_status": control.get("status"),
        "decision_control_status": entry.get("decision_control", {}).get("status")
        if isinstance(entry.get("decision_control"), dict)
        else None,
        "assessment_control_status": entry.get("assessment_control", {}).get("status")
        if isinstance(entry.get("assessment_control"), dict)
        else None,
        "assessment_summary": entry.get("assessment_summary"),
        "decision": entry.get("decision"),
        "evaluation_controls_scope": entry.get("evaluation_controls_scope"),
        "advanced": bool(entry.get("advanced")),
        "selected_candidate": bool(entry.get("selected_candidate")),
        "budget_request": entry.get("budget_request"),
        "budget_decision": entry.get("budget_decision"),
        "requires_refit": bool(patch.get("requires_refit")),
        "n_tool_calls": int(len(patch.get("tool_calls", []))) if isinstance(patch.get("tool_calls", []), list) else 0,
        "tool_names": [
            str(call.get("name"))
            for call in patch.get("tool_calls", [])
            if isinstance(call, dict) and call.get("name")
        ],
        "rationale": patch.get("rationale") or entry.get("rationale") or control.get("rationale"),
        "gate_reasons": list(entry.get("gate_reasons", [])) if isinstance(entry.get("gate_reasons", []), list) else [],
        "gate_warnings": list(entry.get("gate_warnings", [])) if isinstance(entry.get("gate_warnings", []), list) else [],
        "gate_metrics": entry.get("gate_metrics", {}) if isinstance(entry.get("gate_metrics"), dict) else {},
    }


def _findings_from_rounds(rounds: list[dict[str, Any]]) -> list[dict[str, Any]]:
    findings: list[dict[str, Any]] = []
    for item in rounds:
        if item.get("decision") in {"rejected", "needs_human_review"}:
            findings.append(
                {
                    "kind": "gate_decision",
                    "round_index": item.get("round_index"),
                    "summary": f"Round ended with gate decision {item.get('decision')}.",
                    "evidence": [str(value) for value in [*item.get("gate_reasons", []), *item.get("gate_warnings", [])]][:8],
                    "severity": "high" if item.get("decision") == "rejected" else "medium",
                }
            )
        elif item.get("advanced"):
            findings.append(
                {
                    "kind": "accepted_candidate",
                    "round_index": item.get("round_index"),
                    "summary": "Round advanced the mainline fit-control state.",
                    "evidence": item.get("tool_names", [])[:8],
                    "severity": "low",
                }
            )
    return findings


def _budget_events_from_rounds(rounds: list[dict[str, Any]]) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    for item in rounds:
        decision = item.get("budget_decision")
        if not isinstance(decision, dict) or not decision.get("requested"):
            continue
        events.append(
            {
                "round_index": item.get("round_index"),
                "approved": bool(decision.get("approved")),
                "reason": decision.get("reason"),
                "round_limit": decision.get("round_limit"),
                "new_round_limit": decision.get("new_round_limit"),
                "hard_max_rounds": decision.get("hard_max_rounds"),
                "request": decision.get("request"),
            }
        )
    return events


def _human_action_items(stop_reason: str, rounds: list[dict[str, Any]], fit: dict[str, Any]) -> list[str]:
    items: list[str] = []
    if stop_reason in {"needs_human_review", "refit_rejected"} or stop_reason.endswith("_best_candidate_selected"):
        items.append("Inspect the final overview and per-transition plots before accepting the system.")
    if stop_reason == "no_refit_requested":
        items.append("Confirm that no fit-control edits were needed for this packet.")
    if stop_reason == "not_needed_good_fit":
        items.append("Spot-check the good-fit classification against the generated plots.")
    if stop_reason == "max_rounds_reached":
        items.append("Review whether another fit-control round or manual intervention is needed.")
    if any(item.get("gate_warnings") for item in rounds):
        items.append("Review gate warnings and decide whether they block scientific use.")
    if str(fit.get("quality", "")).lower() != "good":
        items.append("Treat the final fit as non-final until a human accepts the caveats.")
    return _dedupe(items)


def _overall_status(stop_reason: str, fit: dict[str, Any]) -> str:
    quality = str(fit.get("quality", "")).lower()
    if stop_reason == "not_needed_good_fit" and quality == "good":
        return "fit_accepted_clean"
    if stop_reason == "no_refit_requested":
        return "fit_inconclusive_human_required"
    if "refit_rejected" in stop_reason:
        return "fit_inconclusive_human_required"
    if "needs_human_review" in stop_reason:
        return "fit_acceptable_with_caveats"
    if "max_rounds_reached" in stop_reason:
        return "fit_acceptable_with_caveats" if quality == "good" else "fit_inconclusive_human_required"
    return "fit_acceptable_with_caveats"


def _executive_summary(
    record: dict[str, Any],
    stop_reason: str,
    rounds: list[dict[str, Any]],
    key_metrics: dict[str, Any],
) -> str:
    sample_id = record.get("sample_id", "unknown")
    n_rounds = len(rounds)
    n_components = key_metrics.get("n_components")
    quality = key_metrics.get("quality") or "unknown"
    return (
        f"Fit-control loop for {sample_id} stopped with `{stop_reason}` after {n_rounds} round(s). "
        f"The final fit quality is `{quality}` with {n_components if n_components is not None else 'unknown'} component(s). "
        "This report summarizes tool actions, deterministic gate outcomes, and human review items; it is not a final scientific adjudication."
    )


def _public_output_paths(paths: dict[str, Path]) -> dict[str, str]:
    return {key: str(value) for key, value in paths.items() if isinstance(value, Path)}


def _render_markdown(payload: dict[str, Any]) -> str:
    lines = [
        f"# Fit-Control Audit Report - {payload.get('sample_id')}",
        "",
        f"**Status:** `{payload.get('overall_status')}`",
        f"**Stop reason:** `{payload.get('stop_reason')}`",
        f"**Generated:** {payload.get('timestamp')}",
        "",
        "## Executive Summary",
        "",
        str(payload.get("executive_summary", "")),
        "",
        "## Key Metrics",
        "",
        "| Metric | Value |",
        "|---|---:|",
    ]
    metrics = payload.get("key_metrics", {})
    for key in (
        "quality",
        "success",
        "fit_rms",
        "reduced_chi2",
        "n_components",
        "work_window_fit_rms",
        "work_window_max_abs_residual_sigma",
    ):
        lines.append(f"| `{key}` | {_display(metrics.get(key))} |")

    lines.extend(["", "## Rounds", ""])
    rounds = payload.get("rounds", [])
    if rounds:
        lines.extend(["| Round | Decision | Assessment | Scope | Tools | Gate | Notes |", "|---:|---|---|---|---|---|---|"])
        for item in rounds:
            notes = "; ".join([*item.get("gate_reasons", []), *item.get("gate_warnings", [])]) or item.get("rationale") or ""
            lines.append(
                "| {round_index} | `{decision_status}` | `{assessment_status}` | `{scope}` | {tools} | `{gate_decision}` | {notes} |".format(
                    round_index=_display(item.get("round_index")),
                    decision_status=item.get("decision_control_status") or item.get("control_status") or "",
                    assessment_status=item.get("assessment_control_status") or "-",
                    scope=item.get("evaluation_controls_scope") or "-",
                    tools=", ".join(item.get("tool_names", [])) or "-",
                    gate_decision=item.get("decision") or "",
                    notes=_escape_table(str(notes))[:500],
                )
            )
            if item.get("assessment_summary"):
                lines.append(
                    "|  |  |  |  |  | assessment | {summary} |".format(
                        summary=_escape_table(str(item.get("assessment_summary")))[:500]
                    )
                )
    else:
        lines.append("No LLM rounds were run.")

    lines.extend(["", "## Findings", ""])
    findings = payload.get("findings", [])
    if findings:
        for finding in findings:
            lines.append(f"- **[{finding.get('severity')}] {finding.get('kind')}**: {finding.get('summary')}")
            for evidence in finding.get("evidence", []):
                lines.append(f"  - {evidence}")
    else:
        lines.append("No gate findings were recorded.")

    lines.extend(["", "## Budget Requests", ""])
    budget_events = payload.get("budget_events", [])
    if budget_events:
        for event in budget_events:
            status = "approved" if event.get("approved") else "denied"
            request = event.get("request") or {}
            lines.append(
                f"- Round {event.get('round_index')}: **{status}** ({event.get('reason')}); "
                f"limit {event.get('round_limit')} -> {event.get('new_round_limit', event.get('round_limit'))}; "
                f"next experiment: {request.get('next_experiment', '-')}"
            )
    else:
        lines.append("No additional budget was requested.")

    lines.extend(["", "## Human Action Items", ""])
    items = payload.get("human_action_items", [])
    if items:
        for item in items:
            lines.append(f"- [ ] {item}")
    else:
        lines.append("- [ ] No specific follow-up was generated; perform standard packet review.")

    lines.extend(["", "## Final Outputs", ""])
    outputs = payload.get("final_outputs", {})
    if outputs:
        for key, value in outputs.items():
            lines.append(f"- `{key}`: `{value}`")
    else:
        lines.append("No refit packet outputs were generated.")
    lines.append("")
    return "\n".join(lines)


def _finite_or_none(value: Any) -> float | None:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if parsed == parsed and abs(parsed) != float("inf") else None


def _int_or_none(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _display(value: Any) -> str:
    if value is None:
        return "-"
    if isinstance(value, float):
        return f"{value:.6g}"
    return str(value)


def _escape_table(value: str) -> str:
    return value.replace("|", "\\|").replace("\n", " ")


def _dedupe(items: list[str]) -> list[str]:
    seen: set[str] = set()
    output: list[str] = []
    for item in items:
        if item in seen:
            continue
        seen.add(item)
        output.append(item)
    return output
