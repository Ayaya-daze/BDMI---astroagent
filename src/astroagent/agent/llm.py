from __future__ import annotations

import json
import os
from base64 import b64encode
import hashlib
from importlib import resources
from pathlib import Path
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any, Protocol, Sequence

import numpy as np


FIT_CONTROL_SYSTEM_PROMPT = "fit_control_system.md"
FIT_CONTROL_USER_PROMPT = "fit_control_user.md"
PROMPT_PAYLOAD_PLACEHOLDER = "{{PROMPT_PAYLOAD_JSON}}"


FIT_CONTROL_REQUIRED_KEYS = {
    "task",
    "status",
    "tool_calls",
    "rationale",
}

FIT_REVIEW_REQUIRED_KEYS = {
    "task",
    "present",
    "system_plausible",
    "consistency_status",
    "human_review_required",
    "review_flags",
    "next_action",
    "quality",
    "rationale",
}


@dataclass(frozen=True)
class LLMMessage:
    role: str
    content: str | list[dict[str, Any]]


@dataclass(frozen=True)
class LLMResult:
    content: str
    model: str
    raw: dict[str, Any]
    tool_calls: list[dict[str, Any]] | None = None


@dataclass(frozen=True)
class FitControlPromptTemplates:
    system_template: str = FIT_CONTROL_SYSTEM_PROMPT
    user_template: str = FIT_CONTROL_USER_PROMPT
    template_dir: str | Path | None = None
    version: str | None = None


class LLMClient(Protocol):
    def complete(
        self,
        messages: list[LLMMessage],
        *,
        temperature: float = 0.0,
        tools: list[dict[str, Any]] | None = None,
    ) -> LLMResult:
        ...


class OfflineReviewClient:
    """Deterministic no-network client for tests and development without API keys."""

    def complete(
        self,
        messages: list[LLMMessage],
        *,
        temperature: float = 0.0,
        tools: list[dict[str, Any]] | None = None,
    ) -> LLMResult:
        is_fit_control = any("Task: fit_control" in _message_text(message.content) for message in messages)
        payload = _offline_fit_control_payload() if is_fit_control else {
            "task": "fit_review",
            "present": None,
            "system_plausible": None,
            "consistency_status": "inspect",
            "human_review_required": True,
            "review_flags": ["offline_llm_client"],
            "matched_expected_lines": [],
            "missing_expected_lines": [],
            "fit_results": [],
            "fit_ok": False,
            "next_action": "inspect",
            "preferred_n_components": None,
            "final_center_shift_kms": None,
            "quality": "unknown",
            "final_confidence": 0.0,
            "rationale": "Offline client did not call an external model.",
        }
        return LLMResult(content=json.dumps(payload, ensure_ascii=False), model="offline-review-client", raw={"offline": True})


class OpenAICompatibleClient:
    """Minimal chat-completions client for OpenAI-compatible providers.

    Configuration comes from environment variables by default:
    `ASTROAGENT_LLM_API_KEY`, `ASTROAGENT_LLM_MODEL`, and optional
    `ASTROAGENT_LLM_BASE_URL` or full `ASTROAGENT_LLM_API_URL`.
    """

    def __init__(
        self,
        *,
        api_key: str | None = None,
        model: str | None = None,
        base_url: str | None = None,
        api_url: str | None = None,
        timeout_s: float = 60.0,
    ) -> None:
        self.api_key = api_key if api_key is not None else os.environ.get("ASTROAGENT_LLM_API_KEY")
        self.model = model if model is not None else os.environ.get("ASTROAGENT_LLM_MODEL", "gpt-4.1-mini")
        resolved_base_url = base_url if base_url is not None else os.environ.get("ASTROAGENT_LLM_BASE_URL", "https://api.openai.com/v1")
        resolved_api_url = api_url if api_url is not None else os.environ.get("ASTROAGENT_LLM_API_URL")
        self.api_url = _chat_completions_url(resolved_base_url, resolved_api_url)
        self.timeout_s = float(timeout_s)
        if not self.api_key:
            raise ValueError("ASTROAGENT_LLM_API_KEY is required for OpenAICompatibleClient")

    def complete(
        self,
        messages: list[LLMMessage],
        *,
        temperature: float = 0.0,
        tools: list[dict[str, Any]] | None = None,
    ) -> LLMResult:
        body = {
            "model": self.model,
            "messages": [{"role": message.role, "content": message.content} for message in messages],
            "temperature": float(temperature),
        }
        if tools:
            body["tools"] = tools
            body["tool_choice"] = "auto"
        else:
            body["response_format"] = {"type": "json_object"}
        request = urllib.request.Request(
            self.api_url,
            data=json.dumps(body).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout_s) as response:
                raw = json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"LLM provider HTTP {exc.code}: {detail}") from exc
        except urllib.error.URLError as exc:
            raise RuntimeError(f"LLM provider request failed: {exc}") from exc

        try:
            message = raw["choices"][0]["message"]
            content = str(message.get("content") or "")
            tool_calls = message.get("tool_calls")
        except (KeyError, IndexError, TypeError) as exc:
            raise RuntimeError(f"LLM provider returned unexpected payload: {raw}") from exc
        return LLMResult(content=content, model=self.model, raw=raw, tool_calls=tool_calls)


def _chat_completions_url(base_url: str, api_url: str | None = None) -> str:
    if api_url:
        return str(api_url).rstrip("/")
    base = str(base_url).rstrip("/")
    if base.endswith("/chat/completions"):
        return base
    return f"{base}/chat/completions"


FIT_CONTROL_TOOLS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "add_absorption_source",
            "description": "Add an absorption component seed to one transition velocity frame.",
            "parameters": {
                "type": "object",
                "properties": {
                    "transition_line_id": {"type": "string"},
                    "center_velocity_kms": {"type": "number"},
                    "center_prior_sigma_kms": {"type": "number"},
                    "center_prior_half_width_kms": {"type": "number"},
                    "logN": {"type": "number"},
                    "logN_lower": {"type": "number"},
                    "logN_upper": {"type": "number"},
                    "b_kms": {"type": "number"},
                    "b_kms_lower": {"type": "number"},
                    "b_kms_upper": {"type": "number"},
                    "reason": {"type": "string"},
                },
                "required": ["transition_line_id", "center_velocity_kms", "reason"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "remove_absorption_source",
            "description": "Remove a fitted component judged to be spurious or redundant.",
            "parameters": {
                "type": "object",
                "properties": {
                    "component_index": {"type": "integer"},
                    "transition_line_id": {"type": "string"},
                    "reason": {"type": "string"},
                },
                "required": ["component_index", "reason"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "update_absorption_source",
            "description": "Adjust initial or bounded parameters for an existing fitted component before refit.",
            "parameters": {
                "type": "object",
                "properties": {
                    "component_index": {"type": "integer"},
                    "transition_line_id": {"type": "string"},
                    "center_velocity_kms": {"type": "number"},
                    "center_prior_sigma_kms": {"type": "number"},
                    "center_prior_half_width_kms": {"type": "number"},
                    "logN": {"type": "number"},
                    "logN_lower": {"type": "number"},
                    "logN_upper": {"type": "number"},
                    "b_kms": {"type": "number"},
                    "b_kms_lower": {"type": "number"},
                    "b_kms_upper": {"type": "number"},
                    "reason": {"type": "string"},
                },
                "required": ["component_index", "reason"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "set_fit_mask_interval",
            "description": "Include or exclude a velocity interval in one transition-frame fit.",
            "parameters": {
                "type": "object",
                "properties": {
                    "transition_line_id": {"type": "string"},
                    "start_velocity_kms": {"type": "number"},
                    "end_velocity_kms": {"type": "number"},
                    "mask_kind": {"type": "string", "enum": ["include", "exclude"]},
                    "reason": {"type": "string"},
                },
                "required": ["transition_line_id", "start_velocity_kms", "end_velocity_kms", "mask_kind", "reason"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "set_fit_window",
            "description": "Adjust transition-frame or local per-source fitting windows before refit.",
            "parameters": {
                "type": "object",
                "properties": {
                    "transition_line_id": {"type": "string"},
                    "transition_half_width_kms": {"type": "number"},
                    "local_fit_half_width_kms": {"type": "number"},
                    "sibling_mask_mode": {"type": "string", "enum": ["exclude", "allow_overlap"]},
                    "reason": {"type": "string"},
                },
                "required": ["transition_line_id", "reason"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "request_refit",
            "description": "Request a deterministic refit after at least one concrete source/window/mask/continuum edit. Calling this alone is only a note and will not change the fit.",
            "parameters": {
                "type": "object",
                "properties": {
                    "reason": {"type": "string"},
                },
                "required": ["reason"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "request_more_budget",
            "description": "Ask the harness for additional experiment rounds after the default round budget is reached. Use only with a concrete next experiment and evidence from the latest refit feedback.",
            "parameters": {
                "type": "object",
                "properties": {
                    "requested_rounds": {"type": "integer", "minimum": 1, "maximum": 5},
                    "next_experiment": {"type": "string"},
                    "reason": {"type": "string"},
                },
                "required": ["requested_rounds", "next_experiment", "reason"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "add_continuum_anchor",
            "description": "Add a continuum anchor point for local continuum reconstruction.",
            "parameters": {
                "type": "object",
                "properties": {
                    "wavelength_A": {"type": "number"},
                    "continuum_flux": {"type": "number"},
                    "reason": {"type": "string"},
                },
                "required": ["wavelength_A", "reason"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "remove_continuum_anchor",
            "description": "Remove a continuum anchor point that is biased by absorption or contamination.",
            "parameters": {
                "type": "object",
                "properties": {
                    "anchor_index": {"type": "integer"},
                    "wavelength_A": {"type": "number"},
                    "reason": {"type": "string"},
                },
                "required": ["anchor_index", "reason"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "update_continuum_mask",
            "description": "Add or remove continuum-mask intervals before rebuilding the continuum.",
            "parameters": {
                "type": "object",
                "properties": {
                    "start_wavelength_A": {"type": "number"},
                    "end_wavelength_A": {"type": "number"},
                    "mask_kind": {"type": "string", "enum": ["include", "exclude"]},
                    "reason": {"type": "string"},
                },
                "required": ["start_wavelength_A", "end_wavelength_A", "mask_kind", "reason"],
            },
        },
    },
]


def build_fit_control_messages(
    record: dict[str, Any],
    plot_image_path: str | Path | Sequence[str | Path] | None = None,
    prompt_templates: FitControlPromptTemplates | None = None,
) -> list[LLMMessage]:
    fit_summary = record.get("fit_results", [{}])[0]
    prompt_payload = {
        "sample_id": record.get("sample_id"),
        "input": record.get("input"),
        "window_summary": record.get("window_summary"),
        "absorber_hypothesis_check": record.get("absorber_hypothesis_check"),
        "plot_data": {
            "image_space": record.get("plot_data", {}).get("image_space"),
            "continuum_exclusion_intervals_A": record.get("plot_data", {}).get("continuum_exclusion_intervals_A"),
            "anchor_wavelengths_A": record.get("plot_data", {}).get("anchor_wavelengths_A"),
            "anchor_fluxes": record.get("plot_data", {}).get("anchor_fluxes"),
            "continuum_anchor_table": _continuum_anchor_table(record.get("plot_data", {})),
            "continuum_anchor_diagnostics": _continuum_anchor_diagnostics(record.get("plot_data", {})),
            "manual_anchor_nodes": record.get("plot_data", {}).get("manual_anchor_nodes"),
        },
        "fit_summary": _compact_fit_summary(fit_summary),
        "fit_control_context": {
            "loop": record.get("fit_control_loop"),
            "last_refit_feedback": _compact_refit_feedback(record.get("fit_control_last_refit_feedback")),
            "last_rejected_refit": _compact_refit_feedback(record.get("fit_control_last_rejected_refit")),
            "source_work_window_kms": record.get("fit_control_hints", {}).get("source_work_window_kms", [-400.0, 400.0]),
            "source_core_window_kms": record.get("fit_control_hints", {}).get("source_core_window_kms", [-360.0, 360.0]),
            "priority_order": record.get("fit_control_hints", {}).get(
                "priority_order",
                [
                    "source_work_window_residual",
                    "source_boundary_residual",
                    "high_sigma_context_mask_candidate",
                    "context_only_mask_candidate",
                ],
            ),
            "diagnostic_hints": {
                **_fit_control_diagnostic_hints(fit_summary),
                "velocity_frame_wavelength_overlaps": _velocity_frame_wavelength_overlaps(fit_summary),
                "sibling_work_window_projections": _sibling_work_window_projections(
                    fit_summary,
                    record.get("fit_control_hints", {}).get("source_work_window_kms", [-400.0, 400.0]),
                ),
                "residual_hints": record.get("fit_control_hints", {}).get("hints", []),
            },
        },
    }
    templates = prompt_templates or FitControlPromptTemplates()
    system = _load_prompt_template(templates.system_template, template_dir=templates.template_dir)
    user_text = _render_prompt_template(
        templates.user_template,
        payload=prompt_payload,
        template_dir=templates.template_dir,
    )
    return [
        LLMMessage(role="system", content=system),
        LLMMessage(role="user", content=_build_multimodal_content(user_text, plot_image_path)),
    ]


def _load_prompt_template(name: str | Path, *, template_dir: str | Path | None = None) -> str:
    if template_dir is not None:
        return (Path(template_dir) / Path(name)).read_text(encoding="utf-8").strip()
    return resources.files("astroagent.agent.prompts").joinpath(str(name)).read_text(encoding="utf-8").strip()


def _render_prompt_template(name: str | Path, *, payload: dict[str, Any], template_dir: str | Path | None = None) -> str:
    template = _load_prompt_template(name, template_dir=template_dir)
    payload_json = json.dumps(payload, ensure_ascii=False, indent=2)
    placeholder_count = template.count(PROMPT_PAYLOAD_PLACEHOLDER)
    if placeholder_count != 1:
        raise ValueError(f"prompt template {name} must contain exactly one {PROMPT_PAYLOAD_PLACEHOLDER}")
    return template.replace(PROMPT_PAYLOAD_PLACEHOLDER, payload_json)


def _prompt_template_metadata(
    templates: FitControlPromptTemplates,
    *,
    messages: list[LLMMessage],
) -> dict[str, Any]:
    system = str(messages[0].content) if messages else ""
    user_content = messages[1].content if len(messages) > 1 else ""
    user = _canonical_message_content(user_content)
    return {
        "system_template": str(templates.system_template),
        "user_template": str(templates.user_template),
        "template_dir": str(templates.template_dir) if templates.template_dir is not None else None,
        "version": templates.version,
        "system_sha256": hashlib.sha256(system.encode("utf-8")).hexdigest(),
        "user_sha256": hashlib.sha256(user.encode("utf-8")).hexdigest(),
    }


def run_fit_control(
    record: dict[str, Any],
    client: LLMClient,
    *,
    temperature: float = 0.0,
    plot_image_path: str | Path | Sequence[str | Path] | None = None,
    prompt_templates: FitControlPromptTemplates | None = None,
    allowed_tool_names: set[str] | None = None,
) -> dict[str, Any]:
    templates = prompt_templates or FitControlPromptTemplates()
    _validate_allowed_tool_names(allowed_tool_names)
    tools = _fit_control_tools_for(allowed_tool_names)
    messages = build_fit_control_messages(record, plot_image_path, prompt_templates=templates)
    result = client.complete(
        messages,
        temperature=temperature,
        tools=tools,
    )
    filtered_tool_calls: list[dict[str, Any]] = []
    if result.tool_calls:
        filtered_tool_calls = [
            {"id": tool_call.get("id"), "name": _tool_call_name(tool_call)}
            for tool_call in result.tool_calls
            if allowed_tool_names is not None and _tool_call_name(tool_call) not in allowed_tool_names
        ]
        tool_calls = [
            _normalize_tool_call(tool_call)
            for tool_call in result.tool_calls
            if allowed_tool_names is None or _tool_call_name(tool_call) in allowed_tool_names
        ]
        control = {
            "task": "fit_control",
            "status": "tool_calls",
            "tool_calls": tool_calls,
            "rationale": "Model requested fitting tool calls."
            if tool_calls
            else "Model requested tool calls outside this step's allowed tool set.",
        }
    else:
        try:
            control = _loads_first_json_object(result.content)
        except json.JSONDecodeError as exc:
            raise ValueError(f"fit-control response is not valid JSON and had no tool calls: {result.content}") from exc
    if not isinstance(control, dict):
        raise ValueError("fit control response must be a JSON object")
    filtered_content_tool_calls = _filter_disallowed_control_tool_calls(control, allowed_tool_names=allowed_tool_names)
    if filtered_content_tool_calls and not control.get("tool_calls") and str(control.get("status", "")).strip().lower() in {
        "tool_calls",
        "tools",
        "refit",
    }:
        control["rationale"] = "Model requested tool calls outside this step's allowed tool set."
    _normalize_fit_control_status(control)
    control.setdefault("rationale", "Model produced no rationale.")
    validate_fit_control(control, allowed_tool_names=allowed_tool_names)
    metadata = {**_llm_metadata(result), "prompt_templates": _prompt_template_metadata(templates, messages=messages)}
    all_filtered_tool_calls = [*filtered_tool_calls, *filtered_content_tool_calls]
    if all_filtered_tool_calls:
        metadata["filtered_tool_calls"] = all_filtered_tool_calls
    control["_llm_metadata"] = metadata
    return control


def build_fit_review_messages(
    record: dict[str, Any],
    plot_image_path: str | Path | Sequence[str | Path] | None = None,
) -> list[LLMMessage]:
    fit_summary = record.get("fit_results", [{}])[0]
    prompt_payload = {
        "sample_id": record.get("sample_id"),
        "input": record.get("input"),
        "window_summary": record.get("window_summary"),
        "absorber_hypothesis_check": record.get("absorber_hypothesis_check"),
        "fit_summary": _compact_fit_summary(fit_summary),
        "human_adjudication": record.get("human_adjudication"),
    }
    system = (
        "You review quasar absorption-line fitting outputs. "
        "Do not invent measurements. Do not make final science claims. "
        "Return one JSON object only."
    )
    user_text = (
        "Review this first-stage transition-frame fit. "
        "Decide whether it should be accepted, rejected, refit, get another component, or inspected by a human/agent. "
        "Required JSON keys: "
        + ", ".join(sorted(FIT_REVIEW_REQUIRED_KEYS))
        + "\n\n"
        + json.dumps(prompt_payload, ensure_ascii=False, indent=2)
    )
    return [
        LLMMessage(role="system", content=system),
        LLMMessage(role="user", content=_build_multimodal_content(user_text, plot_image_path)),
    ]


def run_fit_review(
    record: dict[str, Any],
    client: LLMClient,
    *,
    temperature: float = 0.0,
    plot_image_path: str | Path | Sequence[str | Path] | None = None,
) -> dict[str, Any]:
    result = client.complete(build_fit_review_messages(record, plot_image_path), temperature=temperature)
    try:
        review = json.loads(result.content)
    except json.JSONDecodeError as exc:
        raise ValueError(f"LLM response is not valid JSON: {result.content}") from exc
    validate_fit_review(review)
    review["_llm_metadata"] = _llm_metadata(result)
    return review


def _llm_metadata(result: LLMResult) -> dict[str, Any]:
    metadata: dict[str, Any] = {"model": result.model}
    usage = result.raw.get("usage") if isinstance(result.raw, dict) else None
    if isinstance(usage, dict):
        metadata["usage"] = usage
        for key in ("prompt_tokens", "completion_tokens", "total_tokens"):
            if key in usage:
                metadata[key] = usage[key]
    return metadata


def validate_fit_control(control: dict[str, Any], *, allowed_tool_names: set[str] | None = None) -> None:
    missing = sorted(FIT_CONTROL_REQUIRED_KEYS - set(control))
    if missing:
        raise ValueError(f"fit control is missing required keys: {missing}")
    if control["task"] != "fit_control":
        raise ValueError("fit control task must be 'fit_control'")
    if control["status"] not in {"tool_calls", "no_action", "inspect"}:
        raise ValueError("fit control status is invalid")
    if not isinstance(control["tool_calls"], list):
        raise ValueError("fit control tool_calls must be a list")
    _validate_allowed_tool_names(allowed_tool_names)
    allowed_tools = {tool["function"]["name"] for tool in FIT_CONTROL_TOOLS}
    if allowed_tool_names is not None:
        allowed_tools &= set(allowed_tool_names)
    for tool_call in control["tool_calls"]:
        if tool_call.get("name") not in allowed_tools:
            raise ValueError(f"unknown fit-control tool: {tool_call.get('name')}")
        _validate_fit_control_tool_arguments(tool_call)


def _validate_allowed_tool_names(allowed_tool_names: set[str] | None) -> None:
    if allowed_tool_names is None:
        return
    if not allowed_tool_names:
        raise ValueError("allowed fit-control tools must not be empty")
    known = {tool["function"]["name"] for tool in FIT_CONTROL_TOOLS}
    unknown = sorted(set(allowed_tool_names) - known)
    if unknown:
        raise ValueError(f"unknown allowed fit-control tools: {unknown}")


def _fit_control_tools_for(allowed_tool_names: set[str] | None) -> list[dict[str, Any]]:
    if allowed_tool_names is None:
        return FIT_CONTROL_TOOLS
    allowed = set(allowed_tool_names)
    return [tool for tool in FIT_CONTROL_TOOLS if tool["function"]["name"] in allowed]


def _filter_disallowed_control_tool_calls(
    control: dict[str, Any],
    *,
    allowed_tool_names: set[str] | None,
) -> list[dict[str, Any]]:
    if allowed_tool_names is None:
        return []
    tool_calls = control.get("tool_calls")
    if not isinstance(tool_calls, list):
        return []
    allowed = set(allowed_tool_names)
    kept: list[Any] = []
    filtered: list[dict[str, Any]] = []
    for tool_call in tool_calls:
        name = tool_call.get("name") if isinstance(tool_call, dict) else None
        if name in allowed:
            kept.append(tool_call)
        else:
            filtered.append({"id": tool_call.get("id") if isinstance(tool_call, dict) else None, "name": name})
    control["tool_calls"] = kept
    return filtered


def _canonical_message_content(content: str | list[dict[str, Any]]) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return json.dumps(content, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return str(content)


def _normalize_fit_control_status(control: dict[str, Any]) -> None:
    status = str(control.get("status", "")).strip().lower()
    normalized = status.replace("-", "_").replace(" ", "_")
    tool_calls = control.get("tool_calls", [])
    if isinstance(tool_calls, list) and tool_calls:
        control["status"] = "tool_calls"
        return
    if normalized in {"no_action", "none", "accept", "accepted"}:
        control["status"] = "no_action"
    elif normalized in {"inspect", "needs_inspection", "human_review", "needs_human_review", "review"}:
        control["status"] = "inspect"
    elif normalized in {"tool_calls", "tools", "refit"}:
        control["status"] = "tool_calls" if isinstance(tool_calls, list) and tool_calls else "inspect"


def validate_fit_review(review: dict[str, Any]) -> None:
    missing = sorted(FIT_REVIEW_REQUIRED_KEYS - set(review))
    if missing:
        raise ValueError(f"fit review is missing required keys: {missing}")
    if review["task"] != "fit_review":
        raise ValueError("fit review task must be 'fit_review'")
    if review["next_action"] not in {"accept", "refit", "add_component", "reject", "inspect"}:
        raise ValueError("fit review next_action is invalid")
    if not isinstance(review["review_flags"], list):
        raise ValueError("fit review review_flags must be a list")


def _compact_fit_summary(fit_summary: dict[str, Any]) -> dict[str, Any]:
    transition_frames = _compact_transition_frames(fit_summary.get("transition_frames"))
    components = _compact_components(fit_summary.get("components"))
    return {
        "fit_type": fit_summary.get("fit_type"),
        "fit_method": fit_summary.get("fit_method"),
        "fit_model": fit_summary.get("fit_model"),
        "success": fit_summary.get("success"),
        "quality": fit_summary.get("quality"),
        "chi2": fit_summary.get("chi2"),
        "reduced_chi2": fit_summary.get("reduced_chi2"),
        "fit_rms": fit_summary.get("fit_rms"),
        "source_work_window_metrics": fit_summary.get("source_work_window_metrics"),
        "lsf": fit_summary.get("lsf"),
        "instrument_lsf_applied": fit_summary.get("instrument_lsf_applied"),
        "instrument_lsf_applied_to_fit_likelihood": fit_summary.get("instrument_lsf_applied_to_fit_likelihood"),
        "lsf_model_policy": fit_summary.get("lsf_model_policy"),
        "n_transition_frames": fit_summary.get("n_transition_frames"),
        "n_components_fitted": fit_summary.get("n_components_fitted"),
        "transition_half_width_kms": fit_summary.get("transition_half_width_kms"),
        "agent_review": fit_summary.get("agent_review"),
        "transition_frames": transition_frames,
        "components": components,
    }


def _compact_refit_feedback(value: Any) -> Any:
    if not isinstance(value, dict):
        return value
    compact = _strip_fit_initializer_fields(value)
    if not isinstance(compact, dict):
        return compact
    if isinstance(compact.get("fit_summary"), dict):
        compact["fit_summary"] = _compact_feedback_fit_summary(compact["fit_summary"])
    if isinstance(compact.get("assessment_control"), dict):
        compact["assessment_control"] = _keep_keys(
            compact["assessment_control"],
            {"task", "status", "rationale", "tool_calls"},
        )
    return compact


def _compact_feedback_fit_summary(fit_summary: dict[str, Any]) -> dict[str, Any]:
    return {
        "success": fit_summary.get("success"),
        "quality": fit_summary.get("quality"),
        "fit_rms": fit_summary.get("fit_rms"),
        "reduced_chi2": fit_summary.get("reduced_chi2"),
        "source_work_window_metrics": fit_summary.get("source_work_window_metrics"),
        "lsf": fit_summary.get("lsf"),
        "instrument_lsf_applied_to_fit_likelihood": fit_summary.get("instrument_lsf_applied_to_fit_likelihood"),
        "n_transition_frames": fit_summary.get("n_transition_frames"),
        "n_components_fitted": fit_summary.get("n_components_fitted"),
        "agent_review": fit_summary.get("agent_review"),
        "transition_frame_summaries": _compact_feedback_transition_frames(fit_summary.get("transition_frames")),
        "component_summaries": _compact_feedback_components(fit_summary.get("components")),
    }


def _compact_feedback_transition_frames(value: Any) -> Any:
    frames = _strip_fit_initializer_fields(value)
    if not isinstance(frames, list):
        return frames
    summaries: list[dict[str, Any]] = []
    for frame in frames:
        if not isinstance(frame, dict):
            continue
        diagnostic = frame.get("lsf_diagnostic")
        summary = {
            "transition_line_id": frame.get("transition_line_id"),
            "quality": frame.get("quality"),
            "success": frame.get("success"),
            "fit_metrics": frame.get("fit_metrics"),
            "source_work_window_metrics": frame.get("source_work_window_metrics"),
            "agent_review_required": frame.get("agent_review_required"),
            "agent_review_reasons": frame.get("agent_review_reasons"),
            "n_successful_peak_fits": frame.get("n_successful_peak_fits"),
            "residual_sample_summary": _compact_residual_samples(frame.get("residual_samples", []), max_samples=5),
        }
        if isinstance(diagnostic, dict) and diagnostic.get("available"):
            summary["lsf_diagnostic"] = _compact_lsf_diagnostic(diagnostic)
        elif isinstance(diagnostic, dict):
            summary["lsf_diagnostic"] = _keep_keys(diagnostic, {"available", "comparison"})
        summaries.append(summary)
    return summaries


def _compact_feedback_components(value: Any, *, max_items: int = 8) -> Any:
    components = _strip_fit_initializer_fields(value)
    if not isinstance(components, list):
        return components
    fields = (
        "component_index",
        "transition_line_id",
        "fit_success",
        "center_velocity_kms",
        "logN",
        "b_kms",
        "fit_rms",
        "reduced_chi2",
        "diagnostic_flags",
        "diagnostic_warnings",
    )
    return [
        {key: component.get(key) for key in fields if key in component}
        for component in components[: int(max_items)]
        if isinstance(component, dict)
    ]


def _compact_transition_frames(value: Any) -> Any:
    frames = _strip_fit_initializer_fields(value)
    if not isinstance(frames, list):
        return frames
    return [_compact_transition_frame(frame) for frame in frames]


def _compact_transition_frame(frame: Any) -> Any:
    if not isinstance(frame, dict):
        return frame
    compact = {
        "transition_line_id": frame.get("transition_line_id"),
        "family": frame.get("family"),
        "ion": frame.get("ion"),
        "partner_line_id": frame.get("partner_line_id"),
        "family_context": frame.get("family_context"),
        "rest_wavelength_A": frame.get("rest_wavelength_A"),
        "observed_center_A": frame.get("observed_center_A"),
        "oscillator_strength": frame.get("oscillator_strength"),
        "velocity_frame": _keep_keys(frame.get("velocity_frame"), {"zero", "bounds_kms", "n_good_pixels"}),
        "simultaneous_fit_window": _keep_keys(
            frame.get("simultaneous_fit_window"),
            {
                "bounds_kms",
                "source_fit_bounds_kms",
                "source_fit_policy",
                "sibling_mask_mode",
                "masked_by",
            },
        ),
        "sibling_transition_masks": _compact_list(frame.get("sibling_transition_masks"), max_items=6),
        "quality": frame.get("quality"),
        "success": frame.get("success"),
        "quality_warnings": frame.get("quality_warnings"),
        "agent_review_required": frame.get("agent_review_required"),
        "agent_review_reasons": frame.get("agent_review_reasons"),
        "peak_detection": _keep_keys(
            frame.get("peak_detection"),
            {"method", "source_detection_mode", "n_detected_peaks", "center_use"},
        ),
        "n_successful_peak_fits": frame.get("n_successful_peak_fits"),
        "fit_metrics": frame.get("fit_metrics"),
        "source_work_window_metrics": frame.get("source_work_window_metrics"),
        "peaks": _compact_components(frame.get("peaks")),
        "residual_sample_summary": _compact_residual_samples(frame.get("residual_samples", [])),
        "diagnostic_residual_sample_summary": _compact_residual_samples(
            frame.get("diagnostic_residual_samples", []),
            max_samples=8,
        ),
        "lsf": frame.get("lsf"),
    }
    diagnostic = frame.get("lsf_diagnostic")
    if isinstance(diagnostic, dict) and diagnostic.get("available"):
        compact["lsf_diagnostic"] = _compact_lsf_diagnostic(diagnostic)
    elif isinstance(diagnostic, dict):
        compact["lsf_diagnostic"] = _keep_keys(diagnostic, {"available", "comparison"})
    return compact


def _compact_lsf_diagnostic(diagnostic: dict[str, Any]) -> dict[str, Any]:
    compact = {
        key: value
        for key, value in diagnostic.items()
        if key not in {"residual_samples", "diagnostic_residual_samples"}
    }
    compact["residual_sample_summary"] = _compact_lsf_residual_samples(diagnostic.get("residual_samples", []))
    compact["diagnostic_residual_sample_summary"] = _compact_lsf_residual_samples(
        diagnostic.get("diagnostic_residual_samples", []),
        max_samples=6,
    )
    return compact


def _compact_lsf_residual_samples(samples: Any, *, max_samples: int = 10) -> dict[str, Any]:
    return _compact_residual_samples(samples, max_samples=max_samples)


def _compact_residual_samples(samples: Any, *, max_samples: int = 10) -> dict[str, Any]:
    if not isinstance(samples, list) or not samples:
        return {"n_samples": 0, "top_abs_residual_samples": []}
    rows = [
        sample
        for sample in samples
        if isinstance(sample, dict)
        and _finite_number(sample.get("velocity_kms"))
        and _finite_number(sample.get("residual_sigma"))
    ]
    rows.sort(key=lambda item: abs(float(item.get("residual_sigma", 0.0))), reverse=True)
    fields = ("wavelength_A", "velocity_kms", "model_flux", "residual", "residual_sigma", "used_in_fit")
    return {
        "n_samples": len(samples),
        "top_abs_residual_samples": [
            {key: sample.get(key) for key in fields if key in sample}
            for sample in rows[: int(max_samples)]
        ],
    }


def _compact_components(value: Any, *, max_items: int = 12) -> Any:
    components = _strip_fit_initializer_fields(value)
    if not isinstance(components, list):
        return components
    return [_compact_component(component) for component in components[: int(max_items)]]


def _compact_component(component: Any) -> Any:
    if not isinstance(component, dict):
        return component
    fields = (
        "kind",
        "component_index",
        "transition_line_id",
        "seed_source",
        "reason",
        "fit_success",
        "quality",
        "center_velocity_kms",
        "center_offset_from_seed_kms",
        "seed_velocity_kms",
        "center_wavelength_A",
        "logN",
        "b_kms",
        "tau_scale",
        "fit_rms",
        "reduced_chi2",
        "n_fit_pixels",
        "trough_velocity_min_kms",
        "trough_velocity_max_kms",
        "diagnostic_flags",
        "diagnostic_warnings",
        "observed_equivalent_width_mA",
    )
    compact = {key: component.get(key) for key in fields if key in component}
    intervals = component.get("parameter_intervals")
    if isinstance(intervals, dict):
        compact["parameter_intervals"] = {
            key: intervals.get(key)
            for key in ("center_velocity_kms", "logN", "b_kms")
            if key in intervals
        }
    center_prior = component.get("center_prior")
    if isinstance(center_prior, dict):
        compact["center_prior"] = _keep_keys(center_prior, {"mean_kms", "sigma_kms", "bounds_kms"})
    return compact


def _keep_keys(value: Any, keys: set[str]) -> Any:
    if not isinstance(value, dict):
        return value
    return {key: value.get(key) for key in keys if key in value}


def _compact_list(value: Any, *, max_items: int = 8) -> Any:
    if not isinstance(value, list):
        return value
    return value[: int(max_items)]


def _strip_fit_initializer_fields(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            key: _strip_fit_initializer_fields(item)
            for key, item in value.items()
            if key not in {"map_parameters", "fit_parameter_summary"}
        }
    if isinstance(value, list):
        return [_strip_fit_initializer_fields(item) for item in value]
    return value


def _continuum_anchor_table(plot_data: dict[str, Any]) -> list[dict[str, Any]]:
    wavelengths = plot_data.get("anchor_wavelengths_A") or []
    fluxes = plot_data.get("anchor_fluxes") or []
    rows: list[dict[str, Any]] = []
    for index, (wavelength_A, flux) in enumerate(zip(wavelengths, fluxes, strict=False)):
        if not (_finite_number(wavelength_A) and _finite_number(flux)):
            continue
        rows.append(
            {
                "anchor_index": index,
                "wavelength_A": float(wavelength_A),
                "anchor_flux": float(flux),
            }
        )
    return rows


def _continuum_anchor_diagnostics(plot_data: dict[str, Any]) -> dict[str, Any]:
    rows = _continuum_anchor_table(plot_data)
    fluxes = np.asarray([row["anchor_flux"] for row in rows], dtype=float)
    finite = fluxes[np.isfinite(fluxes) & (fluxes > 0.0)]
    if len(finite) == 0:
        return {
            "median_anchor_flux": None,
            "suspicious_low_anchor_indices": [],
            "suspicious_high_anchor_indices": [],
            "guidance": "No finite positive continuum anchors available; rely on overview image sidebands.",
        }
    median_flux = float(np.nanmedian(finite))
    low_limit = 0.55 * median_flux
    high_limit = 1.45 * median_flux
    suspicious_low = [
        row
        for row in rows
        if np.isfinite(float(row["anchor_flux"])) and float(row["anchor_flux"]) < low_limit
    ]
    suspicious_high = [
        row
        for row in rows
        if np.isfinite(float(row["anchor_flux"])) and float(row["anchor_flux"]) > high_limit
    ]
    return {
        "median_anchor_flux": median_flux,
        "low_anchor_flux_threshold": low_limit,
        "high_anchor_flux_threshold": high_limit,
        "suspicious_low_anchor_indices": suspicious_low,
        "suspicious_high_anchor_indices": suspicious_high,
        "guidance": (
            "If suspicious_low_anchor_indices coincide with saturated troughs in the overview image, "
            "prefer remove_continuum_anchor for those anchor_index values before adding new anchors."
        ),
    }


def _fit_control_diagnostic_hints(fit_summary: dict[str, Any]) -> dict[str, Any]:
    components = fit_summary.get("components", [])
    component_centers = [
        {
            "component_index": component.get("component_index"),
            "transition_line_id": component.get("transition_line_id"),
            "center_velocity_kms": component.get("center_velocity_kms"),
            "b_kms": component.get("b_kms"),
        }
        for component in components
    ]
    hints: list[dict[str, Any]] = []
    for frame in fit_summary.get("transition_frames", []):
        transition_line_id = frame.get("transition_line_id")
        if not frame.get("success") or int(frame.get("n_successful_peak_fits", 0) or 0) == 0:
            hints.append(
                {
                    "kind": "transition_frame_has_no_fitted_sources",
                    "transition_line_id": transition_line_id,
                    "agent_review_reasons": frame.get("agent_review_reasons", []),
                    "suggested_action": (
                        "do not infer this panel is fine from missing residuals; if another frame's edit touches the same "
                        "observed-wavelength overlap, inspect this panel and either add a matching mask/window/source edit "
                        "or leave the overlap for human review"
                    ),
                }
            )
        frame_components = [
            item
            for item in component_centers
            if str(item.get("transition_line_id")) == str(transition_line_id)
            and _finite_number(item.get("center_velocity_kms"))
        ]
        centers = sorted(float(item["center_velocity_kms"]) for item in frame_components)
        for left, right in zip(centers[:-1], centers[1:], strict=False):
            gap = right - left
            if 70.0 <= gap <= 180.0:
                hints.append(
                    {
                        "kind": "possible_unresolved_blend_between_components",
                        "transition_line_id": transition_line_id,
                        "velocity_range_kms": [left, right],
                        "suggested_action": "inspect image; consider add_absorption_source at a real shoulder or update nearby components, not duplicate centers",
                    }
                )
        for component in frame_components:
            center = float(component["center_velocity_kms"])
            if abs(center) >= 0.75 * abs(float(fit_summary.get("transition_half_width_kms", 600.0))):
                half_width = max(35.0, min(100.0, 6.0 * float(component.get("b_kms") or 10.0)))
                hints.append(
                    {
                        "kind": "far_edge_absorption_mask_candidate",
                        "transition_line_id": transition_line_id,
                        "center_velocity_kms": center,
                        "suggested_fit_mask_interval_kms": [center - half_width, center + half_width],
                        "suggested_action": "if visually isolated from the target complex, use set_fit_mask_interval exclude instead of adding target components",
                    }
                )
            b_kms = float(component.get("b_kms") or 0.0)
            if b_kms >= 0.90 * float(fit_summary.get("local_fit_half_width_kms", 180.0)):
                hints.append(
                    {
                        "kind": "broad_component_reposition_candidate",
                        "transition_line_id": transition_line_id,
                        "component_index": component.get("component_index"),
                        "center_velocity_kms": center,
                        "b_kms": b_kms,
                        "suggested_action": (
                            "if masking an edge/blend in this frame, also update or replace this broad component so the "
                            "remaining source seed is centered on the unmasked target core/shoulder"
                        ),
                    }
                )
    return {
        "existing_component_centers": component_centers,
        "hints": hints,
    }


def _sibling_work_window_projections(
    fit_summary: dict[str, Any],
    source_work_window_kms: Sequence[float],
) -> list[dict[str, Any]]:
    try:
        source_start = float(source_work_window_kms[0])
        source_end = float(source_work_window_kms[1])
    except (IndexError, TypeError, ValueError):
        source_start, source_end = -400.0, 400.0
    source_lower, source_upper = sorted((source_start, source_end))
    projections: list[dict[str, Any]] = []
    frames = [
        frame
        for frame in fit_summary.get("transition_frames", [])
        if frame.get("transition_line_id")
        and _finite_number(frame.get("rest_wavelength_A"))
        and isinstance(frame.get("velocity_frame", {}).get("bounds_kms"), list)
        and len(frame.get("velocity_frame", {}).get("bounds_kms", [])) == 2
    ]
    for current in frames:
        current_id = str(current["transition_line_id"])
        current_rest = float(current["rest_wavelength_A"])
        bounds = current.get("velocity_frame", {}).get("bounds_kms", [-600.0, 600.0])
        frame_lower, frame_upper = sorted((float(bounds[0]), float(bounds[1])))
        for sibling in frames:
            sibling_id = str(sibling["transition_line_id"])
            if sibling_id == current_id:
                continue
            sibling_rest = float(sibling["rest_wavelength_A"])
            sibling_zero_in_current = 299792.458 * (sibling_rest / current_rest - 1.0)
            projected_lower = sibling_zero_in_current + source_lower
            projected_upper = sibling_zero_in_current + source_upper
            visible_lower = max(frame_lower, projected_lower)
            visible_upper = min(frame_upper, projected_upper)
            visible = visible_lower < visible_upper
            projections.append(
                {
                    "current_transition_line_id": current_id,
                    "sibling_transition_line_id": sibling_id,
                    "sibling_zero_velocity_in_current_frame_kms": sibling_zero_in_current,
                    "sibling_source_work_window_projected_kms": [projected_lower, projected_upper],
                    "visible_overlap_in_current_frame_kms": [visible_lower, visible_upper] if visible else [],
                    "suggested_audit": (
                        "if visible overlap appears as an edge trough or attracts a fitted source, decide tight mask/window edit "
                        "or explicitly explain why no mask is needed"
                    ),
                }
            )
    return projections


def _velocity_frame_wavelength_overlaps(fit_summary: dict[str, Any]) -> dict[str, Any]:
    frames: list[dict[str, Any]] = []
    for frame in fit_summary.get("transition_frames", []):
        transition_line_id = frame.get("transition_line_id")
        center = _finite_or_none(frame.get("observed_center_A"))
        bounds = frame.get("velocity_frame", {}).get("bounds_kms")
        if not transition_line_id or center is None or not isinstance(bounds, list) or len(bounds) != 2:
            continue
        try:
            v0, v1 = float(bounds[0]), float(bounds[1])
        except (TypeError, ValueError):
            continue
        if not (_finite_number(v0) and _finite_number(v1)):
            continue
        lower_v, upper_v = sorted((v0, v1))
        lower_A = center * (1.0 + lower_v / 299792.458)
        upper_A = center * (1.0 + upper_v / 299792.458)
        frames.append(
            {
                "transition_line_id": str(transition_line_id),
                "observed_center_A": center,
                "velocity_bounds_kms": [lower_v, upper_v],
                "wavelength_interval_A": [lower_A, upper_A],
            }
        )

    overlaps: list[dict[str, Any]] = []
    for index, left in enumerate(frames):
        left_interval = left["wavelength_interval_A"]
        for right in frames[index + 1 :]:
            right_interval = right["wavelength_interval_A"]
            overlap_start = max(left_interval[0], right_interval[0])
            overlap_end = min(left_interval[1], right_interval[1])
            if overlap_start >= overlap_end:
                continue
            overlaps.append(
                {
                    "transition_line_ids": [left["transition_line_id"], right["transition_line_id"]],
                    "overlap_wavelength_A": [overlap_start, overlap_end],
                    "overlap_width_A": overlap_end - overlap_start,
                    "interpretation": "features in this wavelength interval appear in both velocity panels at different velocities",
                }
            )
    return {
        "frames": frames,
        "overlaps": overlaps,
    }


def _finite_number(value: Any) -> bool:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return False
    return parsed == parsed and abs(parsed) != float("inf")


def _finite_or_none(value: Any) -> float | None:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if _finite_number(parsed) else None


def _normalize_tool_call(tool_call: dict[str, Any]) -> dict[str, Any]:
    function = tool_call.get("function", {}) if isinstance(tool_call, dict) else {}
    name = function.get("name") or tool_call.get("name")
    arguments_raw = function.get("arguments") or tool_call.get("arguments") or "{}"
    if isinstance(arguments_raw, str):
        try:
            arguments = json.loads(arguments_raw)
        except json.JSONDecodeError as exc:
            raise ValueError(f"invalid JSON arguments for tool call {name}") from exc
    else:
        arguments = arguments_raw
    return {
        "id": tool_call.get("id"),
        "name": name,
        "arguments": arguments,
    }


def _tool_call_name(tool_call: dict[str, Any]) -> Any:
    function = tool_call.get("function", {}) if isinstance(tool_call, dict) else {}
    return function.get("name") or tool_call.get("name")


def _validate_fit_control_tool_arguments(tool_call: dict[str, Any]) -> None:
    name = tool_call.get("name")
    arguments = tool_call.get("arguments")
    if not isinstance(arguments, dict):
        raise ValueError(f"fit-control tool arguments must be an object for {name}")
    schema = next((tool["function"]["parameters"] for tool in FIT_CONTROL_TOOLS if tool["function"]["name"] == name), None)
    if not isinstance(schema, dict):
        raise ValueError(f"unknown fit-control tool: {name}")
    required = schema.get("required", [])
    if isinstance(required, list):
        missing = [key for key in required if key not in arguments]
        if missing:
            raise ValueError(f"fit-control tool {name} missing required arguments: {missing}")
    properties = schema.get("properties", {})
    if not isinstance(properties, dict):
        return
    for key, value in arguments.items():
        if key not in properties:
            continue
        expected = properties[key].get("type") if isinstance(properties[key], dict) else None
        if expected == "number" and not _strict_finite_number(value):
            raise ValueError(f"fit-control tool {name} argument {key} must be a finite number")
        if expected == "integer":
            if not _strict_integer(value):
                raise ValueError(f"fit-control tool {name} argument {key} must be an integer")
            if "minimum" in properties[key] and int(value) < int(properties[key]["minimum"]):
                raise ValueError(f"fit-control tool {name} argument {key} must be >= {properties[key]['minimum']}")
            if "maximum" in properties[key] and int(value) > int(properties[key]["maximum"]):
                raise ValueError(f"fit-control tool {name} argument {key} must be <= {properties[key]['maximum']}")
        if expected == "string" and not isinstance(value, str):
            raise ValueError(f"fit-control tool {name} argument {key} must be a string")
        enum_values = properties[key].get("enum") if isinstance(properties[key], dict) else None
        if isinstance(enum_values, list) and value not in enum_values:
            raise ValueError(f"fit-control tool {name} argument {key} must be one of {enum_values}")


def _loads_first_json_object(text: str) -> dict[str, Any]:
    decoder = json.JSONDecoder()
    payload, _ = decoder.raw_decode(text.lstrip())
    if not isinstance(payload, dict):
        raise json.JSONDecodeError("fit-control response must be a JSON object", text, 0)
    return payload


def _strict_finite_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and _finite_number(value)


def _strict_integer(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _offline_fit_control_payload() -> dict[str, Any]:
    return {
        "task": "fit_control",
        "status": "no_action",
        "tool_calls": [],
        "rationale": "Offline client did not inspect plots or call fitting tools.",
    }


def _build_multimodal_content(text: str, image_path: str | Path | Sequence[str | Path] | None) -> str | list[dict[str, Any]]:
    if image_path is None:
        return text
    paths = [Path(item) for item in image_path] if isinstance(image_path, (list, tuple)) else [Path(image_path)]
    content: list[dict[str, Any]] = [{"type": "text", "text": text}]
    for index, path in enumerate(paths, start=1):
        if not path.exists():
            raise FileNotFoundError(f"image path does not exist: {path}")
        image_b64 = b64encode(path.read_bytes()).decode("ascii")
        content.append({"type": "text", "text": f"Image {index}: {path.name}"})
        content.append(
            {
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/png;base64,{image_b64}",
                    "detail": "high",
                },
            }
        )
    return content


def _message_text(content: str | list[dict[str, Any]]) -> str:
    if isinstance(content, str):
        return content
    chunks: list[str] = []
    for part in content:
        if isinstance(part, dict) and part.get("type") == "text":
            chunks.append(str(part.get("text", "")))
    return "\n".join(chunks)
