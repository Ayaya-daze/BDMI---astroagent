from __future__ import annotations

import json
import os
from base64 import b64encode
from pathlib import Path
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any, Protocol


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
    `ASTROAGENT_LLM_BASE_URL`.
    """

    def __init__(
        self,
        *,
        api_key: str | None = None,
        model: str | None = None,
        base_url: str | None = None,
        timeout_s: float = 60.0,
    ) -> None:
        self.api_key = api_key if api_key is not None else os.environ.get("ASTROAGENT_LLM_API_KEY")
        self.model = model if model is not None else os.environ.get("ASTROAGENT_LLM_MODEL", "gpt-4.1-mini")
        self.base_url = (base_url if base_url is not None else os.environ.get("ASTROAGENT_LLM_BASE_URL", "https://api.openai.com/v1")).rstrip("/")
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
            f"{self.base_url}/chat/completions",
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
            "description": "Run the deterministic fitter again after applying source/window/mask edits.",
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
    plot_image_path: str | Path | None = None,
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
            "manual_anchor_nodes": record.get("plot_data", {}).get("manual_anchor_nodes"),
        },
        "fit_summary": _compact_fit_summary(fit_summary),
    }
    system = (
        "You are the first-stage fitting agent for quasar absorption spectra. "
        "You may intervene in the fitting loop by calling tools: add sources, remove redundant sources, "
        "adjust source parameters, adjust fit masks/windows, edit continuum anchors, and request a refit. "
        "Continuum anchors follow the ABSpec nodes pattern: wavelength plus optional continuum_flux. "
        "Do not make final astrophysical claims; only improve or inspect the fit."
    )
    user_text = (
        "Task: fit_control. Inspect this transition-frame fit and decide which fitting tools to call. "
        "Use tools when a concrete fitting edit is needed. If no edit is safe, return JSON with task='fit_control', "
        "status='no_action', tool_calls=[], and a rationale.\n\n"
        + json.dumps(prompt_payload, ensure_ascii=False, indent=2)
    )
    return [
        LLMMessage(role="system", content=system),
        LLMMessage(role="user", content=_build_multimodal_content(user_text, plot_image_path)),
    ]


def run_fit_control(
    record: dict[str, Any],
    client: LLMClient,
    *,
    temperature: float = 0.0,
    plot_image_path: str | Path | None = None,
) -> dict[str, Any]:
    result = client.complete(
        build_fit_control_messages(record, plot_image_path),
        temperature=temperature,
        tools=FIT_CONTROL_TOOLS,
    )
    if result.tool_calls:
        control = {
            "task": "fit_control",
            "status": "tool_calls",
            "tool_calls": [_normalize_tool_call(tool_call) for tool_call in result.tool_calls],
            "rationale": "Model requested fitting tool calls.",
        }
    else:
        try:
            control = json.loads(result.content)
        except json.JSONDecodeError as exc:
            raise ValueError(f"fit-control response is not valid JSON and had no tool calls: {result.content}") from exc
    validate_fit_control(control)
    control["_llm_metadata"] = {"model": result.model}
    return control


def build_fit_review_messages(
    record: dict[str, Any],
    plot_image_path: str | Path | None = None,
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
    plot_image_path: str | Path | None = None,
) -> dict[str, Any]:
    result = client.complete(build_fit_review_messages(record, plot_image_path), temperature=temperature)
    try:
        review = json.loads(result.content)
    except json.JSONDecodeError as exc:
        raise ValueError(f"LLM response is not valid JSON: {result.content}") from exc
    validate_fit_review(review)
    review["_llm_metadata"] = {"model": result.model}
    return review


def validate_fit_control(control: dict[str, Any]) -> None:
    missing = sorted(FIT_CONTROL_REQUIRED_KEYS - set(control))
    if missing:
        raise ValueError(f"fit control is missing required keys: {missing}")
    if control["task"] != "fit_control":
        raise ValueError("fit control task must be 'fit_control'")
    if control["status"] not in {"tool_calls", "no_action", "inspect"}:
        raise ValueError("fit control status is invalid")
    if not isinstance(control["tool_calls"], list):
        raise ValueError("fit control tool_calls must be a list")
    allowed_tools = {tool["function"]["name"] for tool in FIT_CONTROL_TOOLS}
    for tool_call in control["tool_calls"]:
        if tool_call.get("name") not in allowed_tools:
            raise ValueError(f"unknown fit-control tool: {tool_call.get('name')}")


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
    return {
        "fit_type": fit_summary.get("fit_type"),
        "fit_method": fit_summary.get("fit_method"),
        "fit_model": fit_summary.get("fit_model"),
        "success": fit_summary.get("success"),
        "quality": fit_summary.get("quality"),
        "fit_rms": fit_summary.get("fit_rms"),
        "n_transition_frames": fit_summary.get("n_transition_frames"),
        "n_components_fitted": fit_summary.get("n_components_fitted"),
        "transition_half_width_kms": fit_summary.get("transition_half_width_kms"),
        "agent_review": fit_summary.get("agent_review"),
        "transition_frames": fit_summary.get("transition_frames"),
        "components": fit_summary.get("components"),
    }


def _normalize_tool_call(tool_call: dict[str, Any]) -> dict[str, Any]:
    function = tool_call.get("function", {}) if isinstance(tool_call, dict) else {}
    name = function.get("name") or tool_call.get("name")
    arguments_raw = function.get("arguments") or tool_call.get("arguments") or "{}"
    if isinstance(arguments_raw, str):
        try:
            arguments = json.loads(arguments_raw)
        except json.JSONDecodeError:
            arguments = {"_raw_arguments": arguments_raw}
    else:
        arguments = arguments_raw
    return {
        "id": tool_call.get("id"),
        "name": name,
        "arguments": arguments,
    }


def _offline_fit_control_payload() -> dict[str, Any]:
    return {
        "task": "fit_control",
        "status": "no_action",
        "tool_calls": [],
        "rationale": "Offline client did not inspect plots or call fitting tools.",
    }


def _build_multimodal_content(text: str, image_path: str | Path | None) -> str | list[dict[str, Any]]:
    if image_path is None:
        return text
    path = Path(image_path)
    if not path.exists():
        raise FileNotFoundError(f"image path does not exist: {path}")
    image_b64 = b64encode(path.read_bytes()).decode("ascii")
    return [
        {"type": "text", "text": text},
        {
            "type": "image_url",
            "image_url": {
                "url": f"data:image/png;base64,{image_b64}",
                "detail": "high",
            },
        },
    ]


def _message_text(content: str | list[dict[str, Any]]) -> str:
    if isinstance(content, str):
        return content
    chunks: list[str] = []
    for part in content:
        if isinstance(part, dict) and part.get("type") == "text":
            chunks.append(str(part.get("text", "")))
    return "\n".join(chunks)
