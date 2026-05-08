"""LLM fit-control interfaces and bounded tool-loop orchestration."""

from importlib import import_module
from typing import Any

__all__ = [
    "LLMClient",
    "LLMMessage",
    "LLMResult",
    "OfflineReviewClient",
    "OpenAICompatibleClient",
    "FitControlLoopResult",
    "SOURCE_CORE_WINDOW_KMS",
    "SOURCE_WORK_WINDOW_KMS",
    "append_fit_control_patch",
    "build_fit_control_messages",
    "build_fit_control_overrides",
    "build_fit_control_patch",
    "build_fit_review_messages",
    "empty_fit_control_overrides",
    "evaluate_refit_patch",
    "merge_fit_control_overrides",
    "refit_record_with_overrides",
    "refit_record_with_patch",
    "run_fit_control",
    "run_fit_control_loop",
    "run_fit_review",
    "summarize_pending_fit_control",
]

_EXPORTS = {
    "SOURCE_CORE_WINDOW_KMS": "astroagent.agent.policy",
    "SOURCE_WORK_WINDOW_KMS": "astroagent.agent.policy",
    "append_fit_control_patch": "astroagent.agent.fit_control",
    "build_fit_control_overrides": "astroagent.agent.fit_control",
    "build_fit_control_patch": "astroagent.agent.fit_control",
    "empty_fit_control_overrides": "astroagent.agent.fit_control",
    "evaluate_refit_patch": "astroagent.agent.fit_control",
    "merge_fit_control_overrides": "astroagent.agent.fit_control",
    "refit_record_with_overrides": "astroagent.agent.fit_control",
    "refit_record_with_patch": "astroagent.agent.fit_control",
    "summarize_pending_fit_control": "astroagent.agent.fit_control",
    "LLMClient": "astroagent.agent.llm",
    "LLMMessage": "astroagent.agent.llm",
    "LLMResult": "astroagent.agent.llm",
    "OfflineReviewClient": "astroagent.agent.llm",
    "OpenAICompatibleClient": "astroagent.agent.llm",
    "build_fit_control_messages": "astroagent.agent.llm",
    "build_fit_review_messages": "astroagent.agent.llm",
    "run_fit_control": "astroagent.agent.llm",
    "run_fit_review": "astroagent.agent.llm",
    "FitControlLoopResult": "astroagent.agent.loop",
    "run_fit_control_loop": "astroagent.agent.loop",
}


def __getattr__(name: str) -> Any:
    module_name = _EXPORTS.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(import_module(module_name), name)
    globals()[name] = value
    return value
