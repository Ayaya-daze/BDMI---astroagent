#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from pathlib import Path
from typing import Any

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from astroagent.agent.fit_control import (  # noqa: E402
    build_fit_control_overrides,
    build_fit_control_patch,
    empty_fit_control_overrides,
    merge_fit_control_overrides,
    refit_record_with_overrides,
)
from astroagent.agent.loop import _candidate_score, _is_better_candidate  # noqa: E402
from astroagent.agent.llm import OfflineReviewClient, OpenAICompatibleClient, run_fit_control, run_fit_review  # noqa: E402
from astroagent.review.packet import build_review_record, write_review_packet  # noqa: E402


Z_PATTERN = re.compile(r"_z(?P<z>[0-9]+p[0-9]+)")
TARGET_PATTERN = re.compile(r"target(?P<targetid>[0-9]+)")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run local DESI spectrum CSVs through review-packet and optional agent passes.")
    parser.add_argument("inputs", nargs="*", type=Path, help="CSV files or directories. Directories are scanned for *.spectrum.csv.")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/local_data_batch"))
    parser.add_argument("--limit", type=int, default=None, help="Optional cap after sorting discovered CSVs.")
    parser.add_argument("--run-agent", action="store_true", help="Also run one LLM pass per written review packet.")
    parser.add_argument("--agent-mode", choices=["fit_control", "fit_review"], default="fit_control")
    parser.add_argument("--max-agent-rounds", type=int, default=1, help="Maximum fit-control/refit rounds per sample.")
    parser.add_argument("--client", choices=["offline", "openai-compatible"], default="offline")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--include-images", action="store_true", help="Send overview and fit PNGs to the agent.")
    parser.add_argument(
        "--layout",
        choices=["structured", "flat"],
        default="structured",
        help="Structured stores one directory per sample and one subdirectory per agent round.",
    )
    parser.add_argument("--summary-name", default="summary", help="Base filename for summary JSON/CSV.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_roots = args.inputs or [Path("data/interim")]
    csv_paths = discover_csvs(input_roots)
    if args.limit is not None:
        csv_paths = csv_paths[: args.limit]

    args.output_dir.mkdir(parents=True, exist_ok=True)
    client = None
    if args.run_agent:
        client = OfflineReviewClient() if args.client == "offline" else OpenAICompatibleClient()

    rows: list[dict[str, Any]] = []
    for index, csv_path in enumerate(csv_paths, start=1):
        row = run_one(
            csv_path,
            output_dir=args.output_dir,
            run_agent=args.run_agent,
            agent_mode=args.agent_mode,
            client=client,
            temperature=args.temperature,
            include_images=args.include_images,
            max_agent_rounds=args.max_agent_rounds,
            layout=args.layout,
        )
        row["batch_index"] = index
        rows.append(row)
        print(status_line(row, index=index, total=len(csv_paths)), flush=True)

    summary = build_summary(rows, input_roots=input_roots, output_dir=args.output_dir)
    summary_json = args.output_dir / f"{args.summary_name}.json"
    summary_csv = args.output_dir / f"{args.summary_name}.csv"
    summary_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    write_csv(summary_csv, rows)
    print(f"summary json: {summary_json}")
    print(f"summary csv:  {summary_csv}")
    print(
        "done: "
        f"{summary['n_success']}/{summary['n_inputs']} review packets, "
        f"{summary['agent']['n_calls']} agent calls, "
        f"{summary['agent']['total_tokens']} total tokens"
    )


def discover_csvs(paths: list[Path]) -> list[Path]:
    discovered: list[Path] = []
    for path in paths:
        if path.is_dir():
            discovered.extend(sorted(path.rglob("*.spectrum.csv")))
        elif path.is_file():
            discovered.append(path)
        else:
            raise FileNotFoundError(path)
    return sorted(dict.fromkeys(discovered))


def run_one(
    csv_path: Path,
    *,
    output_dir: Path,
    run_agent: bool,
    agent_mode: str,
    client: Any,
    temperature: float,
    include_images: bool,
    max_agent_rounds: int,
    layout: str,
) -> dict[str, Any]:
    family, line_id = infer_line_id(csv_path)
    z_sys = infer_z_sys(csv_path)
    targetid = infer_targetid(csv_path)
    sample_id = csv_path.stem.replace(".spectrum", "")
    row: dict[str, Any] = {
        "input_csv": str(csv_path),
        "sample_id": sample_id,
        "targetid": targetid,
        "family": family,
        "line_id": line_id,
        "z_sys": z_sys,
        "status": "started",
    }
    try:
        spectrum = pd.read_csv(csv_path)
        record, window = build_review_record(
            spectrum=spectrum,
            line_id=line_id,
            z_sys=z_sys,
            sample_id=sample_id,
            source={"kind": "local_csv", "path": str(csv_path)},
        )
        sample_dir = sample_output_dir(output_dir, sample_id, layout=layout)
        initial_dir = sample_dir / "initial" if layout == "structured" else output_dir
        paths = write_review_packet(record, window, initial_dir)
        fit = record.get("fit_results", [{}])[0]
        row.update(
            {
                "status": "written",
                "sample_dir": str(sample_dir),
                "initial_dir": str(initial_dir),
                "review_json": str(paths["json"]),
                "window_csv": str(paths["csv"]),
                "plot_png": str(paths["plot_png"]),
                "overview_png": str(paths["overview_png"]),
                "absorber_status": record.get("absorber_hypothesis_check", {}).get("status"),
                "fit_success": fit.get("success"),
                "fit_quality": fit.get("quality"),
                "chi2": fit.get("chi2"),
                "reduced_chi2": fit.get("reduced_chi2"),
                "fit_rms": fit.get("fit_rms"),
                "source_work_chi2": fit.get("source_work_window_metrics", {}).get("chi2"),
                "source_work_reduced_chi2": fit.get("source_work_window_metrics", {}).get("reduced_chi2"),
                "source_work_fit_rms": fit.get("source_work_window_metrics", {}).get("fit_rms"),
                "n_components_fitted": fit.get("n_components_fitted"),
                "n_transition_frames": fit.get("n_transition_frames"),
                "agent_review_required": fit.get("agent_review", {}).get("required"),
                "agent_review_reasons": "|".join(fit.get("agent_review", {}).get("reasons", [])),
            }
        )
        if run_agent:
            try:
                if agent_mode == "fit_control":
                    run_fit_control_rounds(
                        row,
                        record,
                        window,
                        paths,
                        sample_dir=sample_dir,
                        client=client,
                        temperature=temperature,
                        include_images=include_images,
                        max_rounds=max_agent_rounds,
                        layout=layout,
                    )
                else:
                    run_agent_once(row, record, paths, mode=agent_mode, client=client, temperature=temperature, include_images=include_images)
            except Exception as exc:
                row.update(
                    {
                        "agent_status": "failed",
                        "agent_error_type": type(exc).__name__,
                        "agent_error": str(exc),
                    }
                )
        write_sample_manifest(sample_dir, row)
    except Exception as exc:
        row.update({"status": "failed", "error_type": type(exc).__name__, "error": str(exc)})
    return row


def run_agent_once(
    row: dict[str, Any],
    record: dict[str, Any],
    paths: dict[str, Path],
    *,
    mode: str,
    client: Any,
    temperature: float,
    include_images: bool,
) -> None:
    image_paths = None
    if include_images:
        image_paths = [paths["overview_png"], paths["plot_png"]]
    if mode == "fit_control":
        result = run_fit_control(record, client, temperature=temperature, plot_image_path=image_paths)
        patch = build_fit_control_patch(result)
        output_path = paths["json"].with_suffix("").with_suffix(".llm_control.json")
        patch_path = paths["json"].with_suffix("").with_suffix(".fit_control_patch.json")
        patch_path.write_text(json.dumps(patch, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        row["agent_patch_json"] = str(patch_path)
        row["agent_requires_refit"] = patch.get("requires_refit")
        row["agent_n_tool_calls"] = len(patch.get("tool_calls", []))
    else:
        result = run_fit_review(record, client, temperature=temperature, plot_image_path=image_paths)
        output_path = paths["json"].with_suffix("").with_suffix(".llm_review.json")
        row["agent_next_action"] = result.get("next_action")
    output_path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    usage = result.get("_llm_metadata", {}).get("usage", {})
    row.update(
        {
            "agent_status": result.get("status") or result.get("quality"),
            "agent_json": str(output_path),
            "agent_model": result.get("_llm_metadata", {}).get("model"),
            "agent_prompt_tokens": usage.get("prompt_tokens", result.get("_llm_metadata", {}).get("prompt_tokens", 0)),
            "agent_completion_tokens": usage.get("completion_tokens", result.get("_llm_metadata", {}).get("completion_tokens", 0)),
            "agent_total_tokens": usage.get("total_tokens", result.get("_llm_metadata", {}).get("total_tokens", 0)),
        }
    )


def run_fit_control_rounds(
    row: dict[str, Any],
    record: dict[str, Any],
    window: pd.DataFrame,
    paths: dict[str, Path],
    *,
    sample_dir: Path,
    client: Any,
    temperature: float,
    include_images: bool,
    max_rounds: int,
    layout: str,
) -> None:
    state_record = record
    state_paths = paths
    active_controls = empty_fit_control_overrides()
    trial_controls = empty_fit_control_overrides()
    round_summaries: list[dict[str, Any]] = []
    total_tokens = 0
    stop_reason = "max_rounds_reached"
    final_record = record
    final_paths = paths
    best_record: dict[str, Any] | None = None
    best_paths: dict[str, Path] | None = None
    best_score: tuple[float, float, int] | None = None
    last_rejected_refit: dict[str, Any] | None = None
    last_refit_feedback: dict[str, Any] | None = None

    for round_index in range(1, max(1, int(max_rounds)) + 1):
        round_dir = agent_round_dir(sample_dir, round_index, layout=layout)
        round_dir.mkdir(parents=True, exist_ok=True)
        image_paths = None
        if include_images:
            image_paths = [state_paths["overview_png"], state_paths["plot_png"]]
            if last_rejected_refit:
                image_paths.extend(
                    [
                        Path(last_rejected_refit["overview_png"]),
                        Path(last_rejected_refit["plot_png"]),
                    ]
                )

        prompt_record = state_record
        if last_rejected_refit or last_refit_feedback:
            prompt_record = dict(state_record)
        if last_refit_feedback:
            prompt_record["fit_control_last_refit_feedback"] = {
                key: last_refit_feedback[key]
                for key in (
                    "sample_id",
                    "round_index",
                    "review_json",
                    "plot_png",
                    "overview_png",
                    "evaluation",
                    "patch",
                    "fit_summary",
                )
                if key in last_refit_feedback
            }
        if last_rejected_refit:
            prompt_record["fit_control_last_rejected_refit"] = {
                key: last_rejected_refit[key]
                for key in (
                    "sample_id",
                    "round_index",
                    "review_json",
                    "plot_png",
                    "overview_png",
                    "evaluation",
                    "patch",
                    "fit_summary",
                )
                if key in last_rejected_refit
            }

        control = run_fit_control(prompt_record, client, temperature=temperature, plot_image_path=image_paths)
        usage = control.get("_llm_metadata", {}).get("usage", {})
        round_tokens = int(usage.get("total_tokens", control.get("_llm_metadata", {}).get("total_tokens", 0)) or 0)
        total_tokens += round_tokens
        patch = build_fit_control_patch(control)
        round_overrides = build_fit_control_overrides(state_record, patch)
        candidate_controls = merge_fit_control_overrides(trial_controls, round_overrides)
        effective_patch = dict(patch)
        effective_patch["tool_calls"] = candidate_controls.get("tool_calls", patch.get("tool_calls", []))

        control_path = round_dir / f"{row['sample_id']}.round_{round_index:02d}.llm_control.json"
        patch_path = round_dir / f"{row['sample_id']}.round_{round_index:02d}.fit_control_patch.json"
        control_path.write_text(json.dumps(control, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        patch_path.write_text(json.dumps(effective_patch, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

        round_summary: dict[str, Any] = {
            "round_index": round_index,
            "round_dir": str(round_dir),
            "control_json": str(control_path),
            "patch_json": str(patch_path),
            "status": control.get("status"),
            "n_tool_calls": len(patch.get("tool_calls", [])),
            "requires_refit": bool(patch.get("requires_refit")),
            "prompt_tokens": usage.get("prompt_tokens", control.get("_llm_metadata", {}).get("prompt_tokens", 0)),
            "completion_tokens": usage.get("completion_tokens", control.get("_llm_metadata", {}).get("completion_tokens", 0)),
            "total_tokens": round_tokens,
        }

        if not patch.get("requires_refit"):
            stop_reason = "no_refit_requested"
            round_summaries.append(round_summary)
            break

        refit_sample_id = f"{row['sample_id']}_round{round_index:02d}"
        refit_record, applied_controls = refit_record_with_overrides(
            state_record,
            window,
            effective_patch,
            candidate_controls,
            sample_id=refit_sample_id,
        )
        refit_paths = write_review_packet(refit_record, window, round_dir)
        evaluation = refit_record.get("fit_control_evaluation", {})
        refit_fit = refit_record.get("fit_results", [{}])[0]
        feedback = {
            "sample_id": refit_record.get("sample_id"),
            "round_index": round_index,
            "review_json": str(refit_paths["json"]),
            "plot_png": str(refit_paths["plot_png"]),
            "overview_png": str(refit_paths["overview_png"]),
            "evaluation": evaluation,
            "patch": patch,
            "fit_summary": refit_fit,
        }
        round_summary.update(
            {
                "refit_review_json": str(refit_paths["json"]),
                "refit_plot_png": str(refit_paths["plot_png"]),
                "decision": evaluation.get("decision"),
                "outcome": evaluation.get("outcome"),
                "accepted": evaluation.get("accepted"),
                "requires_human_review": evaluation.get("requires_human_review"),
                "gate_reasons": "|".join(evaluation.get("reasons", [])),
                "gate_warnings": "|".join(evaluation.get("warnings", [])),
                "refit_quality": refit_fit.get("quality"),
                "refit_chi2": refit_fit.get("chi2"),
                "refit_reduced_chi2": refit_fit.get("reduced_chi2"),
                "refit_fit_rms": refit_fit.get("fit_rms"),
                "refit_source_work_chi2": refit_fit.get("source_work_window_metrics", {}).get("chi2"),
                "refit_source_work_reduced_chi2": refit_fit.get("source_work_window_metrics", {}).get("reduced_chi2"),
                "refit_source_work_fit_rms": refit_fit.get("source_work_window_metrics", {}).get("fit_rms"),
                "refit_n_components": refit_fit.get("n_components_fitted"),
            }
        )
        candidate_score = _candidate_score(evaluation)
        round_summary["candidate_score"] = json.dumps(list(candidate_score), ensure_ascii=False)
        if evaluation.get("decision") == "rejected":
            stop_reason = "refit_worsened_followup_best_candidate_selected" if final_record is not record else "refit_worsened_followup"
            round_summaries.append(round_summary)
            last_rejected_refit = {
                "sample_id": refit_record.get("sample_id"),
                "round_index": round_index,
                "review_json": str(refit_paths["json"]),
                "plot_png": str(refit_paths["plot_png"]),
                "overview_png": str(refit_paths["overview_png"]),
                "evaluation": evaluation,
                "patch": patch,
                "fit_summary": refit_fit,
            }
            last_refit_feedback = feedback
            trial_controls = candidate_controls
            if round_index == max(1, int(max_rounds)):
                break
            continue
        round_summaries.append(round_summary)
        if _is_better_candidate(candidate_score, best_score):
            for item in round_summaries:
                item["selected_candidate"] = False
            round_summary["selected_candidate"] = True
            best_score = candidate_score
            best_record = refit_record
            best_paths = refit_paths
        final_record = refit_record
        final_paths = refit_paths
        active_controls = applied_controls
        trial_controls = applied_controls
        state_record = refit_record
        state_paths = refit_paths
        last_rejected_refit = None
        last_refit_feedback = feedback
    else:
        stop_reason = "max_rounds_reached"

    if best_record is not None and best_paths is not None and best_record.get("sample_id") != final_record.get("sample_id"):
        final_record = best_record
        final_paths = best_paths
        stop_reason = f"{stop_reason}_best_candidate_selected"
        final_record["fit_control_selected_candidate"] = {
            "sample_id": final_record.get("sample_id"),
            "selection_policy": "minimize fit_rms with penalty for excessive fit-pixel removal",
            "score": list(best_score) if best_score is not None else None,
        }

    row["agent_rounds_json"] = json.dumps(round_summaries, ensure_ascii=False)
    row["agent_n_rounds"] = len(round_summaries)
    row["agent_stop_reason"] = stop_reason
    row["agent_total_tokens"] = total_tokens
    row["agent_prompt_tokens"] = sum(int(item.get("prompt_tokens") or 0) for item in round_summaries)
    row["agent_completion_tokens"] = sum(int(item.get("completion_tokens") or 0) for item in round_summaries)
    row["agent_model"] = round_summaries and control.get("_llm_metadata", {}).get("model")
    row["agent_n_tool_calls"] = sum(int(item.get("n_tool_calls") or 0) for item in round_summaries)
    row["agent_requires_refit"] = any(bool(item.get("requires_refit")) for item in round_summaries)
    row["agent_json"] = round_summaries[-1]["control_json"] if round_summaries else None
    row["agent_patch_json"] = round_summaries[-1]["patch_json"] if round_summaries else None
    row["final_review_json"] = str(final_paths["json"])
    row["final_plot_png"] = str(final_paths["plot_png"])
    final_fit = final_record.get("fit_results", [{}])[0]
    row["final_fit_quality"] = final_fit.get("quality")
    row["final_chi2"] = final_fit.get("chi2")
    row["final_reduced_chi2"] = final_fit.get("reduced_chi2")
    row["final_fit_rms"] = final_fit.get("fit_rms")
    row["final_source_work_chi2"] = final_fit.get("source_work_window_metrics", {}).get("chi2")
    row["final_source_work_reduced_chi2"] = final_fit.get("source_work_window_metrics", {}).get("reduced_chi2")
    row["final_source_work_fit_rms"] = final_fit.get("source_work_window_metrics", {}).get("fit_rms")
    row["final_n_components_fitted"] = final_fit.get("n_components_fitted")
    row["chi2_delta"] = numeric_delta(row.get("chi2"), row.get("final_chi2"))
    row["reduced_chi2_delta"] = numeric_delta(row.get("reduced_chi2"), row.get("final_reduced_chi2"))
    row["fit_rms_delta"] = numeric_delta(row.get("fit_rms"), row.get("final_fit_rms"))
    row["source_work_chi2_delta"] = numeric_delta(row.get("source_work_chi2"), row.get("final_source_work_chi2"))
    row["source_work_reduced_chi2_delta"] = numeric_delta(
        row.get("source_work_reduced_chi2"),
        row.get("final_source_work_reduced_chi2"),
    )
    row["source_work_fit_rms_delta"] = numeric_delta(row.get("source_work_fit_rms"), row.get("final_source_work_fit_rms"))
    row["intervention_effect"] = intervention_effect(row)


def infer_line_id(path: Path) -> tuple[str, str]:
    lowered = str(path).lower()
    if "mgii" in lowered:
        return "MGII", "MGII_doublet"
    if "civ" in lowered:
        return "CIV", "CIV_doublet"
    raise ValueError(f"cannot infer absorber family from path: {path}")


def infer_z_sys(path: Path) -> float:
    match = Z_PATTERN.search(path.name)
    if not match:
        raise ValueError(f"cannot infer z_sys from filename: {path.name}")
    return float(match.group("z").replace("p", "."))


def infer_targetid(path: Path) -> int | None:
    match = TARGET_PATTERN.search(path.name)
    return int(match.group("targetid")) if match else None


def status_line(row: dict[str, Any], *, index: int, total: int) -> str:
    if row["status"] == "written":
        token_text = ""
        if row.get("agent_total_tokens") is not None:
            token_text = f" tokens={row.get('agent_total_tokens')}"
        return (
            f"[{index}/{total}] ok {row['sample_id']} {row['line_id']} "
            f"quality={row.get('fit_quality')} comps={row.get('n_components_fitted')}{token_text}"
        )
    return f"[{index}/{total}] fail {row.get('sample_id')} {row.get('error_type')}: {row.get('error')}"


def build_summary(rows: list[dict[str, Any]], *, input_roots: list[Path], output_dir: Path) -> dict[str, Any]:
    successful = [row for row in rows if row.get("status") == "written"]
    failed = [row for row in rows if row.get("status") != "written"]
    agent_failed = [row for row in successful if row.get("agent_status") == "failed"]
    agent_rows = [row for row in rows if row.get("agent_json")]
    token_values = [int(row.get("agent_total_tokens") or 0) for row in agent_rows]
    effect_counts: dict[str, int] = {}
    for row in agent_rows:
        effect = str(row.get("intervention_effect") or "not_refit")
        effect_counts[effect] = effect_counts.get(effect, 0) + 1
    by_line: dict[str, dict[str, Any]] = {}
    for row in successful:
        key = str(row.get("line_id"))
        bucket = by_line.setdefault(key, {"n": 0, "n_agent_calls": 0, "total_tokens": 0})
        bucket["n"] += 1
        if row.get("agent_json"):
            bucket["n_agent_calls"] += int(row.get("agent_n_rounds") or 1)
            bucket["total_tokens"] += int(row.get("agent_total_tokens") or 0)
    for bucket in by_line.values():
        calls = int(bucket["n_agent_calls"])
        bucket["tokens_per_agent_call"] = float(bucket["total_tokens"] / calls) if calls else 0.0
    return {
        "input_roots": [str(path) for path in input_roots],
        "output_dir": str(output_dir),
        "n_inputs": len(rows),
        "n_success": len(successful),
        "n_failed": len(failed),
        "n_agent_failed": len(agent_failed),
        "by_line_id": by_line,
        "agent": {
            "n_calls": sum(int(row.get("agent_n_rounds") or 1) for row in agent_rows),
            "n_samples": len(agent_rows),
            "total_tokens": sum(token_values),
            "tokens_per_call": (
                float(sum(token_values) / sum(int(row.get("agent_n_rounds") or 1) for row in agent_rows))
                if token_values
                else 0.0
            ),
            "tokens_per_sample": float(sum(token_values) / len(token_values)) if token_values else 0.0,
            "max_tokens": max(token_values) if token_values else 0,
            "intervention_effect_counts": effect_counts,
        },
        "failures": failed,
        "agent_failures": agent_failed,
    }


def sample_output_dir(output_dir: Path, sample_id: str, *, layout: str) -> Path:
    return output_dir / "samples" / sample_id if layout == "structured" else output_dir


def agent_round_dir(sample_dir: Path, round_index: int, *, layout: str) -> Path:
    if layout == "structured":
        return sample_dir / "agent" / f"round_{round_index:02d}"
    return sample_dir


def write_sample_manifest(sample_dir: Path, row: dict[str, Any]) -> None:
    sample_dir.mkdir(parents=True, exist_ok=True)
    (sample_dir / "manifest.json").write_text(json.dumps(row, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def numeric_delta(before: Any, after: Any) -> float | None:
    try:
        return float(after) - float(before)
    except (TypeError, ValueError):
        return None


def intervention_effect(row: dict[str, Any]) -> str:
    if not row.get("agent_requires_refit"):
        return "no_refit"
    delta = row.get("source_work_fit_rms_delta")
    if delta is None:
        delta = row.get("fit_rms_delta")
    if delta is None:
        return "unknown"
    if float(delta) < -0.05:
        return "improved"
    if float(delta) > 0.05:
        return "worsened"
    return "neutral"


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
