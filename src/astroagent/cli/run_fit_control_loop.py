from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from astroagent.agent.loop import run_fit_control_loop
from astroagent.agent.llm import OfflineReviewClient, OpenAICompatibleClient


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a bounded LLM fit-control tool loop over one review packet.")
    parser.add_argument("--review-json", type=Path, required=True, help="Initial *.review.json file.")
    parser.add_argument("--window-csv", type=Path, default=None, help="Initial *.window.csv file. Defaults beside review JSON.")
    parser.add_argument("--plot-image", type=Path, default=None, help="Initial velocity-frame plot PNG. Defaults beside review JSON.")
    parser.add_argument("--overview-image", type=Path, default=None, help="Initial wavelength-space overview PNG. Defaults beside review JSON when present.")
    parser.add_argument("--output-dir", type=Path, default=None, help="Output directory. Defaults beside review JSON.")
    parser.add_argument("--max-rounds", type=int, default=2, help="Initial experiment budget. The agent may request more rounds.")
    parser.add_argument("--hard-max-rounds", type=int, default=6, help="Absolute cap for agent-requested experiment rounds.")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--force", action="store_true", help="Run fit-control even when the initial fit is already good.")
    parser.add_argument(
        "--client",
        choices=["offline", "openai-compatible"],
        default="offline",
        help="'offline' is deterministic and does not call a network provider.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    record = json.loads(args.review_json.read_text(encoding="utf-8"))
    window_csv = args.window_csv or _default_window_csv(args.review_json, record)
    if window_csv is None:
        raise SystemExit("could not infer --window-csv; pass it explicitly")
    window = pd.read_csv(window_csv)
    output_dir = args.output_dir or args.review_json.parent
    image_paths = _initial_image_paths(args.review_json, record, overview_image=args.overview_image, plot_image=args.plot_image)

    client = OfflineReviewClient() if args.client == "offline" else OpenAICompatibleClient()
    result = run_fit_control_loop(
        record=record,
        window=window,
        client=client,
        output_dir=output_dir,
        initial_plot_image_path=image_paths,
        max_rounds=args.max_rounds,
        hard_max_rounds=args.hard_max_rounds,
        temperature=args.temperature,
        force=args.force,
    )

    summary_path = Path(output_dir) / f"{record.get('sample_id', 'fit_control_loop')}.fit_control_loop.json"
    summary_path.write_text(json.dumps(result.final_record["fit_control_loop"], ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"fit-control loop summary: {summary_path}")
    print(f"stop reason:              {result.stop_reason}")
    if result.final_record.get("sample_id") != record.get("sample_id"):
        print(f"final sample id:          {result.final_record.get('sample_id')}")
    if result.final_image_paths:
        print("final image inputs:       " + ", ".join(str(path) for path in result.final_image_paths))
    if result.audit_paths:
        print(f"audit report markdown:    {result.audit_paths['audit_markdown']}")
        print(f"audit report json:        {result.audit_paths['audit_json']}")


def _initial_image_paths(
    review_json: Path,
    record: dict[str, object],
    *,
    overview_image: Path | None,
    plot_image: Path | None,
) -> list[Path] | None:
    sample_id = record.get("sample_id")
    if not sample_id:
        return None
    candidates = [
        overview_image or review_json.parent / f"{sample_id}.overview.png",
        plot_image or review_json.parent / f"{sample_id}.plot.png",
    ]
    paths = [path for path in candidates if path.exists()]
    return paths or None


def _default_window_csv(review_json: Path, record: dict[str, object]) -> Path | None:
    sample_id = record.get("sample_id")
    if not sample_id:
        return None
    candidate = review_json.parent / f"{sample_id}.window.csv"
    return candidate if candidate.exists() else None


if __name__ == "__main__":
    main()
