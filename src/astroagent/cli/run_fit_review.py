from __future__ import annotations

import argparse
import json
from pathlib import Path

from astroagent.agent.fit_control import build_fit_control_patch
from astroagent.agent.llm import OfflineReviewClient, OpenAICompatibleClient, run_fit_control, run_fit_review


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run an LLM pass over one review packet.")
    parser.add_argument("--review-json", type=Path, required=True, help="Path to a *.review.json file.")
    parser.add_argument("--output-json", type=Path, default=None, help="Output path for the LLM result JSON.")
    parser.add_argument("--patch-json", type=Path, default=None, help="Optional path for a normalized fit-control patch JSON.")
    parser.add_argument("--plot-image", type=Path, default=None, help="Optional review plot PNG to send as image input.")
    parser.add_argument(
        "--mode",
        choices=["fit_control", "fit_review"],
        default="fit_control",
        help="First-stage fitting control or second-stage review.",
    )
    parser.add_argument(
        "--client",
        choices=["offline", "openai-compatible"],
        default="offline",
        help="LLM client to use. 'offline' is deterministic and does not call a network provider.",
    )
    parser.add_argument("--temperature", type=float, default=0.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    record = json.loads(args.review_json.read_text(encoding="utf-8"))
    client = OfflineReviewClient() if args.client == "offline" else OpenAICompatibleClient()
    if args.mode == "fit_control":
        review = run_fit_control(record, client, temperature=args.temperature, plot_image_path=args.plot_image)
        patch = build_fit_control_patch(review)
    else:
        review = run_fit_review(record, client, temperature=args.temperature, plot_image_path=args.plot_image)
        patch = None

    output_json = args.output_json
    if output_json is None:
        suffix = ".llm_control.json" if args.mode == "fit_control" else ".llm_review.json"
        output_json = args.review_json.with_suffix("").with_suffix(suffix)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(review, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"llm {args.mode} json: {output_json}")

    if patch is not None:
        patch_json = args.patch_json
        if patch_json is None:
            patch_json = args.review_json.with_suffix("").with_suffix(".fit_control_patch.json")
        patch_json.parent.mkdir(parents=True, exist_ok=True)
        patch_json.write_text(json.dumps(patch, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(f"fit control patch json: {patch_json}")


if __name__ == "__main__":
    main()
