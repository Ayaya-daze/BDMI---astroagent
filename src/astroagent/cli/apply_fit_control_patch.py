from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from astroagent.agent.fit_control import refit_record_with_patch
from astroagent.review.packet import write_review_packet


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Apply one fit-control patch and rerun the review-packet fit.")
    parser.add_argument("--review-json", type=Path, required=True, help="Original *.review.json file.")
    parser.add_argument("--window-csv", type=Path, default=None, help="Original *.window.csv file. Defaults beside review JSON.")
    parser.add_argument("--patch-json", type=Path, required=True, help="*.fit_control_patch.json file.")
    parser.add_argument("--sample-id", default=None, help="Optional output sample id. Defaults to '<old>_refit'.")
    parser.add_argument("--output-dir", type=Path, default=None, help="Output directory. Defaults beside review JSON.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    record = json.loads(args.review_json.read_text(encoding="utf-8"))
    patch = json.loads(args.patch_json.read_text(encoding="utf-8"))
    window_csv = args.window_csv or _default_window_csv(args.review_json, record)
    if window_csv is None:
        raise SystemExit("could not infer --window-csv; pass it explicitly")
    window = pd.read_csv(window_csv)
    refit_record, _ = refit_record_with_patch(record, window, patch, sample_id=args.sample_id)
    output_dir = args.output_dir or args.review_json.parent
    paths = write_review_packet(refit_record, window, output_dir)
    evaluation = refit_record.get("fit_control_evaluation", {})
    print(f"refit review json: {paths['json']}")
    print(f"refit plot csv:    {paths['plot_csv']}")
    print(f"refit model csv:   {paths['model_csv']}")
    print(f"refit plot png:    {paths['plot_png']}")
    if evaluation:
        print(f"fit-control decision: {evaluation.get('decision')}")


def _default_window_csv(review_json: Path, record: dict[str, object]) -> Path | None:
    sample_id = record.get("sample_id")
    if not sample_id:
        return None
    candidate = review_json.parent / f"{sample_id}.window.csv"
    return candidate if candidate.exists() else None


if __name__ == "__main__":
    main()
