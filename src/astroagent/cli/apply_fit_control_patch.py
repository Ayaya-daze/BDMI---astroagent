from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from astroagent.fit_control_apply import refit_record_with_patch
from astroagent.review_packet import write_review_packet


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Apply one fit-control patch and rerun the review-packet fit.")
    parser.add_argument("--review-json", type=Path, required=True, help="Original *.review.json file.")
    parser.add_argument("--window-csv", type=Path, required=True, help="Original *.window.csv file.")
    parser.add_argument("--patch-json", type=Path, required=True, help="*.fit_control_patch.json file.")
    parser.add_argument("--sample-id", default=None, help="Optional output sample id. Defaults to '<old>_refit'.")
    parser.add_argument("--output-dir", type=Path, default=None, help="Output directory. Defaults beside review JSON.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    record = json.loads(args.review_json.read_text(encoding="utf-8"))
    patch = json.loads(args.patch_json.read_text(encoding="utf-8"))
    window = pd.read_csv(args.window_csv)
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


if __name__ == "__main__":
    main()
