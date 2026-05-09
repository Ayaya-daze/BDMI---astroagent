from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from astroagent.review.packet import (
    build_review_record,
    make_demo_quasar_spectrum,
    write_review_packet,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a minimal human-review packet for one quasar absorption window."
    )
    parser.add_argument("--input-csv", type=Path, default=None, help="CSV with wavelength, flux, ivar, pipeline_mask.")
    parser.add_argument("--line-id", default="CIV_doublet", help="Known line hypothesis, e.g. CIV_doublet.")
    parser.add_argument("--z-sys", type=float, default=2.6, help="Known absorber/system redshift.")
    parser.add_argument("--half-width-kms", type=float, default=1500.0, help="Window half width per line center.")
    parser.add_argument("--sample-id", default=None, help="Stable id for output filenames.")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/review_packet"))
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    if args.input_csv is None:
        spectrum = make_demo_quasar_spectrum(z_sys=args.z_sys, line_id=args.line_id)
        source = {"kind": "synthetic_demo", "path": None}
    else:
        spectrum = pd.read_csv(args.input_csv)
        source = {"kind": "csv", "path": str(args.input_csv)}

    sample_id = args.sample_id or f"demo_{args.line_id}_z{args.z_sys:.4f}".replace(".", "p")
    record, window = build_review_record(
        spectrum=spectrum,
        line_id=args.line_id,
        z_sys=args.z_sys,
        sample_id=sample_id,
        source=source,
        half_width_kms=args.half_width_kms,
    )
    paths = write_review_packet(record, window, args.output_dir)
    print(f"review json: {paths['json']}")
    print(f"window csv:  {paths['csv']}")
    print(f"plot csv:    {paths['plot_csv']}")
    print(f"model csv:   {paths['model_csv']}")
    print(f"plot png:    {paths['plot_png']}")
    print(f"readme:      {paths['readme']}")


if __name__ == "__main__":
    main()
