#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from astroagent.desi_public import (  # noqa: E402
    catalog_row_summary,
    choose_best_absorbers,
    fetch_viewer_spectrum,
    load_catalog,
)
from astroagent.review_packet import build_review_record, write_review_packet  # noqa: E402


def _component_summary(fit_result: dict) -> dict[str, str]:
    components = [component for component in fit_result.get("components", []) if component.get("fit_success")]
    centers = [component.get("center_velocity_kms") for component in components]
    logN_values = [component.get("logN") for component in components]
    b_values = [component.get("b_kms") for component in components]
    return {
        "component_centers_kms": "|".join(f"{float(value):.1f}" for value in centers if isinstance(value, (int, float))),
        "component_logN": "|".join(f"{float(value):.3f}" for value in logN_values if isinstance(value, (int, float))),
        "component_b_kms": "|".join(f"{float(value):.1f}" for value in b_values if isinstance(value, (int, float))),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a true-positive DESI absorber review packet.")
    parser.add_argument("--family", default="MGII", choices=["MGII", "CIV"], help="Absorber family to test.")
    parser.add_argument("--selection", default="strongest", choices=["strongest", "unsaturated"], help="Catalog row selection strategy.")
    parser.add_argument("--top-n", type=int, default=25, help="How many catalog rows to try.")
    parser.add_argument("--max-samples", type=int, default=1, help="How many plausible absorbers to write.")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/desi_true_positive"))
    parser.add_argument("--cache-dir", type=Path, default=Path("data/external/desi_catalogs"))
    parser.add_argument("--viewer-dir", type=Path, default=Path("data/interim/desi_true_positive"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    catalog, catalog_path = load_catalog(args.family, cache_dir=args.cache_dir)
    choices = choose_best_absorbers(catalog, args.family, top_n=args.top_n, selection=args.selection)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.viewer_dir.mkdir(parents=True, exist_ok=True)

    written_samples = []
    attempted_rows = []

    for choice in choices:
        if len(written_samples) >= args.max_samples:
            break

        try:
            spectrum, html_path = fetch_viewer_spectrum(choice.targetid, cache_dir=args.viewer_dir / "viewer_pages")
        except Exception as exc:
            attempted_rows.append(
                {
                    "rank": choice.rank,
                    "targetid": choice.targetid,
                    "z_sys": choice.z_sys,
                    "score": choice.score,
                    "status": "viewer_failed",
                    "reason": str(exc),
                }
            )
            print(f"skip TARGETID={choice.targetid}: viewer fetch/parse failed: {exc}")
            continue

        sample_id = f"desi_{choice.family.lower()}_target{choice.targetid}_z{choice.z_sys:.4f}".replace(".", "p")
        source = {
            "kind": "desi_viewer",
            "viewer_url": choice.viewer_url,
            "viewer_html": str(html_path),
            "catalog_path": str(catalog_path),
        }
        try:
            record, window = build_review_record(
                spectrum=spectrum,
                line_id=choice.line_id,
                z_sys=choice.z_sys,
                sample_id=sample_id,
                source=source,
            )
        except ValueError as exc:
            print(f"skip TARGETID={choice.targetid}: {exc}")
            attempted_rows.append(
                {
                    "rank": choice.rank,
                    "targetid": choice.targetid,
                    "z_sys": choice.z_sys,
                    "score": choice.score,
                    "sample_id": sample_id,
                    "status": "window_failed",
                    "reason": str(exc),
                }
            )
            continue

        absorber_status = record["absorber_hypothesis_check"]["status"]
        fit_result = record["fit_results"][0] if record.get("fit_results") else {}
        components = _component_summary(fit_result)
        attempt = {
            "rank": choice.rank,
            "targetid": choice.targetid,
            "z_sys": choice.z_sys,
            "score": choice.score,
            "selection": args.selection,
            "sample_id": sample_id,
            "absorber_status": absorber_status,
            "fit_type": fit_result.get("fit_type"),
            "fit_success": fit_result.get("success"),
            "fit_quality": fit_result.get("quality"),
            "fit_rms": fit_result.get("fit_rms"),
            "n_transition_frames": fit_result.get("n_transition_frames"),
            "n_components_fitted": fit_result.get("n_components_fitted"),
            "component_centers_kms": components["component_centers_kms"],
            "component_logN": components["component_logN"],
            "component_b_kms": components["component_b_kms"],
            "transition_half_width_kms": fit_result.get("transition_half_width_kms"),
            "agent_review_required": fit_result.get("agent_review", {}).get("required"),
            "agent_review_reasons": "|".join(fit_result.get("agent_review", {}).get("reasons", [])),
        }

        if absorber_status != "plausible":
            attempt["status"] = "not_plausible"
            attempted_rows.append(attempt)
            print(f"skip TARGETID={choice.targetid}: absorber status={absorber_status}")
            continue

        paths = write_review_packet(record, window, args.output_dir)
        spectrum_csv = args.viewer_dir / f"{record['sample_id']}.spectrum.csv"
        spectrum.to_csv(spectrum_csv, index=False)

        manifest = {
            "selection": catalog_row_summary(choice),
            "catalog_path": str(catalog_path),
            "viewer_html": str(html_path),
            "spectrum_csv": str(spectrum_csv),
            "outputs": {key: str(value) for key, value in paths.items()},
        }
        manifest_path = args.output_dir / f"{record['sample_id']}.manifest.json"
        manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

        sample_summary = {
            **attempt,
            "status": "written",
            "manifest": str(manifest_path),
            "review_json": str(paths["json"]),
            "window_csv": str(paths["csv"]),
            "plot_csv": str(paths["plot_csv"]),
            "model_csv": str(paths["model_csv"]),
            "plot_png": str(paths["plot_png"]),
            "spectrum_csv": str(spectrum_csv),
        }
        written_samples.append(sample_summary)
        attempted_rows.append(sample_summary)

        print(
            "written "
            f"{len(written_samples)}/{args.max_samples}: TARGETID={choice.targetid} "
            f"z={choice.z_sys:.4f} quality={fit_result.get('quality')} "
            f"components={fit_result.get('n_components_fitted')}"
        )

    if not written_samples:
        raise RuntimeError(f"no plausible absorber found in the top {args.top_n} rows of {args.family} catalog")

    summary_json_path = args.output_dir / f"batch_{args.family.lower()}_n{len(written_samples)}.summary.json"
    summary_csv_path = args.output_dir / f"batch_{args.family.lower()}_n{len(written_samples)}.summary.csv"
    summary = {
        "family": args.family,
        "selection": args.selection,
        "top_n": args.top_n,
        "max_samples": args.max_samples,
        "n_written": len(written_samples),
        "catalog_path": str(catalog_path),
        "written_samples": written_samples,
        "attempted_rows": attempted_rows,
    }
    summary_json_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    fieldnames = sorted({key for row in attempted_rows for key in row})
    with summary_csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(attempted_rows)

    print(f"summary json:      {summary_json_path}")
    print(f"summary csv:       {summary_csv_path}")
    print(f"written samples:   {len(written_samples)}")


if __name__ == "__main__":
    main()
