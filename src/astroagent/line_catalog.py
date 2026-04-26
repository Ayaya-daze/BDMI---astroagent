from __future__ import annotations

import json
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CATALOG_PATH = PROJECT_ROOT / "configs" / "line_catalog.json"


def load_line_catalog(path: str | Path | None = None) -> dict[str, dict[str, Any]]:
    """Load line definitions from a small JSON catalog."""
    catalog_path = Path(path) if path is not None else DEFAULT_CATALOG_PATH
    with catalog_path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, dict):
        raise ValueError("line catalog must be a JSON object keyed by line_id")
    return data


def get_line_definition(
    line_id: str,
    catalog: dict[str, dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Return one line definition and fail loudly for unknown ids."""
    catalog_data = catalog if catalog is not None else load_line_catalog()
    try:
        return catalog_data[line_id]
    except KeyError as exc:
        known = ", ".join(sorted(catalog_data))
        raise KeyError(f"unknown line_id {line_id!r}; known line ids: {known}") from exc


def rest_wavelengths_A(line_id: str, catalog: dict[str, dict[str, Any]] | None = None) -> list[float]:
    """Return all rest wavelengths represented by a line_id."""
    definition = get_line_definition(line_id, catalog)
    if "rest_wavelengths_A" in definition:
        return [float(value) for value in definition["rest_wavelengths_A"]]
    if "rest_wavelength_A" in definition:
        return [float(definition["rest_wavelength_A"])]
    raise ValueError(f"line_id {line_id!r} has no rest wavelength field")


def primary_rest_wavelength_A(
    line_id: str,
    catalog: dict[str, dict[str, Any]] | None = None,
) -> float:
    """Return the wavelength used as the velocity-space zero point."""
    definition = get_line_definition(line_id, catalog)
    if "primary_rest_wavelength_A" in definition:
        return float(definition["primary_rest_wavelength_A"])
    return rest_wavelengths_A(line_id, catalog)[0]
