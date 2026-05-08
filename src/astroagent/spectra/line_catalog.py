from __future__ import annotations

import json
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[3]
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


def oscillator_strengths(line_id: str, catalog: dict[str, dict[str, Any]] | None = None) -> list[float]:
    """Return oscillator strengths aligned with rest_wavelengths_A."""
    definition = get_line_definition(line_id, catalog)
    if "oscillator_strengths" in definition:
        return [float(value) for value in definition["oscillator_strengths"]]
    if "oscillator_strength" in definition:
        return [float(definition["oscillator_strength"])]
    return [1.0 for _ in rest_wavelengths_A(line_id, catalog)]


def primary_rest_wavelength_A(
    line_id: str,
    catalog: dict[str, dict[str, Any]] | None = None,
) -> float:
    """Return the wavelength used as the velocity-space zero point."""
    definition = get_line_definition(line_id, catalog)
    if "primary_rest_wavelength_A" in definition:
        return float(definition["primary_rest_wavelength_A"])
    return rest_wavelengths_A(line_id, catalog)[0]


def transition_definitions(
    line_id: str,
    catalog: dict[str, dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    """Expand a line-family id into individual spectral transitions.

    A transition is one rest-frame spectral line.  Doublets and larger line
    families therefore return multiple entries; each entry should get its own
    local velocity frame downstream.
    """
    catalog_data = catalog if catalog is not None else load_line_catalog()
    definition = get_line_definition(line_id, catalog_data)
    transition_ids = definition.get("transition_line_ids")

    if transition_ids:
        transitions: list[dict[str, Any]] = []
        for transition_line_id in transition_ids:
            transition_definition = get_line_definition(str(transition_line_id), catalog_data)
            transitions.append(
                {
                    "transition_line_id": str(transition_line_id),
                    "family": transition_definition.get("family", definition.get("family", line_id)),
                    "rest_wavelength_A": float(transition_definition["rest_wavelength_A"]),
                    "oscillator_strength": float(transition_definition.get("oscillator_strength", 1.0)),
                    "damping_gamma_kms": float(transition_definition.get("damping_gamma_kms", 0.001)),
                    "atomic_label": transition_definition.get("atomic_label"),
                    "role": transition_definition.get("role", "transition"),
                }
            )
        return transitions

    rests = rest_wavelengths_A(line_id, catalog_data)
    strengths = oscillator_strengths(line_id, catalog_data)
    if len(rests) == 1:
        transition_ids = [line_id]
    else:
        transition_ids = [f"{line_id}_{index + 1}" for index in range(len(rests))]

    return [
        {
            "transition_line_id": str(transition_line_id),
            "family": definition.get("family", line_id),
            "rest_wavelength_A": float(rest),
            "oscillator_strength": float(strength),
            "damping_gamma_kms": float(definition.get("damping_gamma_kms", 0.001)),
            "atomic_label": definition.get("atomic_label"),
            "role": definition.get("role", "transition"),
        }
        for transition_line_id, rest, strength in zip(transition_ids, rests, strengths, strict=True)
    ]

