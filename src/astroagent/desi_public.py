from __future__ import annotations

import base64
import gzip
import html as html_lib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import fitsio
import numpy as np
import pandas as pd
import requests


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CACHE_DIR = PROJECT_ROOT / "data" / "external" / "desi_catalogs"
DEFAULT_VIEWER_DIR = PROJECT_ROOT / "data" / "interim" / "desi_true_positive"

DESI_VIEWER_URL_TEMPLATE = "https://www.legacysurvey.org/viewer/desi-spectrum/dr1/targetid{targetid}"

CATALOGS: dict[str, dict[str, Any]] = {
    "MGII": {
        "catalog_url": "https://data.desi.lbl.gov/public/dr1/vac/dr1/mgii-absorber/v1.0/MgII-Absorbers-DR1.fits",
        "extname": "MGII_ABSORBERS",
        "line_id": "MGII_doublet",
        "z_field": "Z_MGII",
        "score_fields": ["EW_2796", "EW_2803"],
    },
    "CIV": {
        "catalog_url": "https://data.desi.lbl.gov/public/dr1/vac/dr1/civ-absorber/CIV-Absorbers-dr1-v1.0.fits",
        "extname": "ABSORBER",
        "line_id": "CIV_doublet",
        "z_field": "Z_ABS",
        "score_fields": ["CIV_EW_TOTAL"],
    },
}


@dataclass(frozen=True)
class AbsorberChoice:
    family: str
    targetid: int
    z_sys: float
    catalog_row: dict[str, Any]
    score: float
    rank: int

    @property
    def line_id(self) -> str:
        return str(CATALOGS[self.family]["line_id"])

    @property
    def viewer_url(self) -> str:
        return DESI_VIEWER_URL_TEMPLATE.format(targetid=self.targetid)


def _decode_ndarray(payload: dict[str, Any]) -> np.ndarray:
    if "__ndarray__" in payload:
        data = base64.b64decode(payload["__ndarray__"])
        dtype = np.dtype(payload["dtype"])
        shape = payload["shape"]
    elif payload.get("type") == "ndarray" and isinstance(payload.get("array"), dict):
        raw = base64.b64decode(payload["array"]["data"])
        data = gzip.decompress(raw) if raw.startswith(b"\x1f\x8b") else raw
        byteorder = "<" if payload.get("order") == "little" else ">"
        dtype = np.dtype(payload["dtype"]).newbyteorder(byteorder)
        shape = payload["shape"]
    else:
        raise ValueError(f"unsupported ndarray payload keys: {sorted(payload)}")

    array = np.frombuffer(data, dtype=dtype).copy()
    return array.reshape(shape)


def _bokeh_map_to_dict(value: Any) -> Any:
    if isinstance(value, dict) and value.get("type") == "map" and "entries" in value:
        return {key: mapped_value for key, mapped_value in value["entries"]}
    return value


def _iter_bokeh_objects(root: Any):
    stack = [root]
    while stack:
        value = stack.pop()
        if isinstance(value, dict):
            yield value
            stack.extend(value.values())
        elif isinstance(value, list):
            stack.extend(value)


def download_file(url: str, destination: str | Path, timeout_s: int = 120) -> Path:
    destination_path = Path(destination)
    destination_path.parent.mkdir(parents=True, exist_ok=True)
    if destination_path.exists() and destination_path.stat().st_size > 0:
        return destination_path

    with requests.get(url, stream=True, timeout=timeout_s) as response:
        response.raise_for_status()
        with destination_path.open("wb") as handle:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    handle.write(chunk)
    return destination_path


def load_catalog(family: str, cache_dir: str | Path = DEFAULT_CACHE_DIR) -> tuple[np.ndarray, Path]:
    family_key = family.upper()
    if family_key not in CATALOGS:
        raise ValueError(f"unknown absorber family {family!r}; choose from {sorted(CATALOGS)}")

    catalog_info = CATALOGS[family_key]
    cache_path = Path(cache_dir) / f"{family_key.lower()}_absorber_catalog.fits"
    download_file(catalog_info["catalog_url"], cache_path)
    table = fitsio.read(cache_path, ext=catalog_info["extname"])
    return table, cache_path


def _row_to_dict(row: np.void) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for name in row.dtype.names or []:
        value = row[name]
        if isinstance(value, np.ndarray):
            result[name] = value.tolist()
        elif isinstance(value, (np.generic,)):
            result[name] = value.item()
        else:
            result[name] = value
    return result


def _score_rows(table: np.ndarray, family: str) -> np.ndarray:
    info = CATALOGS[family]
    fields = info["score_fields"]
    score = np.zeros(len(table), dtype=float)
    for field in fields:
        if field in table.dtype.names:
            score += np.nan_to_num(table[field].astype(float), nan=0.0)
    return score


def _mgii_unsaturated_score(table: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    names = table.dtype.names or ()
    required = {"EW_2796", "EW_2803", "LINE_SNR_MIN"}
    if not required <= set(names):
        raise ValueError(f"MGII unsaturated selection requires columns {sorted(required)}")

    ew_2796 = table["EW_2796"].astype(float)
    ew_2803 = table["EW_2803"].astype(float)
    snr_min = table["LINE_SNR_MIN"].astype(float)
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = ew_2796 / ew_2803

    valid = (
        np.isfinite(ew_2796)
        & np.isfinite(ew_2803)
        & np.isfinite(ratio)
        & np.isfinite(snr_min)
        & (ew_2796 >= 0.10)
        & (ew_2796 <= 0.90)
        & (ew_2803 > 0.0)
        & (ratio >= 1.55)
        & (ratio <= 2.35)
        & (snr_min >= 3.0)
    )
    ratio_closeness = np.abs(ratio - 2.0)
    ew_preference = np.abs(ew_2796 - 0.35)
    score = -(ratio_closeness + 0.35 * ew_preference) + 0.015 * np.nan_to_num(snr_min, nan=0.0)
    return score, valid


def choose_best_absorbers(
    table: np.ndarray,
    family: str,
    top_n: int = 25,
    selection: str = "strongest",
) -> list[AbsorberChoice]:
    family_key = family.upper()
    if family_key not in CATALOGS:
        raise ValueError(f"unknown absorber family {family!r}; choose from {sorted(CATALOGS)}")

    info = CATALOGS[family_key]
    z_field = info["z_field"]
    if "TARGETID" not in table.dtype.names or z_field not in table.dtype.names:
        raise ValueError(f"catalog missing required columns TARGETID/{z_field}")

    if selection == "strongest":
        score = _score_rows(table, family_key)
        valid = np.isfinite(score) & np.isfinite(table[z_field].astype(float))
    elif selection == "unsaturated":
        if family_key != "MGII":
            raise ValueError("unsaturated selection is currently implemented for MGII only")
        score, valid = _mgii_unsaturated_score(table)
        valid = valid & np.isfinite(table[z_field].astype(float))
    else:
        raise ValueError("selection must be 'strongest' or 'unsaturated'")

    if not np.any(valid):
        raise ValueError(f"no valid rows found in {family_key} absorber catalog for selection={selection!r}")

    order = np.argsort(score[valid])[::-1]
    valid_indices = np.flatnonzero(valid)
    choices: list[AbsorberChoice] = []
    for rank, relative_idx in enumerate(order[:top_n], start=1):
        row_index = int(valid_indices[relative_idx])
        row = table[row_index]
        choices.append(
            AbsorberChoice(
                family=family_key,
                targetid=int(row["TARGETID"]),
                z_sys=float(row[z_field]),
                catalog_row=_row_to_dict(row),
                score=float(score[row_index]),
                rank=rank,
            )
        )
    return choices


def fetch_viewer_html(targetid: int, cache_dir: str | Path = DEFAULT_VIEWER_DIR / "viewer_pages") -> Path:
    cache_path = Path(cache_dir) / f"targetid{targetid}.html"
    if cache_path.exists() and cache_path.stat().st_size > 0:
        return cache_path

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    url = DESI_VIEWER_URL_TEMPLATE.format(targetid=targetid)
    response = requests.get(url, timeout=120)
    response.raise_for_status()
    cache_path.write_text(response.text, encoding="utf-8")
    return cache_path


def parse_viewer_html(html_path: str | Path) -> pd.DataFrame:
    text = Path(html_path).read_text(encoding="utf-8")
    match = re.search(r'<script type="application/json" id="[^"]+">\s*(.*?)\s*</script>', text, re.S)
    if not match:
        raise ValueError(f"could not find embedded Bokeh JSON in {html_path}")

    raw = html_lib.unescape(match.group(1))
    document = json.loads(raw)

    frames: list[pd.DataFrame] = []
    for ref in _iter_bokeh_objects(document):
        if ref.get("type") != "ColumnDataSource" and ref.get("name") != "ColumnDataSource":
            continue
        attributes = ref.get("attributes", {})
        data = _bokeh_map_to_dict(attributes.get("data") or {})
        if not {"origwave", "origflux0", "orignoise0"} <= set(data):
            continue

        wave = _decode_ndarray(data["origwave"])
        flux = _decode_ndarray(data["origflux0"])
        noise = _decode_ndarray(data["orignoise0"])
        ivar = np.zeros(len(noise), dtype=float)
        np.divide(1.0, np.square(noise), out=ivar, where=noise > 0)
        band = str(attributes.get("name") or "")

        frame = pd.DataFrame(
            {
                "wavelength": wave.astype(float),
                "flux": flux.astype(float),
                "ivar": ivar,
                "pipeline_mask": np.zeros(len(wave), dtype=int),
                "band": band,
            }
        )
        frames.append(frame)

    if not frames:
        raise ValueError(f"no spectral data found in {html_path}")

    spectrum = pd.concat(frames, ignore_index=True)
    spectrum = spectrum.sort_values(["wavelength", "band"]).drop_duplicates(subset=["wavelength"], keep="first")
    spectrum = spectrum.reset_index(drop=True)
    spectrum["pipeline_mask"] = spectrum["pipeline_mask"].astype(int)
    return spectrum[["wavelength", "flux", "ivar", "pipeline_mask", "band"]]


def fetch_viewer_spectrum(
    targetid: int,
    cache_dir: str | Path = DEFAULT_VIEWER_DIR / "viewer_pages",
) -> tuple[pd.DataFrame, Path]:
    html_path = fetch_viewer_html(targetid, cache_dir=cache_dir)
    return parse_viewer_html(html_path), html_path


def catalog_row_summary(choice: AbsorberChoice) -> dict[str, Any]:
    return {
        "family": choice.family,
        "targetid": choice.targetid,
        "z_sys": choice.z_sys,
        "score": choice.score,
        "rank": choice.rank,
        "viewer_url": choice.viewer_url,
        "line_id": choice.line_id,
        "catalog_row": choice.catalog_row,
    }
