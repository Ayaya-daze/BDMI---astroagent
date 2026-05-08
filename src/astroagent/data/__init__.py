"""External data adapters and catalog helpers."""

from importlib import import_module
from typing import Any

__all__ = [
    "AbsorberChoice",
    "CATALOGS",
    "choose_best_absorbers",
    "fetch_viewer_spectrum",
    "load_catalog",
    "parse_viewer_html",
]

_EXPORTS = {
    "AbsorberChoice": "astroagent.data.desi_public",
    "CATALOGS": "astroagent.data.desi_public",
    "choose_best_absorbers": "astroagent.data.desi_public",
    "fetch_viewer_spectrum": "astroagent.data.desi_public",
    "load_catalog": "astroagent.data.desi_public",
    "parse_viewer_html": "astroagent.data.desi_public",
}


def __getattr__(name: str) -> Any:
    module_name = _EXPORTS.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(import_module(module_name), name)
    globals()[name] = value
    return value
