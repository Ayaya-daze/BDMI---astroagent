"""Quasar absorption review and fit-control utilities."""

from importlib import import_module
from types import ModuleType

__all__ = ["__version__", "agent", "data", "review", "spectra"]

__version__ = "0.1.0"


def __getattr__(name: str) -> ModuleType:
    if name in {"agent", "data", "review", "spectra"}:
        return import_module(f"{__name__}.{name}")
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
