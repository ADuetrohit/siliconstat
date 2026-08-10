"""REST API layer (FastAPI)."""

from __future__ import annotations

__all__ = ["app"]


def __getattr__(name: str):  # lazy import so `siliconstat` works without FastAPI
    if name == "app":
        from .main import app
        return app
    raise AttributeError(name)
