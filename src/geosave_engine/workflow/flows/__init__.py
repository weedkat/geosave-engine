"""Deployable Prefect workflow entry points."""

from .ingest import ingest
from .preprocess import preprocess

__all__ = ["ingest", "preprocess"]
