"""Typed exceptions for the geodata ingestion pipeline."""
from __future__ import annotations

from enum import Enum


class IngestionErrorReason(Enum):
    NO_ITEMS_FOUND    = "no_items_found"
    RETRIES_EXHAUSTED = "retries_exhausted"
    METADATA_MISSING  = "metadata_missing"


class IngestionError(Exception):
    """Raised when the ingestion pipeline cannot produce a result.

    Callers should inspect ``.reason`` to decide whether to skip the item or
    re-raise, rather than catching the message string.
    """

    def __init__(self, reason: IngestionErrorReason, message: str) -> None:
        super().__init__(message)
        self.reason = reason
