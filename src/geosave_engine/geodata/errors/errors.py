"""Shared exception types for geodata source/pipeline fetch failures.

Lives at the ``geodata`` package root — both ``geodata.stac`` (raises) and
``geodata.pipeline`` (documents/catches) depend on it, and neither of those
should depend on the other just for an exception type.
"""


class AnchorFetchError(RuntimeError):
    """Raised when a source has no usable data for an anchor."""
