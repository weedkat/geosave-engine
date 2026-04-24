from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class BuildRequest:
    """All parameters required to scaffold a new workspace."""

    name: str
    task: str
    method: str
    description: str