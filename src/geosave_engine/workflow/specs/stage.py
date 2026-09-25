"""Ordered call declarations for one model processing stage."""

from collections.abc import Collection, Mapping
from typing import Any

from pydantic import Field, RootModel

from .base import Name
from .call import CallSpec


class StageSpec(RootModel[dict[Name, CallSpec]], Mapping[str, CallSpec]):
    """Validate and expose an ordered mapping of named calls."""

    root: dict[Name, CallSpec] = Field(default_factory=dict)

    def __getitem__(self, name: str) -> CallSpec:
        return self.root[name]

    def __iter__(self) -> Any:
        return iter(self.root)

    def __len__(self) -> int:
        return len(self.root)

    @property
    def external_inputs(self) -> tuple[str, ...]:
        """Return external roots in first-reference order."""
        assigned: set[str] = set()
        external: dict[str, None] = {}
        for output, call in self.root.items():
            for reference in call.references:
                if reference.root not in assigned:
                    external.setdefault(reference.root, None)
            assigned.add(output)
        return tuple(external)

    def validate_inputs(self, names: Collection[str]) -> frozenset[str]:
        """Validate external and ordered references before execution."""
        supplied = set(names)
        declared = set(self.root)
        assigned: set[str] = set()
        required: set[str] = set()
        for output, call in self.root.items():
            for name in call.inputs:
                if name in assigned:
                    continue
                if name in supplied:
                    required.add(name)
                elif name in declared:
                    raise ValueError(f"Forward stage reference: {name!r}")
                else:
                    required.add(name)
            assigned.add(output)
        if missing := required - supplied:
            raise ValueError(f"Missing stage inputs: {sorted(missing)}")
        return frozenset(required)
