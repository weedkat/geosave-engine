"""The attrs an xarray object carries, shaped like the object, detached."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Self

from .namespace import AttrsNamespace


@dataclass(frozen=True)
class DroppedAttr:
    """One attr key a join could not carry over.

    Args:
        variable: Variable that held it, or None for the object's own attrs.
        key: Flat attr key that did not survive the join.

    Examples:
        >>> str(DroppedAttr("B04", "processing_baseline"))
        'B04.processing_baseline'
        >>> str(DroppedAttr(None, "history"))
        'history'
    """

    variable: str | None
    key: str

    def __str__(self) -> str:
        """Return the key qualified by its variable, bare for the own attrs."""
        if self.variable is None:
            return self.key
        return f"{self.variable}.{self.key}"


@dataclass(frozen=True)
class AttrsHeader:
    """Every attrs mapping of one xarray object, detached from it.

    Args:
        root: Namespace for the object's own attrs.
        data_vars: Data variable names mapped to their namespaces.
        coords: Coordinate names mapped to their namespaces.

    Examples:
        >>> header = attrs.create_header(ds)
        >>> header.root.to_attrs()
        {'title': 'Sentinel-2 Level-2A', 'license': 'CC-BY-4.0'}
        >>> header.data_vars["B04"].to_attrs()
        {'units': '1', '_FillValue': 0, 'nodata': 0}
    """

    root: AttrsNamespace = field(default_factory=AttrsNamespace)
    data_vars: Mapping[str, AttrsNamespace] = field(
        default_factory=dict[str, AttrsNamespace]
    )
    coords: Mapping[str, AttrsNamespace] = field(
        default_factory=dict[str, AttrsNamespace]
    )

    @classmethod
    def from_attrs(
        cls,
        *,
        root: Mapping[Any, Any] | None = None,
        data_vars: Mapping[str, Mapping[Any, Any]] | None = None,
        coords: Mapping[str, Mapping[Any, Any]] | None = None,
    ) -> Self:
        """Parse one object's flat attrs mappings into the header they make.

        What `AttrsNamespace.from_attrs` does for one mapping, this does for
        the set of them an object carries, which is what a source's reader
        hands over once it has translated that source into attr keys.

        Args:
            root: The object's own attrs.
            data_vars: Data variable name mapped to that variable's attrs.
            coords: Coordinate name mapped to that coordinate's attrs.

        Returns:
            Header holding the models each mapping carries.

        Raises:
            ValidationError: A value does not satisfy the field owning its key.

        Examples:
            >>> header = AttrsHeader.from_attrs(data_vars={"B04": {"_FillValue": 0}})
            >>> header.data_vars["B04"].to_attrs()
            {'_FillValue': 0, 'nodata': 0}
        """
        return cls(
            root=AttrsNamespace.from_attrs(root or {}),
            data_vars={
                name: AttrsNamespace.from_attrs(attrs)
                for name, attrs in (data_vars or {}).items()
            },
            coords={
                name: AttrsNamespace.from_attrs(attrs)
                for name, attrs in (coords or {}).items()
            },
        )

    @property
    def variables(self) -> Mapping[str, AttrsNamespace]:
        """Return every namespace but the root's, data variables first."""
        return {**self.data_vars, **self.coords}

    @classmethod
    def merge(cls, headers: Sequence[AttrsHeader]) -> tuple[Self, set[DroppedAttr]]:
        """Merge the headers read off the objects being joined.

        The objects' own attrs merge together, and variables merge with their
        namesakes, from whichever objects carry them.

        Args:
            headers: Header read off each object being joined, in call order,
                at least one.

        Returns:
            (merged header, attrs dropped from it — each naming the variable
            that held it)

        Raises:
            ValueError: `headers` is empty, or one object calls a name a data
                variable while another calls it a coordinate.
        """
        if not headers:
            raise ValueError("merging attrs needs at least one header")

        root, dropped_keys = AttrsNamespace.merge([header.root for header in headers])
        dropped = {DroppedAttr(None, key) for key in dropped_keys}

        crossed = sorted(
            {name for header in headers for name in header.data_vars}
            & {name for header in headers for name in header.coords}
        )
        if crossed:
            raise ValueError(
                f"{crossed} are data variables of one object and coordinates "
                f"of another, so a join of them is neither"
            )

        data_vars, dropped_from_vars = _merge_namesakes(
            [header.data_vars for header in headers]
        )
        coords, dropped_from_coords = _merge_namesakes(
            [header.coords for header in headers]
        )
        return cls(root=root, data_vars=data_vars, coords=coords), (
            dropped | dropped_from_vars | dropped_from_coords
        )


def _merge_namesakes(
    groups: Sequence[Mapping[str, AttrsNamespace]],
) -> tuple[dict[str, AttrsNamespace], set[DroppedAttr]]:
    """Merge each name's namespace across the objects carrying that name.

    Args:
        groups: One object's namespaces per entry, keyed by variable name.

    Returns:
        (name mapped to its merged namespace, sorted by name; the attrs
        dropped, each naming the variable that held it)
    """
    merged: dict[str, AttrsNamespace] = {}
    dropped: set[DroppedAttr] = set()
    for name in sorted({name for group in groups for name in group}):
        carried = [group[name] for group in groups if name in group]
        merged[name], dropped_keys = AttrsNamespace.merge(carried)
        dropped.update(DroppedAttr(name, key) for key in dropped_keys)
    return merged, dropped
