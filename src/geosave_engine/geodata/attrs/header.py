"""The attrs an xarray object carries, shaped like the object, detached."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Self

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

    `variables` holds data variables and coordinates together, the way xarray
    keys them; `var_names` and `coord_names` say which is which, so `data_vars`
    and `coords` stay readable once the header leaves its object.

    Args:
        root: Namespace for the object's own attrs.
        variables: Variable and coordinate names mapped to their namespaces.
        coord_names: Which of `variables` are coordinates.
        var_names: Which of `variables` are data variables.

    Raises:
        ValueError: The two name sets overlap, or together they do not name
            exactly the variables carried.

    Examples:
        >>> header = attrs.read(ds)
        >>> header.root.get(ACDD).title
        'Sentinel-2 Level-2A'
        >>> header.data_vars["B04"].get(Nodata).fill_value
        0
    """

    root: AttrsNamespace = field(default_factory=AttrsNamespace)
    variables: Mapping[str, AttrsNamespace] = field(
        default_factory=dict[str, AttrsNamespace]
    )
    coord_names: frozenset[str] = frozenset()
    var_names: frozenset[str] = frozenset()

    def __post_init__(self) -> None:
        """Refuse name sets that do not partition the variables carried.

        Raises:
            ValueError: A name is both a data variable and a coordinate, or the
                two name sets do not name exactly `variables`.
        """
        both = sorted(self.var_names & self.coord_names)
        if both:
            raise ValueError(f"{both} are named as data variables and coordinates")
        mismatched = sorted((self.var_names | self.coord_names) ^ self.variables.keys())
        if mismatched:
            raise ValueError(
                f"var_names and coord_names must together name every variable, "
                f"but {mismatched} are named without a namespace or the reverse"
            )

    @property
    def data_vars(self) -> Mapping[str, AttrsNamespace]:
        """Return the data variables' namespaces, keyed by name, sorted."""
        return {name: self.variables[name] for name in sorted(self.var_names)}

    @property
    def coords(self) -> Mapping[str, AttrsNamespace]:
        """Return the coordinates' namespaces, keyed by name, sorted."""
        return {name: self.variables[name] for name in sorted(self.coord_names)}

    @classmethod
    def merge(cls, headers: Sequence[AttrsHeader]) -> tuple[Self, set[DroppedAttr]]:
        """Merge the headers read off the objects being joined.

        The objects' own attrs merge together, and variables merge with
        their namesakes, from whichever objects carry them. A variable stays a
        coordinate where any object called it one.

        Args:
            headers: Header read off each object being joined, in call order,
                at least one.

        Returns:
            (merged header, attrs dropped from it — each naming the variable
            that held it)

        Raises:
            ValueError: `headers` is empty, a registered model refuses a
                disagreement, or one object calls a name a data variable while
                another calls it a coordinate.
        """
        if not headers:
            raise ValueError("merging attrs needs at least one header")

        root, dropped_keys = AttrsNamespace.merge([header.root for header in headers])
        dropped = {DroppedAttr(None, key) for key in dropped_keys}

        var_names: set[str] = set()
        coord_names: set[str] = set()
        for header in headers:
            var_names.update(header.var_names)
            coord_names.update(header.coord_names)

        variables: dict[str, AttrsNamespace] = {}
        for name in sorted(var_names | coord_names):
            namespaces = [
                header.variables[name] for header in headers if name in header.variables
            ]
            variables[name], dropped_keys = AttrsNamespace.merge(namespaces)
            for attr_key in dropped_keys:
                dropped.add(DroppedAttr(name, attr_key))

        merged = cls(
            root=root,
            variables=variables,
            coord_names=frozenset(coord_names),
            var_names=frozenset(var_names),
        )
        return merged, dropped
