"""The attrs an xarray object carries, shaped like the object, detached."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Self

from .namespace import AttrsNamespace


@dataclass(frozen=True)
class DroppedAttr:
    """One foreign attr a join could not carry over.

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
    """Attrs namespaces read off one xarray object, detached from it.

    Variables key data variables and coordinates together the way xarray keys
    them, and the two name sets partition them, so `data_vars` and `coords`
    stay readable off a header detached from its object.

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
        >>> header.root.get("acdd").title
        'Sentinel-2 Level-2A'
        >>> header.data_vars["B04"].get("packing").fill_value
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
        """Return the namespaces of the variables that are not coordinates."""
        return {name: self.variables[name] for name in sorted(self.var_names)}

    @property
    def coords(self) -> Mapping[str, AttrsNamespace]:
        """Return the namespaces of the variables that are coordinates."""
        return {name: self.variables[name] for name in sorted(self.coord_names)}

    @classmethod
    def combine(cls, headers: Sequence[AttrsHeader]) -> tuple[Self, set[DroppedAttr]]:
        """Combine the headers read off the objects being joined.

        The objects' own attrs combine together, and variables combine with
        their namesakes, from whichever objects carry them. A variable stays a
        coordinate where any object called it one.

        Args:
            headers: Header read off each object being joined, in call order,
                at least one.

        Returns:
            Combined header and the foreign attrs dropped from it, each naming
            the variable that held it.

        Raises:
            ValueError: `headers` is empty, a registered model refuses a
                disagreement, or one object calls a name a data variable while
                another calls it a coordinate.
        """
        if not headers:
            raise ValueError("combining attrs needs at least one header")

        root, dropped_keys = AttrsNamespace.combine([header.root for header in headers])
        dropped = {DroppedAttr(None, key) for key in dropped_keys}

        var_names: set[str] = set()
        coord_names: set[str] = set()
        for header in headers:
            var_names.update(header.var_names)
            coord_names.update(header.coord_names)

        variables: dict[str, AttrsNamespace] = {}
        for name in sorted(var_names | coord_names):
            stated = [
                header.variables[name] for header in headers if name in header.variables
            ]
            variables[name], dropped_keys = AttrsNamespace.combine(stated)
            for attr_key in dropped_keys:
                dropped.add(DroppedAttr(name, attr_key))

        combined = cls(
            root=root,
            variables=variables,
            coord_names=frozenset(coord_names),
            var_names=frozenset(var_names),
        )
        return combined, dropped
