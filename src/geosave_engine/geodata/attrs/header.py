"""The attrs an xarray object carries, shaped like the object, detached."""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Self

from .models import Scope
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

    @classmethod
    def from_keys(cls, variable: str | None, keys: Iterable[str]) -> set[Self]:
        """Attach an attrs location to each dropped key.

        Args:
            variable: Data variable or coordinate name, None for root attrs.
            keys: Flat attr keys that were dropped.

        Returns:
            Set of loss records, one per key at the given location.

        Examples:
            >>> DroppedAttr.from_keys("red", ["long_name"])
            {DroppedAttr(variable='red', key='long_name')}
        """
        return {cls(variable, key) for key in keys}

    def __str__(self) -> str:
        """Return the key qualified by its variable, bare for the own attrs."""
        if self.variable is None:
            return self.key
        return f"{self.variable}.{self.key}"


@dataclass(frozen=True)
class AttrsHeader:
    """The attrs mappings of one xarray object it describes, detached from it.

    Args:
        root: Namespace for the object's own attrs: dataset attrs for a
            Dataset or DataTree, variable attrs for a DataArray. Empty when
            this header says nothing about them.
        data_vars: Data variable names mapped to their variable attrs.
        coords: Coordinate names mapped to their coordinate attrs.

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
        root_scope: Scope = "dataset",
        data_vars: Mapping[str, Mapping[Any, Any]] | None = None,
        coords: Mapping[str, Mapping[Any, Any]] | None = None,
    ) -> Self:
        """Parse one object's flat attrs mappings into the header they make.

        What `AttrsNamespace.from_attrs` does for one mapping, this does for
        the set of them an object carries, which is what a source's reader
        hands over once it has translated that source into attr keys.

        Args:
            root: The object's own attrs. None says nothing about them.
            root_scope: `dataset` for a Dataset or DataTree, `variable` for a
                DataArray.
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
            root=AttrsNamespace.from_attrs(root or {}, root_scope),
            data_vars={
                name: AttrsNamespace.from_attrs(attrs, "variable")
                for name, attrs in (data_vars or {}).items()
            },
            coords={
                name: AttrsNamespace.from_attrs(attrs, "coordinate")
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

        The objects' own attrs merge together, and variables merge with the
        variables of the same name, from whichever objects have them.

        Args:
            headers: Header read off each object being joined, in call order,
                at least one.

        Returns:
            Pair `(header, dropped)`. The header holds merged root, data
            variable and coordinate namespaces. The set `dropped` identifies
            each lost key and its location, with None identifying root attrs.

        Raises:
            ValueError: `headers` is empty, the roots are of different scopes,
                one object has a name as a data variable while another has it
                as a coordinate, or a model refuses what the objects disagree
                on.

        Examples:
            >>> first = AttrsHeader.from_attrs(root={"title": "first"})
            >>> second = AttrsHeader.from_attrs(root={"title": "second"})
            >>> header, dropped = AttrsHeader.merge([first, second])
            >>> header.root.to_attrs()
            {}
            >>> dropped
            {DroppedAttr(variable=None, key='title')}
        """
        if not headers:
            raise ValueError("merging attrs needs at least one header")

        roots: list[AttrsNamespace] = []
        data_mappings: list[Mapping[str, AttrsNamespace]] = []
        coord_mappings: list[Mapping[str, AttrsNamespace]] = []
        data_names: set[str] = set()
        coord_names: set[str] = set()
        for header in headers:
            roots.append(header.root)
            data_mappings.append(header.data_vars)
            coord_mappings.append(header.coords)
            data_names.update(header.data_vars)
            coord_names.update(header.coords)

        scope_conflicts = data_names & coord_names
        if scope_conflicts:
            raise ValueError(
                f"{sorted(scope_conflicts)} hold variable attrs in one object and "
                f"coordinate attrs in another, so a join of them is neither"
            )

        root, dropped = _merge_namespace(roots)
        data_vars, data_dropped = _merge_variables(data_mappings)
        coords, coord_dropped = _merge_variables(coord_mappings)
        dropped.update(data_dropped)
        dropped.update(coord_dropped)

        header = cls(root=root, data_vars=data_vars, coords=coords)
        return header, dropped


def _merge_namespace(
    namespaces: Sequence[AttrsNamespace], *, variable: str | None = None
) -> tuple[AttrsNamespace, set[DroppedAttr]]:
    """Merge one attrs location and identify the keys it loses.

    Args:
        namespaces: Root namespaces, or namespaces for one named variable or
            coordinate, from the headers being joined. At least one.
        variable: Name of that variable or coordinate, None for root attrs.

    Returns:
        Pair `(namespace, dropped)`. The namespace holds the surviving attrs;
        `dropped` is a set of `DroppedAttr(variable, key)` records.

    Raises:
        ValueError: Namespace merging fails. The error includes a note naming
            the attrs location that caused it.

    Examples:
        >>> first = AttrsNamespace.from_attrs({"source": "a"}, "variable")
        >>> second = AttrsNamespace.from_attrs({"source": "b"}, "variable")
        >>> namespace, dropped = _merge_namespace([first, second], variable="red")
        >>> namespace.to_attrs()
        {}
        >>> dropped
        {DroppedAttr(variable='red', key='source')}
    """
    try:
        namespace, dropped_keys = AttrsNamespace.merge(namespaces)
    except ValueError as error:
        if variable is None:
            error.add_note("in the objects' own attrs")
        else:
            error.add_note(f"in {variable!r}")
        raise
    return namespace, DroppedAttr.from_keys(variable, dropped_keys)


def _merge_variables(
    mappings: Sequence[Mapping[str, AttrsNamespace]],
) -> tuple[dict[str, AttrsNamespace], set[DroppedAttr]]:
    """Group data variable or coordinate namespaces by name, then merge them.

    A missing name contributes no namespace. Its attrs are compared only
    across the mappings where that variable or coordinate exists.

    Args:
        mappings: One name-to-namespace mapping per header, all describing
            data variables or all describing coordinates.

    Returns:
        Pair `(variables, dropped)`. The dictionary maps each name to its
        merged namespace, in alphabetical order. The set `dropped` identifies
        lost keys with their variable or coordinate names. Empty mappings
        return `({}, set())`.

    Raises:
        ValueError: Namespace merging fails for a name. The error includes a
            note naming the variable or coordinate that caused it.

    Examples:
        >>> red = AttrsNamespace.from_attrs({"units": "1"}, "variable")
        >>> nir = AttrsNamespace.from_attrs({"long_name": "NIR"}, "variable")
        >>> variables, dropped = _merge_variables([{"red": red}, {"nir": nir}])
        >>> list(variables)
        ['nir', 'red']
        >>> variables["nir"].to_attrs()
        {'long_name': 'NIR'}
        >>> dropped
        set()
    """
    grouped: dict[str, list[AttrsNamespace]] = {}
    for mapping in mappings:
        for name, namespace in mapping.items():
            if name not in grouped:
                grouped[name] = []
            grouped[name].append(namespace)

    variables: dict[str, AttrsNamespace] = {}
    dropped: set[DroppedAttr] = set()
    for name in sorted(grouped):
        namespace, lost = _merge_namespace(grouped[name], variable=name)
        variables[name] = namespace
        dropped.update(lost)
    return variables, dropped
