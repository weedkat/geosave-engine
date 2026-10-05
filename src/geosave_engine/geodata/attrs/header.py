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
        """Qualify the keys a variable lost, None for the object's own attrs."""
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
            (merged header, attrs dropped from it — each naming the variable
            that held it)

        Raises:
            ValueError: `headers` is empty, the roots are of different scopes,
                one object has a name as a data variable while another has it
                as a coordinate, or a model refuses what the objects disagree
                on.
        """
        if not headers:
            raise ValueError("merging attrs needs at least one header")

        data_names = sorted(set().union(*(h.data_vars for h in headers)))
        coord_names = sorted(set().union(*(h.coords for h in headers)))
        if scope_conflicts := set(data_names) & set(coord_names):
            raise ValueError(
                f"{sorted(scope_conflicts)} hold variable attrs in one object and "
                f"coordinate attrs in another, so a join of them is neither"
            )

        header_variables = [h.variables for h in headers]
        # None stands for the root, which no variable can be called.
        namespaces_by_name = {None: [h.root for h in headers]} | {
            name: [variables[name] for variables in header_variables if name in variables]
            for name in (*data_names, *coord_names)
        }
        merged_namespaces: dict[str | None, AttrsNamespace] = {}
        dropped: set[DroppedAttr] = set()
        for name, namespaces in namespaces_by_name.items():
            try:
                merged_namespaces[name], dropped_keys = AttrsNamespace.merge(namespaces)
            except ValueError as error:
                error.add_note("in the objects' own attrs" if name is None else f"in {name!r}")
                raise
            dropped |= DroppedAttr.from_keys(name, dropped_keys)

        header = cls(
            root=merged_namespaces[None],
            data_vars={name: merged_namespaces[name] for name in data_names},
            coords={name: merged_namespaces[name] for name in coord_names},
        )
        return header, dropped
