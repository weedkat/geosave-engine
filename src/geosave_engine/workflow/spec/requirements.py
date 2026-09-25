"""Requirements on native raster structure and registered metadata models."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Annotated, Self

import numpy as np
from pydantic import Field, HttpUrl, JsonValue, field_serializer, model_validator
import xarray as xr

from geosave_engine.geodata import attrs
from geosave_engine.geodata.attrs.model import REGISTERED_ATTR_KEYS

from .base import SpecModel, Text, unique


class FieldRequirement(SpecModel):
    """Require fields, exact values, or one of several accepted values.

    Registered values use the attrs field's parser, so requirements share its
    native types. Predicate consistency is checked when bound to a namespace.

    Args:
        required: Fields that must have non-null values.
        equals: Fields that must equal the supplied values.
        one_of: Fields whose values must belong to the supplied alternatives.
    """

    required: tuple[Text, ...] = ()
    equals: dict[Text, JsonValue] = Field(default_factory=dict)
    one_of: dict[Text, Annotated[tuple[JsonValue, ...], Field(min_length=1)]] = Field(
        default_factory=dict
    )

    @model_validator(mode="after")
    def _validate_required_fields(self) -> Self:
        unique(self.required, "required")
        return self

    @property
    def fields(self) -> set[str]:
        """Return every field constrained by these requirements."""
        return set(self.required) | self.equals.keys() | self.one_of.keys()

    def _parse_predicates(
        self, model: type[attrs.AttrsModel] | None
    ) -> tuple[dict[str, object], dict[str, tuple[object, ...]]]:
        """Read expected values through their owner without building a partial model."""
        if model is not None and (unknown := self.fields - model.model_fields.keys()):
            raise ValueError(
                f"Unknown fields for attrs model {model.NAME!r}: {sorted(unknown)}"
            )

        def parse(field: str, value: JsonValue) -> object:
            parsed = (
                value if model is None else attrs.parse_field_value(model, field, value)
            )
            if parsed is None:
                raise ValueError(f"Metadata requirement {field!r} must be non-null")
            return parsed

        equals = {field: parse(field, value) for field, value in self.equals.items()}
        one_of = {
            field: tuple(parse(field, value) for value in choices)
            for field, choices in self.one_of.items()
        }
        for field in equals.keys() & one_of.keys():
            if not any(
                attrs.attrs_equal(equals[field], value)
                for value in one_of[field]
            ):
                raise ValueError(f"Conflicting equals and one_of for {field!r}")
        return equals, one_of

    def validate_values(
        self,
        values: Mapping[str, object],
        *,
        where: str,
        model: type[attrs.AttrsModel] | None = None,
    ) -> None:
        """Check typed metadata fields without serializing them or reading pixels.

        Args:
            values: Native metadata fields, as read by the attrs module.
            where: Scope used in validation errors.
            model: Registered field owner, or None for foreign metadata.

        Raises:
            ValueError: A required field is absent or a value differs.
        """
        equals, one_of = self._parse_predicates(model)
        for field in sorted(self.fields):
            value = values.get(field)
            if value is None:
                raise ValueError(f"{where}.{field} is required")
            if field in equals and not attrs.attrs_equal(value, equals[field]):
                raise ValueError(f"{where}.{field} must equal {self.equals[field]!r}")
            if field in one_of and not any(
                attrs.attrs_equal(value, choice) for choice in one_of[field]
            ):
                raise ValueError(
                    f"{where}.{field} must be one of {self.one_of[field]!r}"
                )


class NamespaceRequirement(SpecModel):
    """Requirements on one root, variable, or coordinate attrs namespace.

    Args:
        models: Registered attrs model names mapped to field requirements.
        foreign: Requirements on keys that have no registered owner.
    """

    models: dict[Text, FieldRequirement] = Field(default_factory=dict)
    foreign: FieldRequirement = Field(default_factory=FieldRequirement)

    @model_validator(mode="after")
    def _validate_registered_predicates(self) -> Self:
        if collision := self.foreign.fields & REGISTERED_ATTR_KEYS:
            raise ValueError(f"Use registered models for attrs {sorted(collision)}")
        self.foreign._parse_predicates(None)
        for name, requirement in self.models.items():
            try:
                model = attrs.resolve_model(name)
            except KeyError as error:
                raise ValueError(str(error)) from error
            requirement._parse_predicates(model)
        return self

    def validate_namespace(
        self, namespace: attrs.AttrsNamespace, *, where: str
    ) -> None:
        """Check a namespace already parsed by `attrs.create_header`.

        Args:
            namespace: Native typed attrs namespace.
            where: Root, variable, or coordinate path for errors.

        Raises:
            ValueError: A required model or field is absent or incompatible.
        """
        for name, requirement in self.models.items():
            model = namespace.get(name)
            if model is None:
                raise ValueError(f"{where}.{name} metadata is required")
            requirement.validate_values(
                {field: getattr(model, field) for field in requirement.fields},
                model=type(model),
                where=f"{where}.{name}",
            )
        self.foreign.validate_values(namespace.foreign, where=f"{where}.foreign")


class AttrsRequirement(SpecModel):
    """Keep the root, data-variable, and coordinate metadata scopes distinct.

    Args:
        root: Metadata required on the Dataset itself.
        data_vars: Per-variable requirements; `*` applies to selected variables.
        coords: Per-coordinate requirements; `*` applies to all coordinates.
    """

    root: NamespaceRequirement = Field(default_factory=NamespaceRequirement)
    data_vars: dict[Text, NamespaceRequirement] = Field(default_factory=dict)
    coords: dict[Text, NamespaceRequirement] = Field(default_factory=dict)


class RasterRequirement(SpecModel):
    """Select and validate a raster using native variables or positional channels.

    Args:
        variables: Required data variables in their selected order.
        channels: First N channels; mutually exclusive with variables.
        collection: STAC collection ID, paired with endpoints for acquisition.
        endpoints: HTTP(S) STAC endpoints in fallback order; omitted for files.
        dims: Exact dimension order required for each selected variable, if set.
        dtypes: Accepted stored dtypes, if constrained.
        require_crs: Require a locatable geospatial grid.
        resolution: Positive pixel size, scalar or (x, y), in CRS units.
        attrs: Metadata requirements in the existing attrs vocabulary.
    """

    variables: Annotated[tuple[Text, ...], Field(min_length=1)] | None = None
    channels: Annotated[int, Field(gt=0)] | None = None
    collection: Text | None = None
    endpoints: Annotated[tuple[HttpUrl, ...], Field(min_length=1)] | None = None
    dims: Annotated[tuple[Text, ...], Field(min_length=1)] | None = None
    dtypes: Annotated[tuple[Text, ...], Field(min_length=1)] | None = None
    require_crs: bool = False
    resolution: (
        Annotated[float, Field(gt=0)]
        | tuple[Annotated[float, Field(gt=0)], Annotated[float, Field(gt=0)]]
        | None
    ) = None
    attrs: AttrsRequirement = Field(default_factory=AttrsRequirement)

    @field_serializer("endpoints")
    def _serialize_endpoints(
        self, endpoints: tuple[HttpUrl, ...] | None
    ) -> tuple[str, ...] | None:
        return None if endpoints is None else tuple(str(url) for url in endpoints)

    @model_validator(mode="after")
    def _validate_structure(self) -> Self:
        if (self.collection is None) != (self.endpoints is None):
            raise ValueError("collection and endpoints must be supplied together")
        if self.endpoints is not None:
            unique(tuple(str(endpoint) for endpoint in self.endpoints), "endpoints")
        if (self.variables is None) == (self.channels is None):
            raise ValueError("Exactly one of variables or channels is required")
        if self.variables is not None:
            unique(self.variables, "variables")
        if self.dims is not None:
            unique(self.dims, "dims")
        if self.dtypes is not None:
            for dtype in self.dtypes:
                try:
                    np.dtype(dtype)
                except TypeError as error:
                    raise ValueError(f"Unknown raster dtype {dtype!r}") from error
        unknown = self.attrs.data_vars.keys() - set(self.variables or ()) - {"*"}
        if unknown:
            raise ValueError(f"Metadata names unselected variables: {sorted(unknown)}")
        return self

    def validate_raster(self, raster: xr.Dataset) -> None:
        """Validate structure and metadata without computing or changing pixels.

        Args:
            raster: Native xarray Dataset, possibly backed by Dask.

        Raises:
            TypeError: The supplied value is not a Dataset.
            ValueError: Required variables, structure, or metadata are absent.
        """
        self.select_raster(raster)

    def select_raster(self, raster: xr.Dataset) -> xr.Dataset:
        """Return a selected lazy view after checking structure and metadata.

        Args:
            raster: Native xarray Dataset, possibly backed by Dask.

        Returns:
            Dataset with the declared variables or first N channels.

        Raises:
            TypeError: The supplied value is not a Dataset.
            ValueError: Selection is ambiguous or a requirement is unsatisfied.
        """
        if not isinstance(raster, xr.Dataset):
            raise TypeError("Raster requirements expect an xarray.Dataset")
        selected = (
            self._select_variables(raster)
            if self.variables is not None
            else self._select_channels(raster)
        )
        self._validate_grid(selected)
        self._validate_variables(selected)
        self._validate_attrs(selected)
        return selected

    def _select_variables(self, raster: xr.Dataset) -> xr.Dataset:
        """Select named data variables in their declared order."""
        if missing := set(self.variables) - raster.data_vars.keys():
            raise ValueError(f"Missing raster variables: {sorted(missing)}")
        return raster[list(self.variables)]

    def _select_channels(self, raster: xr.Dataset) -> xr.Dataset:
        """Select channels from ordinary variables or a single band variable."""
        names = list(raster.data_vars)
        if any("band" in variable.dims for variable in raster.data_vars.values()):
            if len(names) != 1:
                raise ValueError(
                    "Positional channels are ambiguous with multiple band variables"
                )
            count = raster.sizes["band"]
            if count < self.channels:
                raise ValueError(
                    f"Raster requires {self.channels} channels, got {count}"
                )
            return raster.isel(band=slice(self.channels))
        if len(names) < self.channels:
            raise ValueError(
                f"Raster requires {self.channels} channels, got {len(names)}"
            )
        return raster[names[: self.channels]]

    def _validate_variables(self, raster: xr.Dataset) -> None:
        """Check dimensions and stored dtypes of the selected variables."""
        dtypes = None if self.dtypes is None else tuple(map(np.dtype, self.dtypes))
        for name, variable in raster.data_vars.items():
            if self.dims is not None and variable.dims != self.dims:
                raise ValueError(
                    f"{name} requires dimensions {self.dims}, got {variable.dims}"
                )
            if dtypes is not None and variable.dtype not in dtypes:
                raise ValueError(
                    f"{name} requires dtype in {self.dtypes}, got {variable.dtype}"
                )

    def _validate_grid(self, raster: xr.Dataset) -> None:
        """Check that the raster has the required CRS and pixel size."""
        if self.require_crs and raster.odc.geobox is None:
            raise ValueError("Raster requires a locatable CRS and grid")
        if self.resolution is not None:
            box = raster.odc.geobox
            if box is None:
                raise ValueError("Raster resolution requires a locatable CRS and grid")
            expected = (
                (self.resolution, self.resolution)
                if isinstance(self.resolution, float)
                else self.resolution
            )
            actual = (abs(box.resolution.x), abs(box.resolution.y))
            if not np.allclose(actual, expected, rtol=1e-6, atol=0):
                raise ValueError(f"Raster requires resolution {expected}, got {actual}")

    def _validate_attrs(self, raster: xr.Dataset) -> None:
        """Check metadata on the raster, its bands and its coordinates."""
        header = attrs.create_header(raster)
        self.attrs.root.validate_namespace(header.root, where="root")
        for declarations, namespaces, scope in (
            (self.attrs.data_vars, header.data_vars, "data_vars"),
            (self.attrs.coords, header.coords, "coords"),
        ):
            for selector, requirement in declarations.items():
                targets = tuple(namespaces) if selector == "*" else (selector,)
                for name in targets:
                    if name not in namespaces:
                        raise ValueError(f"Missing {scope}.{name}")
                    requirement.validate_namespace(
                        namespaces[name], where=f"{scope}.{name}"
                    )
