"""Inspect and modify typed attrs on xarray objects."""

from __future__ import annotations

import warnings
from copy import copy
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal, overload

import xarray as xr

from geosave_engine.geodata.errors import DroppedAttrsWarning

from .header import AttrsHeader
from .model import AttrsModel
from .models import Legend, resolve_model
from .namespace import AttrsNamespace

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

type XarrayObject = xr.Dataset | xr.DataArray | xr.DataTree


def create_header(obj: XarrayObject) -> AttrsHeader:
    """Create a detached header from an xarray object's attrs.

    A DataTree node is captured on its own, without traversing its children.

    Args:
        obj: Dataset, DataArray, or DataTree node carrying the attrs.

    Returns:
        Header holding the typed models and foreign attrs from every mapping.

    Raises:
        ValidationError: A value does not satisfy the field that owns its key.

    Examples:
        >>> create_header(ds).data_vars["B04"].to_attrs()
        {'units': '1', 'long_name': 'Red', '_FillValue': 0, 'nodata': 0}
    """
    coords = {str(name): coord.attrs for name, coord in obj.coords.items()}
    # A DataArray is one variable, so its own attrs are a variable's.
    if isinstance(obj, xr.DataArray):
        return AttrsHeader.from_attrs(root=obj.attrs, root_scope="variable", coords=coords)
    data_vars = {str(name): variable.attrs for name, variable in obj.data_vars.items()}
    return AttrsHeader.from_attrs(root=obj.attrs, data_vars=data_vars, coords=coords)


def flag_variables(obj: XarrayObject) -> tuple[str, ...]:
    """Name the variables whose values are class codes, not measurements.

    Dtype does not say: Sentinel-2 reflectance and a land cover map are both
    integers. Only a `Legend` listing marks the values as codes, which
    blending or ranking them would destroy.

    Args:
        obj: Dataset, DataArray, or DataTree to inspect. A DataTree is read
            through its whole tree, its names qualified by group.

    Returns:
        Names carrying a class map, sorted, empty when none do.

    Raises:
        ValidationError: A value does not satisfy the field that owns its key.

    Examples:
        >>> flag_variables(scene)
        ('landcover', 'scl')
    """
    if isinstance(obj, xr.DataTree):
        names = flag_variables(obj.dataset)
        for group, child in obj.children.items():
            names += tuple(f"{group}/{name}" for name in flag_variables(child))
        return tuple(sorted(names))

    # A DataArray is one variable, so its own attrs carry the listing.
    if isinstance(obj, xr.DataArray):
        return (str(obj.name),) if Legend.from_attrs(obj.attrs) is not None else ()

    return tuple(
        sorted(
            str(name)
            for name, variable in obj.data_vars.items()
            if Legend.from_attrs(variable.attrs) is not None
        )
    )


def merge(objects: Sequence[XarrayObject]) -> AttrsHeader:
    """Work out the attrs a join of these objects should carry.

    Objects match their variables and coordinates by name. An attr they carry
    differently describes no join of them, so it drops.

    Args:
        objects: The xarray objects being joined, at least one.

    Returns:
        Header to `rebase` onto the joined result.

    Raises:
        ValueError: `objects` is empty or its roots are of different scopes, or
            same-named variables carry a `MUST_AGREE` field differently.

    Warns:
        DroppedAttrsWarning: The objects carried an attr differently.

    Examples:
        >>> rebase(xr.concat(rasters, dim="time"), merge(rasters))
    """
    if not objects:
        raise ValueError("merging attrs needs at least one object")

    header, dropped = AttrsHeader.merge([create_header(obj) for obj in objects])
    if dropped:
        warnings.warn(
            f"joining drops attrs the objects carried differently: "
            f"{sorted(str(attr) for attr in dropped)}",
            DroppedAttrsWarning,
            stacklevel=2,
        )
    return header


@dataclass(frozen=True)
class AttrsEdit:
    """One namespace written onto one target's attrs.

    Args:
        target: Variable or coordinate name, None for the object's own attrs.
        namespace: Attrs to write; its models must belong to the target's scope.
        replace: Drop the target's existing attrs before writing.

    Examples:
        >>> AttrsEdit("B04", namespace).write(ds)
    """

    target: str | None
    namespace: AttrsNamespace
    replace: bool = False

    def write(self, obj: XarrayObject) -> None:
        """Write the namespace onto the target in `obj`.

        Keys the namespace marks missing are removed from the target.

        Args:
            obj: Dataset, DataArray, or DataTree to write onto.

        Raises:
            ValueError: The target is neither a variable nor a coordinate of
                `obj`, or the namespace belongs to another scope than the target.
        """
        data_vars = () if isinstance(obj, xr.DataArray) else obj.data_vars
        if self.target is None:
            target_obj = obj
            target_scope = "variable" if isinstance(obj, xr.DataArray) else "dataset"
        elif self.target in data_vars:
            target_obj, target_scope = obj[self.target], "variable"
        elif self.target in obj.coords:
            target_obj, target_scope = obj[self.target], "coordinate"
        else:
            names = sorted(str(name) for name in (*data_vars, *obj.coords))
            raise ValueError(
                f"{self.target!r} is neither a variable nor a coordinate of this "
                f"{type(obj).__name__}; it carries {names}"
            )
        if self.namespace.scope not in (None, target_scope):
            where = "the root" if self.target is None else repr(self.target)
            raise ValueError(
                f"{', '.join(self.namespace.models)} belongs on a {self.namespace.scope}; "
                f"{where} of this {type(obj).__name__} holds {target_scope} attrs"
            )

        attrs = {} if self.replace else dict(target_obj.attrs)
        for key in self.namespace.missing_keys:
            attrs.pop(key, None)
        target_obj.attrs = attrs | self.namespace.to_attrs()


@overload
def rebase[T: XarrayObject](
    obj: T,
    *models: AttrsModel | AttrsHeader | AttrsNamespace,
    target: str | Sequence[str] | None = None,
    inplace: Literal[False] = False,
    **model_kwargs: Mapping[str, object] | None,
) -> T: ...


@overload
def rebase(
    obj: XarrayObject,
    *models: AttrsModel | AttrsHeader | AttrsNamespace,
    target: str | Sequence[str] | None = None,
    inplace: Literal[True],
    **model_kwargs: Mapping[str, object] | None,
) -> None: ...


def rebase[T: XarrayObject](
    obj: T,
    *models: AttrsModel | AttrsHeader | AttrsNamespace,
    target: str | Sequence[str] | None = None,
    inplace: bool = False,
    **model_kwargs: Mapping[str, object] | None,
) -> T | None:
    """Write attrs onto an object, one of three ways depending what is given.

    A header replaces its non-empty root and every variable or coordinate it
    names; a namespace or bare models patch `target`. The whole request is
    validated before any attrs change, including inplace writes.

    Args:
        obj: Dataset, DataArray, or DataTree to write onto. A DataTree node
            is written on its own, its children untouched.
        *models: Exactly one `AttrsHeader` to restore its non-empty root and
            every variable or coordinate it names; exactly one
            `AttrsNamespace` to patch onto `target` — every key it carries,
            models' and foreign alike, overwrites `target`'s, but it cannot
            remove a stale key, since a flat mapping cannot tell "never set"
            from "explicitly cleared"; or any number of `AttrsModel`
            instances to patch onto `target`, applied in order, keyword
            models last, where a field set to None does remove its key.
        target: Variable or coordinate name the namespace or models describe,
            or several of them. None writes to the object's own attrs, which
            no name can address since a variable cannot be called None.
            Invalid alongside a header.
        inplace: Write into `obj` rather than returning a new object.
        **model_kwargs: Model name mapped to its field values, e.g.
            ``legend={"color_map": {...}}``, or to None to drop that model.
            Invalid alongside a header or a namespace.

    Returns:
        New object carrying the attrs, or None when `inplace` is set.

    Raises:
        KeyError: A keyword names no GeoSave attrs model.
        TypeError: A positional argument mixes an `AttrsModel` with a header
            or a namespace, or is not one of the three.
        ValueError: `target` or a keyword model accompanies a header, a
            keyword model accompanies a namespace, `target` names neither
            a variable nor a coordinate of `obj`, or a model, namespace, or
            header belongs to another scope than its target.
        ValidationError: A supplied value does not satisfy its field.

    Examples:
        >>> rebase(ds, Nodata(fill_value=0), target=ds.gs.variables)
        >>> rebase(ds, acdd={"title": "Sentinel-2 Level-2A"})
        >>> rebase(stretched, time_spec=None)  # the recorded bucketing no longer holds
        >>> rebase(xr.concat(rasters, dim="time"), merge(rasters))
    """
    targets = [target] if target is None or isinstance(target, str) else list(target)
    match models:
        case (AttrsHeader() as header,):
            if target is not None or model_kwargs:
                raise ValueError(
                    "target and keyword models are not valid alongside an AttrsHeader; "
                    "it already names its own variables and models"
                )
            edits = [
                AttrsEdit(name, namespace, replace=True)
                for name, namespace in header.variables.items()
            ]
            # An empty root says nothing, so a coordinates-only header leaves it alone.
            if header.root.models or header.root.foreign:
                edits.insert(0, AttrsEdit(None, header.root, replace=True))
        case (AttrsNamespace() as namespace,):
            if model_kwargs:
                raise ValueError(
                    "keyword models are not valid alongside an AttrsNamespace; "
                    "pass its models directly instead"
                )
            edits = [AttrsEdit(name, namespace) for name in targets]
        case _:
            attrs_models = [model for model in models if isinstance(model, AttrsModel)]
            if len(attrs_models) != len(models):
                raise TypeError(
                    f"rebase needs AttrsModel instances, got "
                    f"{sorted({type(model).__name__ for model in models})}"
                )
            for name, values in model_kwargs.items():
                model_type = resolve_model(name)
                attrs_models.append(
                    model_type.missing() if values is None else model_type(**values)
                )
            # One namespace per model, so models patch in order and the last wins per key.
            edits = [
                AttrsEdit(name, AttrsNamespace({model.NAME: model}))
                for model in attrs_models
                for name in targets
            ]

    # A failed write must leave `obj` untouched, even inplace, so try every edit on a copy first.
    result = copy(obj)
    for edit in edits:
        edit.write(result)
    if not inplace:
        return result
    for edit in edits:
        edit.write(obj)
    return None
