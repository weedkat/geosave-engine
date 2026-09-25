"""Inspect and modify typed attrs on xarray objects."""

from __future__ import annotations

import warnings
from copy import deepcopy
from typing import TYPE_CHECKING, Literal, cast, overload

import xarray as xr

from geosave_engine.geodata.errors import DroppedAttrsWarning

from .header import AttrsHeader
from .headers.xarray import XarrayObject, create_header
from .namespace import AttrsNamespace
from .model import AttrsModel, attrs_equal, resolve_model
from .models import Legend

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence


# These fields change how stored pixels are decoded or interpreted. A joined
# variable cannot truthfully carry either claim when its inputs disagree.
_SEMANTICS = (
    ("nodata", "nodata"),
    ("scale_factor", "packing.scale_factor"),
    ("add_offset", "packing.add_offset"),
    ("standard_name", "cf.standard_name"),
    ("units", "cf.units"),
    ("cell_methods", "cf.cell_methods"),
    ("flag_values", "legend.flag_values"),
    ("flag_masks", "legend.flag_masks"),
    ("flag_meanings", "legend.flag_meanings"),
)

_MISSING = object()


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
    flagged: list[str] = []
    if isinstance(obj, xr.DataTree):
        flagged.extend(flag_variables(obj.dataset))
        for group, child in obj.children.items():
            flagged.extend(f"{group}/{name}" for name in flag_variables(child))
        return tuple(sorted(flagged))

    header = create_header(obj)
    # A DataArray carries its own models at the root, having no data variables.
    if isinstance(obj, xr.DataArray):
        return (str(obj.name),) if header.root.get(Legend) is not None else ()

    for name, namespace in header.data_vars.items():
        if namespace.get(Legend) is not None:
            flagged.append(name)
    return tuple(sorted(flagged))


def _target_names(target: str | Sequence[str] | None) -> Sequence[str | None]:
    """Read one target, several, or the default as a sequence of names.

    Args:
        target: Variable or coordinate name, several of them, or None for the
            object's own attrs.

    Returns:
        Names to write, where None stands for the object's own attrs.
    """
    if target is None or isinstance(target, str):
        return (target,)
    return target


def merge(objects: Sequence[XarrayObject]) -> AttrsHeader:
    """Work out the attrs a join of these objects should carry.

    Objects match their variables and coordinates by name. An attr they carry
    differently describes no join of them, so it drops.

    Args:
        objects: The xarray objects being joined, at least one.

    Returns:
        Header to `rebase` onto the joined result.

    Raises:
        ValueError: `objects` is empty or namesake variables disagree on how
            their pixels are decoded or interpreted.

    Warns:
        DroppedAttrsWarning: The objects carried an attr differently.

    Examples:
        >>> rebase(xr.concat(rasters, dim="time"), merge(rasters))
    """
    if not objects:
        raise ValueError("merging attrs needs at least one object")

    headers = [create_header(obj) for obj in objects]
    _check_semantics(objects, headers)
    header, dropped = AttrsHeader.merge(headers)
    if dropped:
        warnings.warn(
            f"joining drops attrs the objects carried differently: "
            f"{sorted(str(attr) for attr in dropped)}",
            DroppedAttrsWarning,
            stacklevel=2,
        )
    return header


def _check_semantics(
    objects: Sequence[XarrayObject], headers: Sequence[AttrsHeader]
) -> None:
    """Refuse pixel semantics that disagree across one joined variable."""
    variables: dict[str, list[AttrsNamespace]] = {}
    if all(isinstance(obj, xr.DataArray) for obj in objects):
        first = cast("xr.DataArray", objects[0])
        name = str(first.name) if first.name is not None else "<unnamed>"
        variables[name] = [header.root for header in headers]
    elif all(isinstance(obj, xr.Dataset) for obj in objects):
        for name in sorted({name for header in headers for name in header.data_vars}):
            variables[name] = [
                header.data_vars[name] for header in headers if name in header.data_vars
            ]

    for variable, namespaces in variables.items():
        mappings = [namespace.to_attrs() for namespace in namespaces]
        for key, semantic in _SEMANTICS:
            values = [mapping.get(key, _MISSING) for mapping in mappings]
            first, *others = values
            if any(
                (first is _MISSING) != (value is _MISSING)
                or (first is not _MISSING and not attrs_equal(first, value))
                for value in others
            ):
                raise ValueError(
                    f"{variable!r} carries incompatible {semantic} metadata across "
                    f"the objects; align or remove it before joining them"
                )


@overload
def rebase[T: XarrayObject](
    obj: T,
    *models: AttrsModel,
    target: str | Sequence[str] | None = None,
    inplace: Literal[False] = False,
    **model_kwargs: Mapping[str, object] | None,
) -> T: ...


@overload
def rebase(
    obj: XarrayObject,
    *models: AttrsModel,
    target: str | Sequence[str] | None = None,
    inplace: Literal[True],
    **model_kwargs: Mapping[str, object] | None,
) -> None: ...


@overload
def rebase[T: XarrayObject](
    obj: T, header: AttrsHeader, /, *, inplace: Literal[False] = False
) -> T: ...


@overload
def rebase(
    obj: XarrayObject, header: AttrsHeader, /, *, inplace: Literal[True]
) -> None: ...


@overload
def rebase[T: XarrayObject](
    obj: T,
    namespace: AttrsNamespace,
    /,
    *,
    target: str | Sequence[str] | None = None,
    inplace: Literal[False] = False,
) -> T: ...


@overload
def rebase(
    obj: XarrayObject,
    namespace: AttrsNamespace,
    /,
    *,
    target: str | Sequence[str] | None = None,
    inplace: Literal[True],
) -> None: ...


def rebase[T: XarrayObject](
    obj: T,
    *models: AttrsModel | AttrsHeader | AttrsNamespace,
    target: str | Sequence[str] | None = None,
    inplace: bool = False,
    **model_kwargs: Mapping[str, object] | None,
) -> T | None:
    """Write attrs onto an object, one of three ways depending what is given.

    A header replaces the object's attrs whole; a namespace or bare models
    instead patch just `target`, leaving its other keys alone. The complete
    request is validated before any attrs change, including inplace writes.

    Args:
        obj: Dataset, DataArray, or DataTree to write onto. A DataTree node
            is written on its own, its children untouched.
        *models: Exactly one `AttrsHeader` to restore whole, since it is
            already a full snapshot naming its own variables; exactly one
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
        KeyError: A keyword names no registered model.
        TypeError: A positional argument mixes an `AttrsModel` with a header
            or a namespace, or is not one of the three.
        ValueError: `target` or a keyword model accompanies a header, a
            keyword model accompanies a namespace, or `target` names neither
            a variable nor a coordinate of `obj`.
        ValidationError: A supplied value does not satisfy its field.

    Examples:
        >>> rebase(ds, Nodata(fill_value=0), target=ds.gs.variables)
        >>> rebase(ds, acdd={"title": "Sentinel-2 Level-2A"})
        >>> rebase(stretched, timespec=None)  # the recorded bucketing no longer holds
        >>> rebase(xr.concat(rasters, dim="time"), merge(rasters))
    """
    replacing = len(models) == 1 and isinstance(models[0], AttrsHeader)
    cleared: set[str] = set()
    updates: dict[str | None, dict[str, object]]
    if replacing:
        header = cast("AttrsHeader", models[0])
        if target is not None or model_kwargs:
            raise ValueError(
                "target and keyword models are not valid alongside an AttrsHeader; "
                "it already names its own variables and models"
            )
        updates = {None: header.root.to_attrs()}
        updates.update(
            (name, namespace.to_attrs()) for name, namespace in header.variables.items()
        )
    else:
        if len(models) == 1 and isinstance(models[0], AttrsNamespace):
            if model_kwargs:
                raise ValueError(
                    "keyword models are not valid alongside an AttrsNamespace; "
                    "pass its models directly instead"
                )
            patch = models[0].to_attrs()
            updates = dict.fromkeys(_target_names(target), patch)
        else:
            wrong = sorted(
                {
                    type(model).__name__
                    for model in models
                    if not isinstance(model, AttrsModel)
                }
            )
            if wrong:
                raise TypeError(f"rebase needs AttrsModel instances, got {wrong}")

            patch: dict[str, object] = {}
            for model in models:
                if isinstance(model, AttrsModel):
                    patch.update(model.to_attrs())
            for name, values in model_kwargs.items():
                model = resolve_model(name)
                if values is None:
                    patch.update(
                        {
                            key: None
                            for keys in model.field_keys.values()
                            for key in keys
                        }
                    )
                else:
                    patch.update(model(**values).to_attrs())

            cleared = {key for key, value in patch.items() if value is None}
            patch = {key: value for key, value in patch.items() if key not in cleared}
            # Each variable owns its mutable model values, even in a bulk edit.
            updates = {name: deepcopy(patch) for name in _target_names(target)}

    result = obj if inplace else cast("T", obj.copy(deep=False))
    prepared = []
    for name, patch in updates.items():
        resolved = _resolve_target(result, name)
        held = {} if replacing else dict(resolved.attrs)
        for key in cleared:
            held.pop(key, None)
        prepared.append((resolved, {**held, **patch}))

    # Serialization and target resolution must finish before any attrs change.
    for resolved, held in prepared:
        resolved.attrs = held
    return None if inplace else result


def _coord_names(obj: XarrayObject) -> frozenset[str]:
    """Return the coordinate names an object carries."""
    return frozenset(str(name) for name in obj.coords)


def _var_names(obj: XarrayObject) -> frozenset[str]:
    """Return the data-variable names an object carries, none for a DataArray."""
    if isinstance(obj, xr.DataArray):
        return frozenset()
    return frozenset(str(name) for name in obj.data_vars)


def _resolve_target(obj: XarrayObject, target: str | None) -> XarrayObject:
    """Return the xarray object holding `target`'s attrs, `obj` itself for None."""
    if target is None:
        return obj
    names = _coord_names(obj) | _var_names(obj)
    if target in names:
        return obj[target]
    raise ValueError(
        f"{target!r} is neither a variable nor a coordinate of this "
        f"{type(obj).__name__}; it carries {sorted(names)}"
    )
