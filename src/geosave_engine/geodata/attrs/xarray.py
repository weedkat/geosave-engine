"""Read typed attrs off an xarray object, and write them back."""

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING, Literal, cast, overload

import xarray as xr

from geosave_engine.geodata.errors import DroppedAttrsWarning

from .header import AttrsHeader
from .namespace import AttrsNamespace
from .model import AttrsModel, resolve_model
from .models import Legend

if TYPE_CHECKING:
    from collections.abc import Hashable, Iterable, Mapping, Sequence

type XarrayObject = xr.Dataset | xr.DataArray | xr.DataTree


def read(obj: XarrayObject) -> AttrsHeader:
    """Read every registered model an object carries, off every variable.

    A model appears where at least one of its keys is present. A DataTree node
    is read on its own, its children untouched.

    Args:
        obj: Dataset, DataArray, or DataTree to read.

    Returns:
        Header holding the models and foreign keys the object carries.

    Raises:
        ValidationError: A value does not satisfy the field that owns its key.

    Examples:
        >>> read(ds).data_vars["B04"].get(CFVariable).units
        '1'
    """
    coord_names = _coord_names(obj)
    var_names = _var_names(obj)
    source_variables = _variables(obj)
    variables = {
        name: AttrsNamespace.from_attrs(source_variables[name].attrs)
        for name in (*coord_names, *var_names)
    }
    return AttrsHeader(
        root=AttrsNamespace.from_attrs(obj.attrs),
        variables=variables,
        coord_names=coord_names,
        var_names=var_names,
    )


def flag_variables(obj: XarrayObject) -> tuple[str, ...]:
    """Name the variables whose values are class codes, not measurements.

    Dtype does not say: Sentinel-2 reflectance and a land cover map are both
    integers. Only a `Legend.class_map` marks the values as codes, which
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

    header = read(obj)
    # A DataArray carries its own models at the root, having no data variables.
    if isinstance(obj, xr.DataArray):
        legend = header.root.get(Legend)
        return (str(obj.name),) if legend is not None and legend.class_map else ()

    for name, namespace in header.data_vars.items():
        legend = namespace.get(Legend)
        if legend is not None and legend.class_map:
            flagged.append(name)
    return tuple(sorted(flagged))


@overload
def merge(
    objects: Sequence[XarrayObject], *, action: str | None = None
) -> AttrsHeader: ...


@overload
def merge(
    objects: Sequence[xr.Variable], *, action: str | None = None
) -> AttrsNamespace: ...


def merge(
    objects: Sequence[XarrayObject] | Sequence[xr.Variable],
    *,
    action: str | None = None,
) -> AttrsHeader | AttrsNamespace:
    """Work out the attrs a join of these objects, or Variables, should carry.

    Each model decides for itself what survives, through its own `merge`.
    Objects match variables and coordinates by name; Variables carry no
    names to match, so they all flatten into one namespace.

    Args:
        objects: The xarray objects being joined, or the Variables being
            flattened into one namespace. At least one either way.
        action: Word naming the operation, for the warning a drop raises.
            None picks a default fitting which of the two this call is.

    Returns:
        Header to `rebase` onto the joined xarray result, or the namespace a
        flattened group of Variables merges into.

    Raises:
        ValueError: `objects` is empty, or a model refused a value the objects
            disagreed on.

    Warns:
        DroppedAttrsWarning: A model or foreign attr dropped because the
            objects carried different values.

    Examples:
        >>> rebase(xr.concat(rasters, dim="time"), merge(rasters))
        >>> rebase(stacked, merge(variables, action="stacking"))
    """
    if not objects:
        raise ValueError("merging attrs needs at least one object")

    if isinstance(objects[0], xr.Variable):
        variables = cast("Sequence[xr.Variable]", objects)
        namespace, dropped_keys = AttrsNamespace.merge(
            [AttrsNamespace.from_attrs(variable.attrs) for variable in variables]
        )
        _warn_dropped(dropped_keys, action=action or "merging")
        return namespace

    xarray_objects = cast("Sequence[XarrayObject]", objects)
    header, dropped_attrs = AttrsHeader.merge([read(obj) for obj in xarray_objects])
    _warn_dropped(dropped_attrs, action=action or "joining")
    return header


def _warn_dropped(dropped: Iterable[object], *, action: str) -> None:
    """Warn that a merge could not keep every attr the objects carried.

    Args:
        dropped: Attr keys, or `DroppedAttr` instances, that did not survive.
        action: Word naming the operation, used in the warning message.

    Warns:
        DroppedAttrsWarning: `dropped` is non-empty.
    """
    named = sorted(str(item) for item in dropped)
    if not named:
        return
    warnings.warn(
        f"{action} drops attrs the objects carried differently: {named}",
        DroppedAttrsWarning,
        stacklevel=3,
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
    instead patch just `target`, leaving its other keys alone.

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
            ``legend={"class_map": {...}}``, or to None to drop that model.
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
        >>> rebase(stacked, merge(variables, action="stacking"), target=None)
    """
    if len(models) == 1 and isinstance(models[0], AttrsHeader):
        if target is not None:
            raise ValueError(
                "target is not valid alongside an AttrsHeader; it already "
                "names its own variables"
            )
        if model_kwargs:
            raise ValueError(
                "keyword models are not valid alongside an AttrsHeader; it "
                "already carries its own models"
            )
        return _rebase_header(obj, models[0], inplace=inplace)

    if len(models) == 1 and isinstance(models[0], AttrsNamespace):
        if model_kwargs:
            raise ValueError(
                "keyword models are not valid alongside an AttrsNamespace; "
                "pass its models directly instead"
            )
        return _rebase_namespace(obj, models[0], target=target, inplace=inplace)

    return _rebase_models(
        obj,
        cast("Sequence[AttrsModel]", models),
        target=target,
        inplace=inplace,
        model_kwargs=model_kwargs,
    )


def _rebase_header[T: XarrayObject](
    obj: T, header: AttrsHeader, *, inplace: bool
) -> T | None:
    """Replace an object's attrs with the ones a header holds, whole.

    Args:
        obj: Dataset, DataArray, or DataTree to write onto.
        header: Detached attrs snapshot to write.
        inplace: Write into `obj` rather than returning a new object.

    Returns:
        New object carrying the header, or None when `inplace` is set.

    Raises:
        KeyError: `header` names an unregistered model.
        TypeError: A header model does not match its registered name.
        ValueError: A foreign key collides with a registered attr key, or a
            header target is absent from `obj`.
    """
    result = obj if inplace else obj.copy(deep=False)  # type: ignore

    # Resolve every target before writing so an absent one leaves obj untouched.
    targets = [
        (_resolve_target(result, name), namespace)
        for name, namespace in header.variables.items()
    ]

    result.attrs = header.root.to_attrs()
    for resolved, namespace in targets:
        resolved.attrs = namespace.to_attrs()
    return None if inplace else result


def _rebase_namespace[T: XarrayObject](
    obj: T,
    namespace: AttrsNamespace,
    *,
    target: str | Sequence[str] | None,
    inplace: bool,
) -> T | None:
    """Patch a namespace's keys onto target, leaving its other keys alone.

    Args:
        obj: Dataset, DataArray, or DataTree to write onto.
        namespace: Models and foreign keys to patch onto `target`.
        target: Variable or coordinate name to patch, or several of them.
            None patches the object's own attrs.
        inplace: Write into `obj` rather than returning a new object.

    Returns:
        New object carrying the patch, or None when `inplace` is set.

    Raises:
        ValueError: A foreign key collides with a registered attr key, or
            `target` names neither a variable nor a coordinate of `obj`.
    """
    result = obj if inplace else obj.copy(deep=False)  # type: ignore

    target_names: Sequence[str | None] = (
        (target,) if target is None or isinstance(target, str) else target
    )
    patch = namespace.to_attrs()
    for target_name in target_names:
        resolved = _resolve_target(result, target_name)
        resolved.attrs = {**resolved.attrs, **patch}

    return None if inplace else result


def _rebase_models[T: XarrayObject](
    obj: T,
    models: Sequence[AttrsModel],
    *,
    target: str | Sequence[str] | None,
    inplace: bool,
    model_kwargs: Mapping[str, Mapping[str, object] | None],
) -> T | None:
    """Patch bare models onto target, removing a key where a field is None.

    Args:
        obj: Dataset, DataArray, or DataTree to write onto.
        models: Model instances to apply to `target`.
        target: Variable or coordinate name the models describe, or several
            of them. None patches the object's own attrs.
        inplace: Write into `obj` rather than returning a new object.
        model_kwargs: Model name mapped to its field values, or to None to
            drop that model.

    Returns:
        New object carrying the models, or None when `inplace` is set.

    Raises:
        KeyError: A keyword names no registered model.
        TypeError: A positional argument is not an `AttrsModel` instance.
        ValueError: `target` names neither a variable nor a coordinate of
            `obj`.
        ValidationError: A supplied value does not satisfy its field.
    """
    wrong_models = sorted(
        {type(model).__name__ for model in models if not isinstance(model, AttrsModel)}
    )
    if wrong_models:
        raise TypeError(f"rebase needs AttrsModel instances, got {wrong_models}")

    result = obj if inplace else obj.copy(deep=False)  # type: ignore

    models_to_apply = list(models)
    models_to_remove: list[type[AttrsModel]] = []
    for name, values in model_kwargs.items():
        model = resolve_model(name)
        if values is None:
            models_to_remove.append(model)
        else:
            models_to_apply.append(model(**values))
    # None means the object's own attrs, which no variable name can collide with.
    target_names: Sequence[str | None] = (
        (target,) if target is None or isinstance(target, str) else target
    )

    for target_name in target_names:
        resolved = _resolve_target(result, target_name)
        attrs = dict(resolved.attrs)
        for removed_model in models_to_remove:
            for attr_key in removed_model.attr_keys.values():
                attrs.pop(attr_key, None)
        for applied_model in models_to_apply:
            for key, value in applied_model.to_attrs().items():
                if value is None:
                    attrs.pop(key, None)
                else:
                    attrs[key] = value
        resolved.attrs = attrs

    return None if inplace else result


def _coord_names(obj: XarrayObject) -> frozenset[str]:
    """Return the coordinate names an object carries."""
    return frozenset(str(name) for name in obj.coords)


def _var_names(obj: XarrayObject) -> frozenset[str]:
    """Return the data-variable names an object carries, none for a DataArray."""
    if isinstance(obj, xr.DataArray):
        return frozenset()
    return frozenset(str(name) for name in obj.data_vars)


def _variables(obj: XarrayObject) -> Mapping[Hashable, xr.Variable]:
    """Return the Variables an object carries, building no DataArray per name."""
    if isinstance(obj, xr.DataArray):
        return obj.coords.variables
    return obj.variables


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
