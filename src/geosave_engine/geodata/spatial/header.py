"""GeoHeader: the metadata a raster carries besides pixels, geobox and vector. See GeoHeader."""
from __future__ import annotations

import warnings
from collections.abc import Mapping
from dataclasses import InitVar, dataclass, field, replace
from datetime import datetime as dt
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, Literal

import orjson

from geosave_engine.geodata.extensions import ArraySpec, GeoExtension, Tags, TilingInfo, TimeSpan, TimeSpec
from geosave_engine.geodata.utils.datetime import DateRange

if TYPE_CHECKING:
    import xarray as xr

# Stores differ in one thing only: what an attr value may hold. Zarr takes JSON, netcdf and GDAL take strings.
AttrEncoding = Literal["json", "text"]


@dataclass(frozen=True)
class GeoHeader:
    """A collection of `GeoExtension`s — everything a raster carries but pixels, geobox and vector.

    Every namespace present, including the built-in ones (`tags`, `timespan`,
    `tiling`, `timespec`), lands on disk under its own top-level attr key.
    Read a namespace through its own convenience property (`.tags`,
    `.timespan`, `.tiling`, `.timespec`) rather than `.extensions` directly.

    Args:
        extensions: `{namespace: extension}` or `{namespace: field dict}`,
            e.g. `{"render": RenderHints(...)}` or `{"tags": {"source": "survey"}}`.
            A namespace nothing registered is dropped with a warning —
            import its module to keep it.
        data: Canonical pixel array to reconcile every namespace against,
            via its own `reconcile` hook. None skips that — for a header
            built standalone, with no array attached yet.

    Raises:
        ValueError: A field dict fails its own extension's validation.

    Examples:
        >>> header = GeoHeader({"tags": {"source": "survey"}})
        >>> header.tags
        {'source': 'survey'}

        >>> # attach + validate in one step, e.g. from `_SpatialArray.__post_init__`
        >>> header = GeoHeader(existing.extensions, data=tile.data)
    """

    extensions: Mapping[str, GeoExtension] = field(default_factory=dict)
    data: InitVar[xr.DataArray | None] = None

    def __post_init__(self, data: xr.DataArray | None) -> None:
        """Resolve each namespace to its own registered class, then reconcile it against `data` if given."""
        registry = GeoExtension.registry()
        resolved: dict[str, GeoExtension] = {}
        for namespace, value in self.extensions.items():
            extension_cls = registry.get(namespace)
            if extension_cls is None:
                warnings.warn(f"dropping unregistered extension namespace {namespace!r} — import its module to keep it")
                continue
            resolved[namespace] = value if isinstance(value, extension_cls) else extension_cls.decode(value)

        if data is not None:
            reconciled: dict[str, GeoExtension] = {}
            for namespace, extension in resolved.items():
                result = extension.reconcile(data)
                if result is not None:
                    reconciled[namespace] = result
            resolved = reconciled

        object.__setattr__(self, "extensions", MappingProxyType(resolved))

    def __getstate__(self) -> dict[str, Any]:
        """Need to preserve extensions attribute type

        Returns:
            `{"extensions": {namespace: extension}}` as a plain dict.
        """
        return {"extensions": dict(self.extensions)}

    def __setstate__(self, state: dict[str, Any]) -> None:
        """Restore the read-only view `__getstate__` had to unwrap.

        Namespaces were already resolved and checked before pickling, so
        this rebuilds the view directly rather than re-running `__post_init__`.

        Args:
            state: What `__getstate__` returned.
        """
        object.__setattr__(self, "extensions", MappingProxyType(state["extensions"]))

    def rebase(self, **extensions: GeoExtension | Mapping[str, Any] | None) -> GeoHeader:
        """New GeoHeader with the given namespaces changed.

        Args:
            **extensions: `namespace=value` for any registered extension,
                e.g. `render={"class_map": {0: "bg", 1: "palm"}}` or
                `render=RenderHints(...)`. A dict merges onto that
                namespace's current fields; a built instance replaces it
                whole; `None` drops the namespace.

        Returns:
            New GeoHeader, every other namespace untouched.

        Raises:
            ValueError: A field dict fails its extension's validation, or
                a namespace declares `SETTABLE = False` — it's set only by
                the specific operation that produces it, not by `rebase()`.
            UnknownExtensionError: A keyword names an unregistered namespace.
        """
        if not extensions:
            return self

        updated = dict(self.extensions)
        for namespace, value in extensions.items():
            if value is None:
                updated.pop(namespace, None)
                continue
            extension_cls = GeoExtension.lookup(namespace)
            if not extension_cls.SETTABLE:
                raise ValueError(
                    f"{namespace!r} is not settable through rebase() — {extension_cls.__name__} "
                    "records something only the operation that produces it may set"
                )
            if isinstance(value, extension_cls):
                updated[namespace] = value
                continue
            if isinstance(value, GeoExtension):
                raise TypeError(
                    f"namespace {namespace!r} needs {extension_cls.__name__}, got {type(value).__name__}"
                )
            if not isinstance(value, Mapping):
                raise TypeError(
                    f"namespace {namespace!r} needs {extension_cls.__name__} or a field mapping, "
                    f"got {type(value).__name__}"
                )

            # A field mapping is a partial update. Decode the merged fields so the
            # namespace's own parsing and schema both apply, as they do off a store.
            current = updated.get(namespace)
            merged = {**(current.model_dump() if current is not None else {}), **value}
            updated[namespace] = extension_cls.decode(merged)
        return replace(self, extensions=updated)

    @classmethod
    def combine(cls, *headers: GeoHeader) -> GeoHeader:
        """Fold several arrays' headers into one header, namespace by namespace.

        Each namespace is handed to its own `GeoExtension.combine`, with
        one value per input. Missing namespaces arrive as None so an
        extension cannot mistake partial metadata for complete metadata.

        Args:
            *headers: Headers of the arrays being composed, in composition order.

        Returns:
            Combined header. Namespaces whose hook returns None are absent.

        Raises:
            ValueError: A namespace's values disagree and it declares no merge rule.
        """
        namespaces: list[str] = []
        for header in headers:
            for namespace in header.extensions:
                if namespace not in namespaces:
                    namespaces.append(namespace)

        combined: dict[str, GeoExtension] = {}
        for namespace in namespaces:
            values = [header.extensions.get(namespace) for header in headers]
            extension_cls = GeoExtension.lookup(namespace)
            merged = extension_cls.combine(values)
            if merged is not None:
                combined[namespace] = merged
        return cls(combined)

    @property
    def tags(self) -> dict[str, str]:
        """Free-form descriptive strings this carries. Empty if none.

        Returns:
            `{key: value}`, every value a string.
        """
        ext = self.extensions.get(Tags.NAMESPACE)
        return ext.model_dump() if isinstance(ext, Tags) else {}

    @property
    def array(self) -> ArraySpec | None:
        """Bands, timestamps and fill value of the array this describes.

        Returns:
            The spec, or None for a header no pixels were ever attached to —
            a plain window spec, which has no bands yet.
        """
        ext = self.extensions.get(ArraySpec.NAMESPACE)
        return ext if isinstance(ext, ArraySpec) else None

    @property
    def tiling(self) -> TilingInfo | None:
        """Where a tile sits in its group's grid, if this carries one.

        Returns:
            The stamp, or None if this wasn't cut by `GeoRaster.tiles()`.
        """
        ext = self.extensions.get(TilingInfo.NAMESPACE)
        return ext if isinstance(ext, TilingInfo) else None

    @property
    def timespec(self) -> TimeSpec | None:
        """How `data.time` was bucketed, if this records one.

        Returns:
            The bucketing spec, or None if never resampled.
        """
        ext = self.extensions.get(TimeSpec.NAMESPACE)
        return ext if isinstance(ext, TimeSpec) else None

    @property
    def timespan(self) -> DateRange | None:
        """Declared `(start, end)` window, if this carries one.

        Returns:
            Inclusive `(start, end)`, or None for a timeless header.
        """
        ext = self.extensions.get(TimeSpan.NAMESPACE)
        if not isinstance(ext, TimeSpan) or ext.start_datetime is None or ext.end_datetime is None:
            return None
        return ext.start_datetime, ext.end_datetime

    @property
    def start(self) -> dt | None:
        """Declared window's start. None for a timeless header.

        Returns:
            Range start.
        """
        span = self.timespan
        return None if span is None else span[0]

    @property
    def end(self) -> dt | None:
        """Declared window's end. None for a timeless header.

        Returns:
            Range end.
        """
        span = self.timespan
        return None if span is None else span[1]


def encode_attrs(attrs: Mapping[str, Any], header: GeoHeader, encoding: AttrEncoding) -> dict[str, Any]:
    """Encode a header into attrs one store can hold, one key per namespace.

    Args:
        attrs: Array attrs to encode alongside — this library's own keys in
            `attrs` are ignored; `header` is authoritative.
        header: Header to encode.
        encoding: `"json"` keeps each namespace a nested dict; `"text"`
            stores each as one JSON string.

    Returns:
        `attrs`' foreign keys, plus one key per non-empty namespace in `header`.

    Raises:
        ValueError: a namespace's value fails its own field's validation.
    """
    encoded = _foreign(attrs)
    for namespace, extension in header.extensions.items():
        dumped = extension.encode()
        if dumped is None:
            continue
        encoded[namespace] = orjson.dumps(dumped).decode() if encoding == "text" else dumped
    return encoded


def decode_attrs(attrs: Mapping[str, Any]) -> tuple[dict[str, Any], GeoHeader]:
    """Decode every registered namespace out of raw attrs into a GeoHeader.

    Args:
        attrs: Array attrs holding either form `encode_attrs` writes.

    Returns:
        `(foreign, header)` — attrs outside the registry, and a GeoHeader
        built from every registered namespace present.

    Raises:
        ValueError: a namespace's value fails its own field's validation.
    """
    registry = GeoExtension.registry()
    resolved = {key: value for key, value in attrs.items() if key in registry}
    return _foreign(attrs), GeoHeader(resolved)


def _foreign(attrs: Mapping[str, Any]) -> dict[str, Any]:
    """Everything in attrs that no registered extension owns.

    Args:
        attrs: A DataArray's or Dataset's own `.attrs`.

    Returns:
        Every key outside the registered-extension namespaces.
    """
    registry = GeoExtension.registry()
    return {key: value for key, value in attrs.items() if key not in registry}
