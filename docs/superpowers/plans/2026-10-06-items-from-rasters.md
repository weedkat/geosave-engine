# Items From Rasters Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build STAC Items from the xarray object being written, so `item.from_paths` and its path guessing are deleted and `to_items` exists on a raster, a band and a stack.

**Architecture:** One asset builder takes a raster (`stac.asset.from_raster`), one assembler takes named assets (`stac.item.from_assets`). `io.cogs.layout` states the COG file layout once; the writer and `to_items` both walk it. `to_items` still writes through `to_cog` / `to_zarr` / `to_netcdf` and forwards their options.

**Tech Stack:** xarray, pystac 1.14, stac-geoparquet 0.8, odc-geo, pytest, ruff, ty.

**Spec:** `docs/superpowers/specs/2026-10-06-items-from-rasters-design.md`

## Global Constraints

- `driver` is the only format word in public signatures: `"cog"` (default), `"zarr"`, `"netcdf"`.
- No guessing from paths: no `commonpath`, no regrouping files by time span.
- Readable over compact: one step per statement, a named local instead of a nested comprehension or ternary, a one-line comment heading each block.
- No type suppressions (`cast` to silence, `# type: ignore`). Optional pystac fields are narrowed with explicit checks.
- Names say what a value holds; `validate`, never `check`; Google docstrings with a `{ "key": meaning, }` literal for dict returns.
- No compatibility aliases for anything removed.
- Do not commit. The working tree holds the user's uncommitted work; leave staging to them.
- Do not reformat lines you did not write: run `uv run ruff format --diff <file>` first and apply only if every hunk is yours.

## Review Focus

- A packed raster (`scale_factor` / `add_offset`): the asset built at write time must agree with the one read off the file on dtype, nodata and scale. Pinned in Task 1.
- `from_raster` must open no file: an href that does not exist still builds. Pinned in Task 1.
- A raster with a scalar `time` coordinate and no `time` dimension is one dated scene named by the path alone. Pinned in Task 2 and Task 3.
- `to_items()` on a raster read from a COG folder must raise and say why, not describe the wrong files. Pinned in Task 3.
- A stack whose groups differ in cadence (dated optical, timeless DEM): every Item carries the DEM. Pinned in Task 6.

---

### Task 1: Typed band fields and `asset.from_raster`

**Files:**
- Modify: `src/geosave_engine/geodata/attrs/headers/stac.py` (`band_fields`, ~line 203)
- Modify: `src/geosave_engine/geodata/stac/asset.py` (whole file body)
- Test: `tests/geodata/stac/test_asset.py`

**Interfaces:**
- Consumes: `RasterDriver` from `geosave_engine.geodata.core.raster` (`Literal["cog", "zarr", "netcdf"]`, already defined).
- Produces:
  - `band_fields(variable: xr.DataArray) -> BandFields`
  - `asset.from_raster(raster: xr.Dataset, href: str | PathLike[str], *, driver: RasterDriver) -> pystac.Asset`
  - `asset.from_path(path: str | PathLike[str]) -> pystac.Asset` (unchanged signature)
  - `asset.default_key(raster: xr.Dataset) -> str`

- [ ] **Step 1: Write the failing tests**

Append to `tests/geodata/stac/test_asset.py`:

```python
from geosave_engine.geodata.stac.asset import default_key, from_path, from_raster
from tests.geodata.conftest import build_raster

# What decides how a reader opens the pixels; `description` and a zero `offset`
# are the two fields a COG reads back differently.
_PIXEL_FIELDS = ("href", "type", "roles", "proj:code", "proj:shape", "proj:transform",
                 "start_datetime", "end_datetime")


def _band_facts(described: dict) -> list[tuple]:
    return [
        (raster.get("data_type"), raster.get("nodata"), raster.get("scale"), eo["name"])
        for raster, eo in zip(described["raster:bands"], described["eo:bands"])
    ]


@pytest.mark.parametrize("packed", [False, True])
def test_an_asset_built_at_write_time_agrees_with_the_file(tmp_path, packed) -> None:
    source = build_raster(times=2, packed=packed)
    scene = source.isel(time=0)[["red"]]
    (path, *_) = source.gs.to_cog(tmp_path / "forest", split_bands=True)
    store = source.gs.to_zarr(tmp_path / "forest.zarr")

    for built, read in [
        (from_raster(scene, path, driver="cog"), from_path(path)),
        (from_raster(source, store, driver="zarr"), from_path(store)),
    ]:
        built, read = built.to_dict(), read.to_dict()
        assert [built[key] for key in _PIXEL_FIELDS] == [read[key] for key in _PIXEL_FIELDS]
        assert _band_facts(built) == _band_facts(read)


def test_from_raster_opens_no_file(tmp_path) -> None:
    missing = tmp_path / "never-written.zarr"

    described = from_raster(build_raster(times=2), missing, driver="zarr")

    assert described.href == str(missing)
    assert described.media_type == "application/vnd+zarr"


def test_a_whole_raster_is_keyed_by_its_variable_or_as_image() -> None:
    raster = build_raster()

    assert default_key(raster[["red"]]) == "red"
    assert default_key(raster) == "image"
```

Add `import pytest` to the file's imports if it is absent.

- [ ] **Step 2: Run to verify they fail**

Run: `uv run pytest tests/geodata/stac/test_asset.py -q -p no:warnings`
Expected: collection error, `cannot import name 'default_key'`.

- [ ] **Step 3: Type `band_fields`**

In `src/geosave_engine/geodata/attrs/headers/stac.py`, add `Required, TypedDict` to the `typing` import, add the class above `band_fields`, and replace the function body:

```python
class BandFields(TypedDict, total=False):
    """One saved variable in the names STAC gives a band."""

    name: Required[str]
    data_type: Required[str]
    nodata: float | int
    unit: str
    description: str
    scale: float
    offset: float


def band_fields(variable: xr.DataArray) -> BandFields:
    """Describe one saved variable in the names STAC gives a band.

    Args:
        variable: Variable as its file stores it.

    Returns:
        {
            "name": variable name,
            "data_type": stored dtype name,
            "nodata": fill value, where the variable declares one,
            "unit" | "description" | "scale" | "offset": the attr `_ATTR_KEYS`
                maps to that field, where the variable carries it,
        }
    """
    attrs = variable.attrs
    fields: BandFields = {
        "name": str(variable.name),
        "data_type": variable.dtype.name,
    }

    # JSON holds native numbers, and a file hands back numpy ones.
    nodata = attrs.get("_FillValue")
    if nodata is not None:
        fields["nodata"] = nodata.item() if isinstance(nodata, np.generic) else nodata

    unit = attrs.get("units")
    if unit is not None:
        fields["unit"] = str(unit)
    description = attrs.get("long_name")
    if description is not None:
        fields["description"] = str(description)
    scale = attrs.get("scale_factor")
    if scale is not None:
        fields["scale"] = float(scale)
    offset = attrs.get("add_offset")
    if offset is not None:
        fields["offset"] = float(offset)
    return fields
```

- [ ] **Step 4: Rewrite `stac/asset.py` below its imports**

Replace `_MEDIA_TYPES` and `from_path` with the block below. Add `import xarray as xr` and `from datetime import UTC, datetime as DateTime`; remove `import pandas as pd`. Import `RasterDriver` under `TYPE_CHECKING` from `geosave_engine.geodata.core.raster`.

```python
# What STAC calls the file each driver writes.
_DRIVER_MEDIA_TYPES = {
    "cog": pystac.MediaType.COG,
    "zarr": pystac.MediaType.ZARR,
    "netcdf": pystac.MediaType.NETCDF,
}

# What STAC calls a file read from disk, by its suffix.
_SUFFIX_MEDIA_TYPES = {
    ".tif": pystac.MediaType.GEOTIFF,
    ".tiff": pystac.MediaType.GEOTIFF,
    ".zarr": pystac.MediaType.ZARR,
    ".nc": pystac.MediaType.NETCDF,
    ".nc4": pystac.MediaType.NETCDF,
    ".cdf": pystac.MediaType.NETCDF,
    ".jp2": pystac.MediaType.JPEG2000,
    ".png": pystac.MediaType.PNG,
}


def default_key(raster: xr.Dataset) -> str:
    """Name the asset holding a whole raster.

    Args:
        raster: Raster one file or store holds.

    Returns:
        The raster's variable where it has exactly one, else `"image"`.
    """
    variables = raster.gs.variables
    if len(variables) == 1:
        return variables[0]
    return "image"


def from_raster(
    raster: xr.Dataset, href: str | PathLike[str], *, driver: RasterDriver
) -> pystac.Asset:
    """Describe pixels saved at `href`, from the raster that holds them.

    No file is opened, so the raster has to be the one written there.

    Args:
        raster: Raster the file or store holds.
        href: Local path or URL the raster was saved at.
        driver: Format it was saved in.

    Returns:
        Asset with an absolute href, media type, the `data` role, its grid as
        projection fields, one raster and EO band per variable, and its first
        and last instant where the raster is dated.

    Raises:
        ValueError: The raster carries no locatable grid.

    Examples:
        >>> from_raster(scene, "samples/forest.zarr", driver="zarr").media_type
        'application/vnd+zarr'
    """
    return _describe(raster, absolute_location(href), _DRIVER_MEDIA_TYPES[driver])


def from_path(path: str | PathLike[str]) -> pystac.Asset:
    """Describe one saved raster file or store from its own header.

    Args:
        path: File or store `read_raster` opens.

    Returns:
        Asset as `from_raster` builds it, over what the file itself states.

    Raises:
        ValueError: `path` is a folder or another format than a raster file or
            store, or the file carries no locatable grid.

    Examples:
        >>> from_path("samples/forest/forest_20250601T103031.tif").media_type
        'image/tiff; application=geotiff; profile=cloud-optimized'
    """
    href = absolute_location(path)
    media_type = _SUFFIX_MEDIA_TYPES.get(PurePosixPath(href).suffix.lower())
    if media_type is None:
        raise ValueError(
            f"{path} is not one raster file or store, which an asset is; pass "
            f"each file a writer returned"
        )

    # A GeoTIFF says in its own structure tags whether it is cloud optimized.
    if media_type == pystac.MediaType.GEOTIFF:
        with rasterio.open(gdal_path(href)) as src:
            layout = src.tags(ns="IMAGE_STRUCTURE").get("LAYOUT")
        if layout == "COG":
            media_type = pystac.MediaType.COG

    with read_raster(path) as raster:
        return _describe(raster, href, media_type)


def _describe(raster: xr.Dataset, href: str, media_type: str) -> pystac.Asset:
    """Build the Asset stating a raster's grid, bands and time.

    Args:
        raster: Raster the file holds.
        href: Absolute path or URL of the file.
        media_type: STAC media type of the file.

    Returns:
        Asset carrying projection, raster and EO fields.

    Raises:
        ValueError: The raster carries no locatable grid.
    """
    geobox = raster.gs.geobox
    if not isinstance(geobox, GeoBox) or geobox.crs is None:
        raise ValueError(
            f"{href} carries no locatable grid, which a STAC asset states; "
            f"keep it in an ordinary reference table instead"
        )

    asset = pystac.Asset(href, media_type=media_type, roles=["data"])

    # Projection fields state the grid; a CRS without an EPSG code is spelled as WKT.
    projection = ProjectionExtension.ext(asset)
    if geobox.crs.epsg is None:
        projection.wkt2 = geobox.crs.to_wkt()
    else:
        projection.code = f"EPSG:{geobox.crs.epsg}"
    projection.shape = list(geobox.shape)
    projection.transform = list(geobox.transform)[:6]

    # Raster fields state how each band is stored; EO fields state what it is.
    raster_bands = []
    eo_bands = []
    for name in raster.gs.variables:
        fields = band_fields(raster[name])
        raster_band = RasterBand.create(
            data_type=DataType(fields["data_type"]),
            nodata=fields.get("nodata"),
            scale=fields.get("scale"),
            offset=fields.get("offset"),
            unit=fields.get("unit"),
        )
        raster_bands.append(raster_band)
        eo_band = Band.create(
            name=fields["name"], description=fields.get("description")
        )
        eo_bands.append(eo_band)
    RasterExtension.ext(asset).bands = raster_bands
    EOExtension.ext(asset).bands = eo_bands

    # A dated raster states the first and last instant it covers, in UTC.
    start: DateTime | None = None
    end: DateTime | None = None
    if TIME_COORDINATE in raster.dims:
        span = raster.gs.timespan
        if span is not None:
            start, end = span
    elif TIME_COORDINATE in raster.coords:
        # One scene holds one instant, as a scalar coordinate at microseconds.
        label = raster[TIME_COORDINATE].values.astype("datetime64[us]")
        start = label.item()
        end = start
    if start is not None and end is not None:
        asset.common_metadata.start_datetime = start.replace(tzinfo=UTC)
        asset.common_metadata.end_datetime = end.replace(tzinfo=UTC)
    return asset
```

- [ ] **Step 5: Run tests and type check**

Run: `uv run pytest tests/geodata/stac -q -p no:warnings`
Expected: all pass.
Run: `uvx ty check src/geosave_engine/geodata/stac/asset.py src/geosave_engine/geodata/attrs/headers/stac.py --output-format concise`
Expected: no diagnostic in `asset.py`, and none on `band_fields` or `BandFields`. If `label.item()` is reported as `Any`-typed assignment, annotate nothing further; `start: DateTime | None` already declares it.

---

### Task 2: `cogs.layout`, walked by the COG writer

**Files:**
- Modify: `src/geosave_engine/geodata/io/cogs.py`
- Test: `tests/geodata/io/test_cogs.py`

**Interfaces:**
- Produces:
  - `class CogFile(NamedTuple)`: `scene: str`, `path: str`, `raster: xr.Dataset`, `time: pd.Timestamp | None`
  - `cogs.layout(ds: xr.Dataset, path: str | PathLike[str], *, split_bands: bool = False) -> list[CogFile]`
  - `cogs.write(...)` keeps its signature and return value.

- [ ] **Step 1: Write the failing tests**

Append to `tests/geodata/io/test_cogs.py`:

```python
import pandas as pd

from geosave_engine.geodata.io import cogs


def test_layout_names_each_file_its_scene_and_its_pixels() -> None:
    source = build_raster(times=2)

    files = cogs.layout(source, "samples/forest", split_bands=True)

    assert [file.path for file in files] == [
        "samples/forest/forest_20250601T000000/red.tif",
        "samples/forest/forest_20250601T000000/nir.tif",
        "samples/forest/forest_20250602T000000/red.tif",
        "samples/forest/forest_20250602T000000/nir.tif",
    ]
    assert [file.scene for file in files[::2]] == [
        "forest_20250601T000000",
        "forest_20250602T000000",
    ]
    assert files[0].time == pd.Timestamp("2025-06-01")
    assert files[0].raster.gs.variables == ("red",)
    assert "time" not in files[0].raster.dims


def test_layout_of_a_raster_without_a_time_dimension_is_one_scene() -> None:
    scalar = build_raster(times=1).isel(time=0)

    (file,) = cogs.layout(scalar, "samples/label")

    assert (file.scene, file.path, file.time) == ("label", "samples/label.tif", None)
    assert file.raster.gs.variables == ("red", "nir")


def test_write_writes_exactly_the_layout(tmp_path: Path) -> None:
    source = build_raster(times=2)
    root = tmp_path / "forest"

    written = cogs.write(source, root, split_bands=True)

    planned = cogs.layout(source, root, split_bands=True)
    assert [str(path) for path in written] == [file.path for file in planned]
```

`build_raster` and `Path` are already imported in that file; add them if not.

- [ ] **Step 2: Run to verify they fail**

Run: `uv run pytest tests/geodata/io/test_cogs.py -q -p no:warnings`
Expected: FAIL, `module 'geosave_engine.geodata.io.cogs' has no attribute 'layout'`.

- [ ] **Step 3: Implement**

In `src/geosave_engine/geodata/io/cogs.py`, add `NamedTuple` to the `typing` import and replace everything from `def write(` to the end of the file. The docstring of `write` is unchanged; only its body is.

```python
class CogFile(NamedTuple):
    """One COG a raster is written as.

    Args:
        scene: Name of the scene the file belongs to: the raster's name, with
            its instant where the raster has a time dimension.
        path: Where the file goes, as text.
        raster: Pixels the file holds, without a time dimension.
        time: Instant the file holds, or None where the raster has no time
            dimension.
    """

    scene: str
    path: str
    raster: xr.Dataset
    time: pd.Timestamp | None


def layout(
    ds: xr.Dataset, path: str | PathLike[str], *, split_bands: bool = False
) -> list[CogFile]:
    """Say which COG files a raster is written as, without writing them.

    Args:
        ds: Raster to arrange.
        path: Name of the raster, as a local path or fsspec URL.
        split_bands: True gives each variable its own single-band file. A
            raster with one variable is one file per scene either way.

    Returns:
        One entry per file, in time then variable order.

    Raises:
        ValueError: `ds` carries no locatable grid, spans a non-spatial axis
            other than time, or its instants are empty, repeated or NaT.

    Examples:
        >>> layout(ds, "samples/forest")[0].path
        'samples/forest/forest_20250601T103031.tif'
    """
    spatial = ds.odc.spatial_dims
    if spatial is None:
        raise ValueError(
            "the raster carries no locatable grid, so its files cannot be "
            "georeferenced; assign a CRS with odc.geo.xr.assign_crs first"
        )
    # Only pixels have to fit a file; an axis only a coordinate spans writes nothing.
    pixel_dims = {str(dim) for array in ds.data_vars.values() for dim in array.dims}
    extra_dims = sorted(pixel_dims - {*spatial, TIME_COORDINATE})
    if extra_dims:
        raise ValueError(
            f"a COG holds one instant of one grid, but the raster also spans "
            f"{extra_dims}; select those axes away before writing"
        )

    # Names are joined as text, so a URL keeps its `scheme://`.
    root = str(path).rstrip("/")
    name = PurePosixPath(root).name

    # Each instant is one scene, named after the raster and its time. A raster
    # without a time dimension is one scene named after the raster alone.
    scenes: list[tuple[str, str, xr.Dataset, pd.Timestamp | None]] = []
    if TIME_COORDINATE in ds.dims:
        times = pd.DatetimeIndex(ds[TIME_COORDINATE].values)
        if times.hasnans or not times.is_unique or len(times) == 0:
            raise ValueError("scene times must be nonempty, unique and not NaT")
        for index, instant in enumerate(times):
            scene = f"{name}_{format_instant(instant)}"
            pixels = ds.isel({TIME_COORDINATE: index})
            scenes.append((scene, f"{root}/{scene}", pixels, instant))
    else:
        scenes.append((name, root, ds, None))

    # One variable is already one band per file, and its file keeps the scene's name.
    split = split_bands and len(ds.data_vars) > 1

    # A split scene is a folder holding one file per band; any other is one file.
    files: list[CogFile] = []
    for scene, scene_path, pixels, instant in scenes:
        if split:
            for band in pixels.data_vars:
                file = CogFile(scene, f"{scene_path}/{band}.tif", pixels[[band]], instant)
                files.append(file)
        else:
            file = CogFile(scene, f"{scene_path}.tif", pixels, instant)
            files.append(file)
    return files


def write(
    ds: xr.Dataset,
    path: str | PathLike[str],
    *,
    split_bands: bool = False,
    map_scale: float | None = None,
    overwrite: bool = False,
    storage_options: StorageOptions | None = None,
    **options: Unpack[COGWriteOptions],
) -> tuple[Path | str, ...]:
    """<keep the existing docstring unchanged>"""
    written = []
    for file in layout(ds, path, split_bands=split_bands):
        written_path = write_cog(
            cast("Dataset", file.raster),
            file.path,
            map_scale=map_scale,
            overwrite=overwrite,
            storage_options=storage_options,
            **options,
        )
        written.append(written_path)
    return tuple(written)
```

"Keep the existing docstring unchanged" means: leave the docstring text that is in the file now exactly as it is.

- [ ] **Step 4: Run tests and type check**

Run: `uv run pytest tests/geodata/io/test_cogs.py tests/geodata/core/test_raster.py tests/geodata/core/test_stack.py -q -p no:warnings`
Expected: all pass.
Run: `uvx ty check src/geosave_engine/geodata/io/cogs.py --output-format concise`
Expected: no diagnostic.

---

### Task 3: `GeoRaster.to_items` from the layout

**Files:**
- Modify: `src/geosave_engine/geodata/core/raster.py` (the `to_items` overloads and body, ~lines 778–885)
- Test: `tests/geodata/core/test_catalog.py`

**Interfaces:**
- Consumes: `asset.from_raster`, `asset.from_path`, `asset.default_key` (Task 1); `cogs.layout`, `CogFile` (Task 2); `item.from_assets(assets, *, id, datetime=None, collection=None)` (exists).
- Produces: `GeoRaster.to_items(path=None, *, driver="cog", collection=None, **options) -> tuple[pystac.Item, ...]`. For `driver="cog"`, `split_bands` defaults to true and every other option reaches `to_cog`. The `item.from_paths` call sites in this file are gone.

- [ ] **Step 1: Write the failing tests**

In `tests/geodata/core/test_catalog.py`, replace the body of `test_a_saved_raster_describes_itself_from_where_it_was_read` and add the others:

```python
def test_a_saved_raster_describes_itself_from_where_it_was_read(tmp_path: Path) -> None:
    store = build_raster(times=2).gs.to_zarr(tmp_path / "forest.zarr")

    with read_raster(store) as saved:
        (item,) = saved.gs.to_items()

    assert item.id == "forest"
    assert item.collection_id == "forest"
    assert item.assets["image"].href == str(store)


def test_a_raster_read_from_a_folder_of_cogs_is_indexed_when_written(tmp_path) -> None:
    build_raster(times=2).gs.to_cog(tmp_path / "scenes")

    with read_raster(tmp_path / "scenes") as saved:
        with pytest.raises(ValueError, match="index it when writing"):
            saved.gs.to_items()


def test_a_collection_is_named_by_the_caller_or_by_the_path(tmp_path: Path) -> None:
    cube = build_raster(times=1)

    (named,) = cube.gs.to_items(tmp_path / "a", collection="sentinel-2")
    (pathed,) = cube.gs.to_items(tmp_path / "b")

    assert named.collection_id == "sentinel-2"
    assert pathed.collection_id == "b"


def test_a_scalar_instant_raster_is_one_scene_named_by_the_path(tmp_path: Path) -> None:
    from datetime import UTC, datetime

    (item,) = build_raster(times=1).isel(time=0).gs.to_items(tmp_path / "label")

    assert (item.id, item.datetime) == ("label", datetime(2025, 6, 1, tzinfo=UTC))
    assert sorted(item.assets) == ["nir", "red"]


def test_one_file_per_scene_is_keyed_as_image(tmp_path: Path) -> None:
    items = build_raster(times=2).gs.to_items(tmp_path / "forest", split_bands=False)

    assert [sorted(item.assets) for item in items] == [["image"], ["image"]]
    assert items[0].assets["image"].href == str(
        tmp_path / "forest/forest_20250601T000000.tif"
    )


def test_a_timeless_raster_refuses_and_names_the_way_out(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="datetime="):
        build_raster().gs.to_items(tmp_path / "dem")


def test_writer_options_reach_the_writer(tmp_path: Path) -> None:
    import rasterio

    (item,) = build_raster(times=1).gs.to_items(tmp_path / "forest", compress="ZSTD")

    with rasterio.open(item.assets["red"].href) as src:
        assert src.compression.name == "zstd"
```

- [ ] **Step 2: Run to verify they fail**

Run: `uv run pytest tests/geodata/core/test_catalog.py -q -p no:warnings`
Expected: `test_a_raster_read_from_a_folder_of_cogs_is_indexed_when_written` and `test_a_collection_is_named_by_the_caller_or_by_the_path` FAIL (no `collection` argument; folder currently described). The others may already pass.

- [ ] **Step 3: Implement**

In `raster.py`, add `collection: str | None = None` to all four overloads after `driver` (for the first overload, `def to_items(self, path: None = None, *, collection: str | None = None)`), and replace the implementation:

```python
    def to_items(
        self,
        path: str | PathLike[str] | None = None,
        *,
        driver: RasterDriver = "cog",
        collection: str | None = None,
        **options: Any,
    ) -> tuple[pystac.Item, ...]:
        """Describe this raster's saved files as STAC Items, saving it first if needed.

        A raster read from one file or store is described where it sits. Any
        other raster is written to `path` by the writer `driver` names, then
        described from the pixels written.

        Args:
            path: Where to save an unsaved raster, as `to_cog`, `to_zarr` or
                `to_netcdf` takes it. None describes the file or store this
                raster was read from.
            driver: Format to save in. `"cog"` writes one file per band and
                instant, which every geospatial tool opens; `"zarr"` and
                `"netcdf"` write one store holding every instant.
            collection: Name the Items share. None names them after `path`.
            **options: Forwarded to `to_cog`, `to_zarr` or `to_netcdf`. COGs
                split bands unless `split_bands=False`, so each band is an
                asset of its own.

        Returns:
            One Item per instant for COGs, or one Item for a store.

        Raises:
            ValueError: `path` is None and this raster was not read from one
                file or store, or its pixels changed since; or the raster is
                timeless, which `stac.item.from_assets(..., datetime=...)`
                dates by hand.
            FileExistsError: A file exists and `overwrite` is false.

        Examples:
            >>> items = ds.gs.to_items("samples/forest")
            >>> [item.id for item in items]
            ['forest_20250601T103031', 'forest_20250611T103031']
            >>> ds.gs.to_items("samples/forest.zarr", driver="zarr")[0].id
            'forest'
            >>> read_raster("samples/forest.zarr").gs.to_items()[0].id
            'forest'
        """
        from geosave_engine.geodata.io import cogs
        from geosave_engine.geodata.stac import asset, item

        # A saved raster is described from the one file or store it was read from.
        if path is None:
            source = self._data.encoding.get("source")
            if source is None:
                raise ValueError(
                    "this raster is not saved, so no file describes it; pass a "
                    "path to save it to"
                )
            if Path(source).is_dir() and PurePosixPath(source).suffix != ".zarr":
                raise ValueError(
                    f"{source} is a folder of files, and this raster no longer "
                    f"says which instant each one holds; index it when writing, "
                    f"with to_items(path)"
                )
            name = PurePosixPath(source).stem
            assets = {asset.default_key(self._data): asset.from_path(source)}
            saved = item.from_assets(assets, id=name, collection=collection or name)
            return (saved,)

        # COGs: one Item per scene, each file described from the pixels it holds.
        if driver == "cog":
            name = PurePosixPath(str(path).rstrip("/")).name
            # Catalogs key assets by band, so each band gets a file of its own.
            split_bands = options.pop("split_bands", True)
            self.to_cog(path, split_bands=split_bands, **options)

            scenes: dict[str, dict[str, pystac.Asset]] = {}
            for file in cogs.layout(self._data, path, split_bands=split_bands):
                if file.scene not in scenes:
                    scenes[file.scene] = {}
                key = asset.default_key(file.raster)
                scenes[file.scene][key] = asset.from_raster(
                    file.raster, file.path, driver="cog"
                )

            items = []
            for scene, assets in scenes.items():
                scene_item = item.from_assets(
                    assets, id=scene, collection=collection or name
                )
                items.append(scene_item)
            return tuple(items)

        # A store holds every instant, so it is one Item. It has to be written
        # now: a deferred write leaves nothing to describe.
        if driver == "zarr":
            self.to_zarr(path, compute=True, **options)
        else:
            self.to_netcdf(path, compute=True, **options)
        name = PurePosixPath(str(path)).stem
        assets = {
            asset.default_key(self._data): asset.from_raster(
                self._data, path, driver=driver
            )
        }
        store_item = item.from_assets(assets, id=name, collection=collection or name)
        return (store_item,)
```

- [ ] **Step 4: Run tests**

Run: `uv run pytest tests/geodata/core/test_catalog.py tests/geodata/stac -q -p no:warnings`
Expected: all pass, including `test_a_written_catalog_loads_in_odc_stac` and both `test_a_registered_raster_reads_back_through_a_query` cases.

---

### Task 4: Delete `from_paths`; narrow `from_assets` properly

**Files:**
- Modify: `src/geosave_engine/geodata/stac/item.py`
- Modify: `tests/geodata/stac/test_item.py`
- Modify: `tests/geodata/core/test_catalog.py`

**Interfaces:**
- Consumes: `GeoRaster.to_items` (Task 3).
- Produces: `stac/item.py` exports only `from_assets(assets, *, id, datetime=None, collection=None)`.

- [ ] **Step 1: Move the `from_paths` tests onto `to_items`**

In `tests/geodata/stac/test_item.py`, delete these tests (their behaviour is now pinned in `test_catalog.py` by Task 3, or no longer exists): `test_one_write_becomes_one_item_per_scene`, `test_a_store_is_one_item_named_by_its_stem`, `test_a_scene_is_named_by_the_path_its_files_share`, `test_items_of_one_write_share_a_collection`, `test_a_scalar_instant_file_is_named_by_its_stem`, `test_files_of_two_rasters_in_one_call_refuse`, `test_a_timeless_file_refuses_and_names_the_way_out`. Change the import to `from geosave_engine.geodata.stac.item import from_assets`, and rewrite the two that remain useful:

```python
@pytest.mark.integration
def test_items_validate_against_the_stac_schemas(tmp_path: Path) -> None:
    for item in build_raster(times=2).gs.to_items(tmp_path / "forest"):
        item.validate()


def test_items_survive_the_table(tmp_path: Path) -> None:
    items = build_raster(times=2).gs.to_items(tmp_path / "forest")

    table = GeoVector.from_items(items).gs.to_geoparquet(tmp_path / "catalog.parquet")

    restored = list(stac_table_to_items(pq.read_table(table)))
    assert [entry["id"] for entry in restored] == IDS
    assert [band["name"] for band in restored[0]["assets"]["red"]["eo:bands"]] == [
        "red"
    ]
```

Keep whatever assertions follow `"red"]` in the existing `test_items_survive_the_table` as they are. Add to `test_item.py`:

```python
def test_an_asset_without_a_grid_cannot_place_an_item() -> None:
    import pystac

    bare = pystac.Asset("s0/notes.txt", roles=["data"])

    with pytest.raises(ValueError, match="states no grid"):
        from_assets({"notes": bare}, id="s0")
```

Add to `tests/geodata/core/test_catalog.py` (pins what the deleted per-scene test pinned):

```python
def test_each_scene_is_dated_and_bounded(tmp_path: Path) -> None:
    from datetime import UTC, datetime

    items = build_raster(times=2).gs.to_items(tmp_path / "forest")

    assert [item.datetime for item in items] == [
        datetime(2025, 6, 1, tzinfo=UTC),
        datetime(2025, 6, 2, tzinfo=UTC),
    ]
    assert all(len(item.bbox) == 4 for item in items)
```

- [ ] **Step 2: Run to verify the new test fails**

Run: `uv run pytest tests/geodata/stac/test_item.py -q -p no:warnings`
Expected: `test_an_asset_without_a_grid_cannot_place_an_item` FAILS with a `TypeError` from `tuple(None)`, not the `ValueError`.

- [ ] **Step 3: Implement**

In `src/geosave_engine/geodata/stac/item.py`: delete `from_paths`, `_shared_name`, and the imports only they used (`posixpath`, `Iterable`, `PathLike`, `PurePosixPath`, `EOExtension` stays, `from . import asset as stac_asset`). Update the module docstring to `"""Build native STAC Items from named assets."""`. In `from_assets`, replace the footprint block and the span block:

```python
    # The Item's footprint is the first asset's grid, in WGS84 as STAC states it.
    first_key = next(iter(assets))
    projection = ProjectionExtension.ext(assets[first_key])
    shape = projection.shape
    transform = projection.transform
    if shape is None or transform is None:
        raise ValueError(
            f"asset {first_key!r} states no grid, so the Item has no footprint; "
            f"build assets with stac.asset.from_raster or stac.asset.from_path"
        )
    height, width = shape
    geobox = GeoBox((height, width), Affine(*transform[:6]), projection.crs_string)
    footprint = geobox.extent.to_crs("EPSG:4326").geom

    # The Item covers `datetime`, or else everything its dated assets cover.
    if datetime is not None:
        start = datetime
        end = datetime
    else:
        starts = []
        ends = []
        for asset in assets.values():
            times = asset.common_metadata
            if times.start_datetime is not None and times.end_datetime is not None:
                starts.append(times.start_datetime)
                ends.append(times.end_datetime)
        if not starts:
            raise ValueError(
                f"none of the assets {list(assets)} is dated, and a STAC Item needs a "
                f"time; pass datetime="
            )
        start = min(starts)
        end = max(ends)
```

- [ ] **Step 4: Run tests and type check**

Run: `uv run pytest tests/geodata/stac tests/geodata/core/test_catalog.py tests/workflow tests/cli -q -p no:warnings`
Expected: all pass.
Run: `uvx ty check src/geosave_engine/geodata/stac --output-format concise`
Expected: `Found 0 diagnostics` (15 before this plan). If `GeoBox((height, width), ...)` is still reported, wrap as `wh_(width, height)` from `odc.geo`, which is odc's own typed shape constructor, rather than suppressing.
Run: `grep -rn "from_paths" src tests docs/guides`
Expected: no output.

---

### Task 5: `GeoArray.to_items`

**Files:**
- Modify: `src/geosave_engine/geodata/core/array.py` (after `to_gtiff`, before `to_raster`)
- Test: `tests/geodata/core/test_catalog.py`

**Interfaces:**
- Consumes: `GeoArray.to_raster() -> Dataset` (exists), `GeoRaster.to_items` (Task 3).
- Produces: `GeoArray.to_items(path, *, collection=None, map_scale=None, overwrite=False, storage_options=None, **options: Unpack[COGWriteOptions]) -> tuple[pystac.Item, ...]`.

- [ ] **Step 1: Write the failing test**

```python
def test_a_band_indexes_as_the_raster_it_converts_to(tmp_path: Path) -> None:
    source = build_raster(times=2)

    items = source["red"].gs.to_items(tmp_path / "red", collection="forest")

    assert [item.id for item in items] == ["red_20250601T000000", "red_20250602T000000"]
    assert [sorted(item.assets) for item in items] == [["red"], ["red"]]
    assert {item.collection_id for item in items} == {"forest"}
    restored = GeoVector.from_items(items).gs.to_raster()
    np.testing.assert_array_equal(restored.red, source.red)
```

- [ ] **Step 2: Run to verify it fails**

Run: `uv run pytest tests/geodata/core/test_catalog.py::test_a_band_indexes_as_the_raster_it_converts_to -q -p no:warnings`
Expected: FAIL, `'GeoArray' object has no attribute 'to_items'`.

- [ ] **Step 3: Implement**

Add `import pystac` under `TYPE_CHECKING` in `array.py`, then:

```python
    def to_items(
        self,
        path: str | PathLike[str],
        *,
        collection: str | None = None,
        map_scale: float | None = None,
        overwrite: bool = False,
        storage_options: StorageOptions | None = None,
        **options: Unpack[COGWriteOptions],
    ) -> tuple[pystac.Item, ...]:
        """Save this band as COGs and describe them as STAC Items.

        Args:
            path: Name to save the band under, as a local path or fsspec URL
                without a TIFF suffix.
            collection: Name the Items share. None names them after `path`.
            map_scale: Map denominator for TIFF resolution tags.
            overwrite: Replace existing files.
            storage_options: Options for the filesystem a URL names.
            **options: COG creation options.

        Returns:
            One Item per instant, each holding this band as its one asset.

        Raises:
            ValueError: This band is unnamed or timeless.
            FileExistsError: A file exists and `overwrite` is false.

        Examples:
            >>> [item.id for item in ds["ndvi"].gs.to_items("samples/ndvi")]
            ['ndvi_20250601T103031', 'ndvi_20250611T103031']
        """
        return self.to_raster().gs.to_items(
            path,
            driver="cog",
            collection=collection,
            map_scale=map_scale,
            overwrite=overwrite,
            storage_options=storage_options,
            **options,
        )
```

- [ ] **Step 4: Run the test**

Run: `uv run pytest tests/geodata/core/test_catalog.py -q -p no:warnings`
Expected: all pass.

---

### Task 6: `GeoStack.to_items`

**Files:**
- Modify: `src/geosave_engine/geodata/core/stack.py` (after `to_cog`)
- Test: `tests/geodata/core/test_stack.py`

**Interfaces:**
- Consumes: `cogs.layout` / `CogFile.time` (Task 2), `asset.from_raster` (Task 1), `item.from_assets` (Task 4), `GeoStack.to_cog(destination, ...)`, `GeoStack.rasters`, `format_instant` from `geosave_engine.geodata.utils.datetime`.
- Produces: `GeoStack.to_items(path, *, driver="cog", collection=None, **options) -> tuple[pystac.Item, ...]`.

- [ ] **Step 1: Write the failing tests**

Append to `tests/geodata/core/test_stack.py` (add `from geosave_engine.geodata import GeoVector` to its imports):

```python
def test_a_stack_indexes_one_item_per_instant_with_one_asset_per_group(
    tmp_path: Path,
) -> None:
    optical = build_raster(times=2)
    sample = build_stack({"optical": optical, "dem": build_raster()})

    items = sample.gs.to_items(tmp_path / "s0")

    assert [item.id for item in items] == ["s0_20250601T000000", "s0_20250602T000000"]
    # The timeless group joins every Item.
    assert [sorted(item.assets) for item in items] == [["dem", "optical"]] * 2
    assert {item.collection_id for item in items} == {"s0"}
    assert items[0].assets["dem"].href == str(tmp_path / "s0/dem.tif")

    row = GeoVector.from_items(items).iloc[0]
    restored = row.gs.to_stack()
    assert restored.gs.groups == ("optical", "dem")
    np.testing.assert_array_equal(
        restored["optical"].red.squeeze(), optical.red.isel(time=0)
    )


@pytest.mark.parametrize(("driver", "suffix"), [("zarr", ".zarr"), ("netcdf", ".nc")])
def test_a_stack_store_driver_writes_one_store_per_group(tmp_path, driver, suffix) -> None:
    optical = build_raster(times=2)
    sample = build_stack({"optical": optical, "label": build_raster(times=2)})

    (item,) = sample.gs.to_items(tmp_path / "s0", driver=driver)

    assert item.id == "s0"
    assert item.assets["optical"].href == str(tmp_path / f"s0/optical{suffix}")
    restored = GeoVector.from_items([item]).iloc[0].gs.to_stack()
    np.testing.assert_array_equal(restored["optical"].red, optical.red)
    assert read_stack(tmp_path / "s0").gs.groups == ("label", "optical")
```

- [ ] **Step 2: Run to verify they fail**

Run: `uv run pytest tests/geodata/core/test_stack.py -q -p no:warnings`
Expected: FAIL, `'GeoStack' object has no attribute 'to_items'`.

- [ ] **Step 3: Implement**

In `stack.py`, add `Any` to the `typing` import, `import pystac` and `from .raster import RasterDriver` under `TYPE_CHECKING`, then after `to_cog`:

```python
    def to_items(
        self,
        path: str | PathLike[str],
        *,
        driver: RasterDriver = "cog",
        collection: str | None = None,
        **options: Any,
    ) -> tuple[pystac.Item, ...]:
        """Save every group and describe the stack as STAC Items.

        Each group is one asset, keyed by its name. COGs give one Item per
        instant any group reaches, a timeless group joining every one; a
        store driver gives one Item, with one store per group.

        Args:
            path: Folder the groups are saved into, as a local path or fsspec
                URL.
            driver: Format to save each group in.
            collection: Name the Items share. None names them after `path`.
            **options: Forwarded to each group's `to_cog`, `to_zarr` or
                `to_netcdf`.

        Returns:
            One Item per instant for COGs, or one Item for a store driver.

        Raises:
            ValueError: No group is dated, which
                `stac.item.from_assets(..., datetime=...)` dates by hand.
            FileExistsError: A file exists and `overwrite` is false.

        Examples:
            >>> items = sample.gs.to_items("samples/s0")
            >>> sorted(items[0].assets)
            ['label', 'optical']
        """
        from geosave_engine.geodata.io import cogs
        from geosave_engine.geodata.stac import asset, item
        from geosave_engine.geodata.utils.datetime import format_instant

        # Names are joined as text, so a URL keeps its `scheme://`.
        root = str(path).rstrip("/")
        name = PurePosixPath(root).name
        if collection is None:
            collection = name

        # A store driver writes one store per group, and the stack is one Item.
        if driver != "cog":
            suffix = ".zarr" if driver == "zarr" else ".nc"
            assets: dict[str, pystac.Asset] = {}
            for group, raster in self.rasters.items():
                store = f"{root}/{group}{suffix}"
                if driver == "zarr":
                    raster.gs.to_zarr(store, compute=True, **options)
                else:
                    raster.gs.to_netcdf(store, compute=True, **options)
                assets[group] = asset.from_raster(raster, store, driver=driver)
            stack_item = item.from_assets(assets, id=name, collection=collection)
            return (stack_item,)

        # COGs: every group is written, then its files are sorted by instant.
        self.to_cog(root, **options)
        timeless: dict[str, pystac.Asset] = {}
        dated: dict[pd.Timestamp, dict[str, pystac.Asset]] = {}
        for group, raster in self.rasters.items():
            for file in cogs.layout(raster, f"{root}/{group}"):
                described = asset.from_raster(file.raster, file.path, driver="cog")
                if file.time is None:
                    timeless[group] = described
                    continue
                if file.time not in dated:
                    dated[file.time] = {}
                dated[file.time][group] = described

        # A stack no group dates is still one Item, which `from_assets` refuses
        # until it is given a time.
        if not dated:
            return (item.from_assets(timeless, id=name, collection=collection),)

        items = []
        for instant in sorted(dated):
            # A timeless group, such as a DEM, belongs to every instant.
            scene_assets = {**dated[instant], **timeless}
            scene_item = item.from_assets(
                scene_assets,
                id=f"{name}_{format_instant(instant)}",
                collection=collection,
            )
            items.append(scene_item)
        return tuple(items)
```

`PurePosixPath` must be imported from `pathlib` in `stack.py`; add it to the existing `from pathlib import Path` line.

- [ ] **Step 4: Run the tests**

Run: `uv run pytest tests/geodata/core/test_stack.py -q -p no:warnings`
Expected: all pass. If the asset order assertion `restored.gs.groups == ("optical", "dem")` fails because `{**dated, **timeless}` orders the DEM last while the stack lists it elsewhere, keep the implementation and assert on `sorted(restored.gs.groups)` instead: asset order within an Item carries no meaning.

---

### Task 7: Guides, design record, and full verification

**Files:**
- Modify: `docs/guides/architecture.md` (the catalog section, ~lines 86–135)
- Modify: `docs/superpowers/specs/2026-10-06-catalog-loop-design.md` (status line only)

- [ ] **Step 1: Update the architecture guide**

Replace the paragraph beginning "Each path is one asset." through the sentence ending "how a training sample lists its layers:" with:

```markdown
A file holds one raster, so an asset is built from one:
`stac.asset.from_raster(raster, href, driver=...)` describes pixels as they
are written and opens no file, and `stac.asset.from_path` reads a file's
header and calls it. `to_items` on a raster, a band or a stack saves through
`to_cog`, `to_zarr` or `to_netcdf` and describes what it wrote. An Item holds
named rasters, one asset each: band names for a split raster, group names for
a stack. `stac.item.from_assets` assembles one by hand, which is how a sample
whose layers are already on disk is listed:
```

Add after the `items = read_raster(...)` example line in the code block above it:

```python
items = sample.gs.to_items("samples/s0")              # a stack: one asset per group
```

- [ ] **Step 2: Mark the superseded record**

In `docs/superpowers/specs/2026-10-06-catalog-loop-design.md`, append to the `Status:` paragraph:

```markdown
Rules 1, 3 and 5, `from_paths`, `from_rasters` and `stac=True` are superseded
by `2026-10-06-items-from-rasters-design.md`.
```

- [ ] **Step 3: Full verification**

Run: `uv run pytest -q -p no:warnings`
Expected: all pass, none skipped that passed before (baseline: 1504 passed, 16 deselected, plus the tests this plan adds and minus the seven it deletes).
Run: `uv run ruff check src tests`
Expected: `All checks passed!`
Run: `uvx ty check src/geosave_engine/geodata/stac src/geosave_engine/geodata/io/cogs.py --output-format concise`
Expected: `Found 0 diagnostics`.
Run: `uvx ty check src/geosave_engine/geodata/core/raster.py src/geosave_engine/geodata/core/stack.py src/geosave_engine/geodata/core/array.py --output-format concise | grep -n "to_items"`
Expected: no line inside any `to_items` body. Record the total count before and after so no new diagnostic is hidden among old ones.
Run: `grep -rn "from_paths\|from_rasters\|stac=True\|media_type=" src docs/guides`
Expected: only `pystac.Asset(href, media_type=media_type, ...)` in `stac/asset.py`.

- [ ] **Step 4: Report**

Report what changed, the checks run with their output, and the limits from the spec's "Known limits" section.
