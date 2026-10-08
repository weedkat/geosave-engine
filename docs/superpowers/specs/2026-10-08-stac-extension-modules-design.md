# STAC Extension Modules Design

**Status:** Implemented 2026-10-08. Replaces the deleted

> **Later change, 2026-10-08:** `stac/band.py` and `stac/header.py` no longer
> exist. Band encoding is `stac.asset.band_fields`; band reading and decoding
> (`read_bands`, `band_attrs`) sit with `create_header` in
> `attrs/headers/stac.py`. `StacTableClient` lives in `stac/client.py`.
`plans/2026-10-08-xarray-stac-geoparquet.md`.

**Goal:** Translate between xarray attrs models and community STAC schemas in
one readable place per schema, so `xarray -> pystac.Item -> STAC GeoParquet`
is understood by other tools and reads back into the same attrs.

## Settled decisions

1. GeoSave publishes the STAC 1.1 `bands` list. It reads 1.1 `bands` and the
   legacy `raster:bands` / `eo:bands` lists external catalogs still publish.
2. One module per extension under `stac/extensions/`, plain functions, listed
   in explicit tuples. No registry, no plugin entry points.
3. All six extensions are in this spec: Projection, Raster, EO,
   Classification, Datacube, CF. They are built in that order.
4. Callers keep today's API.

```python
paths = ds.gs.to_cog("data/scenes")
items = ds.gs.to_items(paths, collection="optical")
table.write(items, "data/items.parquet")
```

## Layout

```text
geodata/
├── attrs/models/
│   └── spectral.py          # NEW  Spectral(common_name, center_wavelength, full_width_half_max)
└── stac/
    ├── item.py              # id, footprint, time; declares the schemas its assets use
    ├── asset.py             # href, media type, roles, time; asset-level extensions
    ├── band.py              # NEW  STAC 1.1 core band fields; band-level extensions
    ├── header.py            # MOVED from attrs/headers/stac.py
    └── extensions/
        ├── __init__.py      # BAND, ASSET, schemas()
        ├── types.py         # BandFields, AssetFields, BandExtension, AssetExtension
        ├── raster.py        # Packing            <-> raster:scale, raster:offset
        ├── eo.py            # Spectral           <-> eo:common_name, eo:center_wavelength, eo:full_width_half_max
        ├── classification.py# Legend             <-> classification:classes
        ├── projection.py    # geobox              -> proj:code | proj:wkt2, proj:shape, proj:transform
        ├── datacube.py      # dims, coords        -> cube:dimensions, cube:variables
        └── cf.py            # CFVariable          -> cf:parameter
```

`item > asset > band` are STAC core and sit side by side. `extensions/` holds
only what a schema URI names. `attrs/` no longer knows any STAC spelling;
`StacMetadata` stays there because it is an attrs model.

## Types

```python
# stac/extensions/types.py
type BandFields = dict[str, Any]
type AssetFields = dict[str, Any]


class BandExtension(Protocol):
    """A schema that adds prefixed keys to one band."""

    PREFIX: str
    SCHEMA: str

    def encode_band(self, attrs: FlatAttrs) -> BandFields: ...
    def decode_band(self, band: BandFields) -> AttrsModel | None: ...


class AssetExtension(Protocol):
    """A schema that adds prefixed keys to one asset."""

    PREFIX: str
    SCHEMA: str

    def encode_asset(self, raster: xr.Dataset) -> AssetFields: ...
```

A module satisfies a protocol by defining those names at module level.

`BandFields` is one entry of an asset's `bands` list:

```python
{
    "name": "red",
    "data_type": "uint16",
    "nodata": 0,
    "unit": "1",
    "description": "Red reflectance",
    "raster:scale": 0.0001,
    "raster:offset": -0.1,
    "eo:common_name": "red",
    "eo:center_wavelength": 0.665,
}
```

`AssetFields` is the keys one extension adds beside `href`, `type`, `roles`:

```python
{
    "proj:code": "EPSG:32749",
    "proj:shape": [2, 2],
    "proj:transform": [10.0, 0.0, 500000.0, 0.0, -10.0, 9000000.0],
}
```

## Contracts

```python
# stac/extensions/__init__.py
BAND: tuple[BandExtension, ...] = (raster, eo, classification)
ASSET: tuple[AssetExtension, ...] = (projection, datacube, cf)


def schemas(item: pystac.Item) -> list[str]:
    """Return the schema URIs whose prefix an Item's assets or bands use."""
```

```python
# stac/band.py
def from_variable(variable: xr.DataArray) -> BandFields:
    """Describe one saved variable as a STAC 1.1 band."""

def to_attrs(band: BandFields) -> FlatAttrs:
    """Translate one band into the variable attrs it states."""

def read(asset: pystac.Asset) -> list[BandFields]:
    """Return an asset's bands in STAC 1.1 spelling, in file order."""
```

```python
# stac/extensions/raster.py  (every band module has this shape)
PREFIX = "raster"
SCHEMA = "https://stac-extensions.github.io/raster/v2.0.0/schema.json"


def encode_band(attrs: FlatAttrs) -> BandFields:
    """Spell a variable's packing as Raster fields.

    Examples:
        >>> encode_band({"scale_factor": 0.0001, "add_offset": -0.1})
        {'raster:scale': 0.0001, 'raster:offset': -0.1}
        >>> encode_band({"units": "1"})
        {}
    """


def decode_band(band: BandFields) -> Packing | None:
    """Read the packing a band's Raster fields state.

    Examples:
        >>> decode_band({"name": "red", "raster:scale": 0.0001})
        Packing(scale_factor=0.0001)
    """
```

### Rules

- **Empty means absent.** A module returns `{}` or `None` when its facts are
  missing. `schemas()` declares a URI only when its prefix appears, so a label
  or DEM no longer declares EO.
- **Band modules map attrs, not arrays.** `encode_band` takes the flat attrs a
  writer stores; it never sees pixels and is tested with plain dicts.
- **Asset modules only encode.** On read the file states its own grid and
  dimensions, so nothing decodes them from STAC.
- **One model per band extension.** Raster owns `Packing`, EO owns `Spectral`,
  Classification owns `Legend`. Core band fields own `Nodata` and
  `CFVariable.units` / `long_name`, in `band.py`.

### Stored facts

A band states what is on disk. xarray keeps those facts in `.attrs` for raw
values and moves them to `.encoding` once `xr.decode_cf` has applied them, and
writers re-pack from `.encoding`. `band.from_variable` therefore reads both,
encoding first:

```python
# After xr.decode_cf, the packing, fill and dtype a writer stores sit in .encoding.
stored = {**variable.attrs, **variable.encoding}
data_type = np.dtype(stored.get("dtype", variable.dtype)).name
```

Smoke-tested 2026-10-08: a `decode_cf` raster written with `write_cog` re-read
as `uint16` with the source scale and identical stored values.

`asset.from_raster` keeps its contract: no file is opened, so the raster must
be the one saved there. Encoding passed to a writer call instead of carried on
the raster is not seen; describe the re-read file in that case.

## Mappings

| Module | Level | Reads | Writes | Decodes to |
| --- | --- | --- | --- | --- |
| `band.py` | band | name, stored dtype, `Nodata`, `CFVariable` | `name`, `data_type`, `nodata`, `unit`, `description` | `CFVariable` (the loader that opens the file states nodata and dtype) |
| `raster` v2.0.0 | band | `Packing` | `raster:scale`, `raster:offset` | `Packing` |
| `eo` v2.0.0 | band | `Spectral` | `eo:common_name`, `eo:center_wavelength`, `eo:full_width_half_max` | `Spectral` |
| `classification` v2.0.0 | band | `Legend.class_map`, `color_map` | `classification:classes` | `Legend` |
| `projection` v2.0.0 | asset | `raster.gs.geobox` | `proj:code` or `proj:wkt2`, `proj:shape`, `proj:transform` | — |
| `datacube` v2.2.0 | asset | dims, coordinate labels, geobox | `cube:dimensions`, `cube:variables` | — |
| `cf` v0.2.0 | asset | `CFVariable.standard_name`, `units` | `cf:parameter` | — |

Details that are not a plain rename:

- **Classification.** One class per `flag_values` entry, named by
  `flag_meanings`, with `color_hint` as six uppercase hex digits where
  `color_map` has the value. A `Legend` with `flag_masks` writes nothing: CF
  masks do not state the offset, length and per-value names a bitfield needs.
- **Datacube.** Written only when the raster has a `time` dimension. `x` and
  `y` are `spatial` with label extent, signed step and `reference_system`;
  `time` is `temporal` with first and last label. `cube:variables` lists each
  data variable's dimensions, `type: "data"`, and unit.
- **CF.** One `{"name": standard_name, "unit": units}` per variable carrying a
  standard name; nothing otherwise. `cell_methods` is not exported.
- **Spectral.** A new variable-scope attrs model writing the attr keys
  `common_name`, `center_wavelength`, `full_width_half_max` (micrometres, as
  EO states them). It is filled when a STAC load decodes `eo:` fields or when
  a caller rebases it. `sensors.yaml` is not consulted; no wavelength is
  inferred from a variable name.

### Legacy bands on read

PySTAC 1.14.3 does not migrate `raster:bands` / `eo:bands` to `bands`
(probed with `Item.from_dict(..., migrate=True)`). `band.read` upgrades them:
entries are zipped by index; `nodata`, `data_type`, `unit`, `statistics`,
`name`, `description` keep their name and every other key gains its
extension's prefix. Parquet null fills are dropped in the same function.

## Changes to existing code

- `stac/asset.py`: `_describe` shrinks to media type, roles, `bands` from
  `band.from_variable`, the `ASSET` loop, and time. `_band_fields` and the
  PySTAC Raster / EO / Projection wrappers go. `default_key` reads names from
  `band.read`. The no-grid `ValueError` stays here.
- `stac/item.py`: `from_raster` sets `stac_extensions = extensions.schemas(item)`
  in place of three unconditional `add_to` calls.
- `stac/header.py`: `create_header` reads each variable's band with
  `band.read(asset)[index - 1]` and `band.to_attrs`; the hand-built
  `band_attrs` dict goes. `read_asset_fields` returns asset fields merged with
  that band.
- `attrs/models/__init__.py`: register `Spectral` under `variable`.
- `attrs/model.py`: `from_attrs` reads a NumPy integer as an `int`, so a
  `uint16` fill stays `0` (it read as `0.0`). A float32 scale still widens to
  `9.999999747378752e-05`: the writers store that widened value, so reading
  it shorter would make a described raster disagree with its own file.

## Breaking changes

- Published assets carry `bands` instead of `raster:bands` / `eo:bands`, and
  declare Raster/EO/Classification v2.0.0 only when used. odc-stac 0.5.2 does
  not read dtype, nodata or band names from `bands`; GeoSave's own
  `table.load` reads the files and is unaffected.
- `StacSourceConfig.asset_fields` and `list_asset_fields()` use 1.1 spelling:
  `eo:center_wavelength`, `raster:scale`.
- `attrs.headers.stac` is now `stac.header`.

## Tests

Tests mirror the source: `tests/geodata/stac/test_band.py`,
`tests/geodata/stac/extensions/test_<module>.py`, `tests/geodata/stac/test_header.py`,
`tests/geodata/attrs/models/test_spectral.py`.

- Each band module: dict in, dict out, both directions, and `{}` / `None` when
  the facts are missing.
- `band.read`: 1.1, legacy, and Parquet null-padded input give the same bands.
- `band.from_variable`: raw and `decode_cf` forms of one raster give the same
  band.
- `extensions.schemas`: a label Item declares Classification and Projection
  only; an optical Item declares no Classification.
- Round trip per variable kind (packed optical, label with colours, plain DEM):
  attrs -> Item -> `Item.validate()` -> GeoParquet -> Item -> attrs.
- Laziness: building Items computes no pixels and opens no file (existing
  tests keep passing).

## Follow-up specs

Each gets its own spec; findings carried from the deleted plan:

1. **CF packing lifecycle.** `transform/packing.unpack` returns correct
   physical values but drops `.encoding`, so a later write stores floats.
   `Legend` leaves `flag_values` as Python ints; CF wants the variable's type.
2. **Collections in GeoParquet.** `table.write` passes no Collections to
   stac-geoparquet, and read/edit/rewrite drops ones a file already holds.
   ACDD cannot rebuild a source Collection. Agreed direction:
   `source.load(anchor, return_stac=True)` also returns Items and Collection.
3. **PySTAC loading and xarray engine.** `table.load` merges every asset's
   `xarray:open_kwargs` into one dict, so two groups of one store both open
   the last group. An `xr.open_dataset(item, engine="geosave")` backend is a
   thin layer over this loader and waits for it.
