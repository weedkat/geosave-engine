# Attrs header factories cleanup

## Intent

A header factory takes a source and returns the `AttrsHeader` that source
describes, and `rebase` writes it in one call. Callers stop unpacking headers
into per-coordinate namespace loops, stop parsing a whole header to read one
model, and the STAC factory stops re-implementing agreement checks.

This builds on `2026-10-03-attrs-scoped-models-design.md` and supersedes the
STAC packing contract in `2026-10-03-geodata-attrs-review-design.md`, which
raised on partial packing.

## A header holds only what it describes

```python
@dataclass(frozen=True)
class AttrsHeader:
    root: AttrsNamespace = AttrsNamespace("dataset")   # empty: says nothing about the root
    data_vars: Mapping[str, AttrsNamespace] = {}
    coords: Mapping[str, AttrsNamespace] = {}
```

`header.root` is never None, so `ds.gs.attrs.root.get(ACDD)` needs no
narrowing. `rebase(obj, header)` replaces the root only when the header's root
carries attrs, plus every variable and coordinate the header names. A model
field set to None still counts as carrying attrs, since it clears its key.

Because an empty root writes nothing, every join starts from attrs xarray has
dropped (`combine_attrs="drop"` in `concat_time`, `merge_bands`, and
`read_tree`), so a foreign key that every input carries differently does not
linger from the first input.

## The grid header describes its coordinates completely

```python
# attrs/headers/geobox.py
def create_header(geobox: GeoBox) -> AttrsHeader:
    """Describe a grid's spatial coordinates: odc's attrs plus their CF meaning."""
    names = (
        ("latitude", "longitude")
        if geobox.crs.geographic
        else ("projection_y_coordinate", "projection_x_coordinate")
    )
    grid_coords = [coord for name, coord in xr_coords(geobox).items() if name in geobox.dimensions]
    return AttrsHeader.from_attrs(
        coords={
            str(coord.name): {**coord.attrs, "standard_name": name, "axis": axis}
            for coord, name, axis in zip(grid_coords, names, ("Y", "X"), strict=True)
        }
    )
```

`write_crs` and the array builder apply it whole:

```python
return attrs.rebase(result, create_geobox_header(geobox))
```

The grid coordinates' attrs are replaced: odc's `units`, `resolution`, and
`crs`, plus CF's `standard_name` and `axis`. A stale `crs` attr that odc wrote
for an earlier CRS is corrected. Other keys a user set on `x` or `y` are
dropped, because the grid owns its coordinates.

## Reading one model reads one mapping

Internal code that needs one model parses that model from the mapping it
already knows:

| Site | Today | After |
|---|---|---|
| `utils/io/zarr.py` order | `create_header(ds).root.get(ZarrOrder)` | `ZarrOrder.from_attrs(ds.attrs)` |
| `utils/io/zarr.py` fill encoding | `create_header(ds).data_vars` → `.get(Nodata)` | `Nodata.from_attrs(ds[name].attrs)` |
| `core/base.py` `timespan` | `self.attrs.coords["time"].get(TimeSpec)` | `TimeSpec.from_attrs(coords["time"].attrs)` |
| `transform/nodata.py` (2 sites) | `array.gs.attrs.root.get(Nodata)` | `Nodata.from_attrs(array.attrs)` |
| `transform/packing.py` | `array.gs.attrs.root.get(Packing)` | `Packing.from_attrs(array.attrs)` |

`ds.gs.attrs` stays the public, scope-checked view.

## STAC items that disagree drop with a warning

STAC items are external input, so one bad item must not stop a load. For each
loaded variable, a field is kept only when every item publishes it with one
value. Otherwise it is dropped with a `DroppedAttrsWarning` that lists each
item's value. This covers packing (`scale`, `offset`) and labelling (`unit`,
`description`) alike. Dropping `scale_factor` leaves every item's pixels as
stored digital numbers, which is honest about not knowing the scale.

```python
_ATTR_KEYS = {
    "unit": "units",
    "description": "long_name",
    "scale": "scale_factor",
    "offset": "add_offset",
}

published = [{_ATTR_KEYS[key]: value for key, value in fields.items() if key in _ATTR_KEYS}
             for fields in per_item_fields]
agreed = common_attrs(published)
dropped = sorted(set().union(*published) - agreed.keys())
if dropped:
    warnings.warn(f"STAC items disagree on {name!r} ...", DroppedAttrsWarning)
variables[name] = {**agreed, **variable.attrs}      # the loader still wins
```

`common_attrs(mappings)` in `attrs/model.py` returns the keys every mapping
carries with one value, compared with `attrs_equal`. `GeoRaster.to_array` uses
the same function for the attrs every band shares.

This deletes `_LABELLING`, `_PACKING`, `AssetConflict`, `_shared_fields`, and
`_disagreement`. `StacGroupby` moves to `stac/source.py`, its only user.

## Left as is

- `write_rgb` and the GeoTIFF writer's per-band `GDALVariable` loop, because
  each band gets a different value.
- `rebase(result, source.gs.attrs.root)` in warp, time, concat, and features,
  which is the intended namespace patch onto the root.
- `headers/xarray.py` and `headers/gdal.py`, apart from the root change.

## Risks

- Keys users set on `x`/`y` are dropped by `write_crs`.
- A STAC load whose items disagree on packing now loads without
  `scale_factor` and warns, instead of raising.

## Testing

- **Header:** a header with an empty root leaves the object's root attrs
  untouched; a time concat drops a root key its rasters disagree on.
- **Grid:** `write_crs` gives `x`/`y` odc's attrs plus CF's in one call; a
  stale `crs` attr is replaced; the root and the time coordinate are
  untouched.
- **Single-model reads:** covered by the existing zarr, timespan, nodata, and
  packing suites. They change behaviour nowhere.
- **STAC:**
  - partial and mixed packing warn and drop;
  - a label missing from one item warns and drops;
  - agreeing fields are kept;
  - the loader's nodata still wins.

  Plan A's two raising tests become warning tests.
- **`common_attrs`:** shared keys kept, a differing or missing key dropped,
  NaN equals NaN.
- **Suites:** the full suite, `ruff check src tests`, and the docstring
  checker on touched files.
