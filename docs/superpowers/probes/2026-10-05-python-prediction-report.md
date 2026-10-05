# Plain Python prediction API experiment

Status: executable prototype verified; library API changes remain proposed.

Latest direction: the user rejected prepare_catalog and whole-catalog conversion.
Those caller proposals below are superseded. Use existing read_raster/read_stack,
format writers, GeoVector registration, GeoParquet, and selected-record reopening.
The active storage design is
[native raster storage and GeoVector records](../specs/2026-10-05-native-sample-inputs-design.md).
The numerical experiment remains evidence for tensor-only model execution.

The latest revision separates raster loading and tensor conversion from neural
modules. The encoder receives tensors; the existing segmentation head returns
raw logits. Ordinary functions prepare inputs and interpret outputs. No Prefect
job is needed to exercise this lifecycle.

The runnable experiment is
[`2026-10-05-python-prediction.py`](2026-10-05-python-prediction.py).
It replaces the earlier encoder-owned `prepare_inputs`, head-owned `decode`, and
Zarr-first persistence illustration. Those methods are not the accepted design.

## Catalog and raster ownership

The acquisition catalog is a native GeoDataFrame. Each STAC row describes one
acquisition, carrying geometry, datetime, projection metadata, and named assets.
The experiment stores two dates with two single-band COGs per date. A catalog
selection becomes a lazy xarray Dataset with time, band variables, and an explicit
load grid. A row is one acquisition, not necessarily a complete temporal sample.

```text
catalog.parquet
  acquisition-0: datetime=2024-01-01, assets={red: {...}, nir: {...}}
  acquisition-1: datetime=2024-03-01, assets={red: {...}, nir: {...}}

rasters/
  20240101T000000/red.tif
  20240101T000000/nir.tif
  20240301T000000/red.tif
  20240301T000000/nir.tif
```

Names are a writing arrangement, not the reading contract. Asset hrefs point to
files; the catalog datetime supplies acquisition time. Each COG preserves its
actual pixel grid, dtype, nodata, and band identity. Different source band grids
need per-asset projection metadata and explicit alignment when loaded together.
Zarr remains supported for callers who want cube persistence.

## Caller code exercised today

Fixture setup supplies a real local STAC collection, model stages, and a spec.
Only remote endpoint discovery/search is replaced by the fixture. The spec
loads the declared band assets onto the anchor through the actual STAC recipe.

```python
model = build_model(stages).eval()
acquired = spec.load_rasters(anchor)
acquired["raw"].gs.to_cog(output / "rasters", split_bands=True)
```

The fixture registers the written COG hrefs on copies of its STAC items, then
persists those items with the dependency's STAC-GeoParquet writer. The ordinary
read and inference path uses existing APIs:

```python
from odc.stac import load
from stac_geoparquet import to_item_collection

catalog = io.read_vector(output / "catalog.parquet")
raw = load(
    to_item_collection(catalog),
    geobox=anchor.geobox,
    bands=["red", "nir"],
    groupby="time",
    chunks={"x": 4, "y": 4},
)
prepared = spec.preprocess({"raw": raw})
inputs = prepare_inputs(prepared["image"])
batch = default_collate([inputs])

with torch.inference_mode():
    logits = model(**batch)
    labels = logits.argmax(dim=1)
```

`prepare_inputs` is a model-specific ordinary function. It converts the actual
prepared raster into this encoder's tensor axes and derives temporal/location
tensors from that same raster. It does not need a neural model instance.

```python
recipe = CallSpec(
    call=f"{__name__}.prepare_inputs",
    kwargs={"image": Ref("image")},
)
inputs = recipe.invoke({"image": image})
```

The same function is callable directly or through existing spec declarations.
Exact ModelSpec field placement remains proposed. Keep model-required inputs and
normalization shared by training and inference; method-owned augmentation is a
separate policy. Spatial augmentation must update any affected context.

## Acquisition, reads, and checking requirements

Today's `ModelSpec.load_rasters(anchor)` accepts only an anchor, performs remote
acquisition, and selects/checks each resulting raster. Already loaded rasters can
be checked with `spec.rasters[name].select_raster(data)`; `spec.preprocess` also
uses this selection/checking operation. Checks retain laziness and do not silently
reproject or resample.

The user rejected overloading acquisition, catalog reads, and validation under
`load_rasters`. The refined caller interface below is proposed, not implemented:

```python
scene = spec.ingest(anchor)  # Acquire the declared sources; return xr.DataTree.

# Or prepare a saved collection with a plain Python job, then choose one row.
prepared_path = prepare_catalog(
    "raw.parquet", spec=spec, output="prepared", format="zarr"
)
catalog = io.read_vector(prepared_path)
sample = catalog.iloc[0].gs.to_xarray()  # Proposed Series reader.

# The preparation job uses the spec's native raster processing internally.
```

`GeoStack` is the existing accessor on native `xr.DataTree`, not a wrapper.
Each group is a Dataset of bands on one compatible grid and coordinates. Several
COG band files can form one group; a file is not automatically a group. Catalog
collection/band metadata must identify the logical data, including distinct grids.
An anchor requests acquisition grid/time coverage; actual bands and compatibility
can only be established from matched assets and the resulting raster.

The Series reader owns asset pointers, stored windows, and acquisition datetime.
It is independent of the model spec. Catalog iteration and temporal grouping
belong to the plain Python preparation job with an explicit sample policy.
Preserve native grids when reading; any
alignment/resampling is explicit or declared in preprocessing. Native STAC/ODC
loading is still the useful implementation seam, with an explicit load grid where
alignment is requested. Validation does not read Parquet or acquire pixels.

A focused smoke check also assembled a real two-date optical Dataset and static
DEM into the existing `stack` factory, read them through native DataTree paths,
and rebuilt a prepared stack after cropping/time selection. Laziness, independent
group grids, static/temporal dimensions, and actual temporal input context passed.
This verifies the representation, not the proposed catalog reader or spec methods.

An additional local smoke check persisted a catalog containing a COG-backed row
and a Zarr-backed row, then read each with today's one-row table reader. Pixels,
laziness, and the Zarr's two-date coordinate survived. The many-row table read
correctly rejected ambiguous selection. Native Zarr group reads and native
DataLoader batching with retained IDs passed. Series access, group-aware asset
dispatch, and the Parquet preparation job are still proposed.

Two implementation gaps must be resolved before settling this interface:

- COG tree persistence currently returns no asset records. The storage operation
  should expose the hrefs it actually wrote so registration uses truthful
  pointers, without directory scanning or filename-derived acquisition dates.
- GeoSave's GeoParquet writer forces a covering `bbox` and fails if a standard
  STAC `bbox` column already exists. The experiment uses the dependency's native
  STAC writer. This is an exposed library issue, not a verified GeoSave writer
  round trip. Decide the schema with the dependency before changing that writer.

## Pixel tiling and output policies

Tiler and Merger own pixel splitting and merging. Reference rows come from the
actual prepared image; reference IDs remain outside the neural arguments.
The experiment reverses 20 rows, batches three samples, adds raw logits by native
tile ID, merges with a Hann window and halo removal, and then decodes classes.
Full-image and tiled logits match for the deterministic pointwise fixture.
This does not establish equivalence for arbitrary receptive fields or padding.

| Head | Neural result | Scene processing outside the model |
| --- | --- | --- |
| Segmentation | Pixel class logits | Merge logits, decode, attach prepared grid. |
| Pixelwise regression | Pixel values | Merge values, apply declared inverse transform, attach grid/units. |
| Classification | Sample class logits | Decode and retain reference identity in a table. |
| Detection | Current raw cell predictions | Head-specific decode, map tile-local boxes, resolve duplicates. |

The detection decoder is not implemented by this experiment. Avoid a universal
merger: only dense spatial outputs follow the raster merger path.

## Verification and limits

Run:

```bash
MPLCONFIGDIR=/tmp/geosave-python-mpl UV_CACHE_DIR=/tmp/geosave-uv-cache uv run --no-sync python docs/superpowers/probes/2026-10-05-python-prediction.py
```

Verified in the revised experiment:

- Two acquisitions, four real single-band COG assets, STAC-GeoParquet persistence,
  catalog reopening, and lazy ODC loading. Catalog rasters equal acquired rasters.
- Changing only catalog datetime changes loaded time coordinates; TIFF dates and
  filenames remain unchanged. This proves catalog time is used for assembly.
- Preprocessing changes time selection, grid, band name, and values before input
  conversion. Context observes March 1, 2024, zero-based day 60.
- Direct and declared input functions produce identical tensors without a model
  instance. Neural inputs remain tensors and outputs remain raw predictions.
- Twenty shuffled tiles, batch size three, and native Hann merging reproduce
  full-image logits. Prediction labels and grid survive a COG round trip.
- Regression `[1,1,7,9]`, classification `[1,2]`, and raw detection `[1,63,5]`
  native output contracts. No Prefect module is imported by the experiment.

Remote discovery, trained-model quality, GPU execution, invalid-pixel coverage,
detection decoding, different-resolution bands, and training integration remain
unverified here. The earlier NetCDF nested-attribute serialization limitation
was not addressed. No library implementation, dependency, wheel, or workflow was
changed; only the design, plan, and executable prototype were revised.
