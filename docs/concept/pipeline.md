# GeoPipeline

Reference for how ingestion actually works — what a `GeoPipeline` is, the
two kinds of "source," and the mechanics behind saving to disk or streaming
without saving. For the ordered how-to (write one, validate it, build a
dataset from it), see
[workflow.md](../guide/workflow.md#2-explore-and-validate-a-pipeline-).

## Philosophy

A `GeoPipeline` does exactly one thing: build the samples for **one
anchor**. `ingest(anchor) -> Iterator[GeoStack]` is its whole contract —
one anchor can still yield more than one sample (see
[One anchor, one or more samples](#one-anchor-one-or-more-samples) below).
Turning many anchors into a saved training set, or streaming them live
without touching disk, is deliberately **not** the pipeline's job — that's
a plain loop the caller writes around `ingest()`/`ingest_to_tensor()`.

That split buys two things:

- **A pipeline instance has no notion of "where output goes."** The same
  instance can be looped over disk-saving code, streamed straight into a
  live prediction loop, or both — nothing about the class itself couples
  it to disk.
- **Building model input for one live sample and building a saved dataset in
  bulk share the exact same code path.** `ingest()` is what both a
  bulk-save loop and `ingest_to_tensor()` call — fix a bug in `ingest()`
  once, both paths pick it up.

`GeoTile` (and its data-less counterpart `GeoAnchor`) is what flows through
every stage below — full API in [geotile.md](geotile.md).

## Anatomy of a GeoPipeline

```python
from functools import cached_property

from geosave_engine.geodata.tile import GeoTile
from geosave_engine.geodata.pipeline import GeoPipeline
from geosave_engine.geodata.stac import StacClient
from geosave_engine.geodata.stac.source import StacSource

stac_client = StacClient.cdse()


class Sentinel2OnlyPipeline(GeoPipeline):
    """Fetch Sentinel-2 L1C RGB bands for one anchor tile."""

    @cached_property
    def sources(self) -> dict[str, StacSource]:
        return {
            "sentinel_2_l1c": stac_client.source(
                "sentinel-2-l1c", bands=["B04", "B03", "B02"], max_nodata_fraction=0.1
            )
        }

    def preprocess(self, raw: dict[str, GeoTile]) -> dict[str, GeoTile]:
        tile = raw["sentinel_2_l1c"]
        return {"sentinel_2_l1c": tile.with_data(tile.data.astype("float32"))}
```

| Method | Override when | Default |
| --- | --- | --- |
| `sources` (property) | Your pipeline pulls from named sources (the common case) — one `.load(anchor)` call per source. Use `cached_property` if building a source needs a live client (STAC auth, etc.), so constructing the pipeline just to call `.ingest()` stays network-free until actually used. | `{}` |
| `fetch(anchor)` | Almost never — only if your anchor already carries its own data and there's no I/O left to do (e.g. a label pipeline reading pixels an anchor built via `GeoTile.from_geotiff` already loaded). | Calls `.load(anchor)` on every entry in `sources`, aligning results across sources by real time overlap. |
| `preprocess(raw)` | Deriving final layers from fetched raw layers — pure, no I/O. | Passthrough. |
| `context(tiles)` | Supplying extra per-sample keys a model needs — e.g. a Prithvi/Clay encoder's `temporal_coords`/`location_coords` — beyond the tensor + `"anchors"` every sample already carries. See [Supplying model-specific context](#supplying-model-specific-context) below. | Returns `{}` — no extra keys. |
| `ingest(anchor)` | Rarely — it's just `preprocess(fetch(anchor))`, wrapped into a `GeoStack` per aligned sample. | As described. |

`ingest(anchor) -> Iterator[GeoStack]` is the one method every pipeline is
built around — everything else exists to make writing it easier.

## Two things called "source"

The word "source" means two different things in this codebase depending on
where you're standing:

**1. Ingest source** (`geosave_engine.geodata.pipeline` — `CoordinateSource`,
`GeoJSONSource`, `PolygonSource`). Answers *"where do my anchors come
from?"* — a coordinate, an AOI polygon, or a GeoJSON file/directory. Each is
a small pydantic model implementing one method:

```python
class AnchorSource(BaseModel):
    def to_anchors(self, limit: int | None = None) -> Iterator[GeoAnchor]: ...
```

| Source | Built from | Notes |
| --- | --- | --- |
| `CoordinateSource` | `lat`, `lon`, `area_m` | Always exactly one anchor, before chunking. |
| `PolygonSource` | `geom` (GeoJSON geometry) | One anchor, exact footprint stored, before chunking. |
| `GeoJSONSource` | `src` (file or directory of `.geojson`/`.json`) | One anchor per feature, before chunking. |

`datetime`/`resolution`/`crs`/`tile_size_px` are shared base fields on
every source — `tile_size_px` chunks whatever full-extent anchor the
source builds into consistent, model-input-sized square tiles (default
500px/side), so a source never hands the pipeline one huge anchor no
model could consume in one pass.

`AnyAnchorSource` is the discriminated union of all three (keyed by a
`type` field), so a source spec round-trips through plain dict/YAML/JSON:

```python
from geosave_engine.geodata.pipeline import source_from_dict

source = source_from_dict({"type": "coordinate", "lat": -6.2, "lon": 106.8, "datetime": "2024-01-01", "area_m": 5120})
anchors = source.to_anchors(limit=100)
```

**2. STAC source** (returned by `StacClient.source(...)`, a `StacSource`
instance). Answers *"how do I pull live satellite pixels for one anchor?"*
— given a STAC collection and a band list, `.load(anchor)` searches the
catalog for that anchor's bbox/datetime window and loads matching scenes
via `odc-stac`. You only touch this **inside** a `GeoPipeline.ingest()`
that pulls from a live catalog.

They compose: an ingest source produces the anchors your own bulk-save/stream
loop passes to `pipeline.ingest(anchor)`; a STAC source, if your pipeline's
`sources` declares one, is what actually fetches pixels for each anchor.

## Pulling from a live satellite catalog

```python
from geosave_engine.geodata.stac import StacClient

cdse = StacClient.cdse()                        # or .planetary_computer() / .element84() / .local(url)
source = cdse.source("sentinel-2-l1c", bands=["B02", "B03", "B04"], max_nodata_fraction=0.1)
tile = next(source.load(anchor))                 # anchor: a GeoAnchor with bbox + datetime window
```

| Constructor | Endpoint |
| --- | --- |
| `StacClient.cdse()` | Copernicus Data Space Ecosystem |
| `StacClient.planetary_computer()` | Microsoft Planetary Computer (auto-signs asset URLs) |
| `StacClient.element84()` | Element84 Earth Search |
| `StacClient.local(url)` | Any self-hosted STAC endpoint (e.g. a pgSTAC instance) — same `.search()`/`.source()` interface, no separate caching mechanism needed |

`source.load(anchor)` searches within the anchor's own datetime window,
loads matches via `odc-stac`, downloads with retry, and drops any scene
over `max_nodata_fraction`. Raises `AnchorFetchError` if nothing usable is
found — caught automatically by `GeoPipeline.fetch`, so a bad anchor is
skipped, not fatal.

`load` always returns raw values exactly as the provider published them —
no radiometric scaling, no compositing. Apply those as explicit steps inside
`preprocess()` (see the cloud-mask/NDVI example below).

See [One anchor, one or more samples](#one-anchor-one-or-more-samples)
below for how a source's own temporal config decides how many samples one
anchor's window actually produces.

Credentials: the plain `AWS_ACCESS_KEY_ID` / `AWS_SECRET_ACCESS_KEY` /
`AWS_S3_ENDPOINT` in your workspace's `.env` is enough — GDAL reads those
directly, and since the client is declared once at module level
(`stac_client = StacClient.cdse()` in the example above), there's nothing
per-call to wire up.

## One anchor, one or more samples

Temporal bucketing is owned entirely by each `StacSource`, not by
`GeoPipeline` — two sources in the same pipeline can run completely
independent temporal configs, no coordination between them. `GeoPipeline`
itself never decides to split or merge anything; `ingest(anchor)` yields
however many aligned samples `fetch()` finds, driven by whichever source
produced the most tiles for that anchor.

`StacClient.source(...)` (i.e. `StacSource`) accepts:

- **`temporal_granularity`** (`"scene"` / `"day"` / `"month"` / `"year"`,
  default `"scene"`) — what one sample's time axis is bucketed by.
  `"scene"`: one bucket per real matched acquisition, this source's own
  timestamps. `"day"`/`"month"`/`"year"`: calendar buckets instead, which
  can span more than one real scene each.
- **`temporal_reduce`** (`"first"` / `"last"` / `"median"` / `"mean"`,
  default `"median"`) — how to collapse a bucket with more than one real
  scene down to exactly one time step.
- **`temporal_slots`** (`int`, default `1`) — how many consecutive
  buckets stack into *one* yielded sample's time dimension. `1` is a
  no-op: each bucket is its own sample.
- **`temporal_strides`** (`int | None`, default `None` → equals
  `temporal_slots`) — how many buckets apart consecutive samples start.
  Lower than `temporal_slots` gives overlapping/sliding-window samples
  (e.g. `temporal_slots=4, temporal_strides=1` for a dense sliding
  4-step sequence). A trailing window shorter than `temporal_slots` is
  dropped, not padded.
- **`temporal_fallback`** (`bool`, default `False`) — allow substituting
  the nearest real scene (by absolute time distance, ignoring the bucket
  window) when a bucket has none of its own. Default skips an empty
  bucket rather than silently using stale data.

```python
stac_client.source("sentinel-2-l1c", temporal_granularity="month")                                    # one sample per calendar month, median-composited
stac_client.source("sentinel-2-l1c", temporal_granularity="month", temporal_slots=4)                   # 4 consecutive months, one (time=4, ...) sample
stac_client.source("sentinel-2-l1c", temporal_granularity="month", temporal_slots=4, temporal_strides=1)  # dense sliding 4-month sequences
```

A bucket with nothing usable raises `AnchorFetchError` from the source,
same as any other empty-window case — `GeoPipeline.fetch` skips it and
moves on rather than crash.

## Pulling from local GeoTIFF

For labels or any pre-downloaded raster that isn't self-describing about its
acquisition date, `GeoTile.from_geotiff` derives one from the filename:

```python
from geosave_engine.geodata.tile import GeoTile

anchors = [GeoTile.from_geotiff(p, load_data=True) for p in Path("data/raw/train/labels/").glob("*.tif")]
```

Filename stems must end in `-YYYYMMDD` or `-YYYYMMDD-YYYYMMDD` (pass an
explicit `datetime` to bypass that convention). `load_data=True` populates
`.data` immediately — useful when the raster itself is both the anchor
*and* the thing you need to process (see
[Handling labels](#handling-labels) below). Default `load_data=False`
stays lazy, pixels read only when something actually calls `.to_tensor()`.

## Building a derived layer inside preprocess()

A derived layer just reads the `.data` of a tile already fetched in the same
call — no chaining mechanism, no intermediate directory. Trimmed from
`workspace/modules/data_pipeline.py`'s actual `Pipeline` (full file has the
cloud/shadow mask math in `_ingest_cloud_mask`/`_ingest_ndvi`):

```python
class Pipeline(GeoPipeline):
    """Sentinel-2 imagery + cloud/shadow mask + NDVI for one anchor."""

    @cached_property
    def sources(self) -> dict[str, StacSource]:
        stac_client = StacClient.cdse()
        return {"sentinel_2_l1c": stac_client.source("sentinel-2-l1c", bands=L1C_BANDS, max_nodata_fraction=0.1)}

    def preprocess(self, raw: dict[str, GeoTile]) -> dict[str, GeoTile]:
        s2 = raw["sentinel_2_l1c"]
        # Cloud mask/NDVI need the full L1C_BANDS fetch (B01/B8A/B09/B10) —
        # select down to the model's own input bands only after they're computed.
        cloud_mask = self._ingest_cloud_mask(s2)   # reads s2.data, no re-fetch
        ndvi = self._ingest_ndvi(s2)                # reads s2.data, no re-fetch
        s2_model = s2.with_data(s2.data.sel(band=DW_MODEL_BANDS))
        return {
            "sentinel_2_l1c": s2_model.with_metadata({"description": "Sentinel-2 L1C imagery (DynamicWorld input bands)"}),
            "cloud_mask": cloud_mask.with_metadata({"description": "Cloud and shadow mask, 0=clear, 1=cloud/shadow"}),
            "ndvi": ndvi.with_metadata({"description": "Normalized Difference Vegetation Index"}),
        }
```

Everything `_ingest_cloud_mask`/`_ingest_ndvi` need is already sitting on
`s2` — no separate fetch, no adapter step between layers.

## Supplying model-specific context

Every rendered sample always carries its tensors plus `"anchors"`
(`dict[LayerName, GeoAnchor]`, one bare anchor per layer — see
[geotile.md](geotile.md)). Some models need more than that: a Prithvi
`_tl` encoder wants `temporal_coords`/`location_coords`; Clay wants
`time`/`latlon`/`gsd`. `GeoPipeline.context(tiles)` is where a pipeline
supplies exactly those extra keys — overridable, same pattern as
`preprocess`, default `{}` (no extra keys):

```python
class Pipeline(GeoPipeline):
    def context(self, tiles: dict[str, GeoTile]) -> dict[str, torch.Tensor]:
        tile = tiles["sentinel_2_l1c"]          # picked by name, not first-in-dict
        lon, lat = tile.centroid
        acquired = tile.times[0]                # real per-scene timestamp off the loaded
                                                 # data's own time coordinate — not
                                                 # tile.start/tile.end (GeoAnchor.datetime),
                                                 # which a reduced-precision query date
                                                 # gets widened into a whole-day range
        day_of_year = acquired.timetuple().tm_yday - 1  # Prithvi wants 0-indexed
        return {
            "temporal_coords": torch.tensor([[acquired.year, day_of_year]], dtype=torch.float32),
            "location_coords": torch.tensor([lat, lon], dtype=torch.float32),
            "time": torch.tensor([acquired.isocalendar().week, acquired.hour], dtype=torch.float32),
            "latlon": torch.tensor([lat, lon], dtype=torch.float32),
            "gsd": torch.tensor(tile.resolution, dtype=torch.float32),
        }
```

Every key/shape here mirrors a real `forward()` param name exactly — no
model-specific reshape stage downstream. A key a given model doesn't
declare as a `@model_context` requirement just sits unused in the context
dict (see [model.md](model.md) for `ContextChain`/`model_context`) — one
pipeline's `context()` can serve several different model architectures at
once without any of them needing to know about the others' keys.

Reaches every consumer that renders a sample: `GeoStack.to_tensor(context_fn=...)`,
`GeoDataset(context_fn=...)`, and `ingest_to_tensor` all take/use the same
function. `SemanticSegmentationDataModule`'s own `pipeline` arg wires a
`GeoPipeline` instance's `.context` into every split's `GeoDataset`
automatically (see [model.md](model.md#semanticsegmentationdatamodule)) —
nothing to call by hand for the standardized training path.

## Handling labels

Label remapping (a raw dataset's own class encoding → your training schema)
is usually **specific to one raw data release**, not something a
general-purpose `Pipeline` should own. The pattern used by
`workspace/scripts/ingest.py`: keep imagery ingestion and label writing as
two fully independent steps, not bundled into one `ingest()` call.

```python
def build_label(anchor: GeoTile) -> GeoTile:
    """Remap the anchor's own raw pixel values into the target schema."""
    label = remap(anchor, LABEL_REMAP)   # {raw_value: target_value}
    label = label.with_data(label.data.assign_coords(band=["label"]))
    return label.with_nodata(255)

def ingest_group(raw_dir: Path, out_root: Path) -> None:
    anchors = [GeoTile.from_geotiff(p, load_data=True) for p in raw_dir.glob("*.tif")]

    for anchor in anchors:
        geostack_dir = out_root / f"{anchor.stem}.geostack"
        if geostack_dir.exists():
            continue   # already ingested
        for stack in Pipeline().ingest(anchor):   # imagery, unchanged Pipeline
            stack.add("dynamicworld", build_label(anchor)).save(geostack_dir)
```

Why not fold this into `Pipeline.ingest()`: the raw anchor here already *is*
the label source (`GeoTile.from_geotiff` loaded its pixels), and the remap
table is tied to one specific dataset release's class encoding — bundling
it into the general-purpose imagery `Pipeline` would couple two things that
vary independently. `stack.add(name, tile)` returns a new `GeoStack` with
the label merged in, ready for one single `.save()` call alongside the
imagery layers.

## Saving to disk

```python
root = Path("data/train")
for anchor in anchors:
    for stack in pipeline.ingest(anchor):
        stack.save(root / f"{anchor.stem}.geostack", save_stac=["sentinel_2_l1c"])
```

Writes `root/<anchor_stem>.geostack/<layer_name>.zarr` for every layer in
each yielded `GeoStack`. `save_stac` defaults to `False` (no `.stac.json`
sidecars written at all). Pass `True` to save every layer's, or a list of
layer names to save it for only those — worth doing for the one layer that
actually came from a real STAC search, and skipping derived layers (a
cloud mask, NDVI) that just inherit the same `stac` list unchanged and
would otherwise write a duplicate, no-new-information sidecar per layer.

No manifest, no built-in resumability — if re-running matters, skip
anchors whose folder already exists yourself, same check as
[Handling labels](#handling-labels) above (`if geostack_dir.exists(): continue`).
A failure ingesting one anchor is your own loop's problem to catch and log
if you don't want one bad anchor to stop the whole run.

`anchors` can be anything iterable — a real `AnchorSource.to_anchors()`
call, or a hand-built list for anchors that don't fit an existing source
(the label-ingest pattern above builds its list straight from
`GeoTile.from_geotiff`, no `AnchorSource` involved).

## Streaming without saving

```python
for anchor in anchors:
    for sample in pipeline.ingest_to_tensor(anchor, sel_bands={"sentinel_2_l1c": ["B04", "B03", "B02"]}):
        ...  # tensor dict + "anchors", same shape GeoDataset.__getitem__ returns,
             # plus this pipeline's own context() keys if it overrides one
```

Same one-anchor contract as `ingest()` — looping over many anchors is the
caller's own plain loop around it, not this method's job (see
[Philosophy](#philosophy) above). A plain generator method, not a Dataset
class — for predicting straight from a live source with no disk round
trip. Always applies `self.context`
(see [Supplying model-specific context](#supplying-model-specific-context)
above) — the live-predict path and the disk-training path
(`GeoDataset(context_fn=...)`) both end up calling the same
`GeoStack.to_tensor(context_fn=...)`, so a pipeline's `context()` behaves
identically either way. Wrap it in a one-off `IterableDataset` at the call
site if a `DataLoader` needs one, and shard it by
`torch.utils.data.get_worker_info()` there if using `num_workers > 1` —
`ingest_to_tensor` itself has no worker-awareness.

Built on `ingest()` — same alignment/skip-on-empty-bucket behavior, same
per-source temporal bucketing (see
[One anchor, one or more samples](#one-anchor-one-or-more-samples)): a wide
anchor or a source configured with `temporal_slots > 1` yields several
tensor samples, not one. A bucket with no usable scene (`AnchorFetchError`)
is skipped, not raised; the rest of the stream keeps going.
