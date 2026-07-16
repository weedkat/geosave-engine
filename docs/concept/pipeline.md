# GeoPipeline

Reference for how ingestion actually works — what a `GeoPipeline` is, the
two kinds of "source," and the mechanics behind `save_dataset`/`stream_ingest`.
For the ordered how-to (write one, validate it, build a dataset from it),
see [workflow.md](../guide/workflow.md#2-explore-and-validate-a-pipeline-).

## Philosophy

A `GeoPipeline` does exactly one thing: build the layers for **one anchor**.
`ingest(anchor) -> dict[layer_name, GeoTile]` is its whole contract. Turning
many anchors into a saved training set, or streaming them live without
touching disk, is deliberately **not** the pipeline's job — those are
separate, external functions (`save_dataset`, `stream_ingest`) that take any
`GeoPipeline` and do the looping/saving/streaming around it.

That split buys two things:

- **A pipeline instance has no notion of "where output goes."** The same
  instance saves to as many roots as you call `save_dataset()` against
  (train/val/test splits, different projects) or streams straight into a
  live prediction loop via `stream_ingest()` — nothing about the class
  itself couples it to disk.
- **Building model input for one live sample and building a saved dataset in
  bulk share the exact same code path.** `ingest()` is what both
  `save_dataset()` and a `predict`-time `stream_ingest()` call — fix a bug
  in `ingest()` once, both paths pick it up.

`GeoTile` (and its data-less counterpart `GeoAnchor`) is what flows through
every stage below — full API in [geotile.md](geotile.md).

## Anatomy of a GeoPipeline

```python
from functools import cached_property

from geosave_engine.geodata.tile import GeoTile
from geosave_engine.geodata.pipeline import GeoPipeline, SourceProtocol
from geosave_engine.geodata.stac import StacClient

stac_client = StacClient.cdse()


class Sentinel2OnlyPipeline(GeoPipeline):
    """Fetch Sentinel-2 L1C RGB bands for one anchor tile."""

    @cached_property
    def sources(self) -> dict[str, SourceProtocol]:
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
| `sources` (property) | Your pipeline pulls from named sources (the common case) — one `.load(anchor)` call per source. Use `cached_property` if building a source needs a live client (STAC auth, etc.), so constructing the pipeline just to call `.context()` stays network-free. | `{}` |
| `fetch(anchor)` | Almost never — only if your anchor already carries its own data and there's no I/O left to do (e.g. a label pipeline reading pixels the anchor's own `GeotiffSource` already loaded). | Calls `.load(anchor)` on every entry in `sources`. |
| `preprocess(raw)` | Deriving final layers from fetched raw layers — pure, no I/O. | Passthrough. |
| `ingest(anchor)` | Rarely — it's just `preprocess(fetch(anchor))`. | As described. |
| `context(tiles)` | Attaching per-sample metadata (crs, datetime, STAC item ids, ...) to a rendered sample's `"context"` key. | `{}` |

`ingest(anchor) -> dict[str, GeoTile]` is the one method every pipeline is
built around — everything else exists to make writing it easier.

## Two things called "source"

The word "source" means two different things in this codebase depending on
where you're standing:

**1. Ingest source** (`geosave_engine.geodata.pipeline` — `CoordinateSource`,
`GeoJSONSource`, `PolygonSource`, `GeotiffSource`, `ZarrSource`). Answers
*"where do my anchors come from?"* — a coordinate, an AOI polygon/GeoJSON, a
folder of GeoTIFFs, or an already-ingested zarr layer. Each is a small
pydantic model implementing one method:

```python
class AnchorSource(BaseModel):
    def to_anchors(self, limit: int | None = None) -> Sequence[GeoAnchor]: ...
```

| Source | Built from | Notes |
| --- | --- | --- |
| `CoordinateSource` | `lat`, `lon`, `datetime`, `size_m` | Always exactly one anchor. |
| `PolygonSource` | `geom` (GeoJSON geometry), `datetime` | One anchor, exact footprint stored. |
| `GeoJSONSource` | `src` (file or directory of `.geojson`/`.json`), `datetime` | One anchor per feature. |
| `GeotiffSource` | `src` (file or directory of `.tif`/`.tiff`) | One anchor per file, already carrying data — datetime derived from each filename's `-YYYYMMDD` suffix, no separate `datetime` field. |
| `ZarrSource` | `src` (file or directory of `.zarr` stores) | One anchor per store, already carrying data. |

`GeotiffSource`/`ZarrSource` anchors already carry data (`GeoTile`, not a
bare `GeoAnchor`) — `GeoPipeline.fetch` has no I/O left to do for those
layers.

`AnyAnchorSource` is the discriminated union of all five (keyed by a `type`
field), so a source spec round-trips through plain dict/YAML/JSON:

```python
from geosave_engine.geodata.pipeline import source_from_dict

source = source_from_dict({"type": "geotiff", "src": "data/raw/train/labels/"})
anchors = source.to_anchors(limit=100)
```

**2. STAC source** (returned by `StacClient.source(...)`). Answers *"how do
I pull live satellite pixels for one anchor?"* — given a STAC collection and
a band list, `.load(anchor)` searches the catalog for that anchor's
bbox/datetime window and loads matching scenes via `odc-stac`. You only
touch this **inside** a `GeoPipeline.ingest()` that pulls from a live
catalog.

They compose: an ingest source produces the anchors your pipeline (or
`save_dataset`) loops over; a STAC source, if your pipeline's `sources`
declares one, is what actually fetches pixels for each anchor.

## Pulling from a live satellite catalog

```python
from geosave_engine.geodata.stac import StacClient

cdse = StacClient.cdse()                        # or .planetary_computer() / .element84() / .local(url)
source = cdse.source("sentinel-2-l1c", bands=["B02", "B03", "B04"], max_nodata_fraction=0.1)
tile = source.load(anchor)                       # anchor: a GeoAnchor with bbox + datetime window
```

| Constructor | Endpoint |
| --- | --- |
| `StacClient.cdse()` | Copernicus Data Space Ecosystem |
| `StacClient.planetary_computer()` | Microsoft Planetary Computer (auto-signs asset URLs) |
| `StacClient.element84()` | Element84 Earth Search |
| `StacClient.local(url)` | Any self-hosted STAC endpoint (e.g. a pgSTAC instance) — same `.search()`/`.source()` interface, no separate caching mechanism needed |

`source.load(anchor)` searches every scene within the anchor's own datetime
window (ascending, no lookback), loads them via `odc-stac`, downloads with
retry, and drops any scene over `max_nodata_fraction`. Raises
`AnchorFetchError` if nothing usable is found — catch it in `preprocess()`
if a missing anchor should be skippable rather than fatal.

`load` always returns raw values exactly as the provider published them —
no radiometric scaling, no compositing. Apply those as explicit steps inside
`preprocess()` (see the cloud-mask/NDVI example below).

Credentials: the plain `AWS_ACCESS_KEY_ID` / `AWS_SECRET_ACCESS_KEY` /
`AWS_S3_ENDPOINT` in your workspace's `.env` is enough — GDAL reads those
directly, and since the client is declared once at module level
(`stac_client = StacClient.cdse()` in the example above), there's nothing
per-call to wire up.

## Pulling from local GeoTIFF

For labels or any pre-downloaded raster that isn't self-describing about its
acquisition date, `GeotiffSource` derives one from the filename:

```python
from geosave_engine.geodata.pipeline import GeotiffSource

source = GeotiffSource(src="data/raw/train/labels/")
anchors = source.to_anchors()   # each one already has .data populated
```

Each matched file becomes one anchor via `GeoTile.from_geotiff`, with the
filename datetime range attached. Filename stems must end in `-YYYYMMDD` or
`-YYYYMMDD-YYYYMMDD`. One date expands to its full day; two dates define the
inclusive range. Since `from_geotiff` loads pixel data by default for this
source, each anchor's `.data` is already populated — useful when the raster
itself is both the anchor *and* the thing you need to process (see
[Handling labels](#handling-labels) below).

## Building a derived layer inside preprocess()

A derived layer just reads the `.data` of a tile already fetched in the same
call — no chaining mechanism, no intermediate directory. This is
`workspace/modules/data_pipeline.py`'s actual `Pipeline`, in full:

```python
class Pipeline(GeoPipeline):
    """Sentinel-2 imagery + cloud/shadow mask + NDVI for one anchor."""

    @cached_property
    def sources(self) -> dict[str, SourceProtocol]:
        stac_client = StacClient.cdse()
        return {"sentinel_2_l1c": stac_client.source("sentinel-2-l1c", bands=L1C_BANDS, max_nodata_fraction=0.1)}

    def context(self, tiles: dict[str, GeoTile]) -> dict[str, object]:
        ref = next(iter(tiles.values()))
        return {
            "crs": ref.crs,
            "transform": ref.affine,
            "coordinate": ref.centroid,
            "time": ref.start.timetuple().tm_yday,
            "datetime": ref.start.isoformat(),
            "bbox_wgs84": list(ref.wgs84_bbox),
            "stac_item_ids": [i.id for i in ref.stac],
        }

    def preprocess(self, raw: dict[str, GeoTile]) -> dict[str, GeoTile]:
        s2 = raw["sentinel_2_l1c"]
        cloud_mask = self._ingest_cloud_mask(s2)   # reads s2.data, no re-fetch
        ndvi = self._ingest_ndvi(s2)                # reads s2.data, no re-fetch
        return {
            "sentinel_2_l1c": s2.with_metadata({"description": "Sentinel-2 L1C imagery (all bands)"}),
            "cloud_mask": cloud_mask.with_metadata({"description": "Cloud and shadow mask, 0=clear, 1=cloud/shadow"}),
            "ndvi": ndvi.with_metadata({"description": "Normalized Difference Vegetation Index"}),
        }
```

Everything `_ingest_cloud_mask`/`_ingest_ndvi` need is already sitting on
`s2` — no separate fetch, no adapter step between layers.

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

def ingest_split(raw_dir: Path, out_root: Path) -> None:
    anchors = [GeoTile.from_geotiff(p) for p in raw_dir.rglob("*.tif")]

    save_dataset(Pipeline(), anchors, out_root)   # imagery, unchanged Pipeline

    for anchor in anchors:
        geostack_dir = out_root / f"{anchor.stem}{GEOSTACK_SUFFIX}"
        label_path = geostack_dir / "dynamicworld.zarr"
        if not geostack_dir.exists() or label_path.exists():
            continue   # imagery ingest failed, or label already written
        build_label(anchor).to_zarr(label_path, save_stac=True)
```

Why not fold this into `Pipeline.ingest()`: the raw anchor here already *is*
the label source (`GeotiffSource` loaded its pixels), and the remap table is
tied to one specific dataset release's class encoding — bundling it into
the general-purpose imagery `Pipeline` would couple two things that vary
independently. Writing straight into the anchor's own `.geostack` folder
(which `save_dataset` already created) needs nothing beyond `GeoTile.to_zarr`.

One consequence worth knowing: a layer written this way — straight to disk,
bypassing `save_dataset`'s per-anchor loop — won't appear in that run's
`manifest.json` `"layers"` metadata block (see [Saving to disk](#saving-to-disk)
below), since that block only reflects what `save_dataset` itself observed
each `ingest()` call return.

## Saving to disk

```python
from geosave_engine.geodata.pipeline import save_dataset

pipeline = Pipeline()
anchors = source.to_anchors(limit=None)
save_dataset(pipeline, anchors, root="data/train", save_stac=["sentinel_2_l1c"])
```

`save_stac` defaults to `False` (no `.stac.json` sidecars written at all).
Pass `True` to save every layer's, or a list of layer names to save it for
only those — worth doing for the one layer that actually came from a real
STAC search, and skipping derived layers (a cloud mask, NDVI) that just
inherit the same `stac` list unchanged and would otherwise write a
duplicate, no-new-information sidecar per layer.

`limit` caps how many anchors this call considers, without re-slicing
`anchors` yourself — handy for a quick test run against a handful of
anchors before committing to the full source.

Writes `root/<anchor_stem>.geostack/<layer_name>.zarr` for every layer
`ingest()` returns, plus one `root/manifest.json`:

```json
{
  "metadata": {
    "pipeline": "Pipeline",
    "layers": {"sentinel_2_l1c": {"description": "..."}, "cloud_mask": {"description": "..."}}
  },
  "anchors": {
    "<stem>": {"source": null, "status": "done", "error": null, "store": "<stem>.geostack"}
  }
}
```

Resumable: re-running `save_dataset` against the same `root` skips anchors
already `done`/`error`'d. A failure on one anchor is logged and recorded via
`mark_error`, not raised — the rest of the batch keeps going. `metadata` is
only refreshed on a run that actually ingested ≥1 new anchor; a fully
resumed no-op run leaves it untouched.

`anchors` can be anything iterable — a real `AnchorSource.to_anchors()`
call, or a hand-built list for anchors that don't fit an existing source
(the label-ingest pattern above builds its list straight from
`GeoTile.from_geotiff`, no `AnchorSource` involved).

## Streaming without saving

```python
from geosave_engine.geodata.pipeline import stream_ingest

for sample in stream_ingest(pipeline, anchors, sel_bands={"sentinel_2_l1c": ["B04", "B03", "B02"]}):
    ...  # tensor dict, same shape GeoDataset.__getitem__ returns
```

A plain generator, not a Dataset class — for predicting straight from a live
source with no disk round trip. Wrap it in a one-off `IterableDataset` at
the call site if a `DataLoader` needs one, and shard it by
`torch.utils.data.get_worker_info()` there if using `num_workers > 1` —
`stream_ingest` itself has no worker-awareness.
