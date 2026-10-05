"""Disposable Python API experiment: no Prefect, Lightning, or model downloads.

Run: uv run --no-sync python docs/superpowers/probes/2026-10-05-python-prediction.py
Input conversion and decoding are ordinary functions outside the neural model.
Only STAC endpoint discovery is replaced by a local fixture.
"""

from datetime import UTC, datetime
from itertools import batched
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import numpy as np
from odc.stac import load as load_stac
import pandas as pd
import pystac
from pystac.extensions.projection import ProjectionExtension
import rasterio
from stac_geoparquet import to_item_collection
from stac_geoparquet.arrow import parse_stac_items_to_parquet
import torch
from torch import nn
from torch.utils.data import default_collate
from tiler import Merger
import xarray as xr

from geosave_engine.geodata import GeoAnchor, GeoVector, io
from geosave_engine.geodata.core.array import array
from geosave_engine.geodata.stac.source import StacSource
from geosave_engine.geodata.transform import window
from geosave_engine.model.chain import Published, chain_step
from geosave_engine.model.registry import build_model, register_model
from geosave_engine.model.spec import CallSpec, ModelSpec, Ref


@register_model("encoder", "python_probe")
class Encoder(nn.Module):
    """A tensor-only encoder with a deterministic feature map."""

    feature_channels: Published[int]

    def __init__(self):
        super().__init__()
        self.feature_channels = 1

    @chain_step(outputs=("feature_map", "pyramid"))
    def forward(
        self,
        image: torch.Tensor,
        temporal_coords: torch.Tensor,
        location_coords: torch.Tensor,
    ) -> tuple[torch.Tensor, list[torch.Tensor]]:
        day = temporal_coords[:, :, 1].mean(dim=1)[:, None, None, None]
        feature_map = image.mean(dim=2) + day / 3660
        return feature_map, [feature_map]


def prepare_inputs(image: xr.Dataset) -> dict[str, torch.Tensor]:
    """Convert the actual prepared raster into this model's tensor arguments."""
    dates = pd.DatetimeIndex(image.time.values)
    longitude, latitude = image.gs.anchor.geographic_centroid
    return {
        # Native raster stacking is T,C,H,W; this encoder expects C,T,H,W.
        "image": image.gs.to_tensor(dtype="float32").permute(1, 0, 2, 3),
        "temporal_coords": torch.tensor(
            np.stack([dates.year, dates.dayofyear - 1], axis=-1),
            dtype=torch.float32,
        ),
        "location_coords": torch.tensor([latitude, longitude]),
    }


def preprocess(raw: xr.Dataset) -> xr.Dataset:
    """Change time, grid, bands, and values before deriving model inputs."""
    image = (
        raw[["red"]]
        .isel(time=[1], y=slice(1, 8), x=slice(2, 11))
        .rename({"red": "reflectance"})
    )
    return image.assign(reflectance=image.reflectance / 255)


class LocalCatalog:
    """STAC provider fixture serving two real local raster assets."""

    def __init__(self, metadata, items):
        self.metadata, self.items = metadata, items

    def source(self, collection):
        return StacSource(self, collection=collection)

    def collection(self, collection):
        return self.metadata

    def search(self, query):
        return self.items


def fixture(root: Path) -> tuple[GeoAnchor, LocalCatalog]:
    anchor = GeoAnchor.from_coordinates(
        -7, 110, shape=(10, 12), resolution=10, timespan="2024-01/2024-03"
    )
    grid = anchor.geobox
    bounds = list(grid.geographic_extent.boundingbox)
    dates = [datetime(2024, 1, 1, tzinfo=UTC), datetime(2024, 3, 1, tzinfo=UTC)]
    collection = pystac.Collection(
        "optical",
        "Offline Python probe",
        pystac.Extent(
            pystac.SpatialExtent([bounds]),
            pystac.TemporalExtent([[dates[0], dates[1]]]),
        ),
        license="CC0-1.0",
    )
    items = []
    for index, date in enumerate(dates):
        values = (
            np.arange(grid.width * grid.height, dtype="uint16").reshape(
                tuple(grid.shape)
            )
            * 2
            + index
        )
        item = pystac.Item(
            f"source-{index}",
            grid.geographic_extent.json,
            bounds,
            date,
            {},
            collection="optical",
        )
        ProjectionExtension.ext(item, add_if_missing=True).apply(
            code=str(grid.crs), shape=list(grid.shape), transform=list(grid.transform)
        )
        for band, offset in (("red", 0), ("nir", 100)):
            path = root / f"source-{index}-{band}.tif"
            with rasterio.open(
                path,
                "w",
                driver="COG",
                width=grid.width,
                height=grid.height,
                count=1,
                dtype="uint16",
                crs=str(grid.crs),
                transform=grid.transform,
                nodata=65535,
            ) as output:
                output.write(values + offset, 1)
            item.add_asset(
                band,
                pystac.Asset(
                    str(path),
                    media_type=pystac.MediaType.COG,
                    roles=["data"],
                    extra_fields={
                        "raster:bands": [{"data_type": "uint16", "nodata": 65535}]
                    },
                ),
            )
        items.append(item)
    return anchor, LocalCatalog(collection, items)


def run(root: Path) -> dict[str, object]:
    spec = ModelSpec.model_validate(
        {
            "schema_version": 2,
            "rasters": {
                "raw": {
                    "variables": ["red", "nir"],
                    "require_crs": True,
                    "stac": {
                        "collection": "optical",
                        "endpoints": ["https://offline.test/stac"],
                        "load": {
                            "bands": ["red", "nir"],
                            "groupby": "time",
                            "chunks": {"x": 4, "y": 4},
                        },
                    },
                }
            },
            "preprocessing": {
                "image": {
                    "call": f"{__name__}.preprocess",
                    "kwargs": {"raw": Ref("raw")},
                }
            },
            "tiles": {"size": 4, "overlap": 2, "mode": "edge", "window": "hann"},
            "inputs": {"image": Ref("image")},
        }
    )
    stages = {
        "encoder": {"name": "python_probe"},
        "head": {"name": "segmentation", "init_args": {"classes": ["dark", "bright"]}},
    }
    model = build_model(stages).eval()
    with torch.no_grad():
        model.head.layers[0].weight.copy_(torch.tensor([[[[-1.0]]], [[[1.0]]]]))
        model.head.layers[0].bias.copy_(torch.tensor([0.5, -0.5]))

    # Ingest uses the real recipe, ODC loader, and local GeoTIFFs; no HTTP request.
    anchor, catalog = fixture(root)
    with patch("geosave_engine.model.spec.stac._open_client", return_value=catalog):
        loaded = spec.load_rasters(anchor)
    assert loaded["raw"].red.chunks is not None

    # Persist acquisition separately, then load it for this prediction run.
    loaded["raw"].gs.to_cog(root / "rasters", split_bands=True)
    stored = [item.clone() for item in catalog.items]
    for item in stored:
        timestamp = item.datetime.strftime("%Y%m%dT%H%M%S")
        for band, asset in item.assets.items():
            asset.href = str(root / "rasters" / timestamp / f"{band}.tif")
    # Use the dependency's STAC writer; GeoSave's writer currently conflicts
    # with an existing STAC bbox column when adding a covering bbox.
    parse_stac_items_to_parquet(stored, output_path=root / "catalog.parquet")
    records = io.read_vector(root / "catalog.parquet")
    assert all(set(assets) == {"red", "nir"} for assets in records.assets)

    # Rows supply acquisition time; band files supply pixels and their grid.
    raw = load_stac(
        to_item_collection(records),
        geobox=anchor.geobox,
        bands=["red", "nir"],
        groupby="time",
        chunks={"x": 4, "y": 4},
    )
    xr.testing.assert_equal(raw, loaded["raw"])
    selected = spec.rasters["raw"].select_raster(raw)
    assert selected.red.chunks is not None and selected.red.dtype == raw.red.dtype
    try:
        spec.rasters["raw"].select_raster(raw.drop_vars("nir"))
    except ValueError as error:
        assert "Missing raster variables" in str(error)
    else:
        raise AssertionError("A missing required band must fail compatibility checks")
    dated = records.assign(datetime=records.datetime + pd.Timedelta(days=1))
    redated = load_stac(
        to_item_collection(dated),
        geobox=anchor.geobox,
        bands=["red", "nir"],
        groupby="time",
        chunks={"x": 4, "y": 4},
    )
    np.testing.assert_array_equal(
        redated.time.values, raw.time.values + np.timedelta64(1, "D")
    )
    image = spec.preprocess({"raw": raw})["image"]
    assert list(image.data_vars) == ["reflectance"]
    assert image.sizes["time"] == 1 and image.reflectance.chunks is not None
    assert image.gs.geobox == anchor.geobox.translate_pix(2, 1).crop((7, 9))

    # A spec can invoke the ordinary input function without a model instance.
    input_recipe = CallSpec(
        call=f"{__name__}.prepare_inputs", kwargs={"image": Ref("image")}
    )
    direct = prepare_inputs(image)
    declared = input_recipe.invoke({"image": image})
    for name in direct:
        torch.testing.assert_close(direct[name], declared[name])
    torch.testing.assert_close(
        direct["temporal_coords"], torch.tensor([[2024.0, 60.0]])
    )

    layout = spec.tiles.layout((image.sizes["y"], image.sizes["x"]))
    shape, padding = layout.calculate_padding()
    layout.recalculate(data_shape=shape)
    reference = GeoVector.from_layouts(
        {"scene": image}, {"scene": layout}, padding={"scene": padding}
    ).set_index("id", drop=False, verify_integrity=True)
    merger = Merger(layout, logits=model.head.num_classes, window=spec.tiles.window)

    # Native batching also serves training; IDs remain outside model arguments.
    with torch.inference_mode():
        for rows in batched(reference.iloc[::-1].iterrows(), 3):
            samples = []
            for sample_id, row in rows:
                tile = window.crop(
                    image,
                    (int(row.row_off), int(row.col_off)),
                    (int(row.height), int(row.width)),
                    padding=padding,
                    mode=layout.mode,
                )
                samples.append(prepare_inputs(tile))
            predictions = model(**default_collate(samples))
            for (_, row), prediction in zip(rows, predictions, strict=True):
                merger.add(int(row.tile_id), prediction.numpy())
        merged = torch.from_numpy(merger.merge(extra_padding=padding))
        whole = model(**default_collate([direct]))[0]
        torch.testing.assert_close(merged, whole)
        labels = merged.argmax(dim=-3).numpy().astype("uint8")

    result = array(labels, image.gs.geobox).rename("land_cover")
    path = result.gs.to_cog(root / "prediction.tif")
    restored = io.read_raster(path)
    assert restored.gs.geobox == image.gs.geobox
    np.testing.assert_array_equal(restored.land_cover.values, labels)
    assert set(np.unique(labels)) == {0, 1}

    # The same external input function works with different native heads.
    contracts = {}
    heads = {
        "regression": {"variables": ["biomass"]},
        "classification": {"classes": ["dark", "bright"], "pyramid_channels": [1]},
        "detection": {
            "classes": ["tree"],
            "pyramid_channels": [1],
            "pyramid_strides": [1],
            "hidden_channels": 4,
        },
    }
    with torch.inference_mode():
        for name, arguments in heads.items():
            alternate = build_model(
                {
                    "encoder": stages["encoder"],
                    "head": {"name": name, "init_args": arguments},
                }
            ).eval()
            inputs = prepare_inputs(image)
            output = alternate(**default_collate([inputs]))
            contracts[name] = list(output.shape)
    assert contracts == {
        "regression": [1, 1, 7, 9],
        "classification": [1, 2],
        "detection": [1, 63, 5],
    }
    return {
        "stac_items": 2,
        "cog_band_assets": 4,
        "catalog_datetime_controls_time": "passed",
        "catalog_and_acquired_rasters": "equal",
        "raw_compatibility_checks": "passed",
        "prepared_shape": list(labels.shape),
        "tiles": len(reference),
        "actual_metadata_inputs": "passed",
        "direct_and_declared_inputs": "equal",
        "shuffled_tiled_and_full_logits": "equal",
        "geotiff_round_trip": "passed",
        "batch_size": 3,
        "native_head_contracts": contracts,
        "prefect_imported": __import__("sys").modules.get("prefect") is not None,
    }


if __name__ == "__main__":
    with TemporaryDirectory(prefix="geosave-python-flow-") as directory:
        print(json.dumps(run(Path(directory)), indent=2))
