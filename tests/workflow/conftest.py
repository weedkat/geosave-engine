from dataclasses import dataclass, field
from datetime import datetime, timezone

import dask.array as da
import numpy as np
from odc.geo.geobox import GeoBox
import pytest
import pystac
from pystac.extensions.projection import ProjectionExtension
import rasterio

from geosave_engine.geodata.core.anchor import GeoAnchor
from geosave_engine.geodata.core.raster import raster
from geosave_engine.geodata.stac.query import StacQuery
from geosave_engine.geodata.stac.source import StacSource
from geosave_engine.workflow.specs import ModelSpec, RasterRequirement


@pytest.fixture
def spec():
    return ModelSpec(
        schema_version=2,
        sources={
            "optical": RasterRequirement(variables=("nir", "red"), require_crs=True)
        },
    )


@pytest.fixture
def raw():
    box = GeoBox.from_bbox((0, 0, 40, 40), crs="EPSG:32633", resolution=10)
    return {
        "optical": raster(
            {
                name: da.full((4, 4), value, chunks=(2, 2))
                for name, value in (("red", 2.0), ("nir", 6.0), ("unused", 0.0))
            },
            box,
        )
    }


@pytest.fixture
def anchor(raw):
    return GeoAnchor(geobox=raw["optical"].gs.geobox)


@pytest.fixture
def source():
    # These unit tests replace acquisition; constructing a source performs no I/O.
    return StacSource(client=None, collection="example").set_config(
        bands=("unused",), chunks={"x": 2, "y": 2}, item_properties=("platform",)
    )


@dataclass
class LocalCatalog:
    """Replace only catalog search; assets and the raster loading path are real."""

    metadata: pystac.Collection
    items: list[pystac.Item]
    requests: list[StacQuery] = field(default_factory=list)

    def search(self, query):
        self.requests.append(query)
        return self.items if query.collections == [self.metadata.id] else []

    def collection(self, collection):
        assert collection == self.metadata.id
        return self.metadata


@pytest.fixture
def local_stac(tmp_path):
    anchor = GeoAnchor.from_coordinates(
        45.0,
        12.0,
        shape=4,
        crs="EPSG:32633",
        resolution=10,
        timespan="2025-01",
    )
    box = anchor.geobox
    bounds = list(box.geographic_extent.boundingbox)
    timestamp = datetime(2025, 1, 15, 12, tzinfo=timezone.utc)
    sources = {}
    for name, bands in (
        ("optical", {"red": 2000, "nir": 6000, "unused": 1}),
        ("terrain", {"height": 30}),
    ):
        collection = pystac.Collection(
            id=name,
            description="Local raster smoke test",
            license="CC-BY-4.0",
            extent=pystac.Extent(
                pystac.SpatialExtent([bounds]),
                pystac.TemporalExtent([[timestamp, timestamp]]),
            ),
        )
        item = pystac.Item(
            id=f"{name}-scene",
            geometry=box.geographic_extent.json,
            bbox=bounds,
            datetime=timestamp,
            properties={"platform": "local-sample"},
            collection=name,
        )
        ProjectionExtension.ext(item, add_if_missing=True).apply(
            code="EPSG:32633",
            shape=list(box.shape),
            transform=list(box.transform),
        )
        for band, value in bands.items():
            path = tmp_path / f"{name}-{band}.tif"
            pixels = np.full(tuple(box.shape), value, dtype="uint16")
            pixels[0, 0] = 0
            with rasterio.open(
                path,
                "w",
                driver="GTiff",
                width=box.width,
                height=box.height,
                count=1,
                dtype="uint16",
                crs="EPSG:32633",
                transform=box.transform,
                nodata=0,
            ) as destination:
                destination.write(pixels, 1)
            band_metadata = {"data_type": "uint16", "nodata": 0, "unit": "m"}
            if name == "optical":
                band_metadata.update(scale=0.0001, offset=0, unit="1")
            item.add_asset(
                band,
                pystac.Asset(
                    href=str(path),
                    media_type=pystac.MediaType.GEOTIFF,
                    roles=["data"],
                    extra_fields={"raster:bands": [band_metadata]},
                ),
            )
        sources[name] = (
            StacSource(LocalCatalog(collection, [item]), collection=name)
            .set_config(
                bands=("unused",) if name == "optical" else ("height",),
                groupby="time",
                chunks={"x": 2, "y": 2},
                item_properties=("platform",),
            )
            .set_query(
                datetime="2025-01-15", max_items=1, filter="eo:cloud_cover <= 10"
            )
        )
    return sources, anchor
