"""A reusable geospatial observation flow through real local assets and storage."""

from datetime import UTC, datetime

import dask.array as da
import numpy as np
import pytest

from geosave_engine.geodata import read_raster
from geosave_engine.geodata.attrs import CFVariable, Legend, Spectral
from geosave_engine.geodata.attrs.headers.stac import band_attrs, read_bands
from geosave_engine.geodata.features import ndvi
from geosave_engine.geodata.stac import create_items, table

from tests.geodata.conftest import build_raster


def test_described_rasters_round_trip_through_the_table(tmp_path) -> None:
    optical = build_raster(times=2, packed=True)[["red"]]
    optical = optical.gs.rebase(
        CFVariable(units="1"),
        Spectral(common_name="red", center_wavelength=0.665),
        target="red",
    )
    label = build_raster()[["nir"]].astype("uint8").rename(nir="label")
    label = label.gs.rebase(
        Legend(class_map={0: "background", 1: "forest"}, color_map={1: "#00ff00"}),
        target="label",
    )
    when = datetime(2025, 6, 1, tzinfo=UTC)
    items = [
        *create_items(optical.gs.to_cog(tmp_path / "optical")),
        *create_items(label.gs.to_cog(tmp_path / "label"), datetime=when),
    ]

    saved = table.write(items, tmp_path / "items.parquet")

    restored = table.to_items(table.read(saved))
    red, spectral = read_bands(restored[0].assets["red"])[0]
    classes, named = read_bands(restored[-1].assets["label"])[0]
    assert red.data_type == "uint16"
    assert band_attrs(red, spectral) == {
        "units": "1",
        "long_name": "red",
        "scale_factor": pytest.approx(1e-4),
        "common_name": "red",
        "center_wavelength": 0.665,
    }
    legend = Legend.from_attrs(band_attrs(classes, named))
    assert legend.class_map == {0: "background", 1: "forest"}


@pytest.mark.integration
def test_described_items_validate_against_their_schemas(tmp_path) -> None:
    scene = build_raster(times=1, packed=True)

    (item,) = create_items(scene.gs.to_cog(tmp_path / "scene"))

    item.validate()


def test_observations_to_prepared_index_to_reusable_dataset(local_source, tmp_path):
    source, anchor = local_source
    raw = source.load(anchor)
    reflectance = raw.gs.mask_and_scale()
    derived = reflectance.assign(ndvi=ndvi(reflectance, nir="nir", red="red"))[["ndvi"]]
    composite = derived.median("time", keep_attrs=True)
    assert isinstance(composite.ndvi.data, da.Array)
    assert composite.gs.geobox == anchor.geobox
    path = composite.gs.to_zarr(tmp_path / "ndvi.zarr")

    with read_raster(path, chunks={}) as reopened:
        assert isinstance(reopened.ndvi.data, da.Array)
        assert reopened.gs.geobox == anchor.geobox
        assert reopened.attrs["stac_items"][0]["id"] == "scene"
        assert np.isnan(reopened.ndvi.values[0, 0])
        assert reopened.ndvi.values[1, 1] == pytest.approx((1.1 - 0.2) / (1.1 + 0.2))
        assert "scale_factor" not in reopened.ndvi.attrs
    # Preparation leaves the reusable source configuration and packed input intact.
    assert raw.red.attrs["scale_factor"] == 0.0001
    assert raw.red.values[0, 1, 1] == 2000
