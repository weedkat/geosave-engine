from __future__ import annotations

import numpy as np
import pytest
import xarray as xr
from odc.geo.geobox import GeoBox
from odc.geo.xr import xr_coords

from geosave_engine.geodata.core.stack import stack
from geosave_engine.geodata.datasets import TileDataset
from geosave_engine.geodata.transform.tiling import Tiles

torch = pytest.importorskip("torch")


def _raster(height: int, width: int, seed: int = 0) -> xr.Dataset:
    grid = GeoBox.from_bbox(
        (0, 0, width * 10, height * 10), "EPSG:32748", resolution=10
    )
    rng = np.random.default_rng(seed)
    return xr.Dataset(
        {
            "B04": (("y", "x"), rng.random((height, width)).astype("float32")),
            "B08": (("y", "x"), rng.random((height, width)).astype("float32")),
        },
        coords=xr_coords(grid),
    )


def test_a_sample_states_the_number_it_was_asked_by() -> None:
    tiles = Tiles([_raster(600, 600)], (256, 256), overlap=32)
    samples = TileDataset(tiles)

    assert len(samples) == len(tiles)
    assert samples[3]["index"] == 3
    assert samples[3]["image"].shape == (2, 256, 256)


def test_variables_are_read_in_the_order_they_were_cut_in() -> None:
    source = _raster(600, 600)

    forward = TileDataset(Tiles([source[["B04", "B08"]]], (256, 256)))[0]["image"]
    reversed_ = TileDataset(Tiles([source[["B08", "B04"]]], (256, 256)))[0]["image"]

    assert torch.equal(forward[0], reversed_[1])
    assert torch.equal(forward[1], reversed_[0])


def test_a_stack_reads_one_tensor_per_group() -> None:
    grouped = stack({"optical": _raster(600, 600), "dem": _raster(600, 600, seed=1)})
    samples = TileDataset(Tiles([grouped], (256, 256), overlap=32))

    image = samples[0]["image"]
    assert sorted(image) == ["dem", "optical"]
    assert image["optical"].shape == (2, 256, 256)


def test_a_single_band_reads_as_itself() -> None:
    samples = TileDataset(Tiles([_raster(600, 600).B04], (256, 256)))

    assert samples[0]["image"].shape == (256, 256)
    assert samples[0]["index"] == 0


def test_a_batch_routes_every_result_home() -> None:
    from torch.utils.data import DataLoader

    rasters = [_raster(600, 600, seed=1), _raster(512, 300, seed=2)]
    tiles = Tiles(rasters, (256, 256), overlap=32)
    merger = tiles.merger(window="hann")

    for batch in DataLoader(TileDataset(tiles), batch_size=8):
        bands = batch["image"].mean(dim=1, keepdim=True).repeat(1, 3, 1, 1)
        merger.add(dict(zip(batch["index"].tolist(), bands.numpy(), strict=True)))

    rebuilt = merger.merge()
    assert merger.pending == {}
    for ordinal, source in enumerate(rasters):
        assert rebuilt[ordinal].dims == ("band", "y", "x")
        assert rebuilt[ordinal].shape == (3, *source.B04.shape)
        assert rebuilt[ordinal].odc.geobox == source.odc.geobox
