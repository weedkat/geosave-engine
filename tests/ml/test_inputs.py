from __future__ import annotations

import numpy as np
import pytest
import xarray as xr
from odc.geo.geobox import GeoBox
from odc.geo.xr import xr_coords

from geosave_engine.geodata.core.stack import stack
from geosave_engine.geodata import GeoVector
from geosave_engine.ml.inputs import model_inputs
from torch.utils.data import Dataset
from tiler import Merger, Tiler

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


def test_variables_are_read_in_the_order_they_were_cut_in() -> None:
    source = _raster(600, 600)

    forward, _ = _samples({"scene-a": source[["B04", "B08"]]}, (256, 256))[0]
    reversed_, _ = _samples({"scene-a": source[["B08", "B04"]]}, (256, 256))[0]

    assert torch.equal(forward["image"][0], reversed_["image"][1])
    assert torch.equal(forward["image"][1], reversed_["image"][0])


def test_a_stack_reads_one_tensor_per_group() -> None:
    grouped = stack({"optical": _raster(600, 600), "dem": _raster(600, 600, seed=1)})
    samples = _samples({"scene-a": grouped}, (256, 256), overlap=32)

    inputs, _ = samples[0]
    image = inputs["image"]
    assert sorted(image) == ["dem", "optical"]
    assert image["optical"].shape == (2, 256, 256)


def test_a_single_band_reads_as_itself() -> None:
    samples = _samples({"scene-a": _raster(600, 600).B04}, (256, 256))

    inputs, index = samples[0]
    assert inputs["image"].shape == (256, 256)
    assert index == "scene-a/tile-0"


@pytest.mark.parametrize("window", ["boxcar", "hann"])
def test_a_small_model_and_shuffled_native_merge_rebuild_each_parent(window):
    from torch.utils.data import DataLoader

    parents = {"a": _raster(13, 19, seed=1), "b": _raster(13, 19, seed=2)}
    samples = _samples(parents, (6, 8), overlap=2, halo=window == "hann")
    mergers = {
        key: Merger(layout, logits=3, window=window, save_visits=False)
        for key, layout in samples.layouts.items()
    }
    model = torch.nn.Conv2d(2, 3, 1).eval()
    with torch.inference_mode():
        for inputs, ids in DataLoader(
            samples,
            batch_size=3,
            shuffle=True,
            generator=torch.Generator().manual_seed(7),
        ):
            for sample_id, output in zip(
                ids, model(inputs["image"]).numpy(), strict=True
            ):
                row = samples.reference.loc[sample_id]
                mergers[row.parent_id].add(int(row.tile_id), output)
        for key, parent in parents.items():
            actual = mergers[key].merge(extra_padding=samples.padding[key])
            expected = model(parent.gs.to_tensor()[None])[0].numpy()
            np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-6)


def test_reference_keys_associate_native_outputs_after_shuffle(tmp_path) -> None:
    from torch import nn
    from torch.utils.data import DataLoader

    from geosave_engine.geodata.io import geoparquet
    from geosave_engine.model.chain import ModelChain, chain_step

    # Equal footprints cannot identify which parent supplied a prediction.
    sources = [_raster(12, 16, seed=1), _raster(12, 16, seed=2)]
    samples = _samples(dict(zip(["a", "b"], sources, strict=True)), (6, 8), overlap=2)
    reference = samples.reference
    path = geoparquet.write(reference, tmp_path / "reference.parquet", index=False)
    lookup = (
        geoparquet.read(path)
        .sample(frac=1, random_state=41)
        .set_index("id", verify_integrity=True)
    )

    class Classifier(nn.Module):
        @chain_step(head=True)
        def forward(self, image: torch.Tensor) -> torch.Tensor:
            return image.mean(dim=(-2, -1))

    class Detector(nn.Module):
        @chain_step(head=True)
        def forward(self, image: torch.Tensor) -> list[dict[str, torch.Tensor]]:
            return [
                {
                    "boxes": sample.new_tensor([[1, 1, 3, 3]]),
                    "scores": sample.mean().reshape(1),
                }
                for sample in image
            ]

    model = ModelChain(category=Classifier(), objects=Detector())
    observed = set()
    loader = DataLoader(
        samples, batch_size=3, shuffle=True, generator=torch.Generator().manual_seed(41)
    )
    with torch.inference_mode():
        for inputs, ids in loader:
            outputs = model(**inputs)
            rows = lookup.loc[list(ids)]
            assert len(outputs["objects"]) == len(ids)
            for tile_id, (_, row), scores, detection in zip(
                ids,
                rows.iterrows(),
                outputs["category"],
                outputs["objects"],
                strict=True,
            ):
                position = list(reference.id).index(tile_id)
                expected = samples[position][0]["image"]
                torch.testing.assert_close(scores, expected.mean(dim=(-2, -1)))
                torch.testing.assert_close(
                    detection["scores"], expected.mean().reshape(1)
                )
                box = detection["boxes"][0] + torch.tensor(
                    [row.col_off, row.row_off, row.col_off, row.row_off]
                )
                assert box.tolist() == [
                    row.col_off + 1,
                    row.row_off + 1,
                    row.col_off + 3,
                    row.row_off + 3,
                ]
                assert row.parent_id in ("a", "b")
                assert row.geometry is not None
                observed.add(tile_id)
    assert observed == set(reference.id)


def test_reference_identity_survives_parent_reordering():
    sources = [_raster(12, 16, seed=1), _raster(12, 16, seed=2)]
    samples = _samples(dict(zip(["a", "b"], sources, strict=True)), (6, 8))
    reordered = _samples(dict(zip(["b", "a"], sources[::-1], strict=True)), (6, 8))
    lookup = {reordered[i][1]: reordered[i][0]["image"] for i in range(len(reordered))}
    for i in range(len(samples)):
        inputs, key = samples[i]
        torch.testing.assert_close(inputs["image"], lookup[key])
    assert set(samples.reference.id) == set(reordered.reference.id)


@pytest.mark.parametrize("mode", ["reflect", "edge", "wrap", "constant"])
def test_lazy_reader_matches_native_fringe_padding_and_bounds(mode):
    import dask.array as da
    from dask.callbacks import Callback

    parent = _raster(13, 19)
    tasks = []

    def block(data, block_info=None):
        if block_info is not None:
            tasks.append(block_info[None]["array-location"])
        return data

    lazy = parent.copy()
    for name in parent.data_vars:
        lazy[name].data = da.from_array(parent[name].values, chunks=(4, 5)).map_blocks(
            block, dtype="float32"
        )
    with Callback(pretask=lambda *_: pytest.fail("dataset construction read pixels")):
        samples = _samples({"a": lazy}, (6, 8), overlap=2, mode=mode)
    position = len(samples) - 1
    actual, _ = samples[position]
    native = Tiler((2, 13, 19), (2, 6, 8), overlap=2, channel_dimension=0, mode=mode)
    expected = native.get_tile(parent.gs.to_tensor().numpy(), position)
    np.testing.assert_array_equal(actual["image"].numpy(), expected)
    assert tasks
    assert all(location[-2][0] >= 8 and location[-1][0] >= 10 for location in tasks)


def test_named_inputs_preserve_different_leading_axes_and_tile_grid():
    from geosave_engine.model.spec import ModelSpec, Ref

    optical = _raster(13, 19).expand_dims(
        time=np.array(["2025-01-01", "2025-01-02"], dtype="datetime64[ns]")
    )
    dem = _raster(13, 19)[["B04"]]
    parent = stack({"optical": optical, "dem": dem})
    spec = ModelSpec.model_validate(
        {
            "schema_version": 2,
            "rasters": {
                "optical": {"variables": ["B04", "B08"]},
                "dem": {"variables": ["B04"]},
            },
            "inputs": {"image": Ref("optical"), "height": Ref("dem")},
        }
    )
    samples = _samples({"a": parent}, (6, 8), overlap=2, halo=True, spec=spec)
    inputs, sample_id = samples[-1]
    assert inputs["image"].shape == (2, 2, 6, 8)
    assert inputs["height"].shape == (1, 6, 8)
    tile = samples.read(-1)
    row = samples.reference.loc[sample_id]
    expected = parent.gs.geobox.translate_pix(row.col_off, row.row_off).crop((6, 8))
    assert tile.gs.geobox == expected


class Samples(Dataset):
    """Consumer-owned Dataset exercising the public read and input APIs."""

    def __init__(self, parents, layouts, spec=None, *, padding=None):
        self.parents, self.layouts, self.spec = parents, layouts, spec
        self.padding = padding or {key: [(0, 0), (0, 0)] for key in parents}
        self.reference = GeoVector.from_layouts(
            parents, layouts, padding=self.padding
        ).set_index("id", drop=False)

    def __len__(self):
        return len(self.reference)

    def read(self, position):
        row = self.reference.iloc[position]
        return row.gs.crop(self.parents[row.parent_id])

    def __getitem__(self, position):
        row = self.reference.iloc[position]
        tile = self.read(position)
        inputs = (
            {"image": tile.gs.to_tensor()}
            if self.spec is None
            else model_inputs(self.spec, tile.gs.rasters, row)
        )
        return inputs, row.id


def _samples(parents, shape, *, overlap=0, mode="reflect", halo=False, spec=None):
    layouts, padding = {}, {}
    for key, parent in parents.items():
        spatial = parent.gs.grid_dims
        layout = Tiler(
            tuple(parent.sizes[d] for d in spatial), shape, overlap=overlap, mode=mode
        )
        padding[key] = [(0, 0), (0, 0)]
        if halo:
            padded_shape, padding[key] = layout.calculate_padding()
            layout.recalculate(data_shape=padded_shape)
        layouts[key] = layout
    return Samples(parents, layouts, spec, padding=padding)


@pytest.mark.parametrize("kind", ["array", "dataset", "stack"])
@pytest.mark.parametrize("fill", [0.0, 9.0, float("nan")])
def test_explicit_halo_uses_native_constant_value(kind, fill):
    source = _raster(6, 8)[["B04"]]
    parent = (
        source.B04
        if kind == "array"
        else stack({"image": source})
        if kind == "stack"
        else source
    )
    layout = Tiler((6, 8), (4, 4), overlap=2, mode="constant", constant_value=fill)
    shape, padding = layout.calculate_padding()
    layout.recalculate(data_shape=shape)
    samples = Samples({"a": parent}, {"a": layout}, padding={"a": padding})
    padded = np.pad(source.B04.values, padding, mode="constant", constant_values=fill)
    for index in (0, len(samples) - 1):
        tile = samples.read(index)
        actual = (
            tile["image"].dataset.B04.values
            if isinstance(tile, xr.DataTree)
            else tile.B04.values
            if isinstance(tile, xr.Dataset)
            else tile.values
        )
        expected = layout.get_tile(padded, index)
        np.testing.assert_allclose(actual, expected, equal_nan=True)
