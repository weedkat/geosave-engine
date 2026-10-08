from __future__ import annotations

import numpy as np
import pytest
import xarray as xr
from odc.geo.geobox import GeoBox
from odc.geo.xr import xr_coords

from geosave_engine.geodata import cuts
from geosave_engine.geodata.core.stack import stack
from geosave_engine.ml.inputs import model_inputs, to_tensor
from torch.utils.data import Dataset
from tiler import Merger

from tests.geodata.conftest import whole_windows

torch = pytest.importorskip("torch")


@pytest.mark.parametrize("kind", ["array", "dataset", "tree"])
@pytest.mark.parametrize("dtype", [None, "float32", torch.bfloat16])
def test_native_rasters_convert_with_explicit_tensor_dtype(kind, dtype):
    source = _optical().isel(time=0)
    data = source.B04 if kind == "array" else source
    if kind == "tree":
        data = stack({"optical": source})

    result = to_tensor(data, dtype=dtype)
    if kind == "tree":
        assert tuple(result) == ("optical",)
        result = result["optical"]
    expected_dtype = (
        torch.uint16
        if dtype is None
        else torch.bfloat16
        if dtype is torch.bfloat16
        else torch.float32
    )
    assert result.dtype == expected_dtype
    assert result.shape == ((8, 8) if kind == "array" else (2, 8, 8))
    assert result.is_contiguous()
    expected = source.B04 if kind == "array" else source
    torch.testing.assert_close(
        result, torch.tensor(expected.gs.to_numpy(), dtype=expected_dtype)
    )


@pytest.mark.parametrize("dtype", [None, "float32"])
def test_read_only_pixels_convert_to_a_safely_mutable_tensor(dtype):
    source = _raster(2, 2).B04
    source.values.flags.writeable = False
    before = source.values.copy()

    result = to_tensor(source, dtype=dtype)
    result.fill_(0)

    np.testing.assert_array_equal(source.values, before)
    assert torch.count_nonzero(result) == 0


def test_tensor_conversion_rejects_unknown_dtype():
    with pytest.raises(ValueError, match="Unknown torch dtype"):
        to_tensor(_raster(8, 8), dtype="unknown")


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
    assert index == "scene-a/chip-0"


@pytest.mark.parametrize("window", ["boxcar", "hann"])
def test_a_small_model_and_shuffled_native_merge_rebuild_each_parent(window):
    from torch.utils.data import DataLoader

    parents = {"a": _raster(13, 19, seed=1), "b": _raster(13, 19, seed=2)}
    samples = _samples(parents, (6, 8), overlap=2)
    mergers = {
        key: Merger(layout, logits=3, window=window, save_visits=False)
        for key, layout in samples.tilers.items()
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
                mergers[row.parent].add(int(row.chip), output)
        for key, parent in parents.items():
            actual = mergers[key].merge(extra_padding=samples.padding[key])
            expected = model(to_tensor(parent)[None])[0].numpy()
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
                assert row.parent in ("a", "b")
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
    # Overlapping chips sit over a halo, filled the way the fringe is.
    native, halo = cuts.layout((13, 19), (6, 8), overlap=2, mode=mode)
    fill = {"constant_values": np.nan} if mode == "constant" else {}
    padded = np.pad(to_tensor(parent).numpy(), [(0, 0), *halo], mode=mode, **fill)
    expected = np.stack([native.get_tile(band, position) for band in padded])
    np.testing.assert_array_equal(actual["image"].numpy(), expected)
    assert tasks
    # Only the blocks under the last chip were read; a wrapped halo reads the far edge too.
    if mode != "wrap":
        assert all(location[-2][0] >= 4 and location[-1][0] >= 5 for location in tasks)


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
    samples = _samples({"a": parent}, (6, 8), overlap=2, spec=spec)
    inputs, sample_id = samples[-1]
    assert inputs["image"].shape == (2, 2, 6, 8)
    assert inputs["height"].shape == (1, 6, 8)
    tile = samples.read(-1)
    row = samples.reference.loc[sample_id]
    expected = parent.gs.geobox.translate_pix(row.col_off, row.row_off).crop((6, 8))
    assert tile.gs.geobox == expected


class Samples(Dataset):
    """Consumer-owned Dataset exercising the public read and input APIs."""

    def __init__(self, parents, reference, tilers, padding, spec=None):
        self.parents, self.tilers, self.padding = parents, tilers, padding
        self.spec = spec
        self.reference = reference.set_index("id", drop=False)

    def __len__(self):
        return len(self.reference)

    def read(self, position):
        row = self.reference.iloc[position]
        return cuts.select_pixels(self.parents[row.parent], row.to_dict())

    def __getitem__(self, position):
        row = self.reference.iloc[position]
        tile = self.read(position)
        inputs = (
            {"image": to_tensor(tile)}
            if self.spec is None
            else model_inputs(self.spec, tile.gs.rasters, row)
        )
        return inputs, row.id


def _samples(parents, shape, *, overlap=0, mode="reflect", spec=None):
    reference = cuts.chips(whole_windows(parents), shape, overlap=overlap, mode=mode)
    tilers, padding = {}, {}
    for key, parent in parents.items():
        tilers[key], padding[key] = cuts.layout(
            (parent.sizes["y"], parent.sizes["x"]), shape, overlap=overlap, mode=mode
        )
    return Samples(parents, reference, tilers, padding, spec)


@pytest.mark.parametrize("kind", ["array", "dataset", "stack"])
def test_a_constant_halo_is_filled_with_missing_values(kind):
    source = _raster(6, 8)[["B04"]]
    parent = (
        source.B04
        if kind == "array"
        else stack({"image": source})
        if kind == "stack"
        else source
    )
    samples = _samples({"a": parent}, (4, 4), overlap=2, mode="constant")
    layout, padding = samples.tilers["a"], samples.padding["a"]
    padded = np.pad(source.B04.values, padding, mode="constant", constant_values=np.nan)
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


def _optical() -> xr.Dataset:
    """Build two dated bands with distinct stored integer values."""
    source = _raster(8, 8)
    source = source.assign(
        B04=xr.full_like(source.B04, 4, dtype="uint16"),
        B08=xr.full_like(source.B08, 8, dtype="uint16"),
    )
    return source.expand_dims(
        time=np.array(["2024-01-01", "2024-02-01"], dtype="datetime64[ns]")
    )


def test_to_tensor_preserves_the_prepared_dtype_by_default() -> None:
    tensor = to_tensor(_optical())

    assert tensor.dtype is torch.uint16
    assert tuple(tensor.shape) == (2, 2, 8, 8)


def test_to_tensor_honours_a_requested_dtype() -> None:
    assert to_tensor(_optical(), dtype=torch.int16).dtype is torch.int16


def test_to_tensor_accepts_a_yaml_dtype_name() -> None:
    tensor = to_tensor(_optical(), dtype="float32")

    assert tensor.dtype is torch.float32


def test_to_tensor_rejects_an_unknown_dtype_name() -> None:
    with pytest.raises(ValueError, match="not-a-dtype"):
        to_tensor(_optical(), dtype="not-a-dtype")


def test_a_stack_preserves_and_explicitly_casts_group_dtypes() -> None:
    scene = stack({"optical": _optical()})

    assert to_tensor(scene)["optical"].dtype is torch.uint16
    assert to_tensor(scene, dtype="float32")["optical"].dtype is torch.float32


def test_tensor_conversion_places_spatial_axes_last():
    scrambled = xr.DataArray(np.zeros((8, 8, 3), "uint16"), dims=("y", "x", "band"))
    assert to_tensor(scrambled).shape == (3, 8, 8)


def test_tensor_conversion_of_one_band_preserves_or_casts_dtype():
    band = _optical().B04
    assert to_tensor(band).dtype is torch.uint16
    assert to_tensor(band, dtype="float32").dtype is torch.float32
    assert to_tensor(band, dtype=torch.bfloat16).dtype is torch.bfloat16
