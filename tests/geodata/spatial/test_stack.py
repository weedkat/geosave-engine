"""GeoStack and GeoTileStack — named layers over one grid, at two sizes.

The split mirrors GeoRaster/GeoTile on one axis: GeoStack is unbounded and
owns the disk boundary, GeoTileStack is bounded and owns materialization. The
reference layer's anchor is the stack's identity, so anything that could
move it silently is an error.
"""
from __future__ import annotations

import numpy as np
import pytest
import torch

from geosave_engine.geodata.spatial import GeoTileStack, GeoStack

from .conftest import is_lazy, make_anchor, make_lazy_raster, make_raster, make_vector


def image(**kwargs):
    return make_raster(bands=("B04", "B08"), **kwargs)


def label(**kwargs):
    return make_raster(bands=("cls",), dtype="uint8", nodata=255, **kwargs)


class TestConstruction:
    def test_keyword_layers(self):
        assert list(GeoStack(image=image(), label=label())) == ["image", "label"]

    def test_mapping_positionally(self):
        assert list(GeoStack({"image": image(), "label": label()})) == ["image", "label"]

    def test_extending_an_existing_stack(self):
        base = GeoStack(image=image())
        assert list(GeoStack(base, label=label())) == ["image", "label"]

    def test_mapping_plus_keywords(self):
        assert list(GeoStack({"image": image()}, label=label())) == ["image", "label"]

    def test_extending_keeps_the_base_reference_layer(self):
        base = GeoStack(image=image(), label=label(), reference_layer="label")
        assert GeoStack(base, extra=label()).reference_layer == "label"

    def test_a_keyword_layer_replaces_a_base_layer_of_the_same_name(self):
        base = GeoStack(image=image())
        replaced = GeoStack(base, image=label())
        assert replaced["image"].bands == ("cls",)

    def test_there_is_one_constructor(self):
        assert not hasattr(GeoStack, "from_layers")
        assert not hasattr(GeoTileStack, "from_layers")

    def test_at_least_one_layer_is_required(self):
        with pytest.raises(ValueError, match="at least one layer"):
            GeoStack()

    def test_a_missing_reference_layer_is_rejected(self):
        with pytest.raises(ValueError, match="isn't present"):
            GeoStack(image=image(), reference_layer="nope")


class TestStrictGrid:
    def test_layers_must_share_one_geobox(self):
        with pytest.raises(ValueError, match="isn't on reference layer"):
            GeoStack(image=image(), dem=make_raster(bands=("dem",), resolution=20, width=32, height=32))

    def test_the_error_names_the_fix(self):
        with pytest.raises(ValueError, match=r"reproject_like\(image\)"):
            GeoStack(image=image(), dem=make_raster(bands=("dem",), width=32, height=32))

    def test_aligning_first_makes_it_valid(self):
        target = image()
        dem = make_raster(bands=("dem",), resolution=20, width=32, height=32, dtype="int16", nodata=-9999)
        assert list(GeoStack(image=target, dem=dem.reproject_like(target))) == ["image", "dem"]

    def test_layer_types_are_checked_before_any_pixel_access(self):
        """A non-layer must give TypeError, not AttributeError from reading .anchor."""
        with pytest.raises(TypeError, match="GeoStack holds GeoRaster layers"):
            GeoStack(image=object())

    def test_a_tile_is_not_a_raster_layer(self):
        tile = next(iter(image().tiles(tile_size_px=32)))
        with pytest.raises(TypeError):
            GeoStack(image=tile)


class TestIdentity:
    def test_anchor_is_the_reference_layers_own(self):
        reference = image()
        assert GeoStack(image=reference, label=label()).anchor is reference.anchor

    def test_vector_comes_from_the_reference_layer(self):
        stack = GeoStack(image=image(vector=make_vector()), label=label())
        assert stack.vector is not None and len(stack.vector) == 1

    def test_a_non_reference_layers_vector_is_not_consulted(self):
        """StacSource puts the anchor's vector on every layer, so this must not error."""
        stack = GeoStack(image=image(), label=label(vector=make_vector()))
        assert stack.vector is None

    def test_select_keeps_the_named_layers_in_order(self):
        stack = GeoStack(image=image(), label=label(), extra=label())
        assert list(stack.select("image", "extra")) == ["image", "extra"]

    def test_select_cannot_drop_the_reference_layer(self):
        stack = GeoStack(image=image(), label=label())
        with pytest.raises(ValueError, match="must keep reference layer"):
            stack.select("label")

    def test_select_rejects_an_unknown_layer(self):
        with pytest.raises(KeyError):
            GeoStack(image=image()).select("image", "nope")


class TestSizeSplit:
    def test_stack_cannot_materialize(self):
        assert not hasattr(GeoStack, "to_numpy")
        assert not hasattr(GeoStack, "to_tensor")

    def test_sample_can(self):
        assert hasattr(GeoTileStack, "to_numpy")
        assert hasattr(GeoTileStack, "to_tensor")

    def test_sample_cannot_write(self):
        assert not hasattr(GeoTileStack, "to_zarr")

    def test_promotion_round_trips(self):
        stack = GeoStack(image=image(), label=label())
        assert list(stack.to_sample().to_stack()) == ["image", "label"]

    def test_crop_returns_a_stack_not_a_sample(self):
        window = make_anchor(width=16, height=16).geobox
        assert isinstance(GeoStack(image=image()).crop(window), GeoStack)

    def test_crop_stays_lazy(self):
        window = make_anchor(width=16, height=16).geobox
        assert is_lazy(GeoStack(image=make_lazy_raster()).crop(window)["image"])


class TestTiles:
    def test_yields_samples(self):
        stack = GeoStack(image=image(), label=label())
        assert isinstance(next(iter(stack.tiles(32))), GeoTileStack)

    def test_every_layer_shares_one_tiling_stamp(self):
        stack = GeoStack(image=image(), label=label())
        sample = next(iter(stack.tiles(32)))
        assert len({sample[name].tiling.group_id for name in sample}) == 1

    def test_layer_names_carry_through(self):
        stack = GeoStack(image=image(), label=label())
        assert list(next(iter(stack.tiles(32)))) == ["image", "label"]

    def test_only_the_reference_layer_carries_features(self):
        stack = GeoStack(image=image(vector=make_vector()), label=label())
        sample = next(iter(stack.tiles(32)))
        assert sample["label"].vector is None

    def test_context_is_computed_per_window_not_per_surface(self):
        """A surface-level context would stamp the same centroid on every tile."""
        stack = GeoStack(image=image())
        seen = [
            sample.model_context["lat"]
            for sample in stack.tiles(32, context_fn=lambda a: {"lat": a.geographic_centroid[1]})
        ]
        assert len(set(seen)) == len(seen) > 1

    def test_without_a_context_fn_the_sample_carries_none(self):
        assert next(iter(GeoStack(image=image()).tiles(32))).model_context is None


class TestModelContext:
    @pytest.fixture
    def sample(self) -> GeoTileStack:
        return next(iter(GeoStack(image=image()).tiles(32)))

    def test_dtypes_survive_to_tensor(self, sample):
        context = {
            "f64": np.array([1.5, 2.5], dtype="float64"),
            "i8": np.array([1, 2], dtype="int8"),
        }
        rendered = GeoTileStack(dict(sample.items()), model_context=context).to_sample()["model_context"]
        assert rendered["f64"].dtype is torch.float64
        assert rendered["i8"].dtype is torch.int8

    def test_a_tensor_cannot_be_stored(self, sample):
        """Context has to serialize to ride beside its tile, and a tensor cannot."""
        with pytest.raises(ValueError, match="A tensor cannot be stored"):
            GeoTileStack(dict(sample.items()), model_context={"emb": torch.ones(2)})

    def test_dtypes_survive_to_numpy(self, sample):
        context = {"u8": np.array([1, 2], dtype="uint8"), "f32": np.array([1.5], dtype="float32")}
        rendered = GeoTileStack(dict(sample.items()), model_context=context).to_sample(mode="numpy")["model_context"]
        assert rendered["u8"].dtype == np.dtype("uint8")
        assert rendered["f32"].dtype == np.dtype("float32")

    def test_strings_and_none_pass_through(self, sample):
        rendered = GeoTileStack(
            dict(sample.items()), model_context={"name": "clay", "missing": None}
        ).to_sample()["model_context"]
        assert rendered["name"] == "clay" and rendered["missing"] is None

    def test_a_value_no_model_could_read_is_rejected(self, sample):
        with pytest.raises(ValueError, match="expected an array"):
            GeoTileStack(dict(sample.items()), model_context={"bad": object()})

    def test_a_non_string_key_is_rejected(self, sample):
        with pytest.raises(ValueError, match="non-string key"):
            GeoTileStack(dict(sample.items()), model_context={1: "x"})

    def test_context_is_read_only(self):
        assert not hasattr(GeoTileStack, "with_context")

    def test_context_does_not_survive_promotion_to_a_surface(self, sample):
        carried = GeoTileStack(dict(sample.items()), model_context={"lat": 1.0})
        assert not hasattr(carried.to_stack(), "model_context")


class TestMaterialization:
    @pytest.fixture
    def sample(self) -> GeoTileStack:
        return GeoStack(image=image(), label=label()).to_sample()

    def test_envelope_has_exactly_three_keys(self, sample):
        assert sorted(sample.to_sample(mode="numpy")) == ["anchor", "layers", "model_context"]

    def test_layers_stay_separate_for_the_task_to_combine(self, sample):
        shapes = {name: array.shape for name, array in sample.to_sample(mode="numpy")["layers"].items()}
        assert shapes == {"image": (2, 64, 64), "label": (1, 64, 64)}

    def test_the_anchor_rides_along_as_the_rebuild_handle(self, sample):
        assert sample.to_sample()["anchor"] is sample.anchor

    def test_band_selection_per_layer(self, sample):
        selected = sample.to_numpy(bands={"image": ["B08"]})
        assert selected["image"].shape == (1, 64, 64)

    def test_selecting_an_absent_band_is_rejected(self, sample):
        with pytest.raises(KeyError):
            sample.to_numpy(bands={"image": ["nope"]})

    def test_naming_an_absent_layer_is_rejected(self, sample):
        with pytest.raises(KeyError, match="missing layer"):
            sample.to_numpy(bands={"nope": ["B04"]})

    def test_per_layer_dtype_casting_leaves_other_layers_alone(self, sample):
        rendered = sample.to_tensor(dtype={"label": torch.float32})
        assert rendered["label"].dtype is torch.float32
        assert rendered["image"].dtype is torch.uint16


def test_stack_repr_names_its_layers():
    assert "image" in repr(GeoStack(image=image()))
