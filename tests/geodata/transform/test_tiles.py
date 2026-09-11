from __future__ import annotations

import numpy as np
import pytest

from geosave_engine.geodata.attrs import Packing, Tiling
from geosave_engine.geodata.core.raster import raster
from geosave_engine.geodata.core.stack import stack as build_stack
from geosave_engine.geodata.transform import tiles as transform_tiles

from .conftest import build, geobox

# The default grid is 32 by 32, so a 16-pixel tile with a 4-pixel overlap
# steps by 12 and needs a trailing pad to reach three tiles per axis.
SHAPE = (16, 16)
OVERLAP = 4
COUNT = 9


def test_tile_cuts_every_tile_to_one_shape() -> None:
    cut = transform_tiles.tile(build(geobox()), SHAPE, group_id="s", overlap=OVERLAP)

    assert len(cut) == COUNT
    assert {tile.red.shape for tile in cut} == {(2, *SHAPE)}


def test_tile_places_each_tile_on_the_source_grid() -> None:
    box = geobox()

    cut = transform_tiles.tile(build(box), SHAPE, group_id="s", overlap=OVERLAP)

    assert cut[0].gs.geobox == box[0:16, 0:16]
    assert cut[0].y.attrs["standard_name"] == "projection_y_coordinate"


def test_tile_lets_a_trailing_tile_reach_past_the_source() -> None:
    box = geobox()

    cut = transform_tiles.tile(build(box), SHAPE, group_id="s", overlap=OVERLAP)

    assert cut[-1].gs.bounds.right > box.boundingbox.right


def test_tile_keeps_the_axis_direction_padding_would_mirror() -> None:
    cut = transform_tiles.tile(build(geobox()), SHAPE, group_id="s", overlap=OVERLAP)

    assert bool(np.all(np.diff(cut[-1].y.values) < 0))


def test_tile_stamps_each_tile_with_its_own_position() -> None:
    cut = transform_tiles.tile(build(geobox()), SHAPE, group_id="s", overlap=OVERLAP)

    stamps = [tile.gs.attrs.root.get(Tiling) for tile in cut]
    assert [stamp.tile_index for stamp in stamps] == list(range(COUNT))
    assert {stamp.source_id for stamp in stamps} == {"s"}
    assert stamps[0].source_shape == (32, 32)
    assert stamps[0].tile_shape == SHAPE


def test_tile_copies_no_pixel_of_the_source() -> None:
    box = geobox()
    source = build(box)

    cut = transform_tiles.tile(source, SHAPE, group_id="s", overlap=OVERLAP)

    assert np.array_equal(cut[0].red.values, source.red.values[:, 0:16, 0:16])


def test_tile_keeps_the_source_attrs() -> None:
    cut = transform_tiles.tile(build(geobox()), SHAPE, group_id="s", overlap=OVERLAP)

    assert cut[0].attrs["title"] == "scene"
    assert cut[0].red.attrs["scale_factor"] == pytest.approx(1e-4)


def test_tile_reads_nothing_from_a_lazy_source() -> None:
    source = build(geobox(), chunks={"x": 16, "y": 16})

    cut = transform_tiles.tile(source, SHAPE, group_id="s", overlap=OVERLAP)

    assert cut[-1].red.chunks is not None


def test_tile_pads_a_constant_edge_with_the_declared_fill() -> None:
    source = build(geobox()).gs.rebase(Packing(fill_value=7), target="red")

    cut = transform_tiles.tile(
        source, SHAPE, group_id="s", overlap=OVERLAP, mode="constant"
    )

    assert int(cut[-1].red.values[0, -1, -1]) == 7


def test_tile_refuses_a_constant_edge_without_a_fill() -> None:
    source = build(geobox(), packed=False)

    with pytest.raises(ValueError, match="declare no fill value"):
        transform_tiles.tile(source, SHAPE, group_id="s", mode="constant")


def test_tile_refuses_a_shape_larger_than_the_source() -> None:
    with pytest.raises(ValueError, match="larger than the raster"):
        transform_tiles.tile(build(geobox()), (64, 64), group_id="s")


def test_tile_leaves_an_unplaced_raster_unplaced() -> None:
    source = raster({"band": np.zeros((32, 32), "uint8")})

    cut = transform_tiles.tile(source, SHAPE, group_id="s", overlap=OVERLAP)

    assert len(cut) == COUNT
    with pytest.raises(ValueError, match="no locatable grid"):
        cut[0].gs.geobox


def test_tile_stack_cuts_every_group_on_one_layout() -> None:
    box = geobox()
    scene = build_stack({"image": build(box), "labels": build(box)[["red"]]})

    cut = transform_tiles.tile_stack(scene, SHAPE, group_id="s", overlap=OVERLAP)

    assert len(cut) == COUNT
    assert cut[0].gs.groups == ("image", "labels")
    image = cut[3]["image"].to_dataset()
    labels = cut[3]["labels"].to_dataset()
    assert image.gs.geobox == labels.gs.geobox
    assert image.gs.attrs.root.get(Tiling).tile_index == 3


def test_tile_stack_cuts_a_group_where_the_raster_alone_would() -> None:
    box = geobox()
    scene = build_stack({"image": build(box)})

    cut = transform_tiles.tile_stack(scene, SHAPE, group_id="s", overlap=OVERLAP)
    alone = transform_tiles.tile(build(box), SHAPE, group_id="s", overlap=OVERLAP)

    assert cut[3]["image"].to_dataset().gs.geobox == alone[3].gs.geobox


def test_tile_stack_keeps_colliding_variable_names_apart() -> None:
    box = geobox()
    scene = build_stack({"image": build(box)[["red"]], "labels": build(box)[["red"]]})

    cut = transform_tiles.tile_stack(scene, SHAPE, group_id="s", overlap=OVERLAP)

    assert cut[0]["image"].to_dataset().gs.variables == ("red",)
    assert cut[0]["labels"].to_dataset().gs.variables == ("red",)


def test_tile_stack_refuses_a_stack_holding_no_group() -> None:
    import xarray as xr

    with pytest.raises(ValueError, match="at least one group"):
        transform_tiles.tile_stack(xr.DataTree(), SHAPE, group_id="s")
