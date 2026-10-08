"""An item table lists each saved sample as one whole window."""

from __future__ import annotations

from datetime import UTC, datetime

import numpy as np
import pytest
from odc.geo.geobox import GeoBox

from geosave_engine.geodata import GeoVector, cuts, raster, stac, stack
from geosave_engine.geodata.stac import table


def test_each_sample_is_one_window_over_all_of_it(items, grid) -> None:
    (window,) = cuts.stacks(items).to_dict("records")

    assert (window["id"], window["stack"], window["parent"]) == ("s0", "s0", None)
    assert (window["row_off"], window["col_off"]) == (0, 0)
    assert (window["height"], window["width"]) == tuple(grid.shape)
    assert window["transform"] == pytest.approx(list(grid.transform)[:6])
    assert window["crs"] == "EPSG:32633"


def test_a_window_states_the_time_labels_of_each_group(items) -> None:
    (window,) = cuts.stacks(items).to_dict("records")

    assert window["times"]["s1"] == [
        "2024-03-05T00:00:00",
        "2024-04-04T00:00:00",
        "2024-07-01T00:00:00",
    ]
    assert len(window["times"]["s2"]) == 5
    assert window["times"]["dem"] is None
    assert window["times"]["label"] is None
    assert str(window["start_datetime"])[:10] == "2024-01-10"
    assert str(window["end_datetime"])[:10] == "2024-07-03"


def test_samples_keep_the_order_they_first_appear(sample, tmp_path) -> None:
    built = [
        *stac.create_stack_items(sample.gs.to_zarr(tmp_path / "b"), name="b"),
        *stac.create_stack_items(sample.gs.to_zarr(tmp_path / "a"), name="a"),
    ]

    assert cuts.stacks(table.from_items(built))["id"].tolist() == ["b", "a"]


def test_a_table_naming_no_samples_refuses(sample, tmp_path) -> None:
    paths = sample.gs.rasters["s2"].gs.to_zarr(tmp_path / "s2.zarr")

    with pytest.raises(ValueError, match="geosave:stack"):
        cuts.stacks(table.from_items(stac.create_items(paths)))


def test_groups_on_different_grids_refuse(sample, grid, tmp_path) -> None:
    coarse = GeoBox.from_bbox(grid.boundingbox, grid.crs, resolution=20)
    apart = stack(
        {
            "label": sample.gs.rasters["label"],
            "dem": raster(
                {"height": (("y", "x"), np.zeros(coarse.shape, "float32"))}, coarse
            ),
        }
    )
    first = stac.create_stack_items(
        {"label": apart.gs.rasters["label"].gs.to_zarr(tmp_path / "label.zarr")},
        name="s0",
        datetime=datetime(2024, 1, 1, tzinfo=UTC),
    )
    second = stac.create_stack_items(
        {"dem": apart.gs.rasters["dem"].gs.to_zarr(tmp_path / "dem.zarr")},
        name="s0",
        datetime=datetime(2024, 1, 1, tzinfo=UTC),
    )
    rows = GeoVector.concat([table.from_items(first), table.from_items(second)])

    with pytest.raises(ValueError, match=r"\['dem'\] on a different grid"):
        cuts.stacks(rows)


def test_a_timeless_raster_given_a_date_is_still_timeless(sample, tmp_path) -> None:
    built = stac.create_stack_items(
        {"label": sample.gs.rasters["label"].gs.to_zarr(tmp_path / "label.zarr")},
        name="s0",
        datetime=datetime(2024, 1, 1, tzinfo=UTC),
    )

    (window,) = cuts.stacks(table.from_items(built)).to_dict("records")

    assert window["times"] == {"label": None}
    assert window["start_datetime"] is None
