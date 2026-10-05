from importlib import import_module
import numpy as np
from odc.geo.geobox import GeoBox
from prefect.cache_policies import NO_CACHE
import pytest

from geosave_engine.geodata.core.raster import raster
from geosave_engine.geodata import io
from geosave_engine.model.spec import ModelSpec, RasterRequirement, StacRecipe
from geosave_engine.geodata import read_stack
from geosave_engine.workflow.tasks.sample import write_sample

dense_module = import_module("geosave_engine.workflow.tasks.dense")


def _write_label(path, anchor, *, dated=True):
    path.parent.mkdir(parents=True, exist_ok=True)
    label = raster({"class": np.full((4, 4), 1, dtype="uint8")}, anchor.geobox)
    if dated:
        label = label.assign_coords(time=np.datetime64("2025-01-15T12:00:00"))
    return label, io.geotiff.write_cog(label, path)


def _requirement(url, *, dims=None):
    return RasterRequirement(
        variables=("red", "nir"),
        stac=StacRecipe.model_validate(
            {
                "collection": "optical",
                "endpoints": (url,),
                "load": {"bands": ["red", "nir"]},
            }
        ),
        dims=dims,
        require_crs=True,
    )


def _spec(requirements):
    return ModelSpec(schema_version=2, rasters=requirements)


def test_prepare_dense_sample_is_not_cached_or_persisted() -> None:
    assert dense_module.prepare_dense_sample.cache_policy is NO_CACHE
    assert dense_module.prepare_dense_sample.persist_result is False


def test_prepare_dense_sample_writes_label_and_matching_imagery(
    tmp_path, stac_server
) -> None:
    url, requests, anchor = stac_server
    label, label_path = _write_label(tmp_path / "label.tif", anchor)
    output = tmp_path / "sample"

    result = dense_module.prepare_dense_sample.fn(
        label_path,
        _spec({"optical": _requirement(url)}),
        output,
    )

    assert result.loc[0, "id"] == "sample"
    assert requests == [["optical"]]
    assert sorted(path.name for path in output.iterdir()) == [
        "label.tif",
        "optical.tif",
    ]
    with read_stack(output, chunks="auto") as sample:
        assert set(sample.gs.groups) == {"label", "optical"}
        assert sample.gs.geobox == anchor.geobox
        np.testing.assert_array_equal(sample["label"]["class"], label["class"])
        assert tuple(sample["optical"].data_vars) == ("red", "nir")
        assert sample["label"]["class"].chunks is not None
        assert sample["optical"].red.chunks is not None


def test_prepare_dense_sample_supports_zarr(tmp_path, stac_server) -> None:
    url, requests, anchor = stac_server
    label, label_path = _write_label(tmp_path / "label.tif", anchor)
    output = tmp_path / "sample"

    result = dense_module.prepare_dense_sample.fn(
        label_path,
        _spec({"optical": _requirement(url)}),
        output,
        format="zarr",
    )

    assert result.loc[0, "id"] == "sample"
    assert requests == [["optical"]]
    assert sorted(path.name for path in output.iterdir()) == [
        "label.zarr",
        "optical.zarr",
    ]
    with read_stack(output, chunks="auto") as sample:
        assert set(sample.gs.groups) == {"label", "optical"}
        np.testing.assert_array_equal(sample["label"]["class"], label["class"])


def test_prepare_dense_sample_reuses_a_valid_sample_without_stac(
    tmp_path, stac_server
) -> None:
    url, requests, anchor = stac_server
    _, label_path = _write_label(tmp_path / "label.tif", anchor)
    spec = _spec({"optical": _requirement(url)})
    output = tmp_path / "sample"
    dense_module.prepare_dense_sample.fn(label_path, spec, output)

    again = dense_module.prepare_dense_sample.fn(label_path, spec, output)

    assert again.loc[0, "id"] == "sample"
    assert requests == [["optical"]]


def test_prepare_dense_sample_reuses_geotiff_with_required_time_dimension(
    tmp_path, stac_server
) -> None:
    url, requests, anchor = stac_server
    _, label_path = _write_label(tmp_path / "label.tif", anchor)
    spec = _spec({"optical": _requirement(url, dims=("time", "y", "x"))})
    output = tmp_path / "sample"

    dense_module.prepare_dense_sample.fn(label_path, spec, output)

    again = dense_module.prepare_dense_sample.fn(label_path, spec, output)

    assert again.loc[0, "id"] == "sample"
    assert requests == [["optical"]]


def test_prepare_dense_sample_refuses_rasters_without_a_stac_block(tmp_path) -> None:
    grid = GeoBox.from_bbox((0, 0, 40, 40), crs="EPSG:32633", resolution=10)
    label = raster({"class": np.ones((4, 4), dtype="uint8")}, grid).assign_coords(
        time=np.datetime64("2025-01-15T12:00:00")
    )
    label_path = io.geotiff.write_cog(label, tmp_path / "label.tif")
    output = tmp_path / "sample"

    with pytest.raises(ValueError, match=r"no `stac` block: \['optical'\]"):
        dense_module.prepare_dense_sample.fn(
            label_path,
            _spec({"optical": RasterRequirement(variables=("red",))}),
            output,
        )

    assert not output.exists()


def test_prepare_dense_sample_rejects_an_incomplete_existing_sample(
    tmp_path, stac_server
) -> None:
    url, requests, anchor = stac_server
    label, label_path = _write_label(tmp_path / "label.tif", anchor)
    output = tmp_path / "sample"
    write_sample({"optical": label.rename({"class": "red"})}, output)

    with pytest.raises(ValueError, match="does not match model rasters"):
        dense_module.prepare_dense_sample.fn(
            label_path,
            _spec({"optical": _requirement(url)}),
            output,
        )

    assert requests == []
    with read_stack(output, chunks="auto") as sample:
        assert sample.gs.groups == ("optical",)


def test_validate_dense_sample_rejects_stale_raster_variables(
    tmp_path, stac_server
) -> None:
    url, requests, anchor = stac_server
    label, _ = _write_label(tmp_path / "label.tif", anchor)
    output = tmp_path / "sample"
    optical = raster({"red": np.ones((4, 4), dtype="uint16")}, anchor.geobox)
    write_sample({"label": label, "optical": optical}, output)

    with pytest.raises(ValueError, match="nir"):
        dense_module._validate_dense_sample(output, {"optical": _requirement(url)})

    assert requests == []


def test_prepare_dense_sample_rejects_unexpected_geotiff_asset(
    tmp_path, stac_server
) -> None:
    url, requests, anchor = stac_server
    label, label_path = _write_label(tmp_path / "label.tif", anchor)
    output = tmp_path / "sample"
    optical = raster(
        {
            "red": np.ones((4, 4), dtype="uint16"),
            "nir": np.ones((4, 4), dtype="uint16"),
        },
        anchor.geobox,
    )
    write_sample({"label": label, "optical": optical}, output)
    io.geotiff.write_cog(optical, output / "unexpected.tif")

    with pytest.raises(ValueError, match="does not match model rasters"):
        dense_module.prepare_dense_sample.fn(
            label_path,
            _spec({"optical": _requirement(url)}),
            output,
        )

    assert requests == []
    assert (output / "unexpected.tif").exists()


def test_prepare_dense_sample_rejects_misaligned_geotiff_assets(
    tmp_path, stac_server
) -> None:
    url, requests, anchor = stac_server
    label, label_path = _write_label(tmp_path / "label.tif", anchor)
    output = tmp_path / "sample"
    output.mkdir()
    io.geotiff.write_cog(label, output / "label.tif")
    shifted = GeoBox.from_bbox((10, 0, 50, 40), crs=anchor.geobox.crs, shape=(4, 4))
    optical = raster(
        {
            "red": np.ones((4, 4), dtype="uint16"),
            "nir": np.ones((4, 4), dtype="uint16"),
        },
        shifted,
    )
    io.geotiff.write_cog(optical, output / "optical.tif")

    with pytest.raises(ValueError, match="do not share a grid"):
        dense_module.prepare_dense_sample.fn(
            label_path,
            _spec({"optical": _requirement(url)}),
            output,
        )

    assert requests == []
    assert sorted(path.name for path in output.iterdir()) == [
        "label.tif",
        "optical.tif",
    ]


def test_prepare_dense_sample_requires_label_time(tmp_path, stac_server) -> None:
    url, requests, anchor = stac_server
    _, label_path = _write_label(tmp_path / "timeless.tif", anchor, dated=False)

    with pytest.raises(ValueError, match="Label raster has no time"):
        dense_module.prepare_dense_sample.fn(
            label_path,
            _spec({"optical": _requirement(url)}),
            tmp_path / "sample",
        )

    assert requests == []


def test_prepare_dense_sample_returns_the_samples_row(tmp_path, stac_server) -> None:
    url, _, anchor = stac_server
    _, label_path = _write_label(tmp_path / "label.tif", anchor)

    row = dense_module.prepare_dense_sample.fn(
        label_path,
        _spec({"optical": _requirement(url)}),
        tmp_path / "prepared" / "north" / "a",
        sample_id="north/a",
        properties={"quality": 0.97, "note": None},
    ).iloc[0]

    assert row.id == "north/a"
    assert list(row.assets) == ["label", "optical"]
    assert row.assets["optical"]["href"] == str(
        tmp_path / "prepared" / "north" / "a" / "optical.tif"
    )
    assert row.assets["optical"]["bands"] == [{"name": "red"}, {"name": "nir"}]
    assert str(row.start_datetime).startswith("2025-01-15")
    assert row.quality == 0.97
    assert row.note is None


def test_validate_row_accepts_only_the_label_and_the_specs_rasters(
    tmp_path, stac_server
) -> None:
    url, _, anchor = stac_server
    _, label_path = _write_label(tmp_path / "label.tif", anchor)
    spec = _spec({"optical": _requirement(url)})
    row = dense_module.prepare_dense_sample.fn(label_path, spec, tmp_path / "s")

    dense_module.validate_row(row.iloc[0], spec)
    with pytest.raises(ValueError, match="does not match model rasters"):
        dense_module.validate_row(
            row.iloc[0],
            _spec({"optical": _requirement(url), "radar": _requirement(url)}),
        )
