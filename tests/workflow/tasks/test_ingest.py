import numpy as np
from prefect.cache_policies import NO_CACHE
import pytest

from geosave_engine.geodata.core.raster import raster
from geosave_engine.geodata.utils import io
from geosave_engine.workflow.configs import SourceConfig
from geosave_engine.workflow.specs import RasterRequirement
from geosave_engine.workflow.tasks.ingest import ingest_sample
from geosave_engine.workflow.tasks.save import write_stack


def write_label(path, anchor, *, dated=True):
    values = np.arange(16, dtype="uint8").reshape(4, 4)
    label = raster({"class": values}, anchor.geobox)
    if dated:
        label = label.assign_coords(time=np.datetime64("2025-01-15T12:00:00"))
    return label, io.geotiff.write_cog(label, path)


def optical_requirement(url):
    return RasterRequirement(
        variables=("red", "nir"),
        collection="optical",
        endpoints=(url,),
        require_crs=True,
    )


def test_ingest_sample_writes_label_and_matching_imagery(tmp_path, stac_server):
    url, requests, expected_anchor = stac_server
    label, label_path = write_label(tmp_path / "label.tif", expected_anchor)
    output = tmp_path / "sample.zarr"

    result = ingest_sample.fn(
        label_path,
        {"optical": SourceConfig()},
        {
            "optical": optical_requirement(url)
        },
        output,
    )

    assert result == str(output)
    assert requests == [["optical"]]
    with io.read_stack(output, chunks="auto") as restored:
        assert set(restored.gs.groups) == {"label", "optical"}
        assert restored.gs.geobox == expected_anchor.geobox
        np.testing.assert_array_equal(restored["label"]["class"], label["class"])
        assert tuple(restored["optical"].data_vars) == ("red", "nir")
        assert restored["label"]["class"].chunks is not None
        assert restored["optical"].red.chunks is not None


def test_ingest_sample_reuses_a_valid_completed_sample_without_stac(
    tmp_path, stac_server
):
    url, requests, expected_anchor = stac_server
    _, label_path = write_label(tmp_path / "label.tif", expected_anchor)
    sources = {"optical": SourceConfig()}
    requirements = {"optical": optical_requirement(url)}
    output = tmp_path / "sample.zarr"
    ingest_sample.fn(label_path, sources, requirements, output)

    result = ingest_sample.fn(label_path, sources, requirements, output)

    assert result == str(output)
    assert requests == [["optical"]]


def test_ingest_sample_rejects_an_existing_sample_with_missing_groups(
    tmp_path, stac_server
):
    url, requests, expected_anchor = stac_server
    label, label_path = write_label(tmp_path / "label.tif", expected_anchor)
    output = tmp_path / "sample.zarr"
    write_stack({"optical": label.rename({"class": "red"})}, output)

    with pytest.raises(ValueError, match="groups.*label"):
        ingest_sample.fn(
            label_path,
            {"optical": SourceConfig()},
            {"optical": optical_requirement(url)},
            output,
        )

    assert requests == []
    with io.read_stack(output) as restored:
        assert restored.gs.groups == ("optical",)


def test_ingest_sample_rejects_stale_source_variables(tmp_path, stac_server):
    url, requests, expected_anchor = stac_server
    label, label_path = write_label(tmp_path / "label.tif", expected_anchor)
    output = tmp_path / "sample.zarr"
    optical = raster(
        {"red": np.ones((4, 4), dtype="uint16")}, expected_anchor.geobox
    )
    write_stack({"label": label, "optical": optical}, output)

    with pytest.raises(ValueError, match="nir"):
        ingest_sample.fn(
            label_path,
            {"optical": SourceConfig()},
            {"optical": optical_requirement(url)},
            output,
        )

    assert requests == []
    with io.read_stack(output) as restored:
        assert tuple(restored["optical"].data_vars) == ("red",)


def test_ingest_sample_requires_label_time(tmp_path, stac_server):
    url, requests, expected_anchor = stac_server
    _, label_path = write_label(
        tmp_path / "timeless.tif", expected_anchor, dated=False
    )

    with pytest.raises(ValueError, match="Label raster has no time"):
        ingest_sample.fn(
            label_path,
            {"optical": SourceConfig()},
            {"optical": optical_requirement(url)},
            tmp_path / "sample.zarr",
        )

    assert requests == []


def test_ingest_sample_is_not_cached_or_persisted():
    assert ingest_sample.cache_policy is NO_CACHE
    assert ingest_sample.persist_result is False
