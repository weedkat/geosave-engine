from datetime import datetime

import pytest

from geosave_engine.geodata.attrs import AttrsNamespace, StacItem, StacMetadata


@pytest.mark.parametrize(
    "difference",
    [
        {"datetime": datetime(2025, 1, 2)},
        {"properties": {"view:sun_azimuth": 120}},
        {"assets": {"red": {"scale": 0.2}}},
    ],
)
def test_provenance_retains_different_records_with_the_same_id(difference) -> None:
    first = StacItem(
        id="scene",
        datetime=datetime(2025, 1, 1),
        properties={"view:sun_azimuth": 100},
        assets={"red": {"scale": 0.1}},
    )
    second = first.model_copy(update=difference)
    sources = [
        AttrsNamespace.from_attrs(
            StacMetadata(stac_items=records, stac_groupby="solar_day").to_attrs(),
            "dataset",
        )
        for records in [(first,), (second, first)]
    ]

    merged, dropped = AttrsNamespace.merge(sources)

    provenance = merged.get(StacMetadata)
    assert provenance.stac_items == (first, second)
    assert provenance.stac_groupby == "solar_day"
    assert dropped == set()


def test_provenance_retains_records_when_grouping_differs() -> None:
    first = StacItem(id="first", datetime=datetime(2025, 1, 1))
    second = StacItem(id="second", datetime=datetime(2025, 1, 2))

    merged, dropped = StacMetadata.merge(
        [
            StacMetadata(stac_items=(first,), stac_groupby="id"),
            StacMetadata(stac_items=(second,), stac_groupby="solar_day"),
        ],
        conflicts="drop",
    )

    assert merged.stac_items == (first, second)
    assert merged.stac_groupby is None
    assert dropped == {"stac_groupby"}
